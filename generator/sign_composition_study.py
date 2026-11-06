"""Source-bound, saved-only composition assay. No model construction or evaluation."""
import argparse
from pathlib import Path
import shutil
import time
from encoder_atoms import np, read, save, sha, identity, floats, guard
from sign_composition import support_closed, decompose, account

HERE = Path(__file__).resolve().parent
PORTS = ('R', 'M', 'Hnext')


def verify(root, request):
    unsigned = dict(request); unsigned.pop('requestSHA256')
    if identity(unsigned) != request['requestSHA256']:
        raise ValueError('Request identity differs')
    for name, digest in request['sourceHashes'].items():
        if sha(Path(root) / name) != digest: raise ValueError('Payload source differs: ' + name)


def stage(root, compact, verification):
    from mrun.client.api import Api
    compact = Path(compact).resolve(); verification = Path(verification).resolve()
    v = read(verification); audit = read(compact / 'audit.json')
    if sha(compact / 'audit.json') != v['audit_sha256']:
        raise ValueError('Verified audit differs')
    job_id = audit['job_id']; api = Api(); job = api.json('GET', '/api/jobs/' + job_id)
    request = read(compact / 'source-request.json')
    if job['state'] != 'succeeded' or job['config']['request_sha256'] != request['request_sha256']:
        raise ValueError('Native source job/request differs')
    raw = '/mnt/big/mdb/research/' + job_id + '/qwen-row-sign-result'
    inventory = read(compact / 'source-files.json')
    frontier = read(compact / 'source-source-frontier.json')
    parent = api.json('GET', '/api/jobs/' + frontier['parent_job'])
    if parent['state'] != 'succeeded' or parent['config']['request_sha256'] != frontier['parent_request_sha256']:
        raise ValueError('Own-positive parent differs')
    root = Path(root).resolve(); root.mkdir(exist_ok=False, parents=True)
    payload = root / 'payload'; payload.mkdir()
    for name in ('sign_composition.py', 'sign_composition_study.py', 'encoder_atoms.py'):
        shutil.copyfile(HERE / name, payload / name)
    for name in ('source-files.json', 'source-source-frontier.json', 'audit.json'):
        shutil.copyfile(compact / name, payload / name)
    shutil.copyfile(verification, payload / 'intake-verification.json')
    def original(name):
        descriptor = frontier['source_arrays'][name]
        return {'path': frontier['input_root'] + '/' + descriptor['path'], **{k: descriptor[k] for k in ('bytes', 'sha256', 'shape', 'stored_dtype')}}
    def edited(name):
        return {'path': raw + '/' + name, **inventory['files'][name]}
    source = {'rawManifest': {'path': raw + '/files.json', 'sha256': sha(compact / 'source-files.json')},
              'parentManifest': {'path': frontier['input_root'] + '/files.json', 'sha256': frontier['parent_raw_manifest_sha256']},
              'positive': {k: original(k) for k in PORTS},
              'arms': {arm: {'signs': edited('shortcut/' + arm + '/row-signs.npy'),
                             'permitted': edited('shortcut/' + arm + '/permitted.npy'),
                             'closed': edited('shortcut/' + arm + '/support-closed.npy'),
                             'lanes': {lane: {k: edited(lane + '/' + arm + '/' + k + '.npy') for k in PORTS}
                                       for lane in ('forecast', 'shortcut')}}
                       for arm in ('all-negative', 'prefix-negative')}}
    save(payload / 'source.json', source)
    r = {'schema': 'tce-sign-composition-request-v1', 'nativeJobID': job_id,
         'nativeRequestSHA256': request['request_sha256'], 'auditJobID': v['jobs']['audit']['job_id'],
         'scope': 'Retrospective own-functional saved-state geometry, one context/two dependent edits. No prediction or next-block consumer claim.',
         'law': 'D=Hprime-sH=(Rprime-sR)+(Mprime-sM)+(epsilonprime-s epsilon); epsilon=H-R-M. Best arbitrary row scale is a=<Hprime,H>/<H,H>.',
         'observables': 'Complete signed vectors, all512 row accounts, closed/mixed/negative/positive regions, term/cross energies, finite remainder and best-scaling-family orthogonal defect. No numerical accuracy gates.',
         'sourceHashes': {p.name: sha(p) for p in payload.iterdir()},
         'device': 'cpu', 'backend': 'numpy-fp64', 'cpuThreads': 1, 'timeoutSeconds': 600}
    r['requestSHA256'] = identity(r); save(root / 'request.json', r); save(payload / 'request.json', r)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1',
          'hashed_roots': [p.name for p in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': r['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from response_study import command
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root / 'payload', r)
    if (root / 'job.json').exists(): raise ValueError('Already submitted')
    cmd = command(); cmd[-3] = 'sign_composition_study.py'
    job = launch(cmd, experiment='tce-sign-composition-v1', task='analysis', payload=root / 'payload',
                 pin='beast', needs={'cpu': True},
                 config={'schema': r['schema'], 'requestSHA256': r['requestSHA256'], 'device': 'cpu', 'backend': 'numpy'},
                 timeout_s=600, detach=True, retry_on_kill=False, preflight=True, guarded=True,
                 note='Savedonly CPU1: authenticated own-positive and2edited R/M/H19fields, no model/weights/learned operations. Quantify sign/scaling-family nonclosure and BF16 addition remainder; all512rows/closed/mixed countertrends. Rawderivedvectors staynearcompute, compactaccounts collected. Hostlocal parents,expectedunder1min,finite600s,no retries/no GPU; distinct from neighboring cachedQKV native campaign.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root / 'job.json', value); return value


def execute(root):
    import json
    guard(); root = Path(root).resolve(); r = read(root / 'request.json'); verify(root, r)
    start = time.perf_counter(); cpu = time.process_time(); source = read(root / 'source.json')
    out = root / 'response-result'; out.mkdir(); arrays = root / 'composition-arrays'; arrays.mkdir()
    reads = []
    for k in ('rawManifest', 'parentManifest'):
        if sha(source[k]['path']) != source[k]['sha256']: raise ValueError('Source manifest changed')
        reads.append(source[k])
    def load(d):
        p = Path(d['path'])
        if p.stat().st_size != d['bytes'] or sha(p) != d['sha256']: raise ValueError('Source array differs')
        a = np.load(p, allow_pickle=False)
        if 'shape' in d and (list(a.shape) != d['shape'] or str(a.dtype) != d['stored_dtype']):
            raise ValueError('Source array geometry differs')
        reads.append(d)
        return a
    def field(d):
        a = load(d)
        if a.dtype != np.dtype('uint16') or a.shape != (1, 512, 2560): raise ValueError('Not expected retained BF16 field')
        value = floats(a).astype(np.float64)[0]
        if not np.isfinite(value).all(): raise ValueError('Nonfinite retained field')
        return value
    positive = {k: field(d) for k, d in source['positive'].items()}; reports = []; receipts = []; count = 0
    with (out / 'observations.jsonl').open('x') as stream:
        for arm, spec in source['arms'].items():
            encoded = load(spec['signs']); signs = floats(encoded) if encoded.dtype == np.uint16 else encoded
            signs = signs.reshape(-1); mask = load(spec['permitted']); saved_closed = load(spec['closed']).reshape(-1)
            closed = support_closed(signs, mask)
            if not np.array_equal(closed, saved_closed): raise ValueError('Support predicate differs from saved instrument')
            for lane, fields in spec['lanes'].items():
                edited = {k: field(d) for k, d in fields.items()}; a = decompose(positive, edited, signs)
                dest = arrays / (arm + '-' + lane + '.npz'); np.savez(dest, **a)
                receipts.append({'arm': arm, 'lane': lane, 'path': str(dest), 'bytes': dest.stat().st_size, 'sha256': sha(dest)})
                regions = [('all', np.ones(512, dtype=bool)), ('closed', closed), ('mixed', ~closed),
                           ('negative', signs == -1), ('positive', signs == 1)]
                for region, use in regions:
                    values = account(a, edited['Hnext'], use)
                    if values is not None: reports.append({'arm': arm, 'lane': lane, 'region': region, **values})
                for i in range(512):
                    use = np.arange(512) == i
                    stream.write(json.dumps({'arm': arm, 'lane': lane, 'position': i,
                        'supportClosed': bool(closed[i]), 'sign': int(signs[i]), **account(a, edited['Hnext'], use)}, allow_nan=False) + '\n')
                    count += 1
                print('COMPOSITION_CASE_DONE', arm, lane, flush=True)
    for d in reads:
        if sha(d['path']) != d['sha256']: raise ValueError('Source changed during analysis')
    seal = {'schema': 'tce-sign-composition-analysis-seal-v1', 'requestSHA256': r['requestSHA256'],
            'scope': 'Retrospective accounting seal, not a held-out prediction seal.', 'arrays': receipts, 'verifiedReads': reads}
    save(out / 'analysis-seal.json', seal)
    for name in ('request.json', 'source.json'): shutil.copyfile(root / name, out / name)
    save(out / 'report.json', {'schema': 'tce-sign-composition-result-v1', 'requestSHA256': r['requestSHA256'],
         'scope': r['scope'], 'accounts': reports, 'observations': count,
         'analysisSealSHA256': sha(out / 'analysis-seal.json'),
         'workerSeconds': time.perf_counter() - start, 'cpuSeconds': time.process_time() - cpu,
         'work': {'newNeuralEvaluations': 0, 'originalParameterReads': 0, 'originalProjections': 0, 'nextBlockConsumers': 0, 'newImages': 0},
         'limitations': ['Decomposition uses already evaluated edited ports; closure is accounting, not independent prediction.',
                         'Best scales are retrospective fits, not serving parameters.',
                         'Noncollinearity need not survive a particular learned projection; next-layer and image behavior remain unmeasured.',
                         'Regions and lanes share states; rows are not independent trials.'],
         'mechanics_status': 'complete', 'instrument_status': 'executed', 'terminal_status': 'unassessed'})
    print('COMPOSITION_DONE', flush=True)


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['stage', 'submit', 'execute', 'collect'])
    p.add_argument('run'); p.add_argument('--compact'); p.add_argument('--verification'); a = p.parse_args()
    if a.action == 'stage': result = stage(a.run, a.compact, a.verification)
    elif a.action == 'collect':
        from response_study import collect
        result = collect(a.run)
    else: result = globals()[a.action](a.run)
    if result is not None: print(json.dumps(result))
