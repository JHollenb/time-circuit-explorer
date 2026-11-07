"""One finite saved-only capacity assay: intake→candidate exit→diagnostics."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read, save, sha, identity

HERE = Path(__file__).resolve().parent
OWNED = ['spatial_norm.py', 'spatial_norm_worker.py', 'spatial_norm_study.py',
         'encoder_atoms.py', 'paired_fields.py', 'masked_kv_study.py', 'sparse_text_kv.py']


def stage(root):
    from mrun.client.api import Api
    from publish_source_native import checked
    main, mc, mj, mr = checked(HERE.parent / 'artifacts/coherent-norm-v1')
    parent, pc, pj, pr = checked(HERE.parent / 'artifacts/image-dual-v1')
    api = Api()
    for c, j in [(mc, mj), (pc, pj)]:
        live = api.json('GET', '/api/jobs/' + j['job_id'])
        if live['state'] != 'succeeded' or live['payload_custody'] != c['custody']:
            raise ValueError('Saved-input custody differs')
    p = main / 'result';s, actor = read(p / 'inputs.json'), read(p / 'actor.json')
    if sha(p / 'report.json') != 'cc3b102e8f506ec216cc41fb06a8ea0fefa09da802ccb3ea0515004392d526de':
        raise ValueError('Reviewed parent differs')
    source_job = api.json('GET', '/api/jobs/' + s['sourceJobID'])
    if source_job['state'] != 'succeeded' or source_job['payload_custody'] != s['sourceCustody']:
        raise ValueError('Native source custody differs')
    source = {'sourceCuts': {n: {k: c[k] for k in ['cut', 'cutFile', 'cutID']} for n, c in s['sourceCuts'].items()},
        'store': s['sourceCuts']['Q']['pages']['context']['store'],
        'pNorm': read(parent / 'result/compiler.json')['compile']['fields']['pDenominator'],
        'qNorm': read(parent / 'result/reference.json')['reference']['fields']['qDenominator'],
        'numerator': actor['read']['fields']['imageNumerator'], 'textLogits': actor['read']['fields']['textLogits'],
        'parentJobID': mj['job_id'], 'parentCustody': mc['custody'], 'referenceJobID': pj['job_id'], 'referenceCustody': pc['custody'],
        'nativeJobID': s['sourceJobID'], 'nativeCustody': s['sourceCustody'], 'parentReportSHA256': sha(p / 'report.json'),
        'matchedControlPayloads': read(p / 'report.json')['matchedControlPayloads']}
    root = Path(root).resolve();root.mkdir(parents=True, exist_ok=False);payload = root / 'payload';payload.mkdir()
    for n in OWNED:
        if n in mr['ownedSources'] and sha(HERE / n) != mr['sourceHashes'][n]:
            raise ValueError('Frozen saved-input decoder differs')
        shutil.copyfile(HERE / n, payload / n)
    shutil.copyfile(HERE.parent / 'docs/plans/spatial-key-norm-capacity.md', payload / 'frozen-plan.md')
    save(payload / 'inputs.json', source)
    r = {'schema': 'tce-spatial-norm-request-v1', 'ownedSources': OWNED,
        'sourceHashes': {x.name: sha(x) for x in payload.iterdir()}, 'backend': 'numpy', 'device': 'cpu', 'dtype': 'float64',
        'support': [4, 11, 12, 13], 'knotsPerHead': 256, 'strataPerHead': [256, 705, 63],
        'plannedMatrixMACs': 0, 'newOriginalProjections': 0,
        'protocol': 'Authenticate native THWL IDs; byte-extract fixed even Qnorm rows; fresh knot-only candidate seals/exits before Q/Pnorm and numerator/text score diagnostics. Bilinear squared smoothnorm,31clamps30,no extraepsilon. Separate sampled/interior/edge,fullsigneddelta,fulljointweightedvariance/KL. No valuebank/AV/MLP/nextQ/image/refit/gridsearch.'}
    r['requestSHA256'] = identity(r);save(root / 'request.json', r);save(payload / 'request.json', r)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [x.name for x in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': r['requestSHA256']}


def submit(root):
    from response_study import command
    from mrun.client.submit import launch
    from flux_clock_study import verify
    root = Path(root).resolve();r = read(root / 'request.json');verify(root / 'payload', r)
    if (root / 'job.json').exists():
        raise ValueError('Already submitted; inspect same handle')
    job = launch(command()[:-5] + ['python', '-u', 'spatial_norm_study.py', 'execute', '.'],
        experiment='tce-spatial-norm-capacity-v1', task='analysis', payload=root / 'payload', pin='beast', needs={'cpu': True},
        config={k: r[k] for k in ['schema', 'requestSHA256', 'backend', 'device', 'dtype']},
        timeout_s=180, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='Saved-only fixed physical16x16 normgrid; nativeID custody and byte-only256-row intake. Fresh knot-only candidate seals/exits before complete norms/scores;24,576cell/96query-head outcomes,fulljointVar/KL. CPU1NumPy,zero new learned projections/AV/MLP/image/fit/gridsearch. Nativeknots/source states/numerator paid. Intentfirst,180scap,noretry;arraysnearcompute.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']};save(root / 'job.json', value);return value


def execute(root):
    root = Path(root).resolve();r = read(root / 'request.json')
    if identity({k: v for k, v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request identity differs')
    for n, digest in r['sourceHashes'].items():
        if sha(root / n) != digest:
            raise ValueError('Frozen worker/input differs: ' + n)
    out = root / 'response-result';out.mkdir()
    started = time.perf_counter();phases = []
    for phase in ['intake', 'candidate', 'diagnose']:
        tick = time.perf_counter()
        subprocess.run([sys.executable, '-u', str(root / 'spatial_norm_worker.py'), phase, str(root)], cwd=root, check=True)
        phases.append({'phase': phase, 'secondsIncludingFreshProcess': time.perf_counter()-tick})
        print('SPATIAL_NORM_' + phase.upper() + '_EXITED', flush=True)
    report = read(out / 'diagnostics.json')
    save(out / 'report.json', {**report, 'phases': phases, 'workerWallSeconds': time.perf_counter()-started})
    for n in ['inputs.json', 'request.json', 'frozen-plan.md']:
        shutil.copyfile(root / n, out / n)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']);p.add_argument('run');a = p.parse_args()
    if a.action == 'collect':
        from response_study import collect
        result = collect(a.run)
    else:
        result = globals()[a.action](a.run)
    if result is not None:
        print(json.dumps(result))
