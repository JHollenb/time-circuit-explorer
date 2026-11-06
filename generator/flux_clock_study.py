"""Compile finite clocks, render two candidates, seal, then score saved targets."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import np, read, save, sha, identity

HERE = Path(__file__).resolve().parent
OWNED = ['flux_clock_study.py', 'flux_clock.py', 'flux_clock_compile.py', 'flux_clock_images.py',
    'encoder_atoms.py', 'paired_fields.py', 'masked_kv_study.py', 'complete_masked_study.py']
NATIVE_ROOT = Path('/Users/jakeholl/.cache/mdb/qualification/flux-initial-noise-corrected-20261105/lib/python3.12/site-packages/diffusers')
EXTRA_SOURCES = ['models/normalization.py', 'models/embeddings.py', 'models/activations.py', 'models/cache_utils.py']


def verify(root, request):
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']: raise ValueError('Request changed')
    for name, digest in request['sourceHashes'].items():
        if sha(Path(root) / name) != digest: raise ValueError('Frozen worker/input changed: ' + name)


def source_checks(source, diffusers):
    import torch
    if {'torch': torch.__version__, 'diffusers': diffusers.__version__} != source['versions']: raise ValueError('Native framework versions differ')
    model = Path(source['modelRoot']); assets = []
    for name, digest in source['nativeSources'].items():
        if sha(Path(diffusers.__file__).parent / name) != digest: raise ValueError('Native source changed: ' + name)
    for name, digest in source['imageAssets'].items():
        p = model / name; s = p.stat(); signature = [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
        if sha(p) != digest: raise ValueError('Original image asset changed')
        assets.append({'path': str(p), 'sha256': digest, 'bytesHashed': s.st_size, 'signature': signature})
    return model, assets


def assets_unchanged(assets):
    for item in assets:
        s = Path(item['path']).stat()
        if [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns] != item['signature']:
            raise ValueError('Original asset identity changed during execution')


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from publish_context_statistics import checked_collection
    api = Api(); parent = HERE.parent / 'artifacts/complete-masked-image-v1'
    collection, job, previous = checked_collection(parent, current_workers=False)
    live = api.json('GET', '/api/jobs/' + job['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != collection['custody']: raise ValueError('Image parent differs')
    p = parent / 'result'; source = read(p / 'source.json'); full = read(p / 'images-full.json'); enc = read(p / 'encoder-full.json')
    if sha(p / 'full-seal.json') != read(parent / 'retained-array-verification.json')['fullSealSHA256']: raise ValueError('Parent full-image verification differs')
    if full['schedules']['A'] != full['schedules']['B']: raise ValueError('Parent schedules differ')
    native = dict(source['nativeSources'])
    for name, digest in native.items():
        if sha(NATIVE_ROOT / name) != digest: raise ValueError('Local pinned source differs from parent')
    native.update({name: sha(NATIVE_ROOT / name) for name in EXTRA_SOURCES})
    preview = _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1536], backend='diffusers', dtype='bfloat16', device='cuda')
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); payload = root / 'payload'; payload.mkdir()
    for name in OWNED: shutil.copyfile(HERE / name, payload / name)
    shutil.copyfile(HERE.parent / 'docs/plans/compiled-flux-clock.md', payload / 'frozen-plan.md')
    forbidden = [x['path'] for ports in full['fields'].values() for x in ports.values()] + [x['path'] for x in full['images'].values()]
    inputs = {'modelRoot': source['lexicalSource']['modelRoot'], 'nativeSources': native, 'imageAssets': source['imageAssets'],
        'versions': {'torch': full['torchVersion'], 'diffusers': full['diffusersVersion']},
        'numericalProfile': full['numericalProfile'], 'attentionProcessors': full['attentionProcessors'],
        'E': {c: enc['fields'][c]['E'] for c in ['A', 'B']},
        'noise': source['noise'], 'schedule': full['schedules']['A'], 'forbiddenTargets': forbidden, 'parentJobID': job['job_id'],
        'parentRoot': job['result']['diagnostics']['cwd'], 'noiseParentRoot': source['noiseParentRoot'],
        'parentRequestSHA256': previous['requestSHA256'], 'parentFullSealSHA256': sha(p / 'full-seal.json'),
        'parentByteVerificationSHA256': sha(parent / 'retained-array-verification.json')}
    save(payload / 'inputs.json', inputs)
    save(payload / 'targets.json', {'fields': full['fields'], 'images': full['images'], 'schedules': full['schedules'],
        'parentImagesSHA256': sha(p / 'images-full.json'), 'parentFullSealSHA256': sha(p / 'full-seal.json'),
        'scope': 'Previously observed saved targets, scored only after complete candidates seal; not prospective semantic coverage.'})
    save(payload / 'resource-basis.json', {'intentFirstPreview': preview, 'measuredParentJobID': job['job_id'],
        'measuredParentResult': {k: live['result'][k] for k in ['elapsed_s', 'peak_rss_mb', 'peak_vram_mb']},
        'measuredParentReservation': live['reservation'],
        'exception': 'Generic B64/fallback40.2GBdiffusion sizing is inapplicable to B1/512. Keep compatible measured14336MBRAM/13000MBVRAM envelope. Separate compiler and selected-weight image child, no co-resident full model; only2newimages, noQwen/referenceimages. Actual peaks recorded.'})
    request = {'schema': 'tce-compiled-flux-clock-request-v1', 'scope': 'Finite-domain learned-clock partial evaluation with original live denoiser/VAE paid. Not arbitrary-prompt zero-neural generation.',
        'protocol': 'Compile4actualclockserialB1 with original6affines offline. Fresh serving structurally excludes6learnedtensors and loads remainingweights only. Two candidate ownstate images from supplied2E/inheritednoise. Evaluation target values denied in compile/serve. Sealcompletecandidates then compare alreadysealedfulltargets. No newQwen/referencecampaign, precisionsearch orseedsearch.',
        'parentJobID': job['job_id'], 'planSHA256': sha(HERE.parent / 'docs/plans/compiled-flux-clock.md'), 'ownedSources': OWNED,
        'sourceHashes': {f.name: sha(f) for f in payload.iterdir()}, 'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers', 'device': 'cuda',
        'dtype': 'bfloat16', 'batch_size': 1, 'seq_len': 1536, 'height': 512, 'width': 512, 'steps': 4, 'timeoutSeconds': 600}
    request['requestSHA256'] = identity(request); save(root / 'request.json', request); save(payload / 'request.json', request)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [f.name for f in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': request['requestSHA256']}


def submit(root):
    from run import command
    from mrun.client.submit import launch
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root / 'payload', r)
    if (root / 'job.json').exists(): raise ValueError('Already submitted')
    cmd = command()[:-5] + ['python', '-u', 'flux_clock_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = launch(cmd, experiment='tce-compiled-flux-clock-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs={'cuda': True},
        config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width']},
        ram_mb=14336, vram_mb=13000, cpu_threads=4, disk_gb=1, est_wall_s=120, timeout_s=600, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='Clock-only compiler4serialB1 clocks/24affines, then fresh selected-load serving2images/8denoisers/2VAE; sixlearnedtensors absent. Allliveattention/MLP/VAEpaid. NoQwen/newreferenceimages/noise draw. Candidate seal before savedtarget comparison; noaccuracyabort. Twochildren, CPU4/CUDA B1/512, expected~2min/600sfinite/noretries. Nativeparent394733a01129 measured9148MBVRAM/7926MBRSS; compatible14336RAM/13000VRAM exception vs genericB64badplan.')
    result = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root / 'job.json', result); return result


def seal(root):
    from paired_fields import load
    table = read(root / 'clock-table.json'); images = read(root / 'images-candidate.json')
    if images['tableSHA256'] != sha(root / 'clock-table.json'): raise ValueError('Image table binding differs')
    for row in table['rows']:
        for item in [*row['clockOperands'].values(), *row['fields'].values()]: load(item)
    for ports in images['fields'].values():
        for item in ports.values(): load(item)
    for item in images['images'].values():
        if sha(item['path']) != item['sha256']: raise ValueError('Candidate PNG changed')
    save(root / 'candidate-seal.json', {'requestSHA256': read(root / 'request.json')['requestSHA256'], 'tableSHA256': sha(root / 'clock-table.json'),
        'imagesSHA256': sha(root / 'images-candidate.json'), 'fields': images['fields'], 'images': images['images'],
        'scope': 'Complete candidate arrays and clock constants verified before release of saved evaluation targets.'})


def execute(root):
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root, r); started = time.perf_counter(); phases = []
    out = root / 'response-result'; out.mkdir()
    for action in ['compile', 'images']:
        tick = time.perf_counter(); subprocess.run([sys.executable, '-u', str(root / 'flux_clock_study.py'), action, str(root)], check=True, timeout=480)
        phases.append({'action': action, 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
    seal(root); print('CLOCK_CANDIDATES_SEALED', sha(root / 'candidate-seal.json'), flush=True)
    score(root, out, phases, started)


def score(root, out, phases, started):
    from paired_fields import load, retain, metrics, paired
    seal_sha = sha(root / 'candidate-seal.json'); candidate = read(root / 'images-candidate.json'); targets = read(root / 'targets.json')
    table = read(root / 'clock-table.json'); r = read(root / 'request.json'); inputs = read(root / 'inputs.json')
    if candidate['schedules'] != targets['schedules']: raise ValueError('Candidate/reference schedules differ')
    accounts = []; pair_accounts = []; differences = {}; observations = 0
    kept_differences = ['latent4', 'vaeInput', 'decodedRaw', 'rgbFloat', 'pngRGB']
    with (out / 'observations.jsonl').open('x') as stream:
        for case in ['A', 'B']:
            for port in candidate['fields'][case]:
                a = load(candidate['fields'][case][port]); b = load(targets['fields'][case][port])
                accounts.append({'case': case, 'port': port, **metrics(a, b)})
                if port in kept_differences: differences[case + '.' + port] = retain(root / 'arrays/differences', case + '.' + port, a - b)
                if port in ['rgbFloat', 'pngRGB']:
                    ar = a.reshape(512, -1); br = b.reshape(512, -1)
                    for i in range(512):
                        stream.write(__import__('json').dumps({'case': case, 'port': port, 'scanline': i, **metrics(ar[i], br[i])}, allow_nan=False) + '\n'); observations += 1
        for port in candidate['fields']['A']:
            ca, cb, ra, rb = [load(d['fields'][c][port]) for d, c in [(candidate, 'A'), (candidate, 'B'), (targets, 'A'), (targets, 'B')]]
            pair_accounts.append({'port': port, **paired(ca, cb, ra, rb)})
            if port in kept_differences:
                for name, value in [('candidate', cb - ca), ('full', rb - ra), ('error', (cb - ca) - (rb - ra))]:
                    differences['paired.' + name + '.' + port] = retain(root / 'arrays/differences', 'paired.' + name + '.' + port, value)
    if sha(root / 'candidate-seal.json') != seal_sha: raise ValueError('Candidate seal changed')
    totals = {key: sum(c[key] for c in candidate['counts'].values()) for key in ['denoiser', 'context', 'joint', 'single', 'VAE']}
    save(out / 'differences.json', differences)
    for case in ['A', 'B']:
        for group, prefix in [(candidate, 'candidate'), (targets, 'full')]:
            image = group['images'][case]
            if sha(image['path']) != image['sha256']: raise ValueError('Original image changed')
            shutil.copyfile(image['path'], out / (prefix + '-' + case + '.png'))
    for name in ['clock-table.json', 'images-candidate.json', 'candidate-seal.json', 'inputs.json', 'targets.json', 'request.json', 'resource-basis.json', 'frozen-plan.md']:
        shutil.copyfile(root / name, out / name)
    constants = out / 'clock-constants'; constants.mkdir()
    portable_rows = []
    for row in table['rows']:
        portable = {'clock': row['clock']}
        for kind in ['clockOperands', 'fields']:
            portable[kind] = {}
            for name, item in row[kind].items():
                destination = constants / ('clock' + str(row['clock']) + '-' + kind + '-' + name + '.npy')
                shutil.copyfile(item['path'], destination)
                if sha(destination) != item['sha256']: raise ValueError('Portable constant copy differs')
                portable[kind][name] = {**item, 'path': str(destination.relative_to(out))}
        portable_rows.append(portable)
    save(out / 'portable-table.json', {'schema': table['schema'], 'sourceTableSHA256': sha(root / 'clock-table.json'),
        'contract': table['contract'], 'rows': portable_rows, 'scope': 'Byte-identical clock operands with relative transport paths; original learned compilation paid. No image backend by itself.'})
    save(out / 'report.json', {'schema': 'tce-compiled-flux-clock-result-v1', 'requestSHA256': r['requestSHA256'], 'scope': r['scope'],
        'accounts': accounts, 'pairedPromptResponse': pair_accounts, 'observations': observations, 'phases': phases,
        'workerSecondsAfterNumpyImports': time.perf_counter() - started, 'candidateSealSHA256': seal_sha,
        'parentJobID': inputs['parentJobID'], 'parentFullSealSHA256': inputs['parentFullSealSHA256'],
        'work': {**totals, 'newImages': 2, 'newNoiseDraws': 0, 'newQwenCalls': 0, 'newReferenceImages': 0,
            'offlineClockAffines': len(table['offlineLinears']), 'offlineClockNominalMACs': table['offlineNominalLinearMACs'],
            'onlineClockAffines': 0, 'onlineClockGathers': len(candidate['clockCalls']), 'clockTablePayloadBytes': table['tableOutputPayloadBytes'],
            'removedOriginalParameterBytes': table['removedParameterBytes'], 'remainingTransformerParameterBytesLoaded': candidate['remainingParameterBytesLoaded'],
            'avoidedClockMACsPerDenoiserCall': table['offlineNominalLinearMACs'] // 4,
            'avoidedOnlineClockMACs': (table['offlineNominalLinearMACs'] // 4) * totals['denoiser'],
            'actualRemainingTransformerLinearCalls': sum(x['parameter'].startswith('transformer.') for x in candidate['linears']),
            'actualRemainingTransformerLinearMACs': sum(x['nominalMACs'] for x in candidate['linears'] if x['parameter'].startswith('transformer.')),
            'scope': 'Clock-only six-affine elimination. Original live learned attention/MLP/context/VAE remain; MACs nominal geometry, no matched speedup or energy claim.'},
        'mechanics_status': 'complete', 'instrument_status': 'executed', 'terminal_status': 'unassessed',
        'limitations': ['Two existing prompts with supplied sealed E/noise and previously observed targets; no arbitrary-prompt quality claim.',
            'Original offline clock computation, original online denoiser/VAE and historical native encoder/image costs remain paid.',
            'Table applies only to exact deployed clocks/checkpoint/dtype/profile/mode; no dense clock fallback.',
            'Timing includes separate-process setup, full-file identity hashing, selected loading and captures; no matched speedup.']})
    print('COMPILED_CLOCK_DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['stage', 'submit', 'execute', 'compile', 'images', 'collect']); parser.add_argument('run'); args = parser.parse_args()
    if args.action == 'compile':
        from flux_clock_compile import execute as compile_clock
        result = compile_clock(args.run)
    elif args.action == 'images':
        from flux_clock_images import execute as render
        result = render(args.run)
    elif args.action == 'collect':
        from response_study import collect
        result = collect(args.run)
    else: result = globals()[args.action](args.run)
    if result is not None: print(__import__('json').dumps(result))
