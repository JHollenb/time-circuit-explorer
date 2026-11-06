"""Raw prompt -> complete masked encoder -> four native image consequences."""
import argparse
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from encoder_atoms import np, read, save, sha, identity

HERE = Path(__file__).resolve().parent
PROMPTS = [{'id': 'A', 'prompt': 'A red cube to the left of a blue sphere on a plain white tabletop. Two separate objects, studio photograph.'},
           {'id': 'B', 'prompt': 'A red cube to the right of a blue sphere on a plain white tabletop. Two separate objects, studio photograph.'}]
OWNED = ['complete_masked_study.py', 'complete_masked_encoder.py', 'paired_image_consumer.py', 'paired_fields.py',
    'masked_kv.py', 'kv_accumulation.py', 'masked_kv_study.py', 'encoder_atoms.py', 'conditioning_demand.py']


def verify(root, request):
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']: raise ValueError('Request changed')
    for name, digest in request['sourceHashes'].items():
        if sha(Path(root) / name) != digest: raise ValueError('Frozen source changed: ' + name)


def profile(torch, threads=1):
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8': raise ValueError('Pinned cuBLAS workspace required')
    torch.set_num_threads(threads); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    return {'cublasWorkspaceConfig': os.environ['CUBLAS_WORKSPACE_CONFIG'], 'deterministicAlgorithms': torch.are_deterministic_algorithms_enabled(),
        'threads': torch.get_num_threads(), 'cudaMatmulTF32': torch.backends.cuda.matmul.allow_tf32,
        'float32MatmulPrecision': torch.get_float32_matmul_precision(), 'bf16ReducedPrecisionReduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'cudnnTF32': torch.backends.cudnn.allow_tf32, 'cudnnBenchmark': torch.backends.cudnn.benchmark, 'cudnnDeterministic': torch.backends.cudnn.deterministic}


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from publish_context_statistics import checked_collection
    from conditioning_demand_study import QUALIFIED
    api = Api(); parent = HERE.parent / 'artifacts/conditioning-image-v2'; custody, job, old = checked_collection(parent, current_workers=False)
    live = api.json('GET', '/api/jobs/' + job['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != custody['custody']: raise ValueError('Noise parent differs')
    prefix_root = HERE.parent / 'artifacts/encoder-atoms-full-v1'; prefix_job = read(prefix_root / 'scheduler-job.json')
    prefix_live = api.json('GET', '/api/jobs/' + prefix_job['job_id'])
    if prefix_live['state'] != 'succeeded' or prefix_live['payload_custody'] != prefix_job['payload_custody']: raise ValueError('Lexical parent differs')
    prefix = read(prefix_root / 'result/report.json'); image_source = read(parent / 'result/source.json')
    binding = read(HERE.parent / 'docs/results/cached-qkv-intake/compact/source-metadata/qwen_source_binding.json')
    checkpoint = read(HERE / 'checkpoint-files.json'); manifest = read(HERE / 'kernel-manifest.json')
    if sha(HERE / 'checkpoint-files.json') != manifest['files']['checkpoint-files.json']: raise ValueError('Original checkpoint binding differs')
    spec = read(HERE / 'encoder-atoms-source.json')
    if spec['modelRoot'] != image_source['modelRoot'] or spec['modelRoot'] != binding['checkpoint']['model_root']: raise ValueError('Source model roots differ')
    for item in binding['encoder_sources']:
        if checkpoint[str(Path(item['path']).relative_to(spec['modelRoot']))] != item['sha256']: raise ValueError('Qwen shard differs')
    preview = _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1536], backend='diffusers', dtype='bfloat16', device='cuda')
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); p = root / 'payload'; p.mkdir()
    for name in OWNED: shutil.copyfile(HERE / name, p / name)
    for name, digest in QUALIFIED.items():
        source_path = HERE.parent.parent / 'saturn-pub/packages/mdb/qualification' / name
        if sha(source_path) != digest: raise ValueError('Qualified graph changed')
        shutil.copyfile(source_path, p / name)
    save(p / 'source.json', {'lexicalSource': spec, 'lexicalParentJobID': prefix_job['job_id'], 'qwenShards': binding['encoder_sources'],
        'sdpaSourceSHA256': 'b72c5c96ea070d594f36e774cfc799160a0ecb157353502a61eb3c5ea1e35b46',
        'imageAssets': image_source['assets'], 'nativeSources': image_source['nativeSources'],
        'noise': read(parent / 'result/noise-seal.json'), 'noiseParentRoot': job['result']['diagnostics']['cwd'],
        'noiseParentJobID': job['job_id'], 'noiseParentRequestSHA256': old['requestSHA256'],
        'noiseSealSHA256': sha(parent / 'result/noise-seal.json'), 'noiseParentByteVerificationSHA256': sha(parent / 'retained-array-verification.json')})
    save(p / 'serve.json', {'store': prefix['compile']['store'], 'manifestSHA256': prefix['manifestSHA256'], 'prompts': PROMPTS})
    save(p / 'resource-basis.json', {'intentFirstPreview': preview, 'measuredNativeJobID': job['job_id'],
        'measuredNativeResult': {k: live['result'][k] for k in ['elapsed_s', 'peak_rss_mb', 'peak_vram_mb']}, 'measuredNativeReservation': live['reservation'],
        'encoderHistoryJobID': 'job-0dddb989549a', 'encoderGeometry': 'One~202MBlayer reused for2serialB1/512streams; never co-resident with image model.',
        'exception': 'Generic diffusion B64/fallback config preview40.2GB is inapplicable. Reuse native same-profile14336MBRAM/13000MBVRAM envelope; current nativeB1 measured9168MBVRAM. Four serial model phases plus one lexical child avoid model overlap. Guard14300MB below card physical-minus-margin; actual peaks retained.'})
    request = {'schema': 'tce-complete-masked-encoder-image-request-v1', 'prompts': PROMPTS, 'manifestSHA256': prefix['manifestSHA256'],
        'planSHA256': sha(HERE.parent / 'docs/plans/complete-masked-encoder-image.md'),
        'scope': 'First-prompt dependency reduction through complete encoder and native images. Most learned work paid; not arbitrary-prompt zero-neural serving.',
        'protocol': 'Two frozen rawstrings; compiled block0 lexicalQKV, compactKV blocks1..26 with fixedpolicy. All512queries/residual/MLP. OwnH/E/latent. Candidate2images seal before original native27block fullreferences and2fullimages. Inheritedseed314159noise; no redraw, layerrescue or precisionsearch.',
        'sourceHashes': {f.name: sha(f) for f in p.iterdir()}, 'ownedSources': OWNED, 'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers',
        'device': 'cuda', 'dtype': 'bfloat16', 'batch_size': 1, 'seq_len': 1536, 'height': 512, 'width': 512, 'steps': 4, 'timeoutSeconds': 900}
    request['requestSHA256'] = identity(request); save(root / 'request.json', request); save(p / 'request.json', request)
    save(p / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [f.name for f in p.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': request['requestSHA256']}


def submit(root):
    from run import command
    from mrun.client.submit import launch
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root / 'payload', r)
    if (root / 'job.json').exists(): raise ValueError('Already submitted')
    cmd = command()[:-5] + ['python', '-u', 'complete_masked_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = launch(cmd, experiment='tce-complete-masked-encoder-image-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs={'cuda': True},
        config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width']},
        ram_mb=14336, vram_mb=13000, cpu_threads=4, disk_gb=2, est_wall_s=300, timeout_s=900, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='Freshleft/right2prompts: atomQKV->compactKVblocks1..26->2nativeimages seal, then2native27blockrefs->2images.108SDPA/750encoderlinears;16denoisers/4VAE. Fourserialchildphases, B1/512, CPU1encoder/4image, ownstates/norescue. Measurednative36aca5f049e6RAM14336/VRAM13000 exception vs badB64plan; no model overlap. Expected<8min,900s,noretries. Paid learned work,notzero-neural.')
    result = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root / 'job.json', result); return result


def lexical(root):
    from encoder_atoms import guard, Prefix
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root, r); spec = read(root / 'serve.json'); guard()
    started = time.perf_counter(); prefix = Prefix(spec['store'], spec['manifestSHA256']); rows = []
    out = root / 'lexical'; out.mkdir()
    for item in r['prompts']:
        tick = time.perf_counter(); values, text = prefix.forward(item['prompt']); elapsed = time.perf_counter() - tick
        path = out / (item['id'] + '.npz'); np.savez(path, **values)
        rows.append({'id': item['id'], 'prompt': item['prompt'], 'path': str(path), 'sha256': sha(path), 'activeTokens': int(values['mask'].sum()),
            'renderedTextSHA256': hashlib.sha256(text.encode()).hexdigest(), 'firstArithmeticSeconds': elapsed})
    save(out / 'prediction-seal.json', {'manifestSHA256': spec['manifestSHA256'], 'rows': rows, 'coverageCount': len(prefix.store.ids),
        'uniqueRowsRead': prefix.store.rowsRead, 'compiledRowPayloadBytesRead': prefix.store.rowBytesRead, 'workerSecondsAfterImports': time.perf_counter() - started,
        'frameworkCalls': 0, 'originalProjections': 0, 'scope': 'Two primary-only raw-string prefix evaluations; complete memoized learned table, no contextual state.'})


def cohort(root, lane):
    from paired_fields import load
    encoder = read(root / ('encoder-' + lane + '.json')); images = read(root / ('images-' + lane + '.json'))
    for group in [encoder['fields'], images['fields']]:
        for ports in group.values():
            for item in ports.values(): load(item)
    for image in images['images'].values():
        if sha(image['path']) != image['sha256']: raise ValueError('Original PNG changed')
    result = {'requestSHA256': read(root / 'request.json')['requestSHA256'], 'encoderSHA256': sha(root / ('encoder-' + lane + '.json')),
        'imagesSHA256': sha(root / ('images-' + lane + '.json')), 'encoderFields': encoder['fields'], 'imageFields': images['fields'], 'images': images['images'],
        'scope': 'Complete own-state outputs sealed before downstream comparisons; all remaining learned computation paid.'}
    save(root / (lane + '-seal.json'), result)


def execute(root):
    root = Path(root).resolve(); request = read(root / 'request.json'); verify(root, request); started = time.perf_counter(); phases = []
    out = root / 'response-result'; out.mkdir()
    for action, lane in [('lexical', None), ('encoder', 'candidate'), ('image', 'candidate'), ('encoder', 'full'), ('image', 'full')]:
        if lane == 'full' and action == 'encoder':
            cohort(root, 'candidate'); print('COMPLETE_CANDIDATES_SEALED', sha(root / 'candidate-seal.json'), flush=True)
        tick = time.perf_counter(); cmd = [sys.executable, '-u', str(root / 'complete_masked_study.py'), action, str(root)] + ([lane] if lane else [])
        subprocess.run(cmd, check=True, timeout=780)
        phases.append({'action': action, 'lane': lane, 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
    cohort(root, 'full'); print('COMPLETE_REFERENCES_SEALED', sha(root / 'full-seal.json'), flush=True)
    score(root, out, phases, started)


def score(root, out, phases, started):
    from paired_fields import load, retain, metrics, paired
    r = read(root / 'request.json'); enc = {lane: read(root / ('encoder-' + lane + '.json')) for lane in ['candidate', 'full']}
    images = {lane: read(root / ('images-' + lane + '.json')) for lane in enc}
    seals = {lane: sha(root / (lane + '-seal.json')) for lane in enc}; accounts = []; paired_accounts = []; differences = {}; rows = 0
    with (out / 'observations.jsonl').open('x') as stream:
        for case in ['A', 'B']:
            lexical = next(x for x in read(root / 'lexical/prediction-seal.json')['rows'] if x['id'] == case)
            with np.load(lexical['path'], allow_pickle=False) as packet: mask = packet['mask'].astype(bool)
            ports = [('H' + str(i), 'encoder') for i in range(1, 28)] + [('E', 'encoder')] + [(p, 'image') for p in ['latent4', 'vaeInput', 'decodedRaw', 'rgbFloat', 'pngRGB']]
            for port, kind in ports:
                source = enc if kind == 'encoder' else images
                a, b = [load(source[lane]['fields'][case][port]) for lane in ['candidate', 'full']]
                accounts.append({'case': case, 'port': port, 'region': 'all', **metrics(a, b)})
                if kind == 'encoder':
                    ar = a.reshape(512, -1); br = b.reshape(512, -1)
                    for region, use in [('active', mask), ('padding', ~mask)]:
                        if use.any(): accounts.append({'case': case, 'port': port, 'region': region, **metrics(ar[use], br[use])})
                    for i in range(512):
                        stream.write(json.dumps({'kind': 'encoder', 'case': case, 'port': port, 'position': i, 'attended': bool(mask[i]), **metrics(ar[i], br[i])}, allow_nan=False) + '\n'); rows += 1
                elif port in ['rgbFloat', 'pngRGB']:
                    ar = a.reshape(512, -1); br = b.reshape(512, -1)
                    for i in range(512):
                        stream.write(json.dumps({'kind': 'image', 'case': case, 'port': port, 'scanline': i, **metrics(ar[i], br[i])}, allow_nan=False) + '\n'); rows += 1
                if port in ['E', 'latent4', 'vaeInput', 'decodedRaw', 'rgbFloat', 'pngRGB']:
                    differences[case + '.' + port] = retain(root / 'arrays/differences', case + '.' + port, a - b)
        for port in ['E', 'latent4', 'vaeInput', 'decodedRaw', 'rgbFloat', 'pngRGB']:
            source = enc if port == 'E' else images
            ca, cb, ra, rb = [load(source[lane]['fields'][case][port]) for lane, case in [('candidate', 'A'), ('candidate', 'B'), ('full', 'A'), ('full', 'B')]]
            paired_accounts.append({'port': port, **paired(ca, cb, ra, rb)})
            for name, value in [('candidate', cb - ca), ('full', rb - ra), ('error', (cb - ca) - (rb - ra))]:
                differences['paired.' + name + '.' + port] = retain(root / 'arrays/differences', 'paired.' + name + '.' + port, value)
    for lane in enc:
        if sha(root / (lane + '-seal.json')) != seals[lane]: raise ValueError('Cohort seal changed')
        for p in images[lane]['images'].values(): shutil.copyfile(p['path'], out / Path(p['path']).name)
        for name in ['encoder-' + lane + '.json', 'images-' + lane + '.json', lane + '-seal.json']: shutil.copyfile(root / name, out / name)
    if images['candidate']['schedules'] != images['full']['schedules']: raise ValueError('Candidate/reference image schedules differ')
    candidate_macs = sum(x['nominalMACs'] for x in enc['candidate']['primitives'] if x['kind'] == 'linear')
    full_macs = sum(x['nominalMACs'] for x in enc['full']['primitives'] if x['kind'] == 'linear')
    saving = {case: 26 * (512 - next(x for x in enc['candidate']['lexicalRows'] if x['id'] == case)['activeRows']) * 2560 * 2048 for case in ['A', 'B']}
    save(out / 'differences.json', differences)
    save(out / 'report.json', {'schema': 'tce-complete-masked-encoder-image-result-v1', 'requestSHA256': r['requestSHA256'], 'scope': r['scope'],
        'accounts': accounts, 'pairedPromptResponse': paired_accounts, 'observations': rows, 'phases': phases, 'workerSecondsAfterNumpyImports': time.perf_counter() - started,
        'work': {'encoderSDPA': 108, 'candidateEncoderLinears': 372, 'fullReferenceEncoderLinears': 378, 'nativeReferenceEncoderBlocks': 54,
            'nativeDenoiserCalls': 16, 'nativeContextProjections': 16, 'nativeJointBlocks': 80, 'nativeSingleBlocks': 320, 'nativeVAEDecodes': 4, 'newImages': 4, 'newNoiseDraws': 0,
            'candidateEncoderNominalLinearMACs': candidate_macs, 'fullReferenceEncoderNominalLinearMACs': full_macs,
            'nominalLinearMACsAvoided': full_macs - candidate_macs, 'blocks1to26KVMACsAvoidedPerPrompt': saving,
            'scope': 'Counters instrumented in phase receipts. Candidate still evaluates original learned affines/attention/MLPs. Linear MACs exclude attention, normalization, gathers/scatters, nonlinearities and original table compilation.'},
        'mechanics_status': 'complete', 'instrument_status': 'executed', 'terminal_status': 'unassessed',
        'limitations': ['Two dependent fresh strings/one inherited noise, not arbitrary-prompt coverage or semantic-quality certification.',
            'Reference uses original native27-block chains on lexical embedding; candidate uses compiled block0 QKV and original learned block0 tail/blocks1..26 with maskedKV.',
            'Candidate and reference images use original native FLUX/VAE. No zero-neural encoder, denoiser or decoder claim.',
            'Separate fresh child processes avoid model overlap and incur extra startup/loading. Captures, source hashes and scoring are paid; no matched speedup.']})
    for name in ['request.json', 'source.json', 'resource-basis.json']: shutil.copyfile(root / name, out / name)
    shutil.copyfile(root / 'lexical/prediction-seal.json', out / 'lexical-seal.json'); print('COMPLETE_MASKED_DONE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['stage', 'submit', 'execute', 'collect', 'lexical', 'encoder', 'image']); p.add_argument('run'); p.add_argument('lane', nargs='?'); a = p.parse_args()
    if a.action in ['execute', 'encoder', 'image'] and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8': raise ValueError('cuBLAS pin required before framework import')
    if a.action == 'collect':
        from response_study import collect
        result = collect(a.run)
    elif a.action == 'encoder':
        from complete_masked_encoder import execute as encoder
        result = encoder(a.run, a.lane)
    elif a.action == 'image':
        from paired_image_consumer import execute as image
        result = image(a.run, a.lane)
    else: result = globals()[a.action](a.run)
    if result is not None: print(json.dumps(result))
