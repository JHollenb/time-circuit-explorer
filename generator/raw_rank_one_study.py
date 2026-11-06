"""One frozen reader from raw strings through its own full encoder and images."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import np, read, save, sha, identity

HERE = Path(__file__).resolve().parent
EXPERIMENT = ['raw_rank_one_study.py', 'raw_rank_one_reader.py', 'raw_rank_one_encoder.py', 'raw_rank_one_images.py',
    'paired_fields.py', 'masked_kv_study.py', 'conditioning_demand.py', 'flux_clock.py', 'flux_clock_study.py', 'complete_masked_study.py']
PROMPTS = [{'id': 'A', 'prompt': 'A red cube to the left of a blue sphere on a plain white tabletop. Two separate objects, studio photograph.'},
    {'id': 'B', 'prompt': 'A red cube to the right of a blue sphere on a plain white tabletop. Two separate objects, studio photograph.'}]


def verify(root, request):
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']: raise ValueError('Request changed')
    for name, digest in request['sourceHashes'].items():
        if sha(Path(root) / name) != digest: raise ValueError('Frozen input/source changed: ' + name)


def inherited_clock(root, source):
    path = Path(root) / 'clock-table.json'
    if sha(path) != source['clockTableSHA256']: raise ValueError('Inherited clock table changed')
    table = read(path)
    if table['requestSHA256'] != source['clockParentRequestSHA256']: raise ValueError('Inherited clock source request differs')
    if table['contract']['checkpointSHA256'] != source['imageAssets']['transformer/diffusion_pytorch_model.safetensors']:
        raise ValueError('Inherited clock checkpoint differs')
    return table


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from publish_context_statistics import checked_collection
    api = Api(); fullroot = HERE.parent / 'artifacts/complete-masked-image-v1'; clockroot = HERE.parent / 'artifacts/compiled-flux-clock-v1'; rankroot = HERE.parent / 'artifacts/rank-one-tail-v1'
    parents = {}
    for name, path in [('full', fullroot), ('clock', clockroot), ('rank', rankroot)]:
        collection, job, request = checked_collection(path, current_workers=False); live = api.json('GET', '/api/jobs/' + job['job_id'])
        if live['state'] != 'succeeded' or live['payload_custody'] != collection['custody']: raise ValueError('Authenticated ancestor differs')
        parents[name] = (collection, job, request)
    f = fullroot / 'result'; old_source = read(f / 'source.json'); enc = read(f / 'encoder-full.json'); images = read(f / 'images-full.json')
    prefix = read(fullroot / 'payload/serve.json')
    if sha(fullroot / 'payload/serve.json') != parents['full'][2]['sourceHashes']['serve.json']: raise ValueError('Lexical source binding differs')
    rank_spec = read(rankroot / 'result/serve-source.json')
    if rank_spec['asset']['sha256'] != '25d48680c831e45d6841220a3231be1f79e62ebf1303556877613424bd0adafa' or rank_spec['basis']['sha256'] != 'e891ea238bbc53ba9130d25c33c85c9b6fcc9596ce955a9b41751af5a97a5386':
        raise ValueError('Frozen reader geometry/basis differs')
    geometry_job = api.json('GET', '/api/jobs/job-29760e2a3970')
    if geometry_job['state'] != 'succeeded' or not geometry_job['payload_custody']['required']: raise ValueError('Source geometry ancestor unauthenticated')
    clock_inputs = read(clockroot / 'result/inputs.json'); table_path = clockroot / 'result/clock-table.json'
    if images['schedules']['A'] != images['schedules']['B'] or sha(f / 'full-seal.json') != clock_inputs['parentFullSealSHA256']: raise ValueError('Full reference joins differ')
    preview = _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1536], backend='diffusers', dtype='bfloat16', device='cuda')
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); p = root / 'payload'; p.mkdir()
    frozen = {n: h for n, h in parents['rank'][2]['sourceHashes'].items() if n.endswith('.py')}
    owned = sorted(set(EXPERIMENT) | set(frozen))
    for name in owned:
        if name in frozen and sha(HERE / name) != frozen[name]: raise ValueError('Frozen reader implementation changed: ' + name)
        shutil.copyfile(HERE / name, p / name)
    for name in ['flux_functional_qwen_prefix.py', 'qwen_h18_state_history.py']:
        path = fullroot / 'payload' / name
        if sha(path) != parents['full'][2]['sourceHashes'][name]: raise ValueError('Original graph/history source differs')
        shutil.copyfile(path, p / name)
    shutil.copyfile(table_path, p / 'clock-table.json'); shutil.copyfile(HERE.parent / 'docs/plans/raw-rank-one-image.md', p / 'frozen-plan.md')
    forbidden = [x['path'] for group in [enc['fields'], images['fields']] for ports in group.values() for x in ports.values()] + [x['path'] for x in images['images'].values()]
    inputs = {k: clock_inputs[k] for k in ['modelRoot', 'nativeSources', 'imageAssets', 'versions', 'numericalProfile', 'attentionProcessors', 'noise', 'noiseParentRoot']}
    inputs.update(lexicalStore=prefix['store'], lexicalSource=old_source['lexicalSource'], qwenShards=old_source['qwenShards'],
        sdpaSourceSHA256=old_source['sdpaSourceSHA256'], encoderNumericalProfile=enc['numericalProfile'],
        encoderVersions={'torch': enc['torchVersion'], 'transformers': enc['transformersVersion']},
        readerAsset=rank_spec['asset'], readerBasis=rank_spec['basis'], readerGeometry=rank_spec['geometry'],
        readerPolicy='rank1-fullcentroid-moments-fp64-envelope-v1', forbiddenTargets=forbidden,
        schedule=images['schedules']['A'], parentJobID=parents['full'][1]['job_id'], parentRoot=parents['full'][1]['result']['diagnostics']['cwd'],
        parentFullSealSHA256=sha(f / 'full-seal.json'), parentByteVerificationSHA256=sha(fullroot / 'retained-array-verification.json'),
        clockParentJobID=parents['clock'][1]['job_id'], clockParentRoot=parents['clock'][1]['result']['diagnostics']['cwd'],
        clockParentRequestSHA256=parents['clock'][2]['requestSHA256'], clockTableSHA256=sha(table_path),
        rankParentJobID=parents['rank'][1]['job_id'], rankParentRoot=parents['rank'][1]['result']['diagnostics']['cwd'],
        rankParentRequestSHA256=parents['rank'][2]['requestSHA256'], geometryParentRoot=geometry_job['result']['diagnostics']['cwd'])
    save(p / 'inputs.json', inputs)
    save(p / 'targets.json', {'encoderFields': enc['fields'], 'fields': images['fields'], 'images': images['images'], 'schedules': images['schedules'],
        'parentEncoderSHA256': sha(f / 'encoder-full.json'), 'parentImagesSHA256': sha(f / 'images-full.json'),
        'scope': 'Previously observed complete full targets, read for scoring only after complete own-state candidates seal.'})
    save(p / 'resource-basis.json', {'intentFirstPreview': preview, 'measuredImageJobID': parents['clock'][1]['job_id'],
        'measuredImageResult': {k: parents['clock'][0]['result'][k] for k in ['elapsed_s', 'peak_rss_mb', 'peak_vram_mb']},
        'measuredEncoderJobID': parents['full'][1]['job_id'], 'measuredEncoderResult': {k: parents['full'][0]['result'][k] for k in ['elapsed_s', 'peak_rss_mb', 'peak_vram_mb']},
        'exception': 'GenericB64/fallback40.2GBpreview is inapplicable. Reuse compatible14336MBRAM/13000MBVRAM envelope, measuredselectedimage8848MBVRAM/8201MBRSS. ReaderCPU1, oneencoderlayer/2serialB1streamsCPU1, then imageCPU4; separatechildren/no model co-residency. Expected180s,900sfinite,no retries.'})
    request = {'schema': 'tce-raw-rank-one-image-request-v1', 'prompts': PROMPTS, 'manifestSHA256': parents['full'][2]['manifestSHA256'],
        'scope': 'Frozen dynamic32-head reader from raw strings through own full encoder and native images. Most original learned work paid; not arbitrary-prompt zero-neural generation.',
        'protocol': 'Rawstrings->completeatoms->frozenrankone32headbank->originalblock0tail->own nativefull512-row blocks1..26/E->2images with inheritedexactclocks/noise. No suppliedQKV/nativecontextrefresh/exactheads/newfit/newrefs/seedsearch/compactKV on perturbedstates. Sealcompletecandidates before targetvalues.',
        'planSHA256': sha(p / 'frozen-plan.md'), 'ownedSources': owned, 'frozenReaderSources': frozen,
        'sourceHashes': {x.name: sha(x) for x in p.iterdir()}, 'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers', 'device': 'cuda',
        'dtype': 'bfloat16', 'batch_size': 1, 'seq_len': 1536, 'height': 512, 'width': 512, 'steps': 4, 'timeoutSeconds': 900}
    request['requestSHA256'] = identity(request); save(root / 'request.json', request); save(p / 'request.json', request)
    save(p / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [x.name for x in p.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': request['requestSHA256']}


def submit(root):
    from run import command
    from mrun.client.submit import launch
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root / 'payload', r)
    if (root / 'job.json').exists(): raise ValueError('Already submitted')
    cmd = command()[:-5] + ['python', '-u', 'raw_rank_one_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = launch(cmd, experiment='tce-raw-rank-one-image-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast', needs={'cuda': True},
        config={k: r[k] for k in ['schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width']},
        ram_mb=14336, vram_mb=13000, cpu_threads=4, disk_gb=2, est_wall_s=180, timeout_s=900, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='2 raw prompts->frozen32 rank1 heads->own full512-row encoder/E->2 images.52 encoderSDPA/372 linears;8 denoisers/2VAE/824 image linears.3 children CPU1/1/4,B1 CUDA.FullKV/index/moments/fallback paid.No fit/exactheads/state refresh/newrefs/noise draws.Seal before targets.Inherited exact clocks.14336RAM/13000VRAM measured31811b103879 exception vsbadB64plan.180s expected,900s cap,no retries.')
    result = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root / 'job.json', result); return result


def seal(root):
    from paired_fields import load
    reader = read(root / 'reader-seal.json'); enc = read(root / 'encoder-candidate.json'); images = read(root / 'images-candidate.json')
    encoder_ports = {'H' + str(i) for i in range(28)} | {'E'}
    image_ports = {'C' + str(i) for i in range(4)} | {'velocity' + str(i) for i in range(4)} | {'latent' + str(i) for i in range(1, 5)} | {'vaeInput', 'decodedRaw', 'rgbFloat', 'pngRGB'}
    for group in [reader['fields'], enc['fields'], images['fields']]:
        if set(group) != {'A', 'B'}: raise ValueError('Candidate case coverage differs')
    for case in ['A', 'B']:
        if set(enc['fields'][case]) != encoder_ports or set(images['fields'][case]) != image_ports: raise ValueError('Candidate field coverage differs')
    if enc['readerSealSHA256'] != sha(root / 'reader-seal.json') or images['encoderSHA256'] != sha(root / 'encoder-candidate.json'): raise ValueError('Candidate source joins differ')
    if images['tableSHA256'] != sha(root / 'clock-table.json'): raise ValueError('Image clock identity differs')
    for group in [reader['fields'], *enc['fields'].values(), *images['fields'].values()]:
        for item in group.values(): load(item)
    for item in [*reader['reader']['rows'], *reader['reader']['states'], *reader['lexicalRows'], *images['images'].values()]:
        if sha(item['path']) != item['sha256']: raise ValueError('Candidate packet/state/image changed')
    save(root / 'candidate-seal.json', {'requestSHA256': read(root / 'request.json')['requestSHA256'], 'readerSHA256': sha(root / 'reader-seal.json'),
        'encoderSHA256': sha(root / 'encoder-candidate.json'), 'imagesSHA256': sha(root / 'images-candidate.json'), 'clockTableSHA256': sha(root / 'clock-table.json'),
        'readerFields': reader['fields'], 'encoderFields': enc['fields'], 'imageFields': images['fields'], 'images': images['images'],
        'scope': 'All complete own-state candidate outputs authenticated before saved target values are released to scoring.'})


def execute(root):
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root, r); started = time.perf_counter(); phases = []
    out = root / 'response-result'; out.mkdir()
    for action in ['reader', 'encoder', 'images']:
        tick = time.perf_counter(); subprocess.run([sys.executable, '-u', str(root / 'raw_rank_one_study.py'), action, str(root)], check=True, timeout=780)
        phases.append({'action': action, 'wallSecondsIncludingFreshProcess': time.perf_counter() - tick})
    seal(root); print('RAW_READER_CANDIDATES_SEALED', sha(root / 'candidate-seal.json'), flush=True)
    score(root, out, phases, started)


def score(root, out, phases, started):
    import json
    from paired_fields import load, retain, metrics, paired
    r = read(root / 'request.json'); reader = read(root / 'reader-seal.json'); enc = read(root / 'encoder-candidate.json'); images = read(root / 'images-candidate.json')
    target = read(root / 'targets.json'); seal_sha = sha(root / 'candidate-seal.json'); accounts = []; pair_accounts = []; differences = {}; observations = 0
    if images['schedules'] != target['schedules']: raise ValueError('Image schedule differs from original reference')
    for case in ['A', 'B']:
        if enc['fields'][case]['H0']['payloadSHA256'] != target['encoderFields'][case]['H0']['payloadSHA256']: raise ValueError('Common raw input/embedding identity differs')
    with (out / 'observations.jsonl').open('x') as stream:
        for case in ['A', 'B']:
            row = next(x for x in reader['lexicalRows'] if x['id'] == case)
            with np.load(row['path'], allow_pickle=False) as lexical: mask = lexical['mask'].astype(bool)
            ports = [('H' + str(i), enc['fields'], target['encoderFields']) for i in range(1, 28)] + [('E', enc['fields'], target['encoderFields'])] + [(p, images['fields'], target['fields']) for p in images['fields'][case]]
            for port, candidate_fields, reference_fields in ports:
                a = load(candidate_fields[case][port]); b = load(reference_fields[case][port]); accounts.append({'case': case, 'port': port, 'region': 'all', **metrics(a, b)})
                if port.startswith('H') or port == 'E':
                    ar = a.reshape(512, -1); br = b.reshape(512, -1)
                    for region, use in [('active', mask), ('padding', ~mask)]:
                        if use.any(): accounts.append({'case': case, 'port': port, 'region': region, **metrics(ar[use], br[use])})
                    for i in range(512):
                        stream.write(json.dumps({'case': case, 'port': port, 'position': i, 'attended': bool(mask[i]), **metrics(ar[i], br[i])}, allow_nan=False) + '\n'); observations += 1
                elif port in ['rgbFloat', 'pngRGB']:
                    ar = a.reshape(512, -1); br = b.reshape(512, -1)
                    for i in range(512):
                        stream.write(json.dumps({'case': case, 'port': port, 'scanline': i, **metrics(ar[i], br[i])}, allow_nan=False) + '\n'); observations += 1
                if port in ['E', 'latent4', 'vaeInput', 'decodedRaw', 'rgbFloat', 'pngRGB']:
                    differences[case + '.' + port] = retain(root / 'arrays/differences', case + '.' + port, a - b)
        for port in ['H1', 'H2', 'H9', 'H18', 'H27', 'E', *images['fields']['A']]:
            cand, refs = (enc['fields'], target['encoderFields']) if port.startswith('H') or port == 'E' else (images['fields'], target['fields'])
            ca, cb, ra, rb = [load(fields[case][port]) for fields, case in [(cand, 'A'), (cand, 'B'), (refs, 'A'), (refs, 'B')]]
            pair_accounts.append({'port': port, **paired(ca, cb, ra, rb)})
            if port in ['E', 'latent4', 'vaeInput', 'decodedRaw', 'rgbFloat', 'pngRGB']:
                for name, value in [('candidate', cb - ca), ('full', rb - ra), ('error', (cb - ca) - (rb - ra))]:
                    differences['paired.' + name + '.' + port] = retain(root / 'arrays/differences', 'paired.' + name + '.' + port, value)
    if sha(root / 'candidate-seal.json') != seal_sha: raise ValueError('Candidate seal changed')
    for case in ['A', 'B']:
        for group, prefix in [(images, 'candidate'), (target, 'full')]:
            item = group['images'][case]
            if sha(item['path']) != item['sha256']: raise ValueError('Original image changed')
            shutil.copyfile(item['path'], out / (prefix + '-' + case + '.png'))
    for name in ['request.json', 'inputs.json', 'targets.json', 'resource-basis.json', 'frozen-plan.md', 'clock-table.json', 'reader-seal.json', 'encoder-candidate.json', 'images-candidate.json', 'candidate-seal.json', 'serve-source.json']:
        shutil.copyfile(root / name, out / name)
    save(out / 'differences.json', differences)
    encoder_calls = enc['primitives']; totals = {k: sum(x[k] for x in images['counts'].values()) for k in ['denoiser', 'context', 'joint', 'single', 'VAE']}
    save(out / 'report.json', {'schema': 'tce-raw-rank-one-image-result-v1', 'requestSHA256': r['requestSHA256'], 'scope': r['scope'],
        'accounts': accounts, 'pairedPromptResponse': pair_accounts, 'observations': observations, 'phases': phases,
        'workerSecondsAfterNumpyImports': time.perf_counter() - started, 'candidateSealSHA256': seal_sha,
        'readerWork': reader['reader']['counts'], 'work': {**totals, 'newImages': 2, 'newReferenceImages': 0, 'newNoiseDraws': 0,
            'readerFrameworkCalls': 0, 'readerOriginalDenseProjections': 0, 'sourceBasisRecompiled': False, 'exactHeadReferences': 0,
            'nativeEncoderBlocks': enc['nativeComponentBlocks'], 'encoderSDPA': sum(x['kind'] == 'SDPA' for x in encoder_calls),
            'encoderLinears': sum(x['kind'] == 'linear' for x in encoder_calls), 'encoderNominalLinearMACs': sum(x['nominalMACs'] for x in encoder_calls if x['kind'] == 'linear'),
            'nativeTransformerLinears': sum(x['parameter'].startswith('transformer.') for x in images['linears']),
            'nativeTransformerNominalLinearMACs': sum(x['nominalMACs'] for x in images['linears'] if x['parameter'].startswith('transformer.')),
            'newClockCompilations': 0, 'onlineClockAffines': 0, 'onlineClockGathers': len(images['clockCalls']),
            'scope': 'Only first-layer contextual read is approximated; fullKV/index/moments/envelopes/fallback and original learned encoder/image/VAE remain paid.'},
        'mechanics_status': 'complete', 'instrument_status': 'executed', 'terminal_status': 'unassessed',
        'limitations': ['Two known prompts/one inherited noise and previously observed targets, no arbitrary-prompt quality certificate.',
            'Frozen rank-one implementation mixes approximation and finite arithmetic; no claim of pure rank error.',
            'Native full512-row suffix consumes candidate state, no reference refresh. Original learned maps/VAE and historical lexical/clock compilation remain paid.',
            'First-layer supplied-QKV boundary removed, not a complete zero-neural contextual encoder or image generator. No matched speedup.']})
    print('RAW_RANK_ONE_IMAGE_DONE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['stage', 'submit', 'execute', 'reader', 'encoder', 'images', 'collect']); parser.add_argument('run'); args = parser.parse_args()
    if args.action in ['reader', 'encoder', 'images']:
        module = __import__('raw_rank_one_' + args.action); result = module.execute(args.run)
    elif args.action == 'collect':
        from response_study import collect
        result = collect(args.run)
    else: result = globals()[args.action](args.run)
    if result is not None: print(__import__('json').dumps(result))
