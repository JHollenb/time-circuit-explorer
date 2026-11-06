"""Three presealed H2 states through the actual demanded conditioning suffix."""
import argparse
from pathlib import Path
import runpy
import shutil
import struct
import time
from encoder_atoms import np, read, save, sha, identity, floats

HERE = Path(__file__).resolve().parent
QUALIFIED = {'flux_functional_qwen_prefix.py': '763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a',
             'qwen_h18_state_history.py': '17084d404f5ae00557e4ad8e94a8b84704787c47ebf9c875228cb452c4857036'}
FLUX_SOURCE = '0a3c56ed3ca28a416492b50953a433f20b8a89fd26c7b7a49b1b6b45d5236006'


def verify(root, r):
    if identity({k: v for k, v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request changed')
    for n, h in r['sourceHashes'].items():
        if sha(Path(root) / n) != h: raise ValueError('Staged source changed: ' + n)


def stage(root):
    from mrun.client.api import Api
    from publish_context_statistics import checked_collection
    parent = HERE.parent / 'artifacts/kv-accumulation-v1'; c, j, request = checked_collection(parent, current_workers=False)
    live = Api().json('GET', '/api/jobs/' + j['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != c['custody']: raise ValueError('Parent job differs')
    fields = read(parent / 'result/retained-fields.json')['lanes']; previous = read(parent / 'result/source.json')
    binding = read(HERE.parent / 'docs/results/cached-qkv-intake/compact/source-metadata/qwen_source_binding.json')
    if binding['checkpoint']['model_root'] != previous['sourceSpec']['modelRoot']: raise ValueError('Original checkpoint differs')
    kernel = read(HERE / 'kernel-manifest.json')
    for n in ('parameter-artifacts.json', 'checkpoint-files.json'):
        if sha(HERE / n) != kernel['files'][n]: raise ValueError('Original parameter recipes changed')
    checkpoint = read(HERE / 'checkpoint-files.json')
    for shard in binding['encoder_sources']:
        relative = str(Path(shard['path']).relative_to(previous['sourceSpec']['modelRoot']))
        if checkpoint.get(relative) != shard['sha256']: raise ValueError('Encoder shard differs from original generator binding')
    recipe = next(x['source'] for x in read(HERE / 'parameter-artifacts.json')['artifacts'] if x['source']['tensor'] == 'context_embedder.weight')
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); payload = root / 'payload'; payload.mkdir()
    for n in ('conditioning_demand.py', 'conditioning_demand_study.py', 'masked_kv_study.py', 'encoder_atoms.py'):
        shutil.copyfile(HERE / n, payload / n)
    for n, h in QUALIFIED.items():
        p = HERE.parent.parent / 'saturn-pub/packages/mdb/qualification' / n
        if sha(p) != h: raise ValueError('Qualified arithmetic changed')
        shutil.copyfile(p, payload / n)
    source = {'parentJobID': j['job_id'], 'parentRequestSHA256': request['requestSHA256'],
        'parentSealSHA256': sha(parent / 'result/prediction-seal.json'), 'parentReportSHA256': sha(parent / 'result/report.json'),
        'parentByteVerificationSHA256': sha(parent / 'retained-array-verification.json'),
        'allowedSourceRoots': [j['result']['diagnostics']['cwd'], previous['parentRoot']],
        'states': {label: fields[lane]['Hnext'] for label, lane in [('full', 'original-full'), ('old', 'original-compact'), ('new', 'alternative-compact')]},
        'lexical': previous['lexical'], 'sourceSpec': previous['sourceSpec'], 'sdpaSourceSHA256': previous['sdpaSourceSHA256'],
        'qwenShards': binding['encoder_sources'], 'conditioningPack': binding['tokenizer_runtime']['conditioning_pack'],
        'fluxProjection': recipe, 'fluxConfigSHA256': checkpoint['transformer/config.json'],
        'fluxSourceSHA256': FLUX_SOURCE,
        'scope': 'One already-scored short prompt; consume actual presealed full/oldcompact/newcompact H2. No further approximation or native state refresh.'}
    save(payload / 'source.json', source)
    r = {'schema': 'tce-conditioning-demand-request-v1', 'parentJobID': j['job_id'],
        'parentRequestSHA256': request['requestSHA256'], 'scope': source['scope'],
        'protocol': '75 original blocks2..26, three native context projections; layer-major parameter reuse, original B1/512 geometry. Seal all retained states before comparison. No denoiser/VAE or neural elimination claim.',
        'sourceHashes': {f.name: sha(f) for f in payload.iterdir()}, 'device': 'cuda', 'backend': 'source-bound-torch-bf16', 'timeoutSeconds': 600}
    r['requestSHA256'] = identity(r); save(root / 'request.json', r); save(payload / 'request.json', r)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [f.name for f in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': r['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root / 'payload', r)
    if (root / 'job.json').exists(): raise ValueError('Already submitted')
    cmd = command()[:-5] + ['python', '-u', 'conditioning_demand_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1']
    job = launch(cmd, experiment='tce-conditioning-demand-v1', task='analysis', payload=root / 'payload', pin='beast',
        needs={'cuda': True}, config={'schema': r['schema'], 'requestSHA256': r['requestSHA256'], 'device': 'cuda', 'backend': 'torch', 'seq_len': 512},
        timeout_s=600, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='3presealedH2 states->H9/H18/H27/E/nativecontextprojection.75originalblocks/75SDPA/528linears; no extra compaction. Layer-major11weights~202MB reused across3serial B1/512 branches, then released. Source shards8GB hash, selectedprojection47MB. CUDA CPU1/defaultplan,expected<5min,600s,no retries. Arrays nearcompute; no denoiser/VAE/image.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root / 'job.json', value); return value


def execute(root):
    import hashlib
    import inspect
    import json
    import torch
    from torch.nn import functional as F
    from transformers import Qwen3Config
    from transformers.models.qwen3 import modeling_qwen3 as native
    from transformers.integrations import sdpa_attention
    from diffusers import Flux2Transformer2DModel
    from diffusers.models.transformers import transformer_flux2
    from diffusers.pipelines.flux2 import pipeline_flux2_klein
    from conditioning_demand import pack, alignment, parameter_from_words
    from masked_kv_study import field_metrics
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root, r); source = read(root / 'source.json')
    start = time.perf_counter(); out = root / 'response-result'; out.mkdir(); raw = root / 'demand-arrays'; raw.mkdir()
    spec = source['sourceSpec']; model = Path(spec['modelRoot'])
    for actual, expected in [(native.__file__, spec['nativeSourceSHA256']), (sdpa_attention.__file__, source['sdpaSourceSHA256']),
        (pipeline_flux2_klein.__file__, spec['pipelineSourceSHA256']), (transformer_flux2.__file__, source['fluxSourceSHA256'])]:
        if sha(actual) != expected: raise ValueError('Pinned implementation source differs')
    for n, h in spec['assets'].items():
        if sha(model / n) != h: raise ValueError('Bound metadata changed')
    if sha(model / 'transformer/config.json') != source['fluxConfigSHA256']: raise ValueError('FLUX config differs')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    def signature(p):
        s = p.stat(); return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
    files = {}; file_receipts = []; parameters_read = []
    def register(path, full_hash=None, expected_size=None, header_hash=None):
        path = Path(path); before = signature(path)
        if expected_size is not None and before[2] != expected_size: raise ValueError('Source file size changed')
        if full_hash is not None and sha(path) != full_hash: raise ValueError('Source shard changed')
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if not 0 < length <= 16 << 20: raise ValueError('Invalid safetensors header')
            data = stream.read(length)
        actual_header = hashlib.sha256(data).hexdigest()
        if header_hash and actual_header != header_hash: raise ValueError('Source header changed')
        files[str(path)] = {'offset': 8 + length, 'header': json.loads(data), 'signature': before}
        file_receipts.append({'path': str(path), 'bytes': before[2], 'fullSHA256Verified': full_hash,
            'headerSHA256': actual_header, 'scope': 'Full file byte hash' if full_hash else 'Whole-header digest recorded; selected header entry and payload verified at load, no new full-transformer-file hash claim'})
    for shard in source['qwenShards']:
        register(shard['path'], shard['sha256'], shard['bytes'], shard['header_sha256'])
    projection = source['fluxProjection']; projection_path = model / projection['file']; register(projection_path)
    index = read(model / 'text_encoder/model.safetensors.index.json')['weight_map']
    def load_tensor(path, name, expected=None):
        path = Path(path); info = files[str(path)]; item = info['header'][name]
        if signature(path) != info['signature'] or item['dtype'] != 'BF16': raise ValueError('Source changed/dtype differs')
        lo, hi = item['data_offsets']; count = int(np.prod(item['shape']))
        if hi - lo != 2 * count or hi + info['offset'] > info['signature'][2]: raise ValueError('Parameter byte geometry differs')
        with path.open('rb') as stream: stream.seek(info['offset'] + lo); data = stream.read(hi - lo)
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != hi - lo: raise ValueError('Truncated parameter')
        if expected and (item['shape'] != expected['shape'] or item['data_offsets'] != expected['data_offsets'] or digest != expected['payload_sha256']):
            raise ValueError('Selected original projection differs')
        parameters_read.append({'name': name, 'file': str(path), 'shape': item['shape'], 'bytes': len(data), 'payloadSHA256': digest})
        return parameter_from_words(np.frombuffer(data, dtype='<u2'), item['shape'], 'cuda')
    def inherited(item):
        p = Path(item['path'])
        if p.is_symlink() or not any(p.is_relative_to(x) for x in source['allowedSourceRoots']) or sha(p) != item['sha256']:
            raise ValueError('Inherited state changed')
        return p
    states = {}; histories = {}; taps = {}; consumers = []; times = []
    for branch, item in source['states'].items():
        words = np.load(inherited(item), allow_pickle=False)
        if words.shape != (1, 512, 2560) or words.dtype != np.uint16: raise ValueError('Source H2 geometry differs')
        states[branch] = torch.from_numpy(words.copy()).view(torch.bfloat16).to('cuda')
        histories[branch] = {'H2': {**item, 'payloadSHA256': hashlib.sha256(words.tobytes()).hexdigest()}}; taps[branch] = {}
    with np.load(inherited(source['lexical']), allow_pickle=False) as packet:
        mask = torch.from_numpy(packet['mask'].copy()).to('cuda')[None, :]; ids = torch.from_numpy(packet['ids'].copy()).to('cuda')[None, :]
    if int(mask.sum()) != 28: raise ValueError('Declared source support differs')
    graph = runpy.run_path(str(root / 'flux_functional_qwen_prefix.py')); history = runpy.run_path(str(root / 'qwen_h18_state_history.py'))
    config = Qwen3Config.from_dict(read(model / 'text_encoder/config.json')); profile = graph['_native_profile'](config, 27)
    dispatch = graph['AttentionDispatch'].from_mapping({'backend': 'sdpa', 'gqa_policy': 'repeat_kv', 'source': spec['nativeSourceSHA256']})
    context = history['_context'](graph, ids, mask, profile, dispatch,
        {'init_device': 'cpu', 'init_dtype': 'float32', 'placement_dtype': 'float32', 'source': spec['nativeSourceSHA256']}, torch.bfloat16)
    def retain(branch, name, value):
        if value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()): raise ValueError('Invalid emitted state')
        path = raw / branch / (name + '.npy'); path.parent.mkdir(parents=True, exist_ok=True)
        words = value.detach().contiguous().cpu().view(torch.uint16).numpy().copy()
        np.save(path, words, allow_pickle=False)
        item = {'path': str(path), 'sha256': sha(path), 'payloadSHA256': hashlib.sha256(words.tobytes()).hexdigest(),
            'bytes': path.stat().st_size, 'shape': list(value.shape), 'dtype': 'torch.bfloat16'}
        histories[branch][name] = item; return item
    phase = {'branch': None, 'layer': None}; calls = []; weight_names = {}; old_linear = F.linear; old_sdpa = F.scaled_dot_product_attention
    def linear(x, weight, bias=None):
        calls.append({**phase, 'primitive': 'linear', 'parameter': weight_names.get(weight.data_ptr()), 'rows': x.numel() // x.shape[-1],
            'inputShape': list(x.shape), 'weightShape': list(weight.shape), 'nominalMACs': x.numel() // x.shape[-1] * weight.numel(),
            'allowBF16ReducedPrecisionReduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction})
        return old_linear(x, weight, bias)
    def sdpa(q, k, v, *args, **kwargs):
        actual_mask = kwargs.get('attn_mask', args[0] if args else None)
        causal = kwargs.get('is_causal', False); gqa = kwargs.get('enable_gqa', False)
        scale = kwargs.get('scale'); dropout = kwargs.get('dropout_p', 0.0)
        if actual_mask is not context[0] or actual_mask.dtype != torch.bool or causal != context[1] or gqa or scale != 128 ** -.5 or dropout != 0.0:
            raise ValueError('Declared original SDPA profile differs')
        calls.append({**phase, 'primitive': 'SDPA', 'shapes': [list(x.shape) for x in (q, k, v)], 'maskShape': list(actual_mask.shape),
            'maskDtype': str(actual_mask.dtype), 'isCausal': causal, 'enableGQA': gqa, 'scale': scale, 'dropoutP': dropout})
        return old_sdpa(q, k, v, *args, **kwargs)
    F.linear, F.scaled_dot_product_attention = linear, sdpa
    try:
        with torch.inference_mode(), history['_no_modules']():
            for layer in range(2, 27):
                parameters = {f'model.layers.{layer}.' + suffix: load_tensor(model / 'text_encoder' / index[f'model.layers.{layer}.' + suffix], f'model.layers.{layer}.' + suffix) for suffix in graph['QWEN_LAYER_PARAMETER_SUFFIXES']}
                weight_names = {v.data_ptr(): n for n, v in parameters.items()}
                versions = [(n, v.data_ptr(), v._version) for n, v in parameters.items()]
                for branch in states:
                    phase.update(branch=branch, layer=layer); previous = histories[branch]['H' + str(layer)]
                    actual_input = hashlib.sha256(states[branch].detach().contiguous().cpu().view(torch.uint16).numpy().tobytes()).hexdigest()
                    if actual_input != previous['payloadSHA256']: raise ValueError('Block read wrong previous branch state')
                    torch.cuda.synchronize(); tick = time.perf_counter()
                    states[branch] = history['_block'](graph, states[branch], parameters, profile, layer, context, dispatch, None, branch)
                    emitted = retain(branch, 'H' + str(layer + 1), states[branch]); torch.cuda.synchronize()
                    times.append({**phase, 'wallSecondsIncludingStateWrite': time.perf_counter() - tick})
                    consumers.append({**phase, 'inputSHA256': previous['sha256'], 'actualInputPayloadSHA256': actual_input,
                        'outputSHA256': emitted['sha256'], 'source': 'Own previously emitted branch state; no refresh'})
                    if layer + 1 in (9, 18, 27): taps[branch][layer + 1] = states[branch]
                if versions != [(n, v.data_ptr(), v._version) for n, v in parameters.items()]: raise ValueError('Original weights mutated')
                del parameters; weight_names.clear()
                print('DEMAND_LAYER_COMPLETE', layer, flush=True)
            for branch in states: retain(branch, 'E', pack(taps[branch]))
        # Extract the original native component from a meta-only FLUX container.
        # Other components have no resident parameter payload and never execute.
        with torch.device('meta'): flux = Flux2Transformer2DModel.from_config(read(model / 'transformer/config.json'))
        component = flux.context_embedder.to_empty(device='cuda')
        weight = load_tensor(projection_path, projection['tensor'], projection)
        component.load_state_dict({'weight': weight}, strict=True, assign=True); component.eval().requires_grad_(False)
        if component.bias is not None or tuple(component.weight.shape) != (3072, 7680): raise ValueError('Native consumer geometry differs')
        weight_names = {component.weight.data_ptr(): 'context_embedder.weight'}; projection_consumers = []
        def prehook(module, args):
            value = args[0]; branch = phase['branch']
            words = value.detach().contiguous().cpu().view(torch.uint16).numpy()
            digest = hashlib.sha256(words.tobytes()).hexdigest()
            item = histories[branch]['E']
            if digest != item['payloadSHA256']: raise ValueError('Native projection read wrong branch E')
            projection_consumers.append({'branch': branch, 'EFileSHA256': item['sha256'], 'actualInputPayloadSHA256': digest, 'shape': list(value.shape), 'nativeComponent': type(module).__name__})
        handle = component.register_forward_pre_hook(prehook)
        version = (component.weight.data_ptr(), component.weight._version)
        with torch.inference_mode():
            for branch in states:
                phase.update(branch=branch, layer='context_embedder'); e = pack(taps[branch]); retain(branch, 'C', component(e))
        handle.remove()
        if version != (component.weight.data_ptr(), component.weight._version): raise ValueError('Projection mutated')
    finally: F.linear, F.scaled_dot_product_attention = old_linear, old_sdpa
    if len(projection_consumers) != 3 or sum(c['primitive'] == 'linear' for c in calls) != 528 or sum(c['primitive'] == 'SDPA' for c in calls) != 75:
        raise ValueError('Declared consumer instrument did not execute')
    if any(c['primitive'] == 'linear' and (c['rows'] != 512 or not c['allowBF16ReducedPrecisionReduction']) for c in calls): raise ValueError('Unexpected suffix approximation/profile')
    for path, data in files.items():
        if signature(Path(path)) != data['signature']: raise ValueError('Original source mutated during consumption')
    save(out / 'consumer-seal.json', {'schema': 'tce-conditioning-demand-consumer-seal-v1', 'requestSHA256': r['requestSHA256'], 'histories': histories,
        'projectionConsumers': projection_consumers, 'scope': 'Charged learned consumer outputs, not a zero-neural suffix prediction or image.'})
    print('DEMAND_CONSUMERS_SEALED', sha(out / 'consumer-seal.json'), flush=True)
    def load(item):
        if sha(item['path']) != item['sha256']: raise ValueError('Retained state changed')
        return floats(np.load(item['path'], allow_pickle=False)).astype(np.float64).reshape(512, -1)
    active = mask[0].cpu().numpy().astype(bool); accounts = []; count = 0
    with (out / 'observations.jsonl').open('x') as stream:
        for port in ['H' + str(i) for i in range(2, 28)] + ['E', 'C']:
            baseline, old, new = [load(histories[b][port]) for b in ('full', 'old', 'new')]
            for region, chosen in [('all', np.ones(512, bool)), ('active', active), ('padding', ~active)]:
                accounts.append({'port': port, 'region': region, 'old': field_metrics(old[chosen], baseline[chosen], retained=True),
                    'new': field_metrics(new[chosen], baseline[chosen], retained=True), 'alignment': alignment(old[chosen], new[chosen], baseline[chosen])})
            for i in range(512):
                stream.write(json.dumps({'port': port, 'position': i, 'attended': bool(active[i]), 'old': field_metrics(old[i], baseline[i], retained=True),
                    'new': field_metrics(new[i], baseline[i], retained=True), 'alignment': alignment(old[i], new[i], baseline[i])}, allow_nan=False) + '\n'); count += 1
    for ports in histories.values():
        for item in ports.values():
            if sha(item['path']) != item['sha256']: raise ValueError('Sealed consumer output changed')
    save(out / 'retained-fields.json', {'histories': histories})
    save(out / 'primitive-receipts.json', {'calls': calls, 'stateConsumers': consumers, 'projectionConsumers': projection_consumers,
        'parameterHydrations': 25, 'nativeContextProjectionCalls': 3, 'noNativeEncoderOrDenoiserCalls': True})
    save(out / 'source.json', {**source, 'parameterPayloads': parameters_read, 'sourceFileChecks': file_receipts,
        'pipelineMethod': {'name': '_get_qwen3_prompt_embeds', 'sourceSHA256': hashlib.sha256(inspect.getsource(pipeline_flux2_klein.Flux2KleinPipeline._get_qwen3_prompt_embeds).encode()).hexdigest()},
        'versions': {'torch': torch.__version__, 'transformers': __import__('transformers').__version__, 'diffusers': __import__('diffusers').__version__}})
    save(out / 'report.json', {'schema': 'tce-conditioning-demand-result-v1', 'requestSHA256': r['requestSHA256'], 'scope': r['scope'],
        'accounts': accounts, 'observations': count, 'timings': times, 'workerSecondsAfterFrameworkImports': time.perf_counter() - start,
        'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'work': {'originalQwenBlocks': 75, 'SDPA': 75, 'linear': 528, 'nativeContextProjectionCalls': 3,
            'fullEncoderCalls': 0, 'denoiserCalls': 0, 'VAECalls': 0, 'newImages': 0},
        'mechanics_status': 'complete', 'instrument_status': 'executed', 'terminal_status': 'unassessed',
        'limitations': ['Actual learned consumer observations from three already-sealed H2 states, not zero-neural suffix forecasts.',
            'One retrospectively selected short context, dependent states/rows; no universal stability, semantic or image-quality conclusion.',
            'Full encoder shards are byte-hashed; FLUX projection uses verified original selected payload/header, not a fresh full-transformer-file hash.',
            'Captured phase times include state writes; no matched speedup experiment. All remaining learned operations are paid.']})
    shutil.copyfile(root / 'request.json', out / 'request.json'); print('DEMAND_DONE', flush=True)


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']); p.add_argument('run'); a = p.parse_args()
    if a.action == 'collect':
        from response_study import collect
        result = collect(a.run)
    else: result = globals()[a.action](a.run)
    if result is not None: print(json.dumps(result))
