"""One fixed K/V accumulation contrast on authenticated emitted H1."""
import argparse
from pathlib import Path
import runpy
import shutil
import sys
import time
from encoder_atoms import np, read, save, sha, identity, floats

HERE = Path(__file__).resolve().parent
PORTS = ('q', 'k', 'v', 'head', 'R', 'M', 'Hnext')
QUALIFIED = {'flux_functional_qwen_prefix.py': '763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a',
             'qwen_h18_state_history.py': '17084d404f5ae00557e4ad8e94a8b84704787c47ebf9c875228cb452c4857036'}


def verify(root, request):
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']:
        raise ValueError('Request changed')
    for n, h in request['sourceHashes'].items():
        if sha(Path(root) / n) != h: raise ValueError('Payload changed: ' + n)


def stage(root):
    from mrun.client.api import Api
    from publish_context_statistics import checked_collection
    parent = HERE.parent / 'artifacts/masked-kv-own-state-v2'
    c, j, old_request = checked_collection(parent, current_workers=False)
    live = Api().json('GET', '/api/jobs/' + j['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != c['custody']:
        raise ValueError('Authoritative parent differs')
    fields = read(parent / 'result/retained-fields.json')['rows']; source = read(parent / 'result/source.json')
    def ports(lane, index):
        return next(x['ports'] for x in fields if x['case'] == 'short' and x['lane'] == lane and x['index'] == index)
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); payload = root / 'payload'; payload.mkdir()
    for n in ('kv_accumulation.py', 'kv_accumulation_study.py', 'masked_kv.py', 'masked_kv_study.py', 'encoder_atoms.py'):
        shutil.copyfile(HERE / n, payload / n)
    for n, h in QUALIFIED.items():
        path = HERE.parent.parent / 'saturn-pub/packages/mdb/qualification' / n
        if sha(path) != h: raise ValueError('Qualified source changed')
        shutil.copyfile(path, payload / n)
    bound = {'parentJobID': j['job_id'], 'parentRequestSHA256': old_request['requestSHA256'],
        'parentSealSHA256': sha(parent / 'result/prediction-seal.json'),
        'parentReportSHA256': sha(parent / 'result/report.json'),
        'parentRoot': j['result']['diagnostics']['cwd'], 'H1': ports('candidate', 0)['Hnext'],
        'lexical': next(x for x in source['lexicalSeal']['rows'] if x['id'] == 'short' and x['control'] == 'primary'),
        'originalFull': ports('full', 1), 'originalCompact': ports('candidate', 1),
        'parameters': [x for x in source['parameters'] if x['name'].startswith('model.layers.1.')],
        'sourceSpec': source['sourceSpec'], 'sdpaSourceSHA256': source['sdpaSourceSHA256'],
        'parentByteVerificationSHA256': sha(parent / 'retained-array-verification.json'),
        'sourceKind': 'Already-scored short prompt; inherited actual emitted candidate H1, not a new raw-prompt generalization trial.'}
    save(payload / 'source.json', bound)
    r = {'schema': 'tce-kv-accumulation-request-v1', 'parentJobID': j['job_id'],
        'parentRequestSHA256': old_request['requestSHA256'], 'device': 'cuda', 'backend': 'source-bound-torch-bf16',
        'scope': 'One alternative K/V-only accumulation flag; full512/compact28 from own emitted H1. Original consumer, no encoder/FLUX/VAE.',
        'protocol': 'Seal both complete alternative H2 before original raw projections, saved target reads or CPU FP64 diagnostics. No reference refresh, no numerical gate or fallback.',
        'sourceHashes': {f.name: sha(f) for f in payload.iterdir()}, 'timeoutSeconds': 600}
    r['requestSHA256'] = identity(r); save(root / 'request.json', r); save(payload / 'request.json', r)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [f.name for f in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': r['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root / 'payload', r)
    if (root / 'job.json').exists(): raise ValueError('Already submitted')
    cmd = command()[:-5] + ['python', '-u', 'kv_accumulation_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1']
    job = launch(cmd, experiment='tce-kv-accumulation-v1', task='analysis', payload=root / 'payload', pin='beast',
        needs={'cuda': True}, config={'schema': r['schema'], 'requestSHA256': r['requestSHA256'], 'device': 'cuda', 'backend': 'torch', 'seq_len': 512},
        timeout_s=600, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='Fixed KV-only accumulation flag; emittedH1 reused, full512/compact28 completeH2. 2SDPA/16linears, then2CPUFP64 activeXW diagnostics. All512queries/MLP kept; seal before targets. Hostlocal original11weights~202MB, source arrays nearcompute. CUDA+CPU1,defaultplan,expected<5min,600s,no retries. No encoder/FLUX/VAE or kernel sweep.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root / 'job.json', value); return value


def execute(root):
    import torch
    from torch.nn import functional as F
    from safetensors import safe_open
    from transformers import Qwen3Config
    from transformers.models.qwen3 import modeling_qwen3 as native
    from transformers.integrations import sdpa_attention
    from masked_kv import block
    from masked_kv_study import field_metrics
    from kv_accumulation import projection_policy
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root, r); source = read(root / 'source.json')
    start = time.perf_counter(); out = root / 'response-result'; out.mkdir(); raw = root / 'accumulation-arrays'; raw.mkdir()
    reads = []
    def checked(item, target=False):
        path = Path(item['path'])
        if path.is_symlink() or not path.is_relative_to(source['parentRoot']) or sha(path) != item['sha256']:
            raise ValueError('Inherited source changed: ' + str(path))
        reads.append({'path': str(path), 'sha256': item['sha256'], 'target': target})
        return path
    def from_words(a): return torch.from_numpy(a.copy()).view(torch.bfloat16).to('cuda')
    h1 = from_words(np.load(checked(source['H1']), allow_pickle=False)).reshape(1, 512, 2560)
    with np.load(checked(source['lexical']), allow_pickle=False) as archive:
        mask = torch.from_numpy(archive['mask'].copy()).to('cuda')[None, :]
        ids = torch.from_numpy(archive['ids'].copy()).to('cuda')[None, :]
    if int(mask.sum()) != 28: raise ValueError('Declared support differs')
    spec = source['sourceSpec']; model = Path(spec['modelRoot'])
    if sha(native.__file__) != spec['nativeSourceSHA256'] or sha(sdpa_attention.__file__) != source['sdpaSourceSHA256']:
        raise ValueError('Native primitive source differs')
    for n, h in spec['assets'].items():
        if sha(model / n) != h: raise ValueError('Checkpoint metadata differs')
    shard = model / spec['weightFile']['path']
    if shard.stat().st_size != spec['weightFile']['bytes'] or sha(shard) != spec['weightFile']['sha256']:
        raise ValueError('Checkpoint shard differs')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    graph = runpy.run_path(str(root / 'flux_functional_qwen_prefix.py'))
    history = runpy.run_path(str(root / 'qwen_h18_state_history.py'))
    config = Qwen3Config.from_dict(read(model / 'text_encoder/config.json')); profile = graph['_native_profile'](config, 2)
    dispatch = graph['AttentionDispatch'].from_mapping({'backend': 'sdpa', 'gqa_policy': 'repeat_kv', 'source': spec['nativeSourceSHA256']})
    context = history['_context'](graph, ids, mask, profile, dispatch,
        {'init_device': 'cpu', 'init_dtype': 'float32', 'placement_dtype': 'float32', 'source': spec['nativeSourceSHA256']}, torch.bfloat16)
    parameters = {}
    with safe_open(shard, framework='pt', device='cpu') as bank:
        for item in source['parameters']:
            v = bank.get_tensor(item['name'])
            payload_hash = __import__('hashlib').sha256(v.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()
            if v.dtype != torch.bfloat16 or list(v.shape) != item['shape'] or payload_hash != item['payloadSHA256']:
                raise ValueError('Original parameter changed')
            parameters[item['name']] = v.to('cuda')
    versions = [(n, v.data_ptr(), v._version) for n, v in parameters.items()]
    names = {v.data_ptr(): n for n, v in parameters.items()}; active = torch.nonzero(mask[0].bool(), as_tuple=False).flatten()
    phase = {'lane': None, 'policy': None}; packets = {}; primitives = []; times = []; normalized = {}
    def retain(name, value, stored='bf16'):
        path = raw / phase['lane'] / (name + '.npy'); path.parent.mkdir(parents=True, exist_ok=True)
        if not bool(torch.isfinite(value).all()): raise ValueError('Nonfinite retained field')
        a = value.detach().contiguous().cpu()
        np.save(path, a.view(torch.uint16).numpy().copy() if stored == 'bf16' else a.numpy().copy(), allow_pickle=False)
        item = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size, 'shape': list(value.shape), 'dtype': str(value.dtype), 'stored': stored}
        packets.setdefault(phase['lane'], {})[name] = item
    def capture(n, v): retain(n, v)
    old_linear, old_sdpa = F.linear, F.scaled_dot_product_attention
    def linear(x, w, bias=None):
        name = names.get(w.data_ptr()); is_kv = name is not None and any(x in name for x in ('k_proj', 'v_proj'))
        with projection_policy(phase['policy'], is_kv):
            flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            result = old_linear(x, w, bias)
        primitives.append({**phase, 'primitive': 'linear', 'parameter': name, 'inputShape': list(x.shape), 'weightShape': list(w.shape),
            'rows': x.numel() // x.shape[-1], 'nominalMACs': (x.numel() // x.shape[-1]) * w.numel(), 'actualBF16ReducedPrecisionReduction': flag,
            'restoredBF16ReducedPrecisionReduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction})
        if is_kv:
            port = 'rawK' if 'k_proj' in name else 'rawV'; retain(port, result)
            if port == 'rawK': normalized[phase['lane']] = x.detach().clone(); retain('normalizedInput', x)
        return result
    def sdpa(q, k, v, *args, **kwargs):
        primitives.append({**phase, 'primitive': 'SDPA', 'shapes': [list(x.shape) for x in (q, k, v)],
            'actualBF16ReducedPrecisionReduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction})
        return old_sdpa(q, k, v, *args, **kwargs)
    F.linear, F.scaled_dot_product_attention = linear, sdpa
    try:
        with torch.inference_mode(), history['_no_modules']():
            for lane, mode in [('alternative-full', 'zero'), ('alternative-compact', 'compact')]:
                phase.update(lane=lane, policy='alternative'); torch.cuda.synchronize(); tick = time.perf_counter()
                block(graph, h1, parameters, profile, 1, context, dispatch, mask, mode, capture)
                torch.cuda.synchronize(); times.append({'lane': lane, 'wallSecondsIncludingPortWrites': time.perf_counter() - tick})
        save(out / 'prediction-seal.json', {'schema': 'tce-kv-accumulation-seal-v1', 'requestSHA256': r['requestSHA256'],
            'sourceH1': source['H1'], 'nativeTargets': 0, 'ports': packets.copy(), 'readsBeforeSeal': list(reads)})
        print('ACCUMULATION_COHORT_SEALED', sha(out / 'prediction-seal.json'), flush=True)
        # Every reference and diagnostic starts after the complete alternative H2 seal.
        with torch.inference_mode(), history['_no_modules']():
            phase.update(lane='original-raw', policy='original')
            norm = graph['_rms_norm'](h1, parameters['model.layers.1.input_layernorm.weight'], profile['rms_norm_eps'])
            for port in ('k', 'v'):
                value = F.linear(norm, parameters['model.layers.1.self_attn.' + port + '_proj.weight'])
                if port == 'k':
                    k = graph['_rms_norm'](value.view(1, 512, 8, 128), parameters['model.layers.1.self_attn.k_norm.weight'], profile['rms_norm_eps']).transpose(1, 2)
                    retain('k', k * context[3].unsqueeze(1) + graph['_rotate_half'](k) * context[4].unsqueeze(1))
                else: retain('v', value.view(1, 512, 8, 128).transpose(1, 2))
            phase.update(lane='fp64-diagnostic', policy='cpu-fp64')
            x = normalized['alternative-full'].index_select(1, active).to(device='cpu', dtype=torch.float64)
            for port in ('k', 'v'):
                w = parameters['model.layers.1.self_attn.' + port + '_proj.weight'].to(device='cpu', dtype=torch.float64)
                tick = time.perf_counter(); value = x @ w.T
                primitives.append({**phase, 'primitive': 'matmul', 'port': port, 'rows': 28, 'nominalMACs': x.numel() * w.shape[0], 'wallSeconds': time.perf_counter() - tick})
                retain('raw' + port.upper(), value, stored='float64')
                retain('rounded' + port.upper(), value.to(torch.float32).to(torch.bfloat16))
                del w
    finally:
        F.linear, F.scaled_dot_product_attention = old_linear, old_sdpa
    if versions != [(n, v.data_ptr(), v._version) for n, v in parameters.items()]: raise ValueError('Parameters mutated')
    if not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction: raise ValueError('Original flag not restored')
    if sum(x['primitive'] == 'linear' for x in primitives) != 16 or sum(x['primitive'] == 'SDPA' for x in primitives) != 2:
        raise ValueError('Declared instrument census differs')
    for call in primitives:
        if call['primitive'] == 'linear':
            is_kv = any(x in call['parameter'] for x in ('k_proj', 'v_proj'))
            expected = 28 if call['lane'] == 'alternative-compact' and is_kv else 512
            if call['rows'] != expected or not call['restoredBF16ReducedPrecisionReduction']: raise ValueError('Row/policy instrumentation differs')
            if call['actualBF16ReducedPrecisionReduction'] != (call['policy'] == 'original' or not is_kv):
                raise ValueError('Declared K/V-only policy did not execute')
        if call['primitive'] == 'SDPA' and not call['actualBF16ReducedPrecisionReduction']:
            raise ValueError('Alternative flag escaped into consumer')
    def load(item):
        path = Path(item['path'])
        if sha(path) != item['sha256']: raise ValueError('Retained field changed')
        a = np.load(path, allow_pickle=False)
        return (floats(a) if a.dtype == np.uint16 else a).astype(np.float64)
    for lane, ports in read(out / 'prediction-seal.json')['ports'].items():
        for item in ports.values():
            if sha(item['path']) != item['sha256']: raise ValueError('Forecast mutated after seal')
    for lane, key in [('original-full', 'originalFull'), ('original-compact', 'originalCompact')]:
        packets[lane] = source[key]
        for item in packets[lane].values(): checked(item, target=True)
    active_np = mask[0].cpu().numpy().astype(bool); baseline = h1.float().cpu().numpy().astype(np.float64).reshape(512, -1)
    accounts = []; observation_count = 0
    contrasts = [('alternative-compact', 'alternative-full'), ('alternative-full', 'original-full'),
        ('alternative-compact', 'original-full'), ('original-compact', 'original-full')]
    import json
    with (out / 'observations.jsonl').open('x') as stream:
        for left, right in contrasts:
            for port in PORTS:
                a, b = load(packets[left][port]), load(packets[right][port])
                if port in ('q', 'k', 'v'): a, b = a.transpose(0, 2, 1, 3), b.transpose(0, 2, 1, 3)
                a, b = a.reshape(512, -1), b.reshape(512, -1)
                for region, chosen in [('all', np.ones(512, bool)), ('active', active_np), ('padding', ~active_np)]:
                    accounts.append({'channel': left + '-minus-' + right, 'port': port, 'region': region, 'kind': 'endpoint', **field_metrics(a[chosen], b[chosen], retained=True)})
                if port == 'Hnext':
                    for region, chosen in [('all', np.ones(512, bool)), ('active', active_np), ('padding', ~active_np)]:
                        accounts.append({'channel': left + '-minus-' + right, 'port': port, 'region': region, 'kind': 'block-update', **field_metrics(a[chosen] - baseline[chosen], b[chosen] - baseline[chosen])})
                    for position in range(512):
                        stream.write(json.dumps({'channel': left + '-minus-' + right, 'position': position, 'attended': bool(active_np[position]),
                            'endpoint': field_metrics(a[position], b[position], retained=True), 'blockUpdate': field_metrics(a[position] - baseline[position], b[position] - baseline[position])}, allow_nan=False) + '\n')
                        observation_count += 1
    raw_accounts = []; raw_pair_accounts = []
    for port in ('K', 'V'):
        exact = load(packets['fp64-diagnostic']['raw' + port]).reshape(28, -1)
        rounded = load(packets['fp64-diagnostic']['rounded' + port]).reshape(28, -1)
        full = load(packets['alternative-full']['raw' + port]).reshape(512, -1)[active_np]
        compact = load(packets['alternative-compact']['raw' + port]).reshape(28, -1)
        raw_pair_accounts.append({'port': port, 'channel': 'alternative-compact-minus-alternative-full', **field_metrics(compact, full, retained=True)})
        for lane in ('alternative-full', 'alternative-compact', 'original-raw'):
            a = load(packets[lane]['raw' + port]).reshape(-1, 1024)
            if len(a) == 512: a = a[active_np]
            raw_accounts.append({'port': port, 'lane': lane, 'float64Diagnostic': field_metrics(a, exact), 'roundedFP64ViaFP32': field_metrics(a, rounded, retained=True)})
        if port == 'V':
            a = load(packets['original-compact']['v']).transpose(0, 2, 1, 3).reshape(512, -1)[active_np]
            raw_accounts.append({'port': port, 'lane': 'original-compact', 'float64Diagnostic': field_metrics(a, exact), 'roundedFP64ViaFP32': field_metrics(a, rounded, retained=True)})
    reproduction = {port: field_metrics(load(packets['original-raw'][port]), load(packets['original-full'][port]), retained=True) for port in ('k', 'v')}
    normalized_match = torch.equal(normalized['alternative-full'].index_select(1, active), normalized['alternative-compact'])
    save(out / 'retained-fields.json', {'lanes': packets})
    save(out / 'primitive-receipts.json', {'calls': primitives, 'normalizedActiveInputExact': normalized_match})
    save(out / 'source.json', {**source, 'verifiedReads': reads, 'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'transformers': __import__('transformers').__version__}})
    save(out / 'report.json', {'schema': 'tce-kv-accumulation-result-v1', 'requestSHA256': r['requestSHA256'], 'scope': r['scope'],
        'accounts': accounts, 'rawProjectionDiagnostics': raw_accounts, 'rawPairAccounts': raw_pair_accounts,
        'originalPortReproduction': reproduction, 'observations': observation_count,
        'timings': times, 'workerSecondsAfterFrameworkImports': time.perf_counter() - start,
        'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'normalizedActiveInputExact': normalized_match,
        'work': {'linear': 16, 'SDPA': 2, 'cpuFP64Matmul': 2, 'nativeModuleCalls': 0, 'newH1Blocks': 0, 'fullEncoderCalls': 0, 'FLUXCalls': 0, 'VAECalls': 0},
        'mechanics_status': 'complete', 'instrument_status': 'executed', 'terminal_status': 'unassessed',
        'limitations': ['Already-scored one-context numerical diagnostic, not held-out prompt generation.',
            'Alternative-policy agreement does not imply original-profile parity. K/V on omitted addresses intentionally zero.',
            'Original learned block consumers and parameters still evaluated. Single captured phase times are not speedup measurements.',
            'CPU FP64 reference is a diagnostic from the same BF16 operands; rounded path explicitly goes through FP32.']})
    shutil.copyfile(root / 'request.json', out / 'request.json'); print('ACCUMULATION_DONE', flush=True)


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']); p.add_argument('run'); a = p.parse_args()
    if a.action == 'collect':
        from response_study import collect
        result = collect(a.run)
    else: result = globals()[a.action](a.run)
    if result is not None: print(json.dumps(result))
