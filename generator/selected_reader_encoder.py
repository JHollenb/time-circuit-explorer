"""Selected-only32-head bank -> own original full-row encoder continuation."""
from pathlib import Path
import hashlib
import json
import runpy
import struct
import time
from encoder_atoms import np, read, save, sha
from paired_fields import retain, payload, load


def execute(root):
    import torch
    from torch.nn import functional as F
    from transformers import Qwen3Config
    from transformers.models.qwen3 import modeling_qwen3 as native
    from transformers.integrations import sdpa_attention
    from conditioning_demand import pack, parameter_from_words
    from selected_reader_study import verify
    from complete_masked_study import profile
    from flux_clock import deny_targets
    root = Path(root).resolve(); request = read(root / 'request.json'); verify(root, request); source = read(root / 'inputs.json')
    deny_targets([root / 'targets.json', *source['forbiddenTargets']])
    tick = time.perf_counter(); cpu = time.process_time(); numerical = profile(torch)
    lane = 'candidate'
    if numerical != source['encoderNumericalProfile'] or torch.__version__ != source['encoderVersions']['torch']:
        raise ValueError('Encoder numerical/version profile differs from retained full reference')
    if __import__('transformers').__version__ != source['encoderVersions']['transformers']: raise ValueError('Encoder framework differs')
    reader = read(root / 'reader-seal.json'); reader_sha = sha(root / 'reader-seal.json')
    if reader['requestSHA256'] != request['requestSHA256']: raise ValueError('Reader request differs')
    lexical_sha = sha(root / 'lexical/prediction-seal.json')
    if lexical_sha != reader['lexicalSealSHA256']: raise ValueError('Reader lexical seal differs')
    spec = source['lexicalSource']; model = Path(spec['modelRoot'])
    for path, digest in [(native.__file__, spec['nativeSourceSHA256']), (sdpa_attention.__file__, source['sdpaSourceSHA256'])]:
        if sha(path) != digest: raise ValueError('Qualified encoder implementation changed')
    for name, digest in spec['assets'].items():
        if sha(model / name) != digest: raise ValueError('Original encoder metadata changed')
    graph = runpy.run_path(str(root / 'flux_functional_qwen_prefix.py'))
    history = runpy.run_path(str(root / 'qwen_h18_state_history.py'))
    config = Qwen3Config.from_dict(read(model / 'text_encoder/config.json')); config._attn_implementation = 'sdpa'
    contract = graph['_native_profile'](config, 27)
    def signature(p):
        s = p.stat(); return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
    files = {}; asset_checks = []
    for item in source['qwenShards']:
        p = Path(item['path']); before = signature(p)
        if before[2] != item['bytes'] or sha(p) != item['sha256']: raise ValueError('Original encoder shard changed')
        with p.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            if not 0 < size < 16 << 20: raise ValueError('Invalid parameter header')
            data = f.read(size)
        if hashlib.sha256(data).hexdigest() != item['header_sha256']: raise ValueError('Original encoder header changed')
        files[str(p)] = {'header': json.loads(data), 'offset': size + 8, 'signature': before}
        asset_checks.append({**item, 'verifiedSignature': before})
    index = read(model / 'text_encoder/model.safetensors.index.json')['weight_map']; parameter_reads = []
    def parameter(name):
        path = model / 'text_encoder' / index[name]; source_file = files[str(path)]; entry = source_file['header'][name]
        if signature(path) != source_file['signature'] or entry['dtype'] != 'BF16': raise ValueError('Parameter source changed')
        lo, hi = entry['data_offsets']
        if hi - lo != int(np.prod(entry['shape'])) * 2: raise ValueError('Parameter byte geometry differs')
        with path.open('rb') as f: f.seek(source_file['offset'] + lo); data = f.read(hi - lo)
        if len(data) != hi - lo: raise ValueError('Truncated parameter')
        parameter_reads.append({'name': name, 'bytes': len(data), 'payloadSHA256': hashlib.sha256(data).hexdigest(), 'file': str(path)})
        return parameter_from_words(np.frombuffer(data, dtype='<u2'), entry['shape'], 'cuda')
    lexical = read(root / 'lexical/prediction-seal.json')
    if lexical['manifestSHA256'] != request['manifestSHA256'] or lexical['coverageCount'] != 151936: raise ValueError('Lexical source differs')
    states = {}; contexts = {}; masks = {}; fields = {}; taps = {}; lexical_rows = []; handoffs = []; times = []; heads = {}; head_reads = []
    raw = root / 'arrays' / lane
    for prompt in request['prompts']:
        case = prompt['id']; record = next(x for x in lexical['rows'] if x['id'] == case)
        if sha(record['path']) != record['sha256']: raise ValueError('Lexical packet changed')
        with np.load(record['path'], allow_pickle=False) as f: values = {n: f[n].copy() for n in f.files}
        def tensor(a): return torch.from_numpy(a.copy()).view(torch.bfloat16).to('cuda')
        mask = torch.from_numpy(values['mask']).to('cuda')[None, :]; ids = torch.from_numpy(values['ids']).to('cuda')[None, :]
        if mask.shape != (1, 512) or not 0 < int(mask.sum()) < 512: raise ValueError('Frozen padded prompt support differs')
        dispatch = graph['AttentionDispatch'].from_mapping({'backend': 'sdpa', 'gqa_policy': 'repeat_kv', 'source': spec['nativeSourceSHA256']})
        context = history['_context'](graph, ids, mask, contract, dispatch,
            {'init_device': 'cpu', 'init_dtype': 'float32', 'placement_dtype': 'float32', 'source': spec['nativeSourceSHA256']}, torch.bfloat16)
        states[case] = tensor(values['e']).view(1, 512, 2560); contexts[case] = (context, dispatch); masks[case] = mask
        fields[case] = {'H0': retain(raw / case, 'H0', states[case])}; taps[case] = {}
        lexical_rows.append({**record, 'activeRows': int(mask.sum()), 'maskSHA256': hashlib.sha256(values['mask'].tobytes()).hexdigest(),
            'tokenIDsSHA256': hashlib.sha256(values['ids'].tobytes()).hexdigest()})
        heads[case] = load(reader['fields'][case], 'cuda')
        if tuple(heads[case].shape) != (1, 512, 4096) or heads[case].dtype != torch.bfloat16: raise ValueError('Complete BF16 head bank required')
    calls = []; phase = {}; weight_names = {}; original_linear = F.linear; original_sdpa = F.scaled_dot_product_attention
    def linear(x, weight, bias=None):
        name = weight_names.get(weight.data_ptr())
        if name is None: raise ValueError('Unbound original affine parameter')
        is_kv = name.endswith(('self_attn.k_proj.weight', 'self_attn.v_proj.weight'))
        rows = x.numel() // x.shape[-1]
        if rows != 512 or not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction: raise ValueError('Unchanged full-row affine profile required')
        calls.append({**phase, 'kind': 'linear', 'parameter': name, 'rows': rows, 'inputShape': list(x.shape), 'weightShape': list(weight.shape),
            'nominalMACs': rows * weight.numel(), 'allowBF16ReducedPrecisionReduction': True})
        return original_linear(x, weight, bias)
    def sdpa(q, k, v, *args, **kwargs):
        context = contexts[phase['case']][0]; mask = kwargs.get('attn_mask', args[0] if args else None)
        if mask is not context[0] or mask.dtype != torch.bool or kwargs.get('is_causal', False) != context[1] or kwargs.get('enable_gqa', False) or kwargs.get('dropout_p', 0.0) != 0.0 or kwargs.get('scale') != 128 ** -.5:
            raise ValueError('Original attention mask/dispatch changed')
        calls.append({**phase, 'kind': 'SDPA', 'shapes': [list(x.shape) for x in (q, k, v)],
            'maskPayloadSHA256': payload(mask), 'isCausal': kwargs.get('is_causal', False), 'scale': kwargs.get('scale'), 'enableGQA': False})
        return original_sdpa(q, k, v, *args, **kwargs)
    F.linear, F.scaled_dot_product_attention = linear, sdpa
    native_blocks = 0
    try:
        for layer_index in range(27):
            suffixes = graph['QWEN_LAYER_PARAMETER_SUFFIXES']
            if layer_index == 0:
                suffixes = [x for x in suffixes if x.startswith(('self_attn.o_proj.', 'post_attention_layernorm.', 'mlp.'))]
            params = {f'model.layers.{layer_index}.' + suffix: parameter(f'model.layers.{layer_index}.' + suffix) for suffix in suffixes}
            weight_names = {v.data_ptr(): n for n, v in params.items()}; versions = [(n, v.data_ptr(), v._version) for n, v in params.items()]
            module = None
            if layer_index > 0:
                with torch.device('meta'): module = native.Qwen3DecoderLayer(config, layer_index)
                module.load_state_dict({n.split(f'model.layers.{layer_index}.', 1)[1]: v for n, v in params.items()}, assign=True)
                module.eval().requires_grad_(False)
            with torch.inference_mode():
                for prompt in request['prompts']:
                    case = prompt['id']; phase.update(case=case, layer=layer_index); context, dispatch = contexts[case]
                    actual = payload(states[case]); previous = fields[case]['H' + str(layer_index)]
                    if actual != previous['payloadSHA256']: raise ValueError('Encoder consumed another state')
                    torch.cuda.synchronize(); started = time.perf_counter()
                    if module is not None:
                        native_blocks += 1
                        result = module(states[case], attention_mask=context[0], position_ids=context[2], past_key_values=None,
                            use_cache=False, position_embeddings=(context[3], context[4]))
                    else:
                        def p(n): return params['model.layers.0.' + n + '.weight']
                        with history['_no_modules']():
                            head = heads[case]
                            if payload(head) != reader['fields'][case]['payloadSHA256']: raise ValueError('Wrong actual selected-only head input')
                            head_reads.append({'case': case, 'actualHeadPayloadSHA256': payload(head), 'readerFieldSHA256': reader['fields'][case]['sha256']})
                            residual = states[case] + F.linear(head.reshape(1, 512, -1), p('self_attn.o_proj'))
                            post = graph['_rms_norm'](residual, p('post_attention_layernorm'), contract['rms_norm_eps'])
                            result = residual + F.linear(F.silu(F.linear(post, p('mlp.gate_proj'))) * F.linear(post, p('mlp.up_proj')), p('mlp.down_proj'))
                    states[case] = result
                    item = retain(raw / case, 'H' + str(layer_index + 1), result); fields[case]['H' + str(layer_index + 1)] = item
                    torch.cuda.synchronize(); times.append({**phase, 'wallSecondsWithCapture': time.perf_counter() - started})
                    handoffs.append({**phase, 'actualInputPayloadSHA256': actual, 'inputFieldSHA256': previous['sha256'], 'outputPayloadSHA256': item['payloadSHA256']})
                    if layer_index + 1 in (9, 18, 27): taps[case][layer_index + 1] = result
                    print('ENCODER_BLOCK', lane, case, layer_index, flush=True)
            if versions != [(n, v.data_ptr(), v._version) for n, v in params.items()]: raise ValueError('Original parameters changed')
            del params, module; weight_names = {}; torch.cuda.empty_cache()
        for case in fields: fields[case]['E'] = retain(raw / case, 'E', pack(taps[case]))
    finally: F.linear, F.scaled_dot_product_attention = original_linear, original_sdpa
    if sum(x['kind'] == 'SDPA' for x in calls) != 52 or sum(x['kind'] == 'linear' for x in calls) != 372 or native_blocks != 52:
        raise ValueError('Encoder primitive census differs')
    for path, item in files.items():
        if signature(Path(path)) != item['signature']: raise ValueError('Encoder source mutated')
    if sha(root / 'reader-seal.json') != reader_sha: raise ValueError('Reader seal changed')
    if sha(root / 'lexical/prediction-seal.json') != lexical_sha: raise ValueError('Reader lexical seal changed')
    save(root / ('encoder-' + lane + '.json'), {'lane': lane, 'requestSHA256': request['requestSHA256'], 'fields': fields, 'lexicalRows': lexical_rows,
        'lexicalSealSHA256': sha(root / 'lexical/prediction-seal.json'), 'readerSealSHA256': reader_sha, 'actualHeadReads': head_reads,
        'handoffs': handoffs, 'primitives': calls, 'nativeComponentBlocks': native_blocks, 'parameterReads': parameter_reads,
        'assetChecks': asset_checks, 'timings': times, 'workerSecondsAfterImports': time.perf_counter() - tick,
        'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'cpuSecondsAfterImports': time.process_time() - cpu,
        'torchVersion': torch.__version__, 'transformersVersion': __import__('transformers').__version__, 'numericalProfile': numerical, 'profile': contract,
        'scope': 'Raw lexical H0 plus selected-only causal all32head bank; original block0 tail and own native full512-row blocks1..26. No native-state refresh or compactKV numerical change.'})
