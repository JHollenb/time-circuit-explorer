"""One finite discovery/consumer/own-input transfer transaction near compute."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import numpy as np
from contextual_writer_program import (BUDGET, COORDINATES, FINITE, features, feature_names,
    fit_tree, finite_add_array, read, save, sha, route, restore_program)


def floats(words):
    return (np.asarray(words, dtype='<u2').astype('<u4') << 16).view('<f4')


def field(path, shape, dtype='bf16'):
    path = Path(path)
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size, 'shape': list(shape), 'dtype': dtype}


def native(item):
    path = Path(item['path'])
    if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']: raise ValueError('Native file differs')
    words = np.load(path, allow_pickle=False)
    if words.shape != (1, 512, 2560) or words.dtype != np.dtype('<u2') or hashlib.sha256(words.tobytes()).hexdigest() != item['payloadSHA256']:
        raise ValueError('Native state ABI differs')
    return floats(words)[0]


def raw(item):
    if sha(item['path']) != item['sha256'] or Path(item['path']).stat().st_size != item['bytes']: raise ValueError('Owned prediction differs')
    return floats(np.fromfile(item['path'], dtype='<u2').reshape(item['shape']))


def compile_lexical(root, source):
    start = time.perf_counter(); store = Path(source['atomStore']); manifest = read(root/'atom-manifest.json')
    if sha(store/'manifest.json') != source['atomManifestSHA256']: raise ValueError('Atom manifest differs')
    coverage_path = store/manifest['coverage']['path']
    if sha(coverage_path) != manifest['coverage']['sha256']: raise ValueError('Coverage differs')
    coverage = np.load(coverage_path, allow_pickle=False)
    if not np.array_equal(coverage, np.arange(151936, dtype='<u4')): raise ValueError('Complete vocabulary required')
    out = root/'lexical'; out.mkdir(); hashes = hashlib.sha256(); count = 0; bytes_read = 0
    with (out/'h0.bf16').open('xb') as table, (out/'row-hashes.bin').open('xb') as row_hashes:
        for item in manifest['shards']:
            p = store/item['path']
            if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']: raise ValueError('Atom shard differs')
            a = np.load(p, allow_pickle=False)
            if a.dtype != np.dtype('<u2') or list(a.shape) != item['shape'] or a.shape[1] != 8704: raise ValueError('Atom shard geometry differs')
            e = a[:, :2560].copy(); data = e.tobytes(); table.write(data); hashes.update(data)
            for row in e: row_hashes.write(hashlib.sha256(row.tobytes()).digest())
            count += len(e); bytes_read += item['bytes']
    if count != 151936: raise ValueError('Lexical coverage incomplete')
    for key in ('tokenizer', 'template'):
        item = manifest[key]; p = store/item['path']
        if sha(p) != item['sha256']: raise ValueError('Tokenizer source differs')
        shutil.copyfile(p, out/item['path'])
    lexical = {k: manifest[k] for k in ('tokenizer', 'template', 'tokenPolicy')}
    lexical['files'] = {n: {'sha256': sha(out/n), 'bytes': (out/n).stat().st_size} for n in ('h0.bf16', 'row-hashes.bin')}
    lexical['vocabSize'] = count; lexical['width'] = 2560; save(out/'manifest.json', lexical)
    return {'sourceBytesRead': bytes_read, 'compiledAssetBytes': sum(p.stat().st_size for p in out.iterdir()),
            'sourceLearnedArithmetic': 'Historical full atom compilation remains paid; this extraction copies E only.',
            'seconds': time.perf_counter()-start, 'manifestSHA256': sha(out/'manifest.json')}


def serve(root, source, cases, programs, name):
    output = root/name
    allowed = [str(p.resolve()) for p in (root/'lexical').iterdir()]
    rows = []
    for method, folder in programs.items():
        rows.append({'name': method, 'path': str(folder), 'sha256': sha(folder/'program.json')})
        allowed.extend(str(p.resolve()) for p in folder.iterdir())
    spec = {'lexical': str(root/'lexical'), 'cases': cases, 'programs': rows, 'readAllow': allowed,
            'forbiddenPaths': source['forbiddenServingPaths']}
    path = root/(name+'-spec.json'); save(path, spec); start = time.perf_counter()
    subprocess.run([sys.executable, '-u', str(root/'contextual_writer_program.py'), str(path), str(output)], check=True, timeout=120)
    receipt = read(output/'receipt.json')
    if receipt['loadedNumericalRuntimeRoots'] or receipt['newNeuralEvaluations'] != 0: raise ValueError('Serving isolation differs')
    save(root/(name+'-seal.json'), {'receiptSHA256': sha(output/'receipt.json'), 'programs': rows,
        'outputs': [p for row in receipt['rows'] for p in [row['H0'], *row['predictions']]]})
    return receipt, time.perf_counter()-start


def emit_program(root, name, trees, atoms, literals, prefix, occupancy):
    folder = root/name; folder.mkdir(); dictionaries = []
    for i, values in enumerate(atoms):
        path = folder/('slot'+str(i)+'.f32'); values.astype('<f4').tofile(path)
        dictionaries.append({'path': path.name, 'atoms': len(values), 'sha256': sha(path), 'bytes': path.stat().st_size})
    (folder/'prefix.bf16').write_bytes(prefix)
    p = {'schema': 'tce-contextual-writer-program-v1', 'budget': BUDGET, 'width': 2560, 'length': 512,
        'literals': literals, 'featureNames': feature_names(literals), 'trees': trees, 'dictionaries': dictionaries,
        'prefixSHA256': sha(folder/'prefix.bf16'), 'prefixPositions': [0, 1], 'finitePolicy': FINITE,
        'leafOccupancy': occupancy, 'scope': 'Approximate complete H1 from raw input; no full encoder/image or exact native parity.'}
    save(folder/'program.json', p)
    return folder


def fit(root, source, initializer):
    start = time.perf_counter(); x = []; target = []; h0s = []; weights = []; keys = []; packets = {}
    literals = sorted({t for row in initializer['rows'] for t, m in zip(row['packet']['ids'], row['packet']['mask']) if m})[:64]
    prefix = None
    for row in initializer['rows']:
        key = row['id']; item = source['cases'][key]; h0 = raw({**row['H0'], 'shape': [512, 2560]})
        if not np.array_equal(h0, native(item['H0'])): raise ValueError('Own lexical H0 differs from source')
        h1 = native(item['H1']); prefix = Path(item['H1']['path']) if prefix is None else prefix
        packet = row['packet']; packets[key] = packet
        for label, expected in [('ids', item['tokenIDsSHA256']), ('mask', item['maskSHA256'])]:
            if hashlib.sha256(np.asarray(packet[label], dtype='<i8').tobytes()).hexdigest() != expected: raise ValueError('Own native input differs')
        x.extend(features(packet, h0, literals)[2:]); target.extend(h1[2:]); h0s.extend(h0[2:]); keys.extend((key, p) for p in range(2, 512))
        mask = np.asarray(packet['mask'][2:], bool)
        weights.extend(np.where(mask, .5/max(1, int(mask.sum())), .5/max(1, int((~mask).sum())))/4)
    x = np.asarray(x, np.float64); target = np.asarray(target, np.float64); h0s = np.asarray(h0s, np.float64); weights = np.asarray(weights)
    t0, d0, o0 = fit_tree(x, target-h0s, weights, [0]); a0 = np.array([route(t0, row)[0] for row in x]); stage0 = finite_add_array(h0s, d0[a0])
    residual = target-stage0
    t1, d1, o1 = fit_tree(x, residual, weights, list(range(x.shape[1])))
    tc, dc, oc = fit_tree(x, residual, weights, [0])
    def occupied(trace):
        return [{**{k: v for k, v in leaf.items() if k != 'rows'}, 'histories': dict(__import__('collections').Counter(keys[i][0] for i in leaf['rows']))} for leaf in trace]
    native_a = source['cases'][initializer['rows'][0]['id']]['H1']; prefix_data = np.load(native_a['path'], allow_pickle=False)[0, :2].tobytes()
    occupancy = [occupied(o0), occupied(o1)]
    initial = emit_program(root, 'initial', [t0, t1], [d0, d1], literals, prefix_data, occupancy)
    half = emit_program(root, 'half', [t0, t1], [d0, (d1*np.float32(.5)).astype('<f4')], literals, prefix_data, occupancy)
    control = emit_program(root, 'position', [t0, tc], [d0, dc], literals, prefix_data, [occupied(o0), occupied(oc)])
    return {'initial': initial, 'half': half, 'position': control}, {'seconds': time.perf_counter()-start, 'budget': BUDGET,
        'fitRows': len(x), 'features': x.shape[1], 'literals': literals, 'historyRows': dict(__import__('collections').Counter(k for k, p in keys)),
        'finiteResidual': 'Native H1 minus actually rounded slot0 output.', 'weighting': 'Equal history and balanced active/padding on rows2..511.',
        'proposal': 'One offline half-strength slot1 dictionary; routes unchanged. Consumer-guided amplitude selection only.'}


def metric(pred, target, scale=None):
    pred = np.asarray(pred, np.float64); target = np.asarray(target, np.float64); error = pred-target
    scale = target if scale is None else np.asarray(scale, np.float64)
    a = float(np.mean(pred*pred)); b = float(np.mean(target*target)); e = float(np.mean(error*error)); s = float(np.mean(scale*scale))
    return {'errorRMS': e**.5, 'normalizerRMS': s**.5, 'relativeRMS': (e/s)**.5 if s else None,
            'predictionRMS': a**.5, 'targetRMS': b**.5, 'cosine': float(np.mean(pred*target))/(a*b)**.5 if a*b else None}


def parameters(root, source):
    import torch
    checks = []; selected = {}; headers = {}
    for item in source['parameters']:
        path = Path(item['file'])
        if str(path) not in headers:
            asset = next(x for x in source['parameterAssets'] if x['path'] == str(path))
            if path.stat().st_size != asset['bytes'] or sha(path) != asset['sha256']: raise ValueError('Native weight shard differs')
            with path.open('rb') as stream:
                size = struct.unpack('<Q', stream.read(8))[0]; data = stream.read(size)
            if hashlib.sha256(data).hexdigest() != asset['header_sha256']: raise ValueError('Native header differs')
            headers[str(path)] = (json.loads(data), size+8); checks.append(asset)
        header, start = headers[str(path)]; entry = header[item['name']]; lo, hi = entry['data_offsets']
        with path.open('rb') as stream: stream.seek(start+lo); data = stream.read(hi-lo)
        if entry['dtype'] != 'BF16' or len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['payloadSHA256']: raise ValueError('Native selected weight differs')
        selected[item['name'].split('model.layers.1.', 1)[1]] = torch.from_numpy(np.frombuffer(data, dtype='<u2').copy()).view(torch.bfloat16).reshape(entry['shape']).cuda()
    expected = {'input_layernorm.weight': (2560,), 'self_attn.q_proj.weight': (4096, 2560), 'self_attn.k_proj.weight': (1024, 2560), 'self_attn.v_proj.weight': (1024, 2560)}
    if {k: tuple(v.shape) for k, v in selected.items()} != expected: raise ValueError('Consumer parameter geometry differs')
    return selected, checks


def execute(root):
    import torch
    from saturn.titan import Frame, ActSpec, ActRegistry
    from saturn.tensor_hash import tensor_bytes_sha256
    root = Path(root).resolve(); request = read(root/'request.json'); source = read(root/'inputs.json'); start = time.perf_counter()
    from contextual_writer_study import verify
    verify(root, request); out = root/'response-result'; out.mkdir(); torch.set_num_threads(1)
    released = {'transfer': False}; transfer_paths = {str(Path(source['cases'][k][n]['path']).resolve()) for k in ('01', '11') for n in ('H0', 'H1')}
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)) and not released['transfer'] and str(Path(os.fsdecode(args[0])).resolve()) in transfer_paths:
            raise PermissionError('Transfer targets quarantined until final serving seal')
    sys.addaudithook(audit)
    lexical = compile_lexical(root, source); discovery = [x for x in request['cases'] if x['id'] in ('A', 'B', '00', '10')]
    initializer, init_time = serve(root, source, discovery, {}, 'initializer')
    programs, fit_receipt = fit(root, source, initializer); panel, panel_time = serve(root, source, discovery, programs, 'discovery')
    if torch.__version__ != source['torchVersion']: raise ValueError('Native consumer Torch version differs')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision('highest')
    profile = {'bf16ReducedPrecisionReduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'cublasWorkspaceConfig': os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'cudaMatmulTF32': torch.backends.cuda.matmul.allow_tf32,
        'cudnnBenchmark': torch.backends.cudnn.benchmark, 'cudnnDeterministic': torch.backends.cudnn.deterministic,
        'cudnnTF32': torch.backends.cudnn.allow_tf32, 'deterministicAlgorithms': torch.are_deterministic_algorithms_enabled(),
        'float32MatmulPrecision': torch.get_float32_matmul_precision(), 'threads': torch.get_num_threads()}
    if profile != source['numericalProfile']: raise ValueError('Qualified consumer numerical profile differs')
    parameter_tick = time.perf_counter(); weights, asset_checks = parameters(root, source); parameter_seconds = time.perf_counter()-parameter_tick
    registry = ActRegistry(); consumer_rows = []; outputs = {}; typed = []
    weight_names = {'normWeight': 'input_layernorm.weight', 'qWeight': 'self_attn.q_proj.weight', 'kWeight': 'self_attn.k_proj.weight', 'vWeight': 'self_attn.v_proj.weight'}
    weight_slots = {slot: weights[name] for slot, name in weight_names.items()}
    weight_before = {slot: tensor_bytes_sha256(value) for slot, value in weight_slots.items()}
    expected_weights = {x['name'].split('model.layers.1.', 1)[1]: x['payloadSHA256'] for x in source['parameters']}
    if any(weight_before[slot] != expected_weights[name] for slot, name in weight_names.items()): raise ValueError('Live parameter payload binding differs')
    if source['rmsNormEpsilon'] != 1e-6: raise ValueError('RMS epsilon differs')
    spec = ActSpec('contextual_writer.next_operand', 'v1', reads=('H1', *weight_slots), writes=('Q', 'K', 'V'), numerical_contract='qualified-CUDA-BF16-inputnorm-and-three-original-affines',
        write_abis={p: {'shape': [1, 512, d], 'dtype': 'bfloat16'} for p, d in [('Q', 4096), ('K', 1024), ('V', 1024)]})
    def consume(values):
        h = values['H1']; work = h.float(); work = work*torch.rsqrt(work.pow(2).mean(-1, keepdim=True)+source['rmsNormEpsilon'])
        norm = values['normWeight']*work.to(h.dtype); result = {}; calls = []
        for port in ('Q', 'K', 'V'):
            weight = values[port.lower()+'Weight']; result[port] = torch.nn.functional.linear(norm, weight)
            calls.append({'port': port, 'inputShape': list(norm.shape), 'weightShape': list(weight.shape), 'nominalMACs': 512*weight.numel()})
        if any(not torch.isfinite(v).all() for v in result.values()): raise ValueError('Nonfinite consumer')
        return result, {'linears': len(calls), 'nominalMACs': sum(x['nominalMACs'] for x in calls), 'calls': calls, 'attentionCalls': 0, 'fullBlocks': 0}
    registry.register(spec, 'original-selected-maps', consume)
    parent = Frame.from_values(weight_slots, program_fingerprint=request['requestSHA256']); parent_id = parent.fingerprint
    def run_consumer(key, method, value, input_field):
        words = ((np.asarray(value, '<f4').view('<u4') >> 16).astype('<u2'))
        tensor = torch.from_numpy(words.copy()).view(torch.bfloat16).reshape(1, 512, 2560).cuda()
        input_payload = hashlib.sha256(words.tobytes()).hexdigest()
        if input_payload != input_field.get('payloadSHA256', input_field['sha256']): raise ValueError('Actual consumer input differs from retained field')
        act_name = 'native-reference-install:v1' if method == 'native' else 'owned-lexical-install:v1' if method == 'H0' else 'owned-H1-install:v1'
        frame = parent.with_updates({'H1': tensor}, act=act_name, implementation=method, metadata={'case': key, 'method': method})
        torch.cuda.synchronize(); tick = time.perf_counter()
        with torch.inference_mode(): result, evidence = registry.run(spec.identity, 'original-selected-maps', frame)
        torch.cuda.synchronize(); arrays = {}; fields = {}
        for port in ('Q', 'K', 'V'):
            v = result.values[port]; path = root/'arrays'/key/(method+'-'+port+'.npy'); path.parent.mkdir(parents=True, exist_ok=True)
            retained_words = v.cpu().view(torch.uint16).numpy(); np.save(path, retained_words, allow_pickle=False); arrays[port] = v.float().cpu().numpy()[0].astype(np.float64)
            fields[port] = {**field(path, list(v.shape)), 'payloadSHA256': hashlib.sha256(retained_words.tobytes()).hexdigest()}
        outputs[key, method] = arrays; consumer_rows.append({'case': key, 'method': method, 'inputField': input_field, 'inputPayloadSHA256': input_payload,
            'fields': fields, 'secondsWithHashCapture': time.perf_counter()-tick})
        typed.append({'case': key, 'method': method, 'input': frame.to_dict(), 'output': result.to_dict(), 'evidence': evidence.to_dict()})
        if parent.fingerprint != parent_id: raise ValueError('Immutable weight parent changed')
        print('QKV', key, method, flush=True)
    owned = {}; packets = {}; native_states = {}
    def prepare(rows, methods):
        for row in rows:
            key = row['id']; packets[key] = row['packet']; h0 = raw({**row['H0'], 'shape': [512, 2560]}).astype(np.float64); target = native(source['cases'][key]['H1']).astype(np.float64)
            if not np.array_equal(h0, native(source['cases'][key]['H0'])): raise ValueError('Transfer own H0 differs')
            for label, expected in [('ids', source['cases'][key]['tokenIDsSHA256']), ('mask', source['cases'][key]['maskSHA256'])]:
                if hashlib.sha256(np.asarray(row['packet'][label], '<i8').tobytes()).hexdigest() != expected: raise ValueError('Native input binding differs')
            native_states[key] = target; owned[key, 'H0'] = h0
            values = {'native': target, 'H0': h0}
            for method in methods:
                item = next(x for x in row['predictions'] if x['method'] == method); values[method] = raw({**item, 'shape': [512, 2560]}).astype(np.float64); owned[key, method] = values[method]
            input_fields = {'native': source['cases'][key]['H1'], 'H0': row['H0'], **{x['method']: x for x in row['predictions']}}
            for method, value in values.items(): run_consumer(key, method, value, input_fields[method])
    prepare(panel['rows'], ('initial', 'half', 'position'))
    def feedback(method):
        terms = []; numerator = denominator = 0.0
        for key in ('A', 'B', '00', '10'):
            mask = np.asarray(packets[key]['mask'], bool)
            for stratum, selected in [('active', mask), ('padding', ~mask)]:
                for port in ('Q', 'K', 'V'):
                    delta = outputs[key, 'native'][port][selected]-outputs[key, 'H0'][port][selected]
                    pred = outputs[key, method][port][selected]-outputs[key, 'H0'][port][selected]
                    m = metric(pred, delta); terms.append({'case': key, 'stratum': stratum, 'port': port, **m})
                    numerator += m['errorRMS']**2/24; denominator += m['normalizerRMS']**2/24
        return {'score': numerator/denominator if denominator else None, 'errorMeanSquare': numerator, 'referenceIncrementMeanSquare': denominator, 'terms': terms}
    scores = {m: feedback(m) for m in ('initial', 'half')}
    selected = 'half' if scores['half']['score'] is not None and scores['initial']['score'] is not None and scores['half']['score'] < scores['initial']['score'] else 'initial'
    branch_parent = Frame.from_values({'program': {'path': str(programs['initial']), 'sha256': sha(programs['initial']/'program.json')}}, program_fingerprint=request['requestSHA256'])
    proposed = branch_parent.with_updates({'program': {'path': str(programs['half']), 'sha256': sha(programs['half']/'program.json')}}, act='offline-half-atom-proposal:v1', implementation='fixed-amplitude-only')
    accepted_frame = proposed if selected == 'half' else branch_parent
    accepted = root/'accepted'; restored_files = restore_program(programs[selected], accepted)
    save(out/'feedback.json', {'scores': scores, 'selected': selected, 'baselineFrame': branch_parent.to_dict(), 'proposalFrame': proposed.to_dict(),
        'restoredFrame': accepted_frame.to_dict(), 'restoredFiles': restored_files, 'exactRestore': True, 'acceptedProgramSHA256': sha(accepted/'program.json'),
        'scope': 'Discovery-only reversible immutable-program selection, no optimizer rollback or predicate search. Control has no extra feedback proposal.'})
    final, final_time = serve(root, source, request['cases'], {'accepted': accepted, 'position': programs['position']}, 'final')
    for row in final['rows']:
        if row['id'] not in ('A', 'B', '00', '10'): continue
        prior = next(x for x in panel['rows'] if x['id'] == row['id'])
        if row['packet'] != prior['packet'] or row['H0']['sha256'] != prior['H0']['sha256']: raise ValueError('Fresh own initializer changed')
        for method, previous in [('accepted', selected), ('position', 'position')]:
            if next(x for x in row['predictions'] if x['method'] == method)['sha256'] != next(x for x in prior['predictions'] if x['method'] == previous)['sha256']:
                raise ValueError('Fresh restored program output changed')
    final_seal = sha(root/'final-seal.json'); released['transfer'] = True
    print('TRANSFER_TARGET_RELEASE', final_seal, flush=True)
    prepare([row for row in final['rows'] if row['id'] in ('01', '11')], ('accepted', 'position'))
    if len(consumer_rows) != 28: raise ValueError('Physical consumer geometry differs')
    summaries = []; pairs = []; row_count = 0
    with gzip.open(out/'observations.jsonl.gz', 'xt') as stream:
        for key in ('A', 'B', '00', '10', '01', '11'):
            methods = ('H0', 'initial', 'half', 'position') if key in ('A', 'B', '00', '10') else ('H0', 'accepted', 'position')
            h0 = owned[key, 'H0']; mask = np.asarray(packets[key]['mask'], bool)
            for method in methods:
                fields = [('H1', owned[key, method], native_states[key], native_states[key]-h0)]
                fields += [(port, outputs[key, method][port], outputs[key, 'native'][port], outputs[key, 'native'][port]-outputs[key, 'H0'][port]) for port in ('Q', 'K', 'V')]
                for port, pred, target, scale in fields:
                    for stratum, take in [('all', np.ones(512, bool)), ('active', mask), ('padding', ~mask)]:
                        summaries.append({'case': key, 'method': method, 'port': port, 'stratum': stratum, 'rows': int(take.sum()), **metric(pred[take], target[take], scale[take])})
                    for pos in range(512):
                        stream.write(json.dumps({'case': key, 'method': method, 'port': port, 'position': pos, 'active': bool(mask[pos]), **metric(pred[pos], target[pos], scale[pos])}, allow_nan=False)+'\n'); row_count += 1
        for a, b in [('A', 'B'), ('00', '10'), ('01', '11')]:
            methods = ('H0', 'initial', 'half', 'position') if a != '01' else ('H0', 'accepted', 'position')
            same = np.asarray(packets[a]['ids']) == packets[b]['ids']
            for method in methods:
                for port in ('H1', 'Q', 'K', 'V'):
                    target = native_states[b]-native_states[a] if port == 'H1' else outputs[b, 'native'][port]-outputs[a, 'native'][port]
                    pred = owned[b, method]-owned[a, method] if port == 'H1' else outputs[b, method][port]-outputs[a, method][port]
                    for label, take in [('all', np.ones(512, bool)), ('unchangedToken', same)]:
                        pairs.append({'A': a, 'B': b, 'method': method, 'port': port, 'selection': label, 'rows': int(take.sum()), **metric(pred[take], target[take])})
                    for pos in range(512):
                        stream.write(json.dumps({'pair': [a, b], 'method': method, 'port': port, 'position': pos, 'sameToken': bool(same[pos]), **metric(pred[pos], target[pos])}, allow_nan=False)+'\n'); row_count += 1
    if sha(root/'final-seal.json') != final_seal: raise ValueError('Final serving seal changed')
    weight_after = {slot: tensor_bytes_sha256(value) for slot, value in weight_slots.items()}
    if weight_before != weight_after: raise ValueError('Actual native parameter payload mutated')
    save(out/'typed-consumers.json', {'spec': spec.to_dict(), 'rows': typed, 'unchangedWeightParent': parent.to_dict(), 'parameterPayloadBefore': weight_before, 'parameterPayloadAfter': weight_after})
    save(out/'consumer-fields.json', {'rows': consumer_rows, 'parameterAssets': asset_checks, 'parameters': source['parameters']})
    for name in ('request.json', 'inputs.json', 'frozen-plan.md', 'final-seal.json', 'discovery-seal.json', 'initializer-seal.json'): shutil.copyfile(root/name, out/name)
    shutil.copyfile(root/'lexical/manifest.json', out/'lexical-manifest.json')
    for name in ('initializer', 'discovery', 'final'): shutil.copyfile(root/name/'receipt.json', out/(name+'-serving.json'))
    save(out/'programs.json', {name: {'program': read(path/'program.json'), 'programSHA256': sha(path/'program.json'), 'directory': str(path)} for name, path in {**programs, 'accepted': accepted}.items()})
    save(out/'report.json', {'schema': request['schema'], 'requestSHA256': request['requestSHA256'], 'lexicalCompilation': lexical, 'fitting': fit_receipt,
        'selected': selected, 'feedbackSHA256': sha(out/'feedback.json'), 'transferReleasedAfterSealSHA256': final_seal,
        'summaries': summaries, 'pairs': pairs, 'observationRows': row_count, 'workerWallSeconds': time.perf_counter()-start,
        'freshChildWallSeconds': {'initializer': init_time, 'discovery': panel_time, 'final': final_time},
        'consumerWork': {'units': 28, 'originalLinears': 84, 'nominalMACs': 225485783040, 'normalizations': 28, 'fullBlocks': 0, 'images': 0}, 'numericalProfile': profile,
        'parameterAuthenticationLoadSeconds': parameter_seconds, 'authenticatedWeightShardBytes': sum(x['bytes'] for x in asset_checks),
        'selectedParameterBytes': sum(x['bytes'] for x in source['parameters']),
        'metricSemantics': 'Case metrics compare complete H1/QKV endpoints;error normalized by native increment. Their predictionRMS,targetRMS,cosine are endpoints. Feedback compares actual consumer increments;pair metrics compare signed prompt responses.',
        'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'mechanics_status': 'completed', 'instrument_status': 'own-H1-next-operand-consumer',
        'trend_status': 'inspect-all-state-increment-and-signed-response-panels', 'terminal_status': 'arbitrary-prompt-zero-neural-image-generation-unachieved',
        'limitations': ['Public reused-history transfer,not fresh heldout. One novel raw prompt has no native target.', 'Approximate branch/constant program;no exact-source mechanism claim.',
            'No full encoder continuation,attention/MLP/denoiser/VAE/image. Native inputnorm/QKV are paid evaluation only.', 'Dictionary expressivity diagnostic not executed in this first bounded assay;routing/capacity remain coupled.',
            'Child high-water RSS may inherit pre-exec history;not isolated steady memory or a matched speedup benchmark.']})
    print('CONTEXTUAL_WRITER_COMPLETE', flush=True)
