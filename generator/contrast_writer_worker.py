"""One bounded paired-writer fit, native consumer future, and sealed transfer."""
import collections
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from contextual_writer_program import BUDGET, features, route, finite_add_array, read, save, sha, restore_program
from contextual_writer_worker import native, raw, floats, field, serve, emit_program, parameters, metric
from contrast_writer_fit import fit_contrast_tree


def load_port(item):
    p = Path(item['path'])
    if sha(p) != item['sha256'] or p.stat().st_size != item['bytes']: raise ValueError('Reused port bytes differ')
    words = np.load(p, allow_pickle=False)
    if list(words.shape) != item['shape'] or words.dtype != np.dtype('<u2') or hashlib.sha256(words.tobytes()).hexdigest() != item['payloadSHA256']:
        raise ValueError('Reused port payload differs')
    return floats(words)[0].astype(np.float64)


def fit(root, source, initializer, endpoint):
    p = read(endpoint/'program.json'); literals = p['literals']; rows = initializer['rows']
    actual_literals = sorted({t for row in rows for t, m in zip(row['packet']['ids'], row['packet']['mask']) if m})[:64]
    if actual_literals != literals: raise ValueError('Discovery feature family changed')
    d0 = np.fromfile(endpoint/p['dictionaries'][0]['path'], '<f4').reshape(-1, 2560)
    x = []; h0s = []; targets = []; weights = []; keys = []; packets = {}; offsets = {}
    for row in rows:
        key = row['id']; packet = row['packet']; item = source['cases'][key]
        h0 = raw({**row['H0'], 'shape': [512, 2560]}); h1 = native(item['H1'])
        if not np.array_equal(h0, native(item['H0'])): raise ValueError('Own discovery H0 differs')
        for label, expected in [('ids', item['tokenIDsSHA256']), ('mask', item['maskSHA256'])]:
            if hashlib.sha256(np.asarray(packet[label], '<i8').tobytes()).hexdigest() != expected: raise ValueError('Discovery packet differs')
        offsets[key] = len(keys); packets[key] = packet
        x.extend(features(packet, h0, literals)[2:]); h0s.extend(h0[2:]); targets.extend(h1[2:]); keys.extend((key, pos) for pos in range(2, 512))
        mask = np.asarray(packet['mask'][2:], bool)
        if not mask.any() or mask.all(): raise ValueError('Frozen discovery strata are not both populated')
        weights.extend(np.where(mask, .5/int(mask.sum()), .5/int((~mask).sum()))/4)
    x, h0s, targets, weights = [np.asarray(v, np.float64) for v in (x, h0s, targets, weights)]
    stage0 = finite_add_array(h0s, d0[np.asarray([route(p['trees'][0], row)[0] for row in x])]).astype(np.float64)
    residual = targets-stage0; edges = []; pair_weights = []
    for a, b in [('A', 'B'), ('00', '10')]:
        same = np.asarray(packets[a]['ids'][2:]) == packets[b]['ids'][2:]
        mask = np.asarray(packets[a]['mask'][2:], bool)
        if packets[a]['mask'] != packets[b]['mask']: raise ValueError('Paired strata differ')
        for take in (same & mask, same & ~mask):
            positions = np.flatnonzero(take)
            if not len(positions): raise ValueError('Frozen paired stratum empty')
            for pos in positions:
                edges.append((offsets[a]+int(pos), offsets[b]+int(pos))); pair_weights.append(1/(4*len(positions)))
    edges = np.asarray(edges, np.int64); pair_weights = np.asarray(pair_weights, np.float64)
    weights /= weights.sum(); pair_weights /= pair_weights.sum()
    if abs(float(weights.sum())-1) > len(weights)*np.finfo(np.float64).eps or abs(float(pair_weights.sum())-1) > len(pair_weights)*np.finfo(np.float64).eps:
        raise ValueError('Normalized discovery measures differ beyond finite summation bound')
    energy = float(np.einsum('i,ij,ij->', weights, targets-h0s, targets-h0s))
    change = targets[edges[:, 1]]-targets[edges[:, 0]]
    response = float(np.einsum('i,ij,ij->', pair_weights, change, change))
    tree, atoms, occupancy, diagnostic = fit_contrast_tree(x, residual, weights, edges, pair_weights, energy, response)
    occupied = [{**{k: v for k, v in row.items() if k != 'rows'}, 'histories': dict(collections.Counter(keys[i][0] for i in row['rows']))} for row in occupancy]
    candidate = emit_program(root, 'candidate', [p['trees'][0], tree], [d0, atoms], literals, (endpoint/'prefix.bf16').read_bytes(), [p['leafOccupancy'][0], occupied])
    labels = np.asarray([route(tree, row)[0] for row in x]); finite = finite_add_array(stage0, atoms[labels]).astype(np.float64)
    def losses(pred):
        error = pred-targets; paired = (pred[edges[:, 1]]-pred[edges[:, 0]])-change
        e = float(np.einsum('i,ij,ij->', weights, error, error)); t = float(np.einsum('i,ij,ij->', pair_weights, paired, paired))
        return {'endpointSquaredError': e, 'responseSquaredError': t, 'endpointRatio': e/energy if energy else None, 'responseRatio': t/response if response else None}
    diagnostic.update({'fitRows': len(x), 'features': x.shape[1], 'pairRows': len(edges), 'literals': literals,
        'budget': BUDGET, 'slot0SHA256': p['dictionaries'][0]['sha256'], 'finiteLosses': losses(finite),
        'targetAndSlot0Policy': 'FP64 R=Y-S from actual finite slot0; relaxed q=P*R. Actual finite response measured against separately formed native YB-YA.',
        'weighting': 'Equal history/active-padding endpoints;equal pair/active-padding unchanged learned rows;each measure sums1.'})
    save(root/'fit.json', diagnostic)
    return candidate, diagnostic


def feedback(outputs, packets, method):
    terms = []; ports = {}
    for port in ('Q', 'K', 'V'):
        endpoints = []; responses = []
        for key in ('A', 'B', '00', '10'):
            mask = np.asarray(packets[key]['mask'], bool); learned = np.arange(512) >= 2
            for stratum, take in [('active', mask & learned), ('padding', ~mask & learned)]:
                m = metric(outputs[key, method][port][take]-outputs[key, 'H0'][port][take], outputs[key, 'native'][port][take]-outputs[key, 'H0'][port][take])
                endpoints.append(m); terms.append({'kind': 'endpointIncrement', 'case': key, 'port': port, 'stratum': stratum, **m})
        for a, b in [('A', 'B'), ('00', '10')]:
            same = np.asarray(packets[a]['ids']) == packets[b]['ids']; mask = np.asarray(packets[a]['mask'], bool)
            for stratum, take in [('active', same & mask & learned), ('padding', same & ~mask & learned)]:
                m = metric((outputs[b, method][port]-outputs[a, method][port])[take], (outputs[b, 'native'][port]-outputs[a, 'native'][port])[take])
                responses.append(m); terms.append({'kind': 'signedResponse', 'pair': [a, b], 'port': port, 'stratum': stratum, **m})
        def component(values):
            e = sum(m['errorRMS']**2 for m in values)/len(values); t = sum(m['normalizerRMS']**2 for m in values)/len(values)
            return {'score': e/t if t else e, 'errorMeanSquare': e, 'referenceMeanSquare': t, 'zeroReferenceUsesAbsolute': not bool(t)}
        ports[port] = {'endpoint': component(endpoints), 'response': component(responses)}
    return {'score': sum(v[k]['score'] for v in ports.values() for k in ('endpoint', 'response'))/3, 'ports': ports, 'terms': terms,
            'policy': 'Equal ports;each port separately normalizes learned-row endpoint increments and unchanged-token signed response,balanced history/pair and active/padding;lambda1. Zero reference uses absolute squared error.'}


class Consumer:
    def __init__(self, root, source, request):
        import torch
        from saturn.titan import Frame, ActSpec, ActRegistry
        from saturn.tensor_hash import tensor_bytes_sha256
        self.torch = torch; self.Frame = Frame; self.tensor_hash = tensor_bytes_sha256; self.root = root; self.source = source
        torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision('highest')
        profile = {'bf16ReducedPrecisionReduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'cublasWorkspaceConfig': os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'cudaMatmulTF32': torch.backends.cuda.matmul.allow_tf32,
            'cudnnBenchmark': torch.backends.cudnn.benchmark, 'cudnnDeterministic': torch.backends.cudnn.deterministic,
            'cudnnTF32': torch.backends.cudnn.allow_tf32, 'deterministicAlgorithms': torch.are_deterministic_algorithms_enabled(),
            'float32MatmulPrecision': torch.get_float32_matmul_precision(), 'threads': torch.get_num_threads()}
        if torch.__version__ != source['torchVersion'] or profile != source['numericalProfile'] or source['rmsNormEpsilon'] != 1e-6: raise ValueError('Qualified consumer profile differs')
        tick = time.perf_counter(); weights, self.assets = parameters(root, source); self.load_seconds = time.perf_counter()-tick
        names = {'normWeight': 'input_layernorm.weight', 'qWeight': 'self_attn.q_proj.weight', 'kWeight': 'self_attn.k_proj.weight', 'vWeight': 'self_attn.v_proj.weight'}
        slots = {k: weights[v] for k, v in names.items()}; self.before = {k: tensor_bytes_sha256(v) for k, v in slots.items()}
        expected = {k: next(x['payloadSHA256'] for x in source['parameters'] if x['name'] == 'model.layers.1.'+v) for k, v in names.items()}
        if self.before != expected: raise ValueError('Native tensor parameter binding differs')
        self.parent = Frame.from_values(slots, program_fingerprint=request['requestSHA256']); self.parent_id = self.parent.fingerprint
        self.spec = ActSpec('contrast_writer.next_operand', 'v1', reads=('H1', *slots), writes=('Q', 'K', 'V'),
            numerical_contract='qualified-CUDA-BF16-inputnorm-and-three-original-affines', write_abis={p: {'shape': [1, 512, d], 'dtype': 'bfloat16'} for p, d in [('Q', 4096), ('K', 1024), ('V', 1024)]})
        def consume(values):
            h = values['H1']; work = h.float(); work = work*torch.rsqrt(work.pow(2).mean(-1, keepdim=True)+source['rmsNormEpsilon'])
            norm = values['normWeight']*work.to(h.dtype); result = {}; calls = []
            for port in ('Q', 'K', 'V'):
                weight = values[port.lower()+'Weight']; result[port] = torch.nn.functional.linear(norm, weight)
                calls.append({'port': port, 'inputShape': list(norm.shape), 'weightShape': list(weight.shape), 'nominalMACs': 512*weight.numel()})
            if any(not torch.isfinite(v).all() for v in result.values()): raise ValueError('Nonfinite consumer')
            return result, {'linears': 3, 'nominalMACs': 8053063680, 'calls': calls, 'attentionCalls': 0, 'fullBlocks': 0}
        self.registry = ActRegistry(); self.registry.register(self.spec, 'original-selected-maps', consume); self.rows = []; self.typed = []

    def run(self, key, item):
        torch = self.torch; value = raw({**item, 'shape': [512, 2560]}); words = (value.view('<u4') >> 16).astype('<u2')
        payload = hashlib.sha256(words.tobytes()).hexdigest()
        if payload != item['sha256']: raise ValueError('Actual H1 consumer input differs')
        tensor = torch.from_numpy(words.copy()).view(torch.bfloat16).reshape(1, 512, 2560).cuda()
        frame = self.parent.with_updates({'H1': tensor}, act='own-contrast-H1-install:v1', implementation='candidate', metadata={'case': key, 'method': 'candidate'})
        torch.cuda.synchronize(); tick = time.perf_counter()
        with torch.inference_mode(): result, evidence = self.registry.run(self.spec.identity, 'original-selected-maps', frame)
        torch.cuda.synchronize(); arrays = {}; fields = {}
        for port in ('Q', 'K', 'V'):
            v = result.values[port]; path = self.root/'arrays'/key/('candidate-'+port+'.npy'); path.parent.mkdir(parents=True, exist_ok=True)
            retained = v.cpu().view(torch.uint16).numpy(); np.save(path, retained, allow_pickle=False)
            fields[port] = {**field(path, list(v.shape)), 'payloadSHA256': hashlib.sha256(retained.tobytes()).hexdigest()}; arrays[port] = v.float().cpu().numpy()[0].astype(np.float64)
        self.rows.append({'case': key, 'method': 'candidate', 'inputField': item, 'inputPayloadSHA256': payload, 'fields': fields, 'secondsWithHashCapture': time.perf_counter()-tick})
        self.typed.append({'case': key, 'method': 'candidate', 'input': frame.to_dict(), 'output': result.to_dict(), 'evidence': evidence.to_dict()})
        if self.parent.fingerprint != self.parent_id: raise ValueError('Native immutable parent changed')
        print('QKV', key, 'candidate', flush=True); return arrays

    def save(self, out):
        after = {k: self.tensor_hash(v) for k, v in self.parent.values.items()}
        if after != self.before or len(self.rows) != 6: raise ValueError('Actual parameter immutability or physical geometry differs')
        save(out/'consumer-fields.json', {'rows': self.rows, 'parameterAssets': self.assets, 'parameters': self.source['parameters']})
        save(out/'typed-consumers.json', {'spec': self.spec.to_dict(), 'rows': self.typed, 'unchangedWeightParent': self.parent.to_dict(), 'parameterPayloadBefore': self.before, 'parameterPayloadAfter': after})


def execute(root):
    from contrast_writer_study import verify
    from saturn.titan import Frame
    root = Path(root).resolve(); request = read(root/'request.json'); verify(root, request); source = read(root/'parent-inputs.json'); reuse = read(root/'reuse.json')
    start = time.perf_counter(); out = root/'response-result'; out.mkdir(); old_consumers = read(root/'parent-consumer-fields.json')['rows']
    catalog = read(root/'parent-programs.json'); endpoint = Path(catalog['accepted']['directory']); position = Path(catalog['position']['directory'])
    released = {'transfer': False}; forbidden = set()
    for row in old_consumers:
        if row['case'] in ('01', '11'): forbidden.update([row['inputField']['path'], *(x['path'] for x in row['fields'].values())])
    forbidden.update(source['cases'][k][n]['path'] for k in ('01', '11') for n in ('H0', 'H1'))
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)) and not released['transfer'] and str(Path(os.fsdecode(args[0])).resolve()) in forbidden:
            raise PermissionError('Transfer numeric fields quarantined before final seal')
    sys.addaudithook(audit)
    (root/'lexical').symlink_to(reuse['lexicalDirectory'], target_is_directory=True)
    checked = []
    for name, expected in reuse['servingAssets'].items():
        if sha(name) != expected: raise ValueError('Reused lexical/program bytes differ')
        checked.append({'path': name, 'sha256': expected, 'bytes': Path(name).stat().st_size})
    source['forbiddenServingPaths'] = sorted(set(source['forbiddenServingPaths']) | {r['inputField']['path'] for r in old_consumers} | {x['path'] for r in old_consumers for x in r['fields'].values()})
    discovery_cases = [x for x in request['cases'] if x['id'] in request['discovery']]
    initializer, init_time = serve(root, source, discovery_cases, {}, 'initializer')
    candidate, fitting = fit(root, source, initializer, endpoint)
    panel, discovery_time = serve(root, source, discovery_cases, {'candidate': candidate, 'endpoint': endpoint, 'position': position}, 'discovery')
    consumer = Consumer(root, source, request); outputs = {}; owned = {}; native_states = {}; packets = {}; reused = []
    def prepare(rows):
        for row in rows:
            key = row['id']; packets[key] = row['packet']; h0 = raw({**row['H0'], 'shape': [512, 2560]}).astype(np.float64)
            target = native(source['cases'][key]['H1']).astype(np.float64)
            if not np.array_equal(h0, native(source['cases'][key]['H0'])): raise ValueError('Own fresh H0 differs')
            for label, expected in [('ids', source['cases'][key]['tokenIDsSHA256']), ('mask', source['cases'][key]['maskSHA256'])]:
                if hashlib.sha256(np.asarray(row['packet'][label], '<i8').tobytes()).hexdigest() != expected: raise ValueError('Fresh native packet differs')
            native_states[key] = target; owned[key, 'H0'] = h0
            pred = {p['method']: p for p in row['predictions']}
            for method in ('candidate', 'endpoint', 'position'): owned[key, method] = raw({**pred[method], 'shape': [512, 2560]}).astype(np.float64)
            for method in ('native', 'H0', 'endpoint', 'position'):
                old_method = reuse['selected'] if method == 'endpoint' and key in request['discovery'] else 'accepted' if method == 'endpoint' else method
                old = next(x for x in old_consumers if x['case'] == key and x['method'] == old_method)
                expected_input = source['cases'][key]['H1'] if method == 'native' else row['H0'] if method == 'H0' else pred[method]
                if old['inputPayloadSHA256'] != expected_input.get('payloadSHA256', expected_input['sha256']): raise ValueError('Reused consumer input identity differs')
                outputs[key, method] = {port: load_port(item) for port, item in old['fields'].items()}
                reused.append({'case': key, 'method': method, 'historicalMethod': old_method, 'inputPayloadSHA256': old['inputPayloadSHA256'], 'fields': old['fields']})
            outputs[key, 'candidate'] = consumer.run(key, pred['candidate'])
    prepare(panel['rows'])
    scores = {m: feedback(outputs, packets, m) for m in ('candidate', 'endpoint')}
    selected = 'candidate' if scores['candidate']['score'] < scores['endpoint']['score'] else 'endpoint'
    parent = Frame.from_values({'program': {'path': str(endpoint), 'sha256': sha(endpoint/'program.json')}}, program_fingerprint=request['requestSHA256'])
    proposed = parent.with_updates({'program': {'path': str(candidate), 'sha256': sha(candidate/'program.json')}}, act='offline-contrast-fit:v1', implementation='paired-relaxation-tree')
    restored = restore_program(candidate if selected == 'candidate' else endpoint, root/'accepted')
    save(out/'feedback.json', {'scores': scores, 'selected': selected, 'baselineFrame': parent.to_dict(), 'proposalFrame': proposed.to_dict(),
        'restoredFrame': (proposed if selected == 'candidate' else parent).to_dict(), 'restoredFiles': restored, 'exactRestore': True, 'acceptedProgramSHA256': sha(root/'accepted/program.json')})
    programs = {'candidate': candidate, 'endpoint': endpoint, 'position': position, 'accepted': root/'accepted'}
    final, final_time = serve(root, source, request['cases'], programs, 'final')
    for row in final['rows']:
        predictions = {p['method']: p for p in row['predictions']}
        if predictions['accepted']['sha256'] != predictions[selected]['sha256']: raise ValueError('Accepted immutable program output differs')
        if row['id'] in request['discovery']:
            previous = next(r for r in panel['rows'] if r['id'] == row['id'])
            if row['packet'] != previous['packet'] or row['H0']['sha256'] != previous['H0']['sha256']: raise ValueError('Fresh initializer changed')
            for p in previous['predictions']:
                if p['sha256'] != predictions[p['method']]['sha256']: raise ValueError('Fresh finite prediction changed')
    seal = sha(root/'final-seal.json'); released['transfer'] = True; print('TRANSFER_TARGET_RELEASE', seal, flush=True)
    prepare([r for r in final['rows'] if r['id'] in request['transfer']])
    summaries = []; pairs = []; count = 0; all_rows = np.ones(512, bool); learned = np.arange(512) >= 2
    with gzip.open(out/'observations.jsonl.gz', 'xt') as stream:
        for key in request['discovery']+request['transfer']:
            mask = np.asarray(packets[key]['mask'], bool)
            for method in ('H0', 'candidate', 'endpoint', 'position'):
                for port in ('H1', 'Q', 'K', 'V'):
                    pred, target, base = (owned[key, method], native_states[key], owned[key, 'H0']) if port == 'H1' else (outputs[key, method][port], outputs[key, 'native'][port], outputs[key, 'H0'][port])
                    for label, take in [('all', all_rows), ('learned', learned), ('active', mask), ('padding', ~mask)]:
                        summaries.append({'case': key, 'method': method, 'port': port, 'stratum': label, 'rows': int(take.sum()), **metric(pred[take], target[take], (target-base)[take])})
                    for pos in range(512):
                        stream.write(__import__('json').dumps({'case': key, 'method': method, 'port': port, 'position': pos, 'active': bool(mask[pos]), **metric(pred[pos], target[pos], target[pos]-base[pos])}, allow_nan=False)+'\n'); count += 1
        for a, b in [('A', 'B'), ('00', '10'), ('01', '11')]:
            same = np.asarray(packets[a]['ids']) == packets[b]['ids']; mask = np.asarray(packets[a]['mask'], bool)
            for method in ('H0', 'candidate', 'endpoint', 'position'):
                for port in ('H1', 'Q', 'K', 'V'):
                    target = native_states[b]-native_states[a] if port == 'H1' else outputs[b, 'native'][port]-outputs[a, 'native'][port]
                    pred = owned[b, method]-owned[a, method] if port == 'H1' else outputs[b, method][port]-outputs[a, method][port]
                    for label, take in [('all', all_rows), ('unchangedToken', same), ('unchangedLearned', same & learned), ('unchangedActive', same & learned & mask), ('unchangedPadding', same & learned & ~mask)]:
                        pairs.append({'A': a, 'B': b, 'method': method, 'port': port, 'selection': label, 'rows': int(take.sum()), **metric(pred[take], target[take])})
                    for pos in range(512):
                        stream.write(__import__('json').dumps({'pair': [a, b], 'method': method, 'port': port, 'position': pos, 'sameToken': bool(same[pos]), 'active': bool(mask[pos]), **metric(pred[pos], target[pos])}, allow_nan=False)+'\n'); count += 1
    consumer.save(out)
    if sha(root/'final-seal.json') != seal: raise ValueError('Final seal changed')
    save(out/'reused-fields.json', {'rows': reused, 'servingAssetsAuthenticated': checked, 'scope': 'Historical exact native/H0/endpoint/position consumer outputs reused after actual input/profile bindings;not new evaluations.'})
    for name in request['retainedMetadata']: shutil.copyfile(root/name, out/name)
    for name in ('final-seal.json', 'discovery-seal.json', 'initializer-seal.json', 'fit.json'): shutil.copyfile(root/name, out/name)
    for name in ('initializer', 'discovery', 'final'): shutil.copyfile(root/name/'receipt.json', out/(name+'-serving.json'))
    save(out/'programs.json', {name: {'directory': str(path), 'program': read(path/'program.json'), 'programSHA256': sha(path/'program.json')} for name, path in programs.items()})
    save(out/'report.json', {'schema': request['schema'], 'requestSHA256': request['requestSHA256'], 'fitting': fitting, 'selected': selected,
        'feedbackSHA256': sha(out/'feedback.json'), 'transferReleasedAfterSealSHA256': seal, 'summaries': summaries, 'pairs': pairs, 'observationRows': count,
        'workerWallSeconds': time.perf_counter()-start, 'freshChildWallSeconds': {'initializer': init_time, 'discovery': discovery_time, 'final': final_time},
        'consumerWork': {'units': 6, 'originalLinears': 18, 'nominalMACs': 48318382080, 'normalizations': 6, 'fullBlocks': 0, 'images': 0},
        'parameterAuthenticationLoadSeconds': consumer.load_seconds, 'authenticatedWeightShardBytes': sum(x['bytes'] for x in consumer.assets),
        'selectedParameterBytes': sum(x['bytes'] for x in source['parameters']), 'peakCUDAAllocatedBytes': consumer.torch.cuda.max_memory_allocated(),
        'numericalProfile': source['numericalProfile'], 'metricSemantics': 'Complete endpoints with native-increment-normalized error;endpoint RMS/cosine are not increment alignment. Pair metrics are signed prompt responses.',
        'mechanics_status': 'completed', 'instrument_status': 'owned-finite-H1-and-next-native-QKV', 'trend_status': 'inspect-all-strata-rows-and-signed-responses',
        'terminal_status': 'arbitrary-prompt-zero-neural-image-generation-unachieved',
        'limitations': ['Reused public transfer;novel raw prompt unscored.', 'Offline teacher/fitting/lexical costs and native evaluation paid.',
            'One greedy relaxed objective/search;not a global or finite optimum.', 'No attention,MLP,fullencoder,denoiser,VAE or image. No matched speedup or isolated child memory claim.']})
    print('CONTRAST_WRITER_COMPLETE', flush=True)
