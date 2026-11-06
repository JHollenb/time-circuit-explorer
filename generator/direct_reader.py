"""Current-query score reuse with centers-only inputs and no anchor machinery."""
import copy
import hashlib
from pathlib import Path
import resource
import time
from encoder_atoms import np, floats, words, read, save, sha, guard

POLICY = 'current-top4-direct-score-reuse-fp64-fp32-bf16-v1'
COUNTERS = ['selectorScoreProducts', 'selectedValueProducts', 'queries']


def logsum(logs):
    if not len(logs): return -np.inf
    top = logs.max()
    return float(top + np.log(np.exp(logs - top).sum())) if np.isfinite(top) else -np.inf


def select(score, positions, budget=4):
    score = np.asarray(score); positions = np.asarray(positions)
    if score.shape != positions.shape or score.ndim != 1 or not np.isfinite(score).all() or budget < 0:
        raise ValueError('Invalid selector scores')
    return np.lexsort((positions, -score))[:budget]


def stream(packets, center, state=None):
    heads = len(packets); n, d = packets[0]['q'].shape; center = np.asarray(center)
    tag = hashlib.sha256((POLICY + repr((heads, d))).encode() + np.asarray(center, dtype='<f8').tobytes()).hexdigest()
    if state is None:
        state = {'keys': np.zeros((8, d)), 'values': np.zeros((8, d)), 'positions': np.full(8, -1, dtype=np.int64), 'count': 0, 'steps': 0, 'assetIdentity': tag,
            **{k: 0 for k in COUNTERS}}
    else:
        state = copy.deepcopy(state)
        state['assetIdentity'] = str(np.asarray(state['assetIdentity']).item())
        for k in ['count', 'steps', *COUNTERS]: state[k] = int(state[k])
    if state['assetIdentity'] != tag: raise ValueError('State source geometry differs')
    count = state['count']; capacity = len(state['keys'])
    if state['keys'].shape != (capacity, d) or state['values'].shape != (capacity, d) or state['positions'].shape != (capacity,) or not 0 <= count <= capacity or state['steps'] < count:
        raise ValueError('Typed state dimensions/count differ')
    if count and (np.any(np.diff(state['positions'][:count]) <= 0) or state['positions'][count-1] >= state['steps']):
        raise ValueError('State causal history differs')
    if d % 2 or center.shape != (d,) or center.dtype != np.float64:
        raise ValueError('Grouped source geometry differs')
    for p in packets:
        if not np.array_equal(p['positions'], np.arange(state['steps'], state['steps'] + n)) or not np.array_equal(p['mask'], packets[0]['mask']): raise ValueError('State clock/mask differs')
        if not np.array_equal(p['k'], packets[0]['k']) or not np.array_equal(p['v'], packets[0]['v']): raise ValueError('Shared KV group differs')
        if any(p[k].dtype != np.uint16 or p[k].shape != (n, d) for k in ['q', 'k', 'v']): raise ValueError('BF16 packet shape/type differs')
    mask = packets[0]['mask']; q = np.stack([floats(p['q']).astype(float) / d**.25 for p in packets])
    k = floats(packets[0]['k']).astype(float) / d**.25 - center; v = floats(packets[0]['v']).astype(float)
    if mask.shape != (n,) or np.any((mask != 0) & (mask != 1)) or not all(np.isfinite(x).all() for x in [q, k, v, center, state['keys'], state['values']]):
        raise ValueError('Invalid finite packet/state')
    raw = np.zeros((n, heads*d)); support = np.full((n, heads, 4), -1, dtype=np.int64)
    logits = np.full((n, heads, 4), np.nan); weights = np.zeros((n, heads, 4)); prefix_counts = np.empty(n, dtype=np.int64)
    for i in range(n):
        if mask[i]:
            count = state['count']
            if count == len(state['keys']):
                for key, axis in [('keys', 0), ('values', 0), ('positions', 0)]:
                    shape = list(state[key].shape); shape[axis] *= 2; value = np.zeros(shape, dtype=state[key].dtype)
                    slots = [slice(None)]*value.ndim; slots[axis] = slice(0, count); value[tuple(slots)] = state[key]; state[key] = value
            state['keys'][count] = k[i]; state['values'][count] = v[i]; state['positions'][count] = int(packets[0]['positions'][i]); state['count'] += 1
        for h in range(heads):
            state['queries'] += 1; count = state['count']
            if not count: continue
            current = state['keys'][:count] @ q[h, i]
            chosen = select(current, state['positions'][:count], 4)
            state['selectorScoreProducts'] += count*d; support[i, h, :len(chosen)] = state['positions'][chosen]
            a = current[chosen]; w = np.exp(a - logsum(a))
            raw[i, h*d:(h+1)*d] = w @ state['values'][:count][chosen]
            logits[i, h, :len(chosen)] = a; weights[i, h, :len(chosen)] = w; state['selectedValueProducts'] += len(chosen)*d
        prefix_counts[i] = state['count']; state['steps'] += 1
    return {'raw': raw, 'rounded': words(raw), 'supportPositions': support,
        'selectedLogits': logits, 'selectedWeights': weights, 'prefixCounts': prefix_counts}, state


def serve(spec_path, out):
    guard(); spec = read(spec_path); out = Path(out); out.mkdir(exist_ok=False); start = time.perf_counter(); cpu = time.process_time()
    if sha(spec['asset']['path']) != spec['asset']['sha256']: raise ValueError('Source asset differs')
    centers = np.load(spec['asset']['path'], allow_pickle=False)
    if centers.shape != (8, 128) or centers.dtype != np.float64 or not np.isfinite(centers).all(): raise ValueError('Centers-only payload differs')
    m = spec['geometry']; d = m['headDim']; heads = m['qWidth']//d; groups = m['kvWidth']//d; hpg = heads//groups
    if (heads, groups, d) != (32, 8, 128): raise ValueError('Qualified grouped geometry required')
    rows = []; states = []; counts = []
    for item in spec['ports']:
        if sha(item['path']) != item['sha256']: raise ValueError('Own lexical operands differ')
        with np.load(item['path'], allow_pickle=False) as f: full = {k: f[k] for k in f.files}
        n = len(full['ids']); buffers = {}; group_rows = []; tick = time.perf_counter()
        for g in range(groups):
            first = g*hpg
            packets = [{**{k: full[k] for k in ['ids', 'mask', 'positions']}, 'q': full['qpos'][:, h*d:(h+1)*d],
                'k': full['kpos'][:, g*d:(g+1)*d], 'v': full['v'][:, g*d:(g+1)*d]} for h in range(first, first+hpg)]
            t = time.perf_counter(); result, state = stream(packets, centers[g]); seconds = time.perf_counter()-t
            for key, value in result.items():
                if key == 'prefixCounts':
                    if key in buffers and not np.array_equal(buffers[key], value): raise ValueError('Grouped prefix count differs')
                    buffers[key] = value; continue
                shape = list(value.shape); shape[1] = heads*d if key in ['raw', 'rounded'] else heads
                if key not in buffers: buffers[key] = np.empty(shape, dtype=value.dtype)
                span = slice(first*d, (first+hpg)*d) if key in ['raw', 'rounded'] else slice(first, first+hpg)
                buffers[key][:, span] = value
            path = out/(item['id']+'-group'+str(g)+'-state.npz'); np.savez(path, **{k: np.asarray(v) for k, v in state.items()})
            states.append({'id': item['id'], 'kvGroup': g, 'path': str(path), 'sha256': sha(path)})
            group_rows.append({'kvGroup': g, 'seconds': seconds, 'admittedKeys': state['count'], 'steps': state['steps'], **{k: state[k] for k in COUNTERS},
                'allocatedIndexPayloadBytes': sum(state[k].nbytes for k in ['keys', 'values', 'positions']),
                'usedIndexPayloadBytes': state['count']*(2*d+1)*8, 'countAndClockBytes': 16,
                'anchorAdmissionMACs': 0, 'anchorScoreCacheBytes': 0, 'routingCalls': 0, 'selectedDeltaProducts': 0, 'momentPayloadBytes': 0, 'envelopePayloadBytes': 0, 'fallbackCalls': 0})
        path = out/(item['id']+'-selected.npz'); np.savez(path, **buffers); rows.append({'id': item['id'], 'path': str(path), 'sha256': sha(path)})
        counts.append({'id': item['id'], 'groups': group_rows, 'withSerializationSeconds': time.perf_counter()-tick,
            'outputAndDiagnosticPayloadBytes': sum(v.nbytes for v in buffers.values())})
    save(out/'prediction-seal.json', {'schema': 'tce-direct-reader-prediction-v1', 'policy': POLICY, 'specSHA256': sha(spec_path),
        'asset': spec['asset'], 'ports': spec['ports'], 'rows': rows, 'states': states, 'counts': counts,
        'runtimeCentersPayloadBytes': centers.nbytes, 'basisPayloadBytes': 0,
        'wallSecondsAfterImports': time.perf_counter()-start, 'cpuSecondsAfterImports': time.process_time()-cpu,
        'highWaterRSSKiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope': 'Direct selected scores; full causal K/V and current-query scans paid. No original geometry, routes, anchors, cache, tail, basis or fallback.'})


def install_serve_guard(root, source):
    from flux_clock import deny_targets
    guard(); deny_targets([root/'targets.json', source['originalGeometry']['path'], *source['forbiddenTargets']])
    try:
        with open(source['originalGeometry']['path'], 'rb'): pass
    except PermissionError: geometry_denied = True
    else: raise RuntimeError('Original geometry exclusion is not active')
    return geometry_denied


def execute(root):
    from direct_reader_study import verify
    from encoder_atoms import Prefix
    from paired_fields import retain
    root = Path(root).resolve(); request = read(root/'request.json'); verify(root, request); source = read(root/'inputs.json')
    geometry_denied = install_serve_guard(root, source)
    start = time.perf_counter(); cpu = time.process_time(); prefix = Prefix(source['lexicalStore'], request['manifestSHA256'])
    if len(prefix.store.ids) != 151936: raise ValueError('Complete vocabulary required')
    lexical = root/'lexical'; lexical.mkdir(); rows = []
    for prompt in request['prompts']:
        tick = time.perf_counter(); values, text = prefix.forward(prompt['prompt']); seconds = time.perf_counter()-tick
        path = lexical/(prompt['id']+'.npz'); np.savez(path, **values)
        rows.append({'id': prompt['id'], 'prompt': prompt['prompt'], 'path': str(path), 'sha256': sha(path), 'activeTokens': int(values['mask'].sum()),
            'firstArithmeticSeconds': seconds, 'renderedTextSHA256': hashlib.sha256(text.encode()).hexdigest(),
            'idsPayloadSHA256': hashlib.sha256(values['ids'].tobytes()).hexdigest(), 'maskPayloadSHA256': hashlib.sha256(values['mask'].tobytes()).hexdigest()})
    save(lexical/'prediction-seal.json', {'manifestSHA256': request['manifestSHA256'], 'rows': rows, 'coverageCount': len(prefix.store.ids),
        'uniqueRowsRead': prefix.store.rowsRead, 'compiledRowPayloadBytesRead': prefix.store.rowBytesRead, 'frameworkCalls': 0, 'originalDenseProjections': 0})
    save(root/'serve-source.json', {'asset': read(root/'compile-receipt.json')['centers'], 'geometry': source['readerGeometry'], 'ports': rows})
    tick = time.perf_counter(); serve(root/'serve-source.json', root/'reader'); reader_seconds = time.perf_counter()-tick
    seal = read(root/'reader/prediction-seal.json'); fields = {}; h0 = {}
    for item in seal['rows']:
        if sha(item['path']) != item['sha256']: raise ValueError('Reader output differs')
        with np.load(item['path'], allow_pickle=False) as f:
            raw, bits = f['raw'], f['rounded']
            if raw.shape != (512, 4096) or bits.dtype != np.uint16 or not np.isfinite(raw).all() or not np.array_equal(words(raw), bits): raise ValueError('Head bank finite/shape/rounding differs')
            fields[item['id']] = retain(root/'arrays/reader', item['id']+'-head', bits[None]); fields[item['id']]['logicalDtype'] = 'torch.bfloat16'
    for item in rows:
        with np.load(item['path'], allow_pickle=False) as f: e = f['e']
        if e.shape != (512, 2560) or e.dtype != np.uint16 or not np.isfinite(floats(e)).all(): raise ValueError('Own H0 geometry/finite bits differ')
        h0[item['id']] = retain(root/'arrays/h0', item['id']+'-H0', e[None]); h0[item['id']]['logicalDtype'] = 'torch.bfloat16'
    m = prefix.store.manifest
    atoms = [{'path': str(Path(source['lexicalStore'])/'manifest.json'), 'sha256': request['manifestSHA256']}]
    atoms += [{'path': str(Path(source['lexicalStore'])/m[k]['path']), 'sha256': m[k]['sha256']} for k in ['tokenizer', 'template', 'coverage', 'rowHashes', 'rope']]
    atoms += [{'path': str(Path(source['lexicalStore'])/m['shards'][j]['path']), 'sha256': m['shards'][j]['sha256']} for j in sorted(prefix.store.shards)]
    save(root/'reader-seal.json', {'schema': 'tce-direct-reader-seal-v1', 'policy': POLICY, 'requestSHA256': request['requestSHA256'],
        'lexicalSealSHA256': sha(lexical/'prediction-seal.json'), 'readerPredictionSealSHA256': sha(root/'reader/prediction-seal.json'),
        'fields': fields, 'h0Fields': h0, 'lexicalRows': rows, 'sourceAtomFiles': atoms, 'reader': seal, 'states': seal['states'], 'packets': seal['rows'], 'work': seal['counts'],
        'readerWallSeconds': reader_seconds, 'workerSecondsAfterImports': time.perf_counter()-start, 'cpuSecondsAfterImports': time.process_time()-cpu,
        'highWaterRSSKiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'numpyVersion': np.__version__, 'frameworkCalls': 0, 'originalDenseProjections': 0,
        'originalGeometryOpenDenied': geometry_denied, 'centers': read(root/'compile-receipt.json')['centers'],
        'scope': 'Raw strings through direct-score causal reader; full K/V scans and compiled learned lexical table remain. No anchors/routes/cache/tail/basis/fallback or original learned-weight projections.'})
    print('DIRECT_HEADS_SEALED', sha(root/'reader-seal.json'), flush=True)


def compile_centers(root):
    from direct_reader_study import verify
    from flux_clock import deny_targets
    from paired_fields import retain
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);source=read(root/'inputs.json')
    guard();deny_targets([root/'targets.json',*source['forbiddenTargets']]);tick=time.perf_counter();cpu=time.process_time();item=source['originalGeometry']
    if sha(item['path'])!=item['sha256']:raise ValueError('Frozen original geometry differs')
    with np.load(item['path'],allow_pickle=False) as f:centers=f['keyCenters']
    if centers.shape!=(8,128) or centers.dtype!=np.float64 or not np.isfinite(centers).all():raise ValueError('Source centers differ')
    field=retain(root/'compiled','centers',centers)
    save(root/'compile-receipt.json',{'schema':'tce-direct-reader-centers-compile-v1','requestSHA256':r['requestSHA256'],
        'originalGeometry':item,'originalFileBytesHashed':Path(item['path']).stat().st_size,'centers':field,'numericPayloadBytes':centers.nbytes,
        'compileSecondsAfterImports':time.perf_counter()-tick,'cpuSecondsAfterImports':time.process_time()-cpu,'originalParameterReads':0,
        'scope':'Extract unchanged frozen FP64 key centers only; no fit, contextual target or original weight evaluation. Fresh serving child excludes original geometry.'})
