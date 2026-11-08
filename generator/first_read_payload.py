"""A paid, saved-only allocation/value crossover; never an image generator."""
import hashlib
import math
import time
from encoder_atoms import np, floats, words


POLICY = 'current-top4-uncentered-fp64-fp32-bf16-v1'
ALLOWED = {'lexical': {'ids', 'mask', 'positions', 'v'},
           'selected': {'supportPositions', 'selectedWeights', 'prefixCounts', 'raw', 'rounded'}}


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def member(item, name, selection, phase, ledger):
    """Read one qualified member, copy its selected slice, release the full member."""
    if name not in ALLOWED[item['kind']] or name not in item['members']:
        raise ValueError('Undeclared member')
    header = item['headers'][name]
    with np.load(item['path'], allow_pickle=False) as archive:
        value = archive[name]
    if value.dtype.str != header['dtype'] or list(value.shape) != header['shape'] or not value.flags.c_contiguous or header['fortranOrder']:
        raise ValueError('Retained member ABI differs')
    result = np.array(value[selection], copy=True, order='C')
    ledger.append({'id': item['id'], 'kind': item['kind'], 'member': name, 'phase': phase,
                   'fullMemberBytes': value.nbytes, 'fullMemberSHA256': digest(value),
                   'sliceShape': list(result.shape), 'sliceBytes': result.nbytes, 'sliceSHA256': digest(result)})
    return result


def qualify(a, b):
    for packet in [a, b]:
        if any(packet[k].shape != (512,) for k in ['ids', 'mask', 'positions']):
            raise ValueError('Token geometry')
        if not np.array_equal(packet['positions'], np.arange(512)) or not np.array_equal(packet['mask'], np.arange(512) < 35):
            raise ValueError('Historical clock/mask specimen differs')
        if packet['v'].dtype != np.uint16 or packet['v'].shape != (512, 128) or not np.isfinite(floats(packet['v'])).all():
            raise ValueError('Value words/finite geometry')
        if packet['support'].shape != (477, 4) or packet['weights'].shape != (477, 4) or packet['weights'].dtype != np.float64:
            raise ValueError('Selected-reader geometry')
        if not np.isfinite(packet['weights']).all() or np.any(packet['weights'] < 0) or not np.array_equal(packet['counts'], np.full(477, 35)):
            raise ValueError('Selected-reader finite/prefix contract')
        admitted = packet['positions'][packet['mask'] == 1]
        mapping = {int(p): i for i, p in enumerate(admitted)}
        if len(mapping) != len(admitted):
            raise ValueError('Ambiguous admitted addresses')
        for support in packet['support']:
            if len(set(map(int, support))) != 4 or any(int(p) not in mapping for p in support):
                raise ValueError('Support is not four unique admitted causal addresses')
        packet['admittedPositions'] = admitted
        packet['bank'] = np.ascontiguousarray(floats(packet['v'][packet['mask'] == 1]).astype(np.float64))
    changed = np.flatnonzero(a['ids'] != b['ids']).tolist()
    if changed != [8] or (int(a['ids'][8]), int(b['ids'][8])) != (2115, 1290):
        raise ValueError('Historical token edit differs')
    if any(not np.array_equal(a[k], b[k]) for k in ['mask', 'positions', 'admittedPositions']):
        raise ValueError('Different admitted history')
    rows = np.flatnonzero(np.any(a['v'] != b['v'], axis=1)).tolist()
    if any(p != 8 for p in rows):
        raise ValueError('Single-value-address identity not applicable')
    return {'changedTokenPositions': changed, 'changedValuePositions': rows,
            'changedValueWords': int(np.sum(a['v'][8] != b['v'][8])), 'activeKeys': 35, 'paddingQueries': 477,
            'valueDeltaSquaredNorm': float(np.dot(b['bank'][8]-a['bank'][8], b['bank'][8]-a['bank'][8])),
            'weightSumMaxAbsoluteError': {name: float(np.max(np.abs(packet['weights'].sum(axis=1)-1))) for name, packet in [('A', a), ('B', b)]}}


def selected_read(weights, support, bank, admitted_positions):
    """The source's per-query FP64 [4] @ fancy-gather [4,128], in saved order."""
    weights = np.asarray(weights)
    support = np.asarray(support)
    bank = np.asarray(bank)
    if weights.dtype != np.float64 or weights.shape != support.shape or weights.ndim != 2 or weights.shape[1] != 4:
        raise ValueError('Typed selected weights/support')
    if bank.dtype != np.float64 or bank.ndim != 2 or bank.shape[1] != 128 or len(admitted_positions) != len(bank):
        raise ValueError('Typed source value bank')
    if not np.isfinite(weights).all() or not np.isfinite(bank).all() or np.any(weights < 0):
        raise ValueError('Nonfinite/negative selected read')
    mapping = {int(p): i for i, p in enumerate(admitted_positions)}
    if len(mapping) != len(bank):
        raise ValueError('Duplicate admitted address')
    result = np.empty((len(weights), 128), dtype=np.float64)
    started = time.perf_counter(); cpu = time.process_time()
    for i, addresses in enumerate(support):
        if len(set(map(int, addresses))) != 4 or any(int(p) not in mapping for p in addresses):
            raise ValueError('Unknown/duplicate selected address')
        chosen = np.asarray([mapping[int(p)] for p in addresses], dtype=np.int64)
        w = np.ascontiguousarray(weights[i], dtype=np.float64)
        values = bank[chosen]
        if values.shape != (4, 128) or not values.flags.c_contiguous:
            raise ValueError('Source fancy-gather geometry')
        result[i] = w @ values
    return result, {'queries': len(weights), 'scalarValueProducts': len(weights)*4*128,
                    'wallSeconds': time.perf_counter()-started, 'cpuSeconds': time.process_time()-cpu,
                    'expression': 'separate C-contiguous float64[4] @ fancy-gather float64[4,128] per query; captured order; no renormalization'}


def effects(aa, ab, ba, bb):
    return {'payload': .5*((ab-aa)+(bb-ba)), 'allocation': .5*((ba-aa)+(bb-ab)),
            'interaction': ((bb-ba)-ab)+aa, 'delta': bb-aa}


def row_metrics(effect, ideal_payload, ideal_interaction):
    vectors = {**effect, 'closureRemainder': effect['delta']-(effect['payload']+effect['allocation']),
               'idealPayloadRemainder': effect['payload']-ideal_payload,
               'idealInteractionRemainder': effect['interaction']-ideal_interaction}
    n = len(effect['delta']); rows = []
    for i in range(n):
        norms = {name: float(np.dot(v[i], v[i])) for name, v in vectors.items()}
        dot = float(np.dot(effect['payload'][i], effect['allocation'][i]))
        denominator = math.sqrt(norms['payload']*norms['allocation'])
        rows.append({'squaredNorms': norms, 'payloadDotAllocation': dot, 'twicePayloadDotAllocation': 2*dot,
                     'payloadAllocationCosine': dot/denominator if denominator else None,
                     'deltaSquaredMinusGram': norms['delta']-(norms['payload']+norms['allocation']+2*dot),
                     'payloadNormOverDelta': math.sqrt(norms['payload']/norms['delta']) if norms['delta'] else None,
                     'allocationNormOverDelta': math.sqrt(norms['allocation']/norms['delta']) if norms['delta'] else None})
    return rows, vectors


def aggregate(rows):
    keys = rows[0]['squaredNorms']
    norms = {k: math.fsum(r['squaredNorms'][k] for r in rows) for k in keys}
    dot = math.fsum(r['payloadDotAllocation'] for r in rows)
    return {'rows': len(rows), 'summedSquaredNorms': norms, 'payloadDotAllocation': dot,
            'twicePayloadDotAllocation': 2*dot, 'negativeCrossTermRows': sum(r['payloadDotAllocation'] < 0 for r in rows),
            'zeroDeltaRows': sum(r['squaredNorms']['delta'] == 0 for r in rows),
            'deltaSquaredMinusGram': norms['delta']-(norms['payload']+norms['allocation']+2*dot),
            'payloadNormOverDelta': math.sqrt(norms['payload']/norms['delta']) if norms['delta'] else None,
            'allocationNormOverDelta': math.sqrt(norms['allocation']/norms['delta']) if norms['delta'] else None,
            'scope': 'Dependent query-vector geometry; norm ratios are not semantic shares.'}


def endpoint_comparison(replay, raw, rounded):
    delta = replay-raw
    return {'coordinates': raw.size, 'rawBitDifferences': int(np.sum(replay.view(np.uint64) != raw.view(np.uint64))),
            'rawMaxAbsoluteDifference': float(np.max(np.abs(delta))), 'rawDifferenceSquaredNorm': float(np.sum(delta*delta)),
            'bf16WordDifferences': int(np.sum(words(replay) != rounded)),
            'retainedRawAdmissionWordDifferences': int(np.sum(words(raw) != rounded))}
