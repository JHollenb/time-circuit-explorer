"""Offline paired-loss tree fitting; never imported by the serving child."""
import time
import numpy as np
from contextual_writer_program import BUDGET


def quadratic(residual, weights, edges, pair_weights):
    """q is formed as P R from the retained FP64 residual, not reassociated."""
    r = np.asarray(residual, np.float64); w = np.asarray(weights, np.float64)
    edges = np.asarray(edges, np.int64).reshape(-1, 2); v = np.asarray(pair_weights, np.float64)
    if r.ndim != 2 or w.shape != (len(r),) or v.shape != (len(edges),):
        raise ValueError('Quadratic geometry differs')
    if not np.isfinite(r).all() or not np.isfinite(w).all() or not np.isfinite(v).all() or np.any(w <= 0) or np.any(v < 0):
        raise ValueError('Invalid finite positive endpoint measure')
    if len(edges) and (edges.min() < 0 or edges.max() >= len(r) or np.any(edges[:, 0] == edges[:, 1])):
        raise ValueError('Pair addresses differ')
    qr = w[:, None]*r
    pair_r = r[edges[:, 1]]-r[edges[:, 0]]
    np.add.at(qr, edges[:, 0], -v[:, None]*pair_r)
    np.add.at(qr, edges[:, 1], v[:, None]*pair_r)
    if not np.isfinite(qr).all() or not np.isfinite(pair_r).all(): raise ValueError('Nonfinite quadratic intermediate')
    return {'r': r, 'w': w, 'edges': edges, 'v': v, 'qr': qr, 'pairResidual': pair_r}


def system(q, labels, leaves):
    """Positive diagonal leaf mass plus the signed pair graph Laplacian."""
    labels = np.asarray(labels, np.int64)
    a, b = (labels[q['edges'][:, i]] for i in (0, 1)); distinct = a != b
    a, b, v = a[distinct], b[distinct], q['v'][distinct]
    matrix = np.diag(np.bincount(labels, weights=q['w'], minlength=leaves))
    for i, j, sign in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
        np.add.at(matrix, (i, j), sign*v)
    rhs = np.zeros((leaves, q['r'].shape[1]), np.float64)
    np.add.at(rhs, labels, q['qr'])
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all(): raise ValueError('Nonfinite normal system')
    chol = np.linalg.cholesky(matrix)
    constants = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    if not np.isfinite(constants).all(): raise ValueError('Nonfinite fitted constants')
    return matrix, chol, constants


def loss(q, labels, constants):
    prediction = constants[labels]; error = prediction-q['r']
    pair = (prediction[q['edges'][:, 1]]-prediction[q['edges'][:, 0]])-q['pairResidual']
    value = float(np.einsum('i,ij,ij->', q['w'], error, error)+np.einsum('i,ij,ij->', q['v'], pair, pair))
    if not np.isfinite(value): raise ValueError('Nonfinite relaxed loss')
    return value


def apply_q(q, values):
    result = q['w'][:, None]*values
    change = q['v'][:, None]*(values[q['edges'][:, 1]]-values[q['edges'][:, 0]])
    np.add.at(result, q['edges'][:, 0], -change)
    np.add.at(result, q['edges'][:, 1], change)
    return result


def split_statistics(q, labels, constants, chol, indices, masks, z, diagonal):
    qa = apply_q(q, np.eye(len(constants), dtype=np.float64)[labels])
    c = masks.astype(np.float64)@qa[indices]
    whitened = np.linalg.solve(chol, c.T)
    correction = np.einsum('ik,ik->k', whitened, whitened)
    schur = diagonal-correction
    h = z-c@constants
    if not np.isfinite(c).all() or not np.isfinite(schur).all() or not np.isfinite(h).all(): raise ValueError('Nonfinite Schur intermediate')
    # This bound triggers a direct numerical recomputation, never a quality gate.
    bound = 32*np.finfo(np.float64).eps*(np.abs(diagonal)+np.abs(correction))
    ordinary = schur > bound
    gain = np.full(len(diagonal), -np.inf, np.float64)
    gain[ordinary] = np.einsum('ij,ij->i', h[ordinary], h[ordinary])/schur[ordinary]
    return gain, np.flatnonzero(~ordinary), schur


def candidates(x, q, indices, budget):
    """Cache only eligible cut statistics, not every full-width prefix bank."""
    masks = []; zs = []; diagonal = []; meta = []; seen = set(); enumerated = 0
    n = len(indices); width = q['r'].shape[1]
    if n < 2*budget['minLeafRows']:
        return {'masks': np.empty((0, n), bool), 'z': np.empty((0, width)), 'd': np.empty(0), 'meta': [], 'enumerated': 0}
    for column in range(x.shape[1]):
        order = np.argsort(x[indices, column], kind='stable'); values = x[indices[order], column]
        cuts = np.flatnonzero(values[:-1] < values[1:])
        cuts = cuts[(cuts+1 >= budget['minLeafRows']) & (n-cuts-1 >= budget['minLeafRows'])]
        if len(cuts) > budget['thresholdsPerFeature']:
            cuts = cuts[np.linspace(0, len(cuts)-1, budget['thresholdsPerFeature'], dtype=int)]
        pending = []
        for cut in cuts:
            enumerated += 1; mask = np.zeros(n, bool); mask[order[cut+1:]] = True
            signature = np.packbits(mask).tobytes()
            if signature in seen: continue
            seen.add(signature); pending.append((int(cut), mask))
        if not pending: continue
        prefix = np.cumsum(q['qr'][indices[order]], axis=0)
        for cut, mask in pending:
            global_mask = np.zeros(len(x), bool); global_mask[indices[mask]] = True
            boundary = global_mask[q['edges'][:, 0]] != global_mask[q['edges'][:, 1]]
            diagonal.append(float(q['w'][indices[mask]].sum()+q['v'][boundary].sum()))
            masks.append(mask); zs.append(prefix[-1]-prefix[cut]); meta.append((column, float(values[cut])))
    return {'masks': np.stack(masks) if masks else np.empty((0, n), bool),
            'z': np.stack(zs) if zs else np.empty((0, width)), 'd': np.asarray(diagonal), 'meta': meta, 'enumerated': enumerated}


def fit_contrast_tree(x, residual, endpoint_weights, edges, pair_weights, endpoint_energy, response_energy, budget=BUDGET):
    start = time.perf_counter(); x = np.asarray(x, np.float64)
    if x.ndim != 2 or len(x) != len(residual) or not np.isfinite(x).all(): raise ValueError('Feature ABI differs')
    if not np.isfinite([endpoint_energy, response_energy]).all() or min(endpoint_energy, response_energy) < 0:
        raise ValueError('Invalid reference energy')
    w = np.asarray(endpoint_weights, np.float64)/(endpoint_energy if endpoint_energy > 0 else 1.0)
    v = np.asarray(pair_weights, np.float64)/(response_energy if response_energy > 0 else 1.0)
    q = quadratic(residual, w, edges, v); labels = np.zeros(len(x), np.int64)
    tree = {'leaf': 0}; nodes = {0: tree}; groups = {0: np.arange(len(x))}; depths = {0: 0}; caches = {}
    traces = []; peak_cache = 0; fallback_count = 0; enumerated = 0; unique = 0
    def ordered_leaves(node):
        if 'leaf' in node: return [node['leaf']]
        return ordered_leaves(node['left'])+ordered_leaves(node['right'])
    def cache(leaf):
        nonlocal peak_cache, enumerated, unique
        if depths[leaf] >= budget['depth']: return
        value = candidates(x, q, groups[leaf], budget); caches[leaf] = value
        enumerated += value['enumerated']; unique += len(value['meta'])
        peak_cache = max(peak_cache, sum(sum(v[k].nbytes for k in ('masks', 'z', 'd')) for v in caches.values()))
    cache(0)
    while True:
        leaves = len(groups); matrix, chol, constants = system(q, labels, leaves); current_loss = loss(q, labels, constants); best = None
        if traces and 'lossAfter' not in traces[-1]:
            trace = traces[-1]; trace['lossAfter'] = current_loss
            trace['realizedRelaxedDecrease'] = trace['lossBefore']-current_loss
            trace['schurMinusRealizedDecrease'] = trace['relaxedGain']-trace['realizedRelaxedDecrease']
        if leaves >= budget['maxLeavesPerSlot']: break
        for leaf in ordered_leaves(tree):
            if leaf not in caches or not len(caches[leaf]['meta']): continue
            value = caches[leaf]; gain, fallback, schur = split_statistics(q, labels, constants, chol, groups[leaf], value['masks'], value['z'], value['d'])
            for at in fallback:
                trial = labels.copy(); trial[groups[leaf][value['masks'][at]]] = leaves
                _, _, direct = system(q, trial, leaves+1); gain[at] = current_loss-loss(q, trial, direct); fallback_count += 1
            if not np.isfinite(gain).all(): raise ValueError('Nonfinite split gain')
            at = int(np.argmax(gain)); g = float(gain[at])
            if best is None or g > best[0]: best = (g, leaf, at, float(schur[at]))
        if best is None or best[0] <= 0: break
        gain, leaf, at, schur = best; value = caches.pop(leaf); indices = groups[leaf]; right = value['masks'][at]
        column, threshold = value['meta'][at]; new = leaves
        groups[leaf], groups[new] = indices[~right], indices[right]; labels[groups[new]] = new
        depth = depths[leaf]+1; depths[leaf] = depths[new] = depth
        node = nodes[leaf]; left_node, right_node = {'leaf': leaf}, {'leaf': new}; node.clear()
        node.update({'feature': column, 'threshold': threshold, 'gain': gain, 'left': left_node, 'right': right_node})
        nodes[leaf], nodes[new] = left_node, right_node
        traces.append({'split': len(traces), 'leaf': leaf, 'feature': column, 'threshold': threshold, 'relaxedGain': gain,
                       'schur': schur, 'lossBefore': current_loss, 'leftRows': len(groups[leaf]), 'rightRows': len(groups[new])})
        del value, right
        cache(leaf); cache(new)
    order = ordered_leaves(tree); renumber = {leaf: i for i, leaf in enumerate(order)}
    def serialize(node):
        if 'leaf' in node: return {'atom': renumber[node['leaf']]}
        return {**{k: v for k, v in node.items() if k not in ('left', 'right')}, 'left': serialize(node['left']), 'right': serialize(node['right'])}
    output = constants[order].astype('<f4')
    if not np.isfinite(output).all(): raise ValueError('Nonfinite FP32 atom admission')
    occupancy = [{'atom': renumber[leaf], 'rows': groups[leaf].tolist(), 'weight': float(np.asarray(endpoint_weights)[groups[leaf]].sum())} for leaf in order]
    diagnostics = {'seconds': time.perf_counter()-start, 'splits': traces, 'atoms': len(order), 'relaxedLoss': current_loss,
                   'normalEquationResidualRMS': float(np.sqrt(np.mean((matrix@constants-np.stack([q['qr'][labels == i].sum(0) for i in range(len(groups))]))**2))),
                   'normalMatrixCondition': float(np.linalg.cond(matrix)), 'candidateCutsEnumerated': enumerated,
                   'uniqueCandidatePartitionsCached': unique, 'peakCutCacheBytes': peak_cache, 'directFallbackCandidates': fallback_count,
                   'endpointEnergy': float(endpoint_energy), 'responseEnergy': float(response_energy),
                   'zeroEnergyPolicy': 'Use absolute corresponding component; no divide by zero.',
                   'objective': 'Equal separately normalized endpoint and unchanged-token paired residual loss; q=P*R in FP64.',
                   'search': 'Greedy globally best relaxed split; leafDFS/feature/cut tie order. Exact duplicate right masks use first feature/cut. Cholesky Schur score, direct partition solve for cancellation-dominated denominator; no explicit inverse.'}
    return serialize(tree), output, occupancy, diagnostics
