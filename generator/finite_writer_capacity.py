"""Target-assisted finite dictionary capacity; no fit or source-input resolver."""
import math
import numpy as np
from contextual_writer_program import finite_add_array


SCORING = ('Contiguous coordinate axis; explicit FP64 subtract/multiply and np.sum(axis=-1,dtype=float64) norms. '
           'Pair score=(alpha_i*norm_i+alpha_j*norm_j)+beta*response_norm; no fused expression. '
           'C-order a-major/b-minor enumeration; smallest address wins exact ties; math.fsum outer component scores. '
           'Historical np.einsum losses rebuilt separately; optimum concerns evaluated component scores only.')


def norm2(value):
    value = np.asarray(value, np.float64)
    result = np.sum(value*value, axis=-1, dtype=np.float64)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite coordinate norm')
    return result


def scale(weights, energy):
    if not math.isfinite(energy) or energy < 0: raise ValueError('Invalid reference energy')
    result = np.asarray(weights, np.float64)/(energy if energy else 1.0)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite normalized measure')
    return result


def pair_scores(left, right, yleft, yright, alpha_left, alpha_right, beta):
    ni, nj = norm2(left-yleft), norm2(right-yright)
    # Actual finite response order, not the relaxed P*(Y-S) fitting target.
    response = (right[None, :, :]-left[:, None, :])-(yright-yleft)
    nr = norm2(response)
    endpoint = (alpha_left*ni)[:, None]+(alpha_right*nj)[None, :]
    response_term = beta*nr
    scores = endpoint+response_term
    if not np.isfinite(scores).all(): raise ValueError('Nonfinite component scores')
    return ni, nj, nr, endpoint, response_term, scores


def choose(scores):
    flat = np.asarray(scores, np.float64).ravel(order='C')
    if not len(flat) or not np.isfinite(flat).all(): raise ValueError('Invalid finite choices')
    index = int(np.argmin(flat)); minimum = float(flat[index]); distinct = np.unique(flat)
    return index, {'ties': int(np.sum(flat == minimum)),
                   'nextDistinctGap': float(distinct[1]-distinct[0]) if len(distinct) > 1 else None}


def enumerate_capacity(h0, stage0, targets, atoms, current, labels, weights, edges, pair_weights, energies, emit):
    h0, stage0, targets, atoms, current = [np.asarray(x, np.float64) for x in (h0, stage0, targets, atoms, current)]
    labels = np.asarray(labels); weights = np.asarray(weights, np.float64); edges = np.asarray(edges); pair_weights = np.asarray(pair_weights, np.float64)
    n, width = h0.shape; count = len(atoms)
    if any(x.shape != (n, width) for x in (stage0, targets, current)) or atoms.shape != (count, width) or not count:
        raise ValueError('Finite state/dictionary geometry differs')
    if labels.shape != (n,) or labels.dtype.kind not in 'iu' or np.any(labels < 0) or np.any(labels >= count): raise ValueError('Invalid current labels')
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in 'iu' or np.any(edges < 0) or np.any(edges >= n): raise ValueError('Invalid pair edges')
    if len(set(map(int, edges.ravel()))) != 2*len(edges): raise ValueError('Capacity graph must be disjoint pairs')
    if weights.shape != (n,) or pair_weights.shape != (len(edges),) or np.any(weights <= 0) or np.any(pair_weights <= 0): raise ValueError('Invalid positive measure')
    if any(not np.isfinite(x).all() for x in (h0, stage0, targets, atoms, current, weights, pair_weights)): raise ValueError('Nonfinite capacity inputs')
    if abs(float(weights.sum())-1) > n*np.finfo(np.float64).eps or abs(float(pair_weights.sum())-1) > max(1,len(edges))*np.finfo(np.float64).eps:
        raise ValueError('Capacity measures must sum to one')
    delta = targets-h0; changes = targets[edges[:, 1]]-targets[edges[:, 0]]
    recomputed = [float(np.einsum('i,ij,ij->', weights, delta, delta)), float(np.einsum('i,ij,ij->', pair_weights, changes, changes))]
    energy, response_energy = recomputed if energies is None else map(float, energies)
    alpha, beta = scale(weights, energy), scale(pair_weights, response_energy)
    output = np.empty((n, width), np.float32); selected = np.empty(n, np.int64); covered = np.zeros(n, bool)
    totals = {m: {'joint': [], 'endpoint': [], 'response': []} for m in ('current', 'capacity')}
    components = []; pair_count = 0; single_count = 0
    def options(i):
        value = finite_add_array(stage0[i][None, :], atoms).astype(np.float64)
        if not np.array_equal(value[labels[i]], current[i]): raise ValueError('Frozen finite current route does not reconstruct')
        return value
    def record(row, endpoint, response, scores, current_index):
        chosen, detail = choose(scores); flat = scores.ravel(order='C'); endpoint = endpoint.ravel(order='C'); response = response.ravel(order='C')
        if flat[chosen] > flat[current_index]: raise ValueError('Enumerator did not retain the feasible current choice')
        row.update(detail); row.update({'scores': scores.tolist(), 'currentIndex': int(current_index), 'selectedIndex': chosen})
        for method, idx in [('current', current_index), ('capacity', chosen)]:
            for term, values in [('joint', flat), ('endpoint', endpoint), ('response', response)]: totals[method][term].append(float(values[idx]))
        components.append({'rows': row['rows'], 'currentIndex': int(current_index), 'selectedIndex': chosen, **detail})
        emit(row); return chosen
    for e, (i0, j0) in enumerate(edges):
        i, j = int(i0), int(j0); left, right = options(i), options(j)
        ni, nj, nr, endpoint, response, scores = pair_scores(left, right, targets[i], targets[j], alpha[i], alpha[j], beta[e])
        row = {'kind': 'pair', 'rows': [i, j], 'alpha': [float(alpha[i]), float(alpha[j])], 'beta': float(beta[e]),
               'endpointNorms': [ni.tolist(), nj.tolist()], 'responseNorms': nr.tolist()}
        idx = record(row, endpoint, response, scores, int(labels[i])*count+int(labels[j])); a, b = divmod(idx, count)
        output[i], output[j] = left[a], right[b]; selected[i], selected[j] = a, b; covered[[i, j]] = True; pair_count += 1
    for i0 in np.flatnonzero(~covered):
        i = int(i0); value = options(i); norms = norm2(value-targets[i]); scores = alpha[i]*norms
        row = {'kind': 'single', 'rows': [i], 'alpha': [float(alpha[i])], 'endpointNorms': [norms.tolist()]}
        a = record(row, scores, np.zeros(count, np.float64), scores, int(labels[i]))
        output[i] = value[a]; selected[i] = a; covered[i] = True; single_count += 1
    if not covered.all(): raise ValueError('Incomplete capacity state')
    reduced = {m: {k: math.fsum(v) for k, v in terms.items()} for m, terms in totals.items()}
    def historical(pred):
        error = pred-targets; change_error = (pred[edges[:, 1]]-pred[edges[:, 0]])-changes
        e = float(np.einsum('i,ij,ij->', weights, error, error)); t = float(np.einsum('i,ij,ij->', pair_weights, change_error, change_error))
        return {'endpointSquaredError': e, 'responseSquaredError': t, 'endpointRatio': e/energy if energy else None,
                'responseRatio': t/response_energy if response_energy else None,
                'endpointScore': e/energy if energy else e, 'responseScore': t/response_energy if response_energy else t}
    for method, value in [('current', current), ('capacity', output.astype(np.float64))]:
        reduced[method]['constituentSumMinusJoint'] = (reduced[method]['endpoint']+reduced[method]['response'])-reduced[method]['joint']
        reduced[method]['historicalReduction'] = historical(value)
    return output, selected, {'scores': reduced, 'referenceEnergies': [energy, response_energy], 'recomputedReferenceEnergies': recomputed,
        'referenceEnergyRecomputeDifference': [a-b for a,b in zip(recomputed,[energy,response_energy])], 'components': components,
        'rows': n, 'width': width, 'atoms': count, 'pairedComponents': pair_count, 'unpairedComponents': single_count,
        'evaluatedChoices': pair_count*count*count+single_count*count, 'finiteOptionStates': n*count,
        'finiteOptionCoordinateAdditions': n*count*width, 'endpointNormProducts': n*count*width,
        'pairedResponseNormProducts': pair_count*count*count*width, 'scoring': SCORING,
        'scope': 'Target-assisted finite H1 capacity; no fit, shared resolver, consumer optimum or image.'}
