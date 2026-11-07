"""Fixed physical-grid scalar interpolation and smooth score diagnostics."""
import numpy as np


def geometry(ids):
    ids = np.asarray(ids)
    if ids.shape != (1, 1024, 4) or ids.dtype.kind not in 'iu':
        raise ValueError('Native THWL IDs must be int[1,1024,4]')
    ids = ids[0]
    if any(len(np.unique(ids[:, a])) != 1 for a in [0, 3]):
        raise ValueError('Native T/L axes not constant')
    hw = ids[:, 1:3]
    if set(map(tuple, hw.tolist())) != {(h, w) for h in range(32) for w in range(32)}:
        raise ValueError('Native H/W not one-to-one physical32x32')
    inverse = {tuple(v): j for j, v in enumerate(hw.tolist())}
    knots = np.asarray([inverse[h, w] for h in range(0, 32, 2) for w in range(0, 32, 2)], dtype=np.int64)
    sampled = (hw[:, 0] % 2 == 0) & (hw[:, 1] % 2 == 0)
    edge = (hw[:, 0] == 31) | (hw[:, 1] == 31)
    strata = np.where(sampled, 0, np.where(edge, 2, 1)).astype(np.int64)
    return knots, strata


def reconstruct(ids, knot_norms):
    knots, strata = geometry(ids)
    supplied = np.asarray(knot_norms, dtype=np.float64)
    if supplied.ndim != 2 or supplied.shape[0] != 256 or not np.all(supplied > 0) or not np.isfinite(supplied).all():
        raise ValueError('Invalid positive smooth norm knots')
    hw = np.minimum(np.asarray(ids)[0, :, 1:3], 30)
    lower = hw // 2
    upper = np.minimum(lower + 1, 15)
    fraction = (hw % 2) / 2
    squared = supplied * supplied
    energy = np.zeros((1024, supplied.shape[1]), dtype=np.float64)
    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ih = upper[:, 0] if a else lower[:, 0]
        iw = upper[:, 1] if b else lower[:, 1]
        weight = (fraction[:, 0] if a else 1-fraction[:, 0]) * (fraction[:, 1] if b else 1-fraction[:, 1])
        energy += weight[:, None] * squared[ih * 16 + iw]
    return np.sqrt(energy), energy, knots, strata


def score_diagnostics(reference, candidate):
    """Finite full-joint softmax: delta=candidate-reference, KL(ref||candidate)."""
    reference, candidate = [np.asarray(v, dtype=np.float64) for v in [reference, candidate]]
    if reference.shape != candidate.shape or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('Invalid score geometry/finite inputs')
    def normalize(logits):
        peak = logits.max(-1, keepdims=True)
        logz = np.log(np.exp(logits-peak).sum(-1, keepdims=True)) + peak
        return np.exp(logits-logz), logz[..., 0]
    p, logz = normalize(reference)
    phat, logzhat = normalize(candidate)
    delta = candidate-reference
    mean = np.sum(p*delta, -1)
    centered = delta-mean[..., None]
    variance = np.sum(p*centered*centered, -1)
    partition_difference = logzhat-logz
    kl = partition_difference-mean
    # Algebraically independent finite form avoids log(0) when probabilities underflow.
    direct_kl = np.sum(p*((reference-logz[..., None])-(candidate-logzhat[..., None])), -1)
    return dict(referenceProbability=p, candidateProbability=phat, delta=delta,
                mean=mean, variance=variance, kl=kl, directKLResidual=kl-direct_kl,
                referenceLogPartition=logz, candidateLogPartition=logzhat)
