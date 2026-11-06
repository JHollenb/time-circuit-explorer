"""Retained-state geometry of conditional row-sign reuse; no learned operations."""
import numpy as np


def support_closed(signs, permitted):
    signs = np.asarray(signs)
    permitted = np.asarray(permitted)
    if signs.ndim != 1 or not np.isin(signs, [-1, 1]).all():
        raise ValueError('Expected a complete vector of row signs')
    if permitted.dtype != bool or permitted.shape != (len(signs), len(signs)):
        raise ValueError('Expected the actual Boolean query/key support')
    nonempty = permitted.any(axis=1)
    return nonempty & ~np.any(permitted & (signs[:, None] != signs[None, :]), axis=1)


def decompose(positive, edited, signs):
    """Account for H'-sH, retaining the BF16 addition remainder as observed data.

    Inputs are already evaluated R/M/Hnext. This is retrospective accounting,
    not a prediction of an edited MLP or a next-layer computation.
    """
    positive = {k: np.asarray(positive[k], dtype=np.float64) for k in ('R', 'M', 'Hnext')}
    edited = {k: np.asarray(edited[k], dtype=np.float64) for k in positive}
    shape = positive['Hnext'].shape
    if len(shape) != 2 or any(x.shape != shape or not np.isfinite(x).all()
                              for x in [*positive.values(), *edited.values()]):
        raise ValueError('Expected matching finite row/feature fields')
    signs = np.asarray(signs)
    if signs.shape != (shape[0],) or not np.isin(signs, [-1, 1]).all():
        raise ValueError('Row signs differ from complete state geometry')
    s = signs[:, None]
    remainder = lambda p: p['Hnext'] - p['R'] - p['M']
    terms = {'residual': edited['R'] - s * positive['R'],
             'mlp': edited['M'] - s * positive['M'],
             'additionRemainder': remainder(edited) - s * remainder(positive)}
    defect = edited['Hnext'] - s * positive['Hnext']
    # An enlarged family: allow any real scale, not only the proposed row sign.
    h = positive['Hnext']; target = edited['Hnext']
    norm2 = np.sum(h * h, axis=1)
    scale = np.divide(np.sum(h * target, axis=1), norm2,
                      out=np.zeros(len(h)), where=norm2 != 0)
    orthogonal = target - scale[:, None] * h
    return {'defect': defect, **terms, 'orthogonal': orthogonal,
            'bestScale': scale, 'signedScale': signs * scale, 'scaleDefined': norm2 != 0,
            'closure': defect - sum(terms.values())}


def account(a, target, use):
    """Signed component energies and scale-fit residuals for a declared region."""
    use = np.asarray(use, dtype=bool)
    if not use.any():
        return None
    part = lambda k: a[k][use]
    energy = lambda x: float(np.mean(x * x))
    d = part('defect'); e = energy(d); denominator = energy(np.asarray(target)[use])
    names = ('residual', 'mlp', 'additionRemainder')
    terms = {k: energy(part(k)) for k in names}
    cross = {x + '__' + y: float(2 * np.mean(part(x) * part(y)))
             for i, x in enumerate(names) for y in names[i + 1:]}
    row_changed = np.any(d != 0, axis=1)
    orth = part('orthogonal')
    defined = use & a['scaleDefined']
    return {'rows': int(use.sum()), 'scalars': int(d.size),
            'changedRows': int(row_changed.sum()), 'changedScalars': int(np.count_nonzero(d)),
            'defectRms': e ** .5, 'endpointRms': denominator ** .5,
            'defectOverEndpointRms': (e / denominator) ** .5 if denominator else None,
            'maxAbsDefect': float(np.max(np.abs(d))),
            'termEnergies': terms, 'signedCrossEnergies': cross,
            'energyClosure': e - sum(terms.values()) - sum(cross.values()),
            'vectorClosureMaxAbs': float(np.max(np.abs(part('closure')))),
            'bestScaleOrthogonalRms': energy(orth) ** .5,
            'bestScaleOrthogonalOverEndpointRms': (energy(orth) / denominator) ** .5 if denominator else None,
            'bestScaleOrthogonalOverSignDefectRms': (energy(orth) / e) ** .5 if e else None,
            'bestScaleMin': float(np.min(a['bestScale'][defined])) if defined.any() else None,
            'bestScaleMax': float(np.max(a['bestScale'][defined])) if defined.any() else None,
            'signedScaleMin': float(np.min(a['signedScale'][defined])) if defined.any() else None,
            'signedScaleMax': float(np.max(a['signedScale'][defined])) if defined.any() else None,
            'directionReversedRows': int(np.count_nonzero(a['signedScale'][defined] < 0)),
            'undefinedScaleRows': int(np.count_nonzero(~a['scaleDefined'][use]))}
