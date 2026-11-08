"""Fixed saved-U projection and signed finite-boundary accounts; NumPy only."""
import math
import numpy as np

UPDATE_TERMS = ['projectionResidual', 'finitePolicy', 'admission', 'updateClosure']
EXIT_TERMS = ['gatedProjection', 'gatedFinitePolicy', 'gatedAdmission', 'gateRemainder', 'commitRemainder', 'exitClosure']
DENOMINATORS = [name + suffix for name in ['update82', 'increment82', 'increment83', 'deltaUpdate', 'deltaExit'] for suffix in ['Energy', 'CenteredEnergy']]
SOURCES = ['rowBasis', 'updateFP64', 'update83', 'gated83', 'exit83', 'update82', 'gated82', 'exit82', 'gate', 'text', 'image']


def work(text=512, image=1024, width=3072, rank=128):
    n = text + image
    return {'matrixMACs': 2*n*rank*width, 'projectMACs': n*rank*width, 'reconstructMACs': n*rank*width,
        'matrixProducts': 4, 'gramScalarMultiplications': 2*(10+21)*n*width,
        'gramScalarAdditions': 2*(10+21)*n*(width-1), 'denominatorScalarMultiplications': 10*n*width,
        'denominatorScalarAdditions': 10*n*(width-1), 'newLearnedMapApplications': 0,
        'newNeuralEvaluations': 0, 'newCheckpointLoads': 0, 'newQR': 0, 'newFits': 0, 'newImages': 0,
        'scope': 'Four FP64 projector contractions. Explicit pairwise Gram/denominator reductions charged separately; means, elementwise differences/products, copies, hashes and retention also paid. Counts describe array geometry, not retired hardware instructions. Historical source/model formation remains paid.'}


def abis(text=512, image=1024, width=3072, rank=128):
    n = text + image
    names = ['projectedOutput', 'deltaUpdate', 'deltaExit', 'increment82', 'increment83', *UPDATE_TERMS, *EXIT_TERMS]
    result = {name: {'shape': [1, n, width], 'dtype': 'float64'} for name in names}
    result['projectedCoordinates'] = {'shape': [2, rank, width], 'dtype': 'float64'}
    for prefix, count in [('update', len(UPDATE_TERMS)), ('exit', len(EXIT_TERMS))]:
        result[prefix+'GramRows'] = {'shape': [n, count, count], 'dtype': 'float64'}
        result[prefix+'CenteredGramRows'] = {'shape': [n, count, count], 'dtype': 'float64'}
        result[prefix+'Means'] = {'shape': [2, count, width], 'dtype': 'float64'}
    result['rowDenominators'] = {'shape': [n, len(DENOMINATORS)], 'dtype': 'float64'}
    return result


def gram_rows(values, text):
    n, width = values[0].shape[1:]; count = len(values)
    full = np.empty((n, count, count), dtype=np.float64)
    centered = np.empty_like(full); means = np.empty((2, count, width), dtype=np.float64)
    for stream, (lo, hi) in enumerate([(0, text), (text, n)]):
        for i, value in enumerate(values):
            means[stream, i] = value[0, lo:hi].mean(0, dtype=np.float64)
        for i, a in enumerate(values):
            ac = a[0, lo:hi] - means[stream, i]
            for j in range(i, count):
                b = values[j]; bc = b[0, lo:hi] - means[stream, j]
                full[lo:hi, i, j] = full[lo:hi, j, i] = np.sum(a[0, lo:hi]*b[0, lo:hi], -1, dtype=np.float64)
                centered[lo:hi, i, j] = centered[lo:hi, j, i] = np.sum(ac*bc, -1, dtype=np.float64)
    return full, centered, means


def compute(f, matmul, text=512):
    if set(f) != set(SOURCES) or any(a.dtype != np.float64 for a in f.values()):
        raise ValueError('Admitted sources must be promoted float64 arrays')
    n, width = f['update82'].shape[1:]; rank = f['rowBasis'].shape[-1]
    expected = {name: [1, n, width] for name in SOURCES}
    expected.update(rowBasis=[1, n, rank], gate=[1, 1, width], text=[1, text, width], image=[1, n-text, width])
    if not 0 < text < n or any(list(f[name].shape) != shape for name, shape in expected.items()):
        raise ValueError('Saved source shape differs')
    projected = np.empty_like(f['update82']); coordinates = np.empty((2, rank, width), dtype=np.float64)
    for stream, (lo, hi) in enumerate([(0, text), (text, n)]):
        u = f['rowBasis'][0, lo:hi]
        coordinates[stream] = matmul('project'+str(stream), u.T, f['update82'][0, lo:hi])
        projected[0, lo:hi] = matmul('reconstruct'+str(stream), u, coordinates[stream])
    a = f['update82'] - projected
    b = projected - f['updateFP64']
    c = f['updateFP64'] - f['update83']
    dy = f['update82'] - f['update83']
    ru = dy - ((a+b)+c)
    g = f['gate']
    ga, gb, gc = g*a, g*b, g*c
    dg = f['gated82'] - f['gated83']
    dx = f['exit82'] - f['exit83']
    rg = dg - g*dy
    rx = dx - dg
    closure = dx - ((((ga+gb)+gc)+rg)+rx)
    incoming = np.concatenate([f['text'], f['image']], 1)
    inc82 = f['exit82'] - incoming; inc83 = f['exit83'] - incoming
    result = dict(zip(UPDATE_TERMS, [a, b, c, ru]))
    result.update(zip(EXIT_TERMS, [ga, gb, gc, rg, rx, closure]))
    result.update(projectedCoordinates=coordinates, projectedOutput=projected, deltaUpdate=dy, deltaExit=dx, increment82=inc82, increment83=inc83)
    for prefix, names in [('update', UPDATE_TERMS), ('exit', EXIT_TERMS)]:
        full, centered, means = gram_rows([result[name] for name in names], text)
        result[prefix+'GramRows'] = full; result[prefix+'CenteredGramRows'] = centered; result[prefix+'Means'] = means
    den = np.empty((n, len(DENOMINATORS)), dtype=np.float64)
    for i, value in enumerate([f['update82'], inc82, inc83, dy, dx]):
        den[:, 2*i] = np.sum(value[0]*value[0], -1, dtype=np.float64)
        for lo, hi in [(0, text), (text, n)]:
            centered = value[0, lo:hi] - value[0, lo:hi].mean(0, dtype=np.float64)
            den[lo:hi, 2*i+1] = np.sum(centered*centered, -1, dtype=np.float64)
    result['rowDenominators'] = den
    return result


def observations(result, text=512):
    for i, values in enumerate(result['rowDenominators']):
        yield {'row': i, 'stratum': 'text' if i < text else 'image', 'denominators': dict(zip(DENOMINATORS, map(float, values))),
            **{name: result[name][i].tolist() for name in ['updateGramRows', 'updateCenteredGramRows', 'exitGramRows', 'exitCenteredGramRows']}}


def summarize(rows):
    den = {name: math.fsum(row['denominators'][name] for row in rows) for name in DENOMINATORS}
    result = {'rows': len(rows), 'denominators': den}
    for prefix, terms, delta, baseline in [('update', UPDATE_TERMS, 'deltaUpdate', 'update82'), ('exit', EXIT_TERMS, 'deltaExit', 'increment82')]:
        panel = {'terms': terms}
        for centered in [False, True]:
            suffix = 'CenteredEnergy' if centered else 'Energy'
            key = prefix+('Centered' if centered else '')+'GramRows'
            matrix = [[math.fsum(row[key][i][j] for row in rows) for j in range(len(terms))] for i in range(len(terms))]
            total = den[delta+suffix]; scale = den[baseline+suffix]
            panel['centered' if centered else 'full'] = {'gram': matrix, 'directDifferenceEnergy': total, 'baselineEnergy': scale,
                'differenceOverBaselineRMS': math.sqrt(total/scale) if scale else None,
                'componentNormOverDifference': {name: math.sqrt(matrix[i][i]/total) if total else None for i, name in enumerate(terms)},
                'energyAccountRemainder': total-math.fsum(x for line in matrix for x in line)}
        result[prefix] = panel
    return result
