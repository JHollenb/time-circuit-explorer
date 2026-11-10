"""Public integer census and supplied algebra for a proposed causal-order split.

No scientific tensors, tokenizer, numerical runtime, fitted vector or model access.
Reported old leaf assignments remain source-qualified producer observations.
"""
import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time


PINS = {
    'programs.json': 'd1134a1b7e549a56d1e3e54fd09d47ef6963f9011c88efbd6081bab4ccd1768a',
    'final-serving.json': 'c6eb72f4d3cc3cd1fbb952d5b242fa2a99cb5696b76859aac65fd18577ced265',
}
A, B = 18575, 94607


def order_registers(packet):
    """Current prefix only. Missing/equal first positions have neutral order 0."""
    first = {A: -1, B: -1}
    result = []
    for p, (token, mask) in enumerate(zip(packet['ids'], packet['mask'])):
        if mask and token in first and first[token] == -1:
            first[token] = p
        a, b = first[A], first[B]
        order = (int(a < b) - int(a > b)) if a >= 0 and b >= 0 else 0
        result.append((a, b, order))
    return result


def refine(tree, column, neutral=8, negative=9, positive=10):
    """Two ordinary binary threshold nodes; leave the neutral parent address."""
    if 'atom' in tree:
        if tree['atom'] != neutral:
            return deepcopy(tree)
        return {'feature': column, 'threshold': -0.5, 'left': {'atom': negative},
                'right': {'feature': column, 'threshold': 0.5,
                          'left': {'atom': neutral}, 'right': {'atom': positive}}}
    result = deepcopy(tree)
    result['left'] = refine(tree['left'], column, neutral, negative, positive)
    result['right'] = refine(tree['right'], column, neutral, negative, positive)
    return result


def leaf_depths(tree, depth=0):
    if 'atom' in tree:
        return [(tree['atom'], depth)]
    return leaf_depths(tree['left'], depth + 1) + leaf_depths(tree['right'], depth + 1)


def route(tree, row):
    while 'atom' not in tree:
        tree = tree['left'] if row[tree['feature']] <= tree['threshold'] else tree['right']
    return tree['atom']


def supplied_mechanics():
    # Future input cannot affect an already emitted register; translating both
    # first positions preserves relative order. These are supplied integer cases.
    left = {'ids': [7, A, 4, B, 5, 0], 'mask': [1, 1, 1, 1, 1, 0]}
    right = {'ids': [7, B, 4, A, 5, 0], 'mask': [1, 1, 1, 1, 1, 0]}
    l, r = order_registers(left), order_registers(right)
    assert [v[2] for v in l] == [0, 0, 0, 1, 1, 1]
    assert [v[2] for v in r] == [0, 0, 0, -1, -1, -1]
    shifted = {'ids': [6, 6] + left['ids'], 'mask': [1, 1] + left['mask']}
    assert [v[2] for v in order_registers(shifted)][2:] == [v[2] for v in l]
    for cut in range(1, len(left['ids']) + 1):
        prefix = {k: v[:cut] for k, v in left.items()}
        assert order_registers(prefix) == l[:cut]
    assert order_registers({'ids': [A, B], 'mask': [1, 0]})[-1][2] == 0
    miniature = refine({'feature': 0, 'threshold': 0,
                        'left': {'atom': 7}, 'right': {'atom': 8}}, 1)
    aliases = {7: 7, 8: 8, 9: 8, 10: 8}
    for old_feature in (-1, 1):
        for order in (-1, 0, 1):
            parent = 7 if old_feature <= 0 else 8
            child = route(miniature, [old_feature, order])
            assert aliases[child] == parent
            if order == 0:
                assert child == parent
    # g already contains objective endpoint/pair weights. Do not weight it twice.
    weights = [F(1), F(2), F(3)]
    signs = [1, 1, -1]
    gradients = [(F(2), F(-1)), (F(1), F(3)), (F(-2), F(4))]
    plus = sum(w for w, s in zip(weights, signs) if s == 1)
    minus = sum(w for w, s in zip(weights, signs) if s == -1)
    mass = plus + minus
    b = [minus / mass if s == 1 else -plus / mass for s in signs]
    d = plus * minus / mass
    s = [sum(bi * gi[j] for bi, gi in zip(b, gradients)) for j in range(2)]
    contrast = [-sj / d for sj in s]
    q = [[bi * cj for cj in contrast] for bi in b]
    slope = sum(gj * qj for gi, qi in zip(gradients, q) for gj, qj in zip(gi, qi))
    mean = [sum(w * qi[j] for w, qi in zip(weights, q)) for j in range(2)]
    assert mean == [0, 0]
    assert slope == -sum(sj * sj for sj in s) / d <= 0
    return {'cases': ['causal-prefix', 'translated-order', 'masked-missing-neutral',
                      'symbolic-parent-alias', 'weighted-contrast'],
            'contrastD': str(d), 'contrastS': list(map(str, s)),
            'contrastVector': list(map(str, contrast)), 'gDotQ': str(slope),
            'weightedMean': list(map(str, mean)),
            'scope': 'Supplied exact integer/rational mechanics, not a learned or finite writer result.'}


def investigate(source):
    objects = {}
    for name, expected in PINS.items():
        raw = (source / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Changed public input: ' + name)
        objects[name] = json.loads(raw)
    program = objects['programs.json']['accepted']['program']
    assert len(program['featureNames']) == 155
    assert [v['atoms'] for v in program['dictionaries']] == [14, 9]
    assert A in program['literals'] and B in program['literals']
    old = program['trees'][1]
    depths = leaf_depths(old)
    assert [v for v in depths if v[0] == 8] == [(8, 4)]
    new = refine(old, 155)
    census = []
    for item in objects['final-serving.json']['rows']:
        packet = item['packet']
        assert len(packet['ids']) == len(packet['mask']) == 512
        predictions = [v for v in item['predictions'] if v['method'] == 'accepted']
        assert len(predictions) == 1 and len(predictions[0]['leaves']) == 510
        registers = order_registers(packet)
        counts = Counter()
        first_seen = {}
        for p, leaves in enumerate(predictions[0]['leaves'], 2):
            if leaves[1] == 8:
                a, b, sign = registers[p]
                child = 9 if sign < 0 else 10 if sign > 0 else 8
                counts[(sign, 'active' if packet['mask'][p] else 'padding', child)] += 1
                first_seen.setdefault(str(sign), {'row': p, 'firstA': a, 'firstB': b})
        census.append({'id': item['id'], 'oldLeaf8Rows': sum(counts.values()),
                       'groups': [{'order': s, 'stratum': t, 'newLeaf': c, 'rows': n}
                                  for (s, t, c), n in sorted(counts.items())],
                       'firstExamples': first_seen})
    return {'schema': 'tce-causal-order-public-mechanics-v1', 'inputSHA256': PINS,
            'literals': [A, B], 'oldLeaf8Depth': 4,
            'oldSlot1Depth': max(d for _, d in depths),
            'proposedSlot1Depth': max(d for _, d in leaf_depths(new)),
            'featureCount': {'parent': 155, 'proposed': 156},
            'atomCount': {'parent': 23, 'proposed': 25},
            'addedAtomBytesFP32': 2 * 2560 * 4, 'proposedSlot1Tree': new,
            'census': census, 'suppliedMechanics': supplied_mechanics(),
            'scope': 'Fresh public token/mask integer census using reported old leaves. '
                     'Symbolic null follows from cloning atom bytes, not numerical replay. '
                     'No fitted contrast, refined serving artifact, original consumer, image or scientific array.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    result = investigate(args.source)
    result['sourceSHA256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['secondsAfterStandardLibraryImports'] = time.perf_counter() - started
    with args.destination.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps({'secondsAfterStandardLibraryImports': result['secondsAfterStandardLibraryImports'],
                      'sourceSHA256': result['sourceSHA256'], 'histories': len(result['census'])}))


if __name__ == '__main__':
    main()
