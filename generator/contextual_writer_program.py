"""Two shared branch/constant writers; serving uses only stdlib and a tokenizer.

NumPy is imported only by offline fitting. No learned affine is a serving op.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import struct
import time
from template_cofactor_program import TokenInput, read, save, sha

COORDINATES = list(range(0, 2560, 160))
BUDGET = {'slots': 2, 'depth': 4, 'maxLeavesPerSlot': 16, 'minLeafRows': 8,
          'thresholdsPerFeature': 16, 'maxLiteralFeatures': 64, 'coordinates': COORDINATES}
FINITE = 'Each slot: Python binary64 H+atom, round to IEEE FP32 RNE, then BF16 RNE; slot0 before slot1.'


def decode_words(raw):
    count = len(raw)//2
    words = struct.unpack('<'+str(count)+'H', raw)
    return list(struct.unpack('<'+str(count)+'f', struct.pack('<'+str(count)+'I', *(w << 16 for w in words))))


def add_words(values, atom):
    values = [a+b for a, b in zip(values, atom)]
    bits = struct.unpack('<'+str(len(values))+'I', struct.pack('<'+str(len(values))+'f', *values))
    if any(b & 0x7f800000 == 0x7f800000 for b in bits):
        raise ValueError('Nonfinite writer output')
    words = [((b+0x7fff+((b >> 16) & 1)) >> 16) & 65535 for b in bits]
    if any(w & 0x7f80 == 0x7f80 for w in words): raise ValueError('Nonfinite rounded writer output')
    return struct.pack('<'+str(len(words))+'H', *words)


def validate(packet, length=512, vocab=151936):
    ids, mask, pos = [packet[k] for k in ('ids', 'mask', 'positions')]
    if any(len(v) != length for v in (ids, mask, pos)) or any(type(v) is not int for v in pos) or pos != list(range(length)):
        raise ValueError('Complete ordered packet required')
    if any(type(v) is not int or not 0 <= v < vocab for v in ids):
        raise ValueError('Invalid token')
    if any(type(v) is not int or v not in (0, 1) for v in mask) or any(a < b for a, b in zip(mask, mask[1:])):
        raise ValueError('Right-padded native mask required')


def features(packet, lexical, literals, coordinates=COORDINATES):
    """Online causal registers, including order; no final length or future input."""
    counts = dict.fromkeys(literals, 0); first = dict.fromkeys(literals, -1); last = dict.fromkeys(literals, -1)
    sums = [0.0]*len(coordinates); active = 0; result = []
    for p, (token, mask, row) in enumerate(zip(packet['ids'], packet['mask'], lexical)):
        if mask:
            active += 1
            for j, coordinate in enumerate(coordinates): sums[j] += float(row[coordinate])
            if token in counts:
                counts[token] += 1
                if first[token] == -1: first[token] = p
                last[token] = p
        values = [p, mask, active]
        for token_id in literals: values.extend([counts[token_id], first[token_id], last[token_id]])
        values.extend(float(row[c]) for c in coordinates)
        values.extend(s/active if active else 0.0 for s in sums)
        result.append(values)
    return result


def feature_names(literals, coordinates=COORDINATES):
    return ['position', 'mask', 'causalActiveCount'] + [kind+':'+str(t) for t in literals for kind in ('count', 'first', 'last')] + ['lexical:'+str(c) for c in coordinates] + ['causalMean:'+str(c) for c in coordinates]


def route(tree, row):
    decisions = 0
    while 'atom' not in tree:
        decisions += 1
        tree = tree['left'] if row[tree['feature']] <= tree['threshold'] else tree['right']
    return tree['atom'], decisions


def restore_program(source, destination):
    """Copy one immutable branch, verifying both source and restored bytes."""
    source, destination = Path(source), Path(destination)
    expected = {p.name: sha(p) for p in source.iterdir()}
    shutil.copytree(source, destination)
    if {p.name: sha(p) for p in source.iterdir()} != expected or {p.name: sha(p) for p in destination.iterdir()} != expected:
        raise ValueError('Immutable program changed during restore')
    return expected


def fit_tree(x, y, weights, columns, budget=BUDGET):
    """Deterministic weighted multioutput CART; fixed finite threshold budget."""
    import numpy as np
    atoms = []; trace = []
    def grow(indices, depth):
        w = weights[indices]; target = y[indices]; total = (target*w[:, None]).sum(0); mass = float(w.sum())
        parent = float(total@total/mass); best = None
        if depth < budget['depth'] and len(indices) >= 2*budget['minLeafRows']:
            for column in columns:
                order = np.argsort(x[indices, column], kind='stable'); values = x[indices[order], column]
                cuts = np.flatnonzero(values[:-1] < values[1:])
                cuts = cuts[(cuts+1 >= budget['minLeafRows']) & (len(indices)-cuts-1 >= budget['minLeafRows'])]
                if len(cuts) > budget['thresholdsPerFeature']:
                    cuts = cuts[np.linspace(0, len(cuts)-1, budget['thresholdsPerFeature'], dtype=int)]
                if not len(cuts): continue
                cw = np.cumsum(w[order]); cy = np.cumsum(target[order]*w[order, None], axis=0)
                left = cy[cuts]; right = total-left
                scores = np.einsum('ij,ij->i', left, left)/cw[cuts] + np.einsum('ij,ij->i', right, right)/(mass-cw[cuts])
                at = int(np.argmax(scores)); gain = float(scores[at]-parent)
                if best is None or gain > best[0]:
                    cut = int(cuts[at]); best = (gain, column, float(values[cut]), indices[order[:cut+1]], indices[order[cut+1:]])
        if best is None or best[0] <= 0:
            atom = len(atoms); atoms.append((total/mass).astype('<f4'))
            trace.append({'atom': atom, 'rows': indices.tolist(), 'weight': mass})
            return {'atom': atom}
        gain, column, threshold, left, right = best
        return {'feature': column, 'threshold': threshold, 'gain': gain,
                'left': grow(left, depth+1), 'right': grow(right, depth+1)}
    tree = grow(np.arange(len(y)), 0)
    return tree, np.stack(atoms), trace


def finite_add_array(a, b):
    import numpy as np
    value = (np.asarray(a, np.float64)+np.asarray(b, np.float64)).astype('<f4')
    if not np.isfinite(value).all(): raise ValueError('Nonfinite offline writer output')
    bits = value.view('<u4'); words = ((bits+np.uint32(0x7fff)+((bits >> 16) & 1)) >> 16).astype('<u2')
    if np.any((words & np.uint16(0x7f80)) == np.uint16(0x7f80)): raise ValueError('Nonfinite rounded offline writer output')
    return (words.astype('<u4') << 16).view('<f4')


def serve(spec_path, output):
    from token_history_process import isolate, NoEvaluation
    spec = read(spec_path); base = Path(spec['lexical']); output = Path(output).resolve(); output.mkdir()
    tick = time.perf_counter(); cpu = time.process_time(); manifest = read(base/'manifest.json')
    tokenizer = TokenInput(base, manifest); tokenizer.tokenize('')
    opens, denied, imports = isolate(spec, output)
    for name in ('h0.bf16', 'row-hashes.bin'):
        if sha(base/name) != manifest['files'][name]['sha256']: raise ValueError('Lexical asset changed')
    programs = {}; loaded = time.perf_counter()
    for branch in spec['programs']:
        folder = Path(branch['path']); p = read(folder/'program.json')
        if sha(folder/'program.json') != branch['sha256']: raise ValueError('Program changed')
        dictionaries = []
        for item in p['dictionaries']:
            raw = (folder/item['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest() != item['sha256'] or len(raw) != item['atoms']*2560*4: raise ValueError('Dictionary changed')
            values = struct.unpack('<'+str(len(raw)//4)+'f', raw)
            dictionaries.append([values[i:i+2560] for i in range(0, len(values), 2560)])
        prefix = (folder/'prefix.bf16').read_bytes()
        if hashlib.sha256(prefix).hexdigest() != p['prefixSHA256'] or len(prefix) != 10240: raise ValueError('Prefix changed')
        programs[branch['name']] = (p, dictionaries, prefix)
    loaded = time.perf_counter()-loaded; cache = {}; rows = []
    with (base/'h0.bf16').open('rb') as table, (base/'row-hashes.bin').open('rb') as hashes:
        for case in spec['cases']:
            started = time.perf_counter(); packet, text = tokenizer.tokenize(case['prompt']); validate(packet)
            if packet['ids'][:2] != [151644, 872] or packet['mask'][:2] != [1, 1]: raise ValueError('Unexpected template prefix')
            for token in set(packet['ids']):
                if token not in cache:
                    table.seek(token*5120); raw = table.read(5120); hashes.seek(token*32)
                    if len(raw) != 5120 or hashlib.sha256(raw).digest() != hashes.read(32): raise ValueError('Lexical row changed')
                    cache[token] = (raw, decode_words(raw))
            lexical = [cache[t][1] for t in packet['ids']]
            h0 = b''.join(cache[t][0] for t in packet['ids']); h0_path = output/(case['id']+'-H0.bf16'); h0_path.write_bytes(h0)
            prepared = time.perf_counter(); predictions = []
            for name, (p, dictionaries, prefix) in programs.items():
                begin = time.perf_counter(); x = features(packet, lexical, p['literals']); feature_done = time.perf_counter()
                data = bytearray(prefix); leaves = []; decisions = 0
                for pos in range(2, 512):
                    addresses = []
                    value = lexical[pos]
                    for tree, atoms in zip(p['trees'], dictionaries):
                        address, n = route(tree, x[pos]); decisions += n; addresses.append(address)
                        raw = add_words(value, atoms[address]); value = decode_words(raw)
                    data.extend(raw); leaves.append(addresses)
                path = output/(name+'-'+case['id']+'.bf16'); path.write_bytes(data)
                predictions.append({'method': name, 'path': str(path), 'sha256': sha(path), 'bytes': len(data),
                    'programSHA256': next(b['sha256'] for b in spec['programs'] if b['name'] == name),
                    'featureSeconds': feature_done-begin, 'writerAndSerializeSeconds': time.perf_counter()-feature_done,
                    'branchDecisions': decisions, 'coordinateAdditions': 2*510*2560, 'leaves': leaves})
            rows.append({'id': case['id'], 'packet': packet, 'renderedTextSHA256': hashlib.sha256(text.encode()).hexdigest(),
                'H0': {'path': str(h0_path), 'sha256': sha(h0_path), 'bytes': len(h0)}, 'activeRows': sum(packet['mask']),
                'tokenizeGatherSeconds': prepared-started, 'predictions': predictions})
    save(output/'receipt.json', {'schema': 'tce-contextual-writer-serving-v1', 'rows': rows,
        'dictionaryLoadSeconds': loaded, 'wallSeconds': time.perf_counter()-tick, 'cpuSeconds': time.process_time()-cpu,
        'highWaterRSSKiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'uniqueLexicalRows': len(cache),
        'successfulReadPaths': sorted(opens), 'blockedImports': imports, 'fileCanaryDenials': denied,
        'loadedNumericalRuntimeRoots': sorted(NoEvaluation.roots.intersection(__import__('sys').modules)),
        'finitePolicy': FINITE, 'newNeuralEvaluations': 0, 'scope': 'Trusted-source guard after tokenizer warmup; not an adversarial sandbox. Full H1 predicted, not certified accurate. No native-state or model access.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('spec'); parser.add_argument('output'); args = parser.parse_args()
    serve(args.spec, args.output)
