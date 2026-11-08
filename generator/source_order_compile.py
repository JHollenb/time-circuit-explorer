"""Paid offline source arithmetic and a bank-independent ordering contract."""
import gzip
import hashlib
import math
from pathlib import Path
import platform
import sys
import time
import numpy as np
from encoder_atoms import AtomStore, bf, checked, floats, read, save, sha, words

CONTRACT = 'scaled-fp64-coordinate-products-then-python-math-fsum-v1'


def source_operands(store_path, manifest_sha, alphabet, pad, length):
    store = AtomStore(store_path, manifest_sha)
    m = store.manifest
    if (m['headDim'], m['hiddenSize'], m['qWidth'], m['kvWidth'], m['sequenceLength'], len(store.ids)) != (128, 2560, 4096, 1024, 512, 151936):
        raise ValueError('Qualified token-local source geometry differs')
    if not np.array_equal(store.ids, np.arange(m['vocabSize'], dtype=np.uint32)):
        raise ValueError('Full dense vocabulary identity differs')
    all_ids = alphabet+[pad]
    atoms = store.gather(np.asarray(all_ids, dtype=np.int64))
    rope = np.load(checked(store.root, m['rope']), allow_pickle=False)
    if rope.shape != (2, length, 128) or rope.dtype != np.dtype('<u2'):
        raise ValueError('Qualified rotary words differ')
    cos, sin = [floats(a) for a in rope]

    def rotate(value):
        value = floats(value)[:, None, :]
        turn = np.concatenate([-value[..., 64:], value[..., :64]], axis=-1)
        return words(bf(value*cos[None])+bf(turn*sin[None]))

    q_words = rotate(atoms[-1:, m['hiddenSize']:m['hiddenSize']+128])[0]
    key_start = m['hiddenSize']+m['qWidth']
    k_words = rotate(atoms[:-1, key_start:key_start+128])
    scale = 128**.25
    q = floats(q_words).astype(np.float64)/scale
    k = floats(k_words).astype(np.float64)/scale
    used = [store.root/'manifest.json', store.root/m['coverage']['path'], store.root/m['rowHashes']['path'], store.root/m['rope']['path']]
    used += [store.root/m['shards'][i]['path'] for i in sorted(store.shards)]
    return q, k, q_words, k_words, {'sourceFiles': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in used],
        'compiledRows': len(store.cache), 'compiledRowPayloadBytesRead': store.rowBytesRead,
        'scale': scale, 'scaleHex': scale.hex(), 'operandLogicalBytes': q.nbytes+k.nbytes+q_words.nbytes+k_words.nbytes,
        'dtype': 'source BF16 words; scaled FP64 coordinates', 'numpy': np.__version__, 'python': sys.version, 'platform': platform.platform()}


def scores(query, keys, counters, batch=128):
    """Concrete elementwise products; only math.fsum performs the reduction."""
    query = np.asarray(query, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    if keys.ndim != 2 or query.shape != (keys.shape[1],) or not np.isfinite(query).all() or not np.isfinite(keys).all():
        raise ValueError('Finite score operand geometry differs')
    output = np.empty(len(keys), dtype=np.float64)
    for lo in range(0, len(keys), batch):
        products = np.multiply(keys[lo:lo+batch], query, dtype=np.float64)
        for j, row in enumerate(products.tolist()):
            output[lo+j] = math.fsum(row)
        counters['coordinateProducts'] += products.size
        counters['fsumCalls'] += len(products)
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite deterministic score')
    return output


def compile_order(q, k, alphabet, pad, out):
    out = Path(out)
    out.mkdir(exist_ok=False)
    length, dim = q.shape
    if k.shape != (len(alphabet), length, dim) or len(alphabet)*max(length-1, 0) > 65536:
        raise ValueError('Ordinal source geometry/type range differs')
    count = len(alphabet)*length*(length-1)//2
    table = np.empty(count, dtype='<u2')
    counters = {'coordinateProducts': 0, 'fsumCalls': 0}
    ledger = []
    started = time.perf_counter()
    base = 0
    for clock in range(length):
        tick = time.perf_counter()
        bank = k[:, :clock, :].reshape(-1, dim)
        values = scores(q[clock], bank, counters)
        positions = np.tile(np.arange(clock, dtype=np.int64), len(alphabet))
        tokens = np.repeat(np.asarray(alphabet, dtype=np.int64), clock)
        order = np.lexsort((tokens, positions, -values))
        ranks = np.empty(len(order), dtype='<u2')
        ranks[order] = np.arange(len(order), dtype=np.uint16)
        if not np.array_equal(np.sort(ranks), np.arange(len(order), dtype=np.uint16)):
            raise ValueError('Clock ordinals are not a complete permutation')
        table[base:base+len(ranks)] = ranks
        base += len(ranks)
        ledger.append({'clock': clock, 'candidates': len(bank), 'coordinateProducts': len(bank)*dim,
            'scoreSHA256': hashlib.sha256(values.tobytes()).hexdigest(), 'rankSHA256': hashlib.sha256(ranks.tobytes()).hexdigest(),
            'equalAdjacentScorePairs': int(np.sum(np.diff(values[order]) == 0)), 'seconds': time.perf_counter()-tick})
        if clock % 64 == 0 or clock == length-1:
            print('SOURCE_ORDER_CLOCK', clock, 'products', counters['coordinateProducts'], 'seconds', time.perf_counter()-started, flush=True)
    if base != count or counters != {'coordinateProducts': count*dim, 'fsumCalls': count}:
        raise ValueError('Complete compiler work/count instrument differs')
    computed = time.perf_counter()
    path = out/'order.u16.gz'
    raw = table.tobytes()
    with path.open('xb') as f:
        with gzip.GzipFile(fileobj=f, filename='', mode='wb', compresslevel=9, mtime=0) as stream:
            stream.write(raw)
    metadata = {'schema': 'tce-source-order-program-v1', 'head': 0, 'paddingQueryID': pad, 'alphabet': alphabet,
        'length': length, 'ordinalCount': count, 'rawBytes': len(raw), 'rawSHA256': hashlib.sha256(raw).hexdigest(),
        'compressedBytes': path.stat().st_size, 'compressedSHA256': sha(path), 'scoreContract': CONTRACT,
        'serialization': 'uint16 little-endian; query clock, sorted token ID, key position; gzip9 empty filename mtime0',
        'scope': 'All covered histories under the new bank-independent score contract. Old NumPy finite parity is not universally certified.'}
    save(out/'program.json', metadata)
    save(out/'compiler.json', {'schema': 'tce-source-order-compiler-v1', 'work': counters, 'clocks': ledger,
        'scoreSortAndPermutationSeconds': computed-started, 'serializationSeconds': time.perf_counter()-computed,
        'scope': 'Paid offline learned-attention arithmetic. Coordinate products are not an exhaustive instruction count; rotary/scaling/sorting/hash/NumPy/Python work additional.'})
    return metadata
