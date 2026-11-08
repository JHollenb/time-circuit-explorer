"""Paid offline score compiler; values stay near compute, never in publication."""
import hashlib
from pathlib import Path
import time
import numpy as np
from source_order_compile import CONTRACT, scores
from token_history_process import save, sha


def compile_mass(q, keys, rank_metadata, ledger, rank_metadata_path, output):
    from compiled_mass_program import NORMALIZER
    output = Path(output)
    output.mkdir(exist_ok=False)
    length, dim = q.shape
    alphabet = rank_metadata['alphabet']
    count = len(alphabet)*length*(length-1)//2
    if (keys.shape != (len(alphabet), length, dim) or rank_metadata['length'] != length or
        rank_metadata['ordinalCount'] != count or rank_metadata['scoreContract'] != CONTRACT or len(ledger['clocks']) != length):
        raise ValueError('Frozen score/rank domain mismatch')
    counters = {'coordinateProducts': 0, 'fsumCalls': 0}
    clocks = []
    start = time.perf_counter()
    path = output/'scores.f64'
    with path.open('xb') as stream:
        for clock in range(length):
            before = time.perf_counter()
            values = scores(q[clock], keys[:, :clock, :].reshape(-1, dim), counters)
            raw = values.astype('<f8', copy=False).tobytes()
            digest = hashlib.sha256(raw).hexdigest()
            old = ledger['clocks'][clock]
            if old['clock'] != clock or old['candidates'] != len(values) or old['scoreSHA256'] != digest:
                raise ValueError('Prior qualified full-clock score bytes differ')
            stream.write(raw)
            clocks.append({'clock': clock, 'scores': len(values), 'bytes': len(raw), 'scoreSHA256': digest,
                           'matchesQualified95Clock': True, 'seconds': time.perf_counter()-before})
            if clock % 128 == 0 or clock == length-1:
                print('COMPILED_MASS_CLOCK', clock, counters, flush=True)
    if counters != {'coordinateProducts': count*dim, 'fsumCalls': count} or path.stat().st_size != count*8:
        raise ValueError('Score compiler work/extent mismatch')
    computed = time.perf_counter()
    metadata = {'schema': 'tce-compiled-selected-mass-v1',
                **{k: rank_metadata[k] for k in ['head', 'paddingQueryID', 'alphabet', 'length', 'ordinalCount', 'scoreContract']},
                'normalizer': NORMALIZER, 'scoreBytes': path.stat().st_size, 'scoreSHA256': sha(path),
                'rankMetadataSHA256': sha(rank_metadata_path),
                'serialization': 'FP64 little-endian; clock/token/key-position; same triangular offsets as rank table',
                'scope': 'Input-addressed partial evaluation of one learned primitive; all source work paid. No native softmax finite-parity or general encoder claim.'}
    save(output/'program.json', metadata)
    save(output/'compiler.json', {'schema': 'tce-compiled-mass-compiler-v1', 'work': counters, 'clocks': clocks,
                                'computeAndWriteSeconds': computed-start, 'finalHashAndMetadataSeconds': time.perf_counter()-computed})
    return metadata
