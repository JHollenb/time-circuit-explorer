"""Integer-only execution of a source-derived, padding-query ordering table."""
from array import array
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_history_address import validate_packet
from token_history_process import isolate, read, save, sha, write_rows

METHODS = ('primary', 'clockPlus1', 'reverseRanks', 'recent')


def offset(clock, token_index, key_position, alphabet_size):
    if not (0 <= key_position < clock and 0 <= token_index < alphabet_size):
        raise ValueError('Integer rank address outside causal table')
    return alphabet_size*clock*(clock-1)//2 + token_index*clock + key_position


class OrderProgram:
    def __init__(self, meta, ranks):
        self.meta = meta
        self.alphabet = meta['alphabet']
        self.length = meta['length']
        self.pad = meta['paddingQueryID']
        if meta['schema'] != 'tce-source-order-program-v1' or meta['head'] != 0:
            raise ValueError('Wrong source-order program interface')
        if self.alphabet != sorted(set(self.alphabet)) or not self.alphabet or any(type(x) is not int or x < 0 for x in self.alphabet):
            raise ValueError('Invalid finite alphabet')
        if type(self.length) is not int or not 1 <= self.length <= 512 or len(self.alphabet)*(self.length-1) > 65536:
            raise ValueError('Clock/ordinal type range differs')
        n = len(self.alphabet)*self.length*(self.length-1)//2
        if len(ranks) != n or n != meta['ordinalCount']:
            raise ValueError('Ordinal count differs')
        self.index = {token: i for i, token in enumerate(self.alphabet)}
        self.ranks = ranks

    @classmethod
    def load(cls, metadata_path, bundle_path):
        meta = read(metadata_path)
        if sha(bundle_path) != meta['compressedSHA256']:
            raise ValueError('Compressed order bytes differ')
        with gzip.open(bundle_path, 'rb') as f:
            raw = f.read(meta['rawBytes']+1)
        if len(raw) != meta['rawBytes'] or hashlib.sha256(raw).hexdigest() != meta['rawSHA256']:
            raise ValueError('Raw order bytes differ')
        ranks = array('H')
        if ranks.itemsize != 2:
            raise ValueError('Two-byte unsigned integer representation required')
        ranks.frombytes(raw)
        if sys.byteorder != 'little':
            ranks.byteswap()
        return cls(meta, ranks)

    def serve(self, packet, method='primary'):
        validate_packet(packet)
        if method not in METHODS:
            raise ValueError('Unknown fixed control')
        history = []
        unknown = []
        rows = []
        work = {'steps': 0, 'admissions': 0, 'unknownAdmissions': 0, 'rankLookups': 0, 'integerRankComparisons': 0, 'recentPositionReads': 0, 'supportedQueries': 0}
        for token, active, clock in zip(packet['ids'], packet['mask'], packet['positions']):
            if clock != work['steps']:
                raise ValueError('Own history clock differs')
            if active:
                history.append((token, clock))
                work['admissions'] += 1
                if token not in self.index:
                    unknown.append(clock)
                    work['unknownAdmissions'] += 1
            work['steps'] += 1
            reason = ('clock-outside-domain' if clock >= self.length else 'active-query-outside-padding-contract' if active else
                'wrong-query-token' if token != self.pad else 'unknown-admitted-token' if unknown else
                'control-clock-outside-domain' if method == 'clockPlus1' and clock+1 >= self.length else None)
            if reason:
                rows.append({'clock': clock, 'prefixCount': len(history), 'status': reason, 'addresses': None, 'unknownPositions': unknown.copy()})
                continue
            work['supportedQueries'] += 1
            actual_clock = clock+1 if method == 'clockPlus1' else clock
            top = []
            if method == 'recent':
                addresses = [p for _, p in reversed(history[-4:])]
                work['recentPositionReads'] += len(addresses)
            else:
                for key_token, position in history:
                    rank = self.ranks[offset(actual_clock, self.index[key_token], position, len(self.alphabet))]
                    work['rankLookups'] += 1
                    if method == 'reverseRanks':
                        rank = len(self.alphabet)*actual_clock-1-rank
                    insert = 0
                    for existing, _ in top:
                        work['integerRankComparisons'] += 1
                        if rank < existing:
                            break
                        insert += 1
                    if insert < 4:
                        top.insert(insert, (rank, position))
                        top = top[:4]
                addresses = [p for _, p in top]
            rows.append({'clock': clock, 'readClock': actual_clock, 'prefixCount': len(history), 'status': 'supported',
                'addresses': addresses+[-1]*(4-len(addresses)), 'unknownPositions': []})
        return rows, {'work': work, 'finalHistoryTokens': len(history), 'historyLogicalInt64Bytes': len(history)*16,
            'scope': 'IDs/positions list payload only; table, dictionary, unknown positions, top-four temporaries, outputs and Python objects additional.'}


def execute(spec_path, output):
    start = time.perf_counter()
    cpu = time.process_time()
    spec = read(spec_path)
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    opens, denied, imports = isolate(spec, output)
    if sha(spec['metadata']) != spec['metadataSHA256']:
        raise ValueError('Sealed program metadata differs')
    tick = time.perf_counter()
    program = OrderProgram.load(spec['metadata'], spec['bundle'])
    load_seconds = time.perf_counter()-tick
    receipts = []
    for case in spec['cases']:
        packet = read(case['tokens'])
        for method in METHODS:
            tick = time.perf_counter()
            rows, work = program.serve(packet, method)
            computed = time.perf_counter()
            path = output/(case['id']+'-'+method+'.jsonl.gz')
            write_rows(path, rows)
            receipts.append({'case': case['id'], 'method': method, 'path': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size,
                'computeSeconds': computed-tick, 'serializationAndHashSeconds': time.perf_counter()-computed, **work})
    save(output/'receipt.json', {'schema': 'tce-source-order-serve-v1', 'rows': receipts, 'metadataSHA256': sha(spec['metadata']),
        'compressedSHA256': sha(spec['bundle']), 'loadDecompressAndHashSeconds': load_seconds,
        'fileCanaryDenials': denied, 'blockedImportCanaries': imports, 'successfulReadPaths': sorted(opens),
        'CPUSeconds': time.process_time()-cpu, 'wallSecondsBeforeReceipt': time.perf_counter()-start,
        'highWaterRSSKiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope': 'Trusted-source fresh integer execution; no Q/K/scores/weights/NumPy/model runtime. Exactness only under declared compiled comparator.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('spec')
    p.add_argument('output')
    a = p.parse_args()
    execute(a.spec, a.output)
