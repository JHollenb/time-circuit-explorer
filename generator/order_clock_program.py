"""Fixed COPY/INSERT order recurrence; standard-library serving only."""
from array import array
from collections import Counter
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import resource
import struct
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_history_address import validate_packet
from token_history_process import isolate, read, save, sha, write_rows

METHODS = ('primary', 'clockPlus1', 'reverseRanks', 'recent')


def little_bytes(values):
    out = array('H', values)
    if out.itemsize != 2: raise ValueError('Two-byte unsigned integers required')
    if sys.byteorder != 'little': out.byteswap()
    return out.tobytes()


def compile_program(ranks, alphabet, length):
    size = len(alphabet)
    if len(ranks) != size*length*(length-1)//2 or size*length > 65535:
        raise ValueError('Finite rank geometry')
    previous = []; bytecode = bytearray(); ledger = []; work = Counter(); lengths = Counter(); started = time.perf_counter()
    for clock in range(1, length):
        offset = size*clock*(clock-1)//2
        chunk = ranks[offset:offset+size*clock]
        if sorted(chunk) != list(range(size*clock)): raise ValueError('Source rank permutation')
        slots = sorted(range(size*clock), key=chunk.__getitem__)
        target = [(slot//clock)*length+slot%clock for slot in slots]
        inverse = {identity: p for p, identity in enumerate(previous)}
        ops = []; cursor = 0; copies = inserts = expanded = 0
        while cursor < len(target):
            identity = target[cursor]; work['targetVisits'] += 1
            if identity % length == clock-1:
                ops.append(struct.pack('<BH', 1, identity//length)); inserts += 1; cursor += 1
            else:
                start = inverse[identity]; count = 1
                while cursor+count < len(target) and start+count < len(previous):
                    work['contiguousIdentityComparisons'] += 1
                    if target[cursor+count] != previous[start+count]: break
                    count += 1
                ops.append(struct.pack('<BHH', 0, start, count)); copies += 1; expanded += count; lengths[count] += 1; cursor += count
        bytecode.extend(struct.pack('<H', len(ops)))
        for op in ops: bytecode.extend(op)
        work.update({'sourceRankReads': len(chunk), 'permutationSortCalls': 1, 'targetOrderSortCalls': 1,
                     'copyOperations': copies, 'insertOperations': inserts, 'copiedIdentities': expanded})
        ledger.append({'clock': clock, 'candidates': len(target), 'copies': copies, 'inserts': inserts,
                       'copiedIdentities': expanded, 'bytes': 2+5*copies+3*inserts})
        previous = target
        if clock % 64 == 0 or clock == length-1: print('ORDER_CLOCK_COMPILE', clock, 'bytes', len(bytecode), flush=True)
    return bytes(bytecode), {'work': dict(work), 'clocks': ledger, 'copyLengthHistogram': dict(sorted(lengths.items())), 'compileSeconds': time.perf_counter()-started,
                            'scope': 'Two integer sorts per clock plus fixed greedy maximal-run encoding. No score evaluation or format search.'}


class OrderClock:
    def __init__(self, metadata, bytecode):
        self.meta = metadata; self.data = bytecode
        self.length = metadata['length']; self.alphabet = metadata['alphabet']; self.size = len(self.alphabet)
        if metadata['schema'] != 'tce-order-clock-v1' or metadata['head'] != 0 or not 1 <= self.length <= 512:
            raise ValueError('Clock interface')
        if self.alphabet != sorted(set(self.alphabet)) or not self.alphabet or self.size*self.length > 65535:
            raise ValueError('Clock alphabet range')
        self.clock = 0; self.cursor = 0; self.order = array('H')
        self.inverse = array('H', [65535])*(self.size*self.length)
        self.index = {t: n for n, t in enumerate(self.alphabet)}
        self.work = Counter()

    def advance(self):
        clock = self.clock+1
        if clock >= self.length: raise ValueError('Outside compiled clocks')
        previous = self.order; out = array('H'); count = self.take('<H')[0]
        for _ in range(count):
            opcode = self.take('<B')[0]
            if opcode == 0:
                start, length = self.take('<HH')
                if not length or start+length > len(previous): raise ValueError('Invalid COPY interval')
                out.extend(previous[start:start+length]); self.work['copyOperations'] += 1
                self.work['peakCopySlicePayloadBytes'] = max(self.work['peakCopySlicePayloadBytes'], 2*length)
            elif opcode == 1:
                token = self.take('<H')[0]
                if token >= self.size: raise ValueError('Invalid INSERT token')
                out.append(token*self.length+clock-1); self.work['insertOperations'] += 1
            else: raise ValueError('Invalid bytecode opcode')
        if len(out) != self.size*clock: raise ValueError('Order coverage length')
        seen = bytearray(self.size*self.length)
        for rank, identity in enumerate(out):
            if identity >= len(seen) or identity % self.length >= clock or seen[identity]: raise ValueError('Invalid/duplicate order identity')
            seen[identity] = 1; self.inverse[identity] = rank
        self.work.update({'clockUpdates': 1, 'expandedIdentities': len(out), 'inverseRankWrites': len(out),
                          'validationFlagBytesInitialized': len(seen)})
        self.work['peakCoexistingOrderPayloadBytes'] = max(self.work['peakCoexistingOrderPayloadBytes'], 2*(len(previous)+len(out)))
        self.order = out; self.clock = clock
        if clock == self.length-1 and self.cursor != len(self.data): raise ValueError('Trailing bytecode')

    def take(self, fmt):
        width = struct.calcsize(fmt)
        if self.cursor+width > len(self.data): raise ValueError('Truncated bytecode')
        result = struct.unpack_from(fmt, self.data, self.cursor); self.cursor += width
        return result

    def serve(self, packet, method):
        validate_packet(packet)
        if method not in METHODS or self.clock != 0: raise ValueError('Fresh clock/method required')
        history = []; unknown = []; rows = []
        for clock, (token, active) in enumerate(zip(packet['ids'], packet['mask'])):
            tick = clock+(method == 'clockPlus1')
            while self.clock < min(tick, self.length-1): self.advance()
            if active:
                history.append((token, clock)); self.work['admissions'] += 1
                if token not in self.index: unknown.append(clock); self.work['unknownAdmissions'] += 1
            reason = ('clock-outside-domain' if clock >= self.length else 'active-query-outside-padding-contract' if active else
                      'wrong-query-token' if token != self.meta['paddingQueryID'] else 'unknown-admitted-token' if unknown else
                      'control-clock-outside-domain' if method == 'clockPlus1' and tick >= self.length else None)
            if reason:
                rows.append({'clock': clock, 'prefixCount': len(history), 'status': reason, 'addresses': None, 'unknownPositions': unknown.copy()})
                continue
            top = []; self.work['supportedQueries'] += 1
            if method == 'recent': addresses = [p for _, p in reversed(history[-4:])]
            else:
                for t, position in history:
                    rank = self.inverse[self.index[t]*self.length+position]; self.work['rankLookups'] += 1
                    if method == 'reverseRanks': rank = self.size*tick-1-rank
                    insert = 0
                    for old, _ in top:
                        self.work['integerRankComparisons'] += 1
                        if rank < old: break
                        insert += 1
                    if insert < 4: top.insert(insert, (rank, position)); top = top[:4]
                addresses = [p for _, p in top]
            rows.append({'clock': clock, 'readClock': tick, 'prefixCount': len(history), 'status': 'supported',
                         'addresses': addresses+[-1]*(4-len(addresses)), 'unknownPositions': []})
        return rows, dict(self.work)


class RankTransitions:
    """COPY intervals act as piecewise integer translations on existing ranks."""
    def __init__(self, metadata, code):
        self.meta = metadata; self.length = metadata['length']; self.size = len(metadata['alphabet'])
        OrderClock(metadata, b'')  # Reuse interface validation, not its full-order evaluator.
        self.starts = array('H'); self.deltas = array('h'); self.offsets = array('I', [0, 0])
        self.ends = array('H', [0]); self.inserts = array('H', [65535])*(self.length*self.size)
        self.work = Counter(); cursor = 0
        def take(fmt):
            nonlocal cursor
            width = struct.calcsize(fmt)
            if cursor+width > len(code): raise ValueError('Truncated rank-transition bytecode')
            value = struct.unpack_from(fmt, code, cursor); cursor += width; return value
        for clock in range(1, self.length):
            target = 0; intervals = []; inserted = 0
            for _ in range(take('<H')[0]):
                opcode = take('<B')[0]
                if opcode == 0:
                    start, length = take('<HH')
                    if not length: raise ValueError('Empty rank interval')
                    intervals.append((start, length, target-start)); target += length
                elif opcode == 1:
                    token = take('<H')[0]
                    if token >= self.size or self.inserts[clock*self.size+token] != 65535: raise ValueError('Invalid/duplicate INSERT')
                    self.inserts[clock*self.size+token] = target; target += 1; inserted += 1
                else: raise ValueError('Invalid rank-transition opcode')
            expected = 0
            for start, length, delta in sorted(intervals):
                if start != expected or not -32768 <= delta <= 32767: raise ValueError('COPY source intervals not a partition')
                self.starts.append(start); self.deltas.append(delta); expected += length
            if expected != self.size*(clock-1) or target != self.size*clock or inserted != self.size: raise ValueError('Incomplete clock translation')
            self.ends.append(expected); self.offsets.append(len(self.starts))
            self.work.update({'clockIndexSortCalls': 1, 'copyIntervals': len(intervals), 'insertRanks': inserted})
        if cursor != len(code): raise ValueError('Trailing rank-transition bytes')

    def payload(self):
        return {k+'Bytes': len(v)*v.itemsize for k, v in [('starts', self.starts), ('deltas', self.deltas),
                ('clockOffsets', self.offsets), ('endSentinels', self.ends), ('insertRanks', self.inserts)]}

    def transport(self, clock, rank, work):
        if not 0 <= rank < self.ends[clock]: raise ValueError('Old rank outside interval partition')
        lo, hi = self.offsets[clock], self.offsets[clock+1]
        while lo < hi:
            mid = (lo+hi)//2; work['intervalStartComparisons'] += 1
            if self.starts[mid] <= rank: lo = mid+1
            else: hi = mid
        work['rankTransportBinarySearches'] += 1
        return rank+self.deltas[lo-1]


class RankHistory:
    def __init__(self, transitions):
        self.transitions = transitions; self.clock = 0; self.history = []; self.work = Counter()
        self.index = {t: n for n, t in enumerate(transitions.meta['alphabet'])}

    def advance(self):
        clock = self.clock+1; t = self.transitions
        if clock >= t.length: raise ValueError('Outside sparse rank clocks')
        for item in self.history:
            position, token, rank = item
            if token is None: continue  # Coverage remains unsupported, never an omitted legal key.
            if position == clock-1:
                rank = t.inserts[clock*t.size+token]; self.work['insertRankLookups'] += 1
            else:
                if rank is None: raise ValueError('Uninitialized old rank')
                rank = t.transport(clock, rank, self.work)
            item[2] = rank; self.work['rankWrites'] += 1
        self.clock = clock; self.work['clockUpdates'] += 1

    def serve(self, packet, method):
        validate_packet(packet)
        if method not in METHODS or self.clock != 0 or self.history: raise ValueError('Fresh sparse history required')
        t = self.transitions; unknown = []; rows = []
        for clock, (token, active) in enumerate(zip(packet['ids'], packet['mask'])):
            tick = clock+(method == 'clockPlus1')
            while self.clock < min(tick, t.length-1): self.advance()
            if active:
                token_index = self.index.get(token); rank = None
                if token_index is None: unknown.append(clock); self.work['unknownAdmissions'] += 1
                elif self.clock == clock+1:
                    rank = t.inserts[self.clock*t.size+token_index]; self.work['insertRankLookups'] += 1; self.work['rankWrites'] += 1
                self.history.append([clock, token_index, rank]); self.work['admissions'] += 1
            reason = ('clock-outside-domain' if clock >= t.length else 'active-query-outside-padding-contract' if active else
                      'wrong-query-token' if token != t.meta['paddingQueryID'] else 'unknown-admitted-token' if unknown else
                      'control-clock-outside-domain' if method == 'clockPlus1' and tick >= t.length else None)
            if reason:
                rows.append({'clock': clock, 'prefixCount': len(self.history), 'status': reason, 'addresses': None, 'unknownPositions': unknown.copy()}); continue
            self.work['supportedQueries'] += 1; top = []
            if method == 'recent':
                addresses = [p for p, _, _ in reversed(self.history[-4:])]; self.work['recentPositionReads'] += len(addresses)
            else:
                for position, _, rank in self.history:
                    if rank is None: raise ValueError('Query before rank admission')
                    self.work['rankReads'] += 1
                    if method == 'reverseRanks': rank = t.size*tick-1-rank
                    insert = 0
                    for old, _ in top:
                        self.work['integerRankComparisons'] += 1
                        if rank < old: break
                        insert += 1
                    if insert < 4: top.insert(insert, (rank, position)); top = top[:4]
                addresses = [p for _, p in top]
            rows.append({'clock': clock, 'readClock': tick, 'prefixCount': len(self.history), 'status': 'supported',
                         'addresses': addresses+[-1]*(4-len(addresses)), 'unknownPositions': []})
        return rows, dict(self.work)


def execute(spec_path, output):
    started = time.perf_counter(); cpu = time.process_time(); spec = read(spec_path); out = Path(output); out.mkdir(exist_ok=False)
    opens, denied, imports = isolate(spec, out)
    tick = time.perf_counter(); metadata = read(spec['metadata'])
    if sha(spec['metadata']) != spec['metadataSHA256'] or sha(spec['bytecode']) != metadata['compressedSHA256']: raise ValueError('Compiled bytes differ')
    with gzip.open(spec['bytecode'], 'rb') as f: bytecode = f.read(metadata['rawBytes']+1)
    if len(bytecode) != metadata['rawBytes'] or hashlib.sha256(bytecode).hexdigest() != metadata['rawSHA256']: raise ValueError('Raw bytecode differs')
    indexed_at = time.perf_counter(); transitions = RankTransitions(metadata, bytecode); loaded = time.perf_counter()
    raw_size = len(bytecode); size = len(metadata['alphabet']); length = metadata['length']; proof = OrderClock(metadata, bytecode); hashes = []
    sparse_proof = Counter(); previous = array('H')
    for clock in range(metadata['length']):
        if clock:
            previous = proof.order; proof.advance()
            for old_rank, identity in enumerate(previous):
                predicted = transitions.transport(clock, old_rank, sparse_proof)
                sparse_proof['checkedOrdinals'] += 1
                sparse_proof['mismatchedOrdinals'] += predicted != proof.inverse[identity]
            for token in range(size):
                sparse_proof['checkedOrdinals'] += 1
                sparse_proof['mismatchedOrdinals'] += transitions.inserts[clock*size+token] != proof.inverse[token*length+clock-1]
        raw = little_bytes(proof.inverse[t*proof.length+j] for t in range(proof.size) for j in range(clock))
        hashes.append({'clock': clock, 'ordinals': proof.size*clock, 'rankSHA256': hashlib.sha256(raw).hexdigest()})
    save(out/'decoded-clocks.json', {'clocks': hashes, 'work': dict(proof.work), 'sparseRankProof': dict(sparse_proof),
                                   'scope': 'Fresh full-order replay plus every sparse rank/INSERT checked against it; no expected hashes supplied.'})
    proof_seconds = time.perf_counter()-loaded; proof_work = dict(proof.work); del proof, bytecode, previous, raw
    measurements = []
    for case in spec['cases']:
        packet = read(case['tokens'])
        for method in METHODS:
            begin = time.perf_counter(); machine = RankHistory(transitions); rows, work = machine.serve(packet, method); computed = time.perf_counter()
            path = out/(case['id']+'-'+method+'.jsonl.gz'); write_rows(path, rows); digest = sha(path)
            measurements.append({'case': case['id'], 'method': method, 'computeSeconds': computed-begin,
                                 'serializationAndHashSeconds': time.perf_counter()-computed, 'sha256': digest, 'bytes': path.stat().st_size,
                                 'work': work, 'historyLogicalInt64Bytes': 24*len(machine.history), 'path': path.name})
        print('ORDER_CLOCK_SERVED', case['id'], flush=True)
    save(out/'receipt.json', {'schema': 'tce-order-clock-serve-v1', 'rows': measurements, 'loadDecompressAndHashSeconds': indexed_at-tick,
         'loadAndIndexSeconds': loaded-tick,
         'rankIndexBuildSeconds': loaded-indexed_at, 'indexWork': dict(transitions.work), 'indexPayload': transitions.payload(),
         'completeReplayProofSeconds': proof_seconds, 'completeReplayProofWork': proof_work, 'wallSecondsBeforeReceipt': time.perf_counter()-started,
         'completeSparseRankProof': dict(sparse_proof),
         'CPUSeconds': time.process_time()-cpu, 'highWaterRSSKiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'successfulReadPaths': sorted(opens), 'fileCanaryDenials': denied, 'blockedImportCanaries': imports,
         'bytecodeRawBytes': raw_size, 'verificationOnlyMaximumOrderPayloadBytes': 2*size*(length-1),
         'verificationOnlyInversePayloadBytes': 2*size*length, 'verificationOnlyValidationFlagsPayloadBytes': size*length,
         'scope': 'Serving advances only admitted ranks through packed interval translations. Full order/inverse/raw bytecode released after separate proof; index stays. Logical arrays omit Python/program/outputs/temporaries. High-waterRSS may include inherited pre-exec process history.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('spec'); p.add_argument('output'); args = p.parse_args(); execute(args.spec, args.output)
