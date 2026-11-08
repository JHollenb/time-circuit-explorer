"""Finite input-addressed score lookup and selected normalization, stdlib only.

This still computes scalar attention. Offline learned products are not free.
"""
from array import array
import argparse
import hashlib
import math
from pathlib import Path
import platform
import resource
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_order_program import OrderProgram, offset
from token_history_process import NoEvaluation, isolate, read, save, sha, write_rows

NORMALIZER = 'stdlib-max-subtract-math-exp-math-fsum-divide-selected-order-v1'


def normalize(values):
    if not values:
        return []
    if any(not math.isfinite(x) for x in values):
        raise ValueError('Nonfinite selected score')
    maximum = max(values)
    masses = [math.exp(x-maximum) for x in values]
    total = math.fsum(masses)
    return [x/total for x in masses]


def runtime_profile():
    # Runtime identity only, before the clean-input guard; no model/source arrays.
    libraries = []
    maps = Path('/proc/self/maps')
    if maps.exists():
        paths = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                        if '/libm.so' in line and line.split()[-1].startswith('/')})
        libraries = [{'path': p, 'sha256': sha(p)} for p in paths]
    return {'python': sys.version, 'platform': platform.platform(), 'libc': platform.libc_ver(),
            'byteorder': sys.byteorder, 'mathExtension': getattr(math, '__file__', None),
            'mathExtensionSHA256': sha(math.__file__) if getattr(math, '__file__', None) else None,
            'mappedLibm': libraries, 'normalizer': NORMALIZER,
            'scope': 'Actual Python/math-extension/mapped Linux libm bytes; no cross-platform finite-equivalence claim.'}


class MassProgram:
    def __init__(self, order, metadata, values):
        if metadata['schema'] != 'tce-compiled-selected-mass-v1' or metadata['normalizer'] != NORMALIZER:
            raise ValueError('Wrong mass program interface')
        for key in ['head', 'paddingQueryID', 'alphabet', 'length', 'ordinalCount', 'scoreContract']:
            if metadata[key] != order.meta[key]:
                raise ValueError('Score/rank domain mismatch: '+key)
        if len(values) != metadata['ordinalCount'] or metadata['scoreBytes'] != 8*len(values):
            raise ValueError('Score array extent mismatch')
        self.order, self.metadata, self.values = order, metadata, values

    @classmethod
    def load(cls, metadata_path, rank_metadata_path, ranks_path, scores_path):
        metadata = read(metadata_path)
        if sha(rank_metadata_path) != metadata['rankMetadataSHA256']:
            raise ValueError('Rank metadata identity mismatch')
        order = OrderProgram.load(rank_metadata_path, ranks_path)
        if Path(scores_path).stat().st_size != metadata['scoreBytes'] or sha(scores_path) != metadata['scoreSHA256']:
            raise ValueError('Score bytes mismatch')
        values = array('d')
        if values.itemsize != 8:
            raise ValueError('Eight-byte double required')
        with Path(scores_path).open('rb') as stream:
            values.fromfile(stream, metadata['ordinalCount'])
        if sys.byteorder != 'little':
            values.byteswap()
        return cls(order, metadata, values)

    def serve(self, packet):
        begin = time.perf_counter()
        rows, routing = self.order.serve(packet)
        routed = time.perf_counter()
        work = {'scoreLookups': 0, 'finiteScoreChecks': 0, 'maximumComparisons': 0,
                'subtractions': 0, 'expCalls': 0, 'fsumCalls': 0, 'divisions': 0,
                'supportedEmptyQueries': 0, 'supportedNonemptyQueries': 0, 'unsupportedQueries': 0}
        for row in rows:
            if row['status'] != 'supported':
                row.update(selectedScores=None, weights=None, paddedWeights=None)
                work['unsupportedQueries'] += 1
                continue
            addresses = [j for j in row['addresses'] if j >= 0]
            values = [self.values[offset(row['clock'], self.order.index[packet['ids'][j]], j,
                                         len(self.order.alphabet))] for j in addresses]
            weights = normalize(values)
            row.update(selectedScores=values, weights=weights, paddedWeights=weights+[0.]*(4-len(weights)))
            n = len(values)
            for key in ['scoreLookups', 'finiteScoreChecks', 'subtractions', 'expCalls', 'divisions']:
                work[key] += n
            work['maximumComparisons'] += max(n-1, 0)
            work['fsumCalls'] += int(n > 0)
            work['supportedEmptyQueries' if n == 0 else 'supportedNonemptyQueries'] += 1
        finished = time.perf_counter()
        return rows, {'routing': routing, 'massWork': work, 'routeSeconds': routed-begin,
                      'massAndPacketSeconds': finished-routed, 'completeComputeSeconds': finished-begin,
                      'scope': 'Timing includes Python dispatch/packet formation. Work counts named operations, not exhaustive instructions.'}


def execute(spec_path, output):
    start, cpu = time.perf_counter(), time.process_time()
    initial_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == 'darwin' else 1024)
    spec = read(spec_path)
    profile = runtime_profile()
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    opens, denied, imports = isolate(spec, output)
    if sha(spec['metadata']) != spec['metadataSHA256']:
        raise ValueError('Sealed mass metadata mismatch')
    tick = time.perf_counter()
    program = MassProgram.load(spec['metadata'], spec['rankMetadata'], spec['ranks'], spec['scores'])
    load_seconds = time.perf_counter()-tick
    receipts = []
    for case in spec['cases']:
        tick = time.perf_counter()
        if sha(case['tokens']) != case['tokensSHA256']:
            raise ValueError('Clean token packet mismatch')
        packet = read(case['tokens'])
        authenticated = time.perf_counter()
        rows, work = program.serve(packet)
        computed = time.perf_counter()
        path = output/(case['id']+'.jsonl.gz')
        write_rows(path, rows)
        receipts.append({'case': case['id'], 'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path),
                         'inputAuthenticationSeconds': authenticated-tick,
                         'serializationAndHashSeconds': time.perf_counter()-computed, **work})
    native_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save(output/'receipt.json', {'schema': 'tce-compiled-mass-serve-v1', 'cases': receipts,
         'runtimeProfile': profile, 'metadataSHA256': spec['metadataSHA256'],
         'rankRawBytes': program.order.meta['rawBytes'], 'scoreRawBytes': program.metadata['scoreBytes'],
         'authenticationLoadAndDecodeSeconds': load_seconds, 'fileCanaryDenials': denied,
         'blockedImportCanaries': imports, 'successfulReadPathsAfterGuard': sorted(opens),
         'loadedNumericalRuntimeRoots': sorted(NoEvaluation.roots.intersection(sys.modules)),
         'CPUSeconds': time.process_time()-cpu, 'wallSecondsBeforeReceipt': time.perf_counter()-start,
         'initialHighWaterRSSBytes': initial_rss,
         'highWaterRSSBytes': native_rss*(1 if sys.platform == 'darwin' else 1024),
         'onlineQKCoordinateProducts': 0, 'scalarAttentionComputed': True,
         'scope': 'Trusted-source fresh process; runtime identity before guard, clean score/rank assets and IDs after. High-water RSS may include inherited pre-exec peak; baseline retained. No Q/K/source store/old weights. Not a full zero-neural reader or generator.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec')
    parser.add_argument('output')
    args = parser.parse_args()
    execute(args.spec, args.output)
