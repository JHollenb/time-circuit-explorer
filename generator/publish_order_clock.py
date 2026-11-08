"""Independent public integer-bytecode/custody audit and exact publication."""
from array import array
from collections import Counter
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import sys


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition, message):
    if not condition: raise ValueError(message)
def inventory(path): return {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(Path(path).iterdir()) if p.is_file()}
def save(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2); f.write('\n')


def audit(root, mechanics):
    root = Path(root).resolve(); result = root/'result'; repo = Path(__file__).resolve().parent.parent
    request = read(root/'request.json'); col = read(root/'collection.json'); job = read(root/'scheduler-job.json')
    identity = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != 'requestSHA256'}, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(identity == request['requestSHA256'], 'Canonical request')
    actual = inventory(result); require(actual == col['inventory']['files'], 'Complete collection byte identity')
    require(read(result/'result-files.json') == {k: v for k, v in actual.items() if k != 'result-files.json'}, 'Worker manifest')
    require(job['state'] == 'succeeded' and job['result']['returncode'] == 0 and job['config']['requestSHA256'] == identity, 'Successful job/request')
    require(job['admission'] and job['payload_custody'] == col['custody'] and col['custody']['required'], 'Guarded admission')
    custody = col['custody']
    require(len({(custody[k]['sha256'], custody[k]['size_bytes']) for k in ['declared', 'sealed', 'executed']}) == 1, 'Executed custody')
    for name, digest in request['sourceHashes'].items(): require(sha(root/'payload'/name) == digest, 'Staged source/input '+name)
    for name in request['ownedSources']: require(sha(repo/'generator'/name) == request['sourceHashes'][name], 'Executed source '+name)
    require(sha(repo/'docs/plans/order-clock-recurrence.md') == request['sourceHashes']['plan.md'], 'Frozen plan')
    test = read(mechanics); require(test['returncode'] == 0 and 'Ran 8 tests in 0.050s' in test['stderr'], 'Targeted prior mechanics')
    for name, digest in test['sourceHashes'].items(): require(request['sourceHashes'][name] == digest, 'Test source')
    seal = read(result/'prediction-seal.json')
    require(seal['requestSHA256'] == identity, 'Serving seal request')
    for name, item in seal['files'].items(): require(actual['prediction-'+name] == item, 'Fresh prediction seal '+name)
    spec = read(result/'serve-spec.json'); receipt = read(result/'prediction-receipt.json')
    require(receipt['blockedImportCanaries'] == ['numpy', 'torch'] and receipt['fileCanaryDenials'] == spec['forbiddenPaths'], 'Serving exclusions')
    served = Path(job['result']['diagnostics']['cwd'])/'served'
    output_reads = {str(served/x['path']) for x in receipt['rows']}
    require(set(receipt['successfulReadPaths']) <= set(spec['readAllow']) | output_reads, 'Input/exact-output read allowance')
    parent = repo/'docs/results/source-order-program'; basis = read(result/'basis.json')
    require(sha(parent/'publication.json') == basis['parentPublicationSHA256'], 'Parent publication binding')
    parent_ledger = read(parent/'compile-compiler.json')['clocks']; proof = read(result/'prediction-decoded-clocks.json')
    meta = read(result/'compile-metadata.json'); code = gzip.decompress((result/'compile-program.bin.gz').read_bytes())
    require(len(code) == meta['rawBytes'] and hashlib.sha256(code).hexdigest() == meta['rawSHA256'], 'Raw bytecode identity')
    require(sha(result/'compile-program.bin.gz') == meta['compressedSHA256'] and (result/'compile-program.bin.gz').stat().st_size == meta['compressedBytes'], 'Compressed bytecode identity')
    size, length = len(meta['alphabet']), meta['length']; cursor = 0; previous = array('H'); histogram = Counter(); counts = Counter(); decoded_hashes = []
    def take(fmt):
        nonlocal cursor
        value = struct.unpack_from(fmt, code, cursor); cursor += struct.calcsize(fmt); return value
    for clock in range(length):
        if clock:
            out = array('H'); last_copy_end = None
            for _ in range(take('<H')[0]):
                opcode = take('<B')[0]
                if opcode == 0:
                    start, count = take('<HH')
                    require(count > 0 and start+count <= len(previous), 'Independent COPY bounds')
                    require(last_copy_end != start, 'Canonical maximal adjacent COPY')
                    out.extend(previous[start:start+count]); histogram[count] += 1; counts['copies'] += 1; counts['copiedIdentities'] += count
                    last_copy_end = start+count
                elif opcode == 1:
                    token = take('<H')[0]; require(token < size, 'Independent INSERT bounds')
                    out.append(token*length+clock-1); counts['inserts'] += 1; last_copy_end = None
                else: raise ValueError('Independent opcode')
            previous = out
        require(len(previous) == size*clock and len(set(previous)) == len(previous), 'Independent complete order')
        inverse = {identity: rank for rank, identity in enumerate(previous)}
        packed = array('H', (inverse[t*length+j] for t in range(size) for j in range(clock)))
        if sys.byteorder != 'little': packed.byteswap()
        digest = hashlib.sha256(packed.tobytes()).hexdigest()
        require(digest == parent_ledger[clock]['rankSHA256'] == proof['clocks'][clock]['rankSHA256'], 'Independent full ordinal replay')
        decoded_hashes.append(digest); counts['ordinals'] += len(packed)
    require(cursor == len(code), 'No trailing bytecode')
    require(len(code) == 5*counts['copies']+3*counts['inserts']+2*(length-1), 'Fixed byte formula')
    compiler = read(result/'compile-compiler.json')
    require({str(k): v for k, v in sorted(histogram.items())} == compiler['copyLengthHistogram'], 'Copy histogram')
    require(counts['copies'] == compiler['work']['copyOperations'] and counts['inserts'] == compiler['work']['insertOperations'], 'Operation census')
    require(sum(receipt['indexPayload'].values()) == 4*counts['copies']+4*(length+1)+2*length+2*size*length, 'Resident index byte census')
    report = read(result/'report.json'); method_rows = 0; different = 0; panels = []
    for case in basis['cases']:
        for method in ['primary', 'clockPlus1', 'reverseRanks', 'recent']:
            name = case['id']+'-'+method+'.jsonl.gz'
            with gzip.open(result/('prediction-'+name), 'rt') as f: actual_rows = [json.loads(x) for x in f]
            with gzip.open(parent/('prediction-'+name), 'rt') as f: old_rows = [json.loads(x) for x in f]
            require(len(actual_rows) == len(old_rows) == 512, 'Complete prediction rows')
            clocks = [i for i, (a, b) in enumerate(zip(actual_rows, old_rows)) if a != b]; different += len(clocks); method_rows += len(actual_rows)
            panels.append({'case': case['id'], 'method': method, 'rows': 512, 'mismatchedClocks': clocks, 'supported': sum(r['status'] == 'supported' for r in actual_rows)})
    require(panels == report['panels'] and different == report['servingRowMismatches'] and method_rows == report['servingMethodRows'], 'All method panels')
    sparse = proof['sparseRankProof']
    require(sparse == receipt['completeSparseRankProof'] and sparse['checkedOrdinals'] == counts['ordinals'], 'Complete sparse proof accounting')
    parent_meta = read(parent/'compile-program.json')
    return {'schema': 'tce-order-clock-publication-audit-v1', 'jobID': job['job_id'], 'requestSHA256': identity,
        'rawFiles': len(actual), 'rawBytes': sum(v['bytes'] for v in actual.values()), 'clockHashes': len(decoded_hashes),
        'independentlyDecodedOrdinals': counts['ordinals'], 'methodRows': method_rows, 'methodMismatches': different,
        'sparseProof': sparse, 'copies': counts['copies'], 'inserts': counts['inserts'], 'copyLengthHistogram': dict(sorted(histogram.items())),
        'indexPayloadBytes': sum(receipt['indexPayload'].values()), 'rawBytecodeRatioTo95': len(code)/parent_meta['rawBytes'],
        'compressedBytecodeRatioTo95': meta['compressedBytes']/parent_meta['compressedBytes'],
        'indexRatioTo95RawRanks': sum(receipt['indexPayload'].values())/parent_meta['rawBytes'],
        'singletonCopyFraction': histogram[1]/counts['copies'], 'meanCopiedRunLength': counts['copiedIdentities']/counts['copies'],
        'auditorSHA256': sha(__file__), 'scope': 'Independent stdlib COPY/INSERT reconstruction and source-bound integer/scalar/custody verification. No source scores/models/neural arrays or scientific quality gate. Exact encoding remains the same finite comparator; no new prompt/image coverage.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('mechanics'); p.add_argument('destination'); args = p.parse_args()
    result = audit(args.root, args.mechanics); out = Path(args.destination); out.mkdir(parents=True, exist_ok=False)
    for source in sorted((Path(args.root)/'result').iterdir()):
        with (out/source.name).open('xb') as f: f.write(source.read_bytes())
    for name in ['collection.json', 'scheduler-job.json']:
        with (out/name).open('xb') as f: f.write((Path(args.root)/name).read_bytes())
    with (out/'mechanics.json').open('xb') as f: f.write(Path(args.mechanics).read_bytes())
    save(out/'scalar-audit.json', result)
    save(out/'publication.json', {'schema': 'tce-order-clock-publication-v1', 'files': inventory(out), 'auditorSHA256': sha(__file__),
                                'rawFiles': result['rawFiles'], 'rawBytes': result['rawBytes'], 'scope': result['scope']})
    print(json.dumps(result, indent=2))
