"""Independent stdlib scalar/custody publication; no neural-array hydration."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)
def inventory(root): return {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(Path(root).iterdir()) if p.is_file()}
def save(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def aggregate(rows):
    norms = {k: math.fsum(r['squaredNorms'][k] for r in rows) for k in rows[0]['squaredNorms']}
    dot = math.fsum(r['payloadDotAllocation'] for r in rows)
    return {'rows': len(rows), 'summedSquaredNorms': norms, 'payloadDotAllocation': dot,
        'twicePayloadDotAllocation': 2*dot, 'negativeCrossTermRows': sum(r['payloadDotAllocation'] < 0 for r in rows),
        'zeroDeltaRows': sum(r['squaredNorms']['delta'] == 0 for r in rows),
        'deltaSquaredMinusGram': norms['delta']-(norms['payload']+norms['allocation']+2*dot),
        'payloadNormOverDelta': math.sqrt(norms['payload']/norms['delta']) if norms['delta'] else None,
        'allocationNormOverDelta': math.sqrt(norms['allocation']/norms['delta']) if norms['delta'] else None,
        'scope': 'Dependent query-vector geometry; norm ratios are not semantic shares.'}


def remote_bytes(items):
    code = '''import hashlib,json,stat,zipfile,ast,struct
from pathlib import Path
items=json.loads(%r);rows=[]
for item in items:
 p=Path(item['path']);s=p.lstat()
 if not stat.S_ISREG(s.st_mode):raise ValueError('Nonregular retained file')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 if h.hexdigest()!=item['sha256'] or s.st_size!=item['bytes']:raise ValueError('Retained bytes differ')
 headers={}
 with zipfile.ZipFile(p) as z:
  for name in z.namelist():
   with z.open(name) as f:
    if f.read(6)!=b'\\x93NUMPY':raise ValueError('NPY magic')
    version=tuple(f.read(2));fmt='<H' if version==(1,0) else '<I';n=struct.unpack(fmt,f.read(struct.calcsize(fmt)))[0]
    header=ast.literal_eval(f.read(n).decode('latin1'))
   headers[name]={'dtype':header['descr'],'shape':header['shape'],'fortranOrder':header['fortran_order']}
 rows.append({'path':str(p),'sha256':h.hexdigest(),'bytes':s.st_size,'headers':headers})
print(json.dumps({'files':rows,'bytes':sum(x['bytes'] for x in rows),'scope':'Fresh full-file SHA and ZIP/NPY headers only; no numerical values interpreted or transported.'}))
''' % json.dumps(items)
    return json.loads(subprocess.run(['ssh', '-T', 'beast', 'python3', '-'], input=code, text=True, capture_output=True, check=True, timeout=60).stdout)


def audit(root, mechanics):
    root = Path(root).resolve(); result = root/'result'; repo = Path(__file__).resolve().parent.parent
    request = read(root/'request.json'); col = read(root/'collection.json'); job = read(root/'scheduler-job.json')
    identity = hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != 'requestSHA256'}, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(identity == request['requestSHA256'], 'Canonical request')
    actual = inventory(result); require(actual == col['inventory']['files'], 'Complete collection byte identity')
    require(read(result/'result-files.json') == {k: v for k, v in actual.items() if k != 'result-files.json'}, 'Worker manifest')
    require(job['state'] == 'succeeded' and job['result']['returncode'] == 0 and job['config']['requestSHA256'] == identity, 'Successful job/request')
    require(job['admission'] and job['payload_custody'] == col['custody'] and col['custody']['required'], 'Guarded admission')
    require(len({(col['custody'][k]['sha256'], col['custody'][k]['size_bytes']) for k in ['declared', 'sealed', 'executed']}) == 1, 'Executed payload custody')
    require(col['custody']['executed']['host'] == 'beast', 'Qualified host')
    for name, digest in request['sourceHashes'].items(): require(sha(root/'payload'/name) == digest, 'Staged source/input '+name)
    require(read(result/'request.json') == request, 'Collected request equals submitted request')
    for name in ['inputs.json', 'plan.md', 'implementation.md', 'parent-reader-seal.json', 'parent-request.json', 'parent-provenance.json']:
        require(sha(result/name) == request['sourceHashes'][name], 'Collected input equals staged input '+name)
    for name in ['first_read_payload.py', 'first_read_payload_study.py', 'test_first_read_payload.py', 'encoder_atoms.py']:
        require(sha(repo/'generator'/name) == request['sourceHashes'][name], 'Executed source '+name)
    for source, staged in [('first-read-route-payload.md', 'plan.md'), ('first-read-route-payload-implementation.md', 'implementation.md')]:
        require(sha(repo/'docs/plans'/source) == request['sourceHashes'][staged], 'Frozen plan '+source)
    test = read(mechanics); require(test['returncode'] == 0 and 'Ran 9 tests' in test['stderr'], 'Nine targeted mechanics')
    for name, digest in test['sourceHashes'].items(): require(request['sourceHashes'][name] == digest, 'Test source '+name)
    source = read(result/'inputs.json'); parent = read(result/'parent-reader-seal.json')
    require(sha(result/'parent-reader-seal.json') == source['readerSealSHA256'], 'Historical reader seal')
    require(source['originalPayloadCustody'] == read(result/'parent-provenance.json')['payloadCustody'], 'Historical custody join')
    expected = {(x['id'], 'lexical'): x for x in parent['lexicalRows']}
    expected.update({(x['id'], 'selected'): x for x in parent['packets']})
    for item in source['intake']['files']:
        require(all(item[k] == expected[item['id'], item['kind']][k] for k in ['path', 'sha256']), 'Source seal join')
    seal = read(result/'cross-seal.json'); arrays = read(result/'retained-arrays.json'); accesses = read(result/'member-access.json')['reads']
    require(seal['requestSHA256'] == identity and len(accesses) == 18, 'Cross request/member census')
    before = [x for x in accesses if x['phase'] == 'before-cross-seal']; after = [x for x in accesses if x['phase'] == 'after-cross-seal']
    require(len(before) == 14 and len(after) == 4 and before == seal['memberReadsBeforeSeal'], 'Numerical member access phases')
    require({x['member'] for x in after} == {'raw', 'rounded'}, 'Endpoint members only after seal')
    require(all(x['member'] not in {'raw', 'rounded', 'q', 'k', 'qpos', 'kpos', 'selectedLogits'} for x in before), 'No endpoint or query/key/logit member before seal')
    cross_file = arrays['files']['crosses.npz']
    require(all(cross_file[k] == seal[{'path': 'arrayPath', 'bytes': 'bytes', 'sha256': 'sha256'}[k]] for k in ['path', 'bytes', 'sha256']), 'Cross retained-byte seal')
    with gzip.open(result/'rows.jsonl.gz', 'rt') as f: rows = [json.loads(line) for line in f]
    require([x['queryPosition'] for x in rows] == list(range(35, 512)), 'Every dependent query exactly once')
    report = read(result/'report.json'); require(report['requestSHA256'] == identity, 'Report request')
    max_scalar_formula_remainder = 0.0
    for row in rows:
        for suffix in ['A', 'B']:
            support, weights = row['support'+suffix], row['weights'+suffix]
            require(len(support) == len(set(support)) == len(weights) == 4 and all(0 <= p < 35 for p in support), 'Four admitted addresses')
            require(all(math.isfinite(w) and w >= 0 for w in weights), 'Finite nonnegative weights')
            require(row['p'+suffix+'8'] == sum(w for p, w in zip(support, weights) if p == 8), 'Edited-address mass')
            require(row['editedAddressIn'+suffix] == (8 in support), 'Edited-address membership')
        require(row['orderedSupportChanged'] == (row['supportA'] != row['supportB']) and row['supportSymmetricDifference'] == len(set(row['supportA'])^set(row['supportB'])), 'Support change accounts')
        for panel in ['retainedRaw', 'replayRaw', 'admittedBF16']:
            values = row[panel]; norms = values['squaredNorms']; dot = values['payloadDotAllocation']
            require(all(math.isfinite(v) and v >= 0 for v in norms.values()), 'Finite nonnegative vector norms')
            require(values['twicePayloadDotAllocation'] == 2*dot, 'Signed cross term')
            delta = norms['delta']-(norms['payload']+norms['allocation']+2*dot)
            require(values['deltaSquaredMinusGram'] == delta, 'Scalar Gram expression')
            max_scalar_formula_remainder = max(max_scalar_formula_remainder, abs(delta))
    counts = {'total': 477, 'addressInBoth': sum(x['editedAddressInA'] and x['editedAddressInB'] for x in rows),
        'addressOnlyA': sum(x['editedAddressInA'] and not x['editedAddressInB'] for x in rows),
        'addressOnlyB': sum(x['editedAddressInB'] and not x['editedAddressInA'] for x in rows),
        'addressInNeither': sum(not x['editedAddressInA'] and not x['editedAddressInB'] for x in rows),
        'orderedSupportChanged': sum(x['orderedSupportChanged'] for x in rows)}
    require(counts == report['rowCounts'], 'All row counts')
    for panel in ['retainedRaw', 'replayRaw', 'admittedBF16']:
        require(aggregate([x[panel] for x in rows]) == report['panels'][panel], 'Independent scalar aggregate '+panel)
    require(report['crossReadProducts'] == report['endpointReplayProducts'] == 488448, 'Separate product accounts')
    for label in ['AA', 'AB', 'BA', 'BB']:
        require(report['readWork'][label]['queries'] == 477 and report['readWork'][label]['scalarValueProducts'] == 244224, 'Read geometry/work')
    require(report['newQKProducts'] == report['newSoftmaxEvaluations'] == report['newNeuralModelEncoderDenoiserVAEEvaluations'] == report['newImages'] == 0, 'No new neural/image work')
    require(report['guardCanaryDenials'] == ['torch', 'transformers', 'diffusers', 'checkpoint-open'], 'Actual exclusion canaries')
    require(report['qualification']['changedTokenPositions'] == [8] and set(report['qualification']['changedValuePositions']) <= {8}, 'Single-address applicability without a nonzero-effect gate')
    retained_items = list(arrays['files'].values())
    retained = remote_bytes(retained_items + source['intake']['files'])
    return {'schema': 'tce-first-read-payload-publication-audit-v1', 'jobID': job['job_id'], 'requestSHA256': identity,
        'rawFiles': len(actual), 'rawBytes': sum(x['bytes'] for x in actual.values()), 'scalarRows': len(rows), 'panels': 3,
        'memberReads': len(accesses), 'allScalarAggregatesMatch': True, 'maximumObservedScalarGramRemainder': max_scalar_formula_remainder,
        'retainedArrayBytes': sum(x['bytes'] for x in retained_items), 'freshRemoteByteVerification': retained,
        'auditorSHA256': sha(__file__), 'scope': 'Independent full-byte/custody/member-phase and all-row scalar aggregation checks. Vector calculations are the source-audited worker result, not independently rerun here. No numerical arrays transported/interpreted locally; no scientific quality gate.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('mechanics'); p.add_argument('destination'); args = p.parse_args()
    value = audit(args.root, args.mechanics); out = Path(args.destination); out.mkdir(parents=True, exist_ok=False)
    for source in sorted((Path(args.root)/'result').iterdir()):
        with (out/source.name).open('xb') as f: f.write(source.read_bytes())
    for name in ['collection.json', 'scheduler-job.json']:
        with (out/name).open('xb') as f: f.write((Path(args.root)/name).read_bytes())
    with (out/'mechanics.json').open('xb') as f: f.write(Path(args.mechanics).read_bytes())
    save(out/'scalar-audit.json', value)
    save(out/'publication.json', {'schema': 'tce-first-read-payload-publication-v1', 'files': inventory(out), 'auditorSHA256': sha(__file__),
        'rawFiles': value['rawFiles'], 'rawBytes': value['rawBytes'], 'scope': value['scope']})
    print(json.dumps({k: value[k] for k in ['jobID', 'rawFiles', 'rawBytes', 'scalarRows', 'retainedArrayBytes']}))
