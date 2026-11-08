"""Audit and publish cycle95 integers/scalars; never load neural operands."""
import argparse
from array import array
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    return {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(Path(root).iterdir()) if p.is_file()}


def rows(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-15)
    return a == b


def metrics(predicted, target, weights=None):
    a, b = {p for p in predicted if p >= 0}, {p for p in target if p >= 0}
    out = {'ordered': int(predicted == target), 'set': int(a == b), 'top1': int(predicted[0] == target[0]),
           'overlap': len(a & b)/len(b) if b else None}
    if weights is not None:
        out['oldTop4WeightCoverage'] = sum(w for p, w in zip(target, weights) if p >= 0 and p in a) if b else None
    return out


def audit(root, mechanics):
    root = Path(root).resolve(); result = root/'result'
    request = read(root/'request.json'); col = read(root/'collection.json'); job = read(root/'scheduler-job.json')
    encoded = json.dumps({k: v for k, v in request.items() if k != 'requestSHA256'}, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    require(hashlib.sha256(encoded).hexdigest() == request['requestSHA256'], 'Request identity')
    actual = inventory(result)
    require(actual == col['inventory']['files'] and sum(x['bytes'] for x in actual.values()) == col['inventory']['bytes'], 'Collected bytes')
    require(read(result/'result-files.json') == {k: v for k, v in actual.items() if k != 'result-files.json'}, 'Worker manifest')
    require(read(result/'request.json') == request and job['config']['requestSHA256'] == request['requestSHA256'], 'Job/request join')
    require(job['state'] == 'succeeded' and job['result']['returncode'] == 0 and job['payload_custody'] == col['custody'], 'Successful custody')
    c = col['custody']
    require(c['required'] and len({(c[k]['sha256'], c[k]['size_bytes']) for k in ['declared', 'sealed', 'executed']}) == 1, 'Payload custody')
    for name, digest in request['sourceHashes'].items():
        require(sha(root/'payload'/name) == digest, 'Frozen payload '+name)
    for name in request['ownedSources']:
        require(sha(Path(__file__).parent/name) == request['sourceHashes'][name], 'Executed source '+name)
    for src, dest in [('source-ordering-program-feasibility.md', 'frozen-plan.md'), ('source-ordering-implementation.md', 'implementation.md')]:
        require(sha(Path(__file__).parent.parent/'docs/plans'/src) == request['sourceHashes'][dest], 'Frozen plan '+src)
    tests = read(mechanics)
    require(tests['returncode'] == 0 and 'Ran 8 tests in 0.046s' in tests['stderr'], 'Targeted test receipt')
    for name, digest in tests['sourceHashes'].items():
        require(digest == request['sourceHashes'][name], 'Test/source join')
    meta = read(result/'compile-program.json'); seal = read(result/'prediction-seal.json'); execution = read(result/'execution.json')
    require(seal['programSHA256'] == sha(result/'compile-program.json') and seal['compressedSHA256'] == sha(result/'compile-order.u16.gz'), 'Program seal')
    require(execution['predictionSealSHA256'] == sha(result/'prediction-seal.json') and seal['requestSHA256'] == request['requestSHA256'], 'Prediction seal')
    for name, item in seal['files'].items():
        require(actual['prediction-'+name] == item, 'Sealed predictions')
    receipt = read(result/'prediction-receipt.json'); spec = read(result/'serve-spec.json')
    require(receipt['blockedImportCanaries'] == ['numpy', 'torch'] and receipt['fileCanaryDenials'] == spec['forbiddenPaths'], 'Serving exclusions')
    require(set(receipt['successfulReadPaths']) <= set(spec['readAllow']), 'Read allowlist')
    raw = gzip.decompress((result/'compile-order.u16.gz').read_bytes())
    require(len(raw) == meta['rawBytes'] and hashlib.sha256(raw).hexdigest() == meta['rawSHA256'], 'Raw ranks')
    ranks = array('H'); ranks.frombytes(raw)
    if sys.byteorder != 'little': ranks.byteswap()
    alphabet = meta['alphabet']; size = len(alphabet); length = meta['length']; index = {t: n for n, t in enumerate(alphabet)}
    require(len(ranks) == size*length*(length-1)//2 == meta['ordinalCount'], 'Triangular census')
    ledger = read(result/'compile-compiler.json')['clocks']
    for clock in range(length):
        lo = size*clock*(clock-1)//2; chunk = ranks[lo:lo+size*clock]
        require(sorted(chunk) == list(range(size*clock)), 'Clock rank permutation')
        require(hashlib.sha256(raw[2*lo:2*(lo+size*clock)]).hexdigest() == ledger[clock]['rankSHA256'], 'Clock ledger')
    methods = ['primary', 'clockPlus1', 'reverseRanks', 'recent']
    packets = {c['id']: read(result/('input-'+c['id']+'-tokens.json')) for c in spec['cases']}
    scopes = {c['id']: c['scope'] for c in spec['cases']}
    predictions = {(c, m): rows(result/('prediction-'+c+'-'+m+'.jsonl.gz')) for c in packets for m in methods}
    old = {c: rows(result/(c+'-old-reference.jsonl.gz')) for c in ['A', 'B']}
    observations = rows(result/'observations.jsonl.gz'); observed = {(r['case'], r['clock']): r for r in observations}
    require(len(observed) == len(observations) == len(packets)*512, 'Complete unique scalar stream')
    sums = defaultdict(lambda: {'rows': 0, 'supported': 0, 'orderedMatches': 0, 'setMatches': 0, 'top1Matches': 0,
                              'overlapSum': 0., 'oldRows': 0, 'oldOrderedMatches': 0, 'oldWeightCoverageSum': 0.})
    statuses = Counter(); mismatch = 0; supported = Counter(); empty = 0
    for case, packet in packets.items():
        history = []; unknown = []
        require(packet['positions'] == list(range(512)), 'Absolute clock packet')
        for clock, (token, active) in enumerate(zip(packet['ids'], packet['mask'])):
            if active:
                history.append((token, clock))
                if token not in index: unknown.append(clock)
            obs = observed[case, clock]; group = 'active' if active else 'paddingShort' if len(history) <= 4 else 'paddingNontrivial'
            require((obs['scope'], obs['mask'], obs['prefixCount'], obs['group']) == (scopes[case], active, len(history), group), 'Scalar denominator')
            for method in methods:
                pred = predictions[case, method][clock]
                status = ('clock-outside-domain' if clock >= length else 'active-query-outside-padding-contract' if active else
                          'wrong-query-token' if token != meta['paddingQueryID'] else 'unknown-admitted-token' if unknown else
                          'control-clock-outside-domain' if method == 'clockPlus1' and clock+1 >= length else 'supported')
                require((pred['status'], pred['clock'], pred['prefixCount'], pred['unknownPositions']) == (status, clock, len(history), unknown), 'Own-history status')
                metric = old_metric = None
                if status == 'supported':
                    tick = clock+(method == 'clockPlus1')
                    def key(item):
                        t, p = item
                        rank = ranks[size*tick*(tick-1)//2+index[t]*tick+p]
                        return -rank if method == 'reverseRanks' else rank
                    addresses = [p for _, p in (history[-4:][::-1] if method == 'recent' else sorted(history, key=key)[:4])]
                    addresses += [-1]*(4-len(addresses))
                    require(pred['addresses'] == addresses and pred['readClock'] == tick, 'Independent integer replay')
                    require(obs['declaredReference'] is not None, 'Direct reference present')
                    metric = metrics(addresses, obs['declaredReference'])
                    if method == 'primary':
                        supported[group] += 1; mismatch += 1-metric['ordered']; empty += not history
                    if case in old:
                        target = old[case][clock]['supports'][0]; weights = old[case][clock]['oldTop4Weights'][0]
                        require(obs['oldSupport'] == target and obs['oldWeights'] == weights, 'Old reference join')
                        old_metric = metrics(addresses, target, weights)
                else:
                    require(pred['addresses'] is None, 'Unsupported address refusal')
                supplied = obs['methods'][method]
                require({k: v for k, v in supplied.items() if k not in ['declaredOrder', 'oldReader']} == pred, 'Observation prediction join')
                for a, b in [(metric, supplied['declaredOrder']), (old_metric, supplied['oldReader'])]:
                    require((a is None and b is None) or (a is not None and b is not None and set(a) == set(b) and all(equal(v, b[k]) for k, v in a.items())), 'Per-row metrics')
                statuses[scopes[case], case, method, status] += 1
                for c in ['*', case]:
                    out = sums[scopes[case], c, group, method]; out['rows'] += 1
                    if metric is not None:
                        out['supported'] += 1; out['orderedMatches'] += metric['ordered']; out['setMatches'] += metric['set']; out['top1Matches'] += metric['top1']; out['overlapSum'] += metric['overlap'] or 0
                    if old_metric is not None:
                        out['oldRows'] += 1; out['oldOrderedMatches'] += old_metric['ordered']; out['oldWeightCoverageSum'] += old_metric['oldTop4WeightCoverage'] or 0
    panels = read(result/'panels.json')
    require(len(panels) == len(sums), 'Complete panels')
    for panel in panels:
        out = sums[tuple(panel[k] for k in ['scope', 'case', 'group', 'method'])]; n = out['supported']; old_n = out['oldRows']
        out.update({'declaredOrderedRate': out['orderedMatches']/n if n else None, 'declaredSetRate': out['setMatches']/n if n else None,
                    'declaredOverlapRate': out['overlapSum']/n if n else None, 'oldOrderedRate': out['oldOrderedMatches']/old_n if old_n else None,
                    'oldTop4WeightCoverage': out['oldWeightCoverageSum']/old_n if old_n else None})
        require(all(equal(v, panel[k]) for k, v in out.items()), 'Panel scalar recomputation')
    expected_coverage = [{'scope': a, 'case': b, 'method': c, 'status': d, 'rows': n} for (a, b, c, d), n in sorted(statuses.items())]
    require(read(result/'coverage.json') == expected_coverage, 'Coverage census')
    report = read(result/'report.json')
    require(report['headQueryRows'] == len(observed) and report['methodRows'] == len(observed)*4 and report['primaryDeclaredOrderMismatches'] == mismatch, 'Reported observation counts')
    require(report['pooledPanels'] == [p for p in panels if p['case'] == '*'], 'Report panels')
    a = predictions['A', 'primary']; b = predictions['B', 'primary']
    changed = sum(x['addresses'] != y['addresses'] for x, y in zip(a, b) if x['status'] == y['status'] == 'supported')
    synthetic = read(result/'synthetic-edit-response.json')
    require(synthetic['changedReferenceRows'] == sum(x['referenceChanged'] for x in synthetic['rows']), 'Synthetic contrast census')
    return {'schema': 'tce-source-order-publication-audit-v1', 'jobID': job['job_id'], 'requestSHA256': request['requestSHA256'],
            'rawFiles': len(actual), 'rawBytes': sum(v['bytes'] for v in actual.values()), 'clockPermutations': length,
            'ordinals': len(ranks), 'headQueryRows': len(observed), 'methodRows': len(observed)*4, 'panels': len(panels),
            'supportedPrimaryByGroup': dict(supported), 'primaryDeclaredOrderMismatches': mismatch, 'supportedEmptyPrefixRows': empty,
            'historicalEdit': {'supportedRows': 477, 'changedAddresses': changed, 'distinctSupportsA': len({tuple(x['addresses']) for x in a if x['status'] == 'supported'}),
                               'distinctSupportsB': len({tuple(x['addresses']) for x in b if x['status'] == 'supported'})},
            'syntheticEditChangedReferenceRows': synthetic['changedReferenceRows'], 'sourceHashes': {n: request['sourceHashes'][n] for n in request['ownedSources']},
            'scope': 'Independent stdlib integer subset replay and scalar/custody audit. No source scores recalculated. Direct score references share the compiler score helper; this audit does not independently establish neural numeric parity.',
            'scalarTolerance': {'absolute': 1e-15, 'relative': 1e-14, 'countsAndIndicatorsExact': True},
            'emptyPrefixPanelLimitation': 'Generic panels count absent overlap as zero; current production has no supported empty prefix. Preserve before future reuse.',
            'auditorSHA256': sha(__file__)}


def save_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('mechanics'); p.add_argument('destination'); args = p.parse_args()
    result = audit(args.root, args.mechanics)
    dest = Path(args.destination); dest.mkdir(parents=True, exist_ok=False)
    for source in sorted((Path(args.root)/'result').iterdir()):
        with (dest/source.name).open('xb') as f: f.write(source.read_bytes())
    for name in ['collection.json', 'scheduler-job.json']:
        with (dest/name).open('xb') as f: f.write((Path(args.root)/name).read_bytes())
    with (dest/'mechanics.json').open('xb') as f: f.write(Path(args.mechanics).read_bytes())
    save_new(dest/'scalar-audit.json', result)
    save_new(dest/'publication.json', {'schema': 'tce-source-order-publication-v1', 'files': inventory(dest), 'rawFiles': result['rawFiles'],
                                     'rawBytes': result['rawBytes'], 'auditorSHA256': sha(__file__), 'scope': result['scope']})
    print(json.dumps(result, indent=2))
