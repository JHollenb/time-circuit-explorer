"""V2 publication audit: tolerate calibrated cross-Python float-sum differences."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
from token_history_study import OWNED, identity, read, save, sha, verify


def stream(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def audit(root):
    root = Path(root).resolve()
    result = root/'result'
    request = read(root/'request.json')
    verify(root/'payload', request)
    for name in OWNED:
        if sha(Path(__file__).resolve().parent/name) != request['sourceHashes'][name]:
            raise ValueError('Executed source changed')
    col = read(root/'collection.json')
    job = read(root/'scheduler-job.json')
    actual = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in result.iterdir()}
    if actual != col['inventory']['files'] or job['state'] != 'succeeded' or job['result']['returncode'] != 0:
        raise ValueError('Collection/terminal source identity differs')
    if job['config']['requestSHA256'] != request['requestSHA256'] or job['payload_custody'] != col['custody'] or not job['admission']:
        raise ValueError('Job request/custody/admission join differs')
    custody = col['custody']
    if custody['required'] is not True or len({(custody[k]['sha256'], custody[k]['size_bytes']) for k in ['declared', 'sealed', 'executed']}) != 1:
        raise ValueError('Executed payload custody differs')
    declared = read(result/'result-files.json')
    if declared != {k: v for k, v in actual.items() if k != 'result-files.json'}:
        raise ValueError('Worker result manifest differs')
    seal = read(result/'prediction-seal.json')
    execution = read(result/'execution.json')
    if execution['predictionSealSHA256'] != sha(result/'prediction-seal.json') or seal['programSHA256'] != sha(result/'compile-program.json'):
        raise ValueError('Program/prediction seal join differs')
    for name, item in seal['files'].items():
        if actual['prediction-'+name] != item:
            raise ValueError('Post-seal prediction bytes differ')
    for action in ['compile', 'serve']:
        receipt = execution[action]['receipt']
        if receipt['loadedNumericalRuntimeRoots'] or receipt['blockedImportCanaries'] != ['numpy', 'torch']:
            raise ValueError('Program-child numerical exclusions differ')
        spec = read(result/(action+'-spec.json'))
        if receipt['fileCanaryDenials'] != spec['forbiddenPaths']:
            raise ValueError('Program-child file canaries differ')
    accesses = read(result/'member-access.json')['rows']
    transfer_sources = {c['selected']['path'] for c in read(result/'inputs.json')['cases'] if c['split'] == 'transfer'}
    if any(a['phase'] != 'post-seal-score' for a in accesses if a['path'] in transfer_sources):
        raise ValueError('Transfer label interpreted before seal')
    if any(a['member'] not in {'ids', 'mask', 'positions', 'supportPositions', 'selectedWeights', 'prefixCounts'} for a in accesses):
        raise ValueError('Undeclared numerical member')
    cases = ['A', 'B', '00', '10', '01', '11']
    tokens = {c: read(result/('input-'+c+'-tokens.json')) for c in cases}
    refs = {c: stream(result/('reference-'+c+'.jsonl.gz')) for c in cases}
    methods = ['primary', 'positionOnly', 'earliest', 'recent', 'headRotated', 'tokenNull']
    predictions = {c: {m: stream(result/('prediction-'+c+'-'+m+'.jsonl.gz')) for m in methods} for c in cases}
    sums = defaultdict(lambda: {'rows': 0, 'ordered': 0, 'set': 0, 'top1': 0, 'overlap': 0., 'oldTop4WeightCoverage': 0., 'missingDenominatorRows': 0, 'fallbackSlots': 0, 'exhaustedSlots': 0})
    count = 0
    max_row_difference = 0.0
    query_keys = set()
    with gzip.open(result/'observations.jsonl.gz', 'rt') as f:
        for line in f:
            row = json.loads(line)
            case, p, head = row['case'], row['position'], row['head']
            key = case, p, head
            if key in query_keys:
                raise ValueError('Duplicate observed address')
            query_keys.add(key)
            target = refs[case][p]['supports'][head]
            weights = refs[case][p]['oldTop4Weights'][head]
            if target != row['support'] or weights != row['oldTop4Weights']:
                raise ValueError('Observed reference join differs')
            expected_group = ('active' if tokens[case]['mask'][p] else 'padding')+('Short' if sum(tokens[case]['mask'][:p+1]) <= 4 else 'Nontrivial')
            if row['group'] != expected_group:
                raise ValueError('Active/padding/triviality denominator differs')
            groups = [row['group']]
            if row['unseenQueryToken'] and row['group'] == 'activeNontrivial':
                groups.append('unseenQueryActiveNontrivial')
            for method, observed in row['methods'].items():
                packet = predictions[case][method][p]
                predicted = packet['addresses'][head]
                if predicted != observed['addresses'] or packet['rawSelectors'][head] != observed['raw'] or packet['reasons'][head] != observed['reasons']:
                    raise ValueError('Observed prediction/trace join differs')
                target_set = {a for a in target if a >= 0}
                predicted_set = {a for a in predicted if a >= 0}
                expected = {'ordered': int(predicted == target), 'set': int(predicted_set == target_set), 'top1': int(predicted[0] == target[0]),
                    'overlap': len(target_set & predicted_set)/len(target_set) if target_set else None,
                    'oldTop4WeightCoverage': sum(w for a, w in zip(target, weights) if a >= 0 and a in predicted_set) if target_set else None}
                for name, value in expected.items():
                    if isinstance(value, float):
                        max_row_difference = max(max_row_difference, abs(observed[name]-value))
                        if not math.isclose(observed[name], value, rel_tol=1e-15, abs_tol=1e-15):
                            raise ValueError('Independent scalar float metric differs beyond calibrated sum tolerance')
                    elif observed[name] != value:
                        raise ValueError('Independent scalar indicator differs')
                for group in groups:
                    for c, h in [('*', -1), (case, -1), (case, head)]:
                        a = sums[(row['split'], c, h, group, method)]
                        a['rows'] += 1
                        for name, value in expected.items():
                            if value is not None:
                                a[name] += value
                        a['missingDenominatorRows'] += expected['overlap'] is None
                        a['fallbackSlots'] += sum(why in [1, 2] for why in observed['reasons'])
                        a['exhaustedSlots'] += observed['reasons'].count(3)
            count += 1
    panels = read(result/'panels.json')
    if len(panels) != len(sums) or count != 98304 or len(query_keys) != count:
        raise ValueError('Complete row/panel count differs')
    for panel in panels:
        observed = sums[tuple(panel[k] for k in ['split', 'case', 'head', 'group', 'method'])]
        for name, value in observed.items():
            if not math.isclose(value, panel[name], rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError('Independent panel accumulation differs')
        for name, value in panel['rates'].items():
            denominator = observed['rows']-(observed['missingDenominatorRows'] if name in ['overlap', 'oldTop4WeightCoverage'] else 0)
            expected = observed[name]/denominator if denominator else None
            if value is None or expected is None:
                if value != expected:raise ValueError('Panel empty denominator differs')
            elif not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Panel rate differs beyond accumulated sum tolerance')
    pairs = []
    for a, b in [('A', 'B'), ('00', '10'), ('00', '01'), ('10', '11'), ('01', '11')]:
        changed = [i for i in range(512) if tokens[a]['ids'][i] != tokens[b]['ids'][i]]
        pair = {'pair': [a, b], 'changedInputPositions': changed, 'panels': []}
        for group in ['activeNontrivial', 'paddingNontrivial']:
            indices = [i for i in range(512) if sum(tokens[a]['mask'][:i+1]) > 4 and bool(tokens[a]['mask'][i]) == (group == 'activeNontrivial')]
            addresses = [(i, h) for i in indices for h in range(32)]
            pair['panels'].append({'group': group, 'rows': len(addresses),
                'referenceOrderedChanges': sum(refs[a][i]['supports'][h] != refs[b][i]['supports'][h] for i, h in addresses),
                'referenceSetChanges': sum(set(refs[a][i]['supports'][h]) != set(refs[b][i]['supports'][h]) for i, h in addresses),
                'programOrderedChanges': sum(predictions[a]['primary'][i]['addresses'][h] != predictions[b]['primary'][i]['addresses'][h] for i, h in addresses)})
        for panel in pair['panels']:
            active = panel['group'] == 'activeNontrivial'
            rows = [(i, h) for i in range(512) for h in range(32) if sum(tokens[a]['mask'][:i+1]) > 4 and bool(tokens[a]['mask'][i]) == active and tokens[a]['ids'][i] == tokens[b]['ids'][i]]
            violations = 0
            for i, h in rows:
                left = [p for p in refs[a][i]['supports'][h] if p >= 0 and p not in changed]
                right = [p for p in refs[b][i]['supports'][h] if p >= 0 and p not in changed]
                shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
                violations += shorter != longer[:len(shorter)]
            panel['unchangedQueryRows'] = len(rows)
            panel['unchangedQueryReferenceOrderedChanges'] = sum(refs[a][i]['supports'][h] != refs[b][i]['supports'][h] for i, h in rows)
            panel['unchangedKeySpinePrefixViolations'] = violations
        pairs.append(pair)
    return {'schema': 'tce-token-history-scalar-audit-v1', 'jobID': job['job_id'], 'requestSHA256': request['requestSHA256'],
        'sourceHashes': {n: request['sourceHashes'][n] for n in OWNED}, 'resultFiles': len(actual), 'resultBytes': sum(x['bytes'] for x in actual.values()),
        'floatSumAudit': {'maximumAbsoluteRowFloatDifference': max_row_difference, 'rowTolerance': 1e-15, 'panelRelativeTolerance': 1e-12, 'panelAbsoluteTolerance': 1e-10, 'scope': 'V1 exact equality rejected a 1.1102230246251565e-16 difference. Full589824row localPython3.10 versus worker3.14 sum scan found maximum2.220446049250313e-16. Four nonnegative normalized weights; source outputs remain unchanged. Indicator counts remain exact.'},
        'verifiedHeadQueryRows': count, 'verifiedMethodRows': count*6, 'verifiedPanels': len(panels), 'predictionFiles': len(seal['files']),
        'pairedAddressSensitivity': pairs, 'pairedSensitivityScope': 'Post-hoc public integer/scalar contrast; no refit, new model/array/consumer/image or independent new prompts.',
        'mechanics_status': 'complete', 'terminal_status': 'unassessed',
        'scope': 'Byte/source/custody/member-access/seal/child-exclusion joins and independent scalar formulas/accumulations. Trusted-code audit, not adversarial isolation or native image certification.'}


def publish(root, destination, mechanics):
    root = Path(root).resolve()
    result = audit(root)
    mechanics = Path(mechanics)
    tests = read(mechanics)
    if tests['returncode'] != 0 or tests['sourceHashes'] != result['sourceHashes']:
        raise ValueError('Targeted test receipt/source join differs')
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for source in (root/'result').iterdir():
        shutil.copyfile(source, destination/source.name)
    for name in ['collection.json', 'scheduler-job.json']:
        shutil.copyfile(root/name, destination/name)
    shutil.copyfile(mechanics, destination/'mechanics.json')
    result['publisherSHA256'] = sha(__file__)
    save(destination/'scalar-audit.json', result)
    inventory = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in destination.iterdir()}
    save(destination/'publication.json', {'schema': 'tce-token-history-publication-v1', 'sourceRoot': str(root),
        'files': inventory, 'bytes': sum(a['bytes'] for a in inventory.values()), 'scalarAuditSHA256': sha(destination/'scalar-audit.json'),
        'scope': 'All68original result files copied byte-exact; collection/scheduler/test receipts and independent scalar audit added. Full historical NPZs stay on Beast.'})
    return {'publicRoot': str(destination), 'filesBeforePublicationReceipt': len(inventory), 'bytesBeforePublicationReceipt': sum(a['bytes'] for a in inventory.values()), 'scalarAuditSHA256': sha(destination/'scalar-audit.json')}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root')
    p.add_argument('destination')
    p.add_argument('--mechanics', required=True)
    a = p.parse_args()
    print(json.dumps(publish(a.root, a.destination, a.mechanics)))
