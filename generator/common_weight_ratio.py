"""One stdlib diagnostic of public captured scalar weights; b stays supplied."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import time


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(root):
    root = Path(root); started = time.perf_counter()
    with gzip.open(root/'rows.jsonl.gz', 'rt') as stream:
        original = [json.loads(line) for line in stream]
    rows = []
    for row in original:
        pa = dict(zip(row['supportA'], row['weightsA'])); pb = dict(zip(row['supportB'], row['weightsB']))
        common = sorted((set(pa) & set(pb))-{8})
        ca = math.fsum(pa[j] for j in common); cb = math.fsum(pb[j] for j in common)
        kind = 'both' if 8 in pa and 8 in pb else 'onlyB' if 8 in pb else 'onlyA' if 8 in pa else 'neither'
        item = {'queryPosition': row['queryPosition'], 'kind': kind, 'commonUneditedAddresses': common,
                'commonMassA': ca, 'commonMassB': cb, 'suppliedEditedMassB': pb.get(8, 0.0),
                'droppedFromA': sorted(set(pa)-set(pb)), 'enteredInB': sorted(set(pb)-set(pa))}
        if ca <= 0 or cb <= 0:
            item['instrument_status'] = 'inapplicable-zero-common-mass'
        elif set(pb) != set(common) | ({8} if 8 in pb else set()):
            item['instrument_status'] = 'inapplicable-additional-changed-address'
        else:
            conditional_a = {j: pa[j]/ca for j in common}; conditional_b = {j: pb[j]/cb for j in common}
            item['conditionalWeightMaximumAbsoluteDifference'] = max(abs(conditional_a[j]-conditional_b[j]) for j in common)
            predicted = {j: ((1-pb[8])*conditional_a[j] if j != 8 else pb[8]) for j in pb} if 8 in pb else {j: pa[j] for j in pb}
            item['predictedWeightsBUsingSuppliedMass'] = [predicted[j] for j in row['supportB']]
            item['weightMaximumAbsoluteDifference'] = max(abs(predicted[j]-pb[j]) for j in pb)
            item['weightSumAbsoluteErrorB'] = abs(math.fsum(pb.values())-1)
            item['instrument_status'] = 'applicable'
        rows.append(item)
    applicable = [x for x in rows if x['instrument_status'] == 'applicable']
    return {'schema': 'tce-public-common-weight-ratio-v1', 'sourceRowsSHA256': sha(root/'rows.jsonl.gz'),
        'sourceReportSHA256': sha(root/'report.json'), 'sourceCodeSHA256': sha(__file__), 'rows': rows,
        'summary': {'totalRows': len(rows), 'applicableRows': len(applicable), 'types': dict(Counter(x['kind'] for x in rows)),
            'maximumConditionalWeightDifference': max((x['conditionalWeightMaximumAbsoluteDifference'] for x in applicable), default=None),
            'maximumBWeightReconstructionDifference': max((x['weightMaximumAbsoluteDifference'] for x in applicable), default=None),
            'maximumBWeightSumError': max((x['weightSumAbsoluteErrorB'] for x in applicable), default=None)},
        'secondsBeforeReport': time.perf_counter()-started, 'newModelCalls': 0, 'scientificArrayReads': 0,
        'mechanics_status': 'complete', 'instrument_status': 'all-rows-applicable' if len(applicable) == len(rows) else 'partial-applicability',
        'trend_status': 'Finite coefficient remainders retained without threshold', 'terminal_status': 'No mass or value producer supplied',
        'scope': 'Public scalar postprocessing only. C is joined/sorted by address for analysis, never substituted into source matmul. b and old conditional weights remain supplied. The reconstructed weights are an ideal diagnostic and never replace captured serving weights.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); parser.add_argument('output'); args = parser.parse_args()
    result = analyze(args.root)
    with Path(args.output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result['summary']))
