"""Summarize already-public dependent query scalars; never load vectors."""
import argparse
import gzip
import json
import math
from pathlib import Path


def summarize(root):
    root = Path(root)
    report = json.loads((root/'report.json').read_text())
    with gzip.open(root/'rows.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    summary = {}
    for panel in ['retainedRaw', 'replayRaw', 'admittedBF16']:
        ordered = {k: sorted(row[panel]['squaredNorms'][k]**.5 for row in rows) for k in ['payload', 'allocation', 'interaction', 'delta']}
        summary[panel] = {k: {'min': v[0], 'median': v[len(v)//2], 'max': v[-1], 'nonzeroRows': sum(x != 0 for x in v)} for k, v in ordered.items()}
        summary[panel]['allocationNormGreaterRows'] = sum(row[panel]['squaredNorms']['allocation'] > row[panel]['squaredNorms']['payload'] for row in rows)
        summary[panel]['payloadNormGreaterRows'] = sum(row[panel]['squaredNorms']['payload'] > row[panel]['squaredNorms']['allocation'] for row in rows)
        summary[panel]['idealRemaindersRelativeToDelta'] = {k: math.sqrt(report['panels'][panel]['summedSquaredNorms'][k]/report['panels'][panel]['summedSquaredNorms']['delta']) for k in ['idealPayloadRemainder', 'idealInteractionRemainder', 'closureRemainder']}
    summary['editedAddressMass'] = {k: {'min': min(x[k] for x in rows), 'median': sorted(x[k] for x in rows)[len(rows)//2], 'max': max(x[k] for x in rows)} for k in ['pA8', 'pB8']}
    summary['bf16'] = {k: sum(x[k] for x in rows) for k in ['admittedEndpointWordChanges', 'admittedPayloadAtAllocationAWordChanges', 'admittedPayloadAtAllocationBWordChanges']}
    summary['scope'] = 'Public scalar-row summary only; no vector, image or scientific array read locally. Min/median/max are dependent-query diagnostics.'
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    output = Path(args.root)/'row-summary.json'; value = summarize(args.root)
    if args.verify:
        if json.loads(output.read_text()) != value:
            raise ValueError('Public scalar summary differs')
        print('PUBLIC_SCALAR_SUMMARY_MATCHES')
    else:
        with output.open('x') as stream:
            json.dump(value, stream, indent=2)
