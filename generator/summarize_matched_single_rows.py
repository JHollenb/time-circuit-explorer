"""Describe heterogeneity in already published scalar anatomy rows."""
import gzip
import json
from pathlib import Path
import statistics
from encoder_atoms import save, sha


def execute():
    root = Path(__file__).resolve().parent.parent
    source = root / 'docs/results/matched-single-write/anatomy-observations.jsonl.gz'
    rows = [json.loads(x) for x in gzip.decompress(source.read_bytes()).splitlines()]
    result = {'schema': 'tce-matched-single-row-heterogeneity-v1',
              'sourceSHA256': sha(source), 'sourceRows': len(rows),
              'summarizerSHA256': sha(__file__), 'strata': {},
              'scope': 'Descriptive arithmetic on published scalar RMS/cosine rows only. Half/quarter counts describe distributions and have no gate authority. No model or tensor-array computation.'}
    for name in ['text', 'image']:
        part = [x for x in rows if x['stratum'] == name]
        ratios = [x['finiteRemainderRMS'] / x['actualDeltaRMS'] for x in part if x['actualDeltaRMS']]
        result['strata'][name] = {
            'rows': len(part), 'zeroDeltaRows': sum(x['actualDeltaRMS'] == 0 for x in part),
            'medianRemainderOverDelta': statistics.median(ratios),
            'maxRemainderOverDelta': max(ratios),
            'minOmissionDeltaCosine': min(x['omissionDeltaCosine'] for x in part),
            'rowsAboveHalf': sum(x > .5 for x in ratios),
            'rowsAboveQuarter': sum(x > .25 for x in ratios),
            'largestRemainderRows': [{**x, 'remainderOverDelta': x['finiteRemainderRMS'] / x['actualDeltaRMS']}
                for x in sorted(part, key=lambda y: y['finiteRemainderRMS'] / y['actualDeltaRMS'] if y['actualDeltaRMS'] else 0, reverse=True)[:3]]}
    save(root / 'docs/results/matched-single-write-evidence/row-heterogeneity.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(execute(), sort_keys=True))
