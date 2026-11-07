"""Describe published scalar moments; target-assisted capacity is not a forecast."""
import json
from pathlib import Path
from encoder_atoms import read, save, sha


def execute():
    root = Path(__file__).resolve().parent.parent;p = root / 'docs/results/diagonal-norm-moments'
    out = root / 'docs/results/diagonal-norm-moments-evidence';out.mkdir(parents=True, exist_ok=False)
    report = read(p / 'report.json');rows = report['rows'];pooled = report['pooled']
    if len(rows) != 24 or {x['head'] for x in rows} != set(range(24)):
        raise ValueError('Complete unique head summary required')
    def fractions(x):
        total = x['trueMeanEnergy'] + x['trueVariationEnergy']
        return {'sharedMeanFractionOfTotalEnergy': x['trueMeanEnergy'] / total,
            'sharedMeanFractionOfOmittedEnergy': x['meanOmittedEnergy'] / x['totalOmittedEnergy'],
            'centeredFractionOfOmittedEnergy': x['variationOmittedEnergy'] / x['totalOmittedEnergy'],
            'targetMeanOrthogonalFractionOfTotalEnergy': x['targetMeanOrthogonalEnergy'] / total}
    values = [{'head': x['head'], **fractions(x)} for x in rows]
    result = {'schema': 'tce-diagonal-moments-scalars-v1', 'reportSHA256': sha(p / 'report.json'),
        'pooledFractions': fractions(pooled), 'headFractions': values,
        'headSharedMeanOmissionFractionRange': [min(x['sharedMeanFractionOfOmittedEnergy'] for x in values), max(x['sharedMeanFractionOfOmittedEnergy'] for x in values)],
        'headOrthogonalFractionRange': [min(x['targetMeanOrthogonalFractionOfTotalEnergy'] for x in values), max(x['targetMeanOrthogonalFractionOfTotalEnergy'] for x in values)],
        'newArrays': len(report['retainedMoments']), 'newArrayBytes': sum(x['bytes'] for x in report['retainedMoments'].values()),
        'scope': 'Arithmetic on published scalar head/pooled summaries only; orthogonal capacity uses the correct saved Q mean direction, not an available serving producer.'}
    save(out / 'scalar-summary.json', result)
    with (out / 'observations.jsonl').open('x') as stream:
        for x, f in zip(rows, values):
            stream.write(json.dumps({**x, **f, 'panel': 'post-hoc-head-moments', 'sourceReportSHA256': sha(p / 'report.json')}, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
