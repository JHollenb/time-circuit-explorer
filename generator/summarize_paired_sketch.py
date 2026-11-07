"""Summarize published paired-error scalars without activation loads."""
import json
import math
from pathlib import Path
from encoder_atoms import read, save, sha


def execute():
    root = Path(__file__).resolve().parent.parent;p = root / 'docs/results/paired-sketch-diagnostic'
    out = root / 'docs/results/paired-sketch-evidence';out.mkdir(parents=True, exist_ok=False)
    r = read(p / 'report.json');pooled = r['pooled'];head_errors = [x['anchoredNormResponseValidOnly']['relativeRms'] for x in r['heads'] if x['anchoredNormResponseValidOnly'] is not None]
    ideal = math.sqrt(pooled['idealSharedVarianceMean']) / pooled['energyResponse']['referenceRms']
    value = {'schema': 'tce-paired-sketch-scalars-v1', 'reportSHA256': sha(p / 'report.json'),
        'headNormResponsesBetterThanHold': sum(x < 1 for x in head_errors), 'headNormResponsesWorseThanHold': sum(x > 1 for x in head_errors),
        'definedHeadNormResponses': len(head_errors), 'headNormResponseRange': [min(head_errors), max(head_errors)] if head_errors else None,
        'energyErrorRMSCancellationFactor': math.sqrt(pooled['errorMoments']['qErrorMSE'] / pooled['errorMoments']['differenceErrorMSE']),
        'idealRootExpectedSquaredEnergyResponseError': ideal,
        'hypotheticalIndependentSketchRowsFor10PercentRootExpectedMSE': 16 * (ideal / .1)**2,
        'newArrays': len(r['retainedFields']), 'newArrayBytes': sum(x['bytes'] for x in r['retainedFields'].values()),
        'scope': 'Published scalar arithmetic. Ideal ensemble root expected MSE on this saved geometry; hypothetical row count is diagnostic algebra, not a selected/tested rank or a lower bound on other compression methods. Anchored norms use paid P energy; valid-only subsets explicit.'}
    save(out / 'scalar-summary.json', value)
    return value


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
