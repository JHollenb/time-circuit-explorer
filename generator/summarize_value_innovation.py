"""Compact dependent row trends and exact new-array footprint."""
import gzip
import json
from pathlib import Path
import shutil
import statistics
from encoder_atoms import read, save, sha
from publish_value_innovation import audit, fields


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/value-innovation'
    out = root / 'docs/results/value-innovation-evidence'
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'mechanics.json', audit(p))
    report = read(p / 'report.json')
    rows = [json.loads(line) for line in gzip.decompress((p / 'queries.jsonl.gz').read_bytes()).splitlines()]
    groups = []
    for stage in ['gatedCarrier', 'residualCommit', 'fused', 'product', 'mlpWrite', 'gatedMLP', 'imageExit']:
        for scope in ['all', 'source-nonlocal', 'event-near', 'event-far']:
            chosen = [r for r in rows if r['stage'] == stage and (scope == 'all' or scope == 'source-nonlocal' and r['sourceNonlocal'] or scope == 'event-near' and r['eventNear'] or scope == 'event-far' and not r['eventNear'])]
            relative = [r['response']['relativeRms'] for r in chosen if r['response']['relativeRms'] is not None]
            groups.append({'stage': stage, 'scope': scope, 'dependentQueries': len(chosen), 'nonzeroReferenceRows': len(relative), 'medianRelativeResponseError': statistics.median(relative) if relative else None,
                           'lowerErrorThanZero': sum(r['response']['rms'] < r['zeroResponse']['rms'] for r in chosen), 'higherErrorThanZero': sum(r['response']['rms'] > r['zeroResponse']['rms'] for r in chosen),
                           'equalErrorToZero': sum(r['response']['rms'] == r['zeroResponse']['rms'] for r in chosen), 'negativeResponseCosines': sum(r['response']['cosine'] is not None and r['response']['cosine'] < 0 for r in chosen)})
    provenance = read(p / 'provenance.json')
    base = provenance['schedulerResult']['diagnostics']['cwd'] + '/arrays/'
    arrays = {item['path']: item for name in ['primary.json', 'reference.json', 'response-fields.json'] for item in fields(read(p / name)) if item['path'].startswith(base)}
    verification = read(p / 'retained-array-verification.json')
    save(out / 'scalar-summary.json', {'schema': 'tce-value-innovation-scalars-v1', 'reportSHA256': sha(p / 'report.json'), 'primarySealSHA256': sha(p / 'primary-seal.json'),
         'referenceSealSHA256': sha(p / 'reference-seal.json'), 'newArrays': len(arrays), 'newArrayBytes': sum(x['bytes'] for x in arrays.values()),
         'fullByteFiles': len(verification['files']), 'fullByteCount': verification['totalBytesVerified'], 'VByteWitnessBytesReadAgain': verification['nativeVSwap']['payloadBytesReadAgain'],
         'queryTrends': groups, 'localFreeBytesObserved': shutil.disk_usage(root).free, 'scope': 'Saved scalar/query summaries only. Dependent row outcomes and weak/zero denominators retained. Shared-volume free space is an observation, not cleanup attribution.'})
    j = read(root / 'artifacts/value-innovation-v3/scheduler-job.json')
    save(out / 'scheduler.json', {'schema': 'tce-value-innovation-scheduler-v1', 'jobID': j['job_id'], 'state': j['state'], 'reservation': j['reservation'], 'result': j['result'], 'executedPayload': j['payload_custody']['executed']})
    return {'queryGroups': len(groups), 'arrays': len(arrays), 'bytes': sum(x['bytes'] for x in arrays.values()), 'inheritedA': report['inheritedA']}


if __name__ == '__main__':
    print(execute())
