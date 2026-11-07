"""Published scalar/query summaries and resource observations only."""
import gzip
import json
from pathlib import Path
import shutil
import statistics
from encoder_atoms import read, save, sha


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/product-transport'
    out = root / 'docs/results/product-transport-evidence'
    out.mkdir(exist_ok=False, parents=True)
    from publish_product_transport import audit
    mechanics = audit(p)
    if read(p / 'primary.json')['fields']['residualCommit']['payloadSHA256'] != read(p / 'targets.json')['residualR1B']['payloadSHA256']:
        raise ValueError('Native residual payload bits differ')
    mechanics['nativeResidualPayloadExact'] = True
    save(out / 'mechanics.json', mechanics)
    report = read(p / 'report.json')
    rows = [json.loads(line) for line in gzip.decompress((p / 'queries.jsonl.gz').read_bytes()).splitlines()]
    trends = []
    for stage in ['product', 'mlpWrite', 'gatedMLP', 'imageExit']:
        for scope in ['all', 'source-nonlocal']:
            selected = [x for x in rows if x['stage'] == stage and (scope == 'all' or x['sourceNonlocal'])]
            candidates = [x['response']['recipient-baseline-product'] for x in selected]
            trends.append({'stage': stage, 'scope': scope, 'dependentQueries': len(selected),
                'lowerErrorThanSourceTransfer': sum(x['response']['recipient-baseline-product']['rms'] < x['response']['raw-source-transfer']['rms'] for x in selected),
                'higherErrorThanSourceTransfer': sum(x['response']['recipient-baseline-product']['rms'] > x['response']['raw-source-transfer']['rms'] for x in selected),
                'equalErrorToSourceTransfer': sum(x['response']['recipient-baseline-product']['rms'] == x['response']['raw-source-transfer']['rms'] for x in selected),
                'lowerErrorThanZero': sum(x['response']['recipient-baseline-product']['rms'] < x['response']['zero-response']['rms'] for x in selected),
                'negativeResponseCosines': sum(x['cosine'] is not None and x['cosine'] < 0 for x in candidates),
                'medianRelativeResponseError': statistics.median(x['relativeRms'] for x in candidates if x['relativeRms'] is not None)})
    fields = [*read(p / 'primary.json')['fields'].values(), *[x for group in read(p / 'response-fields.json').values() for x in group.values()]]
    unique = {x['path']: x for x in fields}
    v = read(p / 'retained-array-verification.json')
    save(out / 'scalar-summary.json', {'schema': 'tce-product-transport-scalar-summary-v1', 'reportSHA256': sha(p / 'report.json'), 'primarySealSHA256': sha(p / 'primary-seal.json'), 'newArrays': len(unique), 'newArrayBytes': sum(x['bytes'] for x in unique.values()), 'fullByteFiles': len(v['files']), 'fullByteCount': v['totalBytesVerified'], 'clockViewBytesReadAgain': 6144, 'queryTrends': trends, 'localFreeBytesObserved': shutil.disk_usage(root).free, 'scope': 'Scalar-only publication summary. Query outcomes are dependent and not independent replications. Historical preparation and original learned work paid; storage observation is not cleanup attribution.'})
    job = read(root / 'artifacts/product-transport-v1/scheduler-job.json')
    save(out / 'scheduler.json', {'schema': 'tce-product-transport-scheduler-v1', 'jobID': job['job_id'], 'state': job['state'], 'reservation': job['reservation'], 'result': job['result'], 'executedPayload': job['payload_custody']['executed'], 'rightSizeRecommendation': None, 'futureSizing': 'Measured1663.9MBRSS/494MBVRAM/77.34s. Original reservation4096MBRAM1024MBVRAM CPU4disk4GB. No matching recommendation field in this raw job record. Future sizing should use this one-down/process-separated geometry and actual I/O, not full-model or parent12affine sizing; no rerun for reservations.'})
    return {'arrays': len(unique), 'bytes': sum(x['bytes'] for x in unique.values()), 'queryGroups': len(trends)}


if __name__ == '__main__':
    print(execute())
