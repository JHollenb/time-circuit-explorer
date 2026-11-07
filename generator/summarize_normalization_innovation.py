"""Published scalar-only two-receiver trends and measured resource account."""
import gzip
import json
from pathlib import Path
import shutil
import statistics
from encoder_atoms import read, save, sha


def execute():
    from publish_normalization_innovation import audit
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/normalization-innovation'
    out = root / 'docs/results/normalization-innovation-evidence'
    out.mkdir(exist_ok=False, parents=True)
    save(out / 'mechanics.json', audit(p))
    report = read(p / 'report.json')
    rows = [json.loads(line) for line in gzip.decompress((p / 'queries.jsonl.gz').read_bytes()).splitlines()]
    trends = []
    for receiver in ['R0', 'R1']:
        for stage in ['fused', 'product', 'mlpWrite', 'gatedMLP', 'imageExit']:
            for scope in ['all', 'source-nonlocal']:
                chosen = [x for x in rows if x['receiver'] == receiver and x['stage'] == stage and (scope == 'all' or x['sourceNonlocal'])]
                primary = [x['response']['shared-normalization-innovation'] for x in chosen]
                controls = [key for key in chosen[0]['response'] if key != 'shared-normalization-innovation']
                trends.append({'receiver': receiver, 'stage': stage, 'scope': scope, 'dependentQueries': len(chosen), 'controls': {key: {'lowerError': sum(x['response']['shared-normalization-innovation']['rms'] < x['response'][key]['rms'] for x in chosen), 'higherError': sum(x['response']['shared-normalization-innovation']['rms'] > x['response'][key]['rms'] for x in chosen), 'equalError': sum(x['response']['shared-normalization-innovation']['rms'] == x['response'][key]['rms'] for x in chosen)} for key in controls}, 'negativeResponseCosines': sum(x['cosine'] is not None and x['cosine'] < 0 for x in primary), 'medianRelativeResponseError': statistics.median(x['relativeRms'] for x in primary if x['relativeRms'] is not None)})
    a = read(p / 'primary.json')
    fields = [*a['shared'].values(), *[x for group in a['cells'].values() for x in group.values()], *[x for group in read(p / 'response-fields.json').values() for x in group.values()]]
    unique = {x['path']: x for x in fields}
    v = read(p / 'retained-array-verification.json')
    save(out / 'scalar-summary.json', {'schema': 'tce-normalization-innovation-scalars-v1', 'reportSHA256': sha(p / 'report.json'), 'primarySealSHA256': sha(p / 'primary-seal.json'), 'newArrays': len(unique), 'newArrayBytes': sum(x['bytes'] for x in unique.values()), 'fullByteFiles': len(v['files']), 'fullByteCount': v['totalBytesVerified'], 'clockViewBytesReadAgain': 12288, 'queryTrends': trends, 'localFreeBytesObserved': shutil.disk_usage(root).free, 'scope': 'Scalar/query summaries only; dependent outcomes, no quality gate or independent replication count. Large arrays remain near compute; shared-volume free observation is not cleanup attribution.'})
    j = read(root / 'artifacts/normalization-innovation-v1/scheduler-job.json')
    save(out / 'scheduler.json', {'schema': 'tce-normalization-innovation-scheduler-v1', 'jobID': j['job_id'], 'state': j['state'], 'reservation': j['reservation'], 'result': j['result'], 'executedPayload': j['payload_custody']['executed'], 'futureSizing': 'Use the observed two-CPUFP64/two-CUDA-down process-separated geometry and I/O for matching future runs. Do not infer speedup from MAC count, copy the prior494MB peak to this new working set, or rerun a valid experiment to change reservations.'})
    return {'arrays': len(unique), 'bytes': sum(x['bytes'] for x in unique.values()), 'queryGroups': len(trends)}


if __name__ == '__main__':
    print(execute())
