"""Compact summaries from published observations and custody metadata only."""
import gzip
import json
from pathlib import Path
import shutil
import statistics
from encoder_atoms import read, save, sha
from projected_complement import fields
from publish_propagated_context import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/propagated-context'
    e = root / 'docs/results/propagated-context-evidence';e.mkdir(parents=True, exist_ok=False)
    save(e / 'mechanics.json', audit(p))
    r, receipt, provenance, intake = [read(p / n) for n in ['report.json', 'retained-array-verification.json', 'provenance.json', 'byte-inventory.json']]
    prefix = provenance['schedulerResult']['diagnostics']['cwd'] + '/arrays/'
    arrays = {x['path']: x for n in ['banks', 'primary', 'control'] for x in fields(read(p / (n + '.json'))) if x['path'].startswith(prefix)}
    rows = [json.loads(x) for x in gzip.decompress((p / 'observations.jsonl.gz').read_bytes()).splitlines()]
    heads = [x['difference']['relativeRms'] for x in rows if x['field'] == 'nextQ' and x['panel'] == 'query-head']
    finite = [x for x in heads if x is not None]
    summary = {'schema': 'tce-propagated-context-scalars-v1', 'reportSHA256': sha(p / 'report.json'),
        'newArrays': len(arrays), 'newArrayBytes': sum(x['bytes'] for x in arrays.values()),
        'fullByteFiles': len(receipt['files']), 'fullByteCount': receipt['totalBytesVerified'],
        'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']),
        'NPYPayloadBytesReadAgain': sum(x['payloadBytes'] for x in receipt['npyPayloadWitnesses'].values()),
        'spliceBytesReadAgain': receipt['spliceBytesReadAgain'],
        'sourceChangedRows': {n: len(x['changedRows']) for n, x in intake['rows'].items()},
        'sourceChangedOutsideSupportRows': {n: len(x['changedOutsideSupportRows']) for n, x in intake['rows'].items()},
        'nextQueryHeadRelativeDifference': {'count': len(heads), 'undefined': len(heads)-len(finite),
            'median': statistics.median(finite) if finite else None, 'min': min(finite) if finite else None,
            'max': max(finite) if finite else None, 'zeroDifferenceHeads': sum(x == 0 for x in finite)},
        'localFreeBytesObserved': shutil.disk_usage(root).free,
        'scope': 'Published scalar observations and complete metadata. Relative effects use the full-P control endpoint in each field/head; no new model/array computation or semantic labeling.'}
    save(e / 'scalar-summary.json', summary)
    job = read(root / 'artifacts/propagated-context-v1/scheduler-job.json')
    save(e / 'scheduler.json', {'jobID': job['job_id'], 'state': job['state'], 'reservation': job['reservation'], 'result': job['result'], 'executedPayload': job['payload_custody']['executed']})
    return summary


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
