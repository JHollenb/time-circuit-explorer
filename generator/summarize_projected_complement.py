"""Audited complete rows, work and new-storage summaries from retained evidence."""
import gzip
import json
from pathlib import Path
import shutil
from encoder_atoms import read, save, sha
from publish_projected_complement import audit
from projected_complement import fields


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/projected-complement'
    out = root / 'docs/results/projected-complement-evidence'
    out.mkdir(exist_ok=False, parents=True)
    save(out / 'mechanics.json', audit(p))
    report, old, receipt, provenance = [read(p / n) for n in ['report.json', 'inputs.json', 'retained-array-verification.json', 'provenance.json']]
    base = provenance['schedulerResult']['diagnostics']['cwd'] + '/arrays/'
    arrays = {x['path']: x for name in ['compiler.json', 'primary.json'] for x in fields(read(p / name)) if x['path'].startswith(base)}
    rows = [json.loads(line) for line in gzip.decompress((p / 'queries.jsonl.gz').read_bytes()).splitlines()]
    parent_report = read(root / 'docs/results/sparse-text-kv/report.json')
    trends = {}
    for ref in ['family', 'parent']:
        groups = [x['references'][ref]['response'] for x in rows]
        trends[ref] = {'negativeResponseCosines': sum(x['cosine'] is not None and x['cosine'] < 0 for x in groups),
                       'undefinedRelativeErrors': sum(x['relativeRms'] is None for x in groups)}
    # Compare each row's new native error with the same row's parent native error.
    original_rows = [json.loads(line) for line in gzip.decompress((root / 'docs/results/sparse-text-kv/queries.jsonl.gz').read_bytes()).splitlines()]
    row_change = {'lowerNativeError': 0, 'higherNativeError': 0, 'equalNativeError': 0}
    for a, b in zip(original_rows, rows):
        if a['imageQueryIndex'] != b['imageQueryIndex']:
            raise ValueError('Actual query comparison address differs')
        x, y = a['response']['rms'], b['references']['family']['response']['rms']
        row_change['lowerNativeError' if y < x else 'higherNativeError' if y > x else 'equalNativeError'] += 1
    save(out / 'scalar-summary.json', {'schema': 'tce-projected-complement-scalars-v1', 'reportSHA256': sha(p / 'report.json'),
        'compilerSealSHA256': sha(p / 'compiler-seal.json'), 'primarySealSHA256': sha(p / 'primary-seal.json'),
        'newArrays': len(arrays), 'newArrayBytes': sum(x['bytes'] for x in arrays.values()), 'fullByteFiles': len(receipt['files']), 'fullByteCount': receipt['totalBytesVerified'],
        'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']), 'NPYPayloadBytesReadAgain': sum(x['payloadBytes'] for x in receipt['npyPayloadWitnesses'].values()),
        'queryTrends': trends, 'perQueryNativeErrorChangeFromParent': row_change, 'queryPanels': report['queryPanels'],
        'matrixMACsPerActor': report['actorsMatrixMACs'] // 2, 'parentMatrixMACsPerActor': parent_report['actorsMACs'] // 2,
        'historicalBaselinePlusNewCompilerAndActorsMatrixMACs': report['historicalBaselineCompilerMatrixMACs'] + report['totalIncrementalMatrixMACs'],
        'localFreeBytesObserved': shutil.disk_usage(root).free,
        'scope': 'Already-scored complete dependent rows and exact metadata/byte sizes. Payload rereads are not new storage; free space is an observation, not cleanup attribution.'})
    job = read(root / 'artifacts/projected-complement-v1/scheduler-job.json')
    save(out / 'scheduler.json', {'jobID': job['job_id'], 'state': job['state'], 'reservation': job['reservation'], 'result': job['result'], 'executedPayload': job['payload_custody']['executed']})
    return {'arrays': len(arrays), 'bytes': sum(x['bytes'] for x in arrays.values()), 'queryNativeErrorChange': row_change, 'cosines': trends}


if __name__ == '__main__':
    print(execute())
