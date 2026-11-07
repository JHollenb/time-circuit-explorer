"""Summarize published scalars and byte witnesses without model/array execution."""
import gzip
import json
from pathlib import Path
import shutil
from encoder_atoms import read, save, sha
from projected_complement import fields
from publish_source_complement import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/source-complement'
    out = root / 'docs/results/source-complement-evidence'
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'mechanics.json', audit(p))
    r, receipt, provenance = [read(p / n) for n in ['report.json', 'retained-array-verification.json', 'provenance.json']]
    base = provenance['schedulerResult']['diagnostics']['cwd'] + '/arrays/'
    arrays = {x['path']: x for name in ['shared.json', 'primary.json', 'control.json'] for x in fields(read(p / name)) if x['path'].startswith(base)}
    rows = [json.loads(x) for x in gzip.decompress((p / 'queries.jsonl.gz').read_bytes()).splitlines()]
    old = [json.loads(x) for x in gzip.decompress((root / 'docs/results/selected-source/queries.jsonl.gz').read_bytes()).splitlines()]
    trends = {}
    for arm in ['primary', 'control']:
        actual = [x for x in rows if x['arm'] == arm]
        change = dict(lowerNativeError=0, higherNativeError=0, equalNativeError=0)
        for a, b in zip(old, actual):
            if a['imageQueryIndex'] != b['imageQueryIndex']:
                raise ValueError('Query address differs')
            prior, current = a['references']['family']['response']['rms'], b['nativeResponse']['rms']
            change['lowerNativeError' if current < prior else 'higherNativeError' if current > prior else 'equalNativeError'] += 1
        trends[arm] = {'nativeErrorChangeFromParent': change,
                       'negativeNativeResponseCosines': sum(x['nativeResponse']['cosine'] is not None and x['nativeResponse']['cosine'] < 0 for x in actual)}
    joins = {n: {x['arm']: x for x in receipt['bankSpliceWitnesses'] if x['field'] == n} for n in ['K', 'V']}
    complements = {n: {'primarySHA256': x['primary']['outsideSupportSHA256'], 'controlSHA256': x['control']['outsideSupportSHA256'],
                       'byteIdentical': x['primary']['outsideSupportSHA256'] == x['control']['outsideSupportSHA256']} for n, x in joins.items()}
    if not all(x['byteIdentical'] for x in complements.values()):
        raise ValueError('This summary specifically diagnoses the observed unchanged complement')
    summary = {'schema': 'tce-source-complement-scalars-v1', 'reportSHA256': sha(p / 'report.json'),
               'newArrays': len(arrays), 'newArrayBytes': sum(x['bytes'] for x in arrays.values()),
               'fullByteFiles': len(receipt['files']), 'fullByteCount': receipt['totalBytesVerified'],
               'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']),
               'NPYPayloadBytesReadAgain': sum(x['payloadBytes'] for x in receipt['npyPayloadWitnesses'].values()),
               'spliceBytesReadAgain': receipt['spliceBytesReadAgain'], 'complementByteIdentity': complements,
               'queryTrends': trends, 'localFreeBytesObserved': shutil.disk_usage(root).free,
               'mechanics_status': 'complete', 'instrument_status': 'context-contrast-inapplicable-identical-complements',
               'trend_status': 'conditional-reuse-and-finite-drift-observed; context-sufficiency-untested',
               'terminal_status': 'unestablished',
               'scope': 'Additive post-run interpretation. The immutable raw report remains unchanged. Identical complementary inputs make this substitution uninformative about genuinely changed context. Byte rereads are not additional storage.'}
    save(out / 'scalar-summary.json', summary)
    job = read(root / 'artifacts/source-complement-v1/scheduler-job.json')
    save(out / 'scheduler.json', {'jobID': job['job_id'], 'state': job['state'], 'reservation': job['reservation'],
                                'result': job['result'], 'executedPayload': job['payload_custody']['executed']})
    return summary


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
