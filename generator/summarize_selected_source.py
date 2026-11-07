"""Small summaries of already-published scalars and complete metadata."""
import gzip
import json
from pathlib import Path
import shutil
from encoder_atoms import read, save, sha
from projected_complement import fields
from audit_selected_source import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/selected-source'
    out = root / 'docs/results/selected-source-evidence'
    out.mkdir(exist_ok=False, parents=True)
    save(out / 'mechanics.json', audit(p))
    r, receipt, provenance = [read(p / n) for n in ['report.json', 'retained-array-verification.json', 'provenance.json']]
    base = provenance['schedulerResult']['diagnostics']['cwd'] + '/arrays/'
    arrays = {x['path']: x for name in ['producer.json', 'compiler.json', 'primary.json'] for x in fields(read(p / name)) if x['path'].startswith(base)}
    rows = [json.loads(x) for x in gzip.decompress((p / 'queries.jsonl.gz').read_bytes()).splitlines()]
    parent_path = root / 'docs/results/projected-complement'
    previous = [json.loads(x) for x in gzip.decompress((parent_path / 'queries.jsonl.gz').read_bytes()).splitlines()]
    change = {'lowerNativeError': 0, 'higherNativeError': 0, 'equalNativeError': 0}
    for old, new in zip(previous, rows):
        if old['imageQueryIndex'] != new['imageQueryIndex']:
            raise ValueError('Actual query address differs')
        a, b = old['references']['family']['response']['rms'], new['references']['family']['response']['rms']
        change['lowerNativeError' if b < a else 'higherNativeError' if b > a else 'equalNativeError'] += 1
    trends = {ref: {'negativeResponseCosines': sum(x['references'][ref]['response']['cosine'] is not None and x['references'][ref]['response']['cosine'] < 0 for x in rows),
                    'undefinedRelativeErrors': sum(x['references'][ref]['response']['relativeRms'] is None for x in rows)} for ref in ['family', 'parent']}
    old = next(x for x in read(parent_path / 'report.json')['responses'] if x['reference'] == 'family')
    new = next(x for x in r['responses'] if x['reference'] == 'family')
    delta = next(x for x in r['responses'] if x['reference'] == 'parent')
    cross = new['rms'] ** 2 - old['rms'] ** 2 - delta['rms'] ** 2
    composition = {'law': 'e_new=e_parent+delta; RMS(e_new)^2=RMS(e_parent)^2+RMS(delta)^2+2mean(e_parent*delta)',
                   'oldNativeErrorRMS': old['rms'], 'newNativeErrorRMS': new['rms'], 'changeFromParentRMS': delta['rms'],
                   'twiceMeanErrorDotChange': cross, 'errorChangeCosine': cross / (2 * old['rms'] * delta['rms']),
                   'scope': 'Reconstructed from three published squared RMS values; signed composition identity, not causal attribution to an internal component.'}
    save(out / 'scalar-summary.json', {'schema': 'tce-selected-source-scalars-v1', 'reportSHA256': sha(p / 'report.json'),
         'newArrays': len(arrays), 'newArrayBytes': sum(x['bytes'] for x in arrays.values()),
         'fullByteFiles': len(receipt['files']), 'fullByteCount': receipt['totalBytesVerified'],
         'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']),
         'NPYPayloadBytesReadAgain': sum(x['payloadBytes'] for x in receipt['npyPayloadWitnesses'].values()),
         'perQueryNativeErrorChangeFromParent': change, 'queryTrends': trends, 'signedErrorComposition': composition,
         'historicalCompilersPlusNewSourceAndReaderMatrixMACs': 48318382080 + 9663676416 + r['totalIncrementalMatrixMACs'],
         'localFreeBytesObserved': shutil.disk_usage(root).free,
         'scope': 'Published scalars/complete address rows and metadata only. Byte rereads are not new storage; free space is an observation, not cleanup attribution.'})
    job = read(root / 'artifacts/selected-source-v2/scheduler-job.json')
    save(out / 'scheduler.json', {'jobID': job['job_id'], 'state': job['state'], 'reservation': job['reservation'],
         'result': job['result'], 'executedPayload': job['payload_custody']['executed']})
    return {'arrays': len(arrays), 'bytes': sum(x['bytes'] for x in arrays.values()), 'queryNativeErrorChange': change,
            'queryTrends': trends, 'signedErrorComposition': composition}


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
