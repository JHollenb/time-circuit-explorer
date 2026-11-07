"""Full byte/custody witnesses and lossless scalar publication; no tensor math."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from encoder_atoms import read, save, sha, identity
from publish_source_native import checked
from publish_sparse_text_kv import NPY_BYTES
from rectifier_sign_support_study import bindings
from rectifier_sign_support import abis, EXPRESSIONS


def verify(root):
    from mrun.client.api import Api
    root, collection, job, request = checked(root)
    result = root / 'result'
    source = read(result / 'inputs.json')
    fields = read(result / 'fields.json')['fields']
    parent = Api().json('GET', '/api/jobs/' + source['parent']['jobID'])
    if parent['state'] != 'succeeded' or parent['payload_custody'] != source['parent']['custody']:
        raise ValueError('Parent scheduler custody differs')
    base = job['result']['diagnostics']['cwd']
    items = [*source['fields'].values(), *fields.values()]
    items += [{'path': source['parent']['root']+'/response-result/'+n, 'sha256': h}
              for n, h in source['parent']['records'].items()]
    items += [{'path': base+'/response-result/'+x.name, 'sha256': sha(x)} for x in result.iterdir()]
    unique = {}
    for item in items:
        if item['path'] in unique and unique[item['path']]['sha256'] != item['sha256']:
            raise ValueError('Conflicting byte identity')
        unique[item['path']] = {**unique.get(item['path'], {}), **item}
    spec = {'items': list(unique.values()), 'allowedRoots': [base, source['parent']['root'], *source['sourceRoots']]}
    process = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                              'beast', 'python3 -c '+shlex.quote(NPY_BYTES)], input=json.dumps(spec),
                             text=True, capture_output=True, timeout=180)
    if process.returncode:
        raise RuntimeError(process.stderr)
    witness = json.loads(process.stdout)
    witness.update(schema='tce-rectifier-sign-support-bytes-v1', jobID=job['job_id'],
                   reportSHA256=sha(result / 'report.json'), verifierSourceSHA256=sha(__file__),
                   templateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
                   scope='Complete four source arrays, six support/count arrays, parent metadata and new result bytes. Coefficient arithmetic/source audit and independent tiny scalar/bitwise references are separate; no real-array numerical replay, U/map read or output/semantic certification.')
    save(root / 'retained-array-verification.json', witness)
    return {'files': len(witness['files']), 'bytes': witness['totalBytesVerified'],
            'NPYPayloadWitnesses': len(witness['npyPayloadWitnesses'])}


def publish(root, destination):
    root, collection, job, request = checked(root)
    result = root / 'result'
    witness = read(root / 'retained-array-verification.json')
    if witness['verifierSourceSHA256'] != sha(__file__) or witness['reportSHA256'] != sha(result / 'report.json'):
        raise ValueError('Executed byte verifier/report differs')
    destination = Path(destination)
    destination.mkdir(exist_ok=False, parents=True)
    streams = []
    for path in result.iterdir():
        if path.suffix == '.jsonl':
            raw = path.read_bytes()
            target = destination / (path.name+'.gz')
            target.write_bytes(gzip.compress(raw, mtime=0))
            if gzip.decompress(target.read_bytes()) != raw:
                raise ValueError('Lossless stream differs')
            streams.append({'source': path.name, 'sourceSHA256': sha(path), 'sourceBytes': len(raw),
                            'published': target.name, 'publishedSHA256': sha(target), 'publishedBytes': target.stat().st_size})
        else:
            shutil.copyfile(path, destination / path.name)
    shutil.copyfile(root / 'retained-array-verification.json', destination / 'retained-array-verification.json')
    save(destination / 'scheduler.json', {k: job[k] for k in ['job_id', 'state', 'reservation', 'admission', 'payload_custody', 'result']})
    save(destination / 'provenance.json', {'schema': 'tce-rectifier-sign-support-publication-v1',
         'jobID': job['job_id'], 'requestSHA256': request['requestSHA256'],
         'payloadCustody': collection['custody'], 'completeCollectedInventory': collection['inventory'],
         'losslessStreams': streams,
         'scope': 'Saved-only coefficient-support census, no learned products/new state/image. G/means/support/history paid; prospective arithmetic counts are not physical lowering, runtime or arbitrary-prompt zero-neural serving.'})
    save(destination / 'files.json', {x.name: sha(x) for x in destination.iterdir()})
    return {'files': len(list(destination.iterdir())), 'bytes': sum(x.stat().st_size for x in destination.iterdir())}


def audit(result):
    from publish_rectifier_correction import audit as parent_audit
    result = Path(result)
    request = read(result / 'request.json')
    source = read(result / 'inputs.json')
    report = read(result / 'report.json')
    execution = read(result / 'execution.json')
    fields = read(result / 'fields.json')
    witness = read(result / 'retained-array-verification.json')
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']:
        raise ValueError('Request differs')
    for name, fingerprint in read(result / 'files.json').items():
        if sha(result / name) != fingerprint:
            raise ValueError('Publication bytes differ')
    for name in request['ownedSources']:
        if sha(Path(__file__).with_name(name)) != request['sourceHashes'][name]:
            raise ValueError('Executed source differs: '+name)
    root = Path(__file__).resolve().parent.parent
    parent = root / 'docs/results/rectifier-correction'
    parent_audit(parent)
    if source['fields'] != bindings(parent):
        raise ValueError('Exact four source bindings differ')
    for name, fingerprint in source['parent']['records'].items():
        if sha(parent / name) != fingerprint:
            raise ValueError('Parent metadata differs')
    if witness['verifierSourceSHA256'] != sha(__file__) or witness['templateSourceSHA256'] != sha(Path(__file__).with_name('publish_sparse_text_kv.py')):
        raise ValueError('Verifier/template differs')
    if witness['reportSHA256'] != sha(result / 'report.json') or execution['sourceManifestSHA256'] != identity(source['fields']):
        raise ValueError('Report/source manifest differs')
    for item in [*source['fields'].values(), *fields['fields'].values()]:
        if witness['files'][item['path']]['sha256'] != item['sha256'] or witness['npyPayloadWitnesses'][item['path']]['payloadSHA256'] != item['payloadSHA256']:
            raise ValueError('Full byte witness absent')
    if {n: (x['shape'], x['storedDtype']) for n, x in fields['fields'].items()} != abis():
        raise ValueError('Six complete support/count ABIs differ')
    guard = execution['fileQuarantine']
    allowed = {x['path'] for x in source['fields'].values()}
    denied = set(source['deniedNumericalInputs'])
    if (set(guard['allowedFiles']) != allowed or set(guard['deniedCanaries']) != denied
            or allowed & denied or not set(guard['successfulFileOpenCounts']).issubset(allowed)):
        raise ValueError('Four-input quarantine differs')
    seal = read(result / 'result-seal.json')
    if seal['requestSHA256'] != request['requestSHA256'] or any(sha(result / n) != h for n, h in seal['files'].items() if not n.endswith('.jsonl')):
        raise ValueError('Sealed result differs')
    streams = {}
    for stream in read(result / 'provenance.json')['losslessStreams']:
        raw = gzip.decompress((result / stream['published']).read_bytes())
        if hashlib.sha256(raw).hexdigest() != stream['sourceSHA256'] or seal['files'][stream['source']] != stream['sourceSHA256']:
            raise ValueError('Lossless sealed stream differs')
        streams[stream['source']] = [json.loads(line) for line in raw.splitlines()]
    rows, tiles = streams['observations.jsonl'], streams['tiles.jsonl']
    if [x['row'] for x in rows] != list(range(1536)) or [x['startRow'] for x in tiles] != list(range(0, 1536, 16)):
        raise ValueError('Complete row/tile coverage differs')
    for name, lo, hi in [('all', 0, 1536), ('text', 0, 512), ('image', 512, 1536)]:
        for expression in EXPRESSIONS:
            item = report['panels'][name]['expressions'][expression]
            total = sum(x['activeOmitted'][expression] for x in rows[lo:hi])
            tiled = sum(x['unionOmitted'][expression] for x in tiles[lo//16:hi//16])
            if item['liveCoefficients'] != total or item['idealRowwiseUpDownMACs'] != 6144*total or item['fixedTileUpDownMACs'] != 6144*16*tiled:
                raise ValueError('Scalar observations/prospective arithmetic differ')
    expected = {'newLearnedMatrixMACs': 0, 'newCheckpointTensorLoads': 0, 'newRectifierElements': 14155776,
                'newRetainedArrays': 6, 'newNativeStateUpdates': 0, 'newImages': 0}
    if (report['physicalWork'] != expected or report['requestSHA256'] != request['requestSHA256']
            or fields['bitOrder'] != 'little' or fields['expressionAxis'] != list(EXPRESSIONS)
            or sha(result / 'frozen-plan.md') != request['sourceHashes']['frozen-plan.md']):
        raise ValueError('Work/expression/plan scope differs')
    return {'schema': 'tce-rectifier-sign-support-mechanics-v1', 'sourceInputs': 4,
            'observations': 1536, 'tiles': 96, 'retainedArrays': 6,
            'fullByteFiles': len(witness['files']), 'NPYPayloadWitnesses': len(witness['npyPayloadWitnesses']),
            'newLearnedMatrixMACs': 0,
            'scope': 'Source/ancestry/quarantine/fullbytes/complete support-count ABIs/lossless scalar rows and prospective count joins. No independent real-array numerical replay, U/map evaluation, finite output parity, runtime or semantic certification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['verify', 'publish', 'audit'])
    parser.add_argument('root')
    parser.add_argument('--destination')
    args = parser.parse_args()
    value = publish(args.root, args.destination) if args.action == 'publish' else globals()[args.action](args.root)
    print(json.dumps(value, indent=2))
