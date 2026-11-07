"""Complete byte custody, selected-segment witness and lossless publication."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from encoder_atoms import identity, read, save, sha
from projected_complement import fields
from publish_source_native import checked
from publish_sparse_text_kv import NPY_BYTES

BYTES = NPY_BYTES.rsplit('print(json.dumps', 1)[0] + r'''
intake=s['intake'];source=intake['normSource'];data=Path(source['path']).read_bytes()
assert hashlib.sha256(data).hexdigest()==source['sha256']
assert data[:8]==b'\x93NUMPY\x01\x00' and intake['normDataOffset']==10+struct.unpack('<H',data[8:10])[0]
segments=intake['selectedSegments'];assert len(segments)==256 and len({x['row'] for x in segments})==256
selected=[]
for x in segments:
 assert x['length']==192 and x['offset']==intake['normDataOffset']+x['row']*192 and 0<=x['row']<1024
 b=data[x['offset']:x['offset']+x['length']];assert hashlib.sha256(b).hexdigest()==x['sha256'];selected.append(b)
joined=b''.join(selected);digest=hashlib.sha256(joined).hexdigest()
assert digest==intake['fields']['knots']['payloadSHA256']==witnesses[intake['fields']['knots']['path']]['payloadSHA256']
assert len(joined)==49152
assert intake['nativeDescriptor']['sha256']==witnesses[intake['fields']['ids']['path']]['payloadSHA256']
blobs=[name for name in intake['nativeFiles'] if name.endswith('.safetensors')];assert len(blobs)==1
native=Path(blobs[0]).read_bytes();start=intake['nativeDataOffset'];native_payload=native[start:start+32768]
assert hashlib.sha256(native_payload).hexdigest()==intake['nativeDescriptor']['sha256']
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),
 'npyPayloadWitnesses':witnesses,'selectedSegmentWitness':{'segments':256,'bytes':len(joined),'payloadSHA256':digest},
 'nativeIDPayloadWitness':{'payloadSHA256':intake['nativeDescriptor']['sha256'],'bytes':32768}}))
'''


def execute(root, destination):
    from mrun.client.api import Api
    root, collection, job, request = checked(root)
    p = root / 'result';s, intake = read(p / 'inputs.json'), read(p / 'intake.json')
    roots = [job['result']['diagnostics']['cwd'], s['store']]
    for jkey, ckey in [('parentJobID', 'parentCustody'), ('referenceJobID', 'referenceCustody'), ('nativeJobID', 'nativeCustody')]:
        parent = Api().json('GET', '/api/jobs/' + s[jkey])
        if parent['state'] != 'succeeded' or parent['payload_custody'] != s[ckey]:
            raise ValueError('Saved/native input custody changed')
        roots.append(parent['result']['diagnostics']['cwd'])
    values = list(fields({n: s[n] for n in ['pNorm', 'qNorm', 'numerator', 'textLogits']}))
    for n in ['intake.json', 'candidate.json', 'report.json']:
        values += list(fields(read(p / n)))
    values += [{'path': n, 'sha256': h} for n,h in intake['nativeFiles'].items()]
    values += [{'path': roots[0] + '/response-result/' + n.name, 'sha256': sha(n)} for n in p.iterdir()]
    unique = {}
    for item in values:
        if item['path'] in unique and unique[item['path']]['sha256'] != item['sha256']:
            raise ValueError('Conflicting byte identities')
        # Preserve payload descriptors when a source appears again as a plain file.
        unique[item['path']] = {**unique.get(item['path'], {}), **item}
    proc = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(BYTES)],
        input=json.dumps({'allowedRoots': roots, 'items': list(unique.values()), 'intake': intake}),
        text=True, capture_output=True, timeout=90)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    receipt = json.loads(proc.stdout)
    receipt.update(schema='tce-spatial-norm-bytes-v1', jobID=job['job_id'], verifierSourceSHA256=sha(__file__),
        templateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')), candidateSealSHA256=sha(p / 'candidate-seal.json'))
    save(root / 'retained-array-verification.json', receipt)
    out = Path(destination);out.mkdir(parents=True, exist_ok=False);streams = []
    for n in p.iterdir():
        if n.suffix == '.jsonl':
            target = out / (n.name + '.gz');raw = n.read_bytes();target.write_bytes(gzip.compress(raw, mtime=0))
            streams.append({'source': n.name, 'sourceSHA256': sha(n), 'published': target.name, 'publishedSHA256': sha(target)})
        else:
            shutil.copyfile(n, out / n.name)
    save(out / 'retained-array-verification.json', receipt)
    save(out / 'scheduler.json', {k: job[k] for k in ['job_id', 'state', 'reservation', 'admission', 'payload_custody', 'result']})
    save(out / 'provenance.json', {'schema': 'tce-spatial-norm-publication-v1', 'jobID': job['job_id'],
        'requestSHA256': request['requestSHA256'], 'payloadCustody': collection['custody'],
        'completeCollectedInventory': collection['inventory'], 'losslessStreams': streams,
        'scope': 'Saved-only fixed-grid norm capacity and score diagnostics. Current native knots remain paid; no original projection, new consumer/image or serving speedup.'})
    save(out / 'files.json', {n.name: sha(n) for n in out.iterdir()})
    return {'files': len(receipt['files']), 'bytes': receipt['totalBytesVerified'], 'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses'])}


def audit(source):
    p = Path(source);r, s, report, intake, candidate, seal, verification = [read(p / n) for n in
        ['request.json', 'inputs.json', 'report.json', 'intake.json', 'candidate.json', 'candidate-seal.json', 'retained-array-verification.json']]
    if identity({k: v for k,v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request identity differs')
    for n,h in read(p / 'files.json').items():
        if sha(p / n) != h:
            raise ValueError('Immutable publication changed: ' + n)
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n)) != r['sourceHashes'][n]:
            raise ValueError('Frozen executed source differs: ' + n)
    if verification['verifierSourceSHA256'] != sha(__file__) or verification['templateSourceSHA256'] != sha(Path(__file__).with_name('publish_sparse_text_kv.py')):
        raise ValueError('Full byte verifier changed')
    if seal['candidateSHA256'] != sha(p / 'candidate.json') or seal['intakeSHA256'] != sha(p / 'intake.json') or candidate['intakeSHA256'] != sha(p / 'intake.json') or report['candidateSealSHA256'] != sha(p / 'candidate-seal.json'):
        raise ValueError('Intake/candidate/process-exit diagnostic seal differs')
    expected_allowed = sorted(x['path'] for x in intake['fields'].values())
    expected_denied = [s[n]['path'] for n in ['pNorm', 'qNorm', 'numerator', 'textLogits']]
    guard = candidate['fileQuarantine']
    if guard['allowedFiles'] != expected_allowed or guard['deniedCanaries'] != expected_denied or not set(guard['successfulFileOpenCounts']).issubset(expected_allowed):
        raise ValueError('Actual candidate numerical file closure differs')
    for item in [*fields(intake), *fields(candidate), *fields(report), *fields({n:s[n] for n in ['pNorm','qNorm','numerator','textLogits']})]:
        witness = verification['npyPayloadWitnesses'][item['path']]
        if verification['files'][item['path']]['sha256'] != item['sha256'] or witness['payloadSHA256'] != item['payloadSHA256']:
            raise ValueError('Complete consumed/emitted byte witness absent')
    if verification['selectedSegmentWitness'] != {'segments':256,'bytes':49152,'payloadSHA256':intake['fields']['knots']['payloadSHA256']} or intake['strataRows'] != [256,705,63]:
        raise ValueError('Fixed selected segment/strata witness differs')
    if sha(p / 'frozen-plan.md') != r['sourceHashes']['frozen-plan.md']:
        raise ValueError('Executed plan changed')
    provenance = read(p / 'provenance.json')
    raw = gzip.decompress((p / 'observations.jsonl.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest() != provenance['losslessStreams'][0]['sourceSHA256']:
        raise ValueError('Lossless observations differ')
    rows = [json.loads(line) for line in raw.splitlines()]
    addresses = [(x['panel'], x['row'], x['head']) for x in rows]
    expected = {('image-key-head', j, h) for j in range(1024) for h in range(24)}
    expected |= {('query-head', j, h) for j in r['support'] for h in range(24)}
    if len(addresses) != len(set(addresses)) or set(addresses) != expected:
        raise ValueError('Complete unique norm/query-head observations differ')
    return {'schema':'tce-spatial-norm-mechanics-v1','ownedSourcePins':len(r['ownedSources']),
        'nativeIDPayloadWitness':verification['nativeIDPayloadWitness'],'knots':256,'selectedBytes':49152,
        'strataRows':intake['strataRows'],'candidateNumericalInputFiles':len(expected_allowed),
        'candidateDeniedCanaries':len(expected_denied),'normObservations':24576,'queryHeadObservations':96,
        'observations':len(rows),'fullByteFiles':len(verification['files']),
        'NPYPayloadWitnesses':len(verification['npyPayloadWitnesses']),
        'scope':'Source/byte/geometry/seal/file-closure/address mechanics only; no effect-size or semantic/image/generalization certification.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(execute(a.source,a.destination) if a.action=='publish' else audit(a.source),indent=2))
