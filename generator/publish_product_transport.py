"""Complete byte custody and lossless publication for the sealed local prediction."""
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
from publish_source_context import REMOTE


def items(p):
    s, t = read(p / 'inputs.json'), read(p / 'targets.json')
    primary = read(p / 'primary.json')
    groups = [s['fused'], primary['fields'], *t['cells'].values(), *read(p / 'response-fields.json').values()]
    return [s[k] for k in ['embedding', 'carrier', 'modulation']] + [t[k] for k in ['fusedR1A', 'fusedR1B', 'residualR1B']] + list(s['parentRecords'].values()) + [{'path': str(Path(s['modelRoot']) / n), 'sha256': h} for n, h in s['imageAssets'].items()] + [x for group in groups for x in group.values()]


CLOCK_REMOTE = REMOTE.split('print(json.dumps')[0] + '''import ast,struct
p=Path(s['clock']['path'])
if str(p) not in inventory:raise ValueError('Unverified clock parent')
with p.open('rb') as f:
 if f.read(8)!=b'\\x93NUMPY\\x01\\x00':raise ValueError('Unexpected clock NPY version')
 n=struct.unpack('<H',f.read(2))[0];header=ast.literal_eval(f.read(n).decode('latin1').strip());offset=f.tell()
 if header!={'descr':'<u2','fortran_order':False,'shape':(1,18432)}:raise ValueError('Unexpected clock parent geometry')
 f.seek(offset+2*15360);b=f.read(2*3072)
 if len(b)!=6144:raise ValueError('Short clock gate view')
h=hashlib.sha256(b).hexdigest()
if h!=s['clock']['gatePayloadSHA256']:raise ValueError('Clock gate slice differs')
print(json.dumps({'files':inventory,'totalBytesVerified':sum(v['bytes'] for v in inventory.values()),'clockGateView':{'parentPath':str(p),'lastAxisSlice':[15360,18432],'payloadSHA256':h,'bytesReadAgain':6144}}))
'''


def verify(root):
    root, collection, job, request = checked(root)
    p = root / 'result'
    s = read(p / 'inputs.json')
    a = read(p / 'primary.json')
    seal = sha(p / 'primary-seal.json')
    if read(p / 'report.json')['primarySealSHA256'] != seal or read(p / 'primary-seal.json')['primarySHA256'] != sha(p / 'primary.json'):
        raise ValueError('Candidate release differs')
    base = job['result']['diagnostics']['cwd']
    values = items(p) + [{'path': base + '/response-result/' + x.name, 'sha256': sha(x)} for x in p.iterdir()]
    unique = {}
    for item in values:
        if item['path'] in unique and unique[item['path']]['sha256'] != item['sha256']:
            raise ValueError('Conflicting byte identity')
        unique[item['path']] = item
    spec = {'allowedRoots': [base, s['modelRoot'], s['modulation']['path'], *s['protectedRoots']], 'items': list(unique.values()), 'clock': {'path': s['modulation']['path'], 'gatePayloadSHA256': a['fields']['clockGate']['payloadSHA256']}}
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(CLOCK_REMOTE)], input=json.dumps(spec), text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr)
    value = json.loads(result.stdout)
    value.update(schema='tce-product-transport-byte-verification-v1', jobID=job['job_id'], primarySealSHA256=seal, verifierSourceSHA256=sha(__file__), scope='Full admitted inputs, post-seal native/control operands, original assets and every complete new array. Clock gate independently byte-sliced;6144rereadbytes are not unique storage. SSH byte/metadata only; arrays remain near compute.')
    save(root / 'retained-array-verification.json', value)
    return {'files': len(value['files']), 'bytes': value['totalBytesVerified'], 'clockGateViewBytesReadAgain': 6144}


def publish(root, destination):
    root, collection, job, request = checked(root)
    p = root / 'result'
    verification = read(root / 'retained-array-verification.json')
    if verification['primarySealSHA256'] != sha(p / 'primary-seal.json') or verification['verifierSourceSHA256'] != sha(__file__):
        raise ValueError('Verification source/seal differs')
    out = Path(destination)
    out.mkdir(exist_ok=False, parents=True)
    streams = []
    for file in p.iterdir():
        if file.suffix == '.jsonl':
            raw = file.read_bytes()
            target = out / (file.name + '.gz')
            target.write_bytes(gzip.compress(raw, mtime=0))
            if gzip.decompress(target.read_bytes()) != raw:
                raise ValueError('Lossless stream differs')
            streams.append({'source': file.name, 'sourceSHA256': sha(file), 'sourceBytes': len(raw), 'published': target.name, 'publishedSHA256': sha(target), 'publishedBytes': target.stat().st_size})
        else:
            shutil.copyfile(file, out / file.name)
    shutil.copyfile(root / 'retained-array-verification.json', out / 'retained-array-verification.json')
    save(out / 'provenance.json', {'schema': 'tce-product-transport-publication-v1', 'jobID': job['job_id'], 'requestSHA256': request['requestSHA256'], 'payloadCustody': collection['custody'], 'schedulerResult': collection['result'], 'completeCollectedInventory': collection['inventory'], 'losslessStreams': streams, 'scope': 'One sealed conditional local replacement, with supplied native baselines/increments and one original down/SiLU paid. No cheap producer, complete velocity, image or strictzero-neural claim.'})
    save(out / 'files.json', {file.name: sha(file) for file in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(file.stat().st_size for file in out.iterdir())}


def audit(source):
    from product_transport import DOWN, STAGES
    from source_response import mask
    p = Path(source)
    r, s, a, report, e, v = [read(p / name) for name in ['request.json', 'inputs.json', 'primary.json', 'report.json', 'execution.json', 'retained-array-verification.json']]
    for name, digest in read(p / 'files.json').items():
        if sha(p / name) != digest:
            raise ValueError('Publication member differs')
    if identity({k: x for k, x in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request identity differs')
    for name in r['ownedSources']:
        if sha(Path(__file__).parent / name) != r['sourceHashes'][name]:
            raise ValueError('Executed worker differs')
    seal = read(p / 'primary-seal.json')
    if seal['requestSHA256'] != r['requestSHA256'] or seal['primarySHA256'] != sha(p / 'primary.json') or report['primarySealSHA256'] != sha(p / 'primary-seal.json') or v['primarySealSHA256'] != sha(p / 'primary-seal.json') or v['verifierSourceSHA256'] != sha(__file__):
        raise ValueError('Primary/verification seal chain differs')
    for item in items(p):
        if v['files'][item['path']]['sha256'] != item['sha256']:
            raise ValueError('Complete byte custody absent')
    if e['loadedParameters'] != s['selectedParameters'] or len(e['loadedParameters']) != 1 or e['loadedParameters'][0]['name'] != DOWN or e['loadedParameters'][0]['bytes'] != 56623104 or e['parameterInputRNGClosure'] is not True or e['numericalProfile'] != s['numericalProfile']:
        raise ValueError('Selected original input/parameter/profile closure differs')
    if len(e['linears']) != 1 or e['linears'][0]['nominalMACs'] != 28991029248 or e['linears'][0]['inputShape'] != [1, 1024, 9216] or report['nominalAffineMACs'] != 28991029248:
        raise ValueError('Single-down physical work differs')
    if not set(e['fileQuarantine']['successfulFileOpenCounts']).issubset(e['fileQuarantine']['allowedFiles']) or len(e['fileQuarantine']['deniedCanaries']) != 5:
        raise ValueError('Primary file access closure differs')
    names = {'prequantFP64', 'fused', 'quantizationDifference', 'clockGate', 'activation', 'product', 'mlpWrite', 'gatedMLP', 'residualCommit', 'imageExit'}
    if set(a['fields']) != names or report['nativeResidualPrefixExact'] is not True:
        raise ValueError('Complete candidate/prefix closure differs')
    for name, item in a['fields'].items():
        shape = [1, 1, 3072] if name == 'clockGate' else [1, 1024, 18432 if name in ['fused', 'prequantFP64', 'quantizationDifference'] else 9216 if name in ['activation', 'product'] else 3072]
        if item['shape'] != shape or item['logicalDtype'] != ('torch.float64' if name in ['prequantFP64', 'quantizationDifference'] else 'torch.bfloat16'):
            raise ValueError('Candidate field geometry differs')
    if v['clockGateView']['payloadSHA256'] != a['fields']['clockGate']['payloadSHA256'] or v['clockGateView']['bytesReadAgain'] != 6144:
        raise ValueError('Clock gate semantic byte binding differs')
    src, dst, ev = a['sourceFrame'], a['outputFrame'], a['evidence']
    fingerprints = {'fused' + k: x['payloadSHA256'] for k, x in s['fused'].items()}
    fingerprints.update(embedding=s['embedding']['payloadSHA256'], carrier=s['carrier']['payloadSHA256'], gate=a['fields']['clockGate']['payloadSHA256'])
    if {k: x['fingerprint'] for k, x in src['slots'].items()} != fingerprints or src['frame_id'] != ev['input_frame_id'] or dst['frame_id'] != ev['output_frame_id'] or dst['parent_frame_id'] != src['frame_id'] or ev['act'] != 'tce.recipient_product_transport:v1' or dst['program_fingerprint'] != r['requestSHA256'] or ev['metrics']['capturedImageExitPayloadSHA256'] != a['fields']['imageExit']['payloadSHA256']:
        raise ValueError('Typed consumed/output lineage differs')
    streams = {}
    provenance = read(p / 'provenance.json')
    for name in ['queries', 'operand-queries']:
        raw = gzip.decompress((p / (name + '.jsonl.gz')).read_bytes())
        receipt = next(x for x in provenance['losslessStreams'] if x['source'] == name + '.jsonl')
        if hashlib.sha256(raw).hexdigest() != receipt['sourceSHA256']:
            raise ValueError('Lossless stream identity differs')
        streams[name] = [json.loads(line) for line in raw.splitlines()]
    for key, categories, stream in [('stage', STAGES, streams['queries']), ('operand', ['gate', 'up'], streams['operand-queries'])]:
        for category in categories:
            rows = [x for x in stream if x[key] == category]
            if [x['imageQueryIndex'] for x in rows] != list(range(1024)) or [x['sourceNonlocal'] for x in rows] != mask('bottom-right').tolist():
                raise ValueError('Actual query coverage differs')
    if len(streams['queries']) != report['queryRecords'] or report['queryRecords'] != 4096 or len(streams['operand-queries']) != report['operandRecords'] or report['operandRecords'] != 2048 or len(report['accounts']) != 8 or len(report['operandAccounts']) != 4:
        raise ValueError('Account coverage differs')
    return {'schema': 'tce-product-transport-mechanics-v1', 'ownedSourcePins': len(r['ownedSources']), 'selectedMatrices': 1, 'affineCalls': 1, 'typedActs': 1, 'candidateFields': 10, 'responseQueries': 4096, 'operandQueries': 2048, 'nativeResidualPrefixExact': True, 'primarySealSHA256': sha(p / 'primary-seal.json'), 'scope': 'Source/seal/byte custody, isolated target access, selected work, typed lineage, prefix and complete query mechanics. No scientific quality gate, whole-future parity or generator certification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['verify', 'publish', 'audit'])
    parser.add_argument('source')
    parser.add_argument('destination', nargs='?')
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.destination) if args.action == 'publish' else globals()[args.action](args.source), indent=2))
