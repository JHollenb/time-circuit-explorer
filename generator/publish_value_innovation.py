"""Complete byte custody and lossless addressed-value result publication."""
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


def fields(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            yield value
        else:
            for child in value.values():
                yield from fields(child)
    elif isinstance(value, list):
        for child in value:
            yield from fields(child)


def items(p):
    s = read(p / 'inputs.json')
    result = [x for name in ['inputs.json', 'targets.json', 'primary.json', 'reference.json', 'response-fields.json'] for x in fields(read(p / name))]
    result += [{'path': str(Path(s['modelRoot']) / n), 'sha256': h} for n, h in s['imageAssets'].items()]
    for name in ['conditioning', 'txtIDs']:
        for file in s[name]['files']:
            result.append({'path': file, 'sha256': Path(file).stem})
    return result


BYTE_SWAP = REMOTE.split('print(json.dumps')[0] + '''
import ast,struct
p=Path(s['V']['path'])
with p.open('rb') as f:
 if f.read(8)!=bytes([147,78,85,77,80,89,1,0]):raise ValueError('Unexpected V NPY version')
 n=struct.unpack('<H',f.read(2))[0];header=ast.literal_eval(f.read(n).decode('latin1').strip());raw=f.read()
if header!={'descr':'<u2','fortran_order':False,'shape':(1,1536,24,128)} or len(raw)!=9437184:raise ValueError('V byte geometry differs')
if hashlib.sha256(raw).hexdigest()!=s['V']['payloadSHA256']:raise ValueError('Original V payload differs')
j,k=s['keyIndices'];width=6144
swapped=bytearray(raw);swapped[j*width:(j+1)*width]=raw[k*width:(k+1)*width];swapped[k*width:(k+1)*width]=raw[j*width:(j+1)*width]
digest=hashlib.sha256(swapped).hexdigest()
if digest!=s['nativeVFingerprint']:raise ValueError('Native consumed V does not equal declared byte-row swap')
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'nativeVSwap':{'originalPayloadSHA256':s['V']['payloadSHA256'],'swappedPayloadSHA256':digest,'keyIndices':[j,k],'rowBytes':width,'payloadBytesReadAgain':len(raw)}}))
'''


def verify(root):
    root, c, job, request = checked(root)
    p = root / 'result'
    s, a, reference = [read(p / n) for n in ['inputs.json', 'primary.json', 'reference.json']]
    unique = {}
    base = job['result']['diagnostics']['cwd']
    for item in items(p) + [{'path': base + '/response-result/' + file.name, 'sha256': sha(file)} for file in p.iterdir()]:
        if item['path'] in unique and unique[item['path']]['sha256'] != item['sha256']:
            raise ValueError('Conflicting supplied/produced file identity')
        unique[item['path']] = item
    spec = {'items': list(unique.values()), 'allowedRoots': [base, s['modelRoot'], s['modImage']['path'], s['modText']['path'], *s['protectedRoots']], 'V': a['acquisition']['V'], 'keyIndices': request['event']['keyIndices'],
            'nativeVFingerprint': reference['frame']['sourceFrame']['slots']['V']['fingerprint']}
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(BYTE_SWAP)], input=json.dumps(spec), text=True, capture_output=True, timeout=240)
    if result.returncode:
        raise RuntimeError(result.stderr)
    receipt = json.loads(result.stdout)
    receipt.update(schema='tce-value-innovation-bytes-v1', jobID=job['job_id'], primarySealSHA256=sha(p / 'primary-seal.json'), referenceSealSHA256=sha(p / 'reference-seal.json'), verifierSourceSHA256=sha(__file__),
                   scope='Complete supplied/produced/native/inherited fields, source pages, original checkpoint and compact result files. Independent V-row byte permutation matches the native consumer fingerprint; this is byte I/O, not scientific/model computation over SSH.')
    save(root / 'retained-array-verification.json', receipt)
    return {'files': len(receipt['files']), 'bytes': receipt['totalBytesVerified'], 'nativeVSwapVerified': True}


def publish(root, destination):
    root, c, job, request = checked(root)
    p = root / 'result'
    receipt = read(root / 'retained-array-verification.json')
    if receipt['primarySealSHA256'] != sha(p / 'primary-seal.json') or receipt['referenceSealSHA256'] != sha(p / 'reference-seal.json') or receipt['verifierSourceSHA256'] != sha(__file__):
        raise ValueError('Full-byte verifier source/seals differ')
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    streams = []
    for file in p.iterdir():
        if file.suffix == '.jsonl':
            raw = file.read_bytes()
            target = out / (file.name + '.gz')
            target.write_bytes(gzip.compress(raw, mtime=0))
            if gzip.decompress(target.read_bytes()) != raw:
                raise ValueError('Lossless stream changed')
            streams.append({'source': file.name, 'sourceSHA256': sha(file), 'sourceBytes': len(raw), 'published': target.name, 'publishedSHA256': sha(target), 'publishedBytes': target.stat().st_size})
        else:
            shutil.copyfile(file, out / file.name)
    shutil.copyfile(root / 'retained-array-verification.json', out / 'retained-array-verification.json')
    save(out / 'provenance.json', {'schema': 'tce-value-innovation-publication-v1', 'jobID': job['job_id'], 'requestSHA256': request['requestSHA256'], 'payloadCustody': c['custody'], 'schedulerResult': c['result'], 'completeCollectedInventory': c['inventory'], 'losslessStreams': streams,
                                  'scope': 'One fixed-Q/K addressed value event and own local candidate, sealed before new native B. Baselines, formation, original learned maps and reference arithmetic paid. No whole-image or arbitrary-prompt zero-neural claim.'})
    save(out / 'files.json', {file.name: sha(file) for file in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(file.stat().st_size for file in out.iterdir())}


def audit(source):
    from value_innovation import WO, IN, DOWN, STAGES
    p = Path(source)
    r, s, a, reference, pe, re, report, verification = [read(p / n) for n in ['request.json', 'inputs.json', 'primary.json', 'reference.json', 'execution.json', 'reference-execution.json', 'report.json', 'retained-array-verification.json']]
    for name, digest in read(p / 'files.json').items():
        if sha(p / name) != digest:
            raise ValueError('Publication member differs')
    if identity({k: v for k, v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request identity differs')
    for name in r['ownedSources']:
        if sha(Path(__file__).parent / name) != r['sourceHashes'][name]:
            raise ValueError('Frozen executed source differs')
    seal = read(p / 'primary-seal.json')
    refseal = read(p / 'reference-seal.json')
    if seal['primarySHA256'] != sha(p / 'primary.json') or refseal['referenceSHA256'] != sha(p / 'reference.json') or report['primarySealSHA256'] != sha(p / 'primary-seal.json') or reference['primarySealSHA256'] != sha(p / 'primary-seal.json') or report['referenceSealSHA256'] != sha(p / 'reference-seal.json'):
        raise ValueError('Candidate/reference seal order chain differs')
    if verification['primarySealSHA256'] != sha(p / 'primary-seal.json') or verification['referenceSealSHA256'] != sha(p / 'reference-seal.json') or verification['verifierSourceSHA256'] != sha(__file__):
        raise ValueError('Byte verifier seal/source differs')
    for item in items(p):
        if verification['files'][item['path']]['sha256'] != item['sha256']:
            raise ValueError('Full-byte custody missing')
    if {x['name']: x for x in pe['loadedParameters']} != {x['name']: x for x in s['selectedParameters']} or pe['parameterInputRNGClosure'] is not True or re['parameterInputRNGClosure'] is not True or pe['numericalProfile'] != re['numericalProfile'] or pe['numericalProfile'] != s['numericalProfile']:
        raise ValueError('Parameter/input/profile closure differs')
    if len(pe['linears']) != 33 or len(pe['contractions']) != 3 or sum(x['nominalMACs'] for x in pe['linears'] + pe['contractions']) != 86856695808:
        raise ValueError('Candidate contraction count differs')
    heads = [x for x in pe['linears'] if x['parameter'] == WO]
    if [x['headIndex'] for x in heads] != list(range(24)) or any(x['weightShape'] != [3072, 128] or x['inputShape'] != [128] for x in heads):
        raise ValueError('Physical head-column projection differs')
    if sum(x['phase'] == 'acquisition' for x in pe['linears']) != 7 or sum(x['parameter'] == IN for x in pe['linears']) != 1 or sum(x['parameter'] == DOWN for x in pe['linears']) != 1:
        raise ValueError('Candidate original map use differs')
    if reference['inheritedA'] != all(reference['sham'].values()) or len(re['attention']) != 2 or len(re['linears']) != (4 if reference['inheritedA'] else 6):
        raise ValueError('Reference/sham physical work differs')
    expected_reference = 125627793408 + (0 if reference['inheritedA'] else 86973087744)
    if re['nominalReferenceMACs'] != expected_reference or report['totalNewMACs'] != 86856695808 + expected_reference:
        raise ValueError('Complete work account differs')
    if not set(pe['fileQuarantine']['successfulFileOpenCounts']).issubset(pe['fileQuarantine']['allowedFiles']) or len(pe['fileQuarantine']['deniedCanaries']) != 5:
        raise ValueError('Primary numerical-file guard differs')
    frame = a['frames'][0]
    expected = {n: a['acquisition'][n]['payloadSHA256'] for n in ['Q', 'K', 'V', 'embedding']}
    expected.update({n: s[key]['payloadSHA256'] for n, key in [('carrierA', 'carrierA'), ('fusedA', 'fusedA'), ('beta', 'beta'), ('modulation', 'modImage'), ('onePlusScale', 'onePlusScale')]})
    if {n: x['fingerprint'] for n, x in frame['sourceFrame']['slots'].items()} != expected or frame['evidence']['metrics']['capturedImageExitPayloadSHA256'] != a['fields']['imageExit']['payloadSHA256']:
        raise ValueError('Typed candidate operand/output binding differs')
    for frame in [frame, reference['frame']]:
        src, dst, ev = frame['sourceFrame'], frame['outputFrame'], frame['evidence']
        if src['frame_id'] != ev['input_frame_id'] or dst['parent_frame_id'] != src['frame_id'] or dst['frame_id'] != ev['output_frame_id'] or dst['program_fingerprint'] != r['requestSHA256']:
            raise ValueError('Typed actual state lineage differs')
    if verification['nativeVSwap']['swappedPayloadSHA256'] != reference['frame']['sourceFrame']['slots']['V']['fingerprint']:
        raise ValueError('Independently reconstructed native V swap differs')
    counts = {}
    for name, category, values in [('queries', 'stage', STAGES), ('operand-queries', 'operand', ['gate', 'up'])]:
        raw = gzip.decompress((p / (name + '.jsonl.gz')).read_bytes())
        old = next(x for x in read(p / 'provenance.json')['losslessStreams'] if x['source'] == name + '.jsonl')
        if hashlib.sha256(raw).hexdigest() != old['sourceSHA256']:
            raise ValueError('Lossless row identity differs')
        rows = [json.loads(line) for line in raw.splitlines()]
        for value in values:
            if [x['imageQueryIndex'] for x in rows if x[category] == value] != list(range(1024)):
                raise ValueError('Complete actual query coverage differs')
        counts[name] = len(rows)
    if counts != {'queries': 7168, 'operand-queries': 2048} or len(report['accounts']) != 28 or len(report['operandAccounts']) != 8:
        raise ValueError('Complete response/operand account differs')
    return {'schema': 'tce-value-innovation-mechanics-v1', 'ownedSourcePins': len(r['ownedSources']), 'typedActs': 2, 'candidateAffineCalls': 33, 'candidateOtherContractions': 3, 'nativeSDPACalls': 2,
            'referenceAffineCalls': len(re['linears']), 'nativeVSwapByteWitness': True, 'inheritedA': reference['inheritedA'], 'queryRecords': 7168, 'operandRecords': 2048,
            'scope': 'Source/seal/complete-byte/input/physical-head-block/typed-lineage/query-coverage mechanics. No effect-size, future-closure, image or general generator certification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['verify', 'publish', 'audit'])
    parser.add_argument('source')
    parser.add_argument('destination', nargs='?')
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.destination) if args.action == 'publish' else globals()[args.action](args.source), indent=2))
