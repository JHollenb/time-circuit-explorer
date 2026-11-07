"""Full byte custody, clock-view checks and lossless two-recipient publication."""
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
    s, t, a = [read(p / n) for n in ['inputs.json', 'targets.json', 'primary.json']]
    groups = [s['fusedA'], s['embeddings'], s['carriers'], a['shared'], *a['cells'].values(), *t['cells'].values(), t['residuals'], t['priorConditioned'], *read(p / 'response-fields.json').values()]
    return [s['modulation'], s['onePlusScale'], *s['parentRecords'].values(), *[{'path': str(Path(s['modelRoot']) / n), 'sha256': h} for n, h in s['imageAssets'].items()], *[x for g in groups for x in g.values()]]


CLOCK_REMOTE = REMOTE.split('print(json.dumps')[0] + '''import ast,struct
p=Path(s['clock']['path']);views={}
if str(p) not in inventory:raise ValueError('Unverified modulation parent')
with p.open('rb') as f:
 if f.read(8)!=b'\\x93NUMPY\\x01\\x00':raise ValueError('Unexpected clock NPY version')
 n=struct.unpack('<H',f.read(2))[0];head=ast.literal_eval(f.read(n).decode('latin1').strip());offset=f.tell()
 if head!={'descr':'<u2','fortran_order':False,'shape':(1,18432)}:raise ValueError('Unexpected clock geometry')
 for name,lo,hi in [('shift',9216,12288),('gate',15360,18432)]:
  f.seek(offset+2*lo);b=f.read(2*(hi-lo));h=hashlib.sha256(b).hexdigest()
  if len(b)!=6144 or h!=s['clock'][name]:raise ValueError('Clock semantic view differs')
  views[name]={'parentPath':str(p),'lastAxisSlice':[lo,hi],'payloadSHA256':h,'bytesReadAgain':len(b)}
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'clockViews':views,'clockViewBytesReadAgain':12288}))
'''


def verify(root):
    root, c, j, r = checked(root)
    p = root / 'result'
    s, a = read(p / 'inputs.json'), read(p / 'primary.json')
    seal = sha(p / 'primary-seal.json')
    if read(p / 'report.json')['primarySealSHA256'] != seal or read(p / 'primary-seal.json')['primarySHA256'] != sha(p / 'primary.json'):
        raise ValueError('Primary release differs')
    base = j['result']['diagnostics']['cwd']
    unique = {}
    for item in items(p) + [{'path': base + '/response-result/' + x.name, 'sha256': sha(x)} for x in p.iterdir()]:
        if item['path'] in unique and unique[item['path']]['sha256'] != item['sha256']:
            raise ValueError('Conflicting byte identity')
        unique[item['path']] = item
    spec = {'allowedRoots': [base, s['modelRoot'], s['modulation']['path'], *s['protectedRoots']], 'items': list(unique.values()), 'clock': {'path': s['modulation']['path'], **{n: a['shared'][n]['payloadSHA256'] for n in ['shift', 'gate']}}}
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(CLOCK_REMOTE)], input=json.dumps(spec), text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr)
    value = json.loads(result.stdout)
    value.update(schema='tce-normalization-innovation-bytes-v1', jobID=j['job_id'], primarySealSHA256=seal, verifierSourceSHA256=sha(__file__), scope='Complete supplied/generated/held native/control fields, original assets and result bytes. Two clock views independently byte-sliced;12288rereadbytes are not unique storage. SSH byte/metadata only; complete arrays stay near compute.')
    save(root / 'retained-array-verification.json', value)
    return {'files': len(value['files']), 'bytes': value['totalBytesVerified'], 'clockViewBytesReadAgain': 12288}


def publish(root, destination):
    root, c, j, r = checked(root)
    p = root / 'result'
    v = read(root / 'retained-array-verification.json')
    if v['primarySealSHA256'] != sha(p / 'primary-seal.json') or v['verifierSourceSHA256'] != sha(__file__):
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
    save(out / 'provenance.json', {'schema': 'tce-normalization-innovation-publication-v1', 'jobID': j['job_id'], 'requestSHA256': r['requestSHA256'], 'payloadCustody': c['custody'], 'schedulerResult': c['result'], 'completeCollectedInventory': c['inventory'], 'losslessStreams': streams, 'scope': 'Both supplied B fused operands omitted. Original A baselines/embeddings/carriers, FP64 learned contractions and two native down paths paid. No cheap first prompt, own-state/image or strictzero-neural claim.'})
    save(out / 'files.json', {x.name: sha(x) for x in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    from normalization_innovation import IN, DOWN, STAGES
    from source_response import mask
    p = Path(source)
    r, s, t, a, e, report, v = [read(p / n) for n in ['request.json', 'inputs.json', 'targets.json', 'primary.json', 'execution.json', 'report.json', 'retained-array-verification.json']]
    for name, digest in read(p / 'files.json').items():
        if sha(p / name) != digest:
            raise ValueError('Publication member differs')
    if identity({k: x for k, x in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request identity differs')
    for name in r['ownedSources']:
        if sha(Path(__file__).parent / name) != r['sourceHashes'][name]:
            raise ValueError('Executed source differs')
    seal = read(p / 'primary-seal.json')
    if seal['requestSHA256'] != r['requestSHA256'] or seal['primarySHA256'] != sha(p / 'primary.json') or report['primarySealSHA256'] != sha(p / 'primary-seal.json') or v['primarySealSHA256'] != sha(p / 'primary-seal.json') or v['verifierSourceSHA256'] != sha(__file__):
        raise ValueError('Primary/custody seal chain differs')
    for item in items(p):
        if v['files'][item['path']]['sha256'] != item['sha256']:
            raise ValueError('Complete byte custody missing')
    if e['loadedParameters'] != s['selectedParameters'] or {x['name']: x['shape'] for x in e['loadedParameters']} != {IN: [18432, 3072], DOWN: [3072, 9216]} or sum(x['bytes'] for x in e['loadedParameters']) != 169869312 or e['parameterInputRNGClosure'] is not True or e['numericalProfile'] != s['numericalProfile']:
        raise ValueError('Two-matrix/input/profile closure differs')
    expected = [(IN, 'torch.float64', 'cpu', [1, 1024, 3072]), (IN, 'torch.float64', 'cpu', [1, 1, 3072]), (DOWN, 'torch.bfloat16', 'cuda', [1, 1024, 9216]), (DOWN, 'torch.bfloat16', 'cuda', [1, 1024, 9216])]
    if [(x['parameter'], x['dtype'], x['device'], x['inputShape']) for x in e['linears']] != expected or sum(x['nominalMACs'] for x in e['linears']) != 116020740096 or report['nominalAffineMACs'] != 116020740096:
        raise ValueError('Four-affine physical geometry differs')
    if set(a['cells']) != {'R0', 'R1'} or len(a['frames']) != 2 or len(e['fileQuarantine']['deniedCanaries']) != 6 or not set(e['fileQuarantine']['successfulFileOpenCounts']).issubset(e['fileQuarantine']['allowedFiles']):
        raise ValueError('Receiver/primary target access closure differs')
    names = {'residualA', 'residualB', 'sA', 'sB', 'alpha', 'rho', 'prequantFP64', 'fused', 'quantizationDifference', 'activation', 'product', 'mlpWrite', 'gatedMLP', 'residualCommit', 'imageExit'}
    for receiver, cell in a['cells'].items():
        if set(cell) != names:
            raise ValueError('Complete candidate fields differ')
        for suffix in ['A', 'B']:
            if cell['residual' + suffix]['payloadSHA256'] != t['residuals'][receiver + suffix]['payloadSHA256']:
                raise ValueError('Native residual prefix bits differ')
        if cell['residualCommit']['payloadSHA256'] != cell['residualB']['payloadSHA256']:
            raise ValueError('Final residual differs from candidate statistics input')
    if a['shared']['scale']['payloadSHA256'] != s['onePlusScale']['payloadSHA256']:
        raise ValueError('Actual BF16 one-plus-scale binding differs')
    for name in ['shift', 'gate']:
        if v['clockViews'][name]['payloadSHA256'] != a['shared'][name]['payloadSHA256']:
            raise ValueError('Clock semantic byte binding differs')
    for f in a['frames']:
        receiver = f['receiver']
        src, dst, ev = f['sourceFrame'], f['outputFrame'], f['evidence']
        expected_fp = {'fusedA': s['fusedA'][receiver]['payloadSHA256'], 'embedding': s['embeddings'][receiver]['payloadSHA256'], 'carrierA': s['carriers']['A']['payloadSHA256'], 'carrierB': s['carriers']['B']['payloadSHA256'], **{n: a['shared'][n]['payloadSHA256'] for n in ['gate', 'J', 'beta']}}
        if {k: x['fingerprint'] for k, x in src['slots'].items()} != expected_fp or src['frame_id'] != ev['input_frame_id'] or dst['frame_id'] != ev['output_frame_id'] or dst['parent_frame_id'] != src['frame_id'] or ev['act'] != 'tce.normalization_innovation_exit:v1' or dst['program_fingerprint'] != r['requestSHA256'] or ev['metrics']['capturedImageExitPayloadSHA256'] != a['cells'][receiver]['imageExit']['payloadSHA256']:
            raise ValueError('Typed source/shared/output lineage differs')
    prov = read(p / 'provenance.json')
    streams = {}
    for name in ['queries', 'operand-queries']:
        raw = gzip.decompress((p / (name + '.jsonl.gz')).read_bytes())
        receipt = next(x for x in prov['losslessStreams'] if x['source'] == name + '.jsonl')
        if hashlib.sha256(raw).hexdigest() != receipt['sourceSHA256']:
            raise ValueError('Lossless stream identity differs')
        streams[name] = [json.loads(line) for line in raw.splitlines()]
    for receiver in ['R0', 'R1']:
        for key, categories, rows in [('stage', STAGES, streams['queries']), ('operand', ['gate', 'up'], streams['operand-queries'])]:
            for category in categories:
                chosen = [x for x in rows if x['receiver'] == receiver and x[key] == category]
                if [x['imageQueryIndex'] for x in chosen] != list(range(1024)) or [x['sourceNonlocal'] for x in chosen] != mask('bottom-right').tolist():
                    raise ValueError('Actual query coverage differs')
    if report['queryRecords'] != 10240 or report['operandRecords'] != 4096 or len(streams['queries']) != 10240 or len(streams['operand-queries']) != 4096 or len(report['accounts']) != 20 or len(report['operandAccounts']) != 8 or len(report['prefixes']) != 4:
        raise ValueError('Complete account coverage differs')
    return {'schema': 'tce-normalization-innovation-mechanics-v1', 'ownedSourcePins': len(r['ownedSources']), 'selectedMatrices': 2, 'affineCalls': 4, 'typedActs': 2, 'nativeResidualPrefixesExact': 4, 'responseRecords': 10240, 'operandRecords': 4096, 'primarySealSHA256': sha(p / 'primary-seal.json'), 'scope': 'Source/seal/full-byte/clock/input/shared/physical-work/typed-prefix/complete-query mechanics. No effect-size gate, whole-future parity or generator certification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['verify', 'publish', 'audit'])
    parser.add_argument('source')
    parser.add_argument('destination', nargs='?')
    args = parser.parse_args()
    print(json.dumps(publish(args.source, args.destination) if args.action == 'publish' else globals()[args.action](args.source), indent=2))
