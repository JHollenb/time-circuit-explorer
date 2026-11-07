"""Complete bytes, original-source joins and lossless sparse-reader publication."""
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
from sparse_text_kv_worker import fields

NPY_BYTES = REMOTE.split('print(json.dumps')[0] + '''
import ast,struct
witnesses={}
for item in s['items']:
 if not item.get('payloadSHA256'):continue
 p=Path(item['path'])
 if p.suffix!='.npy':raise ValueError('Expected NPY payload witness')
 with p.open('rb') as f:
  magic=f.read(8)
  if magic[:6]!=bytes([147,78,85,77,80,89]) or magic[6] not in (1,2,3):raise ValueError('Unexpected NPY magic/version')
  n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0]
  header=ast.literal_eval(f.read(n).decode('utf8' if magic[6]==3 else 'latin1').strip())
  if header['fortran_order'] or list(header['shape'])!=item['shape']:raise ValueError('NPY geometry differs')
  h=hashlib.sha256();size=0
  for b in iter(lambda:f.read(1<<20),b''):h.update(b);size+=len(b)
 if h.hexdigest()!=item['payloadSHA256']:raise ValueError('NPY payload descriptor differs')
 witnesses[str(p)]={'payloadSHA256':h.hexdigest(),'payloadBytes':size,'header':header}
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses}))
'''


def items(p):
    source, meta = read(p / 'inputs.json'), read(p / 'family-metadata.json')
    # Original pretrained artifacts are retrieval recipes, not retained run arrays.
    values = list(fields(source['writerPorts'])) + [source['image']]
    values += [x for n in ['targets.json', 'compiler.json', 'primary.json'] for x in fields(read(p / n))]
    values += [{'path': str(Path(source['modelRoot']) / n), 'sha256': h} for n, h in source['imageAssets'].items()]
    values += [{'path': n, 'sha256': h} for n, h in {**meta['files'], **source['sourceMetadataFiles']}.items()]
    return values


def verify(root):
    from mrun.client.api import Api
    from sparse_text_kv_study import FAMILY
    root, collection, job, request = checked(root)
    p = root / 'result'
    source, meta = read(p / 'inputs.json'), read(p / 'family-metadata.json')
    parent = Api().json('GET', '/api/jobs/' + source['parentJobID'])
    if parent['state'] != 'succeeded' or parent['payload_custody'] != source['parentCustody']:
        raise ValueError('Original source custody changed')
    manifest = read(FAMILY / 'fresh-family-02/compact/files.json')['files']
    original_root = source['sourceRootCorrection']['actual']
    joins = []
    for file, digest in {**meta['files'], **source['sourceMetadataFiles']}.items():
        if file.startswith(original_root + '/'):
            relative = file[len(original_root) + 1:]
            if manifest[relative]['sha256'] != digest:
                raise ValueError('Original family manifest join differs: ' + relative)
            joins.append(relative)
    unique = {}
    base = job['result']['diagnostics']['cwd']
    values = items(p) + [{'path': base + '/response-result/' + x.name, 'sha256': sha(x)} for x in p.iterdir()]
    for x in values:
        if x['path'] in unique and unique[x['path']]['sha256'] != x['sha256']:
            raise ValueError('Conflicting byte identity')
        unique[x['path']] = x
    spec = {'allowedRoots': [base, source['modelRoot'], original_root, *meta['files']], 'items': list(unique.values())}
    proc = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(NPY_BYTES)],
        input=json.dumps(spec), text=True, capture_output=True, timeout=240)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    receipt = json.loads(proc.stdout)
    receipt.update(schema='tce-sparse-text-kv-bytes-v1', jobID=job['job_id'], primarySealSHA256=sha(p / 'primary-seal.json'),
        compilerSealSHA256=sha(p / 'compiler-seal.json'), verifierSourceSHA256=sha(__file__), originalManifestJoins=joins,
        scope='Full supplied source writer fields, state pages/blobs, target arrays, produced cache/actor/endpoint arrays, selected-weight source archive and compact results. SSH byte I/O only; no model or scientific computation.')
    save(root / 'retained-array-verification.json', receipt)
    return {'files': len(receipt['files']), 'bytes': receipt['totalBytesVerified'], 'originalManifestJoins': len(joins)}


def publish(root, destination):
    root, collection, job, request = checked(root)
    p = root / 'result'
    receipt = read(root / 'retained-array-verification.json')
    if receipt['primarySealSHA256'] != sha(p / 'primary-seal.json') or receipt['compilerSealSHA256'] != sha(p / 'compiler-seal.json') or receipt['verifierSourceSHA256'] != sha(__file__):
        raise ValueError('Byte verifier seal/source differs')
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    streams = []
    for file in p.iterdir():
        if file.suffix == '.jsonl':
            raw = file.read_bytes()
            target = out / (file.name + '.gz')
            target.write_bytes(gzip.compress(raw, mtime=0))
            if gzip.decompress(target.read_bytes()) != raw:
                raise ValueError('Lossless publication differs')
            streams.append({'source': file.name, 'sourceSHA256': sha(file), 'sourceBytes': len(raw), 'published': target.name,
                            'publishedSHA256': sha(target), 'publishedBytes': target.stat().st_size})
        else:
            shutil.copyfile(file, out / file.name)
    shutil.copyfile(root / 'retained-array-verification.json', out / 'retained-array-verification.json')
    save(out / 'provenance.json', {'schema': 'tce-sparse-text-kv-publication-v1', 'jobID': job['job_id'], 'requestSHA256': request['requestSHA256'],
        'payloadCustody': collection['custody'], 'schedulerResult': collection['result'], 'completeCollectedInventory': collection['inventory'],
        'losslessStreams': streams, 'scope': 'One retrospective fixed-query sparse K/V update; source/cache construction and original maps paid. No complete image or arbitrary-prompt zero-neural serving.'})
    save(out / 'files.json', {x.name: sha(x) for x in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    from sparse_text_kv import ACTOR_PARAMETERS, PREFIX
    p = Path(source)
    r, s, c, a, ce, pe, report, receipt = [read(p / n) for n in ['request.json', 'inputs.json', 'compiler.json', 'primary.json',
        'compiler-execution.json', 'primary-execution.json', 'report.json', 'retained-array-verification.json']]
    if identity({k: v for k, v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request identity differs')
    for name, digest in read(p / 'files.json').items():
        if sha(p / name) != digest:
            raise ValueError('Published member changed')
    for name in r['ownedSources']:
        if sha(Path(__file__).parent / name) != r['sourceHashes'][name]:
            raise ValueError('Frozen executed source differs')
    if read(p / 'primary-seal.json')['primarySHA256'] != sha(p / 'primary.json') or read(p / 'compiler-seal.json')['compilerSHA256'] != sha(p / 'compiler.json'):
        raise ValueError('Complete state seals differ')
    if a['compilerSealSHA256'] != sha(p / 'compiler-seal.json') or report['primarySealSHA256'] != sha(p / 'primary-seal.json') or receipt['primarySealSHA256'] != sha(p / 'primary-seal.json'):
        raise ValueError('Compiler/actor/scoring release chain differs')
    if receipt['verifierSourceSHA256'] != sha(__file__):
        raise ValueError('Verifier source differs')
    for item in items(p):
        if receipt['files'][item['path']]['sha256'] != item['sha256']:
            raise ValueError('Full byte verification absent')
        if item.get('payloadSHA256') and receipt['npyPayloadWitnesses'][item['path']]['payloadSHA256'] != item['payloadSHA256']:
            raise ValueError('Independent original NPY payload witness absent')
    if set(x['name'] for x in pe['loadedParameters']) != set(ACTOR_PARAMETERS) or len(ce['loadedParameters']) != 8:
        raise ValueError('Selected parameter boundary differs')
    for e in [ce, pe]:
        if e['numericalProfile'] != s['numericalProfile'] or e['parameterInputRNGClosure'] is not True:
            raise ValueError('Parameter/profile closure differs')
        guard = e['fileQuarantine']
        if not set(guard['successfulFileOpenCounts']).issubset(guard['allowedFiles']):
            raise ValueError('Undeclared numerical source read')
    if len(ce['linears']) != 5 or len(pe['linears']) != 6 or ce['nominalMACs'] != 48318382080 or pe['nominalMACs'] != 19528679424 or report['totalNewMACs'] != 67847061504:
        raise ValueError('Physical computation ledger differs')
    for line in pe['linears']:
        rows = 1024 if line['parameter'] == PREFIX + 'to_out.0.weight' else 4
        if line['inputShape'] != [1, rows, 3072] or line['weightShape'] != [3072, 3072]:
            raise ValueError('Actual projection geometry differs')
    if c['cacheLogicalBytes'] != 25559040 or c['supportResolution'] != s['supportResolution']:
        raise ValueError('Cache/address identity differs')
    actor_guard = pe['fileQuarantine']
    if any(x['path'] in actor_guard['allowedFiles'] for x in fields(s['writerPorts'])):
        raise ValueError('Full raw source admitted to fresh actor')
    weights = {x['name']: x['payloadSHA256'] for x in pe['loadedParameters']}
    weight_slots = {'Kweight': 'add_k_proj', 'Vweight': 'add_v_proj', 'Woweight': 'to_out.0', 'KnormWeight': 'norm_added_k'}
    for arm, actor in a['actors'].items():
        source, after, evidence = actor['sourceFrame'], actor['outputFrame'], actor['evidence']
        expected = {n: x['payloadSHA256'] for n, x in c['actorInputs'].items() if not n.endswith('.rows')}
        expected['textRows'] = c['actorInputs'][arm + '.rows']['payloadSHA256']
        expected.update({n: weights[PREFIX + name + '.weight'] for n, name in weight_slots.items()})
        if {n: x['fingerprint'] for n, x in source['slots'].items()} != expected:
            raise ValueError('Typed actor inputs differ from admitted values')
        if source['frame_id'] != evidence['input_frame_id'] or after['frame_id'] != evidence['output_frame_id'] or after['parent_frame_id'] != source['frame_id'] or after['program_fingerprint'] != r['requestSHA256']:
            raise ValueError('Typed actor lineage differs')
        if evidence['metrics']['capturedPayloadSHA256'] != actor['fields']['R_attention']['payloadSHA256']:
            raise ValueError('Typed actual output differs')
    raw = gzip.decompress((p / 'queries.jsonl.gz').read_bytes())
    stream = read(p / 'provenance.json')['losslessStreams'][0]
    if hashlib.sha256(raw).hexdigest() != stream['sourceSHA256']:
        raise ValueError('Lossless row bytes differ')
    rows = [json.loads(line) for line in raw.splitlines()]
    if [x['imageQueryIndex'] for x in rows] != list(range(1024)) or report['queryRecords'] != 1024:
        raise ValueError('Complete actual query coverage differs')
    targets = read(p / 'targets.json')
    text_words = {arm: c['diagnostics'][arm + '.text']['payloadSHA256'] == targets['text'][arm]['payloadSHA256'] for arm in ['QQQ', 'PQP']}
    return {'schema': 'tce-sparse-text-kv-mechanics-v1', 'ownedSourcePins': len(r['ownedSources']), 'typedActors': 2,
        'compilerAffines': 5, 'actorAffines': 6, 'queryRecords': 1024, 'fullByteFiles': len(receipt['files']),
        'sourceRootCorrection': s['sourceRootCorrection'], 'originalManifestJoins': len(receipt['originalManifestJoins']),
        'textEndpointBF16WordsExact': text_words, 'independentNPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']),
        'scope': 'Source/seal/full-byte/actual-fourrow-geometry/typed-input/quarantine/query mechanics; no scientific quality or image inheritance gate.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['verify', 'publish', 'audit'])
    p.add_argument('source')
    p.add_argument('destination', nargs='?')
    args = p.parse_args()
    print(json.dumps(publish(args.source, args.destination) if args.action == 'publish' else globals()[args.action](args.source), indent=2))
