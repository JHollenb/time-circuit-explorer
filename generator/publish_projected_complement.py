"""Authenticate complete projected-cache bytes and publish lossless scalar rows."""
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
from projected_complement import fields, WO, ACTOR_PARAMETERS

TEMPLATE = Path(__file__).with_name('publish_sparse_text_kv.py')


def items(p):
    s = read(p / 'inputs.json')
    values = [s['N'], s['parentCompiler'], s['parentCompilerSeal'], *fields(s['actorInputs'])]
    values += [x for name in ['compiler.json', 'primary.json', 'targets.json'] for x in fields(read(p / name))]
    values += [{'path': str(Path(s['modelRoot']) / n), 'sha256': h} for n, h in s['imageAssets'].items()]
    return values


def verify(root):
    from mrun.client.api import Api
    root, collection, job, r = checked(root)
    p = root / 'result'
    s = read(p / 'inputs.json')
    parent = Api().json('GET', '/api/jobs/' + s['parentJobID'])
    if parent['state'] != 'succeeded' or parent['payload_custody'] != s['parentCustody']:
        raise ValueError('Original source custody changed')
    base = job['result']['diagnostics']['cwd']
    unique = {}
    for item in items(p) + [{'path': base + '/response-result/' + x.name, 'sha256': sha(x)} for x in p.iterdir()]:
        if item['path'] in unique and unique[item['path']]['sha256'] != item['sha256']:
            raise ValueError('Conflicting input/output byte identity')
        unique[item['path']] = item
    spec = {'allowedRoots': [base, s['modelRoot'], parent['result']['diagnostics']['cwd'],
                             '/mnt/big/mdb/research/job-a2048028d7bf/flux-binding-family-result'], 'items': list(unique.values())}
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(NPY_BYTES)],
        input=json.dumps(spec), text=True, capture_output=True, timeout=240)
    if result.returncode:
        raise RuntimeError(result.stderr)
    receipt = json.loads(result.stdout)
    receipt.update(schema='tce-projected-complement-bytes-v1', jobID=job['job_id'], compilerSealSHA256=sha(p / 'compiler-seal.json'),
        primarySealSHA256=sha(p / 'primary-seal.json'), verifierSourceSHA256=sha(__file__), byteTemplateSourceSHA256=sha(TEMPLATE),
        scope='Full original complement/admitted inputs,604MBprojectedcache, own outputs, both historical target sets, original checkpoint and compact records. Complete NPY payload headers/bytes independently authenticated. SSH byte I/O only; no scientific/model computation.')
    save(root / 'retained-array-verification.json', receipt)
    return {'files': len(receipt['files']), 'bytes': receipt['totalBytesVerified'], 'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses'])}


def publish(root, destination):
    root, collection, job, r = checked(root)
    p = root / 'result'
    receipt = read(root / 'retained-array-verification.json')
    if receipt['primarySealSHA256'] != sha(p / 'primary-seal.json') or receipt['compilerSealSHA256'] != sha(p / 'compiler-seal.json') or receipt['verifierSourceSHA256'] != sha(__file__) or receipt['byteTemplateSourceSHA256'] != sha(TEMPLATE):
        raise ValueError('Byte verifier source/release differs')
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    streams = []
    for file in p.iterdir():
        if file.suffix == '.jsonl':
            raw = file.read_bytes()
            target = out / (file.name + '.gz')
            target.write_bytes(gzip.compress(raw, mtime=0))
            if gzip.decompress(target.read_bytes()) != raw:
                raise ValueError('Lossless query transport differs')
            streams.append({'source': file.name, 'sourceSHA256': sha(file), 'sourceBytes': len(raw), 'published': target.name,
                            'publishedSHA256': sha(target), 'publishedBytes': target.stat().st_size})
        else:
            shutil.copyfile(file, out / file.name)
    shutil.copyfile(root / 'retained-array-verification.json', out / 'retained-array-verification.json')
    save(out / 'provenance.json', {'schema': 'tce-projected-complement-publication-v1', 'jobID': job['job_id'], 'requestSHA256': r['requestSHA256'],
        'payloadCustody': collection['custody'], 'schedulerResult': collection['result'], 'completeCollectedInventory': collection['inventory'],
        'losslessStreams': streams, 'scope': 'One fixed-query learned reader composition, paid source/cache/weights and changed finite rounding. No new image, first-prompt or arbitrary-prompt zero-neural claim.'})
    save(out / 'files.json', {x.name: sha(x) for x in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
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
            raise ValueError('Executed source changed')
    for name in ['compiler', 'primary']:
        seal = read(p / (name + '-seal.json'))
        if seal[name + 'SHA256'] != sha(p / (name + '.json')) or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Explicit complete JSON/request-to-seal join differs')
    if a['compilerSealSHA256'] != sha(p / 'compiler-seal.json') or receipt['primarySealSHA256'] != sha(p / 'primary-seal.json') or report['primarySealSHA256'] != sha(p / 'primary-seal.json'):
        raise ValueError('Compiler/actor/target release order differs')
    if receipt['verifierSourceSHA256'] != sha(__file__) or receipt['byteTemplateSourceSHA256'] != sha(TEMPLATE):
        raise ValueError('Complete byte verifier source differs')
    for item in items(p):
        if receipt['files'][item['path']]['sha256'] != item['sha256']:
            raise ValueError('Full byte custody absent')
        if item.get('payloadSHA256') and receipt['npyPayloadWitnesses'][item['path']]['payloadSHA256'] != item['payloadSHA256']:
            raise ValueError('Independent complete NPY payload witness absent')
    if [x['name'] for x in ce['loadedParameters']] != [WO] or set(x['name'] for x in pe['loadedParameters']) != set(ACTOR_PARAMETERS):
        raise ValueError('Selected original parameter closure differs')
    for e, expected, calls, rows in [(ce, 9663676416, 24, 1024), (pe, 855638016, 52, 4)]:
        if sum(x['nominalMACs'] for x in e['linears'] + e['contractions']) != expected or e['nominalMatrixMACs'] != expected or len(e['linears']) != calls:
            raise ValueError('Physical operation census differs')
        heads = [x for x in e['linears'] if x['parameter'] == WO]
        if [x['head'] for x in heads] != list(range(24)) * (1 if e is ce else 2) or any(x['weightShape'] != [3072, 128] or x['inputShape'] != [rows, 128] for x in heads):
            raise ValueError('Physical Wo head-block slicing differs')
        if e['parameterInputRNGClosure'] is not True or e['numericalProfile'] != s['numericalProfile']:
            raise ValueError('Input/parameter/profile closure differs')
        guard = e['fileQuarantine']
        if not set(guard['successfulFileOpenCounts']).issubset(guard['allowedFiles']):
            raise ValueError('Undeclared numerical input read')
    actor_guard = pe['fileQuarantine']
    if s['N']['path'] in actor_guard['allowedFiles'] or s['N']['path'] not in actor_guard['deniedCanaries']:
        raise ValueError('Old unprojected complement was admitted to fresh actor')
    if c['projectedCacheLogicalBytes'] != 604372992 or c['sourceN'] != s['N']:
        raise ValueError('Projected cache/source identity differs')
    if {n: x for n, x in c['actorInputs'].items() if n != 'projectedN'} != s['actorInputs']:
        raise ValueError('Unchanged actual actor inputs differ')
    weights = {x['name']: x['payloadSHA256'] for x in pe['loadedParameters']}
    for arm, actor in a['actors'].items():
        src, dst, ev = actor['sourceFrame'], actor['outputFrame'], actor['evidence']
        expected = {n: x['payloadSHA256'] for n, x in c['actorInputs'].items() if not n.endswith('.rows')}
        expected['textRows'] = c['actorInputs'][arm + '.rows']['payloadSHA256']
        expected.update(Kweight=weights['transformer_blocks.1.attn.add_k_proj.weight'], Vweight=weights['transformer_blocks.1.attn.add_v_proj.weight'],
                        KnormWeight=weights['transformer_blocks.1.attn.norm_added_k.weight'], WoFP64=pe['WoFP64']['payloadSHA256'])
        if {n: v['fingerprint'] for n, v in src['slots'].items()} != expected or src['frame_id'] != ev['input_frame_id'] or dst['parent_frame_id'] != src['frame_id'] or dst['frame_id'] != ev['output_frame_id'] or dst['program_fingerprint'] != r['requestSHA256']:
            raise ValueError('Actual typed actor source/lineage differs')
        if ev['metrics']['capturedPayloadSHA256'] != actor['fields']['R_attention']['payloadSHA256']:
            raise ValueError('Actual typed output capture differs')
    raw = gzip.decompress((p / 'queries.jsonl.gz').read_bytes())
    old = read(p / 'provenance.json')['losslessStreams'][0]
    if hashlib.sha256(raw).hexdigest() != old['sourceSHA256']:
        raise ValueError('Lossless stream bytes differ')
    rows = [json.loads(line) for line in raw.splitlines()]
    if [x['imageQueryIndex'] for x in rows] != list(range(1024)) or any(set(x['references']) != {'family', 'parent'} for x in rows):
        raise ValueError('Complete two-reference actual query coverage differs')
    return {'schema': 'tce-projected-complement-mechanics-v1', 'ownedSourcePins': len(r['ownedSources']), 'typedActors': 2,
        'compilerPhysicalHeadBlocks': 24, 'actorPhysicalHeadBlocks': 48, 'actorKVCalls': 4, 'queryRecords': 1024,
        'fullByteFiles': len(receipt['files']), 'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']),
        'compilerNumericalInput': 'Sealed unprojected complement N and original Wo only', 'actorOldNForbidden': True,
        'scope': 'Source/seal/complete-byte/physical-head-block/typed-input/quarantine/row mechanics; no effect-size or image/general-generator certification.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['verify', 'publish', 'audit'])
    p.add_argument('source')
    p.add_argument('destination', nargs='?')
    args = p.parse_args()
    print(json.dumps(publish(args.source, args.destination) if args.action == 'publish' else globals()[args.action](args.source), indent=2))
