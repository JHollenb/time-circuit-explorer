"""Full-byte publication and independent input/own-state/work joins."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read, save, sha, identity
from projected_complement import fields
from publish_source_native import checked
from publish_source_complement import BYTES as PREVIOUS_BYTES, check_frame
from propagated_context import RETURN_PARAMETERS, NEXT_PARAMETERS
from image_dual_worker import KEY_PARAMETERS
from image_dual_analysis import NAMES

TEMPLATES = ['publish_source_complement.py', 'publish_sparse_text_kv.py']
BYTES = PREVIOUS_BYTES.rsplit('print(json.dumps', 1)[0] + r'''
prefix=s['textKeySlice'];original,ho=raw(prefix['source']);selected,hs=raw(prefix)
if ho['shape']!=(1,1536,24,128) or hs['shape']!=(1,512,24,128) or selected!=original[:512*24*128*2]:raise ValueError('Independent text-prefix byte identity differs')
source=Path(prefix['source']['path']).read_bytes();offset=prefix['sourceDataOffset']
if hashlib.sha256(source[:offset]).hexdigest()!=prefix['sourceHeaderSHA256'] or len(original)!=prefix['sourcePayloadBytes'] or offset!=prefix['selectedByteOffset'] or len(selected)!=prefix['selectedPayloadBytes']:raise ValueError('Original header/offset/length binding differs')
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),
 'npyPayloadWitnesses':witnesses,'bankSpliceWitnesses':joins,'spliceBytesReadAgain':reread,
 'textPrefixWitness':{'byteExact':True,'sourceSHA256':prefix['source']['sha256'],
 'sourceHeaderSHA256':prefix['sourceHeaderSHA256'],'payloadSHA256':prefix['payloadSHA256'],'bytes':len(selected)}}))
'''


def items(p):
    s = read(p / 'inputs.json')
    result = [*s['parentRecords'].values(), *s['latestParentRecords'].values()]
    result += [{'path': n, 'sha256': h} for n, h in s['sourceMetadataFiles'].items()]
    result += list(fields({'selected': s['selected'], 'banks': s['banks'], 'textSlice': s['textKeySlice']}))
    for name in ['targets.json', 'compiler.json', 'actor.json', 'reference.json', 'report.json']:
        result += list(fields(read(p / name)))
    result += [{'path': str(Path(s['modelRoot']) / n), 'sha256': h} for n, h in s['imageAssets'].items()]
    return result


def verify(root):
    from mrun.client.api import Api
    root, collection, job, request = checked(root);p = root / 'result';s = read(p / 'inputs.json')
    roots = [job['result']['diagnostics']['cwd'], s['modelRoot'], str(Path(s['textKeySlice']['path']).parent)]
    roots += [str(Path(cut['cutFile']['path']).parent.parent) for cut in s['sourceCuts'].values()]
    for key, custody in [('parentJobID', 'parentCustody'), ('latestParentJobID', 'latestParentCustody'), ('sourceJobID', 'sourceCustody')]:
        parent = Api().json('GET', '/api/jobs/' + s[key])
        if parent['state'] != 'succeeded' or parent['payload_custody'] != s[custody]:
            raise ValueError('Source/parent scheduler custody differs')
        roots.append(parent['result']['diagnostics']['cwd'])
    base = job['result']['diagnostics']['cwd'];unique = {}
    for x in items(p) + [{'path': base + '/response-result/' + n.name, 'sha256': sha(n)} for n in p.iterdir()]:
        if x['path'] in unique and unique[x['path']]['sha256'] != x['sha256']:
            raise ValueError('Conflicting full byte identities')
        unique[x['path']] = x
    actor = read(p / 'actor.json')['read']['fields']
    splices = [{'arm': 'actor', 'field': n, 'support': request['support'], 'selected': s['selected'][n],
                'bank': bank, 'hybrid': actor[field]} for n, bank, field in [
                    ('K', s['textKeySlice'], 'textK_hybrid'), ('V', s['banks']['Q']['V'], 'V_hybrid')]]
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(BYTES)],
        input=json.dumps({'allowedRoots': roots, 'items': list(unique.values()), 'spliceChecks': splices, 'textKeySlice': s['textKeySlice']}),
        text=True, capture_output=True, timeout=240)
    if result.returncode:
        raise RuntimeError(result.stderr)
    receipt = json.loads(result.stdout);receipt.update(schema='tce-image-dual-bytes-v1', jobID=job['job_id'],
        verifierSourceSHA256=sha(__file__), templateSourceHashes={n: sha(Path(__file__).with_name(n)) for n in TEMPLATES},
        phaseSeals={n: sha(p / (n + '-seal.json')) for n in ['compiler', 'actor', 'reference']})
    save(root / 'retained-array-verification.json', receipt)
    return {'files': len(receipt['files']), 'bytes': receipt['totalBytesVerified'],
            'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']), 'hybridSpliceWitnesses': 2, 'textPrefixWitnesses': 1}


def publish(root, destination):
    root, collection, job, request = checked(root);p = root / 'result';receipt = read(root / 'retained-array-verification.json')
    if receipt['verifierSourceSHA256'] != sha(__file__) or receipt['templateSourceHashes'] != {n: sha(Path(__file__).with_name(n)) for n in TEMPLATES}:
        raise ValueError('Byte verifier changed')
    out = Path(destination);out.mkdir(parents=True, exist_ok=False);streams = []
    for member in p.iterdir():
        if member.suffix == '.jsonl':
            raw = member.read_bytes();target = out / (member.name + '.gz');target.write_bytes(gzip.compress(raw, mtime=0))
            streams.append({'source': member.name, 'sourceSHA256': sha(member), 'published': target.name, 'publishedSHA256': sha(target)})
        else:
            shutil.copyfile(member, out / member.name)
    shutil.copyfile(root / 'retained-array-verification.json', out / 'retained-array-verification.json')
    save(out / 'provenance.json', {'schema': 'tce-image-dual-publication-v1', 'jobID': job['job_id'], 'requestSHA256': request['requestSHA256'],
        'payloadCustody': collection['custody'], 'schedulerResult': collection['result'], 'completeCollectedInventory': collection['inventory'],
        'losslessStreams': streams, 'scope': 'Conditional query-specific learned computation; compiler/reference/native ancestry paid, no image or arbitrary-prompt zero-neural result.'})
    save(out / 'files.json', {x.name: sha(x) for x in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    p = Path(source);r, s, report, receipt = [read(p / n) for n in ['request.json', 'inputs.json', 'report.json', 'retained-array-verification.json']]
    if identity({k: v for k, v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request identity differs')
    for n, h in read(p / 'files.json').items():
        if sha(p / n) != h:
            raise ValueError('Immutable publication member differs: ' + n)
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n)) != r['sourceHashes'][n]:
            raise ValueError('Executed source differs: ' + n)
    if receipt['verifierSourceSHA256'] != sha(__file__) or receipt['templateSourceHashes'] != {n: sha(Path(__file__).with_name(n)) for n in TEMPLATES}:
        raise ValueError('Byte verifier/template differs')
    for x in items(p):
        if receipt['files'][x['path']]['sha256'] != x['sha256'] or (x.get('payloadSHA256') and receipt['npyPayloadWitnesses'][x['path']]['payloadSHA256'] != x['payloadSHA256']):
            raise ValueError('Full consumed/emitted byte custody absent')
    arms = {n: read(p / (n + '.json')) for n in ['compiler', 'actor', 'reference']}
    executions = {n: read(p / (n + '-execution.json')) for n in arms};acts = []
    for phase, arm in arms.items():
        seal = read(p / (phase + '-seal.json'));execution = executions[phase]
        if seal[phase + 'SHA256'] != sha(p / (phase + '.json')) or seal['requestSHA256'] != r['requestSHA256'] or receipt['phaseSeals'][phase] != sha(p / (phase + '-seal.json')):
            raise ValueError('Complete phase release differs')
        actual_weights = {x['name']: x['payloadSHA256'] for x in execution['loadedParameters']}
        if set(actual_weights) != set(RETURN_PARAMETERS + NEXT_PARAMETERS if phase == 'actor' else KEY_PARAMETERS):
            raise ValueError('Actual original map hydration differs')
        guard = execution['fileQuarantine'];denied, allowed = set(guard['deniedCanaries']), set(guard['allowedFiles'])
        if not set(guard['successfulFileOpenCounts']).issubset(allowed) or not execution['parameterInputRNGClosure']:
            raise ValueError('Actual numerical-read/parameter/RNG closure differs')
        if phase != 'reference':
            required = {x['path'] for x in fields(read(p / 'targets.json'))} | {s['banks'][role]['K']['path'] for role in ['P', 'Q']}
            if not required.issubset(denied) or required & allowed:
                raise ValueError('Compiler/actor targets/full keys were accessible')
        expected_work = r['plannedMatrixMACs'][phase]
        if execution['nominalMatrixMACs'] != expected_work or sum(x['nominalMACs'] for x in execution['operations']) != expected_work:
            raise ValueError('Physical operation sum differs')
        if phase == 'actor' and len([x for x in execution['operations'] if x['operation'] == 'original-affine']) != 4:
            raise ValueError('Actual receiving affine count differs')
        weights = actual_weights
        for name, act in arm.items():
            if not isinstance(act, dict) or 'sourceFrame' not in act:
                continue
            acts.append(act)
            # Bind every declared input to external evidence or an earlier emitted field.
            expected = {}
            for n in act['sourceFrame']['slots']:
                if n in weights:
                    expected[n] = weights[n]
            if name == 'modln':
                cut = s['sourceCuts']['P' if phase == 'compiler' else 'Q']
                declared = {x['key']['value']: x['value'] for x in cut['cut']['manifest']['payload']['items']}
                context = {x['key']['value']: x['value'] for x in declared['context']['items']}
                expected.update(image=declared['image']['sha256'], imgMod=context['img_mod']['sha256'])
            elif name == 'compile':
                expected.update(x=arm['modln']['fields']['x']['payloadSHA256'], Q=s['selected']['Q']['payloadSHA256'],
                                **{n: arm['rope'][n]['payloadSHA256'] for n in ['cosine', 'sine']})
            elif name == 'read':
                expected.update(x=arm['modln']['fields']['x']['payloadSHA256'],
                    **{n: arms['compiler']['compile']['fields'][n]['payloadSHA256'] for n in ['duals', 'pDenominator', 'groupIDs']},
                    Q=s['selected']['Q']['payloadSHA256'], selectedK=s['selected']['K']['payloadSHA256'], selectedV=s['selected']['V']['payloadSHA256'],
                    textK=s['textKeySlice']['payloadSHA256'], bankV=s['banks']['Q']['V']['payloadSHA256'])
            elif name == 'source':
                expected.update(H=arm['read']['fields']['H']['payloadSHA256'], **{n: s['selected'][n]['payloadSHA256'] for n in ['R', 'txtMod']})
            elif name == 'next_query':
                expected.update(rows=arm['source']['fields']['T']['payloadSHA256'], **{n: s['selected'][n]['payloadSHA256'] for n in ['txtMod', 'ropeCos', 'ropeSin']})
            elif name == 'reference':
                expected.update(x=arms['actor']['modln']['fields']['x']['payloadSHA256'], Q=s['selected']['Q']['payloadSHA256'],
                    nativeKeys=s['banks']['Q']['K']['payloadSHA256'], textLogits=arms['actor']['read']['fields']['textLogits']['payloadSHA256'],
                    bankV=arms['actor']['read']['fields']['V_hybrid']['payloadSHA256'],
                    **{n: arms['compiler']['rope'][n]['payloadSHA256'] for n in ['cosine', 'sine']})
            else:
                raise ValueError('Unknown typed computation')
            check_frame(act, expected, r['requestSHA256'])
            for n in act['evidence']['metrics']['capturedPayloads']:
                x = act['fields'][n]
                if act['outputFrame']['slots'][n]['abi'] != {'shape': x['shape'], 'dtype': x['logicalDtype']}:
                    raise ValueError('Explicit output ABI differs')
    a, c = arms['actor'], arms['compiler']
    if a['compilerSealSHA256'] != sha(p / 'compiler-seal.json') or arms['reference']['actorSealSHA256'] != sha(p / 'actor-seal.json'):
        raise ValueError('Compiler→actor→post-seal reference join differs')
    if a['source']['sourceFrame']['metadata']['headOutputFrame'] != a['read']['outputFrame']['frame_id'] or a['next_query']['sourceFrame']['metadata']['sourceOutputFrame'] != a['source']['outputFrame']['frame_id']:
        raise ValueError('Actual own-H→own-T→own-query join differs')
    if report['totalNewMatrixMACs'] != 21359493120 or sum(x['nominalMatrixMACs'] for x in executions.values()) != report['totalNewMatrixMACs']:
        raise ValueError('Complete matrix work differs')
    if receipt['textPrefixWitness']['byteExact'] is not True or len(receipt['bankSpliceWitnesses']) != 2:
        raise ValueError('Independent text-prefix/hybrid witnesses absent')
    raw = gzip.decompress((p / 'observations.jsonl.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest() != read(p / 'provenance.json')['losslessStreams'][0]['sourceSHA256']:
        raise ValueError('Lossless observations differ')
    rows = [json.loads(line) for line in raw.splitlines()]
    expected = {(n, 'whole-row', row, None) for n in NAMES for row in r['support']}
    expected |= {(n, 'query-head', row, h) for n in ['nextQProjected', 'nextQNormalized', 'nextQ'] for row in r['support'] for h in range(24)}
    expected |= {('H_policy', 'query-head-policy', row, h) for row in r['support'] for h in range(24)}
    actual = [(x['field'], x['panel'], x['row'], x.get('head')) for x in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Complete unique observation addresses differ')
    return {'schema': 'tce-image-dual-mechanics-v1', 'ownedSourcePins': len(r['ownedSources']),
        'typedActs': len(acts), 'explicitOutputABIs': sum(len(a['evidence']['metrics']['capturedPayloads']) for a in acts),
        'ownHeadStateQueryJoins': 2, 'observations': len(rows), 'textPrefixWitnesses': 1, 'hybridSpliceWitnesses': 2,
        'fullByteFiles': len(receipt['files']), 'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']),
        'scope': 'Source/seal/byte/typed-own-state/work/quarantine mechanics. No effect-size gate, semantic/image/first-prompt saving or zero-neural certification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('action', choices=['verify', 'publish', 'audit']);parser.add_argument('source');parser.add_argument('destination', nargs='?');args = parser.parse_args()
    print(json.dumps(publish(args.source, args.destination) if args.action == 'publish' else globals()[args.action](args.source), indent=2))
