"""Complete byte, typed own-norm/own-state and source-only compiler evidence."""
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
from publish_source_complement import check_frame
from publish_image_dual import BYTES
from propagated_context import RETURN_PARAMETERS, NEXT_PARAMETERS
from diagonal_norm_worker import WEIGHT
from image_dual_analysis import NAMES

TEMPLATES = ['publish_image_dual.py', 'publish_source_complement.py', 'publish_sparse_text_kv.py']


def items(p):
    s = read(p / 'inputs.json');result = []
    for key in ['parentRecords', 'latestParentRecords', 'dualParentRecords']:
        result += list(s[key].values())
    result += [{'path': n, 'sha256': h} for n, h in s['sourceMetadataFiles'].items()]
    result += list(fields({'selected': s['selected'], 'banks': s['banks'], 'reuse': s['reuse'], 'textSlice': s['textKeySlice'],
                          'matchedControlFields': s['matchedControlFields'], 'controlModLN': s['controlModLN']}))
    for n in ['targets.json', 'compiler.json', 'actor.json', 'report.json']:
        result += list(fields(read(p / n)))
    result += [{'path': str(Path(s['modelRoot']) / n), 'sha256': h} for n, h in s['imageAssets'].items()]
    return result


def verify(root):
    from mrun.client.api import Api
    root, collection, job, request = checked(root);p = root / 'result';s = read(p / 'inputs.json')
    roots = [job['result']['diagnostics']['cwd'], s['modelRoot'], str(Path(s['textKeySlice']['path']).parent)]
    roots += [str(Path(cut['cutFile']['path']).parent.parent) for cut in s['sourceCuts'].values()]
    for key, custody in [('parentJobID', 'parentCustody'), ('latestParentJobID', 'latestParentCustody'), ('sourceJobID', 'sourceCustody'), ('dualParentJobID', 'dualParentCustody')]:
        parent = Api().json('GET', '/api/jobs/' + s[key])
        if parent['state'] != 'succeeded' or parent['payload_custody'] != s[custody]:
            raise ValueError('Source/parent scheduler custody differs')
        roots.append(parent['result']['diagnostics']['cwd'])
    base = job['result']['diagnostics']['cwd'];unique = {}
    for x in items(p) + [{'path': base + '/response-result/' + n.name, 'sha256': sha(n)} for n in p.iterdir()]:
        if x['path'] in unique and unique[x['path']]['sha256'] != x['sha256']:
            raise ValueError('Conflicting full byte identity')
        unique[x['path']] = x
    reader = read(p / 'actor.json')['read']['fields']
    splices = [{'arm': 'actor', 'field': n, 'support': request['support'], 'selected': s['selected'][n], 'bank': bank, 'hybrid': reader[output]}
        for n, bank, output in [('K', s['textKeySlice'], 'textK_hybrid'), ('V', s['banks']['Q']['V'], 'V_hybrid')]]
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(BYTES)],
        input=json.dumps({'allowedRoots': roots, 'items': list(unique.values()), 'spliceChecks': splices, 'textKeySlice': s['textKeySlice']}),
        text=True, capture_output=True, timeout=240)
    if result.returncode:
        raise RuntimeError(result.stderr)
    receipt = json.loads(result.stdout);receipt.update(schema='tce-diagonal-norm-bytes-v1', jobID=job['job_id'],
        verifierSourceSHA256=sha(__file__), templateSourceHashes={n: sha(Path(__file__).with_name(n)) for n in TEMPLATES},
        phaseSeals={n: sha(p / (n + '-seal.json')) for n in ['compiler', 'actor']})
    save(root / 'retained-array-verification.json', receipt)
    return {'files': len(receipt['files']), 'bytes': receipt['totalBytesVerified'], 'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']),
            'hybridSpliceWitnesses': 2, 'textPrefixWitnesses': 1}


def publish(root, destination):
    root, collection, job, request = checked(root);p = root / 'result';receipt = read(root / 'retained-array-verification.json')
    if receipt['verifierSourceSHA256'] != sha(__file__) or receipt['templateSourceHashes'] != {n: sha(Path(__file__).with_name(n)) for n in TEMPLATES}:
        raise ValueError('Verified source/template changed')
    out = Path(destination);out.mkdir(parents=True, exist_ok=False);streams = []
    for member in p.iterdir():
        if member.suffix == '.jsonl':
            raw = member.read_bytes();target = out / (member.name + '.gz');target.write_bytes(gzip.compress(raw, mtime=0))
            streams.append({'source': member.name, 'sourceSHA256': sha(member), 'published': target.name, 'publishedSHA256': sha(target)})
        else:
            shutil.copyfile(member, out / member.name)
    shutil.copyfile(root / 'retained-array-verification.json', out / 'retained-array-verification.json')
    save(out / 'provenance.json', {'schema': 'tce-diagonal-norm-publication-v1', 'jobID': job['job_id'], 'requestSHA256': request['requestSHA256'],
        'payloadCustody': collection['custody'], 'schedulerResult': collection['result'], 'completeCollectedInventory': collection['inventory'],
        'losslessStreams': streams, 'scope': 'One source-defined norm law and own continuation; all native ancestry/dual cache/receiving maps/old references paid, no new image or zero-neural completion.'})
    save(out / 'files.json', {x.name: sha(x) for x in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    p = Path(source);r, s, report, receipt = [read(p / n) for n in ['request.json', 'inputs.json', 'report.json', 'retained-array-verification.json']]
    if identity({k: v for k, v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request identity differs')
    for n, digest in read(p / 'files.json').items():
        if sha(p / n) != digest:
            raise ValueError('Immutable publication changed: ' + n)
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n)) != r['sourceHashes'][n]:
            raise ValueError('Frozen owned source changed: ' + n)
    if receipt['verifierSourceSHA256'] != sha(__file__) or receipt['templateSourceHashes'] != {n: sha(Path(__file__).with_name(n)) for n in TEMPLATES}:
        raise ValueError('Full byte verifier/template changed')
    for x in items(p):
        if receipt['files'][x['path']]['sha256'] != x['sha256'] or (x.get('payloadSHA256') and receipt['npyPayloadWitnesses'][x['path']]['payloadSHA256'] != x['payloadSHA256']):
            raise ValueError('Full consumed/emitted byte custody absent')
    compiler, actor = read(p / 'compiler.json'), read(p / 'actor.json');acts = []
    executions = {n: read(p / (n + '-execution.json')) for n in ['compiler', 'actor']}
    for phase, arm in [('compiler', compiler), ('actor', actor)]:
        seal = read(p / (phase + '-seal.json'));execution = executions[phase]
        if seal[phase + 'SHA256'] != sha(p / (phase + '.json')) or seal['requestSHA256'] != r['requestSHA256'] or receipt['phaseSeals'][phase] != sha(p / (phase + '-seal.json')):
            raise ValueError('Complete phase seal differs')
        weights = {x['name']: x['payloadSHA256'] for x in execution['loadedParameters']}
        if set(weights) != set((WEIGHT,) if phase == 'compiler' else RETURN_PARAMETERS + NEXT_PARAMETERS):
            raise ValueError('Source-only compiler/consumer hydration differs')
        guard = execution['fileQuarantine'];allowed, denied = set(guard['allowedFiles']), set(guard['deniedCanaries'])
        required = set(s['deniedNumericalInputs']) | (set(s['compilerForbiddenSources']) if phase == 'compiler' else set())
        if not required.issubset(denied) or required & allowed or not set(guard['successfulFileOpenCounts']).issubset(allowed) or not execution['parameterInputRNGClosure']:
            raise ValueError('Actual read/parameter/input/RNG closure differs')
        if phase == 'compiler' and allowed != {str(Path(s['modelRoot']) / n) for n in s['imageAssets']}:
            raise ValueError('Weight-only compiler can read supplied source operands')
        if execution['nominalMatrixMACs'] != r['plannedMatrixMACs'][phase] or sum(x['nominalMACs'] for x in execution['operations']) != execution['nominalMatrixMACs']:
            raise ValueError('Actual matrix-work sum differs')
        for name, act in arm.items():
            if not isinstance(act, dict) or 'sourceFrame' not in act:
                continue
            acts.append(act);expected = {n: weights[n] for n in act['sourceFrame']['slots'] if n in weights}
            if name == 'compile':
                pass
            elif name == 'modln':
                cut = s['sourceCuts']['Q'];declared = {x['key']['value']: x['value'] for x in cut['cut']['manifest']['payload']['items']}
                context = {x['key']['value']: x['value'] for x in declared['context']['items']}
                expected.update(image=declared['image']['sha256'], imgMod=context['img_mod']['sha256'])
            elif name == 'norm':
                expected.update(x=actor['modln']['fields']['x']['payloadSHA256'], diagonal=compiler['compile']['fields']['diagonal']['payloadSHA256'])
            elif name == 'read':
                expected.update(x=actor['modln']['fields']['x']['payloadSHA256'], norm=actor['norm']['fields']['norm']['payloadSHA256'],
                    **{n: s['reuse'][n]['payloadSHA256'] for n in ['duals', 'groupIDs']},
                    Q=s['selected']['Q']['payloadSHA256'], selectedK=s['selected']['K']['payloadSHA256'], selectedV=s['selected']['V']['payloadSHA256'],
                    textK=s['textKeySlice']['payloadSHA256'], bankV=s['banks']['Q']['V']['payloadSHA256'])
            elif name == 'source':
                expected.update(H=actor['read']['fields']['H']['payloadSHA256'], **{n: s['selected'][n]['payloadSHA256'] for n in ['R', 'txtMod']})
            elif name == 'next_query':
                expected.update(rows=actor['source']['fields']['T']['payloadSHA256'], **{n: s['selected'][n]['payloadSHA256'] for n in ['txtMod', 'ropeCos', 'ropeSin']})
            else:
                raise ValueError('Unknown typed operation')
            check_frame(act, expected, r['requestSHA256'])
            for n in act['evidence']['metrics']['capturedPayloads']:
                x = act['fields'][n]
                if act['outputFrame']['slots'][n]['abi'] != {'shape': x['shape'], 'dtype': x['logicalDtype']}:
                    raise ValueError('Explicit typed output ABI differs')
    if actor['compilerSealSHA256'] != sha(p / 'compiler-seal.json'):
        raise ValueError('Source-only compiler release differs')
    joins = [('norm', 'modLNOutputFrame', 'modln'), ('read', 'normOutputFrame', 'norm'), ('source', 'headOutputFrame', 'read'), ('next_query', 'sourceOutputFrame', 'source')]
    if any(actor[a]['sourceFrame']['metadata'][key] != actor[b]['outputFrame']['frame_id'] for a, key, b in joins):
        raise ValueError('Actual own-input/norm/head/state/query join differs')
    if report['totalNewMatrixMACs'] != 1421869056 or sum(x['nominalMatrixMACs'] for x in executions.values()) != report['totalNewMatrixMACs']:
        raise ValueError('Complete new matrix work differs')
    if executions['compiler']['elementwise'] != [{'stage': 'compile', 'weightSquares': 9437184, 'sumAdditions': 9363456, 'meanScalings': 73728, 'input': 'Original W only; no source states/norms'}] or executions['actor']['elementwise'] != [{'stage': 'norm', 'inputSquares': 3145728, 'epsilonAdditions': 24576, 'squareRoots': 24576, 'epsilon': 1e-6}]:
        raise ValueError('Explicit nonmatrix operation accounting differs')
    if receipt['textPrefixWitness']['byteExact'] is not True or len(receipt['bankSpliceWitnesses']) != 2:
        raise ValueError('Text-prefix/hybrid byte witnesses absent')
    raw = gzip.decompress((p / 'observations.jsonl.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest() != read(p / 'provenance.json')['losslessStreams'][0]['sourceSHA256']:
        raise ValueError('Lossless observation stream differs')
    rows = [json.loads(x) for x in raw.splitlines()];actual = [(x['field'], x['panel'], x['row'], x.get('head')) for x in rows]
    expected = {(n, 'whole-row', row, None) for n in NAMES for row in r['support']}
    expected |= {(n, 'query-head', row, h) for n in ['nextQProjected', 'nextQNormalized', 'nextQ'] for row in r['support'] for h in range(24)}
    expected |= {('key_norm', 'image-key-head', row, h) for row in range(1024) for h in range(24)}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Complete unique selected-response/norm addresses differ')
    return {'schema': 'tce-diagonal-norm-mechanics-v1', 'ownedSourcePins': len(r['ownedSources']), 'typedActs': len(acts),
        'explicitOutputABIs': sum(len(a['evidence']['metrics']['capturedPayloads']) for a in acts),
        'ownInputNormHeadStateQueryJoins': len(joins), 'observations': len(rows), 'imageKeyHeadRecords': 24576,
        'sourceOnlyCompiler': True, 'newFullKeyProjections': 0, 'textPrefixWitnesses': 1, 'hybridSpliceWitnesses': 2,
        'fullByteFiles': len(receipt['files']), 'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']),
        'scope': 'Source/seal/bytes/typed-own-state/work/quarantine mechanics; no effect-size gate, semantic/image/generalization/speedup or zero-neural certification.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('action', choices=['verify', 'publish', 'audit']);p.add_argument('source');p.add_argument('destination', nargs='?');a = p.parse_args()
    print(json.dumps(publish(a.source, a.destination) if a.action == 'publish' else globals()[a.action](a.source), indent=2))
