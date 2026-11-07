"""Full-byte custody, independent complementary splices and typed own-state joins."""
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
from publish_source_complement import BYTES, check_frame
from propagated_context import BANK_PARAMETERS, SELECTED_PARAMETERS, RETURN_PARAMETERS, NEXT_PARAMETERS

TEMPLATES = ['publish_source_complement.py', 'publish_sparse_text_kv.py']


def sources(p):
    s = read(p / 'inputs.json')
    values = [x['cutFile'] for x in s['sourceCuts'].values()]
    values += [{'path': path, 'sha256': digest} for cut in s['sourceCuts'].values() for page in cut['pages'].values() for path, digest in page['files'].items()]
    values += list(fields({n: read(p / (n + '.json')) for n in ['banks', 'primary', 'control']}))
    values += [{'path': str(Path(s['modelRoot']) / n), 'sha256': digest} for n, digest in s['imageAssets'].items()]
    return values


def verify(root):
    from mrun.client.api import Api
    root, collection, job, request = checked(root)
    p = root / 'result';s = read(p / 'inputs.json');bank = read(p / 'banks.json')
    old = Api().json('GET', '/api/jobs/' + s['sourceJobID'])
    if old['state'] != 'succeeded' or old['payload_custody'] != s['sourceCustody']:
        raise ValueError('Native source custody differs')
    base = job['result']['diagnostics']['cwd']
    values = sources(p) + [{'path': base + '/response-result/' + x.name, 'sha256': sha(x)} for x in p.iterdir()]
    unique = {}
    for x in values:
        if x['path'] in unique and unique[x['path']]['sha256'] != x['sha256']:
            raise ValueError('Conflicting complete source bytes')
        unique[x['path']] = x
    checks = [{'arm': arm, 'field': n, 'support': request['support'], 'selected': bank['selected']['fields'][n],
               'bank': bank['banks'][role]['fields'][n], 'hybrid': read(p / (arm + '.json'))['source']['fields'][n + '_hybrid']}
              for arm, role in [('primary', 'Q'), ('control', 'P')] for n in ['K', 'V']]
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(BYTES)],
        input=json.dumps({'allowedRoots': [base, s['modelRoot'], '/mnt/big/mdb/research/job-a2048028d7bf/flux-binding-family-result'],
                          'items': list(unique.values()), 'spliceChecks': checks}), text=True, capture_output=True, timeout=240)
    if result.returncode:
        raise RuntimeError(result.stderr)
    receipt = json.loads(result.stdout)
    receipt.update(schema='tce-propagated-context-bytes-v1', jobID=job['job_id'], verifierSourceSHA256=sha(__file__),
        templateSourceHashes={n: sha(Path(__file__).with_name(n)) for n in TEMPLATES},
        phaseSeals={n: sha(p / (n + '-seal.json')) for n in ['banks', 'primary', 'control']})
    save(root / 'retained-array-verification.json', receipt)
    return {'files': len(receipt['files']), 'bytes': receipt['totalBytesVerified'], 'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']), 'bankSpliceWitnesses': len(receipt['bankSpliceWitnesses'])}


def publish(root, destination):
    root, collection, job, request = checked(root)
    p = root / 'result';receipt = read(root / 'retained-array-verification.json')
    if receipt['verifierSourceSHA256'] != sha(__file__) or receipt['templateSourceHashes'] != {n: sha(Path(__file__).with_name(n)) for n in TEMPLATES} or receipt['phaseSeals'] != {n: sha(p / (n + '-seal.json')) for n in ['banks', 'primary', 'control']}:
        raise ValueError('Verified source/seals changed')
    out = Path(destination);out.mkdir(parents=True, exist_ok=False);streams = []
    for x in p.iterdir():
        if x.suffix == '.jsonl':
            raw = x.read_bytes();target = out / (x.name + '.gz');target.write_bytes(gzip.compress(raw, mtime=0))
            if gzip.decompress(target.read_bytes()) != raw:
                raise ValueError('Lossless observations differ')
            streams.append({'source': x.name, 'sourceSHA256': sha(x), 'published': target.name, 'publishedSHA256': sha(target), 'sourceBytes': len(raw)})
        else:
            shutil.copyfile(x, out / x.name)
    shutil.copyfile(root / 'retained-array-verification.json', out / 'retained-array-verification.json')
    save(out / 'provenance.json', {'schema': 'tce-propagated-context-publication-v1', 'jobID': job['job_id'],
         'requestSHA256': request['requestSHA256'], 'payloadCustody': collection['custody'], 'schedulerResult': collection['result'],
         'completeCollectedInventory': collection['inventory'], 'losslessStreams': streams,
         'scope': 'Conditional changed-complement read→ownMLP→ownquery with paid native states and original maps. No native full-width reference or image.'})
    save(out / 'files.json', {x.name: sha(x) for x in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    p = Path(source)
    r, s, bank, report, receipt = [read(p / n) for n in ['request.json', 'inputs.json', 'banks.json', 'report.json', 'retained-array-verification.json']]
    if identity({k: v for k, v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request identity differs')
    for n, digest in read(p / 'files.json').items():
        if sha(p / n) != digest:
            raise ValueError('Published member changed')
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n)) != r['sourceHashes'][n]:
            raise ValueError('Executed scientific source changed')
    if receipt['verifierSourceSHA256'] != sha(__file__) or receipt['templateSourceHashes'] != {n: sha(Path(__file__).with_name(n)) for n in TEMPLATES}:
        raise ValueError('Byte verifier source changed')
    for x in sources(p):
        if receipt['files'][x['path']]['sha256'] != x['sha256'] or (x.get('payloadSHA256') and receipt['npyPayloadWitnesses'][x['path']]['payloadSHA256'] != x['payloadSHA256']):
            raise ValueError('Complete original/output bytes absent')
    executions = {n: read(p / (n + '-execution.json')) for n in ['banks', 'primary', 'control']}
    weights = {x['name']: x['payloadSHA256'] for x in executions['banks']['loadedParameters']}
    if set(weights) != set(BANK_PARAMETERS + SELECTED_PARAMETERS):
        raise ValueError('Bank parameter closure differs')
    declared = {}
    acts = []
    for role, cut_record in s['sourceCuts'].items():
        cut = cut_record['cut'];d = {x['key']['value']: x['value'] for x in cut['manifest']['payload']['items']};declared[role] = d
        if identity(cut['manifest']) != cut_record['cutID']:
            raise ValueError('Source cut identity differs')
        for n, page in cut_record['pages'].items():
            if cut['slots'][n] != page['pageID'] or page['declaredDescriptor'] != d[n] or page['metadata']['value'] != d[n]:
                raise ValueError('Source slot/page membership differs')
        c = {x['key']['value']: x['value'] for x in d['context']['items']}
        expected = dict(image=d['image']['sha256'], text=d['text']['sha256'], imgMod=c['img_mod']['sha256'], txtMod=c['txt_mod']['sha256'],
                        ropeCos=c['rope']['items'][0]['sha256'], ropeSin=c['rope']['items'][1]['sha256'], **{n: weights[n] for n in BANK_PARAMETERS})
        check_frame(bank['banks'][role], expected, r['requestSHA256']);acts.append(bank['banks'][role])
    if declared['P']['context'] != declared['Q']['context']:
        raise ValueError('Shared clock/position premise differs')
    c = {x['key']['value']: x['value'] for x in declared['P']['context']['items']}
    expected = dict(text=declared['P']['text']['sha256'], txtMod=c['txt_mod']['sha256'],
        fullRopeCos=c['rope']['items'][0]['sha256'], fullRopeSin=c['rope']['items'][1]['sha256'],
        bankK=bank['banks']['P']['fields']['K']['payloadSHA256'], bankV=bank['banks']['P']['fields']['V']['payloadSHA256'],
        **{n: weights[n] for n in SELECTED_PARAMETERS})
    check_frame(bank['selected'], expected, r['requestSHA256']);acts.append(bank['selected'])
    if bank['selected']['sourceFrame']['metadata']['sourceBankOutputFrame'] != bank['banks']['P']['outputFrame']['frame_id']:
        raise ValueError('Actual fresh-P-bank→selected join differs')
    for phase, role in [('primary', 'Q'), ('control', 'P')]:
        actual = read(p / (phase + '.json'));ev = executions[phase]
        weights = {x['name']: x['payloadSHA256'] for x in ev['loadedParameters']}
        if set(weights) != set(RETURN_PARAMETERS + NEXT_PARAMETERS):
            raise ValueError('Arm parameter closure differs')
        expected = {n: x['payloadSHA256'] for n, x in bank['selected']['fields'].items()}
        expected.update(bankK=bank['banks'][role]['fields']['K']['payloadSHA256'], bankV=bank['banks'][role]['fields']['V']['payloadSHA256'], **{n: weights[n] for n in RETURN_PARAMETERS})
        check_frame(actual['source'], expected, r['requestSHA256']);acts.append(actual['source'])
        expected = {'rows': actual['source']['fields']['T']['payloadSHA256'], **{n: bank['selected']['fields'][n]['payloadSHA256'] for n in ['txtMod', 'ropeCos', 'ropeSin']}, **{n: weights[n] for n in NEXT_PARAMETERS}}
        check_frame(actual['nextQuery'], expected, r['requestSHA256']);acts.append(actual['nextQuery'])
        if actual['nextQuery']['sourceFrame']['metadata']['sourceOutputFrame'] != actual['source']['outputFrame']['frame_id']:
            raise ValueError('Actual own-T→nextquery join differs')
        guard = ev['fileQuarantine']
        for x in bank['banks']['P' if role == 'Q' else 'Q']['fields'].values():
            if x['path'] in guard['allowedFiles'] or x['path'] not in guard['deniedCanaries']:
                raise ValueError('Opposite full-bank denial differs')
        if phase == 'control' and any(x['path'] not in guard['deniedCanaries'] for x in fields(read(p / 'primary.json'))):
            raise ValueError('Control primary-array denial differs')
    for name, ev in executions.items():
        seal = read(p / (name + '-seal.json'))
        if seal[name + 'SHA256'] != sha(p / (name + '.json')) or seal['requestSHA256'] != r['requestSHA256'] or receipt['phaseSeals'][name] != sha(p / (name + '-seal.json')):
            raise ValueError('Complete phase seal differs')
        if sum(x['nominalMACs'] for x in ev['linears'] + ev['attentions']) != ev['nominalMatrixMACs'] or ev['nominalMatrixMACs'] != (58019807232 if name == 'banks' else 452984832):
            raise ValueError('Physical work sum differs')
        if not ev['parameterInputRNGClosure'] or not set(ev['fileQuarantine']['successfulFileOpenCounts']).issubset(ev['fileQuarantine']['allowedFiles']):
            raise ValueError('Actual numerical input/parameter/profile closure differs')
    for act in acts:
        for n in act['evidence']['metrics']['capturedPayloads']:
            item = act['fields'][n]
            if act['outputFrame']['slots'][n]['abi'] != {'shape': item['shape'], 'dtype': item['logicalDtype']}:
                raise ValueError('Actual output ABI differs from captured tensor')
    primary, control = [read(p / (n + '.json')) for n in ['primary', 'control']]
    if control['primarySealSHA256'] != sha(p / 'primary-seal.json') or any(x['sharedSealSHA256'] != sha(p / 'banks-seal.json') for x in [primary, control]):
        raise ValueError('Sequential release joins differ')
    views = receipt['bankSpliceWitnesses']
    if len(views) != 4 or any(not x['selectedRowsByteExact'] or not x['outsideSupportByteExact'] for x in views):
        raise ValueError('Independent selected/complement splice differs')
    changed = {n: len({x['outsideSupportSHA256'] for x in views if x['field'] == n}) == 2 for n in ['K', 'V']}
    if not any(changed.values()) or not report['outsideSupportKVChanged']:
        raise ValueError('Changed-complement instrument not applicable')
    raw = gzip.decompress((p / 'observations.jsonl.gz').read_bytes())
    stream = read(p / 'provenance.json')['losslessStreams'][0]
    if hashlib.sha256(raw).hexdigest() != stream['sourceSHA256']:
        raise ValueError('Lossless observation bytes differ')
    rows = [json.loads(x) for x in raw.splitlines()]
    if len(rows) != 348 or sum(x['panel'] == 'query-head' for x in rows) != 288 or sum(x['nominalMatrixMACs'] for x in executions.values()) != report['totalNewMatrixMACs']:
        raise ValueError('Complete observation/work coverage differs')
    return {'schema': 'tce-propagated-context-mechanics-v1', 'ownedSourcePins': len(r['ownedSources']), 'sourcePageMembershipJoins': 6,
            'typedActs': len(acts), 'ownStateQueryJoins': 2, 'explicitOutputABIs': sum(len(x['evidence']['metrics']['capturedPayloads']) for x in acts),
            'independentSpliceWitnesses': 4, 'outsideSupportKVByteChanged': changed, 'observations': len(rows),
            'fullByteFiles': len(receipt['files']), 'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']),
            'scope': 'Engineering source/byte/typed-state/work/quarantine evidence, not a general semantic or zero-neural image certification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('action', choices=['verify', 'publish', 'audit']);parser.add_argument('source');parser.add_argument('destination', nargs='?');args = parser.parse_args()
    print(json.dumps(publish(args.source, args.destination) if args.action == 'publish' else globals()[args.action](args.source), indent=2))
