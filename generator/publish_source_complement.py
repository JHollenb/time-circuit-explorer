"""Complete byte custody, independent bank splices, and lossless publication."""
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
from source_complement import SHARED_PARAMETERS, RETURN_PARAMETERS

TEMPLATE = Path(__file__).with_name('publish_sparse_text_kv.py')
BYTES = NPY_BYTES.split('print(json.dumps')[0] + '''
def raw(item):
 p=Path(item['path'])
 if str(p) not in inventory or str(p) not in witnesses:raise ValueError('Splice input lacks full byte/payload custody')
 with p.open('rb') as f:
  magic=f.read(8);n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0]
  f.read(n);return f.read(),witnesses[str(p)]['header']
joins=[];reread=0
for pair in s['spliceChecks']:
 selected,hs=raw(pair['selected']);bank,hb=raw(pair['bank']);hybrid,hh=raw(pair['hybrid'])
 reread+=len(selected)+len(bank)+len(hybrid)
 if hb['shape']!=hh['shape'] or len(hb['shape'])!=4 or hb['shape'][0]!=1 or hb['descr']!='<u2' or hs['descr']!='<u2' or hh['descr']!='<u2':raise ValueError('Splice native byte geometry differs')
 support=pair['support'];shape=hb['shape'];rows=shape[1];rowbytes=shape[2]*shape[3]*2
 if hs['shape']!=(1,len(support),shape[2],shape[3]) or len(set(support))!=len(support) or any(j<0 or j>=rows for j in support):raise ValueError('Splice support geometry differs')
 inside=hashlib.sha256();outside=hashlib.sha256();expectedoutside=hashlib.sha256()
 for j in range(rows):
  actual=hybrid[j*rowbytes:(j+1)*rowbytes]
  if j in support:
   i=support.index(j);expected=selected[i*rowbytes:(i+1)*rowbytes]
   if actual!=expected:raise ValueError('Selected hybrid row differs from shared emitted operand')
   inside.update(actual)
  else:
   expected=bank[j*rowbytes:(j+1)*rowbytes]
   if actual!=expected:raise ValueError('Hybrid complement differs outside declared support')
   outside.update(actual);expectedoutside.update(expected)
 joins.append({'arm':pair['arm'],'field':pair['field'],'support':support,'selectedRowsByteExact':True,
               'outsideSupportByteExact':True,'selectedRowsSHA256':inside.hexdigest(),
               'outsideSupportSHA256':outside.hexdigest(),'bankOutsideSupportSHA256':expectedoutside.hexdigest()})
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),
                  'npyPayloadWitnesses':witnesses,'bankSpliceWitnesses':joins,'spliceBytesReadAgain':reread}))
'''


def items(p):
    s = read(p / 'inputs.json')
    values = list(s['parentRecords'].values()) + [s['pCut']['cutFile']]
    values += [{'path': path, 'sha256': digest} for page in s['pCut']['pages'].values() for path, digest in page['files'].items()]
    values += list(fields(s['banks'])) + [s['qAttention']] + list(s['readerInputs'].values())
    values += [x for name in ['shared.json', 'primary.json', 'control.json', 'targets.json'] for x in fields(read(p / name))]
    values += [{'path': str(Path(s['modelRoot']) / n), 'sha256': h} for n, h in s['imageAssets'].items()]
    return values


def verify(root):
    from mrun.client.api import Api
    root, collection, job, request = checked(root)
    p = root / 'result'
    s, shared = read(p / 'inputs.json'), read(p / 'shared.json')
    allowed_roots = [job['result']['diagnostics']['cwd'], s['modelRoot'],
                     '/mnt/big/mdb/research/job-a2048028d7bf/flux-binding-family-result',
                     '/mnt/hdd8tb/mrun-work/job-cd781e9a2a10', '/mnt/hdd8tb/mrun-work/job-832603af095e']
    for job_key, custody_key in [('parentJobID', 'parentCustody'), ('sourceJobID', 'sourceCustody')]:
        parent = Api().json('GET', '/api/jobs/' + s[job_key])
        if parent['state'] != 'succeeded' or parent['payload_custody'] != s[custody_key]:
            raise ValueError('Parent/source scheduler custody changed')
        allowed_roots.append(parent['result']['diagnostics']['cwd'])
    base = job['result']['diagnostics']['cwd']
    unique = {}
    for item in items(p) + [{'path': base + '/response-result/' + x.name, 'sha256': sha(x)} for x in p.iterdir()]:
        if item['path'] in unique and unique[item['path']]['sha256'] != item['sha256']:
            raise ValueError('Conflicting complete byte identity')
        unique[item['path']] = item
    checks = [{'arm': arm, 'field': n, 'support': request['support'], 'selected': shared['fields'][n],
               'bank': s['banks'][role][n], 'hybrid': read(p / (arm + '.json'))['source']['fields'][n + '_hybrid']}
              for arm, role in [('primary', 'Q'), ('control', 'P')] for n in ['K', 'V']]
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(BYTES)],
                            input=json.dumps({'allowedRoots': allowed_roots, 'items': list(unique.values()), 'spliceChecks': checks}),
                            text=True, capture_output=True, timeout=240)
    if result.returncode:
        raise RuntimeError(result.stderr)
    receipt = json.loads(result.stdout)
    receipt.update(schema='tce-source-complement-bytes-v1', jobID=job['job_id'],
                   phaseSeals={name: sha(p / (name + '-seal.json')) for name in ['shared', 'primary', 'control']},
                   verifierSourceSHA256=sha(__file__), byteTemplateSourceSHA256=sha(TEMPLATE),
                   scope='Full source cut/pages/blobs, retained banks/direct return/reader cache, original assets, own selected/source/reader outputs and targets. Independent raw-byte selected/complement splice comparisons; SSH byte I/O only.')
    save(root / 'retained-array-verification.json', receipt)
    return {'files': len(receipt['files']), 'bytes': receipt['totalBytesVerified'], 'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']), 'bankSpliceWitnesses': len(receipt['bankSpliceWitnesses'])}


def publish(root, destination):
    root, collection, job, request = checked(root)
    p = root / 'result'
    receipt = read(root / 'retained-array-verification.json')
    if receipt['phaseSeals'] != {n: sha(p / (n + '-seal.json')) for n in ['shared', 'primary', 'control']} or receipt['verifierSourceSHA256'] != sha(__file__) or receipt['byteTemplateSourceSHA256'] != sha(TEMPLATE):
        raise ValueError('Byte verification source/release differs')
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    streams = []
    for file in p.iterdir():
        if file.suffix == '.jsonl':
            raw = file.read_bytes()
            target = out / (file.name + '.gz')
            target.write_bytes(gzip.compress(raw, mtime=0))
            if gzip.decompress(target.read_bytes()) != raw:
                raise ValueError('Lossless stream bytes differ')
            streams.append({'source': file.name, 'sourceSHA256': sha(file), 'sourceBytes': len(raw), 'published': target.name,
                            'publishedSHA256': sha(target), 'publishedBytes': target.stat().st_size})
        else:
            shutil.copyfile(file, out / file.name)
    shutil.copyfile(root / 'retained-array-verification.json', out / 'retained-array-verification.json')
    save(out / 'provenance.json', {'schema': 'tce-source-complement-publication-v1', 'jobID': job['job_id'], 'requestSHA256': request['requestSHA256'],
         'payloadCustody': collection['custody'], 'schedulerResult': collection['result'], 'completeCollectedInventory': collection['inventory'],
         'losslessStreams': streams, 'scope': 'Conditional complementary-context experiment with paid banks, Q return, original maps, historical prefixes and recipient cache. No new image or general zero-neural backend.'})
    save(out / 'files.json', {x.name: sha(x) for x in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(x.stat().st_size for x in out.iterdir())}


def structured(item):
    return identity({'dtype': item['logicalDtype'], 'shape': item['shape'], 'bytes': item['payloadSHA256']})


def check_frame(actual, expected, program):
    src, dst, ev = actual['sourceFrame'], actual['outputFrame'], actual['evidence']
    if {n: x['fingerprint'] for n, x in src['slots'].items()} != expected or src['frame_id'] != ev['input_frame_id'] or dst['parent_frame_id'] != src['frame_id'] or dst['frame_id'] != ev['output_frame_id'] or dst['program_fingerprint'] != program:
        raise ValueError('Actual typed input/output lineage differs')
    captures = ev['metrics'].get('capturedPayloads', {'R_attention': ev['metrics'].get('capturedPayloadSHA256')})
    for name, digest in captures.items():
        item = actual['fields'][name]
        if item['payloadSHA256'] != digest or dst['slots'][name]['fingerprint'] != structured(item):
            raise ValueError('Typed output capture/convention differs')


def audit(source):
    p = Path(source)
    r, s, shared, primary, control, report, receipt = [read(p / n) for n in
        ['request.json', 'inputs.json', 'shared.json', 'primary.json', 'control.json', 'report.json', 'retained-array-verification.json']]
    if identity({k: v for k, v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request identity differs')
    for name, digest in read(p / 'files.json').items():
        if sha(p / name) != digest:
            raise ValueError('Published member differs')
    for name in r['ownedSources']:
        if sha(Path(__file__).parent / name) != r['sourceHashes'][name]:
            raise ValueError('Executed source changed')
    if receipt['verifierSourceSHA256'] != sha(__file__) or receipt['byteTemplateSourceSHA256'] != sha(TEMPLATE):
        raise ValueError('Complete byte verifier source differs')
    for name in ['shared', 'primary', 'control']:
        seal = read(p / (name + '-seal.json'))
        if seal[name + 'SHA256'] != sha(p / (name + '.json')) or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Complete phase JSON/seal/request join differs')
    if control['primarySealSHA256'] != sha(p / 'primary-seal.json') or primary['sharedSealSHA256'] != sha(p / 'shared-seal.json') or control['sharedSealSHA256'] != sha(p / 'shared-seal.json'):
        raise ValueError('Shared/primary/control release order differs')
    for item in items(p):
        if receipt['files'][item['path']]['sha256'] != item['sha256']:
            raise ValueError('Full source/output bytes absent')
        if item.get('payloadSHA256') and receipt['npyPayloadWitnesses'][item['path']]['payloadSHA256'] != item['payloadSHA256']:
            raise ValueError('Complete independent NPY payload witness absent')
    if len(receipt['bankSpliceWitnesses']) != 4 or any(x['selectedRowsByteExact'] is not True or x['outsideSupportByteExact'] is not True or x['outsideSupportSHA256'] != x['bankOutsideSupportSHA256'] for x in receipt['bankSpliceWitnesses']):
        raise ValueError('Independent actual hybrid bank splice witness differs')
    cut = s['pCut']['cut']
    declared = {x['key']['value']: x['value'] for x in cut['manifest']['payload']['items']}
    if identity(cut['manifest']) != s['pCut']['cutID']:
        raise ValueError('Authenticated source cut differs')
    for name, page in s['pCut']['pages'].items():
        if cut['slots'][name] != page['pageID'] or page['metadata']['value'] != declared[name] or page['declaredDescriptor'] != declared[name]:
            raise ValueError('Source cut-slot-page membership differs')
    context = {x['key']['value']: x['value'] for x in declared['context']['items']}
    executions = {name: read(p / (name + '-execution.json')) for name in ['shared', 'primary', 'control']}
    weights = {x['name']: x['payloadSHA256'] for x in executions['shared']['loadedParameters']}
    if set(weights) != set(SHARED_PARAMETERS):
        raise ValueError('Shared selected parameter closure differs')
    expected = {'text': declared['text']['sha256'], 'mod': context['txt_mod']['sha256'],
                'cos': context['rope']['items'][0]['sha256'], 'sin': context['rope']['items'][1]['sha256'], **weights}
    check_frame(shared, expected, r['requestSHA256'])
    for phase, arm, role in [('primary', primary, 'Q'), ('control', control, 'P')]:
        execution = executions[phase]
        weights = {x['name']: x['payloadSHA256'] for x in execution['loadedParameters']}
        if set(weights) != set(RETURN_PARAMETERS + ACTOR_PARAMETERS):
            raise ValueError('Arm selected parameter closure differs')
        expected = {n: x['payloadSHA256'] for n, x in shared['fields'].items()}
        expected.update(bankK=s['banks'][role]['K']['payloadSHA256'], bankV=s['banks'][role]['V']['payloadSHA256'],
                        qAttention=s['qAttention']['payloadSHA256'], **{n: weights[n] for n in RETURN_PARAMETERS})
        check_frame(arm['source'], expected, r['requestSHA256'])
        expected = {n: x['payloadSHA256'] for n, x in s['readerInputs'].items()}
        expected.update(textRows=arm['source']['fields']['PQP']['payloadSHA256'], Kweight=weights['transformer_blocks.1.attn.add_k_proj.weight'],
                        Vweight=weights['transformer_blocks.1.attn.add_v_proj.weight'], KnormWeight=weights['transformer_blocks.1.attn.norm_added_k.weight'],
                        WoFP64=execution['WoFP64']['payloadSHA256'])
        check_frame(arm['recipient'], expected, r['requestSHA256'])
        if arm['recipient']['sourceFrame']['metadata']['sourceOutputFrame'] != arm['source']['outputFrame']['frame_id']:
            raise ValueError('Actual source-output-to-recipient join differs')
        heads = [x for x in execution['linears'] if x['parameter'] == WO]
        if [x['head'] for x in heads] != list(range(24)) or any(x['weightShape'] != [3072, 128] or x['inputShape'] != [4, 128] for x in heads):
            raise ValueError('Physical unchanged reader head blocks differ')
    for phase, execution in executions.items():
        total = sum(x['nominalMACs'] for x in execution['linears'] + execution['attentions'] + execution['contractions'])
        if total != execution['nominalMatrixMACs'] or total != (113246208 if phase == 'shared' else 843055104):
            raise ValueError('Actual physical work sum differs')
        guard = execution['fileQuarantine']
        if not set(guard['successfulFileOpenCounts']).issubset(guard['allowedFiles']) or execution['parameterInputRNGClosure'] is not True:
            raise ValueError('Actual numerical-read/input/parameter closure differs')
        if phase in ['shared', 'primary'] and any(x['path'] not in guard['deniedCanaries'] or x['path'] in guard['allowedFiles'] for x in s['banks']['P'].values()):
            raise ValueError('Primary P-complement denial differs')
    control_guard = executions['control']['fileQuarantine']
    if any(x['path'] not in control_guard['deniedCanaries'] or x['path'] in control_guard['allowedFiles'] for x in fields(primary)):
        raise ValueError('Control primary-array quarantine differs')
    if sum(e['nominalMatrixMACs'] for e in executions.values()) != report['totalNewMatrixMACs']:
        raise ValueError('Report complete work sum differs')
    streams = {}
    for stream in read(p / 'provenance.json')['losslessStreams']:
        raw = gzip.decompress((p / stream['published']).read_bytes())
        if hashlib.sha256(raw).hexdigest() != stream['sourceSHA256']:
            raise ValueError('Lossless complete query stream differs')
        streams[stream['source']] = [json.loads(x) for x in raw.splitlines()]
    rows = streams['queries.jsonl']
    if [(x['arm'], x['imageQueryIndex']) for x in rows] != [(a, i) for a in ['primary', 'control'] for i in range(1024)] or [x['imageQueryIndex'] for x in streams['context-effect.jsonl']] != list(range(1024)):
        raise ValueError('Complete actual query-address coverage differs')
    return {'schema': 'tce-source-complement-mechanics-v1', 'ownedSourcePins': len(r['ownedSources']), 'typedSharedActs': 1,
            'typedSourceActs': 2, 'typedRecipientActs': 2, 'sourcePageMembershipJoins': 2, 'independentBankSpliceWitnesses': 4,
            'queryRecords': len(rows), 'contextEffectRecords': len(streams['context-effect.jsonl']), 'sourceAffines': 9,
            'sourceAttentionCalls': 2, 'recipientPhysicalHeadBlocks': 48, 'fullByteFiles': len(receipt['files']),
            'NPYPayloadWitnesses': len(receipt['npyPayloadWitnesses']),
            'scope': 'Source/seal/complete-byte/splice/typed-input/physical-work/quarantine mechanics; no effect-size or general-image certification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['verify', 'publish', 'audit'])
    parser.add_argument('source')
    parser.add_argument('destination', nargs='?')
    args = parser.parse_args()
    value = publish(args.source, args.destination) if args.action == 'publish' else globals()[args.action](args.source)
    print(json.dumps(value, indent=2))
