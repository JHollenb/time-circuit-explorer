"""Full bytes, normalized-input splice and typed own-state value-producer audit."""
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
from publish_image_dual import BYTES as OLD_BYTES
from deferred_value_worker import PARAMETERS, VALUE_PARAMETERS
from propagated_context import RETURN_PARAMETERS, NEXT_PARAMETERS
from image_dual_analysis import NAMES

TEMPLATES=['publish_image_dual.py','publish_source_complement.py','publish_sparse_text_kv.py']
BYTES=OLD_BYTES.rsplit('print(json.dumps',1)[0]+r'''
sp=s['normalizedTextSplice'];q,hq=raw(sp['Q']);p,hp=raw(sp['P']);hybrid,hh=raw(sp['hybrid'])
assert hq['shape']==hp['shape']==hh['shape']==(1,512,3072)
assert hq['descr']==hp['descr']==hh['descr']=='<u2'
inside=hashlib.sha256();outside=hashlib.sha256()
for j in range(512):
 actual=hybrid[j*6144:(j+1)*6144];expected=(p if j in sp['support'] else q)[j*6144:(j+1)*6144]
 if actual!=expected:raise ValueError('Native normalized input splice differs')
 (inside if j in sp['support'] else outside).update(actual)
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),
 'npyPayloadWitnesses':witnesses,'bankSpliceWitnesses':joins,'spliceBytesReadAgain':reread,
 'textPrefixWitness':{'byteExact':True,'sourceSHA256':prefix['source']['sha256'],
 'sourceHeaderSHA256':prefix['sourceHeaderSHA256'],'payloadSHA256':prefix['payloadSHA256'],'bytes':len(selected)},
 'normalizedTextSpliceWitness':{'support':sp['support'],'selectedRowsByteExact':True,'outsideSupportByteExact':True,
 'selectedRowsSHA256':inside.hexdigest(),'outsideSupportSHA256':outside.hexdigest()}}))
'''


def items(p):
    s=read(p/'inputs.json');values=[]
    for key in ['parentRecords','latestParentRecords','deferredParentRecords']:values+=list(s[key].values())
    values += [{'path':n,'sha256':h} for n,h in s['sourceMetadataFiles'].items()]
    for cut in s['sourceCuts'].values():
        values.append(cut['cutFile']);values += [{'path':n,'sha256':h} for page in cut['pages'].values() for n,h in page['files'].items()]
    values += list(fields({n:s[n] for n in ['selected','banks','reuse','textKeySlice','matchedControlFields','controlModLN']}))
    for n in ['targets.json','actor.json','report.json']:values+=list(fields(read(p/n)))
    values += [{'path':str(Path(s['modelRoot'])/n),'sha256':h} for n,h in s['imageAssets'].items()]
    return values


def verify(root):
    from mrun.client.api import Api
    root,collection,job,request=checked(root);p=root/'result';s=read(p/'inputs.json')
    roots=[job['result']['diagnostics']['cwd'],s['modelRoot'],str(Path(s['textKeySlice']['path']).parent)]
    roots += [str(Path(cut['cutFile']['path']).parent.parent) for cut in s['sourceCuts'].values()]
    for jkey,ckey in [('parentJobID','parentCustody'),('latestParentJobID','latestParentCustody'),
                      ('sourceJobID','sourceCustody'),('deferredParentJobID','deferredParentCustody')]:
        parent=Api().json('GET','/api/jobs/'+s[jkey])
        if parent['state']!='succeeded' or parent['payload_custody']!=s[ckey]:raise ValueError('Native/parent custody differs')
        roots.append(parent['result']['diagnostics']['cwd'])
    unique={};base=job['result']['diagnostics']['cwd']
    for item in items(p)+[{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()]:
        if item['path'] in unique and unique[item['path']]['sha256']!=item['sha256']:raise ValueError('Conflicting byte identity')
        unique[item['path']]={**unique.get(item['path'],{}),**item}
    actor=read(p/'actor.json')
    splices=[{'arm':'actor','field':'K','support':request['support'],'selected':s['selected']['K'],
              'bank':s['textKeySlice'],'hybrid':actor['allocate']['fields']['textK_hybrid']}]
    normalized={'Q':actor['text_modln']['fields']['x'],'P':actor['p_text_modln']['fields']['x'],
                'hybrid':actor['splice']['fields']['x'],'support':request['support']}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],
        input=json.dumps({'allowedRoots':roots,'items':list(unique.values()),'spliceChecks':splices,
            'textKeySlice':s['textKeySlice'],'normalizedTextSplice':normalized}),text=True,capture_output=True,timeout=240)
    if proc.returncode:raise RuntimeError(proc.stderr)
    receipt=json.loads(proc.stdout);receipt.update(schema='tce-deferred-value-bytes-v1',jobID=job['job_id'],
        verifierSourceSHA256=sha(__file__),templateSourceHashes={n:sha(Path(__file__).with_name(n)) for n in TEMPLATES},
        actorSealSHA256=sha(p/'actor-seal.json'))
    save(root/'retained-array-verification.json',receipt)
    return {'files':len(receipt['files']),'bytes':receipt['totalBytesVerified'],'NPYPayloadWitnesses':len(receipt['npyPayloadWitnesses']),
            'normalizedInputSplices':1,'textKeySplices':1,'textPrefixWitnesses':1}


def publish(root,destination):
    root,collection,job,request=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['templateSourceHashes']!={n:sha(Path(__file__).with_name(n)) for n in TEMPLATES}:
        raise ValueError('Byte verifier source changed')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for n in p.iterdir():
        if n.suffix=='.jsonl':
            target=out/(n.name+'.gz');raw=n.read_bytes();target.write_bytes(gzip.compress(raw,mtime=0))
            streams.append({'source':n.name,'sourceSHA256':sha(n),'published':target.name,'publishedSHA256':sha(target)})
        else:shutil.copyfile(n,out/n.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'scheduler.json',{k:job[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-deferred-value-publication-v1','jobID':job['job_id'],
        'requestSHA256':request['requestSHA256'],'payloadCustody':collection['custody'],'completeCollectedInventory':collection['inventory'],
        'losslessStreams':streams,'scope':'Current-state deferred value producer at four demanded queries; original maps/state/allocation/Pnorm paid, no full V operand beforeseal, no suffix/image/zero-neural completion.'})
    save(out/'files.json',{n.name:sha(n) for n in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(n.stat().st_size for n in out.iterdir())}


def audit(source):
    p=Path(source);r,s,actor,execution,report,v=[read(p/n) for n in ['request.json','inputs.json','actor.json','actor-execution.json','report.json','retained-array-verification.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Immutable publication changed:'+n)
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]:raise ValueError('Executed source changed:'+n)
    if v['verifierSourceSHA256']!=sha(__file__) or v['templateSourceHashes']!={n:sha(Path(__file__).with_name(n)) for n in TEMPLATES}:
        raise ValueError('Full byte verifier changed')
    for x in items(p):
        if v['files'][x['path']]['sha256']!=x['sha256'] or (x.get('payloadSHA256') and v['npyPayloadWitnesses'][x['path']]['payloadSHA256']!=x['payloadSHA256']):
            raise ValueError('Complete consumed/emitted byte custody absent')
    seal=read(p/'actor-seal.json')
    if seal['actorSHA256']!=sha(p/'actor.json') or seal['requestSHA256']!=r['requestSHA256'] or v['actorSealSHA256']!=sha(p/'actor-seal.json'):
        raise ValueError('Actor release differs')
    weights={x['name']:x['payloadSHA256'] for x in execution['loadedParameters']}
    if set(weights)!=set(PARAMETERS) or not execution['parameterInputRNGClosure']:raise ValueError('Hydrated maps/input/RNG closure differs')
    guard=execution['fileQuarantine'];allowed,denied=set(guard['allowedFiles']),set(guard['deniedCanaries'])
    if not set(s['deniedNumericalInputs']).issubset(denied) or denied&allowed or not set(guard['successfulFileOpenCounts']).issubset(allowed):
        raise ValueError('Actual numerical input closure differs')
    if not {s['selected']['V']['path'],*[s['banks'][n]['V']['path'] for n in ['P','Q']]}.issubset(denied):
        raise ValueError('Full/selected V was not denied')
    acts=[]
    for name,act in actor.items():
        if not isinstance(act,dict) or 'sourceFrame' not in act:continue
        acts.append(act);expected={n:weights[n] for n in act['sourceFrame']['slots'] if n in weights}
        if name in ['image_modln','text_modln','p_text_modln']:
            role='P' if name=='p_text_modln' else 'Q';slot='image' if name=='image_modln' else 'text';mod='img_mod' if slot=='image' else 'txt_mod'
            declared={x['key']['value']:x['value'] for x in s['sourceCuts'][role]['cut']['manifest']['payload']['items']}
            context={x['key']['value']:x['value'] for x in declared['context']['items']}
            expected.update(rows=declared[slot]['sha256'],mod=context[mod]['sha256'])
        elif name=='splice':expected.update(Qtext=actor['text_modln']['fields']['x']['payloadSHA256'],Ptext=actor['p_text_modln']['fields']['x']['payloadSHA256'])
        elif name=='allocate':expected.update(x=actor['image_modln']['fields']['x']['payloadSHA256'],
            **{n:x['payloadSHA256'] for n,x in s['reuse'].items()},Q=s['selected']['Q']['payloadSHA256'],
            selectedK=s['selected']['K']['payloadSHA256'],textK=s['textKeySlice']['payloadSHA256'])
        elif name=='value':expected.update(probability=actor['allocate']['fields']['probability']['payloadSHA256'],
            textInputs=actor['splice']['fields']['x']['payloadSHA256'],imageInputs=actor['image_modln']['fields']['x']['payloadSHA256'])
        elif name=='source':expected.update(H=actor['value']['fields']['H']['payloadSHA256'],**{n:s['selected'][n]['payloadSHA256'] for n in ['R','txtMod']})
        elif name=='next_query':expected.update(rows=actor['source']['fields']['T']['payloadSHA256'],**{n:s['selected'][n]['payloadSHA256'] for n in ['txtMod','ropeCos','ropeSin']})
        else:raise ValueError('Unknown typed operation')
        check_frame(act,expected,r['requestSHA256'])
        for n in act['evidence']['metrics']['capturedPayloads']:
            x=act['fields'][n]
            if act['outputFrame']['slots'][n]['abi']!={'shape':x['shape'],'dtype':x['logicalDtype']}:raise ValueError('Explicit output ABI differs')
    joins=[('splice','QtextFrame','text_modln'),('splice','PtextFrame','p_text_modln'),('allocate','imageModLNFrame','image_modln'),
        ('value','allocationFrame','allocate'),('value','splicedTextFrame','splice'),('value','imageModLNFrame','image_modln'),
        ('source','headOutputFrame','value'),('next_query','sourceOutputFrame','source')]
    if any(actor[a]['sourceFrame']['metadata'][key]!=actor[b]['outputFrame']['frame_id'] for a,key,b in joins):raise ValueError('Own state/producer/consumer join differs')
    if execution['nominalMatrixMACs']!=r['plannedMatrixMACs']['actor'] or sum(x['nominalMACs'] for x in execution['operations'])!=execution['nominalMatrixMACs']:
        raise ValueError('Actual actor work differs')
    if report['totalNewMatrixMACs']!=1874853888 or report['diagnosticMatrixOperations'][0]['nominalMACs']!=18874368:raise ValueError('Complete new work differs')
    if not v['normalizedTextSpliceWitness']['selectedRowsByteExact'] or not v['normalizedTextSpliceWitness']['outsideSupportByteExact'] or not v['textPrefixWitness']['byteExact']:
        raise ValueError('Normalized/native input byte splice absent')
    raw=gzip.decompress((p/'observations.jsonl.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest()!=read(p/'provenance.json')['losslessStreams'][0]['sourceSHA256']:raise ValueError('Lossless observations differ')
    rows=[json.loads(line) for line in raw.splitlines()];addresses=[(x['field'],x['panel'],x['row'],x.get('head')) for x in rows]
    expected={(n,'whole-row',j,None) for n in NAMES for j in r['support']}
    expected|={(n,'query-head',j,h) for n in ['nextQProjected','nextQNormalized','nextQ'] for j in r['support'] for h in range(24)}
    expected|={('H_value_policy','query-head-policy',j,h) for j in r['support'] for h in range(24)}
    if len(addresses)!=len(set(addresses)) or set(addresses)!=expected:raise ValueError('Complete unique observation census differs')
    return {'schema':'tce-deferred-value-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'typedActs':len(acts),
        'explicitOutputABIs':sum(len(a['evidence']['metrics']['capturedPayloads']) for a in acts),'ownStateJoins':len(joins),
        'observations':len(rows),'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),
        'normalizedTextInputSplices':1,'textKeySplices':1,'textPrefixWitnesses':1,'newFullValueProjections':0,
        'scope':'Source/bytes/actual typed own-state/value/consumer/work/denial mechanics; no effect-size or semantic/image/generalization/speedup/zero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source),indent=2))
