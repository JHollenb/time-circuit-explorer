"""Source-only numerical closure, complete byte custody and independent row witnesses."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read,save,sha,identity
from projected_complement import fields
from publish_source_native import checked
from publish_source_complement import BYTES as BASE_BYTES,check_frame
from publish_deferred_value import items as prior_items
from source_reader_worker import PARAMETERS
from image_dual_analysis import NAMES

TEMPLATES=['publish_deferred_value.py','publish_image_dual.py','publish_source_complement.py','publish_sparse_text_kv.py']
BYTES=BASE_BYTES.split('joins=[]')[0]+r'''
sp=s['normalizedTextSplice'];q,hq=raw(sp['Q']);p,hp=raw(sp['P']);hybrid,hh=raw(sp['hybrid'])
assert hq['shape']==hp['shape']==hh['shape']==(1,512,3072)
assert hq['descr']==hp['descr']==hh['descr']=='<u2'
inside=hashlib.sha256();outside=hashlib.sha256()
for j in range(512):
 actual=hybrid[j*6144:(j+1)*6144];expected=(p if j in sp['support'] else q)[j*6144:(j+1)*6144]
 if actual!=expected:raise ValueError('Native normalized input splice differs')
 (inside if j in sp['support'] else outside).update(actual)
selections=[]
for pair in s['sourceSelections']:
 full,hf=raw(pair['full']);selected,hs=raw(pair['selected']);support=pair['support'];rowbytes=pair['rowBytes']
 if hf['descr']!=hs['descr'] or len(selected)!=len(support)*rowbytes:raise ValueError('Selection byte geometry differs')
 expected=b''.join(full[j*rowbytes:(j+1)*rowbytes] for j in support)
 if selected!=expected:raise ValueError('Source row/position selection differs')
 selections.append({'field':pair['field'],'support':support,'byteExact':True,'payloadSHA256':hashlib.sha256(selected).hexdigest(),'payloadBytes':len(selected)})
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),
 'npyPayloadWitnesses':witnesses,'sourceSelectionWitnesses':selections,
 'normalizedTextSpliceWitness':{'support':sp['support'],'selectedRowsByteExact':True,'outsideSupportByteExact':True,
 'selectedRowsSHA256':inside.hexdigest(),'outsideSupportSHA256':outside.hexdigest()}}))
'''


def items(p):
    s=read(p/'inputs.json');values=prior_items(p)+list(s['sourceReaderParentRecords'].values())
    old=Path(__file__).resolve().parent.parent/'artifacts/deferred-value-v1/result/actor.json'
    if sha(old)!=s['sourceReaderParentRecords']['actor.json']['sha256']:raise ValueError('Local assisted actor metadata differs')
    return values+list(fields(read(old)))


def declared(cut):
    slots={x['key']['value']:x['value'] for x in cut['cut']['manifest']['payload']['items']}
    context={x['key']['value']:x['value'] for x in slots['context']['items']}
    return slots,context


def verify(root):
    from mrun.client.api import Api
    root,collection,job,request=checked(root);p=root/'result';s=read(p/'inputs.json')
    roots=[job['result']['diagnostics']['cwd'],s['modelRoot'],str(Path(s['textKeySlice']['path']).parent)]
    roots += [str(Path(cut['cutFile']['path']).parent.parent) for cut in s['sourceCuts'].values()]
    for jkey,ckey in [('parentJobID','parentCustody'),('latestParentJobID','latestParentCustody'),
                      ('sourceJobID','sourceCustody'),('deferredParentJobID','deferredParentCustody'),('sourceReaderParentJobID','sourceReaderParentCustody')]:
        parent=Api().json('GET','/api/jobs/'+s[jkey])
        if parent['state']!='succeeded' or parent['payload_custody']!=s[ckey]:raise ValueError('Native/parent custody differs')
        roots.append(parent['result']['diagnostics']['cwd'])
    unique={};base=job['result']['diagnostics']['cwd']
    for item in items(p)+[{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()]:
        if item['path'] in unique and unique[item['path']]['sha256']!=item['sha256']:raise ValueError('Conflicting byte identity')
        unique[item['path']]={**unique.get(item['path'],{}),**item}
    actor=read(p/'actor.json');derived=actor['source_inputs']['fields']
    normalized={'Q':actor['text_modln']['fields']['x'],'P':actor['p_text_modln']['fields']['x'],
                'hybrid':actor['splice']['fields']['x'],'support':request['support']}
    selections=[{'field':n,'full':derived[f],'selected':derived[n],'support':request['support'],'rowBytes':b}
                for n,f,b in [('R','Ptext',6144),('ropeCos','fullRopeCos',512),('ropeSin','fullRopeSin',512)]]
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],
        input=json.dumps({'allowedRoots':roots,'items':list(unique.values()),'normalizedTextSplice':normalized,'sourceSelections':selections}),
        text=True,capture_output=True,timeout=240)
    if proc.returncode:raise RuntimeError(proc.stderr)
    receipt=json.loads(proc.stdout);receipt.update(schema='tce-source-reader-bytes-v1',jobID=job['job_id'],
        verifierSourceSHA256=sha(__file__),templateSourceHashes={n:sha(Path(__file__).with_name(n)) for n in TEMPLATES},
        actorSealSHA256=sha(p/'actor-seal.json'))
    save(root/'retained-array-verification.json',receipt)
    return {'files':len(receipt['files']),'bytes':receipt['totalBytesVerified'],'NPYPayloadWitnesses':len(receipt['npyPayloadWitnesses']),
            'normalizedInputSplices':1,'sourceRowPositionSelections':3}


def publish(root,destination):
    root,collection,job,request=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['templateSourceHashes']!={n:sha(Path(__file__).with_name(n)) for n in TEMPLATES}:
        raise ValueError('Byte verifier source changed')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for n in p.iterdir():
        if n.suffix=='.jsonl':
            target=out/(n.name+'.gz');raw=n.read_bytes();target.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless compression differs')
            streams.append({'source':n.name,'sourceSHA256':sha(n),'sourceBytes':len(raw),'published':target.name,'publishedSHA256':sha(target),'publishedBytes':target.stat().st_size})
        else:shutil.copyfile(n,out/n.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    shutil.copyfile(root/'prelease-note-rejection.json',out/'prelease-note-rejection.json')
    save(out/'scheduler.json',{k:job[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-source-reader-publication-v1','jobID':job['job_id'],
        'requestSHA256':request['requestSHA256'],'payloadCustody':collection['custody'],'completeCollectedInventory':collection['inventory'],
        'losslessStreams':streams,'scope':'Four-query reader from native source-state interface, original13 maps/fullK/nativeQ/deferredV/ownMLP/nextQ paid. No archived operands beforeseal, no image or zero-neural completion.'})
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
    if set(weights)!=set(PARAMETERS) or sum(x['bytes'] for x in execution['loadedParameters'])!=301990912 or not execution['parameterInputRNGClosure']:
        raise ValueError('Hydrated maps/input/RNG closure differs')
    guard=execution['fileQuarantine'];allowed,denied=set(guard['allowedFiles']),set(guard['deniedCanaries'])
    expected_allowed={str(Path(s['modelRoot'])/n) for n in s['imageAssets']}
    for role,cut in s['sourceCuts'].items():
        if role not in ['P','Q']:continue
        expected_allowed.add(cut['cutFile']['path'])
        for slot,page in cut['pages'].items():
            if slot in (['image','text','context'] if role=='Q' else ['text','context']):expected_allowed.update(page['files'])
    if allowed!=expected_allowed or any(n.endswith('.npy') for n in allowed) or not set(s['deniedNumericalInputs']).issubset(denied) or denied&allowed or not set(guard['successfulFileOpenCounts']).issubset(allowed):
        raise ValueError('Source-only numerical input closure differs')
    for n in ['Ptext','txtMod','fullRopeCos','fullRopeSin']:
        slots,context=declared(s['sourceCuts']['P'])
        expected=slots['text'] if n=='Ptext' else context['txt_mod'] if n=='txtMod' else context['rope']['items'][0 if n=='fullRopeCos' else 1]
        if actor['source_inputs']['fields'][n]['payloadSHA256']!=expected['sha256']:raise ValueError('Full source payload differs:'+n)
    acts=[]
    def emitted(act,*names):return {n:actor[act]['fields'][n]['payloadSHA256'] for n in names}
    for name,act in actor.items():
        acts.append(act);expected={n:weights[n] for n in act['sourceFrame']['slots'] if n in weights}
        if name in ['image_modln','text_modln','p_text_modln']:
            role='P' if name=='p_text_modln' else 'Q';slot='image' if name=='image_modln' else 'text';mod='img_mod' if slot=='image' else 'txt_mod'
            slots,context=declared(s['sourceCuts'][role]);expected.update(rows=slots[slot]['sha256'],mod=context[mod]['sha256'])
        elif name=='source_inputs':
            slots,context=declared(s['sourceCuts']['P']);expected.update(text=slots['text']['sha256'],mod=context['txt_mod']['sha256'],cosine=context['rope']['items'][0]['sha256'],sine=context['rope']['items'][1]['sha256'])
        elif name=='splice':expected.update(Qtext=emitted('text_modln','x')['x'],Ptext=emitted('p_text_modln','x')['x'])
        elif name=='query':expected.update(emitted('source_inputs','R','txtMod','ropeCos','ropeSin'))
        elif name=='keys':
            _,context=declared(s['sourceCuts']['Q']);expected.update(imageInputs=emitted('image_modln','x')['x'],textInputs=emitted('splice','x')['x'],cosine=context['rope']['items'][0]['sha256'],sine=context['rope']['items'][1]['sha256'])
        elif name=='allocate':expected.update(Q=emitted('query','Q')['Q'],K=emitted('keys','K')['K'])
        elif name=='value':expected.update(probability=emitted('allocate','probability')['probability'],textInputs=emitted('splice','x')['x'],imageInputs=emitted('image_modln','x')['x'])
        elif name=='source':expected.update(H=emitted('value','H')['H'],**emitted('source_inputs','R','txtMod'))
        elif name=='next_query':expected.update(rows=emitted('source','T')['T'],**emitted('source_inputs','txtMod','ropeCos','ropeSin'))
        else:raise ValueError('Unknown typed operation')
        check_frame(act,expected,r['requestSHA256'])
        for n in act['evidence']['metrics']['capturedPayloads']:
            x=act['fields'][n]
            if act['outputFrame']['slots'][n]['abi']!={'shape':x['shape'],'dtype':x['logicalDtype']}:raise ValueError('Explicit output ABI differs')
    joins=[('splice','QtextFrame','text_modln'),('splice','PtextFrame','p_text_modln'),('query','sourceInputsFrame','source_inputs'),
        ('keys','imageModLNFrame','image_modln'),('keys','splicedTextFrame','splice'),('allocate','queryFrame','query'),('allocate','keyFrame','keys'),
        ('value','allocationFrame','allocate'),('value','splicedTextFrame','splice'),('value','imageModLNFrame','image_modln'),
        ('source','headOutputFrame','value'),('source','sourceInputsFrame','source_inputs'),('next_query','sourceOutputFrame','source'),('next_query','sourceInputsFrame','source_inputs')]
    if any(actor[a]['sourceFrame']['metadata'][key]!=actor[b]['outputFrame']['frame_id'] for a,key,b in joins):raise ValueError('Own state/producer/consumer join differs')
    if execution['nominalMatrixMACs']!=15495856128 or sum(x['nominalMACs'] for x in execution['operations'])!=15495856128 or report['totalNewMatrixMACs']!=15495856128 or report['workByPhase']['postSealAnalysis']!=0:
        raise ValueError('Actual work differs')
    if not v['normalizedTextSpliceWitness']['selectedRowsByteExact'] or not v['normalizedTextSpliceWitness']['outsideSupportByteExact'] or len(v['sourceSelectionWitnesses'])!=3 or not all(x['byteExact'] for x in v['sourceSelectionWitnesses']):
        raise ValueError('Independent source/splice witness absent')
    raw=gzip.decompress((p/'observations.jsonl.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest()!=read(p/'provenance.json')['losslessStreams'][0]['sourceSHA256']:raise ValueError('Lossless observations differ')
    rows=[json.loads(line) for line in raw.splitlines()];addresses=[(x['field'],x['panel'],x['row'],x.get('head')) for x in rows]
    expected={(n,'whole-row',j,None) for n in NAMES for j in r['support']}
    expected|={(n,'query-head',j,h) for n in ['nextQProjected','nextQNormalized','nextQ'] for j in r['support'] for h in range(24)}
    expected|={('K','key-head-input',j,h) for j in range(1536) for h in range(24)}
    expected|={('allocation','query-head-input',j,h) for j in r['support'] for h in range(24)}
    if len(addresses)!=len(set(addresses)) or set(addresses)!=expected:raise ValueError('Complete unique observation census differs')
    return {'schema':'tce-source-reader-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'typedActs':len(acts),
        'explicitOutputABIs':sum(len(a['evidence']['metrics']['capturedPayloads']) for a in acts),'ownStateJoins':len(joins),
        'observations':len(rows),'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),
        'normalizedTextInputSplices':1,'sourceRowPositionSelections':3,'newFullValueProjections':0,
        'scope':'Source-only numerical closure/bytes/actual typed own-state work mechanics; no effect-size, image/generalization/speedup/zero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source),indent=2))
