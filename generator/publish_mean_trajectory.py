"""Complete bytes, explicit text products and coupled own-state publication."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read,save,sha,identity
from publish_source_native import checked
from publish_terminal_trajectory import items as base_items,check_act
from projected_complement import fields
from publish_sparse_text_kv import NPY_BYTES

BYTES=NPY_BYTES.split('print(json.dumps')[0]+r'''
checkpoint=Path(s['checkpoint']);parameters={};slices={}
def raw_npy(item):
 with Path(item['path']).open('rb') as f:
  magic=f.read(8);n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0];f.read(n);return f.read()
def witness(item,raw):
 digest=hashlib.sha256(raw).hexdigest();key=str(item['step'])+'.'+str(item['blockIndex'])+'.'+item['name']
 if digest!=item['payloadSHA256'] or len(raw)!=item['bytes']:raise ValueError('Actual current-support physical map bytes differ:'+key)
 slices[key]={'payloadSHA256':digest,'bytes':len(raw),'shape':item['shape'],'support':item['support']}
with checkpoint.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n
 if set(header)-{'__metadata__'}!=set(x['name'] for x in s['loaded']):raise ValueError('Full original parameter census differs')
 for item in s['loaded']:
  entry=header[item['name']];a,b=entry['data_offsets'];f.seek(base+a);raw=f.read(b-a)
  if entry['dtype']!='BF16' or entry['shape']!=item['shape'] or [a,b]!=item['dataOffsets'] or b-a!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['payloadSHA256']:raise ValueError('Original parameter bytes/geometry differ')
  parameters[item['name']]={'payloadSHA256':item['payloadSHA256'],'bytes':len(raw),'shape':entry['shape']}
  for physical in [x for x in s['physicalMaps'] if x['sourceParameter']==item['name']]:
   name=physical['name'];width=3072*2
   if name=='QKVG':derived=raw[:18432*width]
   elif name=='KV':derived=raw[3072*width:9216*width]
   elif name=='QG':derived=raw[:3072*width]+raw[9216*width:18432*width]
   elif name=='UAnchor':derived=raw[18432*width:27648*width]
   else:
    support_item=s['candidateFields'][physical['support']['field']];support_raw=raw_npy(support_item)
    if hashlib.sha256(support_raw).hexdigest()!=physical['support']['payloadSHA256']:raise ValueError('Support payload binding differs')
    support=list(struct.unpack('<2304q',support_raw))
    if support!=sorted(set(support)) or support[0]<0 or support[-1]>=9216:raise ValueError('Support ascending/source range differs')
    if name=='USelected':derived=b''.join(raw[(18432+j)*width:(18433+j)*width] for j in support)
    elif name=='outSelected':
     runs=[]
     for j in support:
      if runs and runs[-1][1]==j:runs[-1][1]=j+1
      else:runs.append([j,j+1])
     chunks=[]
     for row in range(3072):
      start=row*12288*2
      chunks.append(raw[start:start+3072*2]+b''.join(raw[start+(3072+a)*2:start+(3072+b)*2] for a,b in runs))
     derived=b''.join(chunks)
    else:raise ValueError('Unknown physical map')
   witness(physical,derived)
completion_witnesses={}
for completion in s['completionRecords']:
 fields=completion['fields'];n=completion['liveRows'];terminal=completion['index']==19
 complete=raw_npy(fields['completedUp']);selected=raw_npy(fields['upRaw']);anchor=raw_npy(fields['anchorUp'])
 support=list(struct.unpack('<2304q',raw_npy(fields['selection_support'])));positions={j:i for i,j in enumerate(support)}
 if len(complete)!=n*9216*2 or len(selected)!=n*2304*2 or len(anchor)!=(1 if terminal else 2)*9216*2:raise ValueError('Completion word inventory differs')
 for channel in range(9216):
  column=complete[2*channel::18432]+complete[2*channel+1::18432]
  if channel in positions:
   pos=positions[channel];expected=selected[2*pos::4608]+selected[2*pos+1::4608]
  else:
   lo=anchor[2*channel:2*channel+1];hi=anchor[2*channel+1:2*channel+2]
   if terminal:expected=lo*n+hi*n
   else:expected=lo*512+anchor[18432+2*channel:18433+2*channel]*1024+hi*512+anchor[18433+2*channel:18434+2*channel]*1024
  if column!=expected:raise ValueError('Complete own anchor broadcast/selected overwrite differs')
 key=str(completion['step'])+'.'+str(completion['index'])
 completion_witnesses[key]={'liveRows':n,'anchorRows':1 if terminal else 2,'completeChannels':9216,'selectedChannels':2304,'completedPayloadSHA256':fields['completedUp']['payloadSHA256'],'anchorPayloadSHA256':fields['anchorUp']['payloadSHA256'],'selectedPayloadSHA256':fields['upRaw']['payloadSHA256'],'allWordsExact':True}
row_witnesses={}
for k in range(4):
 item=s['candidateFields']['terminalRows'+str(k)];raw=raw_npy(item);n=512*3072*2
 if item['shape']!=[1,1536,3072] or len(raw)!=1536*3072*2:raise ValueError('Combined own-row geometry differs')
 th=hashlib.sha256(raw[:n]).hexdigest();ih=hashlib.sha256(raw[n:]).hexdigest()
 if th!=s['candidateFields']['b_single18_textOut'+str(k)]['payloadSHA256'] or ih!=s['candidateFields']['b_single18_imageOut'+str(k)]['payloadSHA256']:raise ValueError('Own terminal row byte concatenation differs')
 row_witnesses[str(k)]={'ownTextPayloadSHA256':th,'ownImagePayloadSHA256':ih,'byteExact':True}
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,
 'originalParameterWitnesses':parameters,'physicalSliceWitnesses':slices,'terminalOwnRowWitnesses':row_witnesses,'completionBroadcastWitnesses':completion_witnesses}))
'''

def items(p):
    return base_items(p)+list(fields(read(p/'comparators.json')))


def verify(root):
    from mrun.client.api import Api
    root,c,j,r=checked(root);p=root/'result';s=read(p/'inputs.json');e=read(p/'execution.json');api=Api();roots=[]
    for job_id,custody in [(s['nativeJobID'],s['nativeCustody']),(s['noiseJobID'],s['noiseCustody']),(s['priorTrajectoryJobID'],s['priorTrajectoryCustody'])]:
        parent=api.json('GET','/api/jobs/'+job_id)
        if parent['state']!='succeeded' or parent['payload_custody']!=custody:raise ValueError('Original parent custody differs')
        roots.append(parent['result']['diagnostics']['cwd'])
    for item in read(p/'comparators.json').values():
        parent=api.json('GET','/api/jobs/'+item['jobID'])
        if parent['state']!='succeeded' or parent['payload_custody']!=item['custody']:raise ValueError('Post-seal comparator custody differs')
        roots.append(parent['result']['diagnostics']['cwd'])
    base=j['result']['diagnostics']['cwd'];values=items(p)+[{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()]
    values.append({'path':s['priorTrajectoryReportPath'],'sha256':s['priorTrajectoryReportSHA256']});unique={}
    for x in values:
        if x['path'] in unique and unique[x['path']]['sha256']!=x['sha256']:raise ValueError('Conflicting byte identity')
        unique[x['path']]={**unique.get(x['path'],{}),**x}
    spec={'items':list(unique.values()),'allowedRoots':[base,s['modelRoot'],*roots],
        'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),'loaded':e['loadedTransformerParameters'],
        'physicalMaps':e['physicalSingleMaps'],'completionRecords':e['completionRecords'],'candidateFields':read(p/'candidate.json')['fields']}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=600)
    if proc.returncode:raise RuntimeError(proc.stderr)
    v=json.loads(proc.stdout);v.update(schema='tce-mean-up-trajectory-bytes-v1',jobID=j['job_id'],verifierSourceSHA256=sha(__file__),
        sharedAuditSourceSHA256=sha(Path(__file__).with_name('publish_terminal_trajectory.py')),byteTemplateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
        candidateSealSHA256=sha(p/'candidate-seal.json'),scope='Complete file/NPY/original parameter/physical slice and own terminal concatenation bytes. SSH byte operations only, no tensor numerics.')
    save(root/'retained-array-verification.json',v)
    return {'files':len(v['files']),'bytes':v['totalBytesVerified'],'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'originalParameterWitnesses':len(v['originalParameterWitnesses']),'uniquePhysicalSlices':len(v['physicalSliceWitnesses'])}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Executed verifier differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for n in p.iterdir():
        if n.suffix=='.jsonl':
            raw=n.read_bytes();target=out/(n.name+'.gz');target.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless publication differs')
            streams.append({'source':n.name,'sourceSHA256':sha(n),'published':target.name,'publishedSHA256':sha(target),'sourceBytes':len(raw),'publishedBytes':target.stat().st_size})
        else:shutil.copyfile(n,out/n.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'scheduler.json',{k:j[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-mean-up-trajectory-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],
        'completeCollectedInventory':c['inventory'],'losslessStreams':streams,'scope':'Frozen current-stream mean-Up law through80ownMLPs/4clocks and actualVAE. Fullbanks retainednearcompute; originallearnedworkpaid. No newreference/fit/sweep/zero-neuralclaim.'})
    save(out/'files.json',{n.name:sha(n) for n in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(n.stat().st_size for n in out.iterdir())}


def audit(source):
    p=Path(source);r,s,c,e,t,report,v=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json','targets.json','report.json','retained-array-verification.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Publication differs:'+n)
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]:raise ValueError('Frozen science source differs:'+n)
    for key,name in [('verifierSourceSHA256','publish_mean_trajectory.py'),('sharedAuditSourceSHA256','publish_terminal_trajectory.py'),('byteTemplateSourceSHA256','publish_sparse_text_kv.py')]:
        if v[key]!=sha(Path(__file__).with_name(name)):raise ValueError('Executed audit source differs')
    for x in items(p):
        if v['files'][x['path']]['sha256']!=x['sha256'] or (x.get('payloadSHA256') and v['npyPayloadWitnesses'][x['path']]['payloadSHA256']!=x['payloadSHA256']):raise ValueError('Complete byte witness absent')
    if read(p/'candidate-seal.json')!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json'),'imageSHA256':sha(p/'candidate-A.png')}:raise ValueError('Candidate seal differs')
    if report['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or sha(p/'candidate-A.png')!=c['image']['sha256'] or sha(p/'native-A.png')!=t['image']['sha256']:raise ValueError('Actual release/image differs')
    if c['schedules']!=[s['schedule']]*4:raise ValueError('Pinned four-clock schedule differs')
    witnesses=v['terminalOwnRowWitnesses']
    if set(witnesses)!=set(map(str,range(4))):raise ValueError('Terminal own-row byte coverage differs')
    for step in range(4):
        w=witnesses[str(step)]
        if not w['byteExact'] or w['ownTextPayloadSHA256']!=c['fields']['b_single18_textOut'+str(step)]['payloadSHA256'] or w['ownImagePayloadSHA256']!=c['fields']['b_single18_imageOut'+str(step)]['payloadSHA256']:raise ValueError('Terminal own-row witness differs')
    manifest=identity({'transformer':e['loadedTransformerParameters'],'vaeCheckpointSHA256':s['imageAssets']['vae/diffusion_pytorch_model.safetensors'],'meanTrajectorySourceSHA256':r['sourceHashes']['mean_up_trajectory.py']})
    if manifest!=e['parameterManifestSHA256'] or not e['sourceParameterBufferRNGClosure'] or e['numericalProfile']!=s['numericalProfile']:raise ValueError('Source/profile closure differs')
    if len(e['loadedTransformerParameters'])!=169 or any(v['originalParameterWitnesses'][x['name']]['payloadSHA256']!=x['payloadSHA256'] for x in e['loadedTransformerParameters']):raise ValueError('Original weights differ')
    maps=e['physicalSingleMaps']
    expected_maps=[(k,index,n) for k in range(4) for index in range(20) for n in (['KV','QG','USelected','UAnchor'] if index==19 else ['QKVG','USelected','UAnchor'])]
    if sorted((x['step'],x['blockIndex'],x['name']) for x in maps)!=sorted(expected_maps):raise ValueError('Physical current-support map coverage differs')
    for x in maps:
        witness=v['physicalSliceWitnesses'][str(x['step'])+'.'+str(x['blockIndex'])+'.'+x['name']]
        if witness!={'payloadSHA256':x['payloadSHA256'],'bytes':x['bytes'],'shape':x['shape'],'support':x['support']}:raise ValueError('Physical source/current-support witness differs')
    if e['mapMaterializationScope']!={'activeSets':0,'maxActiveSets':1,'maxLogicalMapBytes':113246208}:raise ValueError('Ephemeral map scope differs')
    expected_allowed={s['E']['path']}|{s['noise'][n]['path'] for n in ['raw','packed','ids']}|{str(Path(s['modelRoot'])/n) for n in s['imageAssets']};guard=e['fileQuarantine']
    if set(guard['allowedFiles'])!=expected_allowed or set(guard['deniedCanaries'])!=set(s['deniedNumericalInputs']) or expected_allowed&set(guard['deniedCanaries']) or not set(guard['successfulFileOpenCounts']).issubset(expected_allowed):raise ValueError('Numerical source isolation differs')
    acts=e['typedActs'];wanted_stages=['sourceNorms']+[n+str(k) for k in range(4) for n in ['denoise','commit']]+['decode']
    if [a['stage'] for a in acts]!=wanted_stages or c['counts']!=r['expectedCalls'] or len(c['fields'])!=2274:raise ValueError('Actual component census differs')
    check_act(acts[0],{n:c['fields'][n] for n in ['sourceUpNorm','sourceDownNorm']},r['requestSHA256'],manifest)
    if e['sourceNormPreparation']!={'coefficientSquares':1132462080,'roots':368640,'FP32NormBytes':1474560} or e['gateActivations']!=1113587712:raise ValueError('Paid source-norm/gate account differs')
    abis=2;joins=0;latent=s['noise']['packed']['payloadSHA256'];previous=None
    for step in range(4):
        d,commit=acts[1+2*step:3+2*step];src=d['sourceFrame'];meta=src['metadata']
        if meta['sourceNormProducerFrameID']!=acts[0]['outputFrame']['frame_id'] or any(src['slots'][n]['fingerprint']!=c['fields'][n]['payloadSHA256'] for n in ['sourceUpNorm','sourceDownNorm']):raise ValueError('Paid norm producer/selection-consumer join differs')
        if src['slots']['hidden_states']['fingerprint']!=latent or src['slots']['encoder_hidden_states']['fingerprint']!=s['E']['payloadSHA256'] or src['slots']['img_ids']['fingerprint']!=s['noise']['ids']['payloadSHA256'] or meta['precedingCommitFrameID']!=previous:raise ValueError('Own denoiser input differs')
        base=['imageStart','velocity','context','terminalRows','terminalImage','singleMod','temb','cosine','sine']
        outputs={n:c['fields'][('C' if n=='context' else n)+str(step)] for n in base}
        blocks=[b for b in e['blockRecords'] if b['step']==step]
        addresses=[(kind,index) for kind,count in [('joint',5),('single',20)] for index in range(count)]
        if [(b['kind'],b['index']) for b in blocks]!=addresses:raise ValueError('Block execution order differs')
        ih=outputs['imageStart']['payloadSHA256'];th=outputs['context']['payloadSHA256']
        for b in blocks:
            if b['imageInputPayloadSHA256']!=ih or b['textInputPayloadSHA256']!=th:raise ValueError('Coupled own-state handoff differs')
            for label in ['imageOutput','textOutput','alpha','textRead']:
                field=b[label+'Field'];digest=b[label+'PayloadSHA256']
                if field is None:
                    if label=='imageOutput' or (b['kind'],b['index'])!=('single',19) or digest is not None:raise ValueError('Text last-use omission differs')
                    continue
                item=c['fields'][field]
                if item['payloadSHA256']!=digest:raise ValueError('Actual block output binding differs')
                outputs[field[:-1]]=item
            expected_moments={} if (b['kind'],b['index'])==('single',19) else {'mean':[1,24,128],'covariance':[1,24,128,128],'logZI':[1,24,512],'logZT':[1,24,512],'quadratic':[1,24,512]}
            if set(b['moments'])!=set(expected_moments):raise ValueError('Moment field coverage differs')
            for name,shape in expected_moments.items():
                item=b['moments'][name];label='b_'+b['kind']+str(b['index'])+'_moment_'+name+str(step);field=c['fields'][label]
                if item!={'field':label,'payloadSHA256':field['payloadSHA256']} or field['shape']!=shape or field['logicalDtype']!='torch.float32' or field['storedDtype']!='float32':raise ValueError('Actual compact moment binding differs')
                outputs[label[:-1]]=field
            ih=b['imageOutputPayloadSHA256'];th=b['textOutputPayloadSHA256'];joins+=1
        if ih!=outputs['terminalImage']['payloadSHA256']:raise ValueError('Final image state differs')
        selections=[x for x in e['selectionRecords'] if x['step']==step]
        if [x['index'] for x in selections]!=list(range(20)):raise ValueError('Current support coverage differs')
        for selection in selections:
            index=selection['index']
            if selection['liveRows']!=(1024 if index==19 else 1536):raise ValueError('Current selection row demand differs')
            for name,item in selection['fields'].items():
                field=c['fields'][item['field']]
                if item['payloadSHA256']!=field['payloadSHA256']:raise ValueError('Current selector typed binding differs')
                outputs[item['field'][:-1]]=field
        check_act(d,outputs,r['requestSHA256'],manifest);abis+=len(outputs)
        cs=commit['sourceFrame']
        if cs['slots']['sample']['fingerprint']!=latent or cs['slots']['velocity']['fingerprint']!=outputs['velocity']['payloadSHA256'] or cs['metadata']['velocityProducerFrameID']!=d['outputFrame']['frame_id'] or cs['metadata']['nativeSchedulerStepIndexBefore']!=(None if step==0 else step):raise ValueError('Own sampler consumption differs')
        check_act(commit,{'nextLatent':c['fields']['latent'+str(step+1)]},r['requestSHA256'],manifest);abis+=1
        latent=c['fields']['latent'+str(step+1)]['payloadSHA256'];previous=commit['outputFrame']['frame_id']
    if joins!=100:raise ValueError('Coupled block coverage differs')
    decode=acts[-1]
    if decode['sourceFrame']['slots']['vaeInput']['fingerprint']!=c['fields']['vaeInput']['payloadSHA256'] or decode['sourceFrame']['metadata']['finalCommitFrameID']!=previous or decode['sourceFrame']['metadata']['finalOwnLatentPayloadSHA256']!=latent:raise ValueError('Actual decode binding differs')
    check_act(decode,{'decodedRaw':c['fields']['decodedRaw']},r['requestSHA256'],manifest);abis+=1
    products=e['explicitTextProducts']
    expected_products=[(step,kind,index,op) for step in range(4) for kind,count in [('joint',5),('single',19)] for index in range(count) for op in ['imageCovariance','queryCovariance','queryMean','textScores','textAV']]
    if [(x['step'],x['blockKind'],x['blockIndex'],x['operator']) for x in products]!=expected_products:raise ValueError('Explicit text product coverage differs')
    for x in products:
        geometry={'imageCovariance':([1,24,128,1024],[1,24,1024,128],[1,24,128,128]),'queryCovariance':([1,24,512,128],[1,24,128,128],[1,24,512,128]),'queryMean':([1,24,512,128],[1,24,128,1],[1,24,512,1]),'textScores':([1,24,512,128],[1,24,128,512],[1,24,512,512]),'textAV':([1,24,512,512],[1,24,512,128],[1,24,512,128])}
        left,right,out=geometry[x['operator']]
        if x['leftShape']!=left or x['rightShape']!=right or x['outputShape']!=out or any(x[n]!='torch.float32' for n in ['leftDtype','rightDtype','outputDtype']) or x['autocastEnabled']:raise ValueError('Physical FP32 text geometry differs')
    affine=sum(x['nominalMACs'] for x in e['linears']);attention=sum(x['nominalQKAndAVMACs'] for x in e['SDPA'])+sum(x['nominalMACs'] for x in products);conv=sum(x['nominalMACs'] for x in e['convolutions'])
    if [affine,attention,conv]!=[e['affineMACs'],e['attentionMACs'],e['convolutionMACs']] or affine+attention+conv!=18559412469760 or report['work']['modelMatrixAndConvolutionMACs']!=18559412469760:raise ValueError('Actual work differs')
    if len([x for x in e['linears'] if x['kind']=='denoise'])!=600 or len(e['nativeDispatch'])!=100 or len([x for x in e['SDPA'] if x['kind']=='denoise'])!=100:raise ValueError('Denoiser operator census differs')
    streams={}
    for stream in read(p/'provenance.json')['losslessStreams']:
        raw=gzip.decompress((p/stream['published']).read_bytes())
        if hashlib.sha256(raw).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless stream differs')
        streams[stream['source']]=[json.loads(x) for x in raw.splitlines()]
    rows=streams['observations.jsonl'];addresses=[(x['field'],x['panel'],x['index']) for x in rows]
    expected={(name,'row',i) for name in t['fields'] for i in range(512 if name.startswith('C') or name in ['decodedRaw','rgbFloat','pngRGB'] else 64 if name=='vaeInput' else 1024)}
    if len(addresses)!=len(set(addresses)) or set(addresses)!=expected or len(rows)!=11840:raise ValueError('Complete signed comparison differs')
    mass=streams['partition-observations.jsonl'];addresses=[(x['step'],x['kind'],x['index'],x['head']) for x in mass]
    expected={(step,kind,index,head) for step in range(4) for kind,count in [('joint',5),('single',19)] for index in range(count) for head in range(24)}
    if len(addresses)!=len(set(addresses)) or set(addresses)!=expected:raise ValueError('Descriptive own partition coverage differs')
    moments=streams['moment-observations.jsonl']
    if len(moments)!=2304 or {(x['step'],x['kind'],x['index'],x['head']) for x in moments}!=expected:raise ValueError('Moment descriptive coverage differs')
    channel_rows=streams['selection-observations.jsonl']
    if len(channel_rows)!=80*9216 or {(x['step'],x['blockIndex'],x['channel']) for x in channel_rows}!={(k,b,c) for k in range(4) for b in range(20) for c in range(9216)}:raise ValueError('Complete selection channel rows differ')
    completion_abis=0
    if len(e['completionActs'])!=80 or len(e['completionRecords'])!=80 or set(v['completionBroadcastWitnesses'])!={str(k)+'.'+str(i) for k in range(4) for i in range(20)}:raise ValueError('Completion witness census differs')
    loaded={x['name']:x for x in e['loadedTransformerParameters']}
    for record,act in zip(e['completionRecords'],e['completionActs']):
        k,index=record['step'],record['index'];meta=act['sourceFrame']['metadata'];slots=act['sourceFrame']['slots'];fields=record['fields']
        check_act(act,fields,r['requestSHA256'],manifest);completion_abis+=len(fields)
        if act['stage']!='completion'+str(k)+'_'+str(index) or meta['step']!=k or meta['blockIndex']!=index or meta['terminal']!=(index==19) or record['typedOutputFrameID']!=act['outputFrame']['frame_id']:raise ValueError('Nested completion address differs')
        pk,pi=('joint',4) if index==0 else ('single',index-1)
        for slot,field in [('text','b_'+pk+str(pi)+'_textOut'+str(k)),('image','b_'+pk+str(pi)+'_imageOut'+str(k)),('mod','singleMod'+str(k)),('cosine','cosine'+str(k)),('sine','sine'+str(k)),('sourceUpNorm','sourceUpNorm'),('sourceDownNorm','sourceDownNorm')]:
            if slots[slot]['fingerprint']!=c['fields'][field]['payloadSHA256']:raise ValueError('Own completion source join differs')
        if meta['sourceNormProducerFrameID']!=acts[0]['outputFrame']['frame_id']:raise ValueError('Completion norm producer differs')
        for slot,name in meta['sourceParameterNames'].items():
            if slots[slot]['fingerprint']!=loaded[name]['payloadSHA256']:raise ValueError('Completion original parameter binding differs')
        for name,item in fields.items():
            label=('s_single'+str(index)+'_'+name[10:] if name.startswith('selection_') else 'b_single'+str(index)+'_'+name if name in ['textOut','imageOut','alpha','textRead'] else 'u_single'+str(index)+'_'+name)+str(k)
            if item!=c['fields'][label]:raise ValueError('Complete bank descriptor join differs')
        w=v['completionBroadcastWitnesses'][str(k)+'.'+str(index)]
        if not w['allWordsExact'] or w['completedPayloadSHA256']!=fields['completedUp']['payloadSHA256'] or w['anchorPayloadSHA256']!=fields['anchorUp']['payloadSHA256'] or w['selectedPayloadSHA256']!=fields['upRaw']['payloadSHA256']:raise ValueError('Mean completion exact word witness differs')
        if index==19 and record['outputStrides']['imageOut']!=[1536*3072,3072,1]:raise ValueError('Terminal head input stride differs')
    if completion_abis!=1668 or abis!=1471:raise ValueError('Explicit nested ABI inventory differs')
    for k in range(4):
        for index in range(20):
            ops=[x for x in e['linears'] if x['kind']=='denoise' and x['step']==k and x['parameter'].startswith('singlePhysical.'+str(index)+'.')]
            n=1024 if index==19 else 1536
            wanted=([('KV',[1,1536,3072],[6144,3072]),('QG',[1,1024,3072],[12288,3072])] if index==19 else [('QKVG',[1,1536,3072],[18432,3072])])+[('USelected',[1,n,3072],[2304,3072]),('UAnchor',[1,1 if index==19 else 2,3072],[9216,3072])]
            if [(x['parameter'].split('.')[-1],x['inputShape'],x['weightShape']) for x in ops]!=wanted:raise ValueError('Physical block affine geometry differs')
            original=[x for x in e['linears'] if x['kind']=='denoise' and x['step']==k and x['parameter']=='transformer.single_transformer_blocks.'+str(index)+'.attn.to_out.weight']
            if len(original)!=1 or original[0]['inputShape']!=[1,n,12288] or original[0]['weightShape']!=[3072,12288]:raise ValueError('Original full concat output affine differs')
    if sha(p/'prior-report.json')!=s['priorTrajectoryReportSHA256']:raise ValueError('Prior report publication differs')
    return {'schema':'tce-mean-up-trajectory-mechanics-v1','typedActs':90,'outerTypedActs':10,'completionTypedActs':80,'explicitOutputABIs':abis+completion_abis,'outerOutputABIs':abis,'completionOutputABIs':completion_abis,'coupledBlockJoins':joins,'liveTextReads':96,'explicitTextProducts':480,
        'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'originalParameterWitnesses':169,'physicalSliceRecords':244,'sourceNormOutputs':2,'currentSelections':80,'terminalOwnRowByteJoins':4,
        'observations':11840,'descriptivePartitionHeadRows':len(mass),'descriptiveMomentHeadRows':len(moments),'modelMatrixAndConvolutionMACs':affine+attention+conv,'avoidedMatrixMACsVsOriginal':3044322312192,
        'decodeInputScope':'Final latent unpack/BN/unpatchify source-audited, not independently numerically replayed. Actual VAEinput/output/PNG directly bound.',
        'completionBroadcastWordWitnesses':80,'scope':'Execution/custody/source/own-state/work mechanics. No scientific effect-size gate, cheap scalar, native equivalence, prompt generalization or zero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('root');p.add_argument('--destination');a=p.parse_args()
    print(json.dumps(publish(a.root,a.destination) if a.action=='publish' else globals()[a.action](a.root),sort_keys=True))
