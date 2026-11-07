"""Independent byte custody and compiler/cache/own-image structural audit."""
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
from publish_sparse_text_kv import NPY_BYTES
from publish_terminal_trajectory import check_act

BYTES=NPY_BYTES.split('print(json.dumps')[0]+r'''
parameters={};slice_witnesses=[]
with Path(s['checkpoint']).open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n
 if set(header)-{'__metadata__'}!=set(x['name'] for x in s['loaded']):raise ValueError('Full original parameter census differs')
 for item in s['loaded']:
  entry=header[item['name']];a,b=entry['data_offsets'];keep=item['name']=='single_transformer_blocks.19.attn.to_qkv_mlp_proj.weight'
  if entry['dtype']!='BF16' or entry['shape']!=item['shape'] or [a,b]!=item['dataOffsets'] or b-a!=item['bytes']:raise ValueError('Original parameter geometry differs')
  f.seek(base+a);h=hashlib.sha256();left=b-a;parts=[]
  while left:
   chunk=f.read(min(left,1<<20))
   if not chunk:raise ValueError('Short original parameter')
   h.update(chunk);left-=len(chunk)
   if keep:parts.append(chunk)
  if h.hexdigest()!=item['payloadSHA256']:raise ValueError('Original parameter payload differs')
  parameters[item['name']]={'payloadSHA256':h.hexdigest(),'bytes':b-a,'shape':entry['shape']}
  if keep:
   raw=b''.join(parts);width=3072*2;kv=raw[3072*width:9216*width];digest=hashlib.sha256(kv).hexdigest()
   for x in s['physicalMaps']:
    if x['sourceParameter']!=item['name'] or x['name']!='KV' or x['payloadSHA256']!=digest or x['bytes']!=len(kv) or x['shape']!=[6144,3072]:raise ValueError('Physical terminal KV source bytes differ')
    slice_witnesses.append({'step':x['step'],'payloadSHA256':digest,'bytes':len(kv),'shape':x['shape']})
if sorted(x['step'] for x in slice_witnesses)!=[0,1,2,3]:raise ValueError('Four terminal KV slices absent')
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,
 'originalParameterWitnesses':parameters,'physicalSliceWitnesses':slice_witnesses}))
'''


def items(p):
    s=read(p/'inputs.json');values=list(fields(s['E']))+list(fields(s['noise']))
    for name in ['targets.json','candidate.json','differences.json','compiler.json']:values+=list(fields(read(p/name)))
    values += [{'path':str(Path(s['modelRoot'])/n),'sha256':h} for n,h in s['imageAssets'].items()]
    native=Path(s['E']['path']).parents[3]
    values += [{'path':str(native/'response-result'/n),'sha256':h} for n,h in s['sourceRecords'].items()]
    return values


def verify(root):
    from mrun.client.api import Api
    root,c,j,r=checked(root);p=root/'result';s=read(p/'inputs.json');e=read(p/'compiler-execution.json');a=read(p/'execution.json');api=Api();roots=[]
    for job_id,custody in [(s['nativeJobID'],s['nativeCustody']),(s['noiseJobID'],s['noiseCustody']),(s['priorTrajectoryJobID'],s['priorTrajectoryCustody'])]:
        job=api.json('GET','/api/jobs/'+job_id)
        if job['state']!='succeeded' or job['payload_custody']!=custody:raise ValueError('Source custody differs')
        roots.append(job['result']['diagnostics']['cwd'])
    if a['loadedTransformerParameters']!=e['loadedTransformerParameters']:raise ValueError('Two original model loads differ')
    base=j['result']['diagnostics']['cwd'];values=items(p)+[{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()]
    values.append({'path':roots[-1]+'/response-result/report.json','sha256':s['priorTrajectoryReportSHA256']});unique={}
    for x in values:
        if x['path'] in unique and unique[x['path']]['sha256']!=x['sha256']:raise ValueError('Conflicting byte identity')
        unique[x['path']]={**unique.get(x['path'],{}),**x}
    spec={'items':list(unique.values()),'allowedRoots':[base,s['modelRoot'],*roots],
        'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),'loaded':e['loadedTransformerParameters'],'physicalMaps':e['physicalSingleMaps']}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=240)
    if proc.returncode:raise RuntimeError(proc.stderr)
    value=json.loads(proc.stdout);value.update(schema='tce-prompt-program-bytes-v1',jobID=j['job_id'],verifierSourceSHA256=sha(__file__),
        byteTemplateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),sharedAuditSourceSHA256=sha(Path(__file__).with_name('publish_terminal_trajectory.py')),
        candidateSealSHA256=sha(p/'candidate-seal.json'),compilerSealSHA256=sha(p/'compiler-seal.json'),scope='Complete file/NPY/original weight and physical KV slice bytes. SSH byte operations only; no tensor math.')
    save(root/'retained-array-verification.json',value)
    return {'files':len(value['files']),'bytes':value['totalBytesVerified'],'NPYPayloadWitnesses':len(value['npyPayloadWitnesses']),'originalParameterWitnesses':len(value['originalParameterWitnesses']),'physicalSliceWitnesses':len(value['physicalSliceWitnesses'])}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['compilerSealSHA256']!=sha(p/'compiler-seal.json'):raise ValueError('Executed verifier differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for n in p.iterdir():
        if n.suffix=='.jsonl':
            raw=n.read_bytes();target=out/(n.name+'.gz');target.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless stream differs')
            streams.append({'source':n.name,'sourceSHA256':sha(n),'published':target.name,'publishedSHA256':sha(target),'sourceBytes':len(raw),'publishedBytes':target.stat().st_size})
        else:shutil.copyfile(n,out/n.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'scheduler.json',{k:j[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-prompt-program-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],
        'completeCollectedInventory':c['inventory'],'losslessStreams':streams,'scope':'One depth-evolving image-independent text program and own image actor. All learned work paid; no newreference/fit/amortizedspeedup/arbitrary-promptzero-neural completion.'})
    save(out/'files.json',{n.name:sha(n) for n in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(n.stat().st_size for n in out.iterdir())}


def audit(source):
    from prompt_program import matrix_account
    p=Path(source);r,s,c,e,t,report,v,program,ce=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json','targets.json','report.json','retained-array-verification.json','compiler.json','compiler-execution.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Publication differs:'+n)
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]:raise ValueError('Frozen science source differs:'+n)
    for key,name in [('verifierSourceSHA256','publish_prompt_program.py'),('byteTemplateSourceSHA256','publish_sparse_text_kv.py'),('sharedAuditSourceSHA256','publish_terminal_trajectory.py')]:
        if v[key]!=sha(Path(__file__).with_name(name)):raise ValueError('Executed audit source differs')
    for x in items(p):
        if v['files'][x['path']]['sha256']!=x['sha256'] or (x.get('payloadSHA256') and v['npyPayloadWitnesses'][x['path']]['payloadSHA256']!=x['payloadSHA256']):raise ValueError('Full byte witness absent')
    compiler_sha=sha(p/'compiler-seal.json')
    if read(p/'compiler-seal.json')!={'requestSHA256':r['requestSHA256'],'compilerSHA256':sha(p/'compiler.json'),'executionSHA256':sha(p/'compiler-execution.json')}:raise ValueError('Compiler seal differs')
    if read(p/'candidate-seal.json')!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json'),'imageSHA256':sha(p/'candidate-A.png'),'compilerSealSHA256':compiler_sha}:raise ValueError('Actor seal differs')
    if report['compilerSealSHA256']!=compiler_sha or v['compilerSealSHA256']!=compiler_sha or report['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Analysis release differs')
    if sha(p/'native-A.png')!=t['image']['sha256'] or sha(p/'candidate-A.png')!=c['image']['sha256']:raise ValueError('Actual image differs')
    manifest=identity({'transformer':ce['loadedTransformerParameters'],'lawSourceSHA256':r['sourceHashes']['prompt_program.py']})
    actor_manifest=identity({'transformer':e['loadedTransformerParameters'],'vaeCheckpointSHA256':s['imageAssets']['vae/diffusion_pytorch_model.safetensors'],
        'promptProgramLawSourceSHA256':r['sourceHashes']['prompt_program.py'],'compilerSealSHA256':compiler_sha})
    if ce['parameterManifestSHA256']!=manifest or e['parameterManifestSHA256']!=actor_manifest or e['loadedTransformerParameters']!=ce['loadedTransformerParameters']:raise ValueError('Original parameter manifests differ')
    if len(ce['loadedTransformerParameters'])!=169 or any(v['originalParameterWitnesses'][x['name']]['payloadSHA256']!=x['payloadSHA256'] for x in ce['loadedTransformerParameters']):raise ValueError('Original weight witness differs')
    assets={str(Path(s['modelRoot'])/n) for n in s['imageAssets']};compiler_allowed={s['E']['path']}|assets
    actor_allowed=compiler_allowed|{s['noise'][n]['path'] for n in ['raw','packed','ids']}|{x['path'] for row in program['rows'] for group in ['common','cache'] for x in row[group].values()}
    for execution,expected,denied in [(ce,compiler_allowed,set(s['deniedNumericalInputs'])|{s['noise'][n]['path'] for n in ['raw','packed','ids']}),(e,actor_allowed,set(s['deniedNumericalInputs']))]:
        guard=execution['fileQuarantine']
        if set(guard['allowedFiles'])!=expected or set(guard['deniedCanaries'])!=denied or expected&denied or not set(guard['successfulFileOpenCounts']).issubset(expected):raise ValueError('Source isolation differs')
        if not execution['sourceParameterBufferRNGClosure'] or execution['numericalProfile']!=s['numericalProfile']:raise ValueError('Source numeric closure differs')
    if [a['stage'] for a in ce['typedActs']]!=['compile'+str(k) for k in range(4)] or len(ce['blockRecords'])!=100:raise ValueError('Compiler clock/block census differs')
    if [row['step'] for row in program['rows']]!=[0,1,2,3]:raise ValueError('Published compiler clock domain differs')
    abis=0;cache_joins=0;image_joins=0;previous_commit=None;latent=s['noise']['packed']['payloadSHA256']
    wanted_stages=[n+str(k) for k in range(4) for n in ['denoise','commit']]+['decode']
    if [a['stage'] for a in e['typedActs']]!=wanted_stages or c['counts']!=r['expectedCalls'] or len(c['fields'])!=40:raise ValueError('Actor computation census differs')
    for step,row in enumerate(program['rows']):
        ca=ce['typedActs'][step];outputs={**row['common'],**row['cache']};check_act(ca,outputs,r['requestSHA256'],manifest);abis+=len(outputs)
        if row['step']!=step or row['compilerFrameID']!=ca['outputFrame']['frame_id'] or ca['sourceFrame']['slots']['embedding']['fingerprint']!=s['E']['payloadSHA256']:raise ValueError('Compiler source frame differs')
        for n in ['clockArgument','textIDs']:
            if ca['sourceFrame']['slots'][n]['fingerprint']!=row['common'][n]['payloadSHA256']:raise ValueError('Compiler clock/position read differs')
        blocks=[x for x in ce['blockRecords'] if x['step']==step];actor_blocks=[x for x in e['blockRecords'] if x['step']==step]
        addresses=[(kind,index) for kind,count in [('joint',5),('single',20)] for index in range(count)]
        if [(x['kind'],x['index']) for x in blocks]!=addresses or [(x['kind'],x['index']) for x in actor_blocks]!=addresses:raise ValueError('Layer order differs')
        expected_text=row['common']['context']['payloadSHA256'];expected_image=c['fields']['imageStart'+str(step)]['payloadSHA256']
        for b,ib in zip(blocks,actor_blocks):
            key=b['kind']+str(b['index'])
            if b['incomingText']!=row['states'][key] or b['incomingText']['payloadSHA256']!=expected_text or b['cacheTiming']!='incoming-state':raise ValueError('Text continuation/cache timing differs')
            if (b['updated'],b['outgoingTextPayloadSHA256'] is not None)!=(key!='single19',key!='single19'):raise ValueError('Terminal last-use differs')
            expected_text=b['outgoingTextPayloadSHA256']
            for bank,bfield,ifield in [('K','keyPayloadSHA256','textKeyPayloadSHA256'),('V','valuePayloadSHA256','textValuePayloadSHA256')]:
                if row['cache'][key+bank]['payloadSHA256']!=b[bfield] or ib[ifield]!=b[bfield]:raise ValueError('Compiled cache consumer differs')
                cache_joins+=1
            if ib['compilerSealSHA256']!=compiler_sha or ib['compilerFrameID']!=row['compilerFrameID'] or ib['compilerIncomingTextPayloadSHA256']!=b['incomingText']['payloadSHA256'] or ib['imageInputPayloadSHA256']!=expected_image:raise ValueError('Own image/cache lineage differs')
            expected_image=ib['imageOutputPayloadSHA256'];image_joins+=1
        if expected_image!=c['fields']['terminalImage'+str(step)]['payloadSHA256']:raise ValueError('Terminal own image differs')
        d,commit=e['typedActs'][step*2:step*2+2];src=d['sourceFrame'];meta=src['metadata']
        if src['slots']['hidden_states']['fingerprint']!=latent or meta['precedingCommitFrameID']!=previous_commit or meta['compilerFrameID']!=row['compilerFrameID'] or meta['compilerSealSHA256']!=compiler_sha:raise ValueError('Actor source/clock ancestry differs')
        for n,digest in [('encoder_hidden_states',s['E']['payloadSHA256']),('img_ids',s['noise']['ids']['payloadSHA256']),
            ('timestep',row['common']['clockArgument']['payloadSHA256']),('txt_ids',row['common']['textIDs']['payloadSHA256'])]:
            if src['slots'][n]['fingerprint']!=digest:raise ValueError('Actor conditioning/positions/clock differs')
        for name,item in outputs.items():
            slot=('compiled_' if name in row['common'] else 'cache_')+name
            if src['slots'][slot]['fingerprint']!=item['payloadSHA256']:raise ValueError('Explicit compiled read slot differs')
        wanted={n:c['fields'][('C' if n=='context' else n)+str(step)] for n in ['velocity','context','imageStart','terminalImage','singleMod','temb','cosine','sine']}
        check_act(d,wanted,r['requestSHA256'],actor_manifest);abis+=len(wanted)
        cs=commit['sourceFrame']
        if cs['slots']['sample']['fingerprint']!=latent or cs['slots']['velocity']['fingerprint']!=wanted['velocity']['payloadSHA256'] or cs['metadata']['velocityProducerFrameID']!=d['outputFrame']['frame_id'] or cs['metadata']['nativeSchedulerStepIndexBefore']!=(None if step==0 else step):raise ValueError('Own sampler consumption differs')
        check_act(commit,{'nextLatent':c['fields']['latent'+str(step+1)]},r['requestSHA256'],actor_manifest);abis+=1
        latent=c['fields']['latent'+str(step+1)]['payloadSHA256'];previous_commit=commit['outputFrame']['frame_id']
    if cache_joins!=200 or image_joins!=100:raise ValueError('Complete compiled-bank/image join coverage differs')
    decode=e['typedActs'][-1]
    if decode['sourceFrame']['slots']['vaeInput']['fingerprint']!=c['fields']['vaeInput']['payloadSHA256'] or decode['sourceFrame']['metadata']['finalCommitFrameID']!=previous_commit or decode['sourceFrame']['metadata']['finalOwnLatentPayloadSHA256']!=latent:raise ValueError('Actual decode input declaration differs')
    check_act(decode,{'decodedRaw':c['fields']['decodedRaw']},r['requestSHA256'],actor_manifest);abis+=1
    totals=[]
    for execution in [ce,e]:
        affine=sum(x['nominalMACs'] for x in execution['linears']);att=sum(x['nominalQKAndAVMACs'] for x in execution['SDPA'])
        if affine!=execution['affineMACs'] or att!=execution['attentionMACs']:raise ValueError('Actual learned work differs')
        totals.extend([affine,att])
    conv=sum(x['nominalMACs'] for x in e['convolutions']);total=sum(totals)+conv
    if total!=report['work']['modelMatrixAndConvolutionMACs'] or total!=21062568902656 or conv!=e['convolutionMACs']:raise ValueError('Combined actual work differs')
    if ce['textProgramMatrixMACs']!=matrix_account()['compilerMatrixMACs'] or program['logicalCacheBytes']!=629145600 or ce['logicalCacheBytes']!=629145600:raise ValueError('Compiler account differs')
    if len(ce['SDPA'])!=96 or len(e['nativeDispatch'])!=100 or len([x for x in e['linears'] if x['kind']=='denoise'])!=292:raise ValueError('Executed attention/affine geometry differs')
    for x in ce['nativeDispatch']:
        if x['QShape']!=[1,512,24,128] or x['KShape']!=[1,512,24,128] or x['VShape']!=[1,512,24,128]:raise ValueError('TT-only attention differs')
    for x in e['nativeDispatch']:
        if x['QShape']!=[1,1024,24,128] or x['KShape']!=[1,1536,24,128] or x['VShape']!=[1,1536,24,128]:raise ValueError('Own full-bank image attention differs')
    observations=[]
    for stream in read(p/'provenance.json')['losslessStreams']:
        raw=gzip.decompress((p/stream['published']).read_bytes())
        if hashlib.sha256(raw).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless scalar publication differs')
        if stream['source']=='observations.jsonl':observations=[json.loads(x) for x in raw.splitlines()]
    addresses=[(x['field'],x['panel'],x['index']) for x in observations]
    expected={(name,'row',i) for name in t['fields'] for i in range(512 if name.startswith('C') or name in ['decodedRaw','rgbFloat','pngRGB'] else 64 if name=='vaeInput' else 1024)}
    if len(addresses)!=len(set(addresses)) or set(addresses)!=expected or len(observations)!=report['observations'] or len(observations)!=11840:raise ValueError('Complete signed observations differ')
    return {'schema':'tce-prompt-program-mechanics-v1','typedActs':13,'explicitOutputABIs':abis,'compilerBlocks':100,'liveTextUpdates':96,'compiledBankConsumptionJoins':cache_joins,
        'ownImageBlockJoins':image_joins,'originalParameterWitnesses':169,'physicalSliceWitnesses':4,'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),
        'observations':len(observations),'logicalCacheBytes':629145600,'compilerTextProgramMatrixMACs':ce['textProgramMatrixMACs'],'modelMatrixAndConvolutionMACs':total,
        'avoidedMatrixMACsVsOriginal':21603734781952-total,'actualVAEDecodes':1,'newNativeReferences':0,
        'decodeInputScope':'Final latent unpack/BN/unpatchify source-audited, not independently numerically replayed; actual VAE input/output and PNG bytes bound.',
        'scope':'Complete source/cache/own-state/work/custody mechanics. No quality gate, prompt generalization, amortizedspeedup or arbitrary-promptzero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('root');p.add_argument('--destination');a=p.parse_args()
    result=publish(a.root,a.destination) if a.action=='publish' else globals()[a.action](a.root)
    print(json.dumps(result,sort_keys=True))
