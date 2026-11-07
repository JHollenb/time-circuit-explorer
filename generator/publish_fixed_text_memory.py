"""Complete bytes, physical row witnesses and fixed-text/own-image joins."""
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
from publish_sparse_text_kv import NPY_BYTES
from publish_terminal_trajectory import items,check_act


BYTES=NPY_BYTES.split('print(json.dumps')[0]+r'''
parameters={};slices={}
with Path(s['checkpoint']).open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n
 if set(header)-{'__metadata__'}!=set(x['name'] for x in s['loaded']):raise ValueError('Original parameter census differs')
 for item in s['loaded']:
  entry=header[item['name']];a,b=entry['data_offsets'];keep=item['name'].startswith('single_transformer_blocks.') and item['name'].endswith('.attn.to_qkv_mlp_proj.weight')
  if entry['dtype']!='BF16' or entry['shape']!=item['shape'] or [a,b]!=item['dataOffsets'] or b-a!=item['bytes']:raise ValueError('Parameter geometry differs')
  f.seek(base+a);h=hashlib.sha256();left=b-a;parts=[]
  while left:
   chunk=f.read(min(left,1<<20))
   if not chunk:raise ValueError('Short parameter bytes')
   h.update(chunk);left-=len(chunk)
   if keep:parts.append(chunk)
  if h.hexdigest()!=item['payloadSHA256']:raise ValueError('Original map byte witness differs')
  parameters[item['name']]={'payloadSHA256':h.hexdigest(),'bytes':b-a,'shape':entry['shape']}
  if keep:
   raw=b''.join(parts);width=3072*2
   for name,value in [('KV',raw[3072*width:9216*width]),('QGU',raw[:3072*width]+raw[9216*width:])]:
    records=[x for x in s['physicalMaps'] if x['sourceParameter']==item['name'] and x['name']==name];digest=hashlib.sha256(value).hexdigest()
    if len(records)!=4 or sorted(x['step'] for x in records)!=[0,1,2,3]:raise ValueError('Four-clock physical source witness absent')
    for x in records:
     if x['payloadSHA256']!=digest or x['bytes']!=len(value):raise ValueError('Physical row byte witness differs')
    slices[records[0]['parameter']]={'payloadSHA256':digest,'bytes':len(value),'shape':records[0]['shape'],'steps':[0,1,2,3],'sourceParameter':item['name']}
def raw_npy(item):
 with Path(item['path']).open('rb') as f:
  magic=f.read(8);n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0];f.read(n);return f.read()
row_witnesses={}
for k in range(4):
 item=s['candidateFields']['terminalRows'+str(k)];raw=raw_npy(item);n=512*3072*2
 if item['shape']!=[1,1536,3072] or len(raw)!=1536*3072*2:raise ValueError('Combined fixed-text/image byte geometry differs')
 text_hash=hashlib.sha256(raw[:n]).hexdigest();image_hash=hashlib.sha256(raw[n:]).hexdigest()
 last=next(x for x in s['blockRecords'] if x['step']==k and x['kind']=='single' and x['index']==19)
 if text_hash!=s['candidateFields']['C'+str(k)]['payloadSHA256'] or image_hash!=last['imageInputPayloadSHA256']:raise ValueError('Final combined own-row bytes differ')
 row_witnesses[str(k)]={'fixedTextPayloadSHA256':text_hash,'ownImagePayloadSHA256':image_hash,'byteExact':True}
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,
 'originalParameterWitnesses':parameters,'physicalSliceWitnesses':slices,'terminalCombinedRowWitnesses':row_witnesses}))
'''


def verify(root):
    from mrun.client.api import Api
    root,c,j,r=checked(root);p=root/'result';s=read(p/'inputs.json');e=read(p/'execution.json');api=Api();roots=[]
    for job_id,custody in [(s['nativeJobID'],s['nativeCustody']),(s['noiseJobID'],s['noiseCustody']),(s['priorTrajectoryJobID'],s['priorTrajectoryCustody'])]:
        parent=api.json('GET','/api/jobs/'+job_id)
        if parent['state']!='succeeded' or parent['payload_custody']!=custody:raise ValueError('Original source custody differs')
        roots.append(parent['result']['diagnostics']['cwd'])
    base=j['result']['diagnostics']['cwd'];values=items(p)+[{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()]
    values.append({'path':roots[-1]+'/response-result/report.json','sha256':s['priorTrajectoryReportSHA256']});unique={}
    for x in values:
        if x['path'] in unique and unique[x['path']]['sha256']!=x['sha256']:raise ValueError('Conflicting byte identity')
        unique[x['path']]={**unique.get(x['path'],{}),**x}
    spec={'items':list(unique.values()),'allowedRoots':[base,s['modelRoot'],*roots],
        'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),'loaded':e['loadedTransformerParameters'],
        'physicalMaps':e['physicalSingleMaps'],'blockRecords':e['blockRecords'],'candidateFields':read(p/'candidate.json')['fields']}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=240)
    if proc.returncode:raise RuntimeError(proc.stderr)
    v=json.loads(proc.stdout);v.update(schema='tce-fixed-text-memory-bytes-v1',jobID=j['job_id'],verifierSourceSHA256=sha(__file__),
        byteTemplateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),sharedAuditSourceSHA256=sha(Path(__file__).with_name('publish_terminal_trajectory.py')),
        candidateSealSHA256=sha(p/'candidate-seal.json'),scope='Complete file/payload/original weight/physical row and terminal concatenation bytes. SSH byte I/O only, no tensor numerics.')
    save(root/'retained-array-verification.json',v)
    return {'files':len(v['files']),'bytes':v['totalBytesVerified'],'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'originalParameterWitnesses':len(v['originalParameterWitnesses']),'physicalSliceWitnesses':len(v['physicalSliceWitnesses'])}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Byte verifier differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for n in p.iterdir():
        if n.suffix=='.jsonl':
            raw=n.read_bytes();target=out/(n.name+'.gz');target.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless publication differs')
            streams.append({'source':n.name,'sourceSHA256':sha(n),'published':target.name,'publishedSHA256':sha(target),'sourceBytes':len(raw),'publishedBytes':target.stat().st_size})
        else:shutil.copyfile(n,out/n.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'scheduler.json',{k:j[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-fixed-text-memory-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],
        'completeCollectedInventory':c['inventory'],'losslessStreams':streams,'scope':'One own candidate with C fixed at all25blocks, actualoneVAE/PNG. Combined architectural+finitegeometry intervention, all liveKV/imagework paid. No newreference/fit/hidden textstate/arbitrary-promptzero-neural completion.'})
    save(out/'files.json',{n.name:sha(n) for n in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(n.stat().st_size for n in out.iterdir())}


def audit(source):
    p=Path(source);r,s,c,e,t,report,v=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json','targets.json','report.json','retained-array-verification.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request identity differs')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Published member differs:'+n)
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]:raise ValueError('Frozen scientific source differs:'+n)
    if v['verifierSourceSHA256']!=sha(__file__) or v['byteTemplateSourceSHA256']!=sha(Path(__file__).with_name('publish_sparse_text_kv.py')) or v['sharedAuditSourceSHA256']!=sha(Path(__file__).with_name('publish_terminal_trajectory.py')):raise ValueError('Executed byte/shared auditor differs')
    for x in items(p):
        if v['files'][x['path']]['sha256']!=x['sha256'] or (x.get('payloadSHA256') and v['npyPayloadWitnesses'][x['path']]['payloadSHA256']!=x['payloadSHA256']):raise ValueError('Complete byte witness absent')
    seal=read(p/'candidate-seal.json')
    if (seal!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json'),'imageSHA256':sha(p/'candidate-A.png')}
        or report['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['candidateSealSHA256']!=sha(p/'candidate-seal.json')):raise ValueError('Producer release differs')
    if sha(p/'candidate-A.png')!=c['image']['sha256'] or sha(p/'native-A.png')!=t['image']['sha256']:raise ValueError('Actual PNG joins differ')
    manifest=identity({'transformer':e['loadedTransformerParameters'],'vaeCheckpointSHA256':s['imageAssets']['vae/diffusion_pytorch_model.safetensors'],'fixedTextLawSourceSHA256':r['sourceHashes']['fixed_text_memory.py']})
    if manifest!=e['parameterManifestSHA256'] or not e['sourceParameterBufferRNGClosure'] or e['numericalProfile']!=s['numericalProfile']:raise ValueError('Original source/profile closure differs')
    for x in e['loadedTransformerParameters']:
        if v['originalParameterWitnesses'][x['name']]['payloadSHA256']!=x['payloadSHA256']:raise ValueError('Original weight bytes differ')
    maps=e['physicalSingleMaps'];addresses=[(x['step'],x['blockIndex'],x['name']) for x in maps]
    if len(addresses)!=160 or set(addresses)!={(k,i,n) for k in range(4) for i in range(20) for n in ['KV','QGU']}:raise ValueError('Physical map materialization census differs')
    for x in maps:
        if v['physicalSliceWitnesses'][x['parameter']]['payloadSHA256']!=x['payloadSHA256']:raise ValueError('Physical row witness differs')
    if e['mapMaterializationScope']!={'activeSets':0,'maxActiveSets':1,'maxLogicalMapBytes':169869312}:raise ValueError('Logical map callback bound differs')
    guard=e['fileQuarantine'];allowed=set(guard['allowedFiles']);denied=set(guard['deniedCanaries'])
    wanted={s['E']['path']}|{s['noise'][n]['path'] for n in ['raw','packed','ids']}|{str(Path(s['modelRoot'])/n) for n in s['imageAssets']}
    if allowed!=wanted or allowed&denied or denied!=set(s['deniedNumericalInputs']) or not set(guard['successfulFileOpenCounts']).issubset(allowed):raise ValueError('Numerical source closure differs')
    acts=e['typedActs'];stages=[n+str(k) for k in range(4) for n in ['denoise','commit']]+['decode'];ports=['imageStart','velocity','context','terminalRows','terminalImage','singleMod','temb','cosine','sine']
    if [x['stage'] for x in acts]!=stages or c['counts']!=r['expectedCalls'] or len(c['fields'])!=44:raise ValueError('Actual own component census differs')
    state=s['noise']['packed']['payloadSHA256'];previous=None;joins=0
    for k in range(4):
        d,commit=acts[2*k:2*k+2];ds=d['sourceFrame'];cs=commit['sourceFrame'];dm=ds['metadata'];cm=cs['metadata']
        if ds['slots']['hidden_states']['fingerprint']!=state or ds['slots']['encoder_hidden_states']['fingerprint']!=s['E']['payloadSHA256'] or ds['slots']['img_ids']['fingerprint']!=s['noise']['ids']['payloadSHA256'] or dm['precedingCommitFrameID']!=previous or dm['expectedOwnLatentPayloadSHA256']!=state:raise ValueError('Denoiser own-state/source join differs')
        if cs['slots']['velocity']['fingerprint']!=c['fields']['velocity'+str(k)]['payloadSHA256'] or cs['slots']['sample']['fingerprint']!=state or cm['velocityProducerFrameID']!=d['outputFrame']['frame_id'] or cm['nativeSchedulerStepIndexBefore']!=(None if k==0 else k):raise ValueError('Scheduler own-state consumption differs')
        check_act(d,{n:c['fields'][('C' if n=='context' else n)+str(k)] for n in ports},r['requestSHA256'],manifest)
        check_act(commit,{'nextLatent':c['fields']['latent'+str(k+1)]},r['requestSHA256'],manifest)
        state=c['fields']['latent'+str(k+1)]['payloadSHA256'];previous=commit['outputFrame']['frame_id'];joins+=2+(k>0)
        blocks=[x for x in e['blockRecords'] if x['step']==k];order=[('joint',i) for i in range(5)]+[('single',i) for i in range(20)];image=c['fields']['imageStart'+str(k)]['payloadSHA256']
        if [(x['kind'],x['index']) for x in blocks]!=order:raise ValueError('All25 demanded blocks absent')
        for x in blocks:
            if x['fixedTextPayloadSHA256']!=c['fields']['C'+str(k)]['payloadSHA256'] or x['imageInputPayloadSHA256']!=image or x['textShape']!=[1,512,3072] or x['imageShape']!=[1,1024,3072]:raise ValueError('Fixed text/own image block lineage differs')
            image=x['imageOutputPayloadSHA256']
        if image!=c['fields']['terminalImage'+str(k)]['payloadSHA256'] or not v['terminalCombinedRowWitnesses'][str(k)]['byteExact']:raise ValueError('Final own-row consumer join differs')
    decode=acts[-1];meta=decode['sourceFrame']['metadata']
    if decode['sourceFrame']['slots']['vaeInput']['fingerprint']!=c['fields']['vaeInput']['payloadSHA256'] or meta['finalCommitFrameID']!=previous or meta['finalOwnLatentPayloadSHA256']!=state:raise ValueError('Actual decoder declaration differs')
    check_act(decode,{'decodedRaw':c['fields']['decodedRaw']},r['requestSHA256'],manifest)
    affine=[x for x in e['linears'] if x['kind']=='denoise'];att=[x for x in e['SDPA'] if x['kind']=='denoise']
    if len(affine)!=436 or len(att)!=100 or len(e['nativeDispatch'])!=100 or sum(x['nominalQKAndAVMACs'] for x in att)!=sum(x['nominalQKAndAVMACs'] for x in e['nativeDispatch']):raise ValueError('Actual denoiser work census differs')
    if any(x['QShape']!=[1,24,1024,128] or x['KShape']!=[1,24,1536,128] or x['VShape']!=[1,24,1536,128] for x in att):raise ValueError('Full-bank rectangular attention differs')
    if any('.add_q_proj.' in x['parameter'] or '.to_add_out.' in x['parameter'] or '.ff_context.' in x['parameter'] or '.to_qkv_mlp_proj.' in x['parameter'] for x in affine):raise ValueError('Deleted text/full-fused work executed')
    for k in range(4):
        for i in range(5):
            joint=[x for x in affine if x['step']==k and x['parameter'].startswith('transformer.transformer_blocks.'+str(i)+'.')]
            if len(joint)!=8:raise ValueError('Joint reduced affine census differs')
            for x in joint:
                if x['inputShape'][:2]!=[1,512 if '.add_k_proj.' in x['parameter'] or '.add_v_proj.' in x['parameter'] else 1024]:raise ValueError('Joint live source row count differs')
        for i in range(20):
            subset=[x for x in affine if x['step']==k and x['parameter'].startswith('singlePhysical.'+str(i)+'.')]
            if [(x['inputShape'],x['weightShape']) for x in subset]!=[([1,1536,3072],[6144,3072]),([1,1024,3072],[21504,3072])]:raise ValueError('Physical single projection geometry differs')
    total=sum(x['nominalMACs'] for x in e['linears'])+sum(x['nominalQKAndAVMACs'] for x in e['SDPA'])+sum(x['nominalMACs'] for x in e['convolutions'])
    prior=Path(__file__).resolve().parent.parent/'docs/results/terminal-trajectory/report.json'
    if sha(prior)!=s['priorTrajectoryReportSHA256']:raise ValueError('Prior nominal comparator differs')
    pr=read(prior);avoided=pr['work']['modelMatrixAndConvolutionMACs']+pr['work']['avoidedTerminalMatrixMACs']-total
    if total!=report['work']['modelMatrixAndConvolutionMACs'] or avoided!=5798205849600 or e['avoidedTextPathMatrixMACs']!=avoided:raise ValueError('Executed removed work account differs')
    rows=[];block_rows=[]
    for stream in read(p/'provenance.json')['losslessStreams']:
        raw=gzip.decompress((p/stream['published']).read_bytes())
        if hashlib.sha256(raw).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless stream differs')
        if stream['source']=='observations.jsonl':rows=[json.loads(x) for x in raw.splitlines()]
        if stream['source']=='block-progress.jsonl':block_rows=[json.loads(x) for x in raw.splitlines()]
    addresses=[(x['field'],x['panel'],x['index']) for x in rows];expected={(n,'row',i) for n in t['fields'] for i in range(512 if n.startswith('C') or n in ['decodedRaw','rgbFloat','pngRGB'] else 64 if n=='vaeInput' else 1024)}
    if len(addresses)!=len(set(addresses)) or set(addresses)!=expected or len(rows)!=11840 or len(rows)!=report['observations'] or block_rows!=e['blockRecords']:raise ValueError('Complete signed/block progress rows differ')
    return {'schema':'tce-fixed-text-memory-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'typedActs':9,'explicitOutputABIs':41,'ownStateConsumptionJoins':joins,
        'fixedTextOwnImageBlockJoins':100,'completeKVRectangularReads':100,'actualCandidateImages':1,'newNativeReferences':0,'originalParameterByteWitnesses':len(v['originalParameterWitnesses']),
        'physicalSliceByteWitnesses':len(v['physicalSliceWitnesses']),'physicalMapMaterializations':80,'maxLogicalActiveMapSets':1,'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),
        'observations':11840,'denoiserAffineCalls':436,'modelMatrixAndConvolutionMACs':total,'avoidedTextPathMatrixMACs':avoided,'incrementalAvoidedMatrixMACsBeyondTerminal':5566277615616,
        'scope':'Actual own-image/fixed-text/work/byte mechanics. Logical map callback bound is not independently measured GPU residency. Decode normalization source-audited, not independently replayed. No effect-size gate, sole-cause attribution, matchedspeedup or arbitrary-promptzero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source),indent=2))
