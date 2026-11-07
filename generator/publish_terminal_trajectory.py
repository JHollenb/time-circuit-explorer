"""Byte-only custody and component/own-state audit for one actual image trajectory."""
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
from publish_source_complement import structured


BYTES=NPY_BYTES.split('print(json.dumps')[0]+r'''
checkpoint=Path(s['checkpoint']);parameters={};fused=None
with checkpoint.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n
 if set(header)-{'__metadata__'}!=set(x['name'] for x in s['loaded']):raise ValueError('Full original parameter census differs')
 for item in s['loaded']:
  entry=header[item['name']];a,b=entry['data_offsets']
  if entry['dtype']!='BF16' or entry['shape']!=item['shape'] or [a,b]!=item['dataOffsets'] or b-a!=item['bytes']:raise ValueError('Original parameter geometry differs')
  f.seek(base+a);h=hashlib.sha256();left=b-a;parts=[]
  keep=item['name']=='single_transformer_blocks.19.attn.to_qkv_mlp_proj.weight'
  while left:
   chunk=f.read(min(left,1<<20))
   if not chunk:raise ValueError('Short original parameter bytes')
   h.update(chunk);left-=len(chunk)
   if keep:parts.append(chunk)
  if h.hexdigest()!=item['payloadSHA256']:raise ValueError('Original parameter byte witness differs')
  parameters[item['name']]={'payloadSHA256':h.hexdigest(),'bytes':b-a,'shape':entry['shape']}
  if keep:fused=b''.join(parts)
width=3072*2
maps={'KV':fused[3072*width:9216*width],'QGU':fused[:3072*width]+fused[9216*width:]};slices={}
for name,raw in maps.items():
 item=next(x for x in s['physicalMaps'] if x['name']==name);digest=hashlib.sha256(raw).hexdigest()
 if digest!=item['payloadSHA256'] or len(raw)!=item['bytes']:raise ValueError('Physical terminal row bytes differ')
 slices[name]={'payloadSHA256':digest,'bytes':len(raw),'shape':item['shape']}
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,
 'originalParameterWitnesses':parameters,'physicalSliceWitnesses':slices}))
'''


def items(p):
    s=read(p/'inputs.json');values=list(fields(s['E']))+list(fields(s['noise']))
    values += [x for n in ['targets.json','candidate.json','differences.json'] for x in fields(read(p/n))]
    values += [{'path':str(Path(s['modelRoot'])/n),'sha256':h} for n,h in s['imageAssets'].items()]
    native=Path(s['E']['path']).parents[3]
    values += [{'path':str(native/'response-result'/n),'sha256':h} for n,h in s['sourceRecords'].items()]
    return values


def verify(root):
    from mrun.client.api import Api
    root,c,j,r=checked(root);p=root/'result';s=read(p/'inputs.json');e=read(p/'execution.json');api=Api();roots=[]
    for job_id,custody in [(s['nativeJobID'],s['nativeCustody']),(s['noiseJobID'],s['noiseCustody']),(s['parentLivenessJobID'],s['parentLivenessCustody'])]:
        parent=api.json('GET','/api/jobs/'+job_id)
        if parent['state']!='succeeded' or parent['payload_custody']!=custody:raise ValueError('Original source custody differs')
        roots.append(parent['result']['diagnostics']['cwd'])
    base=j['result']['diagnostics']['cwd'];unique={}
    for x in items(p)+[{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()]:
        if x['path'] in unique and unique[x['path']]['sha256']!=x['sha256']:raise ValueError('Conflicting byte identity')
        unique[x['path']]={**unique.get(x['path'],{}),**x}
    spec={'items':list(unique.values()),'allowedRoots':[base,s['modelRoot'],*roots],
        'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),
        'loaded':e['loadedTransformerParameters'],'physicalMaps':e['physicalTerminalMaps']}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],
        input=json.dumps(spec),text=True,capture_output=True,timeout=240)
    if proc.returncode:raise RuntimeError(proc.stderr)
    result=json.loads(proc.stdout);result.update(schema='tce-terminal-trajectory-bytes-v1',jobID=j['job_id'],
        verifierSourceSHA256=sha(__file__),templateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
        candidateSealSHA256=sha(p/'candidate-seal.json'),scope='Complete bytes and original parameter/slice witnesses only; no SSH model or tensor computations.')
    save(root/'retained-array-verification.json',result)
    return {'files':len(result['files']),'bytes':result['totalBytesVerified'],'NPYPayloadWitnesses':len(result['npyPayloadWitnesses']),
        'originalParameterWitnesses':len(result['originalParameterWitnesses']),'physicalSliceWitnesses':len(result['physicalSliceWitnesses'])}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Executed byte verifier differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for n in p.iterdir():
        if n.suffix=='.jsonl':
            raw=n.read_bytes();target=out/(n.name+'.gz');target.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless stream differs')
            streams.append({'source':n.name,'sourceSHA256':sha(n),'published':target.name,'publishedSHA256':sha(target),'sourceBytes':len(raw),'publishedBytes':target.stat().st_size})
        else:shutil.copyfile(n,out/n.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'scheduler.json',{k:j[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-terminal-trajectory-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],
        'payloadCustody':c['custody'],'completeCollectedInventory':c['inventory'],'losslessStreams':streams,
        'scope':'One actual own-state four-clock image. Original conditioning/noise and all remaining learned work paid. No new native shadow/reference, tolerance image inheritance, arbitrary-prompt or zero-neural completion.'})
    save(out/'files.json',{n.name:sha(n) for n in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(n.stat().st_size for n in out.iterdir())}


def audit(source):
    p=Path(source);r,s,c,e,t,report,v=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json','targets.json','report.json','retained-array-verification.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request identity differs')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Published member differs:'+n)
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]:raise ValueError('Frozen science source differs:'+n)
    if v['verifierSourceSHA256']!=sha(__file__) or v['templateSourceSHA256']!=sha(Path(__file__).with_name('publish_sparse_text_kv.py')):raise ValueError('Byte verifier source differs')
    for x in items(p):
        if v['files'][x['path']]['sha256']!=x['sha256'] or (x.get('payloadSHA256') and v['npyPayloadWitnesses'][x['path']]['payloadSHA256']!=x['payloadSHA256']):raise ValueError('Complete byte witness absent')
    seal=read(p/'candidate-seal.json')
    if (seal!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json'),'imageSHA256':sha(p/'candidate-A.png')}
        or report['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['candidateSealSHA256']!=sha(p/'candidate-seal.json')):raise ValueError('Candidate release differs')
    if sha(p/'native-A.png')!=t['image']['sha256'] or sha(p/'candidate-A.png')!=c['image']['sha256']:raise ValueError('Actual image join differs')
    manifest=identity({'transformer':e['loadedTransformerParameters'],'vaeCheckpointSHA256':s['imageAssets']['vae/diffusion_pytorch_model.safetensors'],'physicalTerminalMaps':e['physicalTerminalMaps']})
    if manifest!=e['parameterManifestSHA256'] or not e['sourceParameterBufferRNGClosure'] or e['numericalProfile']!=s['numericalProfile']:raise ValueError('Original parameter/profile closure differs')
    for item in e['loadedTransformerParameters']:
        if v['originalParameterWitnesses'][item['name']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Original parameter bytes differ')
    physical={x['name']:x for x in e['physicalTerminalMaps']}
    for name in ['KV','QGU']:
        if v['physicalSliceWitnesses'][name]['payloadSHA256']!=physical[name]['payloadSHA256']:raise ValueError('Physical terminal map differs')
    guard=e['fileQuarantine'];allowed=set(guard['allowedFiles']);denied=set(guard['deniedCanaries'])
    expected={s['E']['path']}|{s['noise'][n]['path'] for n in ['raw','packed','ids']}|{str(Path(s['modelRoot'])/n) for n in s['imageAssets']}
    if allowed!=expected or allowed&denied or denied!=set(s['deniedNumericalInputs']) or not set(guard['successfulFileOpenCounts']).issubset(allowed):raise ValueError('Numerical source closure differs')
    acts=e['typedActs'];expected_stages=[n+str(k) for k in range(4) for n in ['denoise','commit']]+['decode']
    if [a['stage'] for a in acts]!=expected_stages or c['counts']!=r['expectedCalls'] or len(c['fields'])!=40:raise ValueError('Actual component census differs')
    output_abis=0;state=s['noise']['packed']['payloadSHA256'];previous=None;joins=0
    for k in range(4):
        d,commit=acts[2*k:2*k+2];src=d['sourceFrame'];slots=src['slots'];meta=src['metadata']
        if slots['hidden_states']['fingerprint']!=state or slots['encoder_hidden_states']['fingerprint']!=s['E']['payloadSHA256'] or slots['img_ids']['fingerprint']!=s['noise']['ids']['payloadSHA256']:
            raise ValueError('Denoiser own-state/source join differs')
        if meta['precedingCommitFrameID']!=previous or meta['expectedOwnLatentPayloadSHA256']!=state:raise ValueError('Previous commit join differs')
        ds=commit['sourceFrame'];sm=ds['metadata']
        if ds['slots']['velocity']['fingerprint']!=c['fields']['velocity'+str(k)]['payloadSHA256'] or ds['slots']['sample']['fingerprint']!=state or sm['velocityProducerFrameID']!=d['outputFrame']['frame_id'] or sm['nativeSchedulerStepIndexBefore']!=(None if k==0 else k):raise ValueError('Own scheduler consumption differs')
        state=c['fields']['latent'+str(k+1)]['payloadSHA256'];previous=commit['outputFrame']['frame_id'];joins+=2+(k>0)
        wanted={n:c['fields'][('C' if n=='context' else n)+str(k)] for n in ['velocity','context','terminalRows','terminalImage','singleMod','temb','cosine','sine']}
        for a,outputs in [(d,wanted),(commit,{'nextLatent':c['fields']['latent'+str(k+1)]})]:
            output_abis+=len(outputs);check_act(a,outputs,r['requestSHA256'],manifest)
    decode=acts[-1]
    if decode['sourceFrame']['slots']['vaeInput']['fingerprint']!=c['fields']['vaeInput']['payloadSHA256'] or decode['sourceFrame']['metadata']['finalCommitFrameID']!=previous or decode['sourceFrame']['metadata']['finalOwnLatentPayloadSHA256']!=state:raise ValueError('Final decode declaration differs')
    check_act(decode,{'decodedRaw':c['fields']['decodedRaw']},r['requestSHA256'],manifest);output_abis+=1
    affine=[x for x in e['linears'] if x['kind']=='denoise'];att=[x for x in e['SDPA'] if x['kind']=='denoise']
    if len(affine)!=440 or len(att)!=100 or len(e['nativeDispatch'])!=100 or sum(x['nominalQKAndAVMACs'] for x in att)!=sum(x['nominalQKAndAVMACs'] for x in e['nativeDispatch']):raise ValueError('Denoiser affine/attention census differs')
    for k in range(4):
        subset=[x for x in affine if x['step']==k and x['parameter'].startswith('terminalPhysical.')]
        if [(x['parameter'],x['inputShape'],x['weightShape']) for x in subset]!=[('terminalPhysical.KV',[1,1536,3072],[6144,3072]),('terminalPhysical.QGU',[1,1024,3072],[21504,3072])]:raise ValueError('Reduced physical map geometry differs')
        if not any(x['step']==k and x['QShape']==[1,24,1024,128] and x['KShape']==x['VShape']==[1,24,1536,128] for x in att):raise ValueError('Rectangular complete-bank read absent')
    totals=[sum(x['nominalMACs'] for x in e['linears']),sum(x['nominalQKAndAVMACs'] for x in e['SDPA']),sum(x['nominalMACs'] for x in e['convolutions'])]
    if totals!=[e['affineMACs'],e['attentionMACs'],e['convolutionMACs']] or sum(totals)!=report['work']['modelMatrixAndConvolutionMACs'] or e['avoidedTerminalMatrixMACs']!=231928233984:raise ValueError('Actual work ledger differs')
    rows=[];provenance=read(p/'provenance.json')
    for stream in provenance['losslessStreams']:
        raw=gzip.decompress((p/stream['published']).read_bytes())
        if hashlib.sha256(raw).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless scalar stream differs')
        if stream['source']=='observations.jsonl':rows=[json.loads(x) for x in raw.splitlines()]
    addresses=[(x['field'],x['panel'],x['index']) for x in rows]
    expected={(name,'row',i) for name in t['fields'] for i in range(512 if name.startswith('C') or name in ['decodedRaw','rgbFloat','pngRGB'] else 64 if name=='vaeInput' else 1024)}
    if len(addresses)!=len(set(addresses)) or set(addresses)!=expected or len(rows)!=report['observations'] or len(rows)!=11840:raise ValueError('Complete signed row census differs')
    return {'schema':'tce-terminal-trajectory-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'typedActs':len(acts),'explicitOutputABIs':output_abis,
        'ownStateConsumptionJoins':joins,'actualVAEDecodes':1,'actualCandidateImages':1,'newNativeReferences':0,'originalParameterByteWitnesses':len(v['originalParameterWitnesses']),
        'physicalSliceByteWitnesses':2,'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'observations':len(rows),
        'denoiserAffineCalls':440,'denoiserAttentionCalls':100,'modelMatrixAndConvolutionMACs':sum(totals),'avoidedTerminalMatrixMACs':231928233984,
        'decodeInputScope':'Pinned native final latent unpack/BN/unpatchify is source-audited, not an independent numerical replay. Decode Frame directly binds actual vaeInput and decodedRaw.',
        'scope':'Custody, actual work and own-state mechanics. No effect-size gate, matched speedup, prompt quality/generalization or arbitrary-prompt zero-neural certification.'}


def check_act(a,outputs,request,manifest):
    src,dst,ev=a['sourceFrame'],a['outputFrame'],a['evidence']
    if src['frame_id']!=ev['input_frame_id'] or dst['frame_id']!=ev['output_frame_id'] or dst['parent_frame_id']!=src['frame_id'] or dst['program_fingerprint']!=request or src['metadata']['originalParameterManifestSHA256']!=manifest:raise ValueError('Typed component lineage differs')
    if ev['metrics']['outputPayloads']!={n:x['payloadSHA256'] for n,x in outputs.items()}:raise ValueError('Typed actual output differs')
    for n,x in outputs.items():
        if dst['slots'][n]['fingerprint']!=structured(x):raise ValueError('Actual output array ABI differs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source),indent=2))
