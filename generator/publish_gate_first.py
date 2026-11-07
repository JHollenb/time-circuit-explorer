"""Complete byte witnesses for source-only selection and physical terminal maps."""
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
from terminal_liveness import PARAMETERS

BYTES=NPY_BYTES.split('print(json.dumps')[0]+r'''
raws={};parameters={}
with Path(s['checkpoint']).open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n
 for item in s['loaded']:
  entry=header[item['name']];a,b=entry['data_offsets']
  if entry['dtype']!='BF16' or entry['shape']!=item['shape'] or [a,b]!=item['dataOffsets']:raise ValueError('Original parameter geometry differs')
  f.seek(base+a);raw=f.read(b-a);digest=hashlib.sha256(raw).hexdigest()
  if digest!=item['payloadSHA256']:raise ValueError('Original parameter bytes differ')
  raws[item['name']]=raw;parameters[item['name']]={'payloadSHA256':digest,'bytes':len(raw),'shape':entry['shape']}
def raw_npy(item):
 with Path(item['path']).open('rb') as f:
  magic=f.read(8);n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0];f.read(n);return f.read()
support_raw=raw_npy(s['candidateFields']['selection_support']);support=list(struct.unpack('<2304q',support_raw))
if len(set(support))!=2304 or support!=sorted(support) or support[0]<0 or support[-1]>=9216:raise ValueError('Physical selected support differs')
fused=raws['single_transformer_blocks.19.attn.to_qkv_mlp_proj.weight'];out=raws['single_transformer_blocks.19.attn.to_out.weight'];width=3072*2
maps={'KV':fused[3072*width:9216*width],'QG':fused[:3072*width]+fused[9216*width:18432*width],
 'USelected':b''.join(fused[(18432+j)*width:(18433+j)*width] for j in support)}
chunks=[]
for i in range(3072):
 row=out[i*12288*2:(i+1)*12288*2]
 chunks.append(row[:3072*2]+b''.join(row[(3072+j)*2:(3073+j)*2] for j in support))
maps['outSelected']=b''.join(chunks);slices={}
for name,raw in maps.items():
 expected=next(x for x in s['physicalMaps'] if x['name']==name);digest=hashlib.sha256(raw).hexdigest()
 if digest!=expected['payloadSHA256'] or len(raw)!=expected['bytes'] or raw!=raw_npy(s['candidateFields'][expected['field']]):raise ValueError('Selected physical source-map bytes differ')
 slices[name]={'payloadSHA256':digest,'bytes':len(raw),'shape':expected['shape'],'candidateField':expected['field']}
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,
 'originalParameterWitnesses':parameters,'physicalSliceWitnesses':slices,'supportByteWitness':{'payloadSHA256':hashlib.sha256(support_raw).hexdigest(),'count':len(support),'ascendingUnique':True}}))
'''


def items(p):
    s=read(p/'inputs.json');values=list(s['sourceFields'].values())
    values += [x for n in ['targets.json','candidate.json','differences.json'] for x in fields(read(p/n))]
    return values+[{'path':str(Path(s['modelRoot'])/n),'sha256':h} for n,h in s['imageAssets'].items()]


def verify(root):
    from mrun.client.api import Api
    root,c,j,r=checked(root);p=root/'result';s=read(p/'inputs.json');e=read(p/'execution.json');parent=Api().json('GET','/api/jobs/'+s['parentJobID'])
    if parent['state']!='succeeded' or parent['payload_custody']!=s['parentCustody']:raise ValueError('Own-state parent custody differs')
    base=j['result']['diagnostics']['cwd'];prior=parent['result']['diagnostics']['cwd'];unique={}
    values=items(p)+[{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()]+[{'path':prior+'/response-result/'+n,'sha256':h} for n,h in s['parentRecords'].items()]
    for x in values:
        if x['path'] in unique and unique[x['path']]['sha256']!=x['sha256']:raise ValueError('Conflicting complete byte identity')
        unique[x['path']]={**unique.get(x['path'],{}),**x}
    spec={'items':list(unique.values()),'allowedRoots':[base,prior,s['modelRoot']],'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),
        'loaded':e['loadedParameters'],'physicalMaps':e['physicalMaps'],'candidateFields':read(p/'candidate.json')['fields']}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=240)
    if proc.returncode:raise RuntimeError(proc.stderr)
    v=json.loads(proc.stdout);v.update(schema='tce-gate-first-bytes-v1',jobID=j['job_id'],verifierSourceSHA256=sha(__file__),templateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
        candidateSealSHA256=sha(p/'candidate-seal.json'),scope='Complete files/NPY/original six parameters and four physical maps gathered by actual support. SSH raw byte operations only, no tensor numerics/model execution.')
    save(root/'retained-array-verification.json',v)
    return {'files':len(v['files']),'bytes':v['totalBytesVerified'],'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'originalParameterWitnesses':len(v['originalParameterWitnesses']),'physicalSlices':4}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Executed verifier/release differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for n in p.iterdir():
        if n.suffix=='.jsonl':
            raw=n.read_bytes();target=out/(n.name+'.gz');target.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless stream differs')
            streams.append({'source':n.name,'sourceSHA256':sha(n),'published':target.name,'publishedSHA256':sha(target),'sourceBytes':len(raw),'publishedBytes':target.stat().st_size})
        else:shutil.copyfile(n,out/n.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'scheduler.json',{k:j[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-gate-first-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],'completeCollectedInventory':c['inventory'],'losslessStreams':streams,
        'scope':'One current-gate fixedquarter terminal/head from supplied70 ownstate. Fullgate/maps/KV/nativeattention/head paid; no fullUselection/newimage/reference/fit/sweep/zero-neural claim.'})
    save(out/'files.json',{n.name:sha(n) for n in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(n.stat().st_size for n in out.iterdir())}


def audit(source):
    p=Path(source);r,s,c,e,t,report,v=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json','targets.json','report.json','retained-array-verification.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Published member differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]:raise ValueError('Frozen science source differs:'+n)
    if v['verifierSourceSHA256']!=sha(__file__) or v['templateSourceSHA256']!=sha(Path(__file__).with_name('publish_sparse_text_kv.py')):raise ValueError('Executed verifier differs')
    for x in items(p):
        if v['files'][x['path']]['sha256']!=x['sha256'] or (x.get('payloadSHA256') and v['npyPayloadWitnesses'][x['path']]['payloadSHA256']!=x['payloadSHA256']):raise ValueError('Complete supplied/emitted byte witness absent')
    if read(p/'candidate-seal.json')!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json')} or v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or report['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Candidate release differs')
    weights={x['name']:x['payloadSHA256'] for x in e['loadedParameters']};pre='single_transformer_blocks.19.attn.'
    if set(weights)!=set(PARAMETERS) or e['loadedParameters']!=s['originalParameters'] or e['originalParameterBytes']!=283902464 or not e['sourceInputParameterRNGClosure'] or e['numericalProfile']!=s['numericalProfile']:raise ValueError('Original map/profile/closure differs')
    for n,h in weights.items():
        if v['originalParameterWitnesses'][n]['payloadSHA256']!=h:raise ValueError('Original parameter witness differs')
    physical={x['name']:x for x in e['physicalMaps']}
    if set(physical)!=set(['KV','QG','USelected','outSelected']):raise ValueError('Physical map coverage differs')
    for n,item in physical.items():
        if v['physicalSliceWitnesses'][n]['payloadSHA256']!=item['payloadSHA256'] or c['fields'][item['field']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Actual gathered map differs')
    if v['supportByteWitness']['payloadSHA256']!=c['fields']['selection_support']['payloadSHA256'] or not v['supportByteWitness']['ascendingUnique'] or v['supportByteWitness']['count']!=2304:raise ValueError('Support byte witness differs')
    guard=e['fileQuarantine'];expected={x['path'] for x in s['sourceFields'].values()}|{str(Path(s['modelRoot'])/n) for n in s['imageAssets']}
    if set(guard['allowedFiles'])!=expected or set(guard['deniedCanaries'])!=set(s['deniedNumericalInputs']) or expected&set(guard['deniedCanaries']) or not set(guard['successfulFileOpenCounts']).issubset(expected):raise ValueError('Exact numerical source isolation differs')
    manifest=identity({'originalParameters':e['loadedParameters'],'sourceParentCandidateSealSHA256':s['parentCandidateSealSHA256'],'gateFirstSourceSHA256':r['sourceHashes']['gate_first_mlp.py']})
    if manifest!=e['parameterManifestSHA256'] or c['sourceParentCandidateSealSHA256']!=s['parentCandidateSealSHA256']:raise ValueError('Original source/parameter manifest differs')
    acts=e['typedActs'];stages=['sourceMaps','gates','selection','gather','image','head'];abis=0
    if [a['stage'] for a in acts]!=stages or len(c['fields'])!=24:raise ValueError('Typed stage/output census differs')
    f=lambda label:c['fields'][label]['payloadSHA256'];src=s['sourceFields']
    expected_inputs=[{'fused':weights[pre+'to_qkv_mlp_proj.weight'],'output':weights[pre+'to_out.weight']},
        {'rows':src['rows']['payloadSHA256'],'mod':src['mod']['payloadSHA256'],'KV':f('sourceMaps_KV'),'QG':f('sourceMaps_QG')},
        {'normalized':f('gates_normalizedImage'),'activation':f('gates_activation'),'upNorm':f('sourceMaps_upNorm'),'downNorm':f('sourceMaps_downNorm')},
        {'fused':weights[pre+'to_qkv_mlp_proj.weight'],'output':weights[pre+'to_out.weight'],'support':f('selection_support')},
        {**{n:src[n]['payloadSHA256'] for n in ['rows','mod','cosine','sine']},**{n:f('gates_'+n) for n in ['normalizedImage','queryRaw','keyRaw','valueRaw','activation']},'support':f('selection_support'),
         'USelected':f('gather_USelected'),'outSelected':f('gather_outSelected'),'qNorm':weights[pre+'norm_q.weight'],'kNorm':weights[pre+'norm_k.weight']},
        {'image':f('image_ownImage'),'temb':src['temb']['payloadSHA256'],**{n:weights[n] for n in PARAMETERS[-2:]}}]
    for stage,a,wanted in zip(stages,acts,expected_inputs):
        if {n:x['fingerprint'] for n,x in a['sourceFrame']['slots'].items()}!=wanted:raise ValueError('Typed source/own-state join differs:'+stage)
        outputs={n[len(stage)+1:]:item for n,item in c['fields'].items() if n.startswith(stage+'_')};check_act(a,outputs,r['requestSHA256'],manifest);abis+=len(outputs)
    if acts[-1]['sourceFrame']['slots']['image']['abi']['stride']!=[4718592,3072,1] or c['outputStrides']['image_ownImage']!=[4718592,3072,1]:raise ValueError('Own head image-view ABI differs')
    expected_ops=[('KV',[1,1536,3072],[6144,3072]),('QG',[1,1024,3072],[12288,3072]),('USelected',[1,1024,3072],[2304,3072]),('outSelected',[1,1024,5376],[3072,5376]),('norm_out.linear.weight',[1,3072],[6144,3072]),('proj_out.weight',[1,1024,3072],[128,3072])]
    if [(x['parameter'],x['inputShape'],x['weightShape']) for x in e['operations']]!=expected_ops or any(x['dtype']!='torch.bfloat16' or x['device']!='cuda' for x in e['operations']):raise ValueError('Actual physical affine geometry differs')
    if len(e['attention'])!=1 or len(e['SDPA'])!=1 or [e['attention'][0][n] for n in ['QShape','KShape','VShape']]!=[[1,1024,24,128],[1,1536,24,128],[1,1536,24,128]]:raise ValueError('Actual native attention differs')
    if sum(x['nominalMACs'] for x in e['operations'])!=92226453504 or e['affineMACs']!=92226453504 or e['attentionMACs']!=9663676416 or e['totalMatrixMACs']!=101890129920 or report['newMatrixMACs']!=101890129920:raise ValueError('Actual matrix work differs')
    streams={}
    for item in read(p/'provenance.json')['losslessStreams']:
        raw=gzip.decompress((p/item['published']).read_bytes())
        if hashlib.sha256(raw).hexdigest()!=item['sourceSHA256']:raise ValueError('Lossless observations differ')
        streams[item['source']]=[json.loads(x) for x in raw.splitlines()]
    rows=streams['observations.jsonl'];expected={(name,'image-row',i) for name in ['image','velocity'] for i in range(1024)}|{('image','channel',i) for i in range(3072)}|{('velocity','channel',i) for i in range(128)}
    if len(rows)!=5248 or {(x['field'],x['panel'],x['index']) for x in rows}!=expected or report['observations']!=5248:raise ValueError('Complete signed response coverage differs')
    selection=streams['selection-observations.jsonl']
    if len(selection)!=9216 or {x['channel'] for x in selection}!=set(range(9216)) or sum(x['selected'] for x in selection)!=2304 or not report['selection']['stableRankAndAscendingSupportVerified']:raise ValueError('Selection observation coverage differs')
    return {'schema':'tce-gate-first-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'typedActs':6,'explicitOutputABIs':abis,'ownImageHeadJoins':1,'selectedChannels':2304,'originalParameterWitnesses':6,
        'physicalSliceWitnesses':4,'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'observations':len(rows),'selectionObservations':len(selection),'affineCalls':6,'attentionCalls':1,'matrixMACs':101890129920,'avoidedMatrixMACs':43486543872,
        'scope':'Actual source/ownstate/selection/physicalwork/byte mechanics only. No effectsizegate, finite bound certificate, runtime gain, wholeimage/generalization or arbitrary-promptzero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('root');p.add_argument('--destination');a=p.parse_args()
    print(json.dumps(publish(a.root,a.destination) if a.action=='publish' else globals()[a.action](a.root),sort_keys=True))
