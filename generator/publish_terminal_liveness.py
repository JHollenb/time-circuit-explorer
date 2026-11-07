"""Complete bytes and source-row witnesses for physical terminal liveness."""
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
from terminal_liveness import PARAMETERS

BYTES=NPY_BYTES.split('print(json.dumps')[0]+r'''
checkpoint=Path(s['checkpoint']);parameter_witnesses={};raws={}
with checkpoint.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n
 for item in s['loaded']:
  entry=header[item['name']];a,b=entry['data_offsets']
  if entry['dtype']!='BF16' or entry['shape']!=item['shape'] or [a,b]!=item['dataOffsets']:raise ValueError('Original byte map geometry differs')
  f.seek(base+a);raw=f.read(b-a);digest=hashlib.sha256(raw).hexdigest()
  if digest!=item['payloadSHA256']:raise ValueError('Original parameter byte witness differs')
  parameter_witnesses[item['name']]={'payloadSHA256':digest,'bytes':len(raw),'shape':entry['shape']};raws[item['name']]=raw
fused=raws['single_transformer_blocks.19.attn.to_qkv_mlp_proj.weight'];width=3072*2
maps={'KV':fused[3072*width:9216*width],'QGU':fused[:3072*width]+fused[9216*width:]}
slice_witnesses={}
for name,raw in maps.items():
 expected=next(x for x in s['physicalMaps'] if x['name']==name);digest=hashlib.sha256(raw).hexdigest()
 if digest!=expected['payloadSHA256'] or len(raw)!=expected['bytes']:raise ValueError('Physical weight row byte witness differs')
 slice_witnesses[name]={'payloadSHA256':digest,'bytes':len(raw),'shape':expected['shape'],'source':'single_transformer_blocks.19.attn.to_qkv_mlp_proj.weight'}
def raw_npy(item):
 with Path(item['path']).open('rb') as f:
  magic=f.read(8);n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0];f.read(n);return f.read()
def raw_slot(record):
 p=next(x for x in record['files'] if x.endswith('.safetensors'));raw=Path(p).read_bytes();n=struct.unpack('<Q',raw[:8])[0];header=json.loads(raw[8:8+n])
 entry=header[record['metadata']['encoded']['name']];a,b=entry['data_offsets'];payload=raw[8+n+a:8+n+b]
 if entry['dtype']!='BF16' or entry['shape']!=record['declaredDescriptor']['shape'] or hashlib.sha256(payload).hexdigest()!=record['declaredDescriptor']['sha256']:raise ValueError('Source slot byte witness differs')
 return payload
source=raw_slot(s['sourceCut']['pages']['text'])+raw_slot(s['sourceCut']['pages']['image'])
if source!=raw_npy(s['sourceCombined']):raise ValueError('Complete source concatenation byte witness differs')
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,
 'originalParameterWitnesses':parameter_witnesses,'physicalSliceWitnesses':slice_witnesses,'sourceCombinedWitness':{'payloadSHA256':hashlib.sha256(source).hexdigest(),'byteExact':True,'bytes':len(source)}}))
'''


def items(p):
    s=read(p/'inputs.json');values=[s['sourceCut']['cutFile']]
    values += [{'path':n,'sha256':h} for page in s['sourceCut']['pages'].values() for n,h in page['files'].items()]
    values += [{'path':str(Path(s['modelRoot'])/n),'sha256':h} for n,h in s['imageAssets'].items()]
    return values+[x for name in ['targets.json','candidate.json','differences.json'] for x in fields(read(p/name))]


def verify(root):
    from mrun.client.api import Api
    root,c,j,r=checked(root);p=root/'result';s=read(p/'inputs.json');e=read(p/'execution.json');parent=Api().json('GET','/api/jobs/'+s['parentJobID'])
    if parent['state']!='succeeded' or parent['payload_custody']!=s['parentCustody']:raise ValueError('Native capture custody differs')
    unique={};base=j['result']['diagnostics']['cwd']
    for item in items(p)+[{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()]:
        if item['path'] in unique and unique[item['path']]['sha256']!=item['sha256']:raise ValueError('Conflicting complete byte identity')
        unique[item['path']]={**unique.get(item['path'],{}),**item}
    spec={'items':list(unique.values()),'allowedRoots':[base,s['modelRoot'],parent['result']['diagnostics']['cwd']],
        'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),'loaded':e['loadedParameters'],
        'physicalMaps':e['physicalMaps'],'sourceCut':s['sourceCut'],'sourceCombined':read(p/'targets.json')['sourceCombined']}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','beast','python3 -c '+shlex.quote(BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=240)
    if proc.returncode:raise RuntimeError(proc.stderr)
    result=json.loads(proc.stdout);result.update(schema='tce-terminal-liveness-bytes-v1',jobID=j['job_id'],verifierSourceSHA256=sha(__file__),
        templateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),candidateSealSHA256=sha(p/'candidate-seal.json'))
    save(root/'retained-array-verification.json',result)
    return {'files':len(result['files']),'bytes':result['totalBytesVerified'],'NPYPayloadWitnesses':len(result['npyPayloadWitnesses']),'physicalSliceWitnesses':2}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for n in p.iterdir():
        if n.suffix=='.jsonl':
            raw=n.read_bytes();target=out/(n.name+'.gz');target.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless observations differ')
            streams.append({'source':n.name,'sourceSHA256':sha(n),'published':target.name,'publishedSHA256':sha(target),'sourceBytes':len(raw),'publishedBytes':target.stat().st_size})
        else:shutil.copyfile(n,out/n.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'scheduler.json',{k:j[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-terminal-liveness-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],
        'payloadCustody':c['custody'],'completeCollectedInventory':c['inventory'],'losslessStreams':streams,
        'scope':'One physically reduced native learned terminal producer and original head, plus saved-only comparison. Source/capture/remaininglearnedwork paid; no newreference/image/zero-neural completion.'})
    save(out/'files.json',{n.name:sha(n) for n in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(n.stat().st_size for n in out.iterdir())}


def audit(source):
    p=Path(source);r,s,c,e,report,v=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json','report.json','retained-array-verification.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Published member differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]:raise ValueError('Frozen scientific source differs')
    if v['verifierSourceSHA256']!=sha(__file__) or v['templateSourceSHA256']!=sha(Path(__file__).with_name('publish_sparse_text_kv.py')):raise ValueError('Executed byte verifier differs')
    for x in items(p):
        if v['files'][x['path']]['sha256']!=x['sha256'] or (x.get('payloadSHA256') and v['npyPayloadWitnesses'][x['path']]['payloadSHA256']!=x['payloadSHA256']):raise ValueError('Full supplied/emitted byte custody absent')
    seal=read(p/'candidate-seal.json')
    if seal!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json')} or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Candidate release differs')
    weights={x['name']:x['payloadSHA256'] for x in e['loadedParameters']};physical={x['name']:x['payloadSHA256'] for x in e['physicalMaps']}
    if set(weights)!=set(PARAMETERS) or e['originalParameterBytes']!=283902464 or not e['sourceInputParameterRNGClosure']:raise ValueError('Original map/closure census differs')
    for name,h in weights.items():
        if v['originalParameterWitnesses'][name]['payloadSHA256']!=h:raise ValueError('Original raw weight witness differs')
    for name in ['KV','QGU']:
        if v['physicalSliceWitnesses'][name]['payloadSHA256']!=physical[name]:raise ValueError('Physical row byte witness differs')
    for name,original in [('qNorm',PARAMETERS[1]),('kNorm',PARAMETERS[2]),('out',PARAMETERS[3])]:
        if physical[name]!=weights[original]:raise ValueError('Unsliced original map differs')
    guard=e['fileQuarantine'];allowed=set(guard['allowedFiles']);denied=set(guard['deniedCanaries']);cut=s['sourceCut'];expected={cut['cutFile']['path']}|{n for page in cut['pages'].values() for n in page['files']}|{str(Path(s['modelRoot'])/n) for n in s['imageAssets']}
    if allowed!=expected or allowed&denied or set(s['deniedNumericalInputs'])!=denied or not set(guard['successfulFileOpenCounts']).issubset(allowed):raise ValueError('Exact source-page input closure differs')
    declared={x['key']['value']:x['value'] for x in cut['manifest']['payload']['items']};context={x['key']['value']:x['value'] for x in declared['context']['items']}
    acts=e['typedActs']
    if [x['stage'] for x in acts]!=['image','velocity']:raise ValueError('Own producer/consumer census differs')
    if not v['sourceCombinedWitness']['byteExact'] or v['sourceCombinedWitness']['payloadSHA256']!=s['sourceCombinedPayloadSHA256']:raise ValueError('Original combined source bytes differ')
    for a in acts:
        src,dst,ev=a['sourceFrame'],a['outputFrame'],a['evidence'];actual={n:x['fingerprint'] for n,x in src['slots'].items()}
        expected={**physical,'rows':s['sourceCombinedPayloadSHA256'],'mod':context['single_mod']['sha256'],'cosine':context['rope']['items'][0]['sha256'],'sine':context['rope']['items'][1]['sha256']} if a['stage']=='image' else {
            'image':c['fields']['image']['payloadSHA256'],'temb':context['temb']['sha256'],**{n:weights[n] for n in PARAMETERS[-2:]}}
        if actual!=expected or src['frame_id']!=ev['input_frame_id'] or dst['frame_id']!=ev['output_frame_id'] or dst['parent_frame_id']!=src['frame_id'] or dst['program_fingerprint']!=r['requestSHA256']:raise ValueError('Actual typed source/own-state lineage differs')
        field='image' if a['stage']=='image' else 'velocity';name='ownImage' if field=='image' else 'ownVelocity';item=c['fields'][field]
        if ev['metrics']['outputPayloads']!={name:item['payloadSHA256']} or dst['slots'][name]['fingerprint']!=structured(item):raise ValueError('Typed emitted array/ABI differs')
    if acts[0]['sourceFrame']['slots']['rows']['abi']['stride']!=[4718592,3072,1] or acts[1]['sourceFrame']['slots']['image']['abi']['stride']!=c['imageOutputStride'] or c['imageOutputStride']!=[4718592,3072,1]:raise ValueError('Source/consumed image-view stride differs')
    shapes=[(x['parameter'],x['inputShape'],x['weightShape']) for x in e['operations']]
    if shapes!=[('KV',[1,1536,3072],[6144,3072]),('QGU',[1,1024,3072],[21504,3072]),('out',[1,1024,12288],[3072,12288]),('norm_out.linear.weight',[1,3072],[6144,3072]),('proj_out.weight',[1,1024,3072],[128,3072])]:raise ValueError('Physical affine geometry differs')
    if len(e['attention'])!=1 or [e['attention'][0][n] for n in ['QShape','KShape','VShape']]!=[[1,1024,24,128],[1,1536,24,128],[1,1536,24,128]]:raise ValueError('Full-bank rectangular attention differs')
    if sum(x['nominalMACs'] for x in e['operations'])!=e['affineMACs'] or e['affineMACs']!=135712997376 or e['attentionMACs']!=9663676416 or e['totalMatrixMACs']!=report['newMatrixMACs'] or report['newMatrixMACs']!=145376673792:raise ValueError('Executed reduced work differs')
    raw=gzip.decompress((p/'observations.jsonl.gz').read_bytes());rows=[json.loads(x) for x in raw.splitlines()]
    provenance=read(p/'provenance.json')
    if hashlib.sha256(raw).hexdigest()!=provenance['losslessStreams'][0]['sourceSHA256']:raise ValueError('Lossless scalar stream differs')
    addresses=[(x['field'],x['panel'],x['index']) for x in rows];expected={(f,'image-row',i) for f in ['image','velocity'] for i in range(1024)}|{('image','channel',i) for i in range(3072)}|{('velocity','channel',i) for i in range(128)}
    if len(addresses)!=len(set(addresses)) or set(addresses)!=expected or report['observations']!=len(rows):raise ValueError('Complete unique signed observation census differs')
    return {'schema':'tce-terminal-liveness-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'typedActs':2,'explicitOutputABIs':2,'ownImageConsumerJoins':1,
        'nativeOriginalTensors':6,'physicalSliceByteWitnesses':2,'originalParameterByteWitnesses':6,'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),
        'observations':len(rows),'affineCalls':5,'attentionCalls':1,'totalMatrixMACs':145376673792,'avoidedMatrixMACs':57982058496,
        'scope':'Actual physical work/source/ownstate/byte mechanics only; no effectsizegate, matchedlatency/image/generalization/arbitrary-promptzero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source),indent=2))
