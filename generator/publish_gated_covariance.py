"""Complete bytes, original source views, typed joins and diagnostic publication."""
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
from publish_terminal_trajectory import check_act
from gated_covariance import abis,SOURCE_NAMES

BYTES=NPY_BYTES.split('print(json.dumps')[0]+r'''
parameters={};views={}
with Path(s['checkpoint']).open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n
 for item in s['loaded']:
  entry=header[item['name']];a,b=entry['data_offsets'];f.seek(base+a);raw=f.read(b-a)
  if entry['dtype']!='BF16' or entry['shape']!=item['shape'] or [a,b]!=item['dataOffsets'] or b-a!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['payloadSHA256']:raise ValueError('Original two-parameter bytes differ')
  parameters[item['name']]={'payloadSHA256':item['payloadSHA256'],'bytes':len(raw),'shape':item['shape'],'dataOffsets':[a,b]}
  for name,view in s['views'].items():
   if view['sourceParameter']!=item['name']:continue
   if name=='U':derived=raw[18432*3072*2:27648*3072*2]
   elif name=='D':derived=b''.join(raw[(row*12288+3072)*2:(row*12288+12288)*2] for row in range(3072))
   else:raise ValueError('Unknown source view')
   if len(derived)!=view['bytes'] or hashlib.sha256(derived).hexdigest()!=view['payloadSHA256']:raise ValueError('Original U/D source-view bytes differ')
   views[name]=view
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,'originalParameterWitnesses':parameters,'sourceViewWitnesses':views}))
'''


def items(p):
    s=read(p/'inputs.json');c=read(p/'candidate.json')
    return [*s['sourceFields'].values(),*c['fields'].values(),*[{'path':str(Path(s['modelRoot'])/n),'sha256':h} for n,h in s['imageAssets'].items()],
        *[{'path':s['parentRoot']+'/response-result/'+n,'sha256':h} for n,h in s['parentRecords'].items()]]


def verify(root):
    from mrun.client.api import Api
    root,col,job,r=checked(root);p=root/'result';s=read(p/'inputs.json');e=read(p/'execution.json');parent=Api().json('GET','/api/jobs/'+s['parentJobID'])
    if parent['state']!='succeeded' or parent['payload_custody']!=s['parentCustody'] or parent['result']['diagnostics']['cwd']!=s['parentRoot']:raise ValueError('Own source custody differs')
    base=job['result']['diagnostics']['cwd'];values=items(p)+[{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()];unique={}
    for x in values:
        if x['path'] in unique and unique[x['path']]['sha256']!=x['sha256']:raise ValueError('Conflicting source identities')
        unique[x['path']]={**unique.get(x['path'],{}),**x}
    spec={'items':list(unique.values()),'allowedRoots':[base,s['parentRoot'],s['modelRoot']],'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),'loaded':e['loadedParameters'],'views':e['sourceViews']}
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=300)
    if result.returncode:raise RuntimeError(result.stderr)
    v=json.loads(result.stdout);v.update(schema='tce-gated-covariance-bytes-v1',jobID=job['job_id'],candidateSealSHA256=sha(p/'candidate-seal.json'),verifierSourceSHA256=sha(__file__),byteTemplateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
        scope='Completefiles/NPY/twooriginalBF16parameters/U-Dsourceviewoffsets and rawbytes. FP64promotion/projections/reductions/productidentities sourceauditedandtinytested,not independently numerically replayed bySSHbyteverifier.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified'],'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'originalParameters':len(v['originalParameterWitnesses']),'sourceViews':len(v['sourceViewWitnesses'])}


def publish(root,destination):
    root,col,job,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Executed verifier/release differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for n in p.iterdir():
        if n.suffix=='.jsonl':
            raw=n.read_bytes();dst=out/(n.name+'.gz');dst.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(dst.read_bytes())!=raw:raise ValueError('Lossless stream publication differs')
            streams.append({'source':n.name,'sourceSHA256':sha(n),'sourceBytes':len(raw),'published':dst.name,'publishedSHA256':sha(dst),'publishedBytes':dst.stat().st_size})
        else:shutil.copyfile(n,out/n.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json');save(out/'scheduler.json',{k:job[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-gated-covariance-publication-v1','jobID':job['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':col['custody'],'completeCollectedInventory':col['inventory'],'losslessStreams':streams,
        'scope':'One fixedpaidownstate,1FP64Up/2Down; reconstructedMLPwrite/covariance/oracleconstantrepresentationdiagnostic,no servingproducer/correction/future/image/zero-neuralclaim.'})
    save(out/'files.json',{n.name:sha(n) for n in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(n.stat().st_size for n in out.iterdir())}


def audit(p):
    p=Path(p);r,s,c,e,report,v=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json','report.json','retained-array-verification.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Published file differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]:raise ValueError('Executed source changed:'+n)
    if v['verifierSourceSHA256']!=sha(__file__) or v['byteTemplateSourceSHA256']!=sha(Path(__file__).with_name('publish_sparse_text_kv.py')):raise ValueError('Executed audit source changed')
    for item in items(p):
        if v['files'][item['path']]['sha256']!=item['sha256'] or (item.get('payloadSHA256') and v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']):raise ValueError('Full byte witness absent')
    if read(p/'candidate-seal.json')!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json')} or report['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Release differs')
    manifest=identity({'originalParameters':s['originalParameters'],'ownParentSealSHA256':s['parentSealSHA256'],'completionOutputFrameID':s['completionOutputFrameID'],'helperSHA256':r['sourceHashes']['gated_covariance.py']})
    if manifest!=e['parameterManifestSHA256'] or e['loadedParameters']!=s['originalParameters'] or not e['sourceParameterRNGClosure'] or e['numericalProfile']!=s['numericalProfile']:raise ValueError('Original source/profile closure differs')
    if len(c['fields'])!=28 or {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in c['fields'].items()}!=abis():raise ValueError('28 explicit output ABIs differ')
    if len(e['typedActs'])!=1:raise ValueError('Exactly one diagnostic Act required')
    act=e['typedActs'][0];check_act(act,c['fields'],r['requestSHA256'],manifest)
    expected={n:x['payloadSHA256'] for n,x in s['sourceFields'].items()};expected.update({'parameter_'+x['name']:x['payloadSHA256'] for x in s['originalParameters']});expected.update({n:x['payloadSHA256'] for n,x in e['sourceViews'].items()})
    if {n:x['fingerprint'] for n,x in act['sourceFrame']['slots'].items()}!=expected:raise ValueError('Declared own-source/parameter/view join differs')
    meta=act['sourceFrame']['metadata']
    if meta['parentSealSHA256']!=s['parentSealSHA256'] or meta['completionOutputFrameID']!=s['completionOutputFrameID'] or meta['fixedAddress']!={'clock':3,'singleBlock':0}:raise ValueError('Fixed own source lineage differs')
    if s['parentCompletionAct']['outputFrame']['frame_id']!=s['completionOutputFrameID'] or s['parentCompletionRecord']['typedOutputFrameID']!=s['completionOutputFrameID'] or any(s['sourceFields'][name]!=s['parentCompletionRecord']['fields'][name] for name in SOURCE_NAMES):raise ValueError('Own completion descriptor lineage differs')
    if len(v['originalParameterWitnesses'])!=2 or v['sourceViewWitnesses']!=e['sourceViews']:raise ValueError('Original parameter/view byte witness differs')
    allowed={x['path'] for x in s['sourceFields'].values()}|{str(Path(s['modelRoot'])/n) for n in s['imageAssets']};guard=e['fileQuarantine']
    if set(guard['allowedFiles'])!=allowed or set(guard['deniedCanaries'])!=set(s['deniedNumericalInputs']) or allowed&set(guard['deniedCanaries']) or not set(guard['successfulFileOpenCounts']).issubset(allowed):raise ValueError('Numerical isolation differs')
    ops=e['operations'];wanted=[('fullUp',[1,1536,3072],[9216,3072],[1,1536,9216]),('totalErrorDown',[1,1536,9216],[3072,9216],[1,1536,3072]),('centeredDown',[1,1536,9216],[3072,9216],[1,1536,3072])]
    if [(x['name'],x['inputShape'],x['weightShape'],x['outputShape']) for x in ops]!=wanted or any(x['dtype']!='torch.float64' or x['device']!='cuda' for x in ops):raise ValueError('Three FP64 physical projections differ')
    for x,weight,output,input_field in zip(ops,['weightU','weightD','weightD'],['upReference','totalOutput','centeredOutput'],[None,'totalProductError','centeredProduct']):
        if x['weightPayloadSHA256']!=c['fields'][weight]['payloadSHA256'] or x['outputPayloadSHA256']!=c['fields'][output]['payloadSHA256'] or (input_field and x['inputPayloadSHA256']!=c['fields'][input_field]['payloadSHA256']):raise ValueError('Actual affine bank binding differs')
    if sum(x['nominalMACs'] for x in ops)!=130459631616 or e['totalMatrixMACs']!=130459631616 or report['totalMatrixMACs']!=130459631616:raise ValueError('Actual complete matrix work differs')
    streams={}
    for stream in read(p/'provenance.json')['losslessStreams']:
        raw=gzip.decompress((p/stream['published']).read_bytes())
        if hashlib.sha256(raw).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless stream hash differs')
        streams[stream['source']]=[json.loads(x) for x in raw.splitlines()]
    rows=streams['observations.jsonl'];channels=streams['channel-observations.jsonl']
    if len(rows)!=1536 or [x['row'] for x in rows]!=list(range(1536)) or len(channels)!=18432 or {(x['stratum'],x['channel']) for x in channels}!={(st,j) for st in ['text','image'] for j in range(9216)}:raise ValueError('Complete signed diagnostic rows differ')
    return {'schema':'tce-gated-covariance-mechanics-v1','typedActs':1,'explicitOutputABIs':28,'sourceInputs':len(expected),'originalParameters':2,'sourceViews':2,'physicalFP64Products':3,'totalMatrixMACs':130459631616,'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'observations':1536,'channelObservations':18432,
        'scope':'Ownsource/parameter/view/actualwork/typedandbyteintegrity. FP64promotion/contraction/reduction arithmetic sourceaudited/tinytested; no independent realarray numeric replay, nativefiniteconcaterror, imageattribution, cheapproducer or zero-neuralcertification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('root');p.add_argument('--destination');a=p.parse_args();print(json.dumps(publish(a.root,a.destination) if a.action=='publish' else globals()[a.action](a.root),sort_keys=True))
