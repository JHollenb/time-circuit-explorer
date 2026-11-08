"""Complete source/parameter/physical-join bytes and public scalar publication."""
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
from publish_terminal_trajectory import check_act
from own_successor_intake import BYTES as INTAKE_BYTES
from own_successor import abis,work,PARAMETERS,SOURCE_NAMES
from own_successor_analysis import analysis_abis,ROW_COLUMNS,COLUMNS,summarize,panel

BYTES=INTAKE_BYTES.split('print(json.dumps')[0]+r'''
def raw_npy(item):
 with Path(item['path']).open('rb') as f:
  magic=f.read(8);n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0];f.read(n);return f.read()
def joined(names,widths):
 raw=[raw_npy(s['candidate'][name]) for name in names];h=hashlib.sha256();size=0
 for value,width in zip(raw,widths):
  if len(value)!=1536*width*2:raise ValueError('Join geometry differs')
 for row in range(1536):
  for value,width in zip(raw,widths):
   piece=value[row*width*2:(row+1)*width*2];h.update(piece);size+=len(piece)
 return {'payloadSHA256':h.hexdigest(),'bytes':size}
out_join=joined(['attentionRead','gu_product'],[3072,9216])
qkv_join=joined(['queryRaw','keyRaw','valueRaw'],[3072,3072,3072])
if out_join['payloadSHA256']!=s['outputJoin']['payloadSHA256'] or qkv_join['payloadSHA256']!=s['qkvProjectionPayload']:raise ValueError('Actual own-state operand join differs')
with Path(s['checkpoint']).open('rb') as f:
 entry=head[s['parameters'][0]['name']];a,b=entry['data_offsets'];f.seek(base+a);raw=f.read(9216*3072*2)
physical={'payloadSHA256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
if physical['payloadSHA256']!=s['physicalQKV']['payloadSHA256']:raise ValueError('Physical original QKV slice differs')
sketches={}
for name,key in [('outputSigns','signsPayloadSHA256'),('gatedOutputSigns','gatedSignsPayloadSHA256'),('pulledBackSketch','pulledBackPayloadSHA256')]:
 raw=raw_npy(s['candidate'][name]);digest=hashlib.sha256(raw).hexdigest()
 if digest!=s['outputSketch'][key]:raise ValueError('Observed shared sketch/pullback differs')
 sketches[name]={'bytes':len(raw),'payloadSHA256':digest}
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,'originalParameters':parameters,'physicalQKV':physical,'actualBF16Join':out_join,'actualQKVJoin':qkv_join,'outputSketchWitnesses':sketches}))
'''


def items(p):
    s=read(p/'inputs.json')
    return [*s['sourceFields'].values(),*list(fields(s['comparisonFields'])),
            *list(fields(read(p/'candidate.json'))),*list(fields(read(p/'analysis-fields.json')))]


def verify(root):
    from mrun.client.api import Api
    root,col,job,r=checked(root);p=root/'result';s=read(p/'inputs.json');e=read(p/'execution.json');api=Api()
    for parent in s['parents'].values():
        actual=api.json('GET','/api/jobs/'+parent['jobID'])
        if actual['state']!='succeeded' or actual['payload_custody']!=parent['custody']:raise ValueError('Original parent custody differs')
    base=job['result']['diagnostics']['cwd']
    values=items(p)+[{'path':base+'/response-result/'+x.name,'sha256':sha(x)} for x in p.iterdir()]
    values += [{'path':name,'sha256':digest} for name,digest in s['metadataFiles'].items()]
    unique={}
    for item in values:
        if item['path'] in unique and unique[item['path']]['sha256']!=item['sha256']:raise ValueError('Conflicting byte descriptor')
        unique[item['path']]=item
    spec={'allowedRoots':[base,*s['sourceRoots']],'items':list(unique.values()),
          'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),
          'parameters':e['loadedParameters'],'candidate':read(p/'candidate.json')['fields'],
          'outputJoin':e['actualBF16Join'],'qkvProjectionPayload':e['linears'][0]['outputPayloadSHA256'],
          'physicalQKV':e['physicalQKV'],'outputSketch':e['outputSketch']}
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=300)
    if result.returncode:raise RuntimeError(result.stderr)
    value=json.loads(result.stdout);value.update(schema='tce-own-successor-bytes-v1',jobID=job['job_id'],verifierSourceSHA256=sha(__file__),
        byteTemplateSourceSHA256=sha(Path(__file__).with_name('own_successor_intake.py')),candidateSealSHA256=sha(p/'candidate-seal.json'),
        scope='Complete source/parent/successor/candidate/diagnostic NPY/file bytes,four original parameter bytes,physical QKV slice and actual QKV/J/sketch handoffs. SSH bytes only; no numerical tensor replay.')
    save(root/'retained-array-verification.json',value)
    return {'files':len(value['files']),'NPY':len(value['npyPayloadWitnesses']),'bytes':value['totalBytesVerified'],'parameterBytes':sum(x['bytes'] for x in e['loadedParameters'])}


def publish(root,destination):
    root,col,job,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Verified source/release differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for path in p.iterdir():
        if path.suffix=='.jsonl':
            raw=path.read_bytes();target=out/(path.name+'.gz');target.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless rows differ')
            streams.append({'source':path.name,'sourceSHA256':sha(path),'sourceBytes':len(raw),'published':target.name,'publishedSHA256':sha(target),'publishedBytes':target.stat().st_size})
        else:shutil.copyfile(path,out/path.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'scheduler.json',{k:job[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-own-successor-publication-v1','jobID':job['job_id'],'requestSHA256':r['requestSHA256'],
        'payloadCustody':col['custody'],'completeCollectedInventory':col['inventory'],'losslessStreams':streams,
        'scope':'One ownstate learned single1 transition vs paid70 dense-affine moment-attention parent/successor. Original maps/clocks paid; no independent fullattention/nativeparity/compactclosure/image/arbitrarypromptzero-neural claim.'})
    save(out/'files.json',{x.name:sha(x) for x in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    import math
    import numpy as np
    p=Path(source);r,s,c,e,a,ae,v,report=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json','analysis-fields.json','analysis-execution.json','retained-array-verification.json','report.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request identity differs')
    for name,digest in read(p/'files.json').items():
        if sha(p/name)!=digest:raise ValueError('Published member differs:'+name)
    for name in r['ownedSources']+['inputs.json','resource-basis.json','frozen-plan.md','implementation-notes.md','intake.json']:
        target=Path(__file__).with_name(name) if name in r['ownedSources'] else p/name
        if sha(target)!=r['sourceHashes'][name]:raise ValueError('Frozen source/metadata changed:'+name)
    if v['verifierSourceSHA256']!=sha(__file__) or v['byteTemplateSourceSHA256']!=sha(Path(__file__).with_name('own_successor_intake.py')):raise ValueError('Byte verifier changed')
    intake=read(p/'intake.json')
    if sha(p/'intake.json')!=s['intakeSHA256'] or s['sourceFields']!=intake['sourceFields'] or s['comparisonFields']!=intake['comparisonFields'] or s['originalParameters']!=intake['originalParameters'] or s['parents']!=intake['parents']:raise ValueError('Fresh intake ancestry differs')
    for item in items(p):
        if v['files'][item['path']]['sha256']!=item['sha256'] or v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Complete NPY byte witness missing')
    for name,digest in s['metadataFiles'].items():
        if v['files'][name]['sha256']!=digest:raise ValueError('Original metadata witness missing')
    sealed=sha(p/'candidate-seal.json');exited=read(p/'producer-exit.json')
    if read(p/'candidate-seal.json')!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json')}:raise ValueError('Producer seal differs')
    if any(x.get('candidateSealSHA256')!=sealed for x in [a,ae,v,report,exited]) or ae['producerExitSHA256']!=sha(p/'producer-exit.json') or exited['requestSHA256']!=r['requestSHA256'] or exited['returncode']!=0 or exited['producerPID']==ae['analyzerPID']:raise ValueError('Observed exit/post-exit release differs')
    if set(s['sourceFields'])!=set(SOURCE_NAMES) or e['loadedParameters']!=s['originalParameters'] or [x['name'] for x in e['loadedParameters']]!=PARAMETERS or not e['sourceParameterRNGClosure'] or e['numericalProfile']!=s['numericalProfile'] or ae['numericalProfile']!=s['numericalProfile']:raise ValueError('Declared source/profile/parameter closure differs')
    allowed={x['path'] for x in s['sourceFields'].values()}|{str(Path(s['modelRoot'])/n) for n in s['imageAssets']};guard=e['fileQuarantine']
    if set(guard['allowedFiles'])!=allowed or set(guard['deniedCanaries'])!=set(s['deniedNumericalInputs']) or not set(guard['successfulFileOpenCounts']).issubset(allowed) or not {x['path'] for x in fields(s['comparisonFields'])}.issubset(s['deniedNumericalInputs']):raise ValueError('Source-only numerical isolation differs')
    manifest=identity({'originalParameters':s['originalParameters'],'intakeSHA256':s['intakeSHA256'],'nativeSources':s['nativeSources']})
    expected={**{n:x['payloadSHA256'] for n,x in s['sourceFields'].items()},**{'parameter_'+x['name']:x['payloadSHA256'] for x in s['originalParameters']}}
    if len(e['typedActs'])!=1 or e['parameterManifestSHA256']!=manifest or {n:x['fingerprint'] for n,x in e['typedActs'][0]['sourceFrame']['slots'].items()}!=expected:raise ValueError('Typed source binding differs')
    check_act(e['typedActs'][0],c['fields'],r['requestSHA256'],manifest)
    if {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in c['fields'].items()}!=abis() or {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in a['fields'].items()}!=analysis_abis():raise ValueError('Complete output ABI differs')
    wanted=[]
    for stream,rows in enumerate([512,1024]):
        wanted += [('guSketch'+str(stream),[3072,rows],[rows,128]),('guCoordinates'+str(stream),[rows,3072],[3072,128]),('guCompile'+str(stream),[128,3072],[3072,18432]),('guReconstruct'+str(stream),[rows,128],[128,18432])]
    wanted += [('pullbackSketch',[12288,3072],[3072,128])]
    for stream,rows in enumerate([512,1024]):
        wanted += [('sketch'+str(stream),[rows,12288],[12288,128]),('projectInputs'+str(stream),[128,rows],[rows,12288]),('compressedOut'+str(stream),[128,12288],[12288,3072]),('reconstructUpdate'+str(stream),[rows,128],[128,3072])]
    if [(x['name'],x['leftShape'],x['rightShape']) for x in e['products']]!=wanted or any(x['dtype']!='torch.float64' or x['device']!='cuda' or x['nominalMACs']!=math.prod(x['outputShape'])*x['leftShape'][-1] for x in e['products']):raise ValueError('Actual FP64 geometry differs')
    linear=e['linears'][0]
    if len(e['linears'])!=1 or linear['inputShape']!=[1,1536,3072] or linear['weightShape']!=[9216,3072] or linear['dtype']!='torch.bfloat16' or linear['device']!='cuda' or linear['nominalMACs']!=43486543872:raise ValueError('Physical sliced QKV differs')
    if linear['inputPayloadSHA256']!=c['fields']['normalized']['payloadSHA256'] or linear['weightPayloadSHA256']!=e['physicalQKV']['payloadSHA256'] or linear['weightPayloadSHA256']!=v['physicalQKV']['payloadSHA256'] or linear['outputPayloadSHA256']!=v['actualQKVJoin']['payloadSHA256']:raise ValueError('Physical QKV handoff differs')
    if e['actualBF16Join']['payloadSHA256']!=v['actualBF16Join']['payloadSHA256']:raise ValueError('Current read/product join differs')
    sketch=e['outputSketch']
    if sketch['seedSequence']!=[20261107,86] or sketch['shape']!=[3072,128]:raise ValueError('Frozen output sketch policy differs')
    for name,key in [('outputSigns','signsPayloadSHA256'),('gatedOutputSigns','gatedSignsPayloadSHA256'),('pulledBackSketch','pulledBackPayloadSHA256')]:
        if sketch[key]!=c['fields'][name]['payloadSHA256'] or sketch[key]!=v['outputSketchWitnesses'][name]['payloadSHA256']:raise ValueError('Shared physical sketch differs')
    if e['products'][8]['rightPayloadSHA256']!=sketch['gatedSignsPayloadSHA256'] or e['products'][8]['outputPayloadSHA256']!=sketch['pulledBackPayloadSHA256']:raise ValueError('Learned pullback handoff differs')
    for i in range(2):
        start=9+4*i
        if e['products'][start]['rightPayloadSHA256']!=sketch['pulledBackPayloadSHA256'] or e['products'][start+2]['rightPayloadSHA256']!=e['products'][8]['leftPayloadSHA256']:raise ValueError('Shared V/Wout consumer differs')
    if len(e['momentProducts'])!=5 or [x['name'] for x in e['momentProducts']]!=['imageCovariance','queryCovariance','queryMean','textScores','textAV'] or sum(x['nominalMACs'] for x in e['momentProducts'])!=2216165376 or any(x['dtype']!='torch.float32' or x['device']!='cuda' for x in e['momentProducts']):raise ValueError('Live text-moment work differs')
    if len(e['SDPA'])!=1 or e['SDPA'][0]['QShape']!=[1,24,1024,128] or e['SDPA'][0]['KShape']!=[1,24,1536,128] or e['SDPA'][0]['nominalQKAndAVMACs']!=9663676416:raise ValueError('Live image attention geometry differs')
    if len(e['QR'])!=4 or [x['inputShape'] for x in e['QR']]!=[[3072,128],[3072,128],[512,128],[1024,128]] or e['QRProfile']['requestedPreference']!='cusolver':raise ValueError('Own factorization calls differ')
    costs=work();candidate=sum(x['nominalMACs'] for x in e['products']+e['linears']+e['momentProducts'])+sum(x['nominalQKAndAVMACs'] for x in e['SDPA'])
    if costs!=report['work'] or costs!=e['work'] or candidate!=costs['candidateMACs'] or len(ae['products'])!=8 or sum(x['nominalMACs'] for x in ae['products'])!=costs['diagnosticMACs']:raise ValueError('Actual work ledger differs')
    streams=read(p/'provenance.json')['losslessStreams'];raw=gzip.decompress((p/streams[0]['published']).read_bytes())
    if len(streams)!=1 or hashlib.sha256(raw).hexdigest()!=streams[0]['sourceSHA256']:raise ValueError('Lossless row bytes differ')
    rows=[json.loads(line) for line in raw.splitlines()]
    if len(rows)!=1536 or [x['row'] for x in rows]!=list(range(1536)):raise ValueError('Complete row census differs')
    for name,lo,hi in [('all',0,1536),('text',0,512),('image',512,1536)]:
        if summarize(np.array([[x['statistics'][k] for k in ROW_COLUMNS] for x in rows[lo:hi]]))!=report['panels'][name]:raise ValueError('D+W scalar aggregate differs')
        for field,key in [('inputReconstruction','inputReconstruction'),('outputSketchReconstruction','outputSketchReconstruction')]:
            if panel(np.array([[x[field][k] for k in COLUMNS] for x in rows[lo:hi]]))!=report['factorizationDiagnostics'][name][key]:raise ValueError('Diagnostic scalar aggregate differs')
    return {'schema':'tce-own-successor-mechanics-v1','typedActs':1,'typedInputs':9,'producerABIs':45,'analysisABIs':7,'BF16Projections':1,'FP64CandidateProducts':17,
            'FP32MomentProducts':5,'imageSDPA':1,'QRCalls':4,'diagnosticProducts':8,'dependentRows':1536,
            'candidateMACs':costs['candidateMACs'],'diagnosticMACs':costs['diagnosticMACs'],'assayMACs':costs['assayMACs'],
            'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),
            'scope':'Source/byte/physical joins/typed/geometry/work/fullcentered scalar mechanics. No independent numerical replay,native fused-projection parity,error-source attribution,compact closure,image or strictzero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('root');p.add_argument('--destination');a=p.parse_args()
    print(json.dumps(publish(a.root,a.destination) if a.action=='publish' else globals()[a.action](a.root)))
