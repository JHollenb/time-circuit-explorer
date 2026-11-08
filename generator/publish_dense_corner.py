"""Full native-byte joins, exact operator work and six-corner scalar publication."""
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
from dense_corner import abis,work,PARAMETERS,SOURCE_NAMES
from dense_corner_analysis import analysis_abis,ROW_COLUMNS,summarize
from dense_corner_study import validate_corner_metadata

BYTES=INTAKE_BYTES.split('print(json.dumps')[0]+r'''
def raw_npy(item):
 with Path(item['path']).open('rb') as f:
  magic=f.read(8);n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0];f.read(n);return f.read()
def joined(names,widths):
 raw=[raw_npy(s['candidate'][name]) for name in names];h=hashlib.sha256();size=0
 for value,width in zip(raw,widths):
  if len(value)!=1536*width*2:raise ValueError('Native join geometry differs')
 for row in range(1536):
  for value,width in zip(raw,widths):
   piece=value[row*width*2:(row+1)*width*2];h.update(piece);size+=len(piece)
 return {'payloadSHA256':h.hexdigest(),'bytes':size}
fused=joined(['queryRaw','keyRaw','valueRaw','gateRaw','upRaw'],[3072,3072,3072,9216,9216])
j=joined(['attentionRead','product'],[3072,9216])
if fused['payloadSHA256']!=s['candidate']['fused']['payloadSHA256'] or j['payloadSHA256']!=s['candidate']['outputJoin']['payloadSHA256']:raise ValueError('Actual native fused/J joins differ')
h=hashlib.sha256()
for name in ['textOut','imageOut']:h.update(raw_npy(s['candidate'][name]))
if h.hexdigest()!=s['candidate']['exit']['payloadSHA256']:raise ValueError('Actual native exit join differs')
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,'originalParameters':parameters,'actualFusedJoin':fused,'actualOutputJoin':j,'actualExitJoinPayloadSHA256':h.hexdigest()}))
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
        if actual['state']!='succeeded' or actual['payload_custody']!=parent['custody']:raise ValueError('Original custody differs')
    base=job['result']['diagnostics']['cwd']
    values=items(p)+[{'path':base+'/response-result/'+x.name,'sha256':sha(x)} for x in p.iterdir()]
    values += [{'path':name,'sha256':digest} for name,digest in s['metadataFiles'].items()]
    unique={}
    for item in values:
        if item['path'] in unique and unique[item['path']]['sha256']!=item['sha256']:raise ValueError('Conflicting byte descriptor')
        unique[item['path']]=item
    spec={'allowedRoots':[base,*s['sourceRoots']],'items':list(unique.values()),'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),
          'parameters':e['loadedParameters'],'candidate':read(p/'candidate.json')['fields']}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],
                        input=json.dumps(spec),text=True,capture_output=True,timeout=300)
    if proc.returncode:raise RuntimeError(proc.stderr)
    v=json.loads(proc.stdout);v.update(schema='tce-dense-corner-bytes-v1',jobID=job['job_id'],verifierSourceSHA256=sha(__file__),
        byteTemplateSourceSHA256=sha(Path(__file__).with_name('own_successor_intake.py')),candidateSealSHA256=sha(p/'candidate-seal.json'),
        scope='Complete source/comparison/candidate/analysis/metadata bytes,four original tensors and actual fullfused/J/exit joins. SSHbytes/headers only,no numerical tensor replay.')
    save(root/'retained-array-verification.json',v)
    return {'files':len(v['files']),'NPY':len(v['npyPayloadWitnesses']),'bytes':v['totalBytesVerified'],'parameterBytes':sum(x['bytes'] for x in e['loadedParameters'])}


def publish(root,destination):
    root,col,job,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Verified release differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for path in p.iterdir():
        if path.suffix=='.jsonl':
            raw=path.read_bytes();target=out/(path.name+'.gz');target.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless row bytes differ')
            streams.append({'source':path.name,'sourceSHA256':sha(path),'sourceBytes':len(raw),'published':target.name,'publishedSHA256':sha(target),'publishedBytes':target.stat().st_size})
        else:shutil.copyfile(path,out/path.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'scheduler.json',{k:job[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-dense-corner-publication-v1','jobID':job['job_id'],'requestSHA256':r['requestSHA256'],
        'payloadCustody':col['custody'],'completeCollectedInventory':col['inventory'],'losslessStreams':streams,
        'scope':'One original70dense-affine/momentattention single1 onactual86input;not independentfullattention. Six-corner decisionaudit,allmaps/state/clocks/historypaid;no image/cheapcorrection/compactclosure/zero-neural result.'})
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
    intake=read(p/'intake.json');validate_corner_metadata(intake)
    if sha(p/'intake.json')!=s['intakeSHA256'] or any(s[n]!=intake[n] for n in ['sourceFields','comparisonFields','originalParameters','parents','metadataFiles','frozenConsumerSHA256']):
        raise ValueError('Six-corner intake ancestry differs')
    if intake['verifierSHA256']!=sha(Path(__file__).with_name('dense_corner_intake.py')) or s['frozenConsumerSHA256']!=sha(Path(__file__).with_name('image_partition_moments.py')) or e['frozenConsumerSHA256']!=s['frozenConsumerSHA256']:
        raise ValueError('Original70consumer/intake source differs')
    for item in items(p):
        if v['files'][item['path']]['sha256']!=item['sha256'] or v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Complete NPY witness absent')
    for name,digest in s['metadataFiles'].items():
        if v['files'][name]['sha256']!=digest:raise ValueError('Original metadata witness absent')
    sealed=sha(p/'candidate-seal.json');exited=read(p/'producer-exit.json')
    if read(p/'candidate-seal.json')!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json')}:raise ValueError('Producer seal differs')
    if any(x.get('candidateSealSHA256')!=sealed for x in [a,ae,v,report,exited]) or ae['producerExitSHA256']!=sha(p/'producer-exit.json') or exited['requestSHA256']!=r['requestSHA256'] or exited['returncode']!=0 or exited['producerPID']==ae['analyzerPID']:
        raise ValueError('Observed exit/release differs')
    if set(s['sourceFields'])!=set(SOURCE_NAMES) or e['loadedParameters']!=s['originalParameters'] or [x['name'] for x in e['loadedParameters']]!=PARAMETERS or not e['sourceParameterRNGClosure'] or e['numericalProfile']!=s['numericalProfile']:
        raise ValueError('Original source/profile/maps closure differs')
    allowed={x['path'] for x in s['sourceFields'].values()}|{str(Path(s['modelRoot'])/n) for n in s['imageAssets']};guard=e['fileQuarantine']
    if set(guard['allowedFiles'])!=allowed or set(guard['deniedCanaries'])!=set(s['deniedNumericalInputs']) or not set(guard['successfulFileOpenCounts']).issubset(allowed) or not {x['path'] for x in fields(s['comparisonFields'])}.issubset(s['deniedNumericalInputs']):
        raise ValueError('Comparison-only numerical quarantine differs')
    manifest=identity({'originalParameters':s['originalParameters'],'intakeSHA256':s['intakeSHA256'],'nativeSources':s['nativeSources']})
    expected={**{n:x['payloadSHA256'] for n,x in s['sourceFields'].items()},**{'parameter_'+x['name']:x['payloadSHA256'] for x in s['originalParameters']}}
    if len(e['typedActs'])!=1 or e['parameterManifestSHA256']!=manifest or {n:x['fingerprint'] for n,x in e['typedActs'][0]['sourceFrame']['slots'].items()}!=expected:raise ValueError('Typed original source binding differs')
    check_act(e['typedActs'][0],c['fields'],r['requestSHA256'],manifest)
    if {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in c['fields'].items()}!=abis() or {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in a['fields'].items()}!=analysis_abis():
        raise ValueError('Complete native/six-corner ABI differs')
    if not e['nativeFieldCaptureClosure'] or e['observedFieldPayloads']!={n:x['payloadSHA256'] for n,x in c['fields'].items()} or [x['ordinal'] for x in e['captureEvents']]!=list(range(27)) or {x['field']:x['payloadSHA256'] for x in e['captureEvents']}!=e['observedFieldPayloads']:
        raise ValueError('Observed native fields changed or missed')
    if e['products'] or e['QR'] or ae['products'] or ae['diagnosticMatrixMACs']!=0 or ae['device']!='cpu' or ae['dtype']!='float64':raise ValueError('Undeclared products/QR/analysis profile')
    wanted=[('fullFused',[1,1536,3072],[27648,3072],PARAMETERS[0],'normalized','fused',130459631616),
            ('fullOut',[1,1536,12288],[3072,12288],PARAMETERS[1],'outputJoin','preResidualUpdate',57982058496)]
    if len(e['linears'])!=2:raise ValueError('Original full affine count differs')
    for x,(name,ishape,wshape,param,inp,out,macs) in zip(e['linears'],wanted):
        param_hash=next(p['payloadSHA256'] for p in s['originalParameters'] if p['name']==param)
        if (x['name'],x['inputShape'],x['weightShape'],x['dtype'],x['device'],x['nominalMACs'])!=(name,ishape,wshape,'torch.bfloat16','cuda',macs) or x['weightPayloadSHA256']!=param_hash or x['inputPayloadSHA256']!=c['fields'][inp]['payloadSHA256'] or x['outputPayloadSHA256']!=c['fields'][out]['payloadSHA256']:
            raise ValueError('Original full geometry/payload handoff differs')
    views=e['trace']['actualNativeViews']
    for name,offset in [('queryRaw',0),('keyRaw',3072),('valueRaw',6144),('gateRaw',9216),('upRaw',18432)]:
        if views[name]['storageOffset']!=offset or views[name]['stride']!=[42467328,27648,1]:raise ValueError('Original fused native view differs')
    if v['actualFusedJoin']['payloadSHA256']!=c['fields']['fused']['payloadSHA256'] or v['actualOutputJoin']['payloadSHA256']!=c['fields']['outputJoin']['payloadSHA256'] or v['actualExitJoinPayloadSHA256']!=c['fields']['exit']['payloadSHA256']:
        raise ValueError('Actual complete native join differs')
    if len(e['momentProducts'])!=5 or [x['name'] for x in e['momentProducts']]!=['imageCovariance','queryCovariance','queryMean','textScores','textAV'] or sum(x['nominalMACs'] for x in e['momentProducts'])!=2216165376 or any(x['dtype']!='torch.float32' or x['device']!='cuda' for x in e['momentProducts']):raise ValueError('Original moment work differs')
    if len(e['SDPA'])!=1 or e['SDPA'][0]['QShape']!=[1,24,1024,128] or e['SDPA'][0]['KShape']!=[1,24,1536,128] or e['SDPA'][0]['nominalQKAndAVMACs']!=9663676416:raise ValueError('Original image attention differs')
    costs=work();macs=sum(x['nominalMACs'] for x in e['linears']+e['momentProducts'])+sum(x['nominalQKAndAVMACs'] for x in e['SDPA'])
    if costs!=report['work'] or costs!=e['work'] or macs!=costs['candidateMACs']:raise ValueError('Actual matrix work differs')
    streams=read(p/'provenance.json')['losslessStreams'];raw=gzip.decompress((p/streams[0]['published']).read_bytes())
    if len(streams)!=1 or hashlib.sha256(raw).hexdigest()!=streams[0]['sourceSHA256']:raise ValueError('Lossless row stream differs')
    rows=[json.loads(line) for line in raw.splitlines()]
    if len(rows)!=1536 or [x['row'] for x in rows]!=list(range(1536)):raise ValueError('Complete row census differs')
    for name,lo,hi in [('all',0,1536),('text',0,512),('image',512,1536)]:
        if summarize(np.array([[x['statistics'][k] for k in ROW_COLUMNS] for x in rows[lo:hi]]))!=report['panels'][name]:raise ValueError('Six-corner scalar aggregate differs')
    return {'schema':'tce-dense-corner-mechanics-v1','typedActs':1,'typedInputs':9,'producerABIs':27,'analysisABIs':5,'BF16Projections':2,
        'FP64CandidateProducts':0,'FP32MomentProducts':5,'imageSDPA':1,'QRCalls':0,'diagnosticProducts':0,'dependentRows':1536,
        'candidateMACs':costs['candidateMACs'],'diagnosticMACs':0,'assayMACs':costs['assayMACs'],'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),
        'scope':'Source/byte/originalfullgeometry/nativefieldhistory/joins/typed/work/allrow six-corner scalar mechanics. No independent real-array replay or originalfullattention/globalcausal/cheapcorrection/image/zero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('root');p.add_argument('--destination');a=p.parse_args()
    print(json.dumps(publish(a.root,a.destination) if a.action=='publish' else globals()[a.action](a.root)))
