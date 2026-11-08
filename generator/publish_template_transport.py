"""Complete byte joins and scalar/operator audit; no independent numerical replay."""
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
from template_transport import abis,work,PARAMETERS,SOURCE_NAMES,TEMPLATE_NAMES
from template_transport_analysis import analysis_abis,ROW_COLUMNS,summarize
from template_transport_intake import validate_metadata

BYTES=INTAKE_BYTES.split('print(json.dumps')[0]+r'''
def raw_npy(item):
 with Path(item['path']).open('rb') as f:
  magic=f.read(8);n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0];f.read(n);return f.read()
def joined(candidate,names,widths):
 raw=[raw_npy(candidate[name]) for name in names];h=hashlib.sha256();size=0
 for value,width in zip(raw,widths):
  if len(value)!=1536*width*2:raise ValueError('Actual join geometry differs')
 for row in range(1536):
  for value,width in zip(raw,widths):
   piece=value[row*width*2:(row+1)*width*2];h.update(piece);size+=len(piece)
 return {'payloadSHA256':h.hexdigest(),'bytes':size}
joins={}
for arm,candidate in s['candidate'].items():
 qkv=joined(candidate,['queryRaw','keyRaw','valueRaw'],[3072,3072,3072])
 j=joined(candidate,['attentionRead','gu_product'],[3072,9216])
 h=hashlib.sha256(raw_npy(candidate['textOut'])+raw_npy(candidate['imageOut'])).hexdigest()
 if qkv['payloadSHA256']!=s['qkvProjectionPayloads'][arm] or j['payloadSHA256']!=candidate['outputJoin']['payloadSHA256'] or h!=candidate['exit']['payloadSHA256']:
  raise ValueError('Actual two-input QKV/J/exit joins differ')
 joins[arm]={'QKV':qkv,'J':j,'exitPayloadSHA256':h}
with Path(s['checkpoint']).open('rb') as f:
 entry=head[s['parameters'][0]['name']];a,b=entry['data_offsets'];f.seek(base+a);raw=f.read(9216*3072*2)
physical={'payloadSHA256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
if any(v['payloadSHA256']!=physical['payloadSHA256'] for v in s['physicalQKV'].values()):raise ValueError('Physical original QKV slice differs')
raw={n:raw_npy(v) for n,v in s['templates'].items()};template_handoffs=[]
for stream,(lo,hi) in enumerate([(0,512),(512,1536)]):
 basis=raw['gu_basis'][stream*3072*128*8:(stream+1)*3072*128*8]
 compiled=raw['gu_projectedGU'][stream*128*18432*8:(stream+1)*128*18432*8]
 row=raw['rowBasis'][lo*128*8:hi*128*8];transposed=hashlib.sha256()
 for col in range(128):
  for i in range(hi-lo):transposed.update(row[(i*128+col)*8:(i*128+col+1)*8])
 template_handoffs.append({'guBasis':hashlib.sha256(basis).hexdigest(),'guMap':hashlib.sha256(compiled).hexdigest(),
  'rowBasis':hashlib.sha256(row).hexdigest(),'rowBasisTranspose':transposed.hexdigest()})
for arm in ['parent','own']:
 products={x['name']:x for x in s['products'] if x['arm']==arm}
 for i,t in enumerate(template_handoffs):
  if products['guCoordinates'+str(i)]['rightPayloadSHA256']!=t['guBasis'] or products['guReconstruct'+str(i)]['rightPayloadSHA256']!=t['guMap'] or products['projectInputs'+str(i)]['leftPayloadSHA256']!=t['rowBasisTranspose'] or products['reconstructUpdate'+str(i)]['leftPayloadSHA256']!=t['rowBasis']:
   raise ValueError('Actual template operand handoff differs')
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,
 'originalParameters':parameters,'physicalQKV':physical,'actualJoins':joins,'templateHandoffs':template_handoffs}))
'''


def items(p):
    s=read(p/'inputs.json')
    return [*s['sourceFields'].values(),*s['templateFields'].values(),*list(fields(s['comparisonFields'])),
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
    spec={'allowedRoots':[base,*s['sourceRoots']],'items':list(unique.values()),
          'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),
          'parameters':e['loadedParameters'],'candidate':read(p/'candidate.json')['fields'],
          'qkvProjectionPayloads':{x['arm']:x['outputPayloadSHA256'] for x in e['linears']},
          'physicalQKV':e['physicalQKV'],'templates':s['templateFields'],'products':e['products']}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],
                        input=json.dumps(spec),text=True,capture_output=True,timeout=360)
    if proc.returncode:raise RuntimeError(proc.stderr)
    value=json.loads(proc.stdout);value.update(schema='tce-template-transport-bytes-v1',jobID=job['job_id'],
        verifierSourceSHA256=sha(__file__),byteTemplateSourceSHA256=sha(Path(__file__).with_name('own_successor_intake.py')),
        candidateSealSHA256=sha(p/'candidate-seal.json'),
        scope='Complete source/template/comparison/candidate/analysis/metadata bytes,four original tensors,physical QKV and actual QKV/J/exit/template joins. SSH bytes/header/permutation only,no numerical tensor replay.')
    save(root/'retained-array-verification.json',value)
    return {'files':len(value['files']),'NPY':len(value['npyPayloadWitnesses']),'bytes':value['totalBytesVerified'],'parameterBytes':sum(x['bytes'] for x in e['loadedParameters'])}


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
    save(out/'provenance.json',{'schema':'tce-template-transport-publication-v1','jobID':job['job_id'],'requestSHA256':r['requestSHA256'],
        'payloadCustody':col['custody'],'completeCollectedInventory':col['inventory'],'losslessStreams':streams,
        'scope':'Sameexplicitfinite-template,parent+own. Parentassociationmeasurednotassumed;alllearnedmaps/templates/source/clocks/historypaid. No image/originalfullattention/globalbound/promptgeneralization/zero-neural claim.'})
    save(out/'files.json',{x.name:sha(x) for x in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    import numpy as np
    p=Path(source)
    r,s,c,e,a,ae,v,report=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json',
        'analysis-fields.json','analysis-execution.json','retained-array-verification.json','report.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request identity differs')
    for name,digest in read(p/'files.json').items():
        if sha(p/name)!=digest:raise ValueError('Published member differs:'+name)
    for name in r['ownedSources']+['inputs.json','resource-basis.json','frozen-plan.md','implementation-notes.md','intake.json']:
        target=Path(__file__).with_name(name) if name in r['ownedSources'] else p/name
        if sha(target)!=r['sourceHashes'][name]:raise ValueError('Frozen source/metadata differs:'+name)
    if v['verifierSourceSHA256']!=sha(__file__) or v['byteTemplateSourceSHA256']!=sha(Path(__file__).with_name('own_successor_intake.py')):
        raise ValueError('Byte verifier differs')
    intake=read(p/'intake.json');validate_metadata(intake)
    if sha(p/'intake.json')!=s['intakeSHA256'] or any(s[n]!=intake[n] for n in ['sourceFields','comparisonFields','templateFields','originalParameters','parents','metadataFiles']):
        raise ValueError('Complete eight-corner ancestry differs')
    if intake['verifierSHA256']!=sha(Path(__file__).with_name('template_transport_intake.py')):raise ValueError('Intake source differs')
    for item in items(p):
        if v['files'][item['path']]['sha256']!=item['sha256'] or v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:
            raise ValueError('Complete NPY witness absent')
    for name,digest in s['metadataFiles'].items():
        if v['files'][name]['sha256']!=digest:raise ValueError('Metadata witness absent')
    sealed=sha(p/'candidate-seal.json');exited=read(p/'producer-exit.json')
    if read(p/'candidate-seal.json')!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json')}:
        raise ValueError('Complete two-arm seal differs')
    if any(x.get('candidateSealSHA256')!=sealed for x in [a,ae,v,report,exited]) or ae['producerExitSHA256']!=sha(p/'producer-exit.json') or exited['requestSHA256']!=r['requestSHA256'] or exited['returncode']!=0 or exited['producerPID']==ae['analyzerPID']:
        raise ValueError('Observed exit/release differs')
    if e['loadedParameters']!=s['originalParameters'] or not e['sourceParameterTemplateRNGClosure'] or e['numericalProfile']!=s['numericalProfile']:
        raise ValueError('Original map/template/profile closure differs')
    allowed={x['path'] for x in [*s['sourceFields'].values(),*s['templateFields'].values(),*s['comparisonFields']['parentIncoming'].values()]}|{str(Path(s['modelRoot'])/n) for n in s['imageAssets']}
    guard=e['fileQuarantine']
    if set(guard['allowedFiles'])!=allowed or set(guard['deniedCanaries'])!=set(s['deniedNumericalInputs']) or not set(guard['successfulFileOpenCounts']).issubset(allowed):
        raise ValueError('Source/template allowlist differs')
    comparisons={x['path'] for x in fields(s['comparisonFields'])}-allowed
    if not comparisons.issubset(s['deniedNumericalInputs']):raise ValueError('Comparison-only quarantine differs')
    manifest=identity({'originalParameters':s['originalParameters'],'intakeSHA256':s['intakeSHA256'],'nativeSources':s['nativeSources'],'templateFields':s['templateFields']})
    if e['parameterManifestSHA256']!=manifest or len(e['typedActs'])!=2:raise ValueError('Typed template manifest differs')
    for arm,record in zip(['parent','own'],e['typedActs']):
        current={**s['sourceFields'],**(s['comparisonFields']['parentIncoming'] if arm=='parent' else {})}
        expected={**{n:x['payloadSHA256'] for n,x in current.items()},**{n:x['payloadSHA256'] for n,x in s['templateFields'].items()},
                  **{'parameter_'+x['name']:x['payloadSHA256'] for x in s['originalParameters']}}
        if record['stage']!=arm or {n:x['fingerprint'] for n,x in record['sourceFrame']['slots'].items()}!=expected:
            raise ValueError('Typed per-arm source/template binding differs')
        check_act(record,c['fields'][arm],r['requestSHA256'],manifest)
        if {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in c['fields'][arm].items()}!=abis() or e['retainedFingerprints'][arm]!={n:x['payloadSHA256'] for n,x in c['fields'][arm].items()}:
            raise ValueError('Complete per-arm ABI/fingerprint differs')
        if v['actualJoins'][arm]['J']['payloadSHA256']!=c['fields'][arm]['outputJoin']['payloadSHA256'] or v['actualJoins'][arm]['exitPayloadSHA256']!=c['fields'][arm]['exit']['payloadSHA256']:
            raise ValueError('Actual join witness differs')
    if {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in a['fields'].items()}!=analysis_abis():
        raise ValueError('Complete analysis ABI differs')
    if e['QR'] or ae['products'] or ae['diagnosticMatrixMACs']!=0 or ae['device']!='cpu' or ae['dtype']!='float64':
        raise ValueError('Undeclared QR/analysis products')
    if [len(e[k]) for k in ['products','linears','momentProducts','SDPA']]!=[20,2,10,2]:raise ValueError('Operator census differs')
    for arm in ['parent','own']:
        products={x['name']:x for x in e['products'] if x['arm']==arm}
        for stream,rows in enumerate([512,1024]):
            desired=[('guCoordinates',[rows,3072],[3072,128]),('guReconstruct',[rows,128],[128,18432]),
                     ('projectInputs',[128,rows],[rows,12288]),('compressedOut',[128,12288],[12288,3072]),
                     ('reconstructUpdate',[rows,128],[128,3072])]
            for name,left,right in desired:
                x=products[name+str(stream)]
                if (x['leftShape'],x['rightShape'],x['dtype'],x['device'])!=(left,right,'torch.float64','cuda') or x['nominalMACs']!=left[0]*left[1]*right[1]:
                    raise ValueError('Fixed-template FP64 geometry/work differs')
                for key,shape in [('leftGeometry',left),('rightGeometry',right)]:
                    if x[key]['shape']!=shape or x[key]['dtype']!='torch.float64':raise ValueError('Actual operand geometry absent')
                if name=='guCoordinates' and x['rightGeometry']['stride']!=[128,1]:raise ValueError('Reloaded GU basis policy differs')
                if name=='projectInputs' and x['leftGeometry']['stride']!=[1,128]:raise ValueError('Reloaded row basis policy differs')
            t=v['templateHandoffs'][stream]
            if products['guCoordinates'+str(stream)]['rightPayloadSHA256']!=t['guBasis'] or products['guReconstruct'+str(stream)]['rightPayloadSHA256']!=t['guMap'] or products['projectInputs'+str(stream)]['leftPayloadSHA256']!=t['rowBasisTranspose'] or products['reconstructUpdate'+str(stream)]['leftPayloadSHA256']!=t['rowBasis']:
                raise ValueError('Template byte/operator handoff differs')
        linear=next(x for x in e['linears'] if x['arm']==arm)
        if (linear['inputShape'],linear['weightShape'],linear['dtype'],linear['device'],linear['nominalMACs'])!=([1,1536,3072],[9216,3072],'torch.bfloat16','cuda',43486543872) or linear['inputPayloadSHA256']!=c['fields'][arm]['normalized']['payloadSHA256'] or linear['outputPayloadSHA256']!=v['actualJoins'][arm]['QKV']['payloadSHA256'] or linear['weightPayloadSHA256']!=v['physicalQKV']['payloadSHA256']:
            raise ValueError('Physical original QKV handoff differs')
        moments=[x for x in e['momentProducts'] if x['arm']==arm]
        if [x['name'] for x in moments]!=['imageCovariance','queryCovariance','queryMean','textScores','textAV'] or sum(x['nominalMACs'] for x in moments)!=2216165376 or any(x['dtype']!='torch.float32' or x['device']!='cuda' for x in moments):
            raise ValueError('Per-arm moment work differs')
    if any(x['QShape']!=[1,24,1024,128] or x['KShape']!=[1,24,1536,128] or x['nominalQKAndAVMACs']!=9663676416 for x in e['SDPA']):
        raise ValueError('Image attention work differs')
    costs=work();macs=sum(x['nominalMACs'] for x in e['products']+e['linears']+e['momentProducts'])+sum(x['nominalQKAndAVMACs'] for x in e['SDPA'])
    if costs!=report['work'] or costs!=e['work'] or macs!=costs['candidateMACs']:raise ValueError('Actual matrix ledger differs')
    streams=read(p/'provenance.json')['losslessStreams'];raw=gzip.decompress((p/streams[0]['published']).read_bytes())
    if len(streams)!=1 or hashlib.sha256(raw).hexdigest()!=streams[0]['sourceSHA256']:raise ValueError('Lossless scalar stream differs')
    rows=[json.loads(line) for line in raw.splitlines()]
    if len(rows)!=1536 or [x['row'] for x in rows]!=list(range(1536)):raise ValueError('Complete row census differs')
    for name,lo,hi in [('all',0,1536),('text',0,512),('image',512,1536)]:
        if summarize(np.array([[x['statistics'][k] for k in ROW_COLUMNS] for x in rows[lo:hi]]))!=report['panels'][name]:raise ValueError('Eight-corner scalar aggregate differs')
    return {'schema':'tce-template-transport-mechanics-v1','typedActs':2,'typedInputsPerAct':12,'producerABIsPerArm':33,
        'analysisABIs':8,'FP64Products':20,'BF16QKVProjections':2,'FP32MomentProducts':10,'imageSDPA':2,'QRCalls':0,
        'dependentRows':1536,'scalarColumns':len(ROW_COLUMNS),'candidateMACs':macs,'diagnosticMACs':0,
        'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),
        'scope':'Recorded sources/bytes/templates/joins/typed/actual finite geometry/work/all scalar rows. No independent real-array replay,old CUDA layout attestation,originalfullattention/globalbound/prompt/image/zero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('root');p.add_argument('--destination');a=p.parse_args()
    print(json.dumps(publish(a.root,a.destination) if a.action=='publish' else globals()[a.action](a.root)))
