"""Full byte/view custody and lossless first-image-consumer publication."""
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
from publish_source_context import REMOTE


def items(p):
    s=read(p/'inputs.json');a=read(p/'primary.json');groups=[a['embeddings'],a['constants'],*a['shared'].values(),*a['cells'].values(),*read(p/'response-fields.json').values(),*read(p/'factor-fields.json').values()]
    return [s['modulation'],*s['donors'].values(),*[x['z'] for x in s['receivers'].values()],*s['parentRecords'].values(),*[{'path':str(Path(s['modelRoot'])/n),'sha256':h} for n,h in s['imageAssets'].items()],*[x for g in groups for x in g.values()]]


VIEW_REMOTE=REMOTE.split('print(json.dumps')[0]+'''import ast,struct
views={}
for v in s.get('views',[]):
 p=Path(v['path'])
 if str(p) not in inventory:raise ValueError('Unverified view parent')
 with p.open('rb') as f:
  if f.read(8)!=b'\\x93NUMPY\\x01\\x00':raise ValueError('Unexpected NPY view version')
  n=struct.unpack('<H',f.read(2))[0];header=ast.literal_eval(f.read(n).decode('latin1').strip());offset=f.tell()
  if header!={'descr':'<u2','fortran_order':False,'shape':(1,1024,18432)}:raise ValueError('Unexpected view geometry')
  h=hashlib.sha256();lo,hi=v['slice'];count=0
  for row in range(1024):
   f.seek(offset+2*(row*18432+lo));b=f.read(2*(hi-lo))
   if len(b)!=2*(hi-lo):raise ValueError('Short view row')
   h.update(b);count+=len(b)
 if h.hexdigest()!=v['payloadSHA256']:raise ValueError('Semantic view bytes differ')
 views[v['id']]={'payloadSHA256':h.hexdigest(),'bytes':count,'parentPath':str(p)}
print(json.dumps({'files':inventory,'totalBytesVerified':sum(v['bytes'] for v in inventory.values()),'allowedRoots':s['allowedRoots'],'views':views,'totalViewBytesVerified':sum(v['bytes'] for v in views.values())}))
'''


def verify(root):
    root,c,j,r=checked(root);p=root/'result';s=read(p/'inputs.json');a=read(p/'primary.json');base=j['result']['diagnostics']['cwd'];seal=sha(p/'primary-seal.json')
    if read(p/'report.json')['primarySealSHA256']!=seal or read(p/'primary-seal.json')['primarySHA256']!=sha(p/'primary.json'):raise ValueError('Primary release differs')
    values=items(p)+[{'path':base+'/response-result/'+x.name,'sha256':sha(x)} for x in p.iterdir()];unique={}
    for x in values:
        if x['path'] in unique and unique[x['path']]['sha256']!=x['sha256']:raise ValueError('Conflicting byte identity')
        unique[x['path']]=x
    views=[{'id':cell+'.'+name,'path':a['cells'][cell]['fused']['path'],'slice':x['lastAxisSlice'],'payloadSHA256':x['payloadSHA256']} for cell,vs in a['views'].items() for name,x in vs.items()]
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(VIEW_REMOTE)],input=json.dumps({'allowedRoots':[base,*s['protectedRoots'],s['modulation']['path']],'items':list(unique.values()),'views':views}),text=True,capture_output=True,timeout=180)
    if proc.returncode:raise RuntimeError(proc.stderr)
    v=json.loads(proc.stdout);v.update(schema='tce-first-consumer-bytes-v1',jobID=j['job_id'],primarySealSHA256=seal,verifierSourceSHA256=sha(__file__),scope='Full supplied/native/response/factor files and original transformer assets; eight semantic BF16 gate/up views independently row-byte-hashed. SSH byte/metadata only, no numerical array/model computation; tensors remain near compute.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified'],'views':len(v['views']),'viewBytes':v['totalViewBytesVerified']}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['primarySealSHA256']!=sha(p/'primary-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Byte verification differs')
    out=Path(destination);out.mkdir(exist_ok=False,parents=True);streams=[]
    for x in p.iterdir():
        if x.suffix=='.jsonl':
            raw=x.read_bytes();to=out/(x.name+'.gz');to.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(to.read_bytes())!=raw:raise ValueError('Lossless stream differs')
            streams.append({'source':x.name,'sourceSHA256':sha(x),'sourceBytes':len(raw),'published':to.name,'publishedSHA256':sha(to),'publishedBytes':to.stat().st_size})
        else:shutil.copyfile(x,out/x.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json');save(out/'provenance.json',{'schema':'tce-first-consumer-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreams':streams,
        'scope':'Source-order local image consumer and saved-operand finite-change diagnostics. Four original matrices/norms/MLPs paid; no whole-future mediation, cheap complete velocity or arbitrary-prompt zero-neural claim.'});save(out/'files.json',{x.name:sha(x) for x in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    from first_consumer import SELECTED,STAGES
    from source_response import mask
    p=Path(source);r=read(p/'request.json');s=read(p/'inputs.json');a=read(p/'primary.json');seal=read(p/'primary-seal.json');e=read(p/'execution.json');report=read(p/'report.json');v=read(p/'retained-array-verification.json');prov=read(p/'provenance.json')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Publication member differs')
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request identity differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Executed worker differs')
    if report['primarySealSHA256']!=sha(p/'primary-seal.json') or v['primarySealSHA256']!=sha(p/'primary-seal.json') or seal['primarySHA256']!=sha(p/'primary.json') or seal['requestSHA256']!=r['requestSHA256']:raise ValueError('Primary seal chain differs')
    for x in items(p):
        if v['files'][x['path']]['sha256']!=x['sha256']:raise ValueError('Full byte custody absent')
    if e['loadedParameters']!=s['selectedParameters'] or {x['name']:x['shape'] for x in e['loadedParameters']}!=SELECTED or sum(x['bytes'] for x in e['loadedParameters'])!=189530112:raise ValueError('Four-parameter closure differs')
    if e['parameterInputVersionPayloadClosure'] is not True or e['consumerRNGClosure'] is not True or e['numericalProfile']!=s['numericalProfile'] or len(e['fileQuarantine']['deniedCanaries'])!=3:raise ValueError('Qualified model/input/quarantine closure absent')
    if len(e['linears'])!=12 or report['nominalAffineMACs']!=368025010176 or sum(x['nominalMACs'] for x in e['linears'])!=report['nominalAffineMACs']:raise ValueError('Affine work ledger differs')
    if [sum(x['role']==role for x in e['linears']) for role in ['input-projection','carrier-projection','local-image-path']]!=[2,2,8]:raise ValueError('Shared/cell physical work differs')
    for x in e['linears']:
        if x['parameter'] not in SELECTED:raise ValueError('Unadmitted parameter')
        if x['role']=='carrier-projection' and (x['inputStride']!=s['carrierGeometry']['stride'] or x['inputStorageOffset']!=s['carrierGeometry']['storageOffset']):raise ValueError('Carrier projection layout differs')
        if x['role']=='input-projection' and x['inputStride']!=s['receivers'][x['receiver']]['z']['nativeStride']:raise ValueError('Receiver input layout differs')
    expected=['R0A','R0B','R1A','R1B'];names={'residualCommit','normalization','normScaled','mlpInput','fused','activation','product','mlpWrite','gatedMLP','imageExit'}
    if sorted(a['cells'])!=expected or [x['cell'] for x in a['frames']]!=expected:raise ValueError('Four-cell native coverage differs')
    for cell in expected:
        if set(a['cells'][cell])!=names:raise ValueError('Complete native stage capture differs')
        for name,x in a['cells'][cell].items():
            width=18432 if name=='fused' else 9216 if name in ['activation','product'] else 3072
            if x['shape']!=[1,1024,width] or x['logicalDtype']!='torch.bfloat16':raise ValueError('Native stage geometry differs')
        for name,x in a['views'][cell].items():
            if v['views'][cell+'.'+name]['payloadSHA256']!=x['payloadSHA256'] or x['lastAxisSlice']!=([0,9216] if name=='gateOperand' else [9216,18432]):raise ValueError('Semantic gate/up byte view differs')
    for f in a['frames']:
        rr=f['cell'][:2];h=f['cell'][-1];src=f['sourceFrame'];dst=f['outputFrame'];ev=f['evidence']
        if ev['act']!='tce.first_image_consumer:v1' or src['frame_id']!=ev['input_frame_id'] or dst['frame_id']!=ev['output_frame_id'] or dst['parent_frame_id']!=src['frame_id'] or dst['program_fingerprint']!=r['requestSHA256']:raise ValueError('Typed local lineage differs')
        expected_fp={'embedding':a['embeddings'][rr]['payloadSHA256'],'gatedCarrier':a['shared'][h]['gatedCarrier']['payloadSHA256'],'modulation':s['modulation']['payloadSHA256']}
        if {k:x['fingerprint'] for k,x in src['slots'].items()}!=expected_fp or ev['metrics']['capturedImageExitPayloadSHA256']!=a['cells'][f['cell']]['imageExit']['payloadSHA256']:raise ValueError('Typed consumed/output content differs')
    if any(a['embeddings'][rr]['payloadSHA256']!=s['expectedPrefixes'][rr]['hidden_states'] for rr in ['R0','R1']):raise ValueError('Archived input-prefix reconstruction differs')
    streams={}
    for name in ['queries','factors']:
        raw=gzip.decompress((p/(name+'.jsonl.gz')).read_bytes());receipt=next(x for x in prov['losslessStreams'] if x['source']==name+'.jsonl')
        if hashlib.sha256(raw).hexdigest()!=receipt['sourceSHA256']:raise ValueError('Lossless stream identity differs')
        streams[name]=[json.loads(x) for x in raw.splitlines()]
    for stage in STAGES:
        rows=[x for x in streams['queries'] if x['stage']==stage]
        if [x['imageQueryIndex'] for x in rows]!=list(range(1024)) or [x['sourceNonlocal'] for x in rows]!=mask('bottom-right').tolist():raise ValueError('Actual stage/query coverage differs')
    for rr in ['R0','R1']:
        for kind in ['normalization','product']:
            if [x['imageQueryIndex'] for x in streams['factors'] if x['receiver']==rr and x['kind']==kind]!=list(range(1024)):raise ValueError('Factor/query coverage differs')
    if len(streams['queries'])!=report['queryRecords'] or report['queryRecords']!=13312 or len(streams['factors'])!=report['factorRecords'] or report['factorRecords']!=4096:raise ValueError('Complete row counts differ')
    responses=read(p/'response-fields.json');factors=read(p/'factor-fields.json')
    if set(responses)!=set(STAGES) or len(factors)!=4 or len(report['stageAccounts'])!=26 or len(report['factorAccounts'])!=8:raise ValueError('Response/factor account coverage differs')
    return {'schema':'tce-first-consumer-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'nativeCells':4,'typedActs':4,'selectedMatrices':4,'affineCalls':12,'stages':13,'queryRecords':13312,'factorRecords':4096,'independentlyByteVerifiedViews':8,'primarySealSHA256':sha(p/'primary-seal.json'),
        'scope':'Frozen source/request/seal/full-byte/semantic-view/layout/shared-work/native-stage/typed input-output/prefix/completequery mechanics. No factor magnitude gate, full-block-exit parity, whole-future mediation or generator certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args();print(json.dumps(publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source),indent=2))
