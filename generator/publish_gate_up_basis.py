"""Independent retained-byte and source/typed/operation/row publication audit."""
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
from gate_up_basis import PARAMETERS,SOURCE_NAMES,COMPARISON_NAMES,abis,work
from gate_up_basis_analysis import PORTS,COLUMNS,analysis_abis,panel

BYTES=NPY_BYTES.split('print(json.dumps')[0]+r'''
parameters={};raws={}
with Path(s['checkpoint']).open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n
 for item in s['loaded']:
  entry=header[item['name']];a,b=entry['data_offsets'];f.seek(base+a);raw=f.read(b-a)
  if entry['dtype']!='BF16' or entry['shape']!=item['shape'] or [a,b]!=item['dataOffsets'] or len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['payloadSHA256']:raise ValueError('Original parameter bytes differ')
  raws[item['name']]=raw;parameters[item['name']]={'payloadSHA256':item['payloadSHA256'],'bytes':len(raw),'shape':entry['shape']}
raw=raws[s['loaded'][0]['name']][9216*3072*2:27648*3072*2]
physical={'payloadSHA256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'shape':[18432,3072]}
if any(physical[k]!=s['physical'][k] for k in physical):raise ValueError('Actual original GU view differs')
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,'originalParameterWitnesses':parameters,'physicalGUWitness':physical}))
'''


def items(p):
    s=read(p/'inputs.json')
    return [*s['sourceFields'].values(),*s['comparisonFields'].values(),*list(fields(read(p/'candidate.json'))),*list(fields(read(p/'analysis-fields.json')))]


def verify(root):
    from mrun.client.api import Api
    root,col,job,r=checked(root);p=root/'result';s=read(p/'inputs.json');e=read(p/'execution.json');api=Api();parent=api.json('GET','/api/jobs/'+s['parentJobID'])
    if parent['state']!='succeeded' or parent['payload_custody']!=s['parentCustody']:raise ValueError('Source parent custody changed')
    base=job['result']['diagnostics']['cwd'];values=items(p)+[{'path':base+'/response-result/'+x.name,'sha256':sha(x)} for x in p.iterdir()]
    unique={}
    for item in values:
        if item['path'] in unique and item['sha256']!=unique[item['path']]['sha256']:raise ValueError('Conflicting byte identity')
        unique[item['path']]=item
    spec={'allowedRoots':[base,*s['sourceRoots']],'items':list(unique.values()),'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),
          'loaded':e['loadedParameters'],'physical':e['physicalGUView']}
    process=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=300)
    if process.returncode:raise RuntimeError(process.stderr)
    value=json.loads(process.stdout);value.update(schema='tce-current-state-gate-up-bytes-v1',jobID=job['job_id'],verifierSourceSHA256=sha(__file__),byteTemplateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
        candidateSealSHA256=sha(p/'candidate-seal.json'),scope='Complete source/target/candidate/diagnostic files and NPY payloads, two selected original tensors and physical GU rows. SSH bytes only. Original full checkpoint identity separately checked by producer; full arrays remain near compute.')
    save(root/'retained-array-verification.json',value)
    return {'files':len(value['files']),'NPY':len(value['npyPayloadWitnesses']),'bytes':value['totalBytesVerified'],'originalParameterBytes':sum(x['bytes'] for x in e['loadedParameters'])}


def publish(root,destination):
    root,col,job,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Verified release/source differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for item in p.iterdir():
        if item.suffix=='.jsonl':
            raw=item.read_bytes();target=out/(item.name+'.gz');target.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless stream differs')
            streams.append({'source':item.name,'sourceSHA256':sha(item),'published':target.name,'publishedSHA256':sha(target),'sourceBytes':len(raw),'publishedBytes':target.stat().st_size})
        else:shutil.copyfile(item,out/item.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'scheduler.json',{k:job[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-current-state-gate-up-publication-v1','jobID':job['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':col['custody'],
        'completeCollectedInventory':col['inventory'],'losslessStreams':streams,'scope':'One paid current-input rank128 completeGU/native local consumer. Original learned maps/source paid; no image or arbitrary-prompt zero-neural serving.'})
    save(out/'files.json',{x.name:sha(x) for x in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    import numpy as np
    p=Path(source);r,s,c,e,a,ae,report,v=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json','analysis-fields.json','analysis-execution.json','report.json','retained-array-verification.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request identity differs')
    for name,digest in read(p/'files.json').items():
        if sha(p/name)!=digest:raise ValueError('Published member differs:'+name)
    for name in r['ownedSources']:
        if sha(Path(__file__).with_name(name))!=r['sourceHashes'][name]:raise ValueError('Executed source changed:'+name)
    if v['verifierSourceSHA256']!=sha(__file__) or v['byteTemplateSourceSHA256']!=sha(Path(__file__).with_name('publish_sparse_text_kv.py')):raise ValueError('Byte verifier source differs')
    for item in items(p):
        if v['files'][item['path']]['sha256']!=item['sha256'] or v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Complete byte witness missing')
    if read(p/'candidate-seal.json')!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json')}:raise ValueError('Producer seal differs')
    sealed=sha(p/'candidate-seal.json');exited=read(p/'producer-exit.json')
    if any(x.get('candidateSealSHA256')!=sealed for x in [v,a,ae,report,exited]) or ae['producerExitSHA256']!=sha(p/'producer-exit.json') or exited['returncode']!=0 or ae['analyzerPID']==exited['producerPID']:raise ValueError('Post-exit release differs')
    if set(s['sourceFields'])!=set(SOURCE_NAMES) or set(s['comparisonFields'])!=set(COMPARISON_NAMES) or e['loadedParameters']!=s['originalParameters'] or [x['name'] for x in e['loadedParameters']]!=PARAMETERS:raise ValueError('Admitted field/parameter census differs')
    for item in e['loadedParameters']:
        if v['originalParameterWitnesses'][item['name']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Original parameter byte join differs')
    if any(v['physicalGUWitness'][k]!=e['physicalGUView'][k] for k in ['shape','bytes','payloadSHA256']):raise ValueError('Physical GU byte join differs')
    if not e['sourceParameterRNGClosure'] or e['numericalProfile']!=s['numericalProfile'] or ae['numericalProfile']!=s['numericalProfile']:raise ValueError('Numerical/source closure differs')
    guard=e['fileQuarantine'];allowed={x['path'] for x in s['sourceFields'].values()}|{str(Path(s['modelRoot'])/n) for n in s['imageAssets']}
    if set(guard['allowedFiles'])!=allowed or set(guard['deniedCanaries'])!=set(s['deniedNumericalInputs']) or not set(guard['successfulFileOpenCounts']).issubset(allowed) or not {x['path'] for x in s['comparisonFields'].values()}.issubset(s['deniedNumericalInputs']):raise ValueError('Five-field pre-release isolation differs')
    manifest=identity({'originalParameters':s['originalParameters'],'parentDenseSealSHA256':s['parentDenseSealSHA256'],'nativeSources':s['nativeSources']})
    if e['parameterManifestSHA256']!=manifest or len(e['typedActs'])!=1:raise ValueError('Producer manifest/Act differs')
    act=e['typedActs'][0];expected={**{n:x['payloadSHA256'] for n,x in s['sourceFields'].items()},**{'parameter_'+x['name']:x['payloadSHA256'] for x in s['originalParameters']}}
    if {n:x['fingerprint'] for n,x in act['sourceFrame']['slots'].items()}!=expected:raise ValueError('Typed source/parameter joins differ')
    check_act(act,c['fields'],r['requestSHA256'],manifest)
    if {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in c['fields'].items()}!=abis() or {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in a['fields'].items()}!=analysis_abis():raise ValueError('Complete output ABIs differ')
    products=[]
    for stream,rows in enumerate([512,1024]):
        products += [('sketch'+str(stream),[3072,rows],[rows,128]),('coordinates'+str(stream),[rows,3072],[3072,128]),
                     ('compileGU'+str(stream),[128,3072],[3072,18432]),('reconstructGU'+str(stream),[rows,128],[128,18432])]
    products += [('outFull',[1,1536,12288],[12288,3072])]
    if [(x['name'],x['leftShape'],x['rightShape']) for x in e['products']]!=products or any(x['dtype']!=('torch.bfloat16' if x['name']=='outFull' else 'torch.float64') or x['device']!='cuda' for x in e['products']):raise ValueError('Actual candidate contraction geometry differs')
    diagnostics=[]
    for stream,rows in enumerate([512,1024]):diagnostics += [('inputReconstruction'+str(stream),[rows,128],[128,3072]),('basisGram'+str(stream),[128,3072],[3072,128])]
    if [(x['name'],x['leftShape'],x['rightShape']) for x in ae['products']]!=diagnostics or any(x['dtype']!='torch.float64' or x['device']!='cuda' for x in ae['products']):raise ValueError('Post-exit contraction geometry differs')
    for event in e['products']+ae['products']:
        if event['nominalMACs']!=math_product(event['outputShape'])*event['leftShape'][-1]:raise ValueError('Actual product work differs')
    costs=work()
    if e['work']!=costs or report['work']!=costs or sum(x['nominalMACs'] for x in e['products'])!=costs['candidateMACs'] or sum(x['nominalMACs'] for x in ae['products'])!=costs['diagnosticMACs']:raise ValueError('Complete computation ledger differs')
    if len(e['QR'])!=2 or any(x['inputShape']!=[3072,128] or x['mode']!='reduced' or x['dtype']!='torch.float64' for x in e['QR']) or e['QRProfile']['requestedPreference']!='cusolver':raise ValueError('QR instrument differs')
    t=e['trace']
    for name,offset,width in [('nativeGUView',9216,18432),('nativeGateView',9216,9216),('nativeUpView',18432,9216)]:
        if t[name]['shape']!=[1,1536,width] or t[name]['stride']!=[42467328,27648,1] or t[name]['storageOffset']!=offset:raise ValueError('Source-derived native GU geometry differs')
    if t['recordedProduct']['stride']!=[18874368,12288,1] or t['recordedProduct']['storageOffset']!=3072 or t['activation']['stride']!=[14155776,9216,1]:raise ValueError('Native activation/product capture differs')
    streams=read(p/'provenance.json')['losslessStreams']
    if len(streams)!=1:raise ValueError('Full scalar row stream missing')
    stream=streams[0];raw=gzip.decompress((p/stream['published']).read_bytes())
    if hashlib.sha256(raw).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless observations differ')
    rows=[json.loads(line) for line in raw.splitlines()]
    if len(rows)!=1536 or [x['row'] for x in rows]!=list(range(1536)) or any(set(x['ports'])!=set(PORTS) or x['stratum']!=('text' if x['row']<512 else 'image') for x in rows):raise ValueError('Dependent row coverage differs')
    for name,lo,hi in [('all',0,1536),('text',0,512),('image',512,1536)]:
        for port in PORTS:
            values=np.array([[x['ports'][port][col] for col in COLUMNS] for x in rows[lo:hi]])
            if panel(values)!=report['panels'][name][port]:raise ValueError('Published scalar aggregation differs')
        values=np.array([[x['inputReconstruction'][col] for col in COLUMNS] for x in rows[lo:hi]])
        if panel(values)!=report['inputReconstruction'][name]:raise ValueError('Input scalar aggregation differs')
    return {'schema':'tce-current-state-gate-up-mechanics-v1','typedActs':1,'producerABIs':18,'analysisABIs':4,'candidateProducts':9,'QRCalls':2,'diagnosticProducts':4,
        'candidateMACs':costs['candidateMACs'],'diagnosticMACs':costs['diagnosticMACs'],'assayMACs':costs['assayMACs'],'dependentRows':1536,'portsPerRow':7,
        'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'scope':'Source/byte/physicalmap/typed/geometry/work/row mechanics only; no quality, numerical-rank, speedup, trajectory, generalization or zero-neural certification.'}


def math_product(shape):
    import math
    return math.prod(shape)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['verify','publish','audit']);parser.add_argument('root');parser.add_argument('--destination');args=parser.parse_args()
    print(json.dumps(publish(args.root,args.destination) if args.action=='publish' else globals()[args.action](args.root)))
