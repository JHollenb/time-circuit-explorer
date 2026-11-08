"""Complete byte/actual join witnesses and primary-secondary publication checks."""
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
from row_output import SOURCE_NAMES,PARAMETER,abis,work
from row_output_analysis import PORTS,ROW_COLUMNS,COLUMNS,panel,analysis_abis,summarize

BYTES=NPY_BYTES.split('print(json.dumps')[0]+r'''
item=s['loaded'][0]
with Path(s['checkpoint']).open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n;entry=header[item['name']];a,b=entry['data_offsets'];f.seek(base+a);raw=f.read(b-a)
 if entry['dtype']!='BF16' or entry['shape']!=item['shape'] or [a,b]!=item['dataOffsets'] or len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['payloadSHA256']:raise ValueError('Original output parameter bytes differ')
parameter={'payloadSHA256':item['payloadSHA256'],'bytes':len(raw),'shape':entry['shape']}
def raw_npy(item):
 with Path(item['path']).open('rb') as f:
  magic=f.read(8);n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0];f.read(n);return f.read()
attention=raw_npy(s['attention']);product=raw_npy(s['product']);h=hashlib.sha256();size=0
if len(attention)!=1536*3072*2 or len(product)!=1536*9216*2:raise ValueError('Join sources differ')
for row in range(1536):
 for raw,width in [(attention,3072*2),(product,9216*2)]:
  piece=raw[row*width:(row+1)*width];h.update(piece);size+=len(piece)
joined={'payloadSHA256':h.hexdigest(),'bytes':size,'shape':[1,1536,12288]}
if any(joined[k]!=s['actualJoin'][k] for k in joined):raise ValueError('Actual pre-promotion BF16 join differs')
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,'originalParameterWitness':parameter,'actualBF16JoinWitness':joined}))
'''


def items(p):
    s=read(p/'inputs.json')
    return [*s['sourceFields'].values(),*list(fields(s['comparisonFields'])),*list(fields(read(p/'candidate.json'))),*list(fields(read(p/'analysis-fields.json')))]


def verify(root):
    from mrun.client.api import Api
    root,col,job,r=checked(root);p=root/'result';s=read(p/'inputs.json');e=read(p/'execution.json');parent=Api().json('GET','/api/jobs/'+s['parentJobID'])
    if parent['state']!='succeeded' or parent['payload_custody']!=s['parentCustody']:raise ValueError('Parent custody differs')
    base=job['result']['diagnostics']['cwd'];values=items(p)+[{'path':base+'/response-result/'+item.name,'sha256':sha(item)} for item in p.iterdir()]
    values += [{'path':s['parentRoot']+'/response-result/'+name,'sha256':digest} for name,digest in s['parentRecords'].items()]
    unique={}
    for item in values:
        if item['path'] in unique and item['sha256']!=unique[item['path']]['sha256']:raise ValueError('Conflicting byte identity')
        unique[item['path']]=item
    spec={'allowedRoots':[base,*s['sourceRoots']],'items':list(unique.values()),'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),
        'loaded':e['loadedParameters'],'attention':s['sourceFields']['attentionRead'],'product':s['sourceFields']['product'],'actualJoin':e['actualBF16Join']}
    process=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=300)
    if process.returncode:raise RuntimeError(process.stderr)
    value=json.loads(process.stdout);value.update(schema='tce-current-row-output-bytes-v1',jobID=job['job_id'],verifierSourceSHA256=sha(__file__),byteTemplateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
        candidateSealSHA256=sha(p/'candidate-seal.json'),scope='Complete source/82/73targets/newoutputs/diagnostics/parentmetadata NPY/file bytes, original complete out tensor and actual BF16 read/product concatenation. SSH bytes only, no FP64 promotion or tensor numerical replay.')
    save(root/'retained-array-verification.json',value);return {'files':len(value['files']),'NPY':len(value['npyPayloadWitnesses']),'bytes':value['totalBytesVerified'],'originalParameterBytes':e['loadedParameters'][0]['bytes'],'joinBytes':value['actualBF16JoinWitness']['bytes']}


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
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json');save(out/'scheduler.json',{key:job[key] for key in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-current-row-output-publication-v1','jobID':job['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':col['custody'],'completeCollectedInventory':col['inventory'],'losslessStreams':streams,
        'scope':'One same-input row-space output cut with82primary73secondary/fullcenteredsignedallrows. Source82GU/currentstate/attention/originalmap remainpaid;no wholetrajectory/runtime/image/arbitrarypromptzero-neuralclaim.'})
    save(out/'files.json',{item.name:sha(item) for item in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(item.stat().st_size for item in out.iterdir())}


def audit(source):
    import math
    import numpy as np
    p=Path(source);r,s,c,e,a,ae,report,v=[read(p/name) for name in ['request.json','inputs.json','candidate.json','execution.json','analysis-fields.json','analysis-execution.json','report.json','retained-array-verification.json']]
    if identity({key:value for key,value in r.items() if key!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request identity differs')
    for name,digest in read(p/'files.json').items():
        if sha(p/name)!=digest:raise ValueError('Published member differs:'+name)
    for name in r['ownedSources']:
        if sha(Path(__file__).with_name(name))!=r['sourceHashes'][name]:raise ValueError('Executed source changed:'+name)
    if v['verifierSourceSHA256']!=sha(__file__) or v['byteTemplateSourceSHA256']!=sha(Path(__file__).with_name('publish_sparse_text_kv.py')):raise ValueError('Byte verifier source differs')
    for item in items(p):
        if v['files'][item['path']]['sha256']!=item['sha256'] or v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Complete byte witness missing')
    for name,digest in s['parentRecords'].items():
        if v['files'][s['parentRoot']+'/response-result/'+name]['sha256']!=digest:raise ValueError('Parent metadata byte join differs')
    if read(p/'candidate-seal.json')!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json')}:raise ValueError('Producer seal differs')
    sealed=sha(p/'candidate-seal.json');exited=read(p/'producer-exit.json')
    if any(item.get('candidateSealSHA256')!=sealed for item in [v,a,ae,report,exited]) or ae['producerExitSHA256']!=sha(p/'producer-exit.json') or exited['returncode']!=0 or ae['analyzerPID']==exited['producerPID']:raise ValueError('Post-exit release differs')
    if set(s['sourceFields'])!=set(SOURCE_NAMES) or set(s['comparisonFields'])!={'82','73'} or any(set(value)!={'preResidualUpdate','exit'} for value in s['comparisonFields'].values()) or e['loadedParameters']!=s['originalParameters'] or [item['name'] for item in e['loadedParameters']]!=[PARAMETER]:raise ValueError('Source/target/parameter census differs')
    if v['originalParameterWitness']['payloadSHA256']!=e['loadedParameters'][0]['payloadSHA256'] or any(v['actualBF16JoinWitness'][key]!=e['actualBF16Join'][key] for key in ['shape','bytes','payloadSHA256']):raise ValueError('Actual parameter/join witness differs')
    if not e['sourceParameterRNGClosure'] or e['numericalProfile']!=s['numericalProfile'] or ae['numericalProfile']!=s['numericalProfile']:raise ValueError('Source/numerical closure differs')
    allowed={item['path'] for item in s['sourceFields'].values()}|{str(Path(s['modelRoot'])/name) for name in s['imageAssets']};guard=e['fileQuarantine']
    if set(guard['allowedFiles'])!=allowed or set(guard['deniedCanaries'])!=set(s['deniedNumericalInputs']) or not set(guard['successfulFileOpenCounts']).issubset(allowed) or not {item['path'] for item in fields(s['comparisonFields'])}.issubset(s['deniedNumericalInputs']):raise ValueError('Source-only target isolation differs')
    manifest=identity({'originalParameters':s['originalParameters'],'parentCandidateSealSHA256':s['parentCandidateSealSHA256'],'nativeSources':s['nativeSources']})
    if e['parameterManifestSHA256']!=manifest or len(e['typedActs'])!=1:raise ValueError('Producer manifest/Act differs')
    expected={**{name:item['payloadSHA256'] for name,item in s['sourceFields'].items()},'originalOut':s['originalParameters'][0]['payloadSHA256']}
    if {name:item['fingerprint'] for name,item in e['typedActs'][0]['sourceFrame']['slots'].items()}!=expected:raise ValueError('Typed input joins differ')
    check_act(e['typedActs'][0],c['fields'],r['requestSHA256'],manifest)
    if {name:{'shape':item['shape'],'dtype':item['logicalDtype']} for name,item in c['fields'].items()}!=abis() or {name:{'shape':item['shape'],'dtype':item['logicalDtype']} for name,item in a['fields'].items()}!=analysis_abis():raise ValueError('Full retained ABI census differs')
    products=[];diagnostics=[]
    for stream,rows in enumerate([512,1024]):
        products += [('projectInputs'+str(stream),[128,rows],[rows,12288]),('compressedOut'+str(stream),[128,12288],[12288,3072]),('reconstructUpdate'+str(stream),[rows,128],[128,3072])]
        diagnostics += [('coordinateReconstruction'+str(stream),[rows,128],[128,128]),('basisGram'+str(stream),[128,rows],[rows,128])]
    for actual,wanted in [(e['products'],products),(ae['products'],diagnostics)]:
        if [(item['name'],item['leftShape'],item['rightShape']) for item in actual]!=wanted or any(item['dtype']!='torch.float64' or item['device']!='cuda' or item['nominalMACs']!=math.prod(item['outputShape'])*item['leftShape'][-1] for item in actual):raise ValueError('Actual FP64 contraction geometry/work differs')
    costs=work()
    if report['work']!=costs or e['work']!=costs or sum(item['nominalMACs'] for item in e['products'])!=costs['candidateMACs'] or sum(item['nominalMACs'] for item in ae['products'])!=costs['diagnosticMACs']:raise ValueError('Actual/composed work ledger differs')
    parent_report=read(Path(__file__).resolve().parent.parent/'docs/results/current-state-gate-up/report.json')
    if costs['inherited82GateUpFormationMACs']!=parent_report['work']['candidateMACs']-parent_report['work']['candidate']['outFull'] or sha(Path(__file__).resolve().parent.parent/'docs/results/current-state-gate-up/report.json')!=s['parentRecords']['report.json']:raise ValueError('Paid inherited82 work join differs')
    if len(e['QR'])!=2 or [item['inputShape'] for item in e['QR']]!=[[512,128],[1024,128]] or any(item['dtype']!='torch.float64' or item['mode']!='reduced' for item in e['QR']) or e['QRProfile']['requestedPreference']!='cusolver':raise ValueError('Actual QR profile differs')
    update=e['trace']['admittedUpdate']
    if update['shape']!=[1,1536,3072] or update['stride']!=[4718592,3072,1] or update['storageOffset']!=0 or update['dtype']!='torch.bfloat16':raise ValueError('Admitted native-shaped update layout differs')
    if e['trace']['dataWork']['completeFP64ToFP32Elements']!=4718592 or e['trace']['dataWork']['finalFP32ToBF16Elements']!=4718592:raise ValueError('Final double-admission element census differs')
    streams=read(p/'provenance.json')['losslessStreams']
    if len(streams)!=1:raise ValueError('Scalar stream absent')
    stream=streams[0];raw=gzip.decompress((p/stream['published']).read_bytes())
    if hashlib.sha256(raw).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless observations differ')
    rows=[json.loads(line) for line in raw.splitlines()]
    if len(rows)!=1536 or [item['row'] for item in rows]!=list(range(1536)) or any(set(item['ports'])!=set(PORTS) or item['stratum']!=('text' if item['row']<512 else 'image') for item in rows):raise ValueError('Dependent row coverage differs')
    for name,lo,hi in [('all',0,1536),('text',0,512),('image',512,1536)]:
        for port in PORTS:
            values=np.array([[item['ports'][port][col] for col in ROW_COLUMNS] for item in rows[lo:hi]])
            if summarize(values)!=report['panels'][name][port]:raise ValueError('Primary/secondary/signed scalar aggregation differs')
        values=np.array([[item['coordinateReconstruction'][col] for col in COLUMNS] for item in rows[lo:hi]])
        if panel(values)!=report['coordinateReconstruction'][name]:raise ValueError('Coordinate scalar aggregation differs')
    return {'schema':'tce-current-row-output-mechanics-v1','typedActs':1,'producerABIs':11,'analysisABIs':4,'candidateProducts':6,'QRCalls':2,'diagnosticProducts':4,
        'candidateMACs':costs['candidateMACs'],'diagnosticMACs':costs['diagnosticMACs'],'assayMACs':costs['assayMACs'],'prospectiveComposedProducerMACs':costs['prospectiveComposedProducerMACs'],
        'dependentRows':1536,'portsPerRow':3,'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),
        'scope':'Source/byte/actualBF16join/typed/geometry/work/allrow primary-secondarysigned mechanics. No independent real-array numerical replay, quality/rank/runtime/trajectory/generalization/zero-neural certification.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['verify','publish','audit']);parser.add_argument('root');parser.add_argument('--destination');args=parser.parse_args()
    print(json.dumps(publish(args.root,args.destination) if args.action=='publish' else globals()[args.action](args.root)))
