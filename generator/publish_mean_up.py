"""Independent byte joins, source/work/ABI coverage and compact publication."""
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
from mean_up_worker import PARAMETERS
from mean_up_analysis import specifications

BYTES=NPY_BYTES.split('print(json.dumps')[0]+r'''
raws={};parameters={}
with Path(s['checkpoint']).open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n
 for item in s['loaded']:
  entry=header[item['name']];a,b=entry['data_offsets'];f.seek(base+a);raw=f.read(b-a)
  if entry['dtype']!='BF16' or entry['shape']!=item['shape'] or [a,b]!=item['dataOffsets'] or len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['payloadSHA256']:raise ValueError('Original source parameter bytes differ')
  raws[item['name']]=raw;parameters[item['name']]={'payloadSHA256':item['payloadSHA256'],'bytes':len(raw),'shape':entry['shape'],'dataOffsets':[a,b]}
def raw_npy(item):
 with Path(item['path']).open('rb') as f:
  magic=f.read(8);n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0];f.read(n);return f.read()
c=s['candidate'];support=list(struct.unpack('<2304q',raw_npy(c['selection_support'])))
if support!=sorted(set(support)) or support[0]<0 or support[-1]>=9216:raise ValueError('Actual source support byte geometry differs')
fused=raws[s['parameterNames'][0]];width=3072*2;up=fused[18432*width:]
maps={'QKVG':fused[:18432*width],'USelected':b''.join(up[j*width:(j+1)*width] for j in support),'UAnchor':up};slices={}
for name,raw in maps.items():
 item=next(x for x in s['physicalMaps'] if x['name']==name);h=hashlib.sha256(raw).hexdigest()
 if h!=item['payloadSHA256'] or len(raw)!=item['bytes']:raise ValueError('Physical source map differs')
 slices[name]={'payloadSHA256':h,'bytes':len(raw),'shape':item['shape']}
anchors=raw_npy(c['anchorUp']);selected=raw_npy(c['upRaw']);completed=raw_npy(c['completedUp']);rowbytes=9216*2;selbytes=2304*2
for row in range(1536):
 stream=0 if row<512 else 1;expected=bytearray(anchors[stream*rowbytes:(stream+1)*rowbytes]);row_selected=selected[row*selbytes:(row+1)*selbytes]
 for index,channel in enumerate(support):expected[channel*2:channel*2+2]=row_selected[index*2:index*2+2]
 if bytes(expected)!=completed[row*rowbytes:(row+1)*rowbytes]:raise ValueError('Own selected overwrite/stream anchor byte completion differs')
text=raw_npy(c['textOut']);image=raw_npy(c['imageOut']);combined=text+image
if hashlib.sha256(combined).hexdigest()!=c['exit']['payloadSHA256']:raise ValueError('Emitted text/image exit byte join differs')
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,
 'originalParameterWitnesses':parameters,'physicalSliceWitnesses':slices,'originalUViewWitness':{'sourceRowStart':18432,'sourceRowStop':27648,'payloadSHA256':hashlib.sha256(up).hexdigest(),'bytes':len(up),'shape':[9216,3072]},
 'completionWitness':{'rows':1536,'selectedWordsPerRow':2304,'omittedWordsPerRow':6912,'textRows':[0,512],'imageRows':[512,1536],'payloadSHA256':hashlib.sha256(completed).hexdigest(),'policy':'Raw BF16 word gather/overwrite only; no numeric mean/projection/product replay.'},
 'exitJoinWitness':{'payloadSHA256':hashlib.sha256(combined).hexdigest(),'textBytes':len(text),'imageBytes':len(image)}}))
'''


def items(p):
    s=read(p/'inputs.json');values=list(s['sourceFields'].values())
    values += [x for n in ['targets.json','candidate.json','differences.json','derived-reference.json'] for x in fields(read(p/n))]
    values += [{'path':str(Path(s['modelRoot'])/n),'sha256':h} for n,h in s['imageAssets'].items()]
    values += [{'path':s['parents'][mode]['root']+'/response-result/'+n,'sha256':h} for mode,records in s['parentRecords'].items() for n,h in records.items()]
    return values


def verify(root):
    from mrun.client.api import Api
    root,col,job,r=checked(root);p=root/'result';s=read(p/'inputs.json');e=read(p/'execution.json');api=Api()
    for parent in s['parents'].values():
        live=api.json('GET','/api/jobs/'+parent['jobID'])
        if live['state']!='succeeded' or live['payload_custody']!=parent['custody']:raise ValueError('Source parent custody differs')
    base=job['result']['diagnostics']['cwd'];unique={}
    for item in items(p)+[{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()]:
        if item['path'] in unique and unique[item['path']]['sha256']!=item['sha256']:raise ValueError('Conflicting file identities')
        unique[item['path']]={**unique.get(item['path'],{}),**item}
    spec={'items':list(unique.values()),'allowedRoots':[base,s['modelRoot'],*[x['root'] for x in s['parents'].values()]],
        'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),'loaded':e['loadedParameters'],
        'candidate':read(p/'candidate.json')['fields'],'physicalMaps':e['trace']['physicalMaps'],'parameterNames':PARAMETERS}
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=300)
    if result.returncode:raise RuntimeError(result.stderr)
    v=json.loads(result.stdout);v.update(schema='tce-mean-up-bytes-v1',jobID=job['job_id'],verifierSourceSHA256=sha(__file__),byteTemplateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
        candidateSealSHA256=sha(p/'candidate-seal.json'),scope='Complete files/NPY/original4params/3physicalmaps/originalUview/streamanchor-selectedoverwrite/exitjoin. SSH raw bytes only. Mean/projection/product arithmetic source-audited, not independently replayed by byte verifier.')
    save(root/'retained-array-verification.json',v)
    return {'files':len(v['files']),'bytes':v['totalBytesVerified'],'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'originalParameterWitnesses':len(v['originalParameterWitnesses']),'physicalMapWitnesses':len(v['physicalSliceWitnesses'])}


def publish(root,destination):
    root,col,job,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Verified release/source differs')
    out=Path(destination);out.mkdir(exist_ok=False,parents=True);streams=[]
    for n in p.iterdir():
        if n.suffix=='.jsonl':
            raw=n.read_bytes();target=out/(n.name+'.gz');target.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless scalar stream differs')
            streams.append({'source':n.name,'sourceSHA256':sha(n),'published':target.name,'publishedSHA256':sha(target),'sourceBytes':len(raw),'publishedBytes':target.stat().st_size})
        else:shutil.copyfile(n,out/n.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'scheduler.json',{k:job[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-mean-up-publication-v1','jobID':job['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':col['custody'],'completeCollectedInventory':col['inventory'],'losslessStreams':streams,
        'scope':'One source-input mean-Up completion, ownquarter/fullgates/fulloutput, immutable73comparisons afterseal/exit. Originallearnedmaps/history paid. No newimage/reference/zero-neuralclaim.'})
    save(out/'files.json',{n.name:sha(n) for n in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(n.stat().st_size for n in out.iterdir())}


def audit(source):
    p=Path(source);r,s,c,e,t,report,v=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json','targets.json','report.json','retained-array-verification.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request identity differs')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Published member differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]:raise ValueError('Executed source changed:'+n)
    if v['verifierSourceSHA256']!=sha(__file__) or v['byteTemplateSourceSHA256']!=sha(Path(__file__).with_name('publish_sparse_text_kv.py')):raise ValueError('Byte verifier source differs')
    for item in items(p):
        if v['files'][item['path']]['sha256']!=item['sha256'] or (item.get('payloadSHA256') and v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']):raise ValueError('Byte witness absent')
    if read(p/'candidate-seal.json')!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json')} or report['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Candidate release differs')
    manifest=identity({'originalParameters':s['originalParameters'],'parentSeals':s['parentSeals'],'helpers':{n:r['sourceHashes'][n] for n in ['mean_up_completion.py','gate_first_mlp.py','image_partition_moments.py']}})
    if e['parameterManifestSHA256']!=manifest or e['loadedParameters']!=s['originalParameters'] or len(e['loadedParameters'])!=4 or not e['sourceParameterRNGClosure'] or e['numericalProfile']!=s['numericalProfile']:raise ValueError('Original parameters/numerical closure differs')
    if len(e['typedActs'])!=1 or len(c['fields'])!=36:raise ValueError('One source candidate with36outputs required')
    act=e['typedActs'][0];expected={**{n:x['payloadSHA256'] for n,x in s['sourceFields'].items()},**{'parameter_'+x['name']:x['payloadSHA256'] for x in s['originalParameters']}}
    if {n:x['fingerprint'] for n,x in act['sourceFrame']['slots'].items()}!=expected:raise ValueError('Source/parameter typed joins differ')
    if e['sourceComparatorSealSHA256']!=s['parentSeals']['matched73'] or act['sourceFrame']['metadata']['sourceComparatorSealSHA256']!=s['parentSeals']['matched73']:raise ValueError('Immutable parent binding differs')
    check_act(act,c['fields'],r['requestSHA256'],manifest)
    allowed={x['path'] for x in s['sourceFields'].values()}|{str(Path(s['modelRoot'])/n) for n in s['imageAssets']};guard=e['fileQuarantine']
    if set(guard['allowedFiles'])!=allowed or set(guard['deniedCanaries'])!=set(s['deniedNumericalInputs']) or not set(guard['successfulFileOpenCounts']).issubset(allowed):raise ValueError('Candidate numeric isolation differs')
    trace=e['trace'];ops=trace['operations'];wanted=[('QKVG',[1,1536,3072],[18432,3072]),('USelected',[1,1536,3072],[2304,3072]),('UAnchor',[1,2,3072],[9216,3072]),('outFull',[1,1536,12288],[3072,12288])]
    if [(x['name'],x['inputShape'],x['weightShape']) for x in ops]!=wanted or any(x['dtype']!='torch.bfloat16' or x['device']!='cuda' for x in ops):raise ValueError('Physical four-affine geometry differs')
    if len(trace['attention'])!=1 or len(e['SDPA'])!=1 or trace['attention'][0]['QShape']!=[1,1024,24,128] or trace['attention'][0]['KShape']!=[1,1536,24,128] or e['SDPA'][0]['nominalQKAndAVMACs']!=9663676416:raise ValueError('Image attention instrument differs')
    if len(trace['products'])!=5 or [x['operator'] for x in trace['products']]!=['imageCovariance','queryCovariance','queryMean','textScores','textAV'] or any(x['dtype']!='torch.float32' or x['device']!='cuda' for x in trace['products']):raise ValueError('Moment products differ')
    total=sum(x['nominalMACs'] for x in ops+trace['attention']+trace['products'])
    if total!=167763247104 or report['totalMatrixMACs']!=total or e['totalMatrixMACs']!=total:raise ValueError('Actual complete work differs')
    if trace['streamMeanDomains']!=[{'name':'text','start':0,'stop':512},{'name':'image','start':512,'stop':1536}] or trace['mapScope']!={'activeSets':0,'maxActiveSets':1,'maxLogicalBytes':113246208}:raise ValueError('Mean domain/map scope differs')
    maps=trace['physicalMaps']
    if [x['name'] for x in maps]!=['QKVG','USelected','UAnchor']:raise ValueError('Physical map census differs')
    for x in maps:
        if v['physicalSliceWitnesses'][x['name']]!={'payloadSHA256':x['payloadSHA256'],'bytes':x['bytes'],'shape':x['shape']}:raise ValueError('Actual source map witness differs')
    if v['originalUViewWitness']!={'sourceRowStart':18432,'sourceRowStop':27648,'payloadSHA256':maps[2]['payloadSHA256'],'bytes':56623104,'shape':[9216,3072]}:raise ValueError('Original U source offset differs')
    if len(v['originalParameterWitnesses'])!=4 or any(v['originalParameterWitnesses'][x['name']]['payloadSHA256']!=x['payloadSHA256'] for x in s['originalParameters']):raise ValueError('Four original parameter byte witnesses differ')
    witness=v['completionWitness']
    if {k:witness[k] for k in ['rows','selectedWordsPerRow','omittedWordsPerRow','textRows','imageRows','payloadSHA256']}!={'rows':1536,'selectedWordsPerRow':2304,'omittedWordsPerRow':6912,'textRows':[0,512],'imageRows':[512,1536],'payloadSHA256':c['fields']['completedUp']['payloadSHA256']}:raise ValueError('Stream anchor/selected overwrite witness differs')
    if v['exitJoinWitness']['payloadSHA256']!=c['fields']['exit']['payloadSHA256']:raise ValueError('Own emitted exit byte join differs')
    streams={}
    for item in read(p/'provenance.json')['losslessStreams']:
        raw=gzip.decompress((p/item['published']).read_bytes())
        if hashlib.sha256(raw).hexdigest()!=item['sourceSHA256']:raise ValueError('Lossless scalar stream differs')
        streams[item['source']]=[json.loads(x) for x in raw.splitlines()]
    def count(item):
        shape=item['shape'];return shape[1] if len(shape)>=3 and shape[0]==1 else shape[0] if len(shape)==1 else 1
    specs=specifications()
    if read(p/'comparison-specifications.json')['comparisons']!=specs:raise ValueError('Exact comparison specification differs')
    support_matched=c['fields']['selection_support']['payloadSHA256']==t['fields']['sparse_selection_support']['payloadSHA256']
    if report['supportMatchedSparse']!=support_matched:raise ValueError('Selected-coordinate applicability differs')
    skipped={x['name'] for x in specs if x['requiresSameSupport'] and not support_matched}
    for name in skipped:
        if report['comparisons'][name].get('status')!='not-applicable':raise ValueError('Unaligned selected coordinates not qualified')
    expected_addresses={(x['name'],i) for x in specs if x['name'] not in skipped for i in range(count(c['fields'][x['candidate']]))}|{('dense_streamAnchor',i) for i in range(2)}
    observations=streams['observations.jsonl'];addresses=[(x['comparison'],x['row']) for x in observations]
    writes=streams['write-observations.jsonl']
    if len(addresses)!=len(set(addresses)) or set(addresses)!=expected_addresses or len(observations)!=report['observations'] or set(report['comparisons'])!={x[0] for x in expected_addresses}|skipped:raise ValueError('Complete comparison addresses differ')
    if len(writes)!=1536 or {x['row'] for x in writes}!=set(range(1536)):raise ValueError('Complete signed write observations differ')
    derived=read(p/'derived-reference.json')
    if derived['source']!=t['fields']['dense_upRaw'] or derived['field']['shape']!=[1,2,9216]:raise ValueError('Post-seal anchor reference provenance differs')
    return {'schema':'tce-mean-up-mechanics-v1','typedActs':1,'explicitOutputABIs':36,'localAffines':4,'imageSDPA':1,'momentProducts':5,'totalMatrixMACs':total,
        'originalParameters':4,'physicalMaps':3,'completedUpByteRows':1536,'sourceMeanRows':2,'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),
        'signedComparisonRows':len(observations),'signedWriteRows':len(writes),'scope':'Source/input/isolation/map/completion/state/work/custody mechanics. Mean/projection/product source-audited and toy-tested, not independently real-array numerically replayed. No quality/generalization/zero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('root');p.add_argument('--destination');a=p.parse_args()
    print(json.dumps(verify(a.root) if a.action=='verify' else publish(a.root,a.destination) if a.action=='publish' else audit(a.root),sort_keys=True))
