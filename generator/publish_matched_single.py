"""Byte-only witnesses and typed/operator coverage for the matched write anatomy."""
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
from matched_single_worker import PARAMETERS

BYTES=NPY_BYTES.split('print(json.dumps')[0]+r'''
raws={};parameters={}
with Path(s['checkpoint']).open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n
 for item in s['loaded']:
  entry=header[item['name']];a,b=entry['data_offsets'];f.seek(base+a);raw=f.read(b-a)
  if entry['dtype']!='BF16' or entry['shape']!=item['shape'] or [a,b]!=item['dataOffsets'] or len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['payloadSHA256']:raise ValueError('Original single0 parameter bytes differ')
  raws[item['name']]=raw;parameters[item['name']]={'payloadSHA256':item['payloadSHA256'],'bytes':len(raw),'shape':entry['shape']}
def raw_npy(item):
 with Path(item['path']).open('rb') as f:
  magic=f.read(8);n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0];f.read(n);return f.read()
support_raw=raw_npy(s['candidateFields']['sparse_selection_support']);support=list(struct.unpack('<2304q',support_raw))
if support!=sorted(set(support)) or support[0]<0 or support[-1]>=9216:raise ValueError('Actual support byte geometry differs')
fused=raws[s['parameterNames'][0]];output=raws[s['parameterNames'][1]];width=3072*2
maps={'QKVG':fused[:18432*width],'USelected':b''.join(fused[(18432+j)*width:(18433+j)*width] for j in support)}
chunks=[];down=[]
for i in range(3072):
 row=output[i*12288*2:(i+1)*12288*2]
 chunks.append(row[:3072*2]+b''.join(row[(3072+j)*2:(3073+j)*2] for j in support));down.append(row[3072*2:])
maps['outSelected']=b''.join(chunks);down_raw=b''.join(down);slices={}
for name,raw in maps.items():
 item=next(x for x in s['physicalMaps'] if x['name']==name);h=hashlib.sha256(raw).hexdigest()
 if h!=item['payloadSHA256'] or len(raw)!=item['bytes']:raise ValueError('Physical actual support/source map differs')
 slices[name]={'payloadSHA256':h,'bytes':len(raw),'shape':item['shape']}
if hashlib.sha256(down_raw).hexdigest()!=s['diagnosticDownSHA256']:raise ValueError('Diagnostic actual original down slice differs')
joins={}
for mode in ['sparse','dense']:
 text=raw_npy(s['candidateFields'][mode+'_textOut']);image=raw_npy(s['candidateFields'][mode+'_imageOut']);combined=text+image
 if hashlib.sha256(combined).hexdigest()!=s['candidateFields'][mode+'_exit']['payloadSHA256']:raise ValueError('Actual emitted text/image exit join differs')
 joins[mode]={'payloadSHA256':hashlib.sha256(combined).hexdigest(),'textBytes':len(text),'imageBytes':len(image)}
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,
 'originalParameterWitnesses':parameters,'physicalSliceWitnesses':slices,'diagnosticDownWitness':{'payloadSHA256':hashlib.sha256(down_raw).hexdigest(),'bytes':len(down_raw),'shape':[3072,9216]},'exitJoinWitnesses':joins}))
'''


def items(p):
    s=read(p/'inputs.json');values=list(s['sourceFields'].values())
    values += [item for name in ['targets.json','candidate.json','differences.json','anatomy.json'] for item in fields(read(p/name))]
    values += [{'path':str(Path(s['modelRoot'])/name),'sha256':h} for name,h in s['imageAssets'].items()]
    values += [{'path':s['parents'][mode]['root']+'/response-result/'+name,'sha256':h} for mode,records in s['parentRecords'].items() for name,h in records.items()]
    return values


def verify(root):
    from mrun.client.api import Api
    root,col,job,r=checked(root);p=root/'result';s=read(p/'inputs.json');e=read(p/'execution.json');d=read(p/'anatomy-execution.json');api=Api()
    for parent in s['parents'].values():
        live=api.json('GET','/api/jobs/'+parent['jobID'])
        if live['state']!='succeeded' or live['payload_custody']!=parent['custody']:raise ValueError('Common-input parent custody differs')
    base=job['result']['diagnostics']['cwd'];unique={}
    for item in items(p)+[{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()]:
        if item['path'] in unique and unique[item['path']]['sha256']!=item['sha256']:raise ValueError('Conflicting byte identity')
        unique[item['path']]={**unique.get(item['path'],{}),**item}
    spec={'items':list(unique.values()),'allowedRoots':[base,s['modelRoot'],*[x['root'] for x in s['parents'].values()]],
        'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),'loaded':e['replays']['sparse']['loadedParameters'],
        'candidateFields':read(p/'candidate.json')['fields'],'physicalMaps':e['replays']['sparse']['trace']['physicalMaps'],'parameterNames':PARAMETERS,'diagnosticDownSHA256':d['originalDownPayloadSHA256']}
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=300)
    if result.returncode:raise RuntimeError(result.stderr)
    v=json.loads(result.stdout);v.update(schema='tce-matched-single-bytes-v1',jobID=job['job_id'],verifierSourceSHA256=sha(__file__),byteTemplateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
        candidateSealSHA256=sha(p/'candidate-seal.json'),scope='Complete files/NPY/original4parameters/3actualsupportmaps/originalDownslice/2exitjoins. SSH raw byte operations only, no tensor numerics/model execution.')
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
    save(out/'provenance.json',{'schema':'tce-matched-single-publication-v1','jobID':job['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':col['custody'],'completeCollectedInventory':col['inventory'],'losslessStreams':streams,
        'scope':'One common-input sparse72/dense70local replay plusFP64commonP signedwrite diagnosis. Originalmaps/input/prefix/norms paid. No new consumer/image/oraclesupport/fit/sweep/zero-neuralclaim.'})
    save(out/'files.json',{n.name:sha(n) for n in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(n.stat().st_size for n in out.iterdir())}


def audit(source):
    p=Path(source);r,s,c,e,t,a,d,report,v=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json','targets.json','anatomy.json','anatomy-execution.json','report.json','retained-array-verification.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request identity differs')
    for name,h in read(p/'files.json').items():
        if sha(p/name)!=h:raise ValueError('Published member differs')
    for name in r['ownedSources']:
        if sha(Path(__file__).with_name(name))!=r['sourceHashes'][name]:raise ValueError('Executed science changed:'+name)
    if v['verifierSourceSHA256']!=sha(__file__) or v['byteTemplateSourceSHA256']!=sha(Path(__file__).with_name('publish_sparse_text_kv.py')):raise ValueError('Executed byte source differs')
    for item in items(p):
        if v['files'][item['path']]['sha256']!=item['sha256'] or (item.get('payloadSHA256') and v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']):raise ValueError('Complete byte witness missing')
    expected_seal={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json'),'sparseSealSHA256':sha(p/'seal-sparse.json'),'denseSealSHA256':sha(p/'seal-dense.json')}
    if read(p/'candidate-seal.json')!=expected_seal or report['candidateSealSHA256']!=sha(p/'candidate-seal.json') or a['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Both-actor release differs')
    manifest=identity({'originalParameters':s['originalParameters'],'parentSeals':s['parentSeals'],'helpers':{n:r['sourceHashes'][n] for n in ['gate_first_trajectory.py','image_partition_moments.py']}})
    output_abis=0;all_affines=[];attention=0;moment=0
    expected_allowed={x['path'] for x in s['sourceFields'].values()}|{str(Path(s['modelRoot'])/n) for n in s['imageAssets']}
    for mode in ['sparse','dense']:
        case,ex=c['replays'][mode],e['replays'][mode];act=ex['typedActs'][0]
        if case!=read(p/('replay-'+mode+'.json')) or ex!=read(p/('execution-'+mode+'.json')):raise ValueError('Combined actor record differs from separately sealed file')
        if read(p/('seal-'+mode+'.json'))!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/('replay-'+mode+'.json')),'executionSHA256':sha(p/('execution-'+mode+'.json'))}:raise ValueError('Local actor seal differs')
        if ex['parameterManifestSHA256']!=manifest or ex['loadedParameters']!=s['originalParameters'] or not ex['sourceParameterRNGClosure'] or ex['numericalProfile']!=s['numericalProfile']:raise ValueError('Local original source/profile closure differs')
        if len(ex['typedActs'])!=1 or act['stage']!=mode or len(case['fields'])!=(32 if mode=='sparse' else 25):raise ValueError('Local typed output census differs')
        expected_inputs={**{n:x['payloadSHA256'] for n,x in s['sourceFields'].items()},**{'parameter_'+x['name']:x['payloadSHA256'] for x in s['originalParameters']}}
        if {n:x['fingerprint'] for n,x in act['sourceFrame']['slots'].items()}!=expected_inputs:raise ValueError('Common-input/parameter typed joins differ')
        check_act(act,case['fields'],r['requestSHA256'],manifest);output_abis+=len(case['fields'])
        for name,item in case['fields'].items():
            if c['fields'][mode+'_'+name]!=item:raise ValueError('Combined actor field differs')
        guard=ex['fileQuarantine'];expected_denied=set(s['deniedNumericalInputs'])|({x['path'] for x in c['replays']['sparse']['fields'].values()} if mode=='dense' else set())
        if set(guard['allowedFiles'])!=expected_allowed or set(guard['deniedCanaries'])!=expected_denied or not set(guard['successfulFileOpenCounts']).issubset(expected_allowed):raise ValueError('Pre-release input isolation differs')
        if mode=='dense' and (ex['precedingSparseSealSHA256']!=sha(p/'seal-sparse.json') or act['sourceFrame']['metadata']['precedingSparseSealSHA256']!=sha(p/'seal-sparse.json')):raise ValueError('Sparse-before-dense join differs')
        trace=ex['trace'];ops=trace['operations'];all_affines+=ops
        wanted=([('QKVG',[18432,3072]),('USelected',[2304,3072]),('outSelected',[3072,5376])] if mode=='sparse' else [('QKVGU',[27648,3072]),('outFull',[3072,12288])])
        if [(x['name'],x['weightShape']) for x in ops]!=wanted or any(x['inputShape']!=[1,1536,5376 if x['name']=='outSelected' else 12288 if x['name']=='outFull' else 3072] or x['dtype']!='torch.bfloat16' or x['device']!='cuda' for x in ops):raise ValueError('Physical fused local affine geometry differs')
        if len(ex['SDPA'])!=1 or len(trace['attention'])!=1 or trace['attention'][0]['QShape']!=[1,1024,24,128] or trace['attention'][0]['KShape']!=[1,1536,24,128]:raise ValueError('Complete image attention instrument differs')
        if len(trace['products'])!=5 or [x['operator'] for x in trace['products']]!=['imageCovariance','queryCovariance','queryMean','textScores','textAV'] or any(x['dtype']!='torch.float32' or x['device']!='cuda' for x in trace['products']):raise ValueError('Moment product coverage differs')
        attention+=sum(x['nominalMACs'] for x in trace['attention']);moment+=sum(x['nominalMACs'] for x in trace['products'])
    maps=e['replays']['sparse']['trace']['physicalMaps']
    if [x['name'] for x in maps]!=['QKVG','USelected','outSelected'] or e['replays']['sparse']['trace']['mapScope']!={'activeSets':0,'maxActiveSets':1,'maxLogicalBytes':113246208} or e['replays']['dense']['trace']['physicalMaps']:raise ValueError('Physical sparse map scope differs')
    for item in maps:
        if v['physicalSliceWitnesses'][item['name']]!={'payloadSHA256':item['payloadSHA256'],'bytes':item['bytes'],'shape':item['shape']}:raise ValueError('Actual map byte witness differs')
    if len(v['originalParameterWitnesses'])!=4 or any(v['originalParameterWitnesses'][x['name']]['payloadSHA256']!=x['payloadSHA256'] for x in s['originalParameters']):raise ValueError('Original4parameter witnesses differ')
    if v['diagnosticDownWitness']['payloadSHA256']!=d['originalDownPayloadSHA256']:raise ValueError('Diagnostic original down input differs')
    if len(d['typedActs'])!=1:raise ValueError('One anatomy Act required')
    act=d['typedActs'][0];expected={'admittedProduct':c['fields']['dense_product']['payloadSHA256'],'originalDown':d['originalDownPayloadSHA256'],'ownSupport':c['fields']['sparse_selection_support']['payloadSHA256'],
        'sparseExit':c['fields']['sparse_exit']['payloadSHA256'],'denseExit':c['fields']['dense_exit']['payloadSHA256'],'gate':c['fields']['dense_gate']['payloadSHA256']}
    if {n:x['fingerprint'] for n,x in act['sourceFrame']['slots'].items()}!=expected or d['parameterManifestSHA256']!=manifest or not d['sourceInputClosure']:raise ValueError('Common-product anatomy input joins differ')
    if act['sourceFrame']['metadata']['candidateSealSHA256']!=sha(p/'candidate-seal.json') or d['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Diagnostic before actor release')
    for mode in ['sparse','dense']:
        if act['sourceFrame']['metadata'][mode+'ProducerFrameID']!=e['replays'][mode]['typedActs'][0]['outputFrame']['frame_id']:raise ValueError('Diagnostic actual producer frame differs')
    if d['loadedParameters']!=[s['originalParameters'][1]]:raise ValueError('Diagnostic original parameter scope differs')
    check_act(act,a['fields'],r['requestSHA256'],manifest);output_abis+=len(a['fields'])
    products=d['products']
    if [(x['operator'],x['leftShape'],x['rightShape']) for x in products]!=[('fullDown',[1,1536,9216],[9216,3072]),('selectedDown',[1,1536,2304],[2304,3072])] or any(x['dtype']!='torch.float64' or x['device']!='cuda' for x in products):raise ValueError('Paid FP64 contraction geometry differs')
    total=sum(x['nominalMACs'] for x in all_affines)+attention+moment+sum(x['nominalMACs'] for x in products)
    if len(all_affines)!=5 or attention!=19327352832 or moment!=4432330752 or total!=389771427840 or report['totalMatrixMACs']!=total or output_abis!=66:raise ValueError('Actual complete work/output account differs')
    streams={}
    for item in read(p/'provenance.json')['losslessStreams']:
        raw=gzip.decompress((p/item['published']).read_bytes())
        if hashlib.sha256(raw).hexdigest()!=item['sourceSHA256']:raise ValueError('Lossless signed stream differs')
        streams[item['source']]=[json.loads(x) for x in raw.splitlines()]
    rows=streams['observations.jsonl'];addresses=[(x['comparison'],x['row']) for x in rows]
    def count_rows(item):
        shape=item['shape']
        return shape[1] if len(shape)>=3 and shape[0]==1 else shape[0] if len(shape)==1 else 1
    expected_addresses={(('archive_'+mode+'_'+name),i) for mode,targets in t['fields'].items() for name,item in targets.items() for i in range(count_rows(item))}
    operands=['normalized','queryRaw','keyRaw','valueRaw','gateRaw','activation','attentionQuery','attentionKey','attentionValue','imageAttention','attentionRead','preResidualUpdate','textOut','imageOut','exit','alpha','textRead','gate','upRaw','product']
    expected_addresses|={('operand_'+name,i) for name in operands for i in range(count_rows(c['fields']['sparse_'+name]))}
    if len(rows)!=report['observations'] or len(addresses)!=len(set(addresses)) or set(addresses)!=expected_addresses or set(report['comparisons'])!={name for name,_ in expected_addresses}:raise ValueError('Complete comparison row identity differs')
    channels=streams['channel-observations.jsonl'];anatomy=streams['anatomy-observations.jsonl']
    if len(channels)!=27648 or {(x['stratum'],x['channel']) for x in channels}!={(n,j) for n in ['all','text','image'] for j in range(9216)} or len(anatomy)!=1536 or {x['row'] for x in anatomy}!=set(range(1536)):raise ValueError('Complete channel/row anatomy coverage differs')
    return {'schema':'tce-matched-single-mechanics-v1','typedActs':3,'explicitOutputABIs':output_abis,'actorFields':57,'anatomyFields':9,'localAffines':5,'imageSDPA':2,'momentProducts':10,'FP64DownProducts':2,'totalMatrixMACs':total,
        'originalParameters':4,'physicalMaps':3,'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'signedComparisonRows':len(rows),'channelRows':len(channels),'anatomyRows':len(anatomy),
        'scope':'Execution/source/common-input/operand/map/work/custody coverage only. Numeric replay/errors/cancellation magnitudes are observations, not quality or zero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('root');p.add_argument('--destination');a=p.parse_args()
    print(json.dumps(publish(a.root,a.destination) if a.action=='publish' else globals()[a.action](a.root),sort_keys=True))
