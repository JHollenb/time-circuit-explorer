"""Complete byte custody and actual typed native-capture ledger, without model work."""
import argparse
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

META=r'''
import hashlib,json,sys
from pathlib import Path
s=json.load(sys.stdin);result={}
for role,record in s['cuts'].items():
 p=Path(record['cutFile']['path']);raw=p.read_bytes();cut=json.loads(raw)
 if hashlib.sha256(raw).hexdigest()!=record['cutFile']['sha256'] or cut['manifest']!=record['manifest']:raise ValueError('New cut identity differs')
 root=p.parent.parent;declared={x['key']['value']:x['value'] for x in cut['manifest']['payload']['items']};pages={}
 for slot,page_id in cut['slots'].items():
  path=root/'pages'/(page_id+'.json');raw=path.read_bytes();page=json.loads(raw)
  if hashlib.sha256(raw).hexdigest()!=page_id or page['value']!=declared[slot]:raise ValueError('Own page identity differs')
  files={str(path):page_id}
  if page['tensor_blob'] is not None:files[str(root/'blobs'/(page['tensor_blob']+'.safetensors'))]=page['tensor_blob']
  pages[slot]={'pageID':page_id,'declaredDescriptor':declared[slot],'metadata':page,'files':files}
 result[role]={**record,'cut':cut,'pages':pages}
print(json.dumps(result))
'''
BYTES=NPY_BYTES.split('print(json.dumps')[0]+r'''
def raw_npy(item):
 p=Path(item['path'])
 if str(p) not in witnesses:raise ValueError('NPY byte custody absent')
 with p.open('rb') as f:
  magic=f.read(8);n=struct.unpack('<H' if magic[6]==1 else '<I',f.read(2 if magic[6]==1 else 4))[0];f.read(n);return f.read()
def raw_slot(record):
 page=record['metadata'];path=next(n for n in record['files'] if n.endswith('.safetensors'));raw=Path(path).read_bytes()
 n=struct.unpack('<Q',raw[:8])[0];header=json.loads(raw[8:8+n]);entry=header[page['encoded']['name']];a,b=entry['data_offsets'];payload=raw[8+n+a:8+n+b]
 d=record['declaredDescriptor']
 if entry['dtype']!='BF16' or entry['shape']!=d['shape'] or hashlib.sha256(payload).hexdigest()!=d['sha256']:raise ValueError('Own raw tensor page differs')
 return payload
source=s['cuts']['source'];target=s['cuts']['target'];fs=s['fields'];combined=raw_npy(fs['sourceFields']['combined'])
st,si=raw_slot(source['pages']['text']),raw_slot(source['pages']['image'])
tt,ti=raw_slot(target['pages']['text']),raw_slot(target['pages']['image'])
if combined!=st+si or raw_npy(fs['targetFields']['text'])!=tt or raw_npy(fs['targetFields']['image'])!=ti:raise ValueError('Complete residual cut/array byte join differs')
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,
 'stateByteJoins':{'sourceCombinedSHA256':hashlib.sha256(st+si).hexdigest(),'targetCombinedSHA256':hashlib.sha256(tt+ti).hexdigest(),
 'sourceResidualSplitExact':True,'targetImageExact':True,'targetTextExact':True}}))
'''


def source_items(s):
    values=[s['sourceCut']['cutFile']]
    values += [{'path':n,'sha256':h} for page in s['sourceCut']['pages'].values() for n,h in page['files'].items()]
    values += [{'path':str(Path(s['modelRoot'])/n),'sha256':h} for n,h in s['imageAssets'].items()]
    return values


def verify(root):
    from mrun.client.api import Api
    root,c,j,r=checked(root);p=root/'result';s=read(p/'inputs.json');capture=read(p/'capture.json')
    for job_id,custody in [(s['sourceJobID'],s['sourceCustody']),(s['parentJobID'],s['parentCustody'])]:
        old=Api().json('GET','/api/jobs/'+job_id)
        if old['state']!='succeeded' or old['payload_custody']!=custody:raise ValueError('Original custody differs')
    def remote(code,spec):
        proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','beast','python3 -c '+shlex.quote(code)],input=json.dumps(spec),text=True,capture_output=True,timeout=240)
        if proc.returncode:raise RuntimeError(proc.stderr)
        return json.loads(proc.stdout)
    cuts=remote(META,{'cuts':{k:capture[k] for k in ['source','target']}})
    items=source_items(s)+list(fields(capture));base=j['result']['diagnostics']['cwd']
    for x in cuts.values():
        items += [x['cutFile']]+[{'path':n,'sha256':h} for page in x['pages'].values() for n,h in page['files'].items()]
    items += [{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()]
    unique={}
    for item in items:
        if item['path'] in unique and unique[item['path']]['sha256']!=item['sha256']:raise ValueError('Conflicting complete byte identity')
        unique[item['path']]={**unique.get(item['path'],{}),**item}
    receipt=remote(BYTES,{'allowedRoots':[base,s['modelRoot'],s['sourceCut']['store']],'items':list(unique.values()),'cuts':cuts,'fields':capture})
    receipt.update(schema='tce-terminal-capture-byte-verification-v1',jobID=j['job_id'],verifierSourceSHA256=sha(__file__),
        byteTemplateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),captureSealSHA256=sha(p/'capture-seal.json'))
    save(root/'state-cut-records.json',cuts);save(root/'retained-array-verification.json',receipt)
    return {'files':len(receipt['files']),'bytes':receipt['totalBytesVerified'],'NPYPayloadWitnesses':len(receipt['npyPayloadWitnesses'])}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    for n in p.iterdir():shutil.copyfile(n,out/n.name)
    for n in ['state-cut-records.json','retained-array-verification.json']:shutil.copyfile(root/n,out/n)
    save(out/'scheduler.json',{k:j[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-terminal-capture-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],
        'payloadCustody':c['custody'],'completeCollectedInventory':c['inventory'],'scope':'Paid native source/target capture only; prefix and failed predecessor remain charged. No reduced producer, image, speedup or zero-neural claim.'})
    save(out/'files.json',{n.name:sha(n) for n in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(n.stat().st_size for n in out.iterdir())}


def audit(source):
    p=Path(source);r,s,cap,e,report,v,cuts=[read(p/n) for n in ['request.json','inputs.json','capture.json','execution.json','report.json','retained-array-verification.json','state-cut-records.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request identity differs')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Published member differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]:raise ValueError('Frozen scientific source changed')
    if v['verifierSourceSHA256']!=sha(__file__) or v['byteTemplateSourceSHA256']!=sha(Path(__file__).with_name('publish_sparse_text_kv.py')):raise ValueError('Byte verifier source changed')
    if read(p/'capture-seal.json')!={'requestSHA256':r['requestSHA256'],'captureSHA256':sha(p/'capture.json')} or v['captureSealSHA256']!=sha(p/'capture-seal.json'):raise ValueError('Capture release differs')
    for x in source_items(s)+list(fields(cap)):
        if v['files'][x['path']]['sha256']!=x['sha256']:raise ValueError('Full source/capture byte custody absent')
        if x.get('payloadSHA256') and v['npyPayloadWitnesses'][x['path']]['payloadSHA256']!=x['payloadSHA256']:raise ValueError('NPY payload witness differs')
    for role,record in cuts.items():
        if identity(record['manifest'])!=record['cutID'] or record['manifest']!=record['cut']['manifest'] or len(record['pages'])!=13 or not record['roundtripExact']:raise ValueError('Complete StateCut identity differs')
        for page in record['pages'].values():
            for n,h in page['files'].items():
                if v['files'][n]['sha256']!=h:raise ValueError('Own cut page/blob byte custody absent')
        if record['manifest']['execution']['nativeAdapterSessionCompatible'] is not False:raise ValueError('Capture compatibility scope differs')
    weights={x['name']:x['payloadSHA256'] for x in e['loadedParameters']}
    if len(e['loadedParameters'])!=len(weights) or len(weights)!=146 or not e['sourceInputParameterRNGClosure']:raise ValueError('Original parameter/source closure differs')
    declared={x['key']['value']:x['value'] for x in s['sourceCut']['cut']['manifest']['payload']['items']};context={x['key']['value']:x['value'] for x in declared['context']['items']}
    expected_source={'image':declared['image']['sha256'],'text':declared['text']['sha256']};stages=e['typedActs'];stage_names=['joint'+str(i) for i in range(1,5)]+['single'+str(i) for i in range(20)]+['velocity']
    if [x['stage'] for x in stages]!=stage_names:raise ValueError('Native stage census differs')
    projected=[]
    for a in stages:
        src,dst,ev=a['sourceFrame'],a['outputFrame'],a['evidence'];name=a['stage'];actual={n:x['fingerprint'] for n,x in src['slots'].items()}
        expected={n:weights[n] for n in actual if n in weights}
        if name.startswith('joint'):expected.update(expected_source,imgMod=context['img_mod']['sha256'],txtMod=context['txt_mod']['sha256'],cosine=context['rope']['items'][0]['sha256'],sine=context['rope']['items'][1]['sha256'])
        elif name.startswith('single'):
            if name=='single0':expected['rows']=actual['rows'];projected.append({'kind':'source-audited-concatenation','from':'joint4 text/image','to':'single0 rows','independentIntermediateBytesRetained':False})
            else:expected['rows']=expected_source['combined']
            expected.update(mod=context['single_mod']['sha256'],cosine=context['rope']['items'][0]['sha256'],sine=context['rope']['items'][1]['sha256'])
        else:expected.update(image=cuts['target']['pages']['image']['declaredDescriptor']['sha256'],temb=context['temb']['sha256'])
        if actual!=expected or src['frame_id']!=ev['input_frame_id'] or dst['parent_frame_id']!=src['frame_id'] or dst['frame_id']!=ev['output_frame_id'] or dst['program_fingerprint']!=r['requestSHA256']:raise ValueError('Typed actual source/output lineage differs')
        for n,h in ev['metrics']['outputPayloads'].items():
            abi=dst['slots'][n]['abi'];item={'payloadSHA256':h,'shape':abi['shape'],'logicalDtype':abi['dtype']}
            if dst['slots'][n]['fingerprint']!=structured(item):raise ValueError('Typed output ABI/fingerprint differs')
        expected_source=ev['metrics']['outputPayloads']
    joins=v['stateByteJoins'];by_name={x['stage']:x for x in stages}
    if joins['sourceCombinedSHA256']!=cap['sourceFields']['combined']['payloadSHA256'] or joins['sourceCombinedSHA256']!=by_name['single18']['evidence']['metrics']['outputPayloads']['combined'] or joins['targetCombinedSHA256']!=by_name['single19']['evidence']['metrics']['outputPayloads']['combined'] or stages[-1]['evidence']['metrics']['outputPayloads']['velocity']!=cap['targetFields']['velocity']['payloadSHA256']:raise ValueError('Actual terminal own-state byte join differs')
    if len(e['operations'])!=90 or len(e['attention'])!=24 or e['nativeAffineMACs']!=r['plannedNativeAffineMACs'] or sum(x['nominalMACs'] for x in e['operations'])!=4523022090240 or e['nativeAttentionMACs']!=347892350976 or sum(x['nominalQKAndAVMACs'] for x in e['attention'])!=347892350976:raise ValueError('Paid native work differs')
    progress=[json.loads(x) for x in (p/'prefix-progress.jsonl').read_text().splitlines()]
    if [x['typedAct'] for x in progress]!=stages or progress[-1]['cumulativeAffineMACs']!=e['nativeAffineMACs'] or progress[-1]['cumulativeAttentionMACs']!=e['nativeAttentionMACs']:raise ValueError('Durable completed stage/work stream differs')
    mechanics=read(p/'capture-mechanics.json')
    if not mechanics['roundtripExact'] or mechanics['completeSlots']!=13 or mechanics['sharedStoreModuleAvailable'] or mechanics['externalPoolOpened'] or mechanics['newModelEvaluations']!=0:raise ValueError('Actual old-wheel mechanics scope differs')
    return {'schema':'tce-terminal-capture-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'typedNativeActs':25,'explicitOutputABIs':sum(len(a['evidence']['metrics']['outputPayloads']) for a in stages),
        'distinctOriginalTensors':146,'nativeAffines':90,'nativeAttentionCalls':24,'durableStageRows':len(progress),'completeStateCuts':2,'fullByteFiles':len(v['files']),
        'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'sourceAuditedIntermediateConcatenations':projected,
        'scope':'Actual native component/cut/byte/work mechanics only. One unretained joint-to-single concatenation is source audited, not independently byte replayed. No independent full-native parity, reduced producer, image or zero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source),indent=2))
