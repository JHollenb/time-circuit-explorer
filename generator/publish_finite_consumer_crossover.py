"""Scalar/typed/byte custody audit; full scientific tensors remain on Beast."""
import argparse
import collections
import gzip
import json
import math
from pathlib import Path
import shutil
import subprocess
from contextual_writer_program import read,save,sha
from publish_contextual_writer import check_frame,digest
from publish_contrast_writer import pooled,close

VECTORS = ('N','O','Cno','Con','E','Uno','Vno','gno','Uon','Von','gon')
DOTS = ('N:O','N:Cno','N:Con','Uno:Vno','Uon:Von','sum:gno','sum:gon')


def validate_scalar_analysis(row,width):
    n,d = row['normSquared'],row['innerProducts']
    if row['width']!=width or set(n)!=set(VECTORS) or set(d)!=set(DOTS) or any(not math.isfinite(v) or v<0 for v in n.values()) or any(not math.isfinite(v) for v in d.values()):raise ValueError('Scalar analysis ABI/finiteness differs')
    ratio=lambda key:math.sqrt(n[key]/n['N']) if n['N'] else None
    cosine=lambda a,b:d[a+':'+b]/math.sqrt(n[a]*n[b]) if n[a]*n[b] else None
    expected={'rms':{k:math.sqrt(n[k]/width) for k in VECTORS},'relativeToNative':{k:ratio(k) for k in VECTORS},'responseCosines':{k:cosine('N',k) for k in ('O','Cno','Con')},'paths':{}}
    for suffix in ('no','on'):
        u,v,g='U'+suffix,'V'+suffix,'g'+suffix;terms=[n[u],n[v],2*d[u+':'+v],2*d['sum:'+g],n[g]];total=math.fsum(terms)
        expected['paths'][suffix]={'terms':terms,'gramSum':total,'normSquaredMinusGram':n['E']-total,'termCosine':cosine(u,v),'Urelative':ratio(u),'Vrelative':ratio(v),'gregroupingRMS':math.sqrt(n[g]/width)}
    count=0
    def compare(a,b):
        nonlocal count
        if isinstance(b,dict):
            if set(a)!=set(b):raise ValueError('Scalar summary fields differ')
            for k in b:compare(a[k],b[k])
        elif isinstance(b,list):
            if len(a)!=len(b):raise ValueError('Gram term count differs')
            for x,y in zip(a,b):compare(x,y)
        else:close(a,b);count+=1
    for key,value in expected.items():compare(row[key],value)
    return count


def fingerprint(item):
    return digest({'dtype':'torch.bfloat16' if item['dtype']=='bf16' else item['dtype'],'shape':item['shape'],'bytes':item['payloadSHA256']})


def check_transaction(record,inputs,outputs,request_sha,spec=None):
    before,after,evidence=record['input'],record['output'],record['evidence'];spec=record['spec'] if spec is None else spec
    check_frame(before);check_frame(after)
    if set(spec['reads'])!=set(inputs) or set(spec['writes'])!=set(outputs) or set(before['slots'])!=set(inputs) or set(after['slots'])!=set(inputs)|set(outputs):raise ValueError('Typed complete read/write domain differs')
    if before['program_fingerprint']!=request_sha or after['program_fingerprint']!=request_sha or after['parent_frame_id']!=before['frame_id'] or evidence['input_frame_id']!=before['frame_id'] or evidence['output_frame_id']!=after['frame_id'] or evidence['act']!=spec['identity']:raise ValueError('Typed ancestry differs')
    for name,item in inputs.items():
        if before['slots'][name]['fingerprint']!=fingerprint(item) or after['slots'][name]!=before['slots'][name]:raise ValueError('Typed input payload differs')
    for name,item in outputs.items():
        if after['slots'][name]['fingerprint']!=fingerprint(item):raise ValueError('Typed output payload differs')
    if 'inputPayloadBefore' in record:
        expected={k:v['payloadSHA256'] for k,v in inputs.items()}
        if record['inputPayloadBefore']!=expected or record['inputPayloadAfter']!=expected:raise ValueError('Actual immutable input hash map differs')


def publish(root,mechanics,destination):
    from mrun.client.api import Api
    from finite_consumer_crossover_study import OWNED,PARENTS,WORK,verify
    from finite_consumer_crossover_worker import BRANCHES
    from finite_consumer_crossover import STATE_FIELDS,POLICY,SCORING
    root=Path(root).resolve();destination=Path(destination).resolve();result=root/'result';req=read(root/'request.json');verify(root/'payload',req)
    for name in OWNED:
        if sha(Path(__file__).parent/name)!=req['sourceHashes'][name]:raise ValueError('Current source differs from execution')
    tests=read(mechanics)
    if (tests['testsRun'],tests['failures'],tests['errors'])!=(8,0,0):raise ValueError('Eight mechanics incomplete')
    for name,h in tests['sourceHashes'].items():
        if req['sourceHashes'].get(name)!=h or sha(Path(__file__).parent/name)!=h:raise ValueError('Tested source differs')
    job=read(root/'scheduler-job.json');live=Api().json('GET','/api/jobs/'+job['job_id']);collection=read(root/'collection.json');custody=job['payload_custody'];phases=[custody[k] for k in ('declared','sealed','executed')]
    if live['state']!='succeeded' or live['result']['returncode']!=0 or live['result']!=job['result'] or collection['result']!=job['result'] or live['payload_custody']!=custody or not custody['required'] or not live.get('admission'):raise ValueError('Guarded successful result differs')
    if len({(x['sha256'],x['size_bytes']) for x in phases})!=1 or phases[-1]['host']!='beast' or collection['custody']!=custody or collection['jobID']!=job['job_id']:raise ValueError('Collection/execution custody differs')
    if any(x['config']['requestSHA256']!=req['requestSHA256'] for x in (job,live)):raise ValueError('Job request differs')
    actual={str(p.relative_to(result)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in result.rglob('*') if p.is_file()}
    if actual!=collection['inventory']['files']:raise ValueError('Complete collection inventory differs')
    for name in req['retainedMetadata']:
        if name=='request.json':
            if read(result/name)!=req:raise ValueError('Collected request differs')
        elif sha(result/name)!=req['sourceHashes'][name]:raise ValueError('Retained metadata differs')
    parent=read(result/'qualified-publication.json');old=Api().json('GET','/api/jobs/'+req['parent']['jobID'])
    if sha(result/'qualified-publication.json')!=req['parent']['publicationSHA256'] or old['state']!='succeeded' or old['payload_custody']!=req['parent']['custody'] or parent['jobID']!=old['job_id']:raise ValueError('Parent authority differs')
    for name in PARENTS:
        if name!='publication.json' and sha(result/('qualified-'+name))!=parent['files'][name]['sha256']:raise ValueError('Parent metadata lineage differs')
    source=read(result/'qualified-parent-inputs.json');final=read(result/'qualified-final-serving.json');catalog=read(result/'qualified-programs.json')
    previous_own=read(result/'qualified-consumer-fields.json')['rows'];previous_native=read(result/'qualified-reused-fields.json')['rows'];old_report=read(result/'qualified-report.json')
    if old_report['selected']!='candidate' or catalog['accepted']['programSHA256']!=req['acceptedProgramSHA256'] or catalog['candidate']['programSHA256']!=req['acceptedProgramSHA256'] or source['parameters']!=req['parameters'] or source['numericalProfile']!=req['numericalProfile'] or source['torchVersion']!=req['torchVersion'] or source['rmsNormEpsilon']!=req['rmsNormEpsilon']:raise ValueError('Qualified state/consumer family differs')
    for case,native_name,own_name in [('00','a','p'),('10','b','q')]:
        row=next(x for x in final['rows'] if x['id']==case)
        if row['packet']!=req['packets'][case]:raise ValueError('Qualified packet differs')
        own=next(x for x in row['predictions'] if x['method']=='accepted')
        if own['programSHA256']!=req['acceptedProgramSHA256']:raise ValueError('Accepted state program differs')
        for name,item,method,records in [(native_name,source['cases'][case]['H1'],'native',previous_native),(own_name,own,'candidate',previous_own)]:
            endpoint=req['endpoints'][name];old_record=next(x for x in records if x['case']==case and x['method']==method)
            if any(endpoint[k]!=item[k] for k in ('path','bytes','sha256')) or endpoint['payloadSHA256']!=item.get('payloadSHA256',item['sha256']) or endpoint['shape']!=[1,512,2560] or endpoint['storedDtype']!='<u2' or endpoint['format']!=('npy' if name in ('a','b') else 'raw'):raise ValueError('Qualified endpoint ABI/identity differs')
            if req['reusedConsumers'][name]!={'case':case,'method':method,'inputPayloadSHA256':old_record['inputPayloadSHA256'],'fields':old_record['fields']} or old_record['inputPayloadSHA256']!=endpoint['payloadSHA256']:raise ValueError('Historical exact consumer binding differs')
    report=read(result/'report.json');arrays=read(result/'array-manifest.json');construction=read(result/'typed-construction.json');analyses=read(result/'analysis.json');typed=read(result/'typed-analysis.json');consumers=read(result/'consumer-fields.json');native=read(result/'typed-consumers.json')
    if req['position']!=9 or req['cases']!=['00','10'] or report['physicalWork']!=WORK or req['physicalWork']!=WORK or report['position']!=9 or report['branches']!=BRANCHES or report['requestSHA256']!=req['requestSHA256'] or report['finitePolicy']!=POLICY or report['scoring']!=SCORING:raise ValueError('Fixed executed contract differs')
    if report['numericalProfile']!=req['numericalProfile'] or report['torchVersion']!=req['torchVersion'] or report['selectedParameterBytes']!=sum(x['bytes'] for x in req['parameters']):raise ValueError('Qualified runtime/cost differs')
    if report['analysisSealSHA256']!=sha(result/'analysis-seal.json'):raise ValueError('Analysis seal differs')
    for name,h in read(result/'analysis-seal.json').items():
        if sha(result/name)!=h:raise ValueError('Sealed member differs')
    expected={k:([1,512,2560],'torch.bfloat16','<u2','raw') for k in BRANCHES}
    expected.update({k:([2560],'torch.float64','<f8','npy') for k in STATE_FIELDS})
    for branch in BRANCHES:
        for port,width in [('Q',4096),('K',1024),('V',1024)]:expected[branch+'-'+port]=([1,512,width],'torch.bfloat16','<u2','npy')
    for port,width in [('H1',2560),('Q',4096),('K',1024),('V',1024)]:
        for name in VECTORS:expected[port+'-'+name]=([width],'torch.float64','<f8','npy')
    if set(arrays)!=set(expected) or len({x['path'] for x in arrays.values()})!=60:raise ValueError('Complete unique array domain differs')
    cwd=job['result']['diagnostics']['cwd']
    for name,abi in expected.items():
        x=arrays[name]
        if (x['shape'],x['dtype'],x['storedDtype'],x['format'])!=abi or not x['path'].startswith(cwd+'/arrays/'):raise ValueError('Expected retained ABI/path differs')
        if x['format']=='raw' and (x['bytes']!=math.prod(x['shape'])*2 or x['sha256']!=x['payloadSHA256']):raise ValueError('Actual raw tensor/file payload join differs')
    endpoints={k:{**v,'dtype':'torch.bfloat16'} for k,v in req['endpoints'].items()}
    if set(endpoints)!={'a','b','p','q'} or construction['branches']!=BRANCHES:raise ValueError('Construction domain differs')
    check_transaction(construction,endpoints,{k:arrays[k] for k in (*BRANCHES,*STATE_FIELDS)},req['requestSHA256'])
    if set(construction['rowFields'])!=set(STATE_FIELDS):raise ValueError('State row observations incomplete')
    for x in construction['rowFields'].values():
        if not math.isfinite(x['normSquared']) or x['normSquared']<0 or type(x['nonzeroCoordinates']) is not int or not 0<=x['nonzeroCoordinates']<=2560:raise ValueError('Invalid state remainder observation')
    widths={'H1':2560,'Q':4096,'K':1024,'V':1024}
    if len(analyses)!=4 or {x['port'] for x in analyses}!=set(widths) or len(typed)!=4 or {x['port'] for x in typed}!=set(widths) or report['analyses']!=analyses:raise ValueError('Complete unique analysis domain differs')
    scalar_count=0
    for row in analyses:
        port=row['port'];scalar_count+=validate_scalar_analysis(row,widths[port]);record=next(x for x in typed if x['port']==port)
        inputs=endpoints if port=='H1' else {k:req['reusedConsumers'][k]['fields'][port] for k in endpoints}
        inputs={**inputs,**{k:arrays[k if port=='H1' else k+'-'+port] for k in BRANCHES}}
        check_transaction(record,inputs,{k:arrays[port+'-'+k] for k in VECTORS},req['requestSHA256'])
    names={'normWeight':('input_layernorm.weight',[2560]),'qWeight':('self_attn.q_proj.weight',[4096,2560]),'kWeight':('self_attn.k_proj.weight',[1024,2560]),'vWeight':('self_attn.v_proj.weight',[1024,2560])}
    parameters={k:{'payloadSHA256':next(x['payloadSHA256'] for x in req['parameters'] if x['name']=='model.layers.1.'+name),'dtype':'torch.bfloat16','shape':shape} for k,(name,shape) in names.items()}
    param_hashes={k:v['payloadSHA256'] for k,v in parameters.items()};weight_parent=native['unchangedWeightParent'];check_frame(weight_parent)
    if native['parameterPayloadBefore']!=param_hashes or native['parameterPayloadAfter']!=param_hashes or set(weight_parent['slots'])!=set(names):raise ValueError('Actual immutable parameter set differs')
    for k,v in parameters.items():
        if weight_parent['slots'][k]['fingerprint']!=fingerprint(v):raise ValueError('Actual weight fingerprint differs')
    if consumers['parameters']!=req['parameters'] or consumers['parameterAssets']!=req['selectedParameterAssets'] or consumers['reused']!=req['reusedConsumers']:raise ValueError('Consumer/reuse source binding differs')
    if len(consumers['rows'])!=2 or {x['branch'] for x in consumers['rows']}!=set(BRANCHES) or len(native['rows'])!=2 or {x['branch'] for x in native['rows']}!=set(BRANCHES):raise ValueError('Complete consumer branch domain differs')
    for row in consumers['rows']:
        branch=row['branch'];record=next(x for x in native['rows'] if x['branch']==branch)
        if row['case']!=branch or record['case']!=branch or row['method']!='candidate' or record['method']!='candidate' or row['semantics']!=BRANCHES[branch] or record['semantics']!=BRANCHES[branch] or row['inputField']!=arrays[branch] or row['inputPayloadSHA256']!=arrays[branch]['payloadSHA256']:raise ValueError('Explicit branch/full-input binding differs')
        check_transaction(record,{**parameters,'H1':arrays[branch]},{p:arrays[branch+'-'+p] for p in ('Q','K','V')},req['requestSHA256'],native['spec'])
        if record['input']['parent_frame_id']!=weight_parent['frame_id'] or weight_parent['program_fingerprint']!=req['requestSHA256']:raise ValueError('Weight parent ancestry differs')
        metrics=record['evidence']['metrics']
        if metrics['linears']!=3 or metrics['nominalMACs']!=8053063680 or len(metrics['calls'])!=3 or {x['port'] for x in metrics['calls']}!={'Q','K','V'}:raise ValueError('Executed original work differs')
        for port in ('Q','K','V'):
            item=row['fields'][port];call=next(x for x in metrics['calls'] if x['port']==port);width=widths[port];field=arrays[branch+'-'+port]
            if any(item[k]!=field[k] for k in ('path','bytes','sha256','payloadSHA256','shape')) or call['inputShape']!=[1,512,2560] or call['weightShape']!=[width,2560] or call['nominalMACs']!=512*2560*width:raise ValueError('Original affine geometry/output differs')
    domain={(b,p,i) for b in BRANCHES for p in ('Q','K','V') for i in range(512)};raw={};groups=collections.defaultdict(list)
    with gzip.open(result/'observations.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);i=row['position'];key=(row['branch'],row['port'],i)
            if type(i) is not int or key not in domain or key in raw or row['intervened'] is not (i==9):raise ValueError('Complete comparison row domain differs')
            raw[key]=row
            for selection in ('all','intervened' if i==9 else 'unmodified'):groups[(*key[:2],selection)].append(row)
    if set(raw)!=domain or report['comparisonRows']!=3072 or report['unmodifiedRowControlRows']!=3066:raise ValueError('Comparison count differs')
    pooled_count=0
    for item in report['summaries']:
        rows=groups.pop((item['branch'],item['port'],item['selection']))
        if item['rows']!=len(rows):raise ValueError('Pooled coverage differs')
        for k,v in pooled(rows).items():close(item[k],v);pooled_count+=1
    if groups:raise ValueError('Incomplete pooled domain')
    if len(report['controls'])!=6 or {(x['branch'],x['port']) for x in report['controls']}!={(b,p) for b in BRANCHES for p in ('Q','K','V')}:raise ValueError('Control domain differs')
    for item in report['controls']:
        changed=sum(raw[item['branch'],item['port'],i]['errorRMS']>0 for i in range(512) if i!=9)
        if item['unmodifiedRows']!=511 or item['unmodifiedRowsChanged']!=changed:raise ValueError('Unmodified-row diagnostic count differs')
    remote={k:dict(v) for k,v in req['remoteInputs'].items()};npys={}
    for x in req['selectedParameterAssets']:remote[x['path']]={k:x[k] for k in ('sha256','bytes')}
    for x in arrays.values():remote[x['path']]={k:x[k] for k in ('sha256','bytes')}
    for x in [*req['endpoints'].values(),*(f for r in req['reusedConsumers'].values() for f in r['fields'].values()),*arrays.values()]:
        if x.get('format','npy')=='npy':npys[x['path']]={'payloadSHA256':x['payloadSHA256'],'shape':x['shape'],'dtype':x.get('storedDtype','<u2')}
    segments=[]
    for branch,info in BRANCHES.items():
        for offset,size in [(0,9*5120),(10*5120,(512-10)*5120)]:segments.append({'left':arrays[branch]['path'],'right':req['endpoints'][info['base']]['path'],'offset':offset,'bytes':size})
    code='''import ast,hashlib,json,stat,struct
from pathlib import Path
expected=json.loads(%r);npys=json.loads(%r);segments=json.loads(%r);rows=[];offsets={}
for name,item in expected.items():
 p=Path(name);s=p.lstat()
 if not stat.S_ISREG(s.st_mode) or s.st_size!=item['bytes'] or not name.startswith(('/mnt/hdd8tb/mrun-work/','/mnt/big/llm-models/')):raise ValueError('Unsafe retained file')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 if h.hexdigest()!=item['sha256']:raise ValueError('Fresh retained file differs')
 if name in npys:
  with p.open('rb') as f:
   if f.read(8)!=b'\\x93NUMPY\\x01\\x00':raise ValueError('NPY version differs')
   n=struct.unpack('<H',f.read(2))[0];header=ast.literal_eval(f.read(n).decode('latin1'));offsets[name]=f.tell();x=npys[name]
   if header!={'descr':x['dtype'],'fortran_order':False,'shape':tuple(x['shape'])}:raise ValueError('NPY ABI differs')
   data=f.read()
   if len(data)!=__import__('math').prod(x['shape'])*{'<u2':2,'<f8':8}[x['dtype']] or hashlib.sha256(data).hexdigest()!=x['payloadSHA256']:raise ValueError('NPY actual payload differs')
 rows.append({'path':name,'bytes':s.st_size,'sha256':item['sha256']})
for x in segments:
 data=[]
 for key in ('left','right'):
  with open(x[key],'rb') as f:f.seek(offsets.get(x[key],0)+x['offset']);data.append(f.read(x['bytes']))
 if len(data[0])!=x['bytes'] or data[0]!=data[1]:raise ValueError('Crossed state changed outside row9')
print(json.dumps(rows))
''' % tuple(json.dumps(x) for x in (remote,npys,segments))
    witnesses=json.loads(subprocess.run(['ssh','-T','beast','python3','-'],input=code,text=True,capture_output=True,check=True,timeout=120).stdout)
    destination.mkdir(exist_ok=False)
    for p in result.iterdir():shutil.copyfile(p,destination/p.name)
    for p,name in [(root/'collection.json','collection.json'),(root/'scheduler-job.json','scheduler-job.json'),(Path(mechanics),'mechanics.json')]:shutil.copyfile(p,destination/name)
    publication={'schema':'tce-finite-consumer-crossover-publication-v1','jobID':job['job_id'],'requestSHA256':req['requestSHA256'],'publisherSHA256':sha(__file__),
                 'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in destination.iterdir()},'rawComparisonRows':len(raw),'rebuiltPooledScalars':pooled_count,'rebuiltSignedSummaryScalars':scalar_count,
                 'freshRemoteFiles':witnesses,'unmodifiedStateByteSegmentChecks':len(segments),'retainedNumericFields':len(arrays),
                 'scope':'Complete guarded/source/metadata/scalar/typed ABI and fresh full-byte/payload audit. Vector norms,inner products,crossed-state construction and historical model outputs remain source-qualified producer measurements,not independent vector replay. No local tensor hydration or new scientific/model computation.'}
    save(destination/'publication.json',publication);return publication


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('mechanics');parser.add_argument('destination');a=parser.parse_args();print(json.dumps(publish(a.root,a.mechanics,a.destination)))
