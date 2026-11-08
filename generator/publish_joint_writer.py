"""Independent public-scalar, typed and byte audit; no local model arrays."""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
from contextual_writer_program import read,save,sha,BUDGET,FINITE
from publish_contextual_writer import check_frame,digest
from publish_contrast_writer import close,pooled
from publish_finite_consumer_crossover import fingerprint
from joint_writer_math import POLICY
from coupled_normalization import SOURCE_POLICY

KEYS=('Hendpoint','Hresponse','Zendpoint','Zresponse')
FUTURES=('slot0-dose1','slot0-dosehalf','slot1-dose1','slot1-dosehalf')
DISCOVERY=['A','B','00','10']


def scalar_objective(errors,scales):
    if set(errors)!=set(KEYS) or set(scales)!=set(KEYS) or any(not math.isfinite(x) or x<0 for x in [*errors.values(),*scales.values()]):raise ValueError('Finite objective primitive domain differs')
    terms={k:errors[k]/(scales[k] if scales[k] else 1.) for k in KEYS}
    return {'errorEnergies':dict(errors),'nativeEnergies':dict(scales),'terms':terms,'score':math.fsum(terms.values())/4,'zeroReferenceUsesAbsolute':{k:scales[k]==0 for k in KEYS}}


def compare(actual,expected):
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('Scalar field domain differs')
        return sum(compare(actual[k],v) for k,v in expected.items())
    if isinstance(expected,bool):
        if actual is not expected:raise ValueError('Boolean scalar policy differs')
        return 1
    close(actual,expected);return 1


def objective_record(row,scales):
    return compare({k:row[k] for k in ('errorEnergies','nativeEnergies','terms','score','zeroReferenceUsesAbsolute')},scalar_objective(row['errorEnergies'],scales))


def eligible_futures(rows,keys=DISCOVERY):
    if len(rows)!=4 or [r['id'] for r in rows]!=list(FUTURES):raise ValueError('Four attempted futures absent')
    good=[]
    for at,r in enumerate(rows):
        if r['slot']!=at//2 or r['dose']!=(1. if at%2==0 else .5):raise ValueError('Future slot/dose differs')
        done=r['completedHistories']
        if done!=keys[:len(done)]:raise ValueError('Partial history order differs')
        if r['status']=='complete':
            if done!=keys or r['programSHA256'] is None or r['score'] is None or not math.isfinite(r['score']):raise ValueError('Incomplete future cannot be selected')
            good.append(r)
        elif r['status'] in ('construction-incomplete','numerically-inadmissible'):
            if r['score'] is not None or not r['reason'] or r['stage'] not in ('tree-fit','FP32-dictionary-admission','BF16-state-admission'):raise ValueError('Unavailable future has synthetic score or unknown cause')
            if r['stage']!='BF16-state-admission' and (done or r['programSHA256'] is not None):raise ValueError('Unconstructed future has fabricated evidence')
        else:raise ValueError('Unknown future status')
    return good


def transaction(record,inputs,outputs,request_sha):
    before,after,evidence,spec=[record[k] for k in ('input','output','evidence','spec')]
    check_frame(before);check_frame(after)
    if set(spec['reads'])!=set(inputs) or set(spec['writes'])!=set(outputs) or set(before['slots'])!=set(inputs) or set(after['slots'])!=set(inputs)|set(outputs):raise ValueError('Complete typed transaction domain differs')
    if before['program_fingerprint']!=request_sha or after['program_fingerprint']!=request_sha or after['parent_frame_id']!=before['frame_id'] or after['act']!=spec['identity'] or evidence['input_frame_id']!=before['frame_id'] or evidence['output_frame_id']!=after['frame_id'] or evidence['act']!=spec['identity']:raise ValueError('Typed transaction ancestry differs')
    for k,v in inputs.items():
        if before['slots'][k]['fingerprint']!=fingerprint(v) or after['slots'][k]!=before['slots'][k]:raise ValueError('Typed input payload differs')
    for k,v in outputs.items():
        expected=fingerprint(v) if isinstance(v,dict) else digest({'type':type(v).__name__,'repr':repr(v)})
        if after['slots'][k]['fingerprint']!=expected:raise ValueError('Typed output payload differs')
    expected={k:v['payloadSHA256'] for k,v in inputs.items()}
    if record['inputPayloadBefore']!=expected or record['inputPayloadAfter']!=expected:raise ValueError('Actual immutable input hashes differ')


def fresh_bytes(remote,npys):
    code='''import ast,hashlib,json,math,stat,struct
from pathlib import Path
expected=json.loads(%r);npys=json.loads(%r);rows=[]
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
   n=struct.unpack('<H',f.read(2))[0];header=ast.literal_eval(f.read(n).decode('latin1'));x=npys[name]
   if header!={'descr':x['dtype'],'fortran_order':False,'shape':tuple(x['shape'])}:raise ValueError('NPY ABI differs')
   body=hashlib.sha256();size=0
   for block in iter(lambda:f.read(8<<20),b''):body.update(block);size+=len(block)
   if size!=math.prod(x['shape'])*{'<u2':2,'<f8':8,'<i8':8}[x['dtype']] or body.hexdigest()!=x['payloadSHA256']:raise ValueError('NPY payload differs')
 rows.append({'path':name,'bytes':s.st_size,'sha256':item['sha256']})
print(json.dumps(rows))
''' % (json.dumps(remote),json.dumps(npys))
    return json.loads(subprocess.run(['ssh','-T','beast','python3','-'],input=code,text=True,capture_output=True,check=True,timeout=120).stdout)


def publish(root,mechanics,destination):
    from mrun.client.api import Api
    from joint_writer_study import OWNED,PARENTS,WORK,verify,FAMILY_SHA
    root=Path(root).resolve();result=root/'result';destination=Path(destination).resolve();req=read(root/'request.json');verify(root/'payload',req)
    tests=read(mechanics)
    if (tests['testsRun'],tests['failures'],tests['errors'])!=(12,0,0) or set(tests['sourceHashes'])!=set(OWNED):raise ValueError('Twelve source-bound supplied mechanics absent')
    for name in OWNED:
        if sha(Path(__file__).parent/name)!=req['sourceHashes'][name] or tests['sourceHashes'][name]!=req['sourceHashes'][name]:raise ValueError('Current/tested/executed source differs')
    job=read(root/'scheduler-job.json');live=Api().json('GET','/api/jobs/'+job['job_id']);collection=read(root/'collection.json');custody=job['payload_custody'];phases=[custody[k] for k in ('declared','sealed','executed')]
    if live['state']!='succeeded' or live['result']['returncode']!=0 or live['result']!=job['result'] or collection['result']!=job['result'] or live['payload_custody']!=custody or not custody['required'] or not live.get('admission'):raise ValueError('Guarded successful result differs')
    if len({(x['sha256'],x['size_bytes']) for x in phases})!=1 or phases[-1]['host']!='beast' or collection['custody']!=custody or collection['jobID']!=job['job_id'] or any(x['config']['requestSHA256']!=req['requestSHA256'] for x in (job,live)):raise ValueError('Execution/collection/request custody differs')
    cwd=job['result']['diagnostics']['cwd']
    if cwd!='/mnt/hdd8tb/mrun-work/'+job['job_id']:raise ValueError('Host-local root differs')
    members={'typed-normalization.json','typed-proposals.json','selection.json','futures.json','programs.json','array-manifest.json','observations.jsonl.gz','initializer-seal.json','final-seal.json','initializer-serving.json','final-serving.json'}
    actual={str(p.relative_to(result)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in result.rglob('*') if p.is_file()}
    if actual!=collection['inventory']['files'] or set(actual)!=set(req['retainedMetadata'])|members|{'analysis-seal.json','report.json'}:raise ValueError('Complete collection differs')
    for name in req['retainedMetadata']:
        if name=='request.json':
            if read(result/name)!=req:raise ValueError('Collected request differs')
        elif sha(result/name)!=req['sourceHashes'][name]:raise ValueError('Retained source metadata differs')
    parent=read(result/'parent-publication.json');old=Api().json('GET','/api/jobs/'+req['parent']['jobID'])
    if sha(result/'parent-publication.json')!='0a3a231312dc619a3266a7b15b6e35640b95616575765176c9fed10f041b5d54' or sha(result/'parent-publication.json')!=req['parent']['publicationSHA256'] or old['state']!='succeeded' or old['payload_custody']!=req['parent']['custody'] or parent['jobID']!=old['job_id']:raise ValueError('Qualified104 authority differs')
    for name in PARENTS:
        if name!='publication.json' and sha(result/('parent-'+name))!=parent['files'][name]['sha256']:raise ValueError('Qualified104 member differs')
    if req['cases']!=read(result/'parent-request.json')['cases'] or [r['id'] for r in req['cases']]!=['A','B','00','10','01','11','novel']:raise ValueError('Exact original seven raw prompts differ')
    normpub=read(result/'normalization109-publication.json');normreq=read(result/'normalization109-request.json');source=read(result/'parent-parent-inputs.json')
    if sha(result/'normalization109-publication.json')!='8bb04982247fdcaeab39b4d98565851d4988bde2991da23660570da8a89acce4' or sha(result/'normalization109-request.json')!=normpub['files']['request.json']['sha256']:raise ValueError('109 gamma authority differs')
    if req['gammaParameter']!=normreq['gammaParameter'] or req['selectedParameterAsset']!=normreq['selectedParameterAsset'] or req['gammaParameter'] not in source['parameters']:raise ValueError('Gamma/shard binding differs')
    if sha(result/'family-plan.md')!=FAMILY_SHA or req['physicalWork']!=WORK or req['objectivePolicy']!=POLICY or req['budget']!=BUDGET or req['finitePolicy']!=FINITE or req['discovery']!=DISCOVERY or req['transfer']!=['01','11'] or req['novelNoTarget']!=['novel']:raise ValueError('Fixed construction contract differs')
    report=read(result/'report.json');arrays=read(result/'array-manifest.json');normal=read(result/'typed-normalization.json');proposals=read(result/'typed-proposals.json');futures=read(result/'futures.json');selections=read(result/'selection.json');catalog=read(result/'programs.json')
    for k,v in [('requestSHA256',req['requestSHA256']),('physicalWork',WORK),('objectivePolicy',POLICY),('sourcePolicy',SOURCE_POLICY),('budget',BUDGET),('torchVersion',req['torchVersion']),('numericalProfile',req['numericalProfile'])]:
        if report[k]!=v:raise ValueError('Executed contract/profile differs: '+k)
    if any(req[k]!=source[k] for k in ('numericalProfile','torchVersion','rmsNormEpsilon')):raise ValueError('Original source profile differs')
    seal=read(result/'analysis-seal.json')
    if set(seal)!=members or report['analysisSealSHA256']!=sha(result/'analysis-seal.json') or any(sha(result/n)!=h for n,h in seal.items()):raise ValueError('Complete analysis seal differs')
    eligible=eligible_futures(futures);scales=report['initialObjective']['nativeEnergies'];scalar_count=objective_record(report['initialObjective'],scales)+objective_record(report['selectedDiscoveryObjective'],scales)
    for row in eligible:scalar_count+=objective_record(row,scales)
    emitted={r['id'] for r in futures if r['programSHA256'] is not None}
    if set(catalog)!={'parent','accepted'}|emitted or {k:catalog['parent'][k] for k in req['parent']['program']}!=req['parent']['program'] or catalog['accepted']['programSHA256']!=catalog[report['selected']]['programSHA256']:raise ValueError('Program catalog/accepted parent differs')
    remote={};npys={}
    def retain(item):
        entry={k:item[k] for k in ('sha256','bytes')};path=item['path']
        if path in remote and remote[path]!=entry:raise ValueError('Conflicting byte authorities')
        remote[path]=entry
        if item.get('format')=='npy':npys[path]={'shape':item['shape'],'dtype':item['storedDtype'],'payloadSHA256':item['payloadSHA256']}
    inventory={r['path']:r for r in parent['freshRemoteFiles']}
    for path,h in req['servingAssets'].items():
        item=inventory[path]
        if item['sha256']!=h:raise ValueError('Historical serving asset differs')
        retain(item)
    if report['authenticatedServingAssets']!=[{'path':p,'sha256':h,'bytes':inventory[p]['bytes']} for p,h in req['servingAssets'].items()]:raise ValueError('Actual asset authentication differs')
    retain(req['selectedParameterAsset'])
    for item in catalog.values():
        p=item['program'];encoded=json.dumps(p,indent=2,sort_keys=True,allow_nan=False).encode()
        if hashlib.sha256(encoded).hexdigest()!=item['programSHA256'] or p['budget']!=BUDGET or p['finitePolicy']!=FINITE or p['literals']!=catalog['parent']['program']['literals'] or p['featureNames']!=catalog['parent']['program']['featureNames'] or len(p['featureNames'])!=155 or p['prefixSHA256']!=catalog['parent']['program']['prefixSHA256'] or len(p['trees'])!=2 or len(p['dictionaries'])!=2 or p['prefixPositions']!=[0,1] or (p['width'],p['length'])!=(2560,512):raise ValueError('Program identity/grammar differs')
        folder=Path(item['directory']);files={'program.json':item['programSHA256'],'prefix.bf16':p['prefixSHA256'],**{d['path']:d['sha256'] for d in p['dictionaries']}}
        if item['files']!=files:raise ValueError('Complete program bundle differs')
        retain({'path':str(folder/'program.json'),'sha256':item['programSHA256'],'bytes':len(encoded)});retain({'path':str(folder/'prefix.bf16'),'sha256':p['prefixSHA256'],'bytes':10240})
        for tree,d in zip(p['trees'],p['dictionaries']):
            leaves=[]
            def visit(node,depth):
                if 'atom' in node:
                    if type(node['atom']) is not int:raise ValueError('Atom address type differs')
                    leaves.append(node['atom']);return
                if depth>=4 or type(node['feature']) is not int or not 0<=node['feature']<len(p['featureNames']) or not math.isfinite(node['threshold']):raise ValueError('Branch grammar differs')
                visit(node['left'],depth+1);visit(node['right'],depth+1)
            visit(tree,0)
            if len(leaves)>16 or set(leaves)!=set(range(d['atoms'])) or d['atoms']>16 or d['bytes']!=d['atoms']*2560*4:raise ValueError('Occupied atom budget differs')
            retain({'path':str(folder/d['path']),'sha256':d['sha256'],'bytes':d['bytes']})
    children={};packets={};forbidden=sorted(set(source['forbiddenServingPaths'])|{p['path'] for r in read(result/'parent-final-serving.json')['rows'] for p in [r['H0'],*r['predictions']]})
    for phase in ('initializer','final'):
        child=read(result/(phase+'-serving.json'));seal=read(result/(phase+'-seal.json'));cases=DISCOVERY if phase=='initializer' else [r['id'] for r in req['cases']];methods=['parent'] if phase=='initializer' else ['parent','accepted']
        if [r['id'] for r in child['rows']]!=cases or child['loadedNumericalRuntimeRoots'] or child['newNeuralEvaluations']!=0 or child['blockedImports']!=['numpy','torch'] or child['fileCanaryDenials']!=forbidden or seal['receiptSHA256']!=sha(result/(phase+'-serving.json')) or seal['outputs']!=[p for r in child['rows'] for p in [r['H0'],*r['predictions']]]:raise ValueError('Complete fresh serving seal/isolation differs')
        for row in child['rows']:
            key=row['id'];packet=row['packet'];children[phase,key]=row;packets[key]=packet
            if packet['positions']!=list(range(512)) or len(packet['ids'])!=512 or any(type(t) is not int or not 0<=t<151936 for t in packet['ids']) or len(packet['mask'])!=512 or any(type(m) is not int or m not in (0,1) for m in packet['mask']) or any(a<b for a,b in zip(packet['mask'],packet['mask'][1:])) or row['activeRows']!=sum(packet['mask']):raise ValueError('Fresh packet ABI differs')
            if key in source['cases']:
                for label,wanted in [('ids',source['cases'][key]['tokenIDsSHA256']),('mask',source['cases'][key]['maskSHA256'])]:
                    if hashlib.sha256(struct.pack('<512q',*packet[label])).hexdigest()!=wanted:raise ValueError('Native input packet differs')
            if [p['method'] for p in row['predictions']]!=methods:raise ValueError('Fresh prediction domain differs')
            for method,item in [('H0',row['H0'])]+[(p['method'],p) for p in row['predictions']]:
                name=key+'-H0.bf16' if method=='H0' else method+'-'+key+'.bf16'
                if item['path']!=cwd+'/'+phase+'/'+name or item['bytes']!=2621440:raise ValueError('Fresh output ABI differs')
                if method!='H0' and (item['programSHA256']!=catalog[method]['programSHA256'] or item['coordinateAdditions']!=2611200 or len(item['leaves'])!=510 or not 0<=item['branchDecisions']<=4080):raise ValueError('Fresh output/program work differs')
                retain(item)
    if report['transferReleasedAfterSealSHA256']!=sha(result/'final-seal.json'):raise ValueError('Transfer release seal differs')
    for key in DISCOVERY:
        a,b=children['initializer',key],children['final',key]
        if a['packet']!=b['packet'] or a['H0']['sha256']!=b['H0']['sha256']:raise ValueError('Fresh initializer replay differs')
        retain({**source['cases'][key]['H0'],'format':'npy','storedDtype':'<u2'})
    for key in DISCOVERY+req['transfer']:retain({**source['cases'][key]['H1'],'format':'npy','storedDtype':'<u2'})
    # Selection reconstructs parent-first minimum, without quality thresholds.
    if len(selections)!=2 or [r['slot'] for r in selections]!=[0,1] or len(proposals)!=2 or [r['slot'] for r in proposals]!=[0,1]:raise ValueError('Single sequential sweep differs')
    selected='parent';previous_frame=None;current_objective=report['initialObjective']
    for slot,row in enumerate(selections):
        offered=[r for r in eligible if r['slot']==slot]
        if row['parent']['id']!=selected or row['candidates']!=offered:raise ValueError('Selection candidate/parent domain differs')
        scalar_count+=objective_record(row['parent'],scales)
        scalar_count+=compare({k:row['parent'][k] for k in current_objective},current_objective)
        winner=min([row['parent'],*offered],key=lambda x:x['score'])['id']
        if row['selected']!=winner or row['programSHA256']!=catalog[winner]['programSHA256'] or row['restoredFiles']!=catalog[winner]['files']:raise ValueError('Parent-first selection/exact restore differs')
        for name,h in row['restoredFiles'].items():
            original=next(v for p,v in remote.items() if p==catalog[winner]['directory']+'/'+name);retain({'path':row['restoredDirectory']+'/'+name,**original})
        check_frame(row['parentFrame']);check_frame(row['selectedFrame'])
        if row['parentFrame']['slots']['programSHA256']['fingerprint']!=digest({'type':'str','repr':repr(catalog[selected]['programSHA256'])}) or previous_frame is not None and row['parentFrame']!=previous_frame:raise ValueError('Selected parent program Frame differs')
        if set(row['candidateFrames'])!={r['id'] for r in offered}:raise ValueError('Candidate program Frame domain differs')
        for name,frame in row['candidateFrames'].items():
            check_frame(frame)
            if frame['parent_frame_id']!=row['parentFrame']['frame_id'] or frame['slots']['programSHA256']['fingerprint']!=digest({'type':'str','repr':repr(catalog[name]['programSHA256'])}):raise ValueError('Candidate program Frame differs')
        if row['selectedFrame']!=(row['parentFrame'] if winner==selected else row['candidateFrames'][winner]):raise ValueError('Exact selected Frame restore differs')
        winning_objective=next(x for x in [row['parent'],*offered] if x['id']==winner)
        current_objective={k:winning_objective[k] for k in report['initialObjective']};selected=winner;previous_frame=row['selectedFrame']
    if selected!=report['selected']:raise ValueError('Final selection differs')
    scalar_count+=compare(report['selectedDiscoveryObjective'],current_objective)
    # Numeric descriptors and transactions authenticate producer calculations.
    normal_rows=normal['rows'];expected_states={x+'-'+k for x in ('target','parent') for k in DISCOVERY+req['transfer']}|{'accepted-'+k for k in req['transfer']}|{r['id']+'-'+k for r in futures for k in r['completedHistories']}
    if len(normal_rows)!=len(expected_states) or {r['state'] for r in normal_rows}!=expected_states or len(normal_rows)>30:raise ValueError('Actual complete/partial normalization work differs')
    expected={'gamma':([2560],'torch.bfloat16','<u2','raw')}
    for state in expected_states:
        for port in ('H','Z'):expected[state+'-'+port]=([1,512,2560],'torch.bfloat16','<u2','raw')
    edge_count=sum(sum(i>=2 and a==b for i,(a,b,m) in enumerate(zip(packets[l]['ids'],packets[r]['ids'],packets[l]['mask']))) for l,r in [('A','B'),('00','10')])
    features=len(catalog['parent']['program']['featureNames'])
    shapes={'h':[2040,2560],'z':[2040,2560],'target':[2040,2560],'zt':[2040,2560],'oldSlot':[2040,2560],'features':[2040,features],'weights':[2040],'edges':[edge_count,2],'pairWeights':[edge_count],'gamma':[2560],'scaleVector':[4],'gradient':[2040,2560],'step':[2040,2560]}
    for slot in (0,1):
        for k,shape in shapes.items():expected['fit'+str(slot)+'-'+k]=(shape,'torch.int64' if k=='edges' else 'torch.float64','<i8' if k=='edges' else '<f8','npy')
    if set(arrays)!=set(expected) or len({v['path'] for v in arrays.values()})!=len(arrays):raise ValueError('Complete unique retained numeric domain differs')
    for name,abi in expected.items():
        item=arrays[name]
        if (item['shape'],item['dtype'],item['storedDtype'],item['format'])!=abi or not item['path'].startswith(cwd+'/arrays/'):raise ValueError('Retained numeric ABI/root differs')
        if item['format']=='raw' and (item['bytes']!=math.prod(item['shape'])*2 or item['sha256']!=item['payloadSHA256']):raise ValueError('Raw payload/file join differs')
        retain(item)
    if arrays['gamma']['payloadSHA256']!=req['gammaParameter']['payloadSHA256'] or normal['parameterPayloadBefore']!=arrays['gamma']['payloadSHA256'] or normal['parameterPayloadAfter']!=normal['parameterPayloadBefore']:raise ValueError('Immutable actual gamma differs')
    parent_frame=normal['weightParent'];check_frame(parent_frame)
    if set(parent_frame['slots'])!={'normWeight'} or parent_frame['program_fingerprint']!=req['requestSHA256'] or parent_frame['slots']['normWeight']['fingerprint']!=fingerprint(arrays['gamma']):raise ValueError('Gamma parent Frame differs')
    for record in normal_rows:
        name=record['state'];transaction(record,{'H1':arrays[name+'-H'],'normWeight':arrays['gamma']},{'Z':arrays[name+'-Z']},req['requestSHA256'])
        if record['spec']['numerical_contract']!=SOURCE_POLICY or record['input']['parent_frame_id']!=parent_frame['frame_id'] or record['inputField']!=arrays[name+'-H'] or record['outputField']!=arrays[name+'-Z']:raise ValueError('Source normalization boundary differs')
        if record['evidence']['metrics']!={'normalizationUnits':1,'originalLinears':0,'nominalMatrixMACs':0,'inputShape':[1,512,2560],'inputStride':[1310720,2560,1],'outputStride':[1310720,2560,1]}:raise ValueError('Original normalization physical geometry differs')
        method,key=name.rsplit('-',1)
        program_sha=None if method=='target' else catalog[method]['programSHA256']
        if record['programSHA256']!=program_sha or record['input']['metadata']!={'state':name,'programSHA256':program_sha} or record['output']['metadata']!=record['input']['metadata']:raise ValueError('Normalization named source/program metadata differs')
        if method=='target':
            if arrays[name+'-H']['payloadSHA256']!=source['cases'][key]['H1']['payloadSHA256']:raise ValueError('Native target input differs')
        if method in ('parent','accepted'):
            wanted=next(p for p in children['final',key]['predictions'] if p['method']==method)
            if arrays[name+'-H']['sha256']!=wanted['sha256']:raise ValueError('Consumed fresh serving input differs')
    for key in DISCOVERY:
        accepted=next(p for p in children['final',key]['predictions'] if p['method']=='accepted')
        if accepted['sha256']!=arrays[selected+'-'+key+'-H']['sha256']:raise ValueError('Accepted discovery full-byte replay differs')
    for slot,record in enumerate(proposals):
        names=set(shapes)-{'gradient','step'};inputs={k:arrays['fit'+str(slot)+'-'+k] for k in names};outputs={k:arrays['fit'+str(slot)+'-'+k] for k in ('gradient','step')}
        expected_shas={k:catalog.get('slot'+str(slot)+'-'+dose,{}).get('programSHA256') for k,dose in [('fullCandidateSHA','dose1'),('halfCandidateSHA','dosehalf')]}
        if record['candidateSHAs']!=expected_shas or record['inputFields']!=inputs or record['outputFields']!=outputs or record['parent']!=selections[slot]['parent']['id'] or record['sourceProgramSHA256']!=catalog[record['parent']]['programSHA256']:raise ValueError('Proposal declared numeric/program inputs differ')
        transaction(record,inputs,{**outputs,**expected_shas},req['requestSHA256'])
        if record['spec']['numerical_contract']!=POLICY or record['input']['metadata']!={'slot':slot,'programSHA256':record['sourceProgramSHA256'],'inputProgram':record['parent']}:raise ValueError('Proposal contract/parent differs')
        for future in (r for r in futures if r['slot']==slot):
            if future['sourceProgramSHA256']!=record['sourceProgramSHA256']:raise ValueError('Second-slot recomputation parent differs')
            if future['id'] in catalog:
                p,q=catalog[future['id']]['program'],catalog[record['parent']]['program'];other=1-slot
                if p['trees'][other]!=q['trees'][other] or p['dictionaries'][other]!=q['dictionaries'][other]:raise ValueError('Other slot changed during coordinate proposal')
    # Every raw dependent row and every pooled scalar, including regressions.
    domain={('case',k,m,p,i) for k in DISCOVERY+req['transfer'] for m in ('parent','accepted') for p in ('H','Z') for i in range(512)}|{('pair',l,r,m,p,i) for l,r in [('A','B'),('00','10'),('01','11')] for m in ('parent','accepted') for p in ('H','Z') for i in range(512)}
    raw={};groups=collections.defaultdict(list)
    with gzip.open(result/'observations.jsonl.gz','rt') as stream:
        for line in stream:
            r=json.loads(line);i=r['position']
            if type(i) is not int or not 0<=i<512:raise ValueError('Raw row position differs')
            if 'case' in r:
                key=('case',r['case'],r['method'],r['port'],i);mask=bool(packets[r['case']]['mask'][i]);labels=['all','active' if mask else 'padding']+(['learned'] if i>=2 else [])
            else:
                left,right=r['pair'];key=('pair',left,right,r['method'],r['port'],i);same=packets[left]['ids'][i]==packets[right]['ids'][i];mask=bool(packets[left]['mask'][i])
                if r['sameToken'] is not same:raise ValueError('Raw unchanged-token binding differs')
                labels=['all']+(['unchangedToken'] if same else [])+(['unchangedLearned','unchangedActive' if mask else 'unchangedPadding'] if same and i>=2 else [])
            if key not in domain or key in raw or r['active'] is not mask:raise ValueError('Raw row identity/domain differs')
            raw[key]=r
            for label in labels:groups[key[:-1]+(label,)].append(r)
    if set(raw)!=domain or report['observationRows']!=18432:raise ValueError('Complete raw row coverage differs')
    pooled_count=0
    for paired,rows in [(False,report['summaries']),(True,report['pairs'])]:
        for r in rows:
            key=('pair',*r['pair'],r['method'],r['port'],r['selection']) if paired else ('case',r['case'],r['method'],r['port'],r['selection']);items=groups.pop(key)
            if len(items)!=r['rows']:raise ValueError('Pooled population differs')
            pooled_count+=compare({k:r[k] for k in pooled(items)},pooled(items))
    if groups or len(report['summaries'])!=96 or len(report['pairs'])!=60:raise ValueError('Complete pooled domain differs')
    for method,claimed in [('parent',report['initialObjective']),('accepted',report['selectedDiscoveryObjective'])]:
        errors={k:[] for k in KEYS};native={k:[] for k in KEYS}
        for case in DISCOVERY:
            for active in (True,False):
                positions=[i for i in range(2,512) if bool(packets[case]['mask'][i]) is active]
                for port in ('H','Z'):
                    for i in positions:
                        r=raw['case',case,method,port,i];errors[port+'endpoint'].append(2560*r['errorRMS']**2/(8*len(positions)));native[port+'endpoint'].append(2560*r['normalizerRMS']**2/(8*len(positions)))
        for left,right in [('A','B'),('00','10')]:
            for active in (True,False):
                positions=[i for i in range(2,512) if packets[left]['ids'][i]==packets[right]['ids'][i] and bool(packets[left]['mask'][i]) is active]
                for port in ('H','Z'):
                    for i in positions:
                        r=raw['pair',left,right,method,port,i];errors[port+'response'].append(2560*r['errorRMS']**2/(4*len(positions)));native[port+'response'].append(2560*r['normalizerRMS']**2/(4*len(positions)))
        rebuilt=scalar_objective({k:math.fsum(v) for k,v in errors.items()},{k:math.fsum(v) for k,v in native.items()});scalar_count+=compare({k:claimed[k] for k in rebuilt},rebuilt)
    if report['normalizationUnits']!=len(normal_rows) or report['proposalFitAttempts']!=2 or report['candidateFuturesAttempted']!=4 or report['candidatePrograms']!=len(emitted) or report['completeCandidateFutures']!=len(eligible) or report['retainedNumericFileBytes']!=sum(x['bytes'] for x in arrays.values()):raise ValueError('Actual work/retention counts differ')
    load=report['gammaLoad']
    if load['materializedParameterBytes']!=5120 or load['materializedParameterNames']!=[req['gammaParameter']['name']] or load['authenticatedShardBytes']!=req['selectedParameterAsset']['bytes']:raise ValueError('Gamma-only load differs')
    witnesses=fresh_bytes(remote,npys);destination.mkdir(exist_ok=False)
    for p in result.iterdir():shutil.copyfile(p,destination/p.name)
    for p,name in [(root/'collection.json','collection.json'),(root/'scheduler-job.json','scheduler-job.json'),(Path(mechanics),'mechanics.json')]:shutil.copyfile(p,destination/name)
    publication={'schema':'tce-joint-writer-publication-v1','jobID':job['job_id'],'requestSHA256':req['requestSHA256'],'publisherSHA256':sha(__file__),
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in destination.iterdir()},'retainedNumericFields':len(arrays),'rawObservationRows':len(raw),'rebuiltObjectiveScalars':scalar_count,'rebuiltPooledScalars':pooled_count,'freshRemoteFiles':witnesses,
        'scope':'Guarded/source/metadata/complete scalar and typed ABI audit plus fresh full-byte/payload custody. Native normalization,gradients,paired solves,tree fitting,norms and offline candidate reconstruction remain source-qualified producer measurements;no independent vector replay or local tensor hydration. Fresh accepted serving is full-byte bound to evaluated H. Original normalization/gamma paid offline;no general zero-neural image claim.'}
    save(destination/'publication.json',publication);return publication


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('mechanics');parser.add_argument('destination');a=parser.parse_args();p=publish(a.root,a.mechanics,a.destination);print(json.dumps({k:v for k,v in p.items() if k not in ('files','freshRemoteFiles')}))
