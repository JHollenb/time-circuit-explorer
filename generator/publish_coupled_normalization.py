"""Independent scalar arithmetic and typed/byte custody; no vector replay."""
import argparse
import collections
import gzip
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
from contextual_writer_program import read,save,sha
from publish_contextual_writer import check_frame
from publish_contrast_writer import close,pooled
from publish_finite_consumer_crossover import check_transaction,fingerprint,validate_scalar_analysis
from coupled_normalization import STATES,EDGES,CONTRASTS,ANALYSIS_NAMES,SCORING,SOURCE_POLICY,EXPLANATORY_POLICY


def compare(actual,expected):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('Scalar summary field domain differs')
        return sum(compare(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):
        if len(actual)!=len(expected):raise ValueError('Scalar list domain differs')
        return sum(compare(a,b) for a,b in zip(actual,expected))
    close(actual,expected);return 1


def validate_sum(row,components,reference,width):
    n,d=row['normSquared'],row['innerProducts'];pairs=list(itertools.combinations(components,2));names={'target','closure',*components}
    if row['width']!=width or row['components']!=list(components) or set(n)!=names or set(d)!={a+':'+b for a,b in pairs}:raise ValueError('Signed scalar ABI differs')
    if any(not math.isfinite(v) or v<0 for v in n.values()) or any(not math.isfinite(v) for v in d.values()):raise ValueError('Nonfinite signed scalar primitive')
    close(row['referenceSquared'],reference);terms=[n[k] for k in components]+[2*d[a+':'+b] for a,b in pairs];total=math.fsum(terms)
    expected={'rms':{k:math.sqrt(v/width) for k,v in n.items()},'relativeToNative':{k:math.sqrt(v/reference) if reference else None for k,v in n.items()},
              'gramTerms':terms,'gramSum':total,'normSquaredMinusGram':n['target']-total,
              'componentCosines':{a+':'+b:d[a+':'+b]/math.sqrt(n[a]*n[b]) if n[a]*n[b] else None for a,b in pairs}}
    return sum(compare(row[k],v) for k,v in expected.items())


def validate_analysis(row,width,epsilon):
    from coupled_normalization_study import identity
    if row['width']!=width or row['scoring']!=SCORING or row['explanatoryPolicy']!=EXPLANATORY_POLICY or identity(row['edgeMap'])!=identity(EDGES) or identity(row['contrastMap'])!=identity(CONTRASTS):raise ValueError('Explanatory ABI differs')
    if set(row['states'])!=set(STATES) or set(row['edges'])!=set(EDGES) or set(row['factors'])!=set(CONTRASTS) or set(row['endpoints'])!={'p','q'}:raise ValueError('Complete analysis domain differs')
    count=validate_scalar_analysis(row['response'],width);reference=row['response']['normSquared']['N']
    for state in row['states'].values():
        if any(not math.isfinite(state[k]) or state[k]<0 for k in ('inputNormSquared','idealNormSquared','rhoNormSquared')):raise ValueError('State primitive differs')
        squared=state['inputNormSquared']/width+epsilon;count+=compare({k:state[k] for k in ('squaredRadius','radius','inverseRadius')},
            {'squaredRadius':squared,'radius':math.sqrt(squared),'inverseRadius':1/math.sqrt(squared)})
    for name,response in [('native','N'),('own','O'),('no','Cno'),('on','Con')]:
        item=row['edges'][name];close(item['normSquared']['target'],row['response']['normSquared'][response]);count+=validate_sum(item,('D','B','R','g'),reference,width)
    for name,item in row['factors'].items():
        close(item['normSquared']['target'],row['response']['normSquared'][name]);count+=validate_sum(item,('D','B','R','G','g'),reference,width)
    for item in row['endpoints'].values():
        p,t,e,ip=[item[k] for k in ('predictionSquared','targetSquared','errorSquared','innerProduct')]
        if any(not math.isfinite(v) for v in (p,t,e,ip)) or min(p,t,e)<0:raise ValueError('Endpoint primitive differs')
        count+=compare({k:item[k] for k in ('errorRMS','normalizerRMS','relativeRMS','predictionRMS','targetRMS','cosine')},
            {'errorRMS':math.sqrt(e/width),'normalizerRMS':math.sqrt(t/width),'relativeRMS':math.sqrt(e/t) if t else None,
             'predictionRMS':math.sqrt(p/width),'targetRMS':math.sqrt(t/width),'cosine':ip/math.sqrt(p*t) if p*t else None})
    return count


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
   data=f.read()
   if len(data)!=math.prod(x['shape'])*{'<u2':2,'<f8':8}[x['dtype']] or hashlib.sha256(data).hexdigest()!=x['payloadSHA256']:raise ValueError('NPY actual payload differs')
 rows.append({'path':name,'bytes':s.st_size,'sha256':item['sha256']})
print(json.dumps(rows))
''' % (json.dumps(remote),json.dumps(npys))
    return json.loads(subprocess.run(['ssh','-T','beast','python3','-'],input=code,text=True,capture_output=True,check=True,timeout=120).stdout)


def publish(root,mechanics,destination):
    from mrun.client.api import Api
    from coupled_normalization_study import OWNED,PARENTS,WORK,verify,identity,FAMILY_SHA,PARENT_PUBLICATION_SHA
    from coupled_normalization_worker import BRANCHES
    root=Path(root).resolve();result=root/'result';destination=Path(destination).resolve();req=read(root/'request.json');verify(root/'payload',req)
    for name in OWNED:
        if sha(Path(__file__).parent/name)!=req['sourceHashes'][name]:raise ValueError('Current executed source differs')
    tests=read(mechanics)
    if (tests['testsRun'],tests['failures'],tests['errors'])!=(7,0,0) or set(tests['sourceHashes'])!=set(OWNED):raise ValueError('Complete seven-test source-bound mechanics absent')
    for name,h in tests['sourceHashes'].items():
        if req['sourceHashes'][name]!=h:raise ValueError('Tested source differs')
    job=read(root/'scheduler-job.json');live=Api().json('GET','/api/jobs/'+job['job_id']);collection=read(root/'collection.json');custody=job['payload_custody'];phases=[custody[k] for k in ('declared','sealed','executed')]
    if live['state']!='succeeded' or live['result']['returncode']!=0 or live['result']!=job['result'] or collection['result']!=job['result'] or live['payload_custody']!=custody or not custody['required'] or not live.get('admission'):raise ValueError('Successful guarded result differs')
    if len({(x['sha256'],x['size_bytes']) for x in phases})!=1 or phases[-1]['host']!='beast' or collection['custody']!=custody or collection['jobID']!=job['job_id']:raise ValueError('Collection/execution custody differs')
    if any(x['config']['requestSHA256']!=req['requestSHA256'] for x in (job,live)):raise ValueError('Scheduler request differs')
    actual={str(p.relative_to(result)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in result.rglob('*') if p.is_file()}
    required=set(req['retainedMetadata'])|{'typed-normalization.json','typed-analysis.json','analysis.json','array-manifest.json','observations.jsonl.gz','analysis-seal.json','report.json'}
    if actual!=collection['inventory']['files'] or set(actual)!=required:raise ValueError('Complete collection inventory differs')
    for name in req['retainedMetadata']:
        if name=='request.json':
            if read(result/name)!=req:raise ValueError('Retained request differs')
        elif sha(result/name)!=req['sourceHashes'][name]:raise ValueError('Source metadata differs')
    parent=read(result/'parent-publication.json');previous=read(result/'parent-request.json');old=Api().json('GET','/api/jobs/'+req['parent']['jobID'])
    if sha(result/'parent-publication.json')!=PARENT_PUBLICATION_SHA or req['parent']['publicationSHA256']!=PARENT_PUBLICATION_SHA or old['state']!='succeeded' or old['payload_custody']!=req['parent']['custody'] or parent['jobID']!=old['job_id']:raise ValueError('Parent107 authority differs')
    for name in PARENTS:
        if name!='publication.json' and sha(result/('parent-'+name))!=parent['files'][name]['sha256']:raise ValueError('Parent107 member differs')
    old_arrays=read(result/'parent-array-manifest.json');expected_endpoints={**previous['endpoints'],**{k:old_arrays[k] for k in BRANCHES}}
    if req['endpoints']!=expected_endpoints or set(expected_endpoints)!=set(STATES) or req['gammaParameter']!=next(x for x in previous['parameters'] if x['name']=='model.layers.1.input_layernorm.weight') or req['selectedParameterAsset'] not in previous['selectedParameterAssets']:raise ValueError('Actual historical endpoints/gamma differ')
    if sha(result/'family-plan.md')!=FAMILY_SHA or req['physicalWork']!=WORK or req['position']!=9 or req['cases']!=['00','10'] or req['states']!=list(STATES) or identity(req['edges'])!=identity(EDGES) or identity(req['contrasts'])!=identity(CONTRASTS):raise ValueError('Fixed contract differs')
    report=read(result/'report.json');arrays=read(result/'array-manifest.json');normal=read(result/'typed-normalization.json');analysis=read(result/'analysis.json');typed=read(result/'typed-analysis.json')
    for key,value in [('requestSHA256',req['requestSHA256']),('physicalWork',WORK),('sourcePolicy',SOURCE_POLICY),('explanatoryPolicy',EXPLANATORY_POLICY),('scoring',SCORING),('numericalProfile',req['numericalProfile']),('torchVersion',req['torchVersion'])]:
        if report[key]!=value:raise ValueError('Executed contract/profile differs: '+key)
    if any(req[k]!=previous[k] for k in ('numericalProfile','torchVersion','rmsNormEpsilon')):raise ValueError('Historical runtime profile differs')
    if report['analysisSealSHA256']!=sha(result/'analysis-seal.json'):raise ValueError('Analysis seal differs')
    seal=read(result/'analysis-seal.json')
    if set(seal)!={'typed-normalization.json','typed-analysis.json','analysis.json','array-manifest.json','observations.jsonl.gz'}:raise ValueError('Complete sealed domain differs')
    for name,h in seal.items():
        if sha(result/name)!=h:raise ValueError('Sealed member differs')
    expected={'Z-'+k:([1,512,2560],'torch.bfloat16','<u2','raw') for k in STATES};expected['gamma']=([2560],'torch.bfloat16','<u2','raw')
    expected.update({k:([2560],'torch.float64','<f8','npy') for k in ANALYSIS_NAMES})
    if set(arrays)!=set(expected) or len({v['path'] for v in arrays.values()})!=86:raise ValueError('Complete unique retained field domain differs')
    cwd=job['result']['diagnostics']['cwd']
    for key,abi in expected.items():
        v=arrays[key]
        if (v['shape'],v['dtype'],v['storedDtype'],v['format'])!=abi or not v['path'].startswith(cwd+'/arrays/'):raise ValueError('Retained ABI/root differs')
        if v['format']=='raw' and (v['bytes']!=math.prod(v['shape'])*2 or v['sha256']!=v['payloadSHA256']):raise ValueError('Raw file/payload join differs')
    if arrays['gamma']['payloadSHA256']!=req['gammaParameter']['payloadSHA256'] or report['retainedNumericFileBytes']!=sum(v['bytes'] for v in arrays.values()):raise ValueError('Actual gamma/retained cost differs')
    load=report['gammaLoad']
    if load['materializedParameterBytes']!=5120 or load['materializedParameterNames']!=[req['gammaParameter']['name']] or load['authenticatedShardBytes']!=req['selectedParameterAsset']['bytes']:raise ValueError('Gamma-only loading cost differs')
    parent_frame=normal['weightParent'];check_frame(parent_frame)
    if set(parent_frame['slots'])!={'normWeight'} or parent_frame['program_fingerprint']!=req['requestSHA256'] or parent_frame['slots']['normWeight']['fingerprint']!=fingerprint(arrays['gamma']) or normal['parameterPayloadBefore']!=arrays['gamma']['payloadSHA256'] or normal['parameterPayloadAfter']!=normal['parameterPayloadBefore']:raise ValueError('Actual immutable gamma parent differs')
    if len(normal['rows'])!=6 or {r['state'] for r in normal['rows']}!=set(STATES):raise ValueError('Complete normalization Act domain differs')
    for record in normal['rows']:
        key=record['state'];inputs={'normWeight':arrays['gamma'],'H1':{**req['endpoints'][key],'dtype':'torch.bfloat16'}}
        check_transaction(record,inputs,{'Z':arrays['Z-'+key]},req['requestSHA256'])
        if record['input']['parent_frame_id']!=parent_frame['frame_id'] or record['spec']['numerical_contract']!=SOURCE_POLICY:raise ValueError('Original normalization boundary differs')
        metrics=record['evidence']['metrics']
        if metrics!={'normalizationUnits':1,'rows':512,'coordinates':1310720,'originalLinears':0,'nominalMatrixMACs':0,'QKVMatricesMaterialized':0,
                     'inputShape':[1,512,2560],'inputStride':[1310720,2560,1],'outputStride':[1310720,2560,1],'dtype':'torch.bfloat16'}:raise ValueError('Executed normalization geometry differs')
    inputs={**{'H-'+k:{**v,'dtype':'torch.bfloat16'} for k,v in req['endpoints'].items()},**{'Z-'+k:arrays['Z-'+k] for k in STATES},'gamma':arrays['gamma']}
    check_transaction(typed,inputs,{k:arrays[k] for k in ANALYSIS_NAMES},req['requestSHA256'])
    if typed['spec']['numerical_contract']!=SCORING+' '+EXPLANATORY_POLICY or typed['evidence']['metrics']!={'position':9,'width':2560,'originalLinears':0,'retainedRowFields':79}:raise ValueError('Executed analysis boundary differs')
    scalar_count=validate_analysis(analysis,2560,req['rmsNormEpsilon']);domain={(b,i) for b in BRANCHES for i in range(512)};raw={};groups=collections.defaultdict(list)
    with gzip.open(result/'observations.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line);i=row['position'];key=(row['branch'],i)
            if type(i) is not int or key not in domain or key in raw or row['intervened'] is not (i==9):raise ValueError('Complete comparison row domain differs')
            raw[key]=row
            for selection in ('all','intervened' if i==9 else 'unmodified'):groups[(key[0],selection)].append(row)
    if set(raw)!=domain or report['comparisonRows']!=1024 or report['unmodifiedRowControlRows']!=1022:raise ValueError('Comparison coverage differs')
    pooled_count=0
    for row in report['summaries']:
        rows=groups.pop((row['branch'],row['selection']))
        if row['rows']!=len(rows):raise ValueError('Pooled coverage differs')
        pooled_count+=compare({k:row[k] for k in pooled(rows)},pooled(rows))
    if groups or len(report['controls'])!=2 or {c['branch'] for c in report['controls']}!=set(BRANCHES):raise ValueError('Complete pooled/control domain differs')
    for row in report['controls']:
        if row['unmodifiedRows']!=511 or row['unmodifiedRowsChanged']!=sum(raw[row['branch'],i]['errorRMS']>0 for i in range(512) if i!=9):raise ValueError('Unmodified-row diagnostic differs')
    remote={v['path']:{k:v[k] for k in ('sha256','bytes')} for v in [*req['endpoints'].values(),req['selectedParameterAsset'],*arrays.values()]}
    npys={v['path']:{'payloadSHA256':v['payloadSHA256'],'shape':v['shape'],'dtype':v['storedDtype']} for v in [*req['endpoints'].values(),*arrays.values()] if v['format']=='npy'}
    witnesses=fresh_bytes(remote,npys)
    destination.mkdir(exist_ok=False)
    for p in result.iterdir():shutil.copyfile(p,destination/p.name)
    for p,name in [(root/'collection.json','collection.json'),(root/'scheduler-job.json','scheduler-job.json'),(Path(mechanics),'mechanics.json')]:shutil.copyfile(p,destination/name)
    publication={'schema':'tce-coupled-normalization-publication-v1','jobID':job['job_id'],'requestSHA256':req['requestSHA256'],'publisherSHA256':sha(__file__),
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in destination.iterdir()},'retainedNumericFields':86,'rawComparisonRows':len(raw),
        'rebuiltAnalysisScalars':scalar_count,'rebuiltPooledScalars':pooled_count,'freshRemoteFiles':witnesses,
        'scope':'Complete guarded/source/metadata/scalar/typed ABI and fresh full-byte/payload audit. Vector norms,inner products and reconstructed normalized fields remain source-qualified producer measurements;no independent numerical vector replay or local tensor hydration. This is original normalization arithmetic,not zero-neural serving.'}
    save(destination/'publication.json',publication);return publication


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('mechanics');parser.add_argument('destination');a=parser.parse_args();print(json.dumps(publish(a.root,a.mechanics,a.destination)))
