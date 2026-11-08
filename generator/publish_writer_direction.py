"""Compact scalar/typed/byte audit without local scientific tensor hydration."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
from template_cofactor_program import read,save,sha
from writer_direction_study import OWNED,PARENTS,PUB_SHA,FAMILY_SHA,SURVEY_SHA,WORK,CASES,CANDIDATES,identity,verify
from writer_direction_math import columns,TERMS,DIRECTIONS,GRADIENTS,POLICY


def fresh_bytes(expected,npys,joins):
    code='''import ast,hashlib,json,math,stat,struct
from pathlib import Path
expected=json.loads(%r);npys=json.loads(%r);joins=json.loads(%r);rows=[]
for name,item in expected.items():
 p=Path(name);s=p.lstat()
 if not stat.S_ISREG(s.st_mode) or not name.startswith('/mnt/hdd8tb/mrun-work/') or s.st_size!=item['bytes']:raise ValueError('Unsafe retained file')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 if h.hexdigest()!=item['sha256']:raise ValueError('Fresh retained bytes differ')
 if name in npys:
  with p.open('rb') as f:
   if f.read(8)!=b'\\x93NUMPY\\x01\\x00':raise ValueError('NPY version differs')
   n=struct.unpack('<H',f.read(2))[0];header=ast.literal_eval(f.read(n).decode('latin1'));x=npys[name]
   if header!={'descr':x['storedDtype'],'fortran_order':False,'shape':tuple(x['shape'])}:raise ValueError('NPY ABI differs')
   h=hashlib.sha256();size=0
   for block in iter(lambda:f.read(8<<20),b''):h.update(block);size+=len(block)
   if size!=math.prod(x['shape'])*{'<f8':8,'<i8':8}[x['storedDtype']] or h.hexdigest()!=x['payloadSHA256']:raise ValueError('NPY payload differs')
 rows.append({'path':name,**item})
for name,item in joins.items():
 h=hashlib.sha256()
 for source in item['files']:
  with Path(source).open('rb') as f:
   f.seek(10240)
   for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 if h.hexdigest()!=item['payloadSHA256']:raise ValueError('Actual BF16 input row byte-join differs')
print(json.dumps(rows))
''' % (json.dumps(expected),json.dumps(npys),json.dumps(joins))
    return json.loads(subprocess.run(['ssh','-T','beast','python3','-'],input=code,text=True,capture_output=True,check=True,timeout=120).stdout)


def publish(root,mechanics,destination):
    from mrun.client.api import Api
    root=Path(root).resolve();dest=Path(destination).resolve();req=read(root/'request.json');verify(root/'payload',req);tests=read(mechanics)
    if (tests['testsRun'],tests['failures'],tests['errors'])!=(9,0,0) or set(tests['sourceHashes'])!=set(OWNED):raise ValueError('Nine source-bound mechanics absent')
    for name in OWNED:
        if sha(Path(__file__).parent/name)!=req['sourceHashes'][name] or tests['sourceHashes'][name]!=req['sourceHashes'][name]:raise ValueError('Tested/current/staged source differs')
    job=read(root/'scheduler-job.json');live=Api().json('GET','/api/jobs/'+job['job_id']);collection=read(root/'collection.json');custody=job['payload_custody'];phases=[custody[k] for k in ('declared','sealed','executed')]
    if live['state']!='succeeded' or live['result']['returncode']!=0 or live['result']!=job['result'] or live['payload_custody']!=custody or not custody['required'] or not live.get('admission') or len({(x['sha256'],x['size_bytes']) for x in phases})!=1 or phases[-1]['host']!='beast':raise ValueError('Successful guarded112 custody differs')
    if collection['custody']!=custody or collection['result']!=job['result'] or collection['jobID']!=job['job_id'] or any(x['config']['requestSHA256']!=req['requestSHA256'] for x in (job,live)):raise ValueError('Request/collection/actual result differs')
    result=root/'result';actual={str(p.relative_to(result)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in result.rglob('*') if p.is_file()}
    members={'analysis.json','typed-analysis.json','array-manifest.json','observations.jsonl.gz','analysis-seal.json','report.json'}
    if actual!=collection['inventory']['files'] or set(actual)!=set(req['retainedMetadata'])|members:raise ValueError('Complete compact result differs')
    for name in req['retainedMetadata']:
        if name=='request.json':
            if read(result/name)!=req:raise ValueError('Collected request differs')
        elif sha(result/name)!=req['sourceHashes'][name]:raise ValueError('Collected frozen metadata differs')
    parent=read(result/'parent-publication.json');old=read(result/'parent-scheduler-job.json');current_old=Api().json('GET','/api/jobs/'+req['parent']['jobID'])
    if sha(result/'parent-publication.json')!=PUB_SHA or old['job_id']!=parent['jobID'] or current_old['state']!='succeeded' or current_old['result']!=old['result'] or current_old['payload_custody']!=req['parent']['custody']:raise ValueError('Qualified110 authority differs')
    for name in PARENTS:
        if name!='publication.json' and sha(result/('parent-'+name))!=parent['files'][name]['sha256']:raise ValueError('Qualified110 member differs')
    if sha(result/'family-plan.md')!=FAMILY_SHA or sha(result/'review111-survey.json')!=SURVEY_SHA or req['physicalWork']!=WORK:raise ValueError('Selected111 contract differs')
    report=read(result/'report.json');analysis=read(result/'analysis.json');typed=read(result/'typed-analysis.json');arrays=read(result/'array-manifest.json');seal=read(result/'analysis-seal.json')
    if report['physicalWork']!=WORK or report['policy']!=POLICY or report['requestSHA256']!=req['requestSHA256'] or report['actualWork']['directionCoordinateProducts']!=WORK['directionCoordinateProducts'] or report['actualWork']['gradientReconstructionNormProducts']!=WORK['gradientReconstructionNormProducts']:raise ValueError('Actual work differs')
    if set(seal)!=members-{'report.json','analysis-seal.json'} or any(sha(result/n)!=h for n,h in seal.items()) or report['analysisSealSHA256']!=sha(result/'analysis-seal.json'):raise ValueError('Analysis seal differs')
    before,after,evidence,spec=[typed[k] for k in ('input','output','evidence','spec')]
    shapes={k:[2040,2560] for k in ('h','z','target','zt','gradient','step','oldSlot0','oldSlot1')}
    shapes.update({'features':[2040,155],'weights':[2040],'edges':[1017,2],'pairWeights':[1017],'gamma':[2560],'scaleVector':[4]})
    for name in CANDIDATES:
        shapes[name+'-write']=[2040,2560];shapes[name+'-H']=[2040,2560]
    if set(typed['inputABIs'])!=set(shapes):raise ValueError('Exact22-input domain differs')
    for name,shape in shapes.items():
        expected_dtype='torch.int64' if name=='edges' else 'torch.bfloat16' if name.endswith('-H') else 'torch.float64'
        if typed['inputABIs'][name]['shape']!=shape or typed['inputABIs'][name]['dtype']!=expected_dtype:raise ValueError('Exact input shape/dtype differs')
    write_abis={n:{'shape':[2040,2 if n=='gradient-reconstruction' else 52],'dtype':'float64'} for n in arrays}
    if spec['identity']!='writer_direction.saved_futures:v1' or spec['name']!='writer_direction.saved_futures' or spec['abi_version']!='v1' or spec['numerical_contract']!=POLICY or spec['write_abis']!=write_abis:raise ValueError('Named Act/numerical/write contract differs')
    for frame in (before,after):
        if identity({k:frame[k] for k in ('schema','slots','parent_frame_id','act','implementation','program_fingerprint','metadata')})!=frame['frame_id'] or frame['program_fingerprint']!=req['requestSHA256']:raise ValueError('Canonical frame identity differs')
    if after['parent_frame_id']!=before['frame_id'] or after['act']!=spec['identity'] or evidence['input_frame_id']!=before['frame_id'] or evidence['output_frame_id']!=after['frame_id'] or evidence['act']!=spec['identity'] or set(spec['reads'])!=set(typed['inputABIs']) or set(spec['writes'])!=set(arrays):raise ValueError('Canonical transaction ancestry/domain differs')
    if set(before['slots'])!=set(typed['inputABIs']) or set(after['slots'])!=set(before['slots'])|set(arrays) or typed['inputPayloadBefore']!=typed['inputPayloadAfter']:raise ValueError('Actual immutable typed input differs')
    for name,item in typed['inputABIs'].items():
        expected=identity({'dtype':item['dtype'],'shape':item['shape'],'bytes':item['payloadSHA256']})
        if before['slots'][name]['fingerprint']!=expected or after['slots'][name]!=before['slots'][name] or typed['inputPayloadBefore'][name]!=item['payloadSHA256']:raise ValueError('Input ABI/fingerprint differs')
    direct={k[5:]:v for k,v in req['fields'].items() if k.startswith('fit0-') and k!='fit0-oldSlot'}
    direct.update({'oldSlot'+str(s):req['fields']['fit'+str(s)+'-oldSlot'] for s in (0,1)})
    for name,item in direct.items():
        if typed['inputABIs'][name]!={k:item[k] for k in ('shape','dtype','payloadSHA256')}:raise ValueError('Direct original typed-field input join differs')
    for name,item in arrays.items():
        shape=write_abis[name]['shape']
        if item['shape']!=shape or item['dtype']!='torch.float64' or item['storedDtype']!='<f8' or item['format']!='npy' or item['bytes']!=128+math.prod(shape)*8:raise ValueError('Exact five-output ABI/bytes differ')
        if after['slots'][name]['fingerprint']!=identity({'dtype':item['dtype'],'shape':item['shape'],'bytes':item['payloadSHA256']}):raise ValueError('Output ABI/fingerprint differs')
    if set(arrays)!={n+'-products' for n in CANDIDATES}|{'gradient-reconstruction'} or len(analysis['positions'])!=2040 or analysis['columns']!=columns():raise ValueError('Output field/column domain differs')
    rows={name:[] for name in CANDIDATES};payloads={name:hashlib.sha256() for name in CANDIDATES};positions=analysis['positions'];packets={x['id']:x['packet'] for x in read(result/'parent-final-serving.json')['rows']}
    expected_positions=[]
    for at,case in enumerate(CASES):
        other=CASES[at+1 if at%2==0 else at-1]
        for position in range(2,512):expected_positions.append({'case':case,'position':position,'active':bool(packets[case]['mask'][position]),'paired':packets[case]['ids'][position]==packets[other]['ids'][position]})
    if positions!=expected_positions:raise ValueError('All discovery history/position/strata metadata differ')
    groups={'all':list(range(2040)),'active':[i for i,r in enumerate(positions) if r['active']],'padding':[i for i,r in enumerate(positions) if not r['active']]}
    for case in CASES:
        for label,active in [('all',None),('active',True),('padding',False)]:groups[case+':'+label]=[i for i,r in enumerate(positions) if r['case']==case and (active is None or r['active']==active)]
    if analysis['groups']!=groups:raise ValueError('Complete dependent group membership differs')
    with gzip.open(result/'observations.jsonl.gz','rt') as stream:
        for line in stream:
            r=json.loads(line);name=r.pop('candidate');values=r.pop('values');index=len(rows[name])
            if r!=positions[index] or len(values)!=52 or any(not math.isfinite(v) for v in values):raise ValueError('Raw signed row domain differs')
            payloads[name].update(struct.pack('<52d',*values))
            rows[name].append(values)
    if any(len(v)!=2040 for v in rows.values()) or report['observationRows']!=8160:raise ValueError('Complete raw direction rows absent')
    if any(payloads[name].hexdigest()!=arrays[name+'-products']['payloadSHA256'] for name in CANDIDATES):raise ValueError('Public scalar rows differ from actual typed numerical output bytes')
    rebuilt=[]
    for summary in analysis['summaries']:
        name,group=summary['candidate'],summary['group'];indices=analysis['groups'][group]
        if summary['rows']!=len(indices):raise ValueError('Group population differs')
        if set(summary['values'])!=set(columns()) or any(not math.isfinite(x) for x in [*summary['values'].values(),*summary['localModelSlopes'].values()]):raise ValueError('Nonfinite or incomplete scalar summary')
        for at,col in enumerate(columns()):
            expected=max(rows[name][i][at] for i in indices) if col=='vectorClosureMaximumAbsolute' else math.fsum(rows[name][i][at] for i in indices)
            if not math.isfinite(expected) or not math.isfinite(summary['values'][col]-expected):raise ValueError('Nonfinite scalar reaggregation')
            rebuilt.append({'candidate':name,'group':group,'column':col,'producer':summary['values'][col],'scalarReaggregation':expected,'difference':summary['values'][col]-expected})
        if any(summary['localModelSlopes'][g+'.'+d]!=summary['values'][g+'.'+d]/2 for g in GRADIENTS for d in DIRECTIONS):raise ValueError('Declared half-factor differs')
    if len(analysis['summaries'])!=4*15 or {(x['candidate'],x['group']) for x in analysis['summaries']}!={(c,g) for c in CANDIDATES for g in groups}:raise ValueError('Full group coverage differs')
    future={x['id']:x for x in read(result/'parent-futures.json')};prior=read(result/'parent-report.json')['initialObjective']
    if len(analysis['objectives'])!=4 or {x['candidate'] for x in analysis['objectives']}!=set(CANDIDATES):raise ValueError('Four inherited objectives absent')
    for row in analysis['objectives']:
        f=future[row['candidate']]
        if row['parent']!=prior or row['finiteScoreChange']!=f['score']-prior['score'] or row['finiteTermChanges']!={k:f['terms'][k]-prior['terms'][k] for k in TERMS}:raise ValueError('Actual finite objective comparison differs')
    remote=dict(req['remoteInputs']);npys={x['path']:x for x in req['fields'].values() if x['format']=='npy'}
    for item in arrays.values():remote[item['path']]={k:item[k] for k in ('sha256','bytes')};npys[item['path']]=item
    joins={name:{'files':[req['fields'][name+'-'+case+'-H']['path'] for case in CASES],'payloadSHA256':typed['inputPayloadBefore'][name+'-H']} for name in CANDIDATES}
    fresh=fresh_bytes(remote,npys,joins)
    if dest.exists():raise ValueError('Publication destination already exists')
    dest.mkdir(parents=True)
    for name in actual:shutil.copyfile(result/name,dest/name)
    for src,name in [(root/'scheduler-job.json','scheduler-job.json'),(root/'collection.json','collection.json'),(Path(mechanics),'mechanics.json')]:shutil.copyfile(src,dest/name)
    save(dest/'scalar-reaggregation.json',{'scope':'Independent public scalar-row reaggregation,not vector dot replay. Different FP64 reduction orders remain diagnostics;no scientific threshold.','rows':rebuilt,'maximumAbsoluteDifference':max(abs(x['difference']) for x in rebuilt)})
    files={str(p.relative_to(dest)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in dest.rglob('*') if p.is_file()}
    publication={'schema':'tce-writer-direction-publication-v1','jobID':job['job_id'],'requestSHA256':req['requestSHA256'],'publisherSHA256':sha(__file__),'files':files,'freshRemoteFiles':fresh,'admittedInputByteJoins':joins,'rawOutputPayloadJoins':{n:payloads[n].hexdigest() for n in CANDIDATES},'observationRows':8160,'scalarReaggregations':len(rebuilt),'retainedNumericFields':len(arrays),
        'scope':'Complete compact scalar/typed/byte audit. Original vector norms/products and routed gathers remain source-qualified producer measurements;not independent vector replay. All directions,strata,regrouping residuals and inherited finite objectives retained. Scientific arrays remain on Beast;goal unresolved.'}
    save(dest/'publication.json',publication);return publication


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('mechanics');parser.add_argument('destination');args=parser.parse_args()
    result=publish(args.root,args.mechanics,args.destination);print(json.dumps({k:v for k,v in result.items() if k not in ('files','freshRemoteFiles')}))
