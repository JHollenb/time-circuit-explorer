"""One saved-only CPU lease for the four completed110 writer futures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from template_cofactor_program import read,save,sha

HERE=Path(__file__).resolve().parent
OWNED=['writer_direction_math.py','writer_direction_study.py','writer_direction_worker.py','publish_writer_direction.py','test_writer_direction.py',
       'contextual_writer_program.py','template_cofactor_program.py','contextual_writer_saturn_init.py','response_study.py','run.py','worker.py']
PARENTS=['publication.json','request.json','scheduler-job.json','report.json','array-manifest.json','typed-proposals.json','typed-normalization.json','programs.json','selection.json','futures.json','final-serving.json']
PUB_SHA='4f15854368616a14910558de1a7a003f9da1e35d2880b45259c9940ea4b7fde5'
FAMILY_SHA='eea40e41786e8a000ed998e41dd19250f5e67cbd6c9d3c5b75b2c96d11059c34'
SURVEY_SHA='3006cad0f01e7aeaa4ddb878afc52d0fe4bbca276c25841fe905cc37e839b291'
FITS=('h','z','target','zt','oldSlot','features','weights','edges','pairWeights','gamma','scaleVector','gradient','step')
CANDIDATES=('slot0-dose1','slot0-dosehalf','slot1-dose1','slot1-dosehalf')
CASES=('A','B','00','10')
WORK={'savedFutures':4,'discoveryHistories':4,'learnedRows':2040,'coordinates':2560,'features':155,'pairedEdges':1017,'singleRows':6,
      'analysisActs':1,'newFits':0,'newDoses':0,'originalNormalizations':0,'originalLinears':0,'newNeuralEvaluations':0,'images':0,
      'directionCoordinateProducts':4*31*2040*2560,'gradientReconstructionNormProducts':2040*2560,'outputRowFields':5,'observationRows':4*2040}


def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify(root,request):
    if identity({k:v for k,v in request.items() if k!='requestSHA256'})!=request['requestSHA256']:raise ValueError('Request identity differs')
    for name,h in request['sourceHashes'].items():
        if sha(Path(root)/name)!=h:raise ValueError('Frozen source differs: '+name)


def stage(root):
    from mrun.client.api import Api
    from mdb.job_support import one_lease_scheduling_contract
    base=HERE.parent/'docs/results/joint-writer';pub=read(base/'publication.json');old=read(base/'scheduler-job.json');live=Api().json('GET','/api/jobs/'+old['job_id'])
    if sha(base/'publication.json')!=PUB_SHA or pub['jobID']!=old['job_id'] or live['state']!='succeeded' or live['result']!=old['result'] or live['payload_custody']!=old['payload_custody'] or not live.get('admission'):raise ValueError('Qualified110 authority differs')
    custody=old['payload_custody'];phases=[custody[k] for k in ('declared','sealed','executed')]
    if not custody['required'] or len({(x['sha256'],x['size_bytes']) for x in phases})!=1 or phases[-1]['host']!='beast':raise ValueError('Qualified110 payload differs')
    for name,item in pub['files'].items():
        if sha(base/name)!=item['sha256'] or (base/name).stat().st_size!=item['bytes']:raise ValueError('Published110 member differs')
    report=read(base/'report.json');catalog=read(base/'programs.json');fits=read(base/'typed-proposals.json');futures=read(base/'futures.json');arrays=read(base/'array-manifest.json')
    if report['selected']!='parent' or catalog['accepted']['programSHA256']!=catalog['parent']['programSHA256'] or [x['id'] for x in futures]!=list(CANDIDATES) or any(x['status']!='complete' for x in futures) or len(fits)!=2:raise ValueError('Actual four complete parent-retained futures differ')
    fields={}
    for slot,record in enumerate(fits):
        if record['slot']!=slot or record['parent']!='parent' or record['sourceProgramSHA256']!=catalog['parent']['programSHA256']:raise ValueError('Actual slot parent differs')
        for name in FITS:
            item=(record['outputFields'] if name in ('gradient','step') else record['inputFields'])[name]
            if item!=arrays['fit'+str(slot)+'-'+name]:raise ValueError('Typed fit/array join differs')
            fields['fit'+str(slot)+'-'+name]=item
    for name in FITS:
        if name!='oldSlot' and any(fields['fit0-'+name][k]!=fields['fit1-'+name][k] for k in ('payloadSHA256','shape','storedDtype')):raise ValueError('Actual shared110 input differs')
    for state in ('parent',*CANDIDATES):
        for case in CASES:
            for port in (('H','Z') if state=='parent' else ('H',)):
                fields[state+'-'+case+'-'+port]=arrays[state+'-'+case+'-'+port]
    inventory={r['path']:r for r in pub['freshRemoteFiles']};remote={}
    def need(path,digest,size=None):
        row=inventory[path]
        if row['sha256']!=digest or size is not None and row['bytes']!=size:raise ValueError('Input lacks110 byte custody')
        remote[path]={'sha256':digest,'bytes':row['bytes']}
    for item in fields.values():need(item['path'],item['sha256'],item['bytes'])
    for state in ('parent',*CANDIDATES):
        program=catalog[state];folder=Path(program['directory']);need(str(folder/'program.json'),program['programSHA256'])
        for d in program['program']['dictionaries']:need(str(folder/d['path']),d['sha256'],d['bytes'])
    family=HERE.parent/'docs/plans/writer-proposal-direction.md';survey=HERE.parent/'docs/results/research-review-cycle111/advisor-survey.json'
    if sha(family)!=FAMILY_SHA or sha(survey)!=SURVEY_SHA:raise ValueError('Selected111 authority differs')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for name in OWNED:shutil.copyfile(HERE/name,payload/name)
    exported=payload/'saturn';exported.mkdir()
    for name in ('titan.py','tensor_hash.py'):shutil.copyfile(HERE.parents[1]/'saturn/src/saturn'/name,exported/name)
    shutil.copyfile(HERE/'contextual_writer_saturn_init.py',exported/'__init__.py')
    for name in PARENTS:shutil.copyfile(base/name,payload/('parent-'+name))
    for src,name in [(family,'family-plan.md'),(survey,'review111-survey.json'),(HERE.parent/'docs/plans/writer-direction-assay.md','frozen-plan.md')]:shutil.copyfile(src,payload/name)
    retained=['request.json','family-plan.md','review111-survey.json','frozen-plan.md']+['parent-'+x for x in PARENTS]
    scheduling=one_lease_scheduling_contract(model_loads=0,model_free=True,logical_branches=4,logical_operations=1,logical_checkpoint_replays=0,
        extra={'task_family':'analysis','branchIsolation':'One CPU saved-field analysis Act over four immutable110 futures;identical input payloads loaded once,no model or child leases.'})
    req={'schema':'tce-writer-direction-v1','parent':{'jobID':old['job_id'],'custody':custody,'publicationSHA256':PUB_SHA,'requestSHA256':read(base/'request.json')['requestSHA256']},
         'cases':list(CASES),'candidates':list(CANDIDATES),'fields':fields,'remoteInputs':remote,'physicalWork':WORK,'retainedMetadata':retained,
         'rmsNormEpsilon':read(base/'request.json')['rmsNormEpsilon'],'sourceHashes':{str(p.relative_to(payload)):sha(p) for p in payload.rglob('*') if p.is_file()},
         'device':'cpu','backend':'NumPy-FP64-saved-field-products;Torch-CPU-typed-custody-only','dtype':'saved-BF16-FP32-FP64;FP64-explanation','needs':{'cpu':True},'scheduling_contract':scheduling,'timeoutSeconds':900,
         'protocol':'Four already complete110 futures only. Reconstruct exact routed FP32 writes from saved features;compare saved finite H. Global frozen measures/scales;original ordered pair errors. No new finite serving,norm,fit,dose,QKV,prompt,image,precision sweep,retry or expansion.'}
    req['requestSHA256']=identity(req);save(root/'request.json',req);save(payload/'request.json',req)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':sorted([x.name for x in payload.iterdir()]+['preflight.v1.json'])})
    return {'root':str(root),'requestSHA256':req['requestSHA256'],'remoteFiles':len(remote),'physicalWork':WORK}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();req=read(root/'request.json');verify(root/'payload',req)
    if (root/'job.json').exists():raise ValueError('Already submitted;inspect existing job handle')
    cmd=command()[:-5]+['python','-u','writer_direction_study.py','execute','.'];i=cmd.index('uv');cmd[i:i]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=']
    job=guarded_mrun_launch(cmd,request=req,experiment='tce-writer-direction-v1',task='analysis',model=None,payload=root/'payload',pin='beast',gpu=False,needs=req['needs'],
        mrun_config={k:req[k] for k in ('schema','requestSHA256','device','backend','dtype','scheduling_contract')},timeout_s=900,detach=True,retry_on_kill=False,preflight=True,
        note='CPU1 saved110 direction audit:4 complete futures,2040x2560 rows,1017pairededges. One canonical Act,647.6M direction norm/dot products+5.22M gradient norm products and ideal pullbacks;no model,norm,fit,dose,QKV,CUDA,image. Authenticate retained aliases,load identical inputs once,all vectors stay Beast. Expected<5min/900scap,one hostlocal lease,intentfirst sizing,no retries/expansion.')
    value={'jobID':str(job),'requestSHA256':req['requestSHA256']};save(root/'job.json',value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','execute','collect']);parser.add_argument('root');args=parser.parse_args()
    if args.action=='execute':
        from writer_direction_worker import execute
        result=execute(args.root)
    elif args.action=='collect':
        from response_study import collect
        result=collect(args.root)
    else:result=globals()[args.action](args.root)
    if result is not None:print(json.dumps(result))
