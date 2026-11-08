"""V2 staging uses authenticated parameter metadata plus fresh worker/publisher hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from contextual_writer_program import read,save,sha
from finite_consumer_crossover import POLICY,SCORING

HERE = Path(__file__).resolve().parent
OWNED = ['finite_consumer_crossover.py','finite_consumer_crossover_worker.py','finite_consumer_crossover_study.py','publish_finite_consumer_crossover.py','test_finite_consumer_crossover.py',
         'contrast_writer_worker.py','contrast_writer_fit.py','contextual_writer_program.py','contextual_writer_worker.py','template_cofactor_program.py','token_history_process.py',
         'token_history_address.py','contextual_writer_saturn_init.py','publish_contextual_writer.py','publish_contrast_writer.py','finite_consumer_crossover_study_v2.py']
PARENTS = ['parent-inputs.json','consumer-fields.json','reused-fields.json','final-serving.json','programs.json','publication.json','report.json','request.json','typed-consumers.json','parent-typed-consumers.json']
WORK = {'constructionActs':1,'consumerUnits':2,'normalizationUnits':2,'originalLinears':6,'nominalMatrixMACs':16106127360,'analysisActs':4,
        'crossedStateFields':2,'stateRowFields':8,'analysisRowFields':44,'comparisonRows':3072,'unmodifiedRowControlRows':3066,'fullEncoderEvaluations':0,'images':0}


def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify(root,request):
    if identity({k:v for k,v in request.items() if k!='requestSHA256'})!=request['requestSHA256']:raise ValueError('Request identity differs')
    for name,h in request['sourceHashes'].items():
        if sha(Path(root)/name)!=h:raise ValueError('Frozen source differs: '+name)


def stage(root):
    from mrun.client.api import Api
    from mdb.job_support import one_lease_scheduling_contract
    base=HERE.parent/'docs/results/contrast-writer';pub=read(base/'publication.json');old=read(base/'scheduler-job.json');live=Api().json('GET','/api/jobs/'+old['job_id'])
    custody=old['payload_custody'];phases=[custody[k] for k in ('declared','sealed','executed')]
    if live['state']!='succeeded' or live['result']!=old['result'] or live['payload_custody']!=custody or not custody['required'] or not live.get('admission') or pub['jobID']!=old['job_id']:raise ValueError('Qualified parent job differs')
    if len({(x['sha256'],x['size_bytes']) for x in phases})!=1 or phases[-1]['host']!='beast':raise ValueError('Qualified executed payload differs')
    for name,x in pub['files'].items():
        if sha(base/name)!=x['sha256'] or (base/name).stat().st_size!=x['bytes']:raise ValueError('Qualified publication differs')
    source=read(base/'parent-inputs.json');catalog=read(base/'programs.json');final=read(base/'final-serving.json');own=read(base/'consumer-fields.json')['rows'];reused=read(base/'reused-fields.json')['rows']
    if read(base/'report.json')['selected']!='candidate' or catalog['accepted']['programSHA256']!=catalog['candidate']['programSHA256']:raise ValueError('Frozen accepted104 identity differs')
    endpoints={};consumers={};packets={};inventory={x['path']:x for x in pub['freshRemoteFiles']};inputs={}
    def need(item):
        previous=inventory[item['path']]
        if any(previous[k]!=item[k] for k in ('sha256','bytes')):raise ValueError('Input lacks published byte custody')
        inputs[item['path']]={k:item[k] for k in ('sha256','bytes')}
    for key,native_name,own_name in [('00','a','p'),('10','b','q')]:
        row=next(x for x in final['rows'] if x['id']==key);packets[key]=row['packet'];accepted=next(x for x in row['predictions'] if x['method']=='accepted')
        if accepted['programSHA256']!=catalog['accepted']['programSHA256']:raise ValueError('Accepted state program differs')
        for name,item,method,records in [(native_name,source['cases'][key]['H1'],'native',reused),(own_name,accepted,'candidate',own)]:
            endpoint={k:item[k] for k in ('path','sha256','bytes')};endpoint.update({'shape':[1,512,2560],'storedDtype':'<u2','format':'npy' if name in ('a','b') else 'raw','payloadSHA256':item.get('payloadSHA256',item['sha256'])})
            record=next(x for x in records if x['case']==key and x['method']==method)
            if record['inputPayloadSHA256']!=endpoint['payloadSHA256']:raise ValueError('Reused consumer/full state binding differs')
            endpoints[name]=endpoint;consumers[name]={'case':key,'method':method,'inputPayloadSHA256':record['inputPayloadSHA256'],'fields':record['fields']};need(endpoint)
            for field in record['fields'].values():need(field)
    selected_files={x['file'] for x in source['parameters']};assets=[x for x in source['parameterAssets'] if x['path'] in selected_files]
    if len(assets)!=1 or len(source['parameters'])!=4 or {x['path'] for x in assets}!=selected_files:raise ValueError('Expected one covering selected shard/four tensors')
    parent_consumer=read(base/'consumer-fields.json')
    if parent_consumer['parameters']!=source['parameters'] or parent_consumer['parameterAssets']!=assets:
        raise ValueError('Selected shard source/consumer metadata custody differs')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for name in OWNED:shutil.copyfile(HERE/name,payload/name)
    exported=payload/'saturn';exported.mkdir()
    for name in ('titan.py','tensor_hash.py'):shutil.copyfile(HERE.parents[1]/'saturn/src/saturn'/name,exported/name)
    shutil.copyfile(HERE/'contextual_writer_saturn_init.py',exported/'__init__.py')
    for name in PARENTS:shutil.copyfile(base/name,payload/('qualified-'+name))
    for name in ('advisor-actual.json','publication.json'):
        folder='finite-writer-capacity-evidence' if name=='advisor-actual.json' else 'finite-writer-capacity'
        shutil.copyfile(HERE.parent/'docs/results'/folder/name,payload/('capacity-'+name))
    shutil.copyfile(HERE.parent/'docs/plans/finite-consumer-crossover.md',payload/'family-plan.md')
    shutil.copyfile(HERE.parent/'docs/plans/finite-consumer-crossover-assay.md',payload/'frozen-plan.md')
    shutil.copyfile(HERE.parent/'docs/plans/finite-consumer-crossover-staging-v2.md',payload/'staging-plan-v2.md')
    retained=['staging-plan-v2.md','request.json','family-plan.md','frozen-plan.md','capacity-advisor-actual.json','capacity-publication.json']+['qualified-'+n for n in PARENTS]
    contract=one_lease_scheduling_contract(model_loads=1,logical_branches=2,logical_operations=7,logical_checkpoint_replays=0,
        extra={'modelLoadScope':'One original selected layer1 norm/QKV set; no full model. Four historical diagonal units reused.',
               'branchIsolation':'One CPU construction Act,two original CUDA consumer Acts,four CPU signed-analysis Acts; no child leases or retry.'})
    request={'schema':'tce-finite-consumer-crossover-v1','position':9,'cases':['00','10'],'endpoints':endpoints,'reusedConsumers':consumers,'packets':packets,
             'parent':{'jobID':old['job_id'],'custody':custody,'publicationSHA256':sha(base/'publication.json')},'acceptedProgramSHA256':catalog['accepted']['programSHA256'],
             'remoteInputs':inputs,'selectedParameterAssets':assets,'parameterAssetCustody':{'historical':'Authenticated qualified parent-inputs and consumer-fields records agree on the selected full shard and four tensors; shard absent from parent freshRemoteFiles.','current':'Unchanged worker parameters() authenticates full shard/header/four tensor payloads before loading; publisher freshly hashes full shard afterward.'},'parameters':source['parameters'],'numericalProfile':source['numericalProfile'],'torchVersion':source['torchVersion'],'rmsNormEpsilon':source['rmsNormEpsilon'],
             'finitePolicy':POLICY,'scoring':SCORING,'physicalWork':WORK,'retainedMetadata':retained,'sourceHashes':{str(p.relative_to(payload)):sha(p) for p in payload.rglob('*') if p.is_file()},
             'device':'cuda','backend':'qualified-original-inputnorm-QKV;CPU-FP64-finite-crossover-analysis','dtype':'bfloat16','batch_size':1,'seq_len':512,'needs':{'cuda':True},'scheduling_contract':contract,'timeoutSeconds':900,
             'protocol':'Offline row9 source-bound finite crossover,not serving. Stored diagonals,2crosses,full512geometry,6linears16.106BMACs. No fit,attention,MLP,suffix,image,newprompt,retry or automatic expansion.'}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(payload/'request.json',request)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':sorted([p.name for p in payload.iterdir()]+['preflight.v1.json'])})
    return {'root':str(root),'requestSHA256':request['requestSHA256'],'physicalWork':WORK}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();req=read(root/'request.json');verify(root/'payload',req)
    if (root/'job.json').exists():raise ValueError('Already submitted;inspect existing job')
    cmd=command()[:-5]+['python','-u','finite_consumer_crossover_study_v2.py','execute','.'];i=cmd.index('uv')
    cmd[i:i]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=req,experiment='tce-finite-consumer-crossover-v1',task='analysis',model=None,payload=root/'payload',pin='beast',gpu=True,needs=req['needs'],
        mrun_config={k:req[k] for k in ('schema','requestSHA256','device','backend','dtype','batch_size','seq_len','scheduling_contract')},timeout_s=900,detach=True,retry_on_kill=False,preflight=True,
        note='Finite row9 vessel00to10 crossover:2 full512 originalRMS/QKV units,6linears16.106BMACs;reuse4 exact diagonals. One selected31.46MBparameter set,4.97GBshard hashed once;CPU1FP64 construction/signedGram analysis. 3072comparisons include6interventions/3066unchangedrow controls. No fit,attention,MLP,suffix,image,newprompt,retry/expansion. Expected<5min/900scap,oneCUDAhostlocal lease,intentfirst sizing.')
    value={'jobID':str(job),'requestSHA256':req['requestSHA256']};save(root/'job.json',value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','execute','collect']);parser.add_argument('root');args=parser.parse_args()
    if args.action=='execute':
        from finite_consumer_crossover_worker import execute
        result=execute(args.root)
    elif args.action=='collect':
        from response_study import collect
        result=collect(args.root)
    else:result=globals()[args.action](args.root)
    if result is not None:print(json.dumps(result))
