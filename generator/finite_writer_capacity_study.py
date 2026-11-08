"""One guarded CPU lease for fixed-nine-atom target-assisted capacity."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from contextual_writer_program import read, save, sha, FINITE

HERE = Path(__file__).resolve().parent
OWNED = ['finite_writer_capacity.py','finite_writer_capacity_worker.py','finite_writer_capacity_study.py','publish_finite_writer_capacity.py','test_finite_writer_capacity.py',
    'contextual_writer_program.py','contextual_writer_worker.py','template_cofactor_program.py','token_history_process.py','token_history_address.py','contextual_writer_saturn_init.py',
    'publish_contextual_writer.py','publish_contrast_writer.py']
PARENTS = ['programs.json','final-serving.json','parent-inputs.json','fit.json','publication.json','report.json','request.json']


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify(root,request):
    if identity({k:v for k,v in request.items() if k!='requestSHA256'}) != request['requestSHA256']: raise ValueError('Request differs')
    for name,h in request['sourceHashes'].items():
        if sha(Path(root)/name) != h: raise ValueError('Frozen source differs: '+name)


def stage(root):
    from mrun.client.api import Api
    from mdb.job_support import one_lease_scheduling_contract
    base = HERE.parent/'docs/results/contrast-writer'; pub = read(base/'publication.json'); old = read(base/'scheduler-job.json'); live = Api().json('GET','/api/jobs/'+old['job_id'])
    custody = old['payload_custody']; phases = [custody[k] for k in ('declared','sealed','executed')]
    if live['state']!='succeeded' or live['result']!=old['result'] or live['payload_custody']!=custody or not custody['required'] or pub['jobID']!=old['job_id'] or not live.get('admission'): raise ValueError('Qualified parent job differs')
    if any((p['sha256'],p['size_bytes'])!=(phases[0]['sha256'],phases[0]['size_bytes']) for p in phases) or phases[2]['host']!='beast': raise ValueError('Qualified parent payload phases differ')
    for name,x in pub['files'].items():
        if sha(base/name)!=x['sha256'] or (base/name).stat().st_size!=x['bytes']: raise ValueError('Parent publication differs')
    catalog = read(base/'programs.json'); child = read(base/'final-serving.json'); source = read(base/'parent-inputs.json'); fit = read(base/'fit.json')
    if catalog['accepted']['programSHA256']!=catalog['candidate']['programSHA256'] or read(base/'report.json')['selected']!='candidate' or fit['atoms']!=9: raise ValueError('Qualified nine-atom family differs')
    inventory = {x['path']:x for x in pub['freshRemoteFiles']}; inputs = {}
    def need(path,digest,size=None):
        item = inventory[path]
        if item['sha256']!=digest or size is not None and item['bytes']!=size: raise ValueError('Input lacks prior full-byte custody')
        inputs[path] = {'sha256':digest,'bytes':item['bytes']}
    selected = catalog['accepted']; folder = Path(selected['directory']); program = selected['program']
    need(str(folder/'program.json'),selected['programSHA256']); need(str(folder/'prefix.bf16'),program['prefixSHA256'],10240)
    for x in program['dictionaries']: need(str(folder/x['path']),x['sha256'],x['bytes'])
    cases = ['A','B','00','10','01','11']; packets = {}
    for key in cases:
        row = next(x for x in child['rows'] if x['id']==key); packets[key] = row['packet']; cur = next(x for x in row['predictions'] if x['method']=='candidate')
        for x in (row['H0'],cur,source['cases'][key]['H1']): need(x['path'],x['sha256'],x['bytes'])
    pairs = [('A','B'),('00','10'),('01','11')]; paired = sum(sum(a==b for a,b in zip(packets[x]['ids'][2:],packets[y]['ids'][2:])) for x,y in pairs)
    n = 6*510; singles = n-2*paired
    work = {'newNeuralEvaluations':0,'newOriginalLinears':0,'newImages':0,'groups':2,'rows':n,'pairedComponents':paired,'unpairedComponents':singles,
        'evaluatedChoices':paired*81+singles*9,'finiteOptionStates':n*9,'finiteOptionCoordinateAdditions':n*9*2560,
        'endpointNormProducts':n*9*2560,'pairedResponseNormProducts':paired*81*2560}
    root = Path(root).resolve(); root.mkdir(parents=True,exist_ok=False); payload = root/'payload'; payload.mkdir()
    for name in OWNED: shutil.copyfile(HERE/name,payload/name)
    saturn = HERE.parents[1]/'saturn/src/saturn'; exported = payload/'saturn'; exported.mkdir()
    for name in ('titan.py','tensor_hash.py'): shutil.copyfile(saturn/name,exported/name)
    shutil.copyfile(HERE/'contextual_writer_saturn_init.py',exported/'__init__.py')
    for name in PARENTS: shutil.copyfile(base/name,payload/('parent-'+name))
    shutil.copyfile(HERE.parent/'docs/plans/finite-writer-capacity-assay.md',payload/'frozen-plan.md')
    shutil.copyfile(HERE.parent/'docs/plans/finite-writer-capacity.md',payload/'family-plan.md')
    retained = ['request.json','frozen-plan.md','family-plan.md']+['parent-'+n for n in PARENTS]
    scheduling = one_lease_scheduling_contract(model_loads=0,model_free=True,logical_branches=2,logical_operations=2,logical_checkpoint_replays=0,
        extra={'task_family':'analysis','exhaustiveScalarChoices':work['evaluatedChoices'],
               'branchIsolation':'Two canonical CPU capacity Acts. Scalar choices are arithmetic inside each Act,not scheduler work items. No child leases,model or GPU.'})
    request = {'schema':'tce-fixed-writer-capacity-v1','cases':cases,'discovery':cases[:4],'transfer':cases[4:],
        'finitePolicy':FINITE,'physicalWork':work,'remoteInputs':inputs,'retainedMetadata':retained,
        'parent':{'jobID':old['job_id'],'custody':custody,'publicationSHA256':sha(base/'publication.json'),'requestSHA256':read(base/'request.json')['requestSHA256']},
        'sourceHashes':{str(p.relative_to(payload)):sha(p) for p in payload.rglob('*') if p.is_file()},
        'device':'cpu','backend':'numpy-FP64-finite-enumeration;torch-CPU-canonical-custody-only','dtype':'BF16-states-FP32-atoms-FP64-scores',
        'needs':{'cpu':True},'scheduling_contract':scheduling,'timeoutSeconds':900,
        'protocol':'Fixed own slot0/nine atoms;target-assisted81/9 enumeration. No fit,resolver,native projection,image,teacher/data/family expansion. One CPU1host-local lease,no retries. Historical neural work paid.'}
    request['requestSHA256'] = identity(request); save(root/'request.json',request); save(payload/'request.json',request)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':sorted([p.name for p in payload.iterdir()]+['preflight.v1.json'])})
    return {'root':str(root),'requestSHA256':request['requestSHA256'],'physicalWork':work}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root = Path(root).resolve(); req = read(root/'request.json'); verify(root/'payload',req)
    if (root/'job.json').exists(): raise ValueError('Already submitted;inspect existing job')
    cmd = command()[:-5]+['python','-u','finite_writer_capacity_study.py','execute','.']; i = cmd.index('uv')
    cmd[i:i] = ['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=']
    job = guarded_mrun_launch(cmd,request=req,experiment='tce-fixed-writer-capacity-v1',task='analysis',model=None,payload=root/'payload',pin='beast',gpu=False,needs=req['needs'],
        mrun_config={k:req[k] for k in ('schema','requestSHA256','device','backend','dtype','scheduling_contract')},timeout_s=900,detach=True,retry_on_kill=False,preflight=True,
        note='CPU1 saved-state fixed9atom H1 capacity:3060learnedrows,1525disjointpairs/10single,123615choices. Stream81x2560 perpair;70.5Mfinite adds/316.2Mresponse norm products plus metadata/metrics. Zero model/originallinears/CUDA/image. Canonical SaturnCPUFrames/Acts,all endpoints/countertrends. Expected<5min/900scap,intentfirst RAM sizing,no retry/expansion;one hostlocal lease.')
    value = {'jobID':str(job),'requestSHA256':req['requestSHA256']}; save(root/'job.json',value); return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action',choices=['stage','submit','execute','collect']); parser.add_argument('root'); args = parser.parse_args()
    if args.action=='execute':
        from finite_writer_capacity_worker import execute
        result = execute(args.root)
    elif args.action=='collect':
        from response_study import collect
        result = collect(args.root)
    else: result = globals()[args.action](args.root)
    if result is not None: print(json.dumps(result))
