"""One bounded constant update and six nearest-consumer evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from template_cofactor_program import read,save,sha
from joint_writer_study import OWNED as PREVIOUS_OWNED
from cauchy_writer_math import POLICY,ETA

HERE=Path(__file__).resolve().parent
OWNED=sorted(set(PREVIOUS_OWNED+['writer_direction_math.py','cauchy_writer_math.py','cauchy_writer_study.py','cauchy_writer_worker.py','publish_cauchy_writer.py','test_cauchy_writer.py','publish_writer_direction.py','writer_direction_study.py']))
PARENTS=['publication.json','request.json','scheduler-job.json','report.json','array-manifest.json','typed-proposals.json','typed-normalization.json','programs.json','final-serving.json','parent-parent-inputs.json']
PUB_SHA='4f15854368616a14910558de1a7a003f9da1e35d2880b45259c9940ea4b7fde5'
FAMILY_SHA='89be70bac884b943d9d53272b564498e19e2b1b7369d2cbcca50877c8082a41d'
WORK={'proposalActs':1,'candidatePrograms':1,'routeFits':0,'discoveryHistories':4,'discoveryLearnedRows':2040,'pairedEdges':1017,'atomCounts':[14,9],
      'maximumNormalizationUnits':6,'originalLinears':0,'nominalMatrixMACs':0,'freshServingChildren':1,'knownTransferHistories':2,'novelUnscoredHistories':1,'images':0}


def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify(root,req):
    if identity({k:v for k,v in req.items() if k!='requestSHA256'})!=req['requestSHA256']:raise ValueError('Request identity differs')
    for name,digest in req['sourceHashes'].items():
        if sha(Path(root)/name)!=digest:raise ValueError('Frozen source/member differs: '+name)


def stage(root):
    from mrun.client.api import Api
    from mdb.job_support import one_lease_scheduling_contract
    base=HERE.parent/'docs/results/joint-writer';pub=read(base/'publication.json');old=read(base/'scheduler-job.json');previous=read(base/'request.json');live=Api().json('GET','/api/jobs/'+old['job_id'])
    custody=old['payload_custody'];phases=[custody[k] for k in ('declared','sealed','executed')]
    if sha(base/'publication.json')!=PUB_SHA or live['state']!='succeeded' or live['result']!=old['result'] or live['payload_custody']!=custody or not live.get('admission') or not custody['required'] or len({(x['sha256'],x['size_bytes']) for x in phases})!=1 or phases[-1]['host']!='beast':raise ValueError('Qualified110 authority differs')
    for name,s in pub['files'].items():
        if sha(base/name)!=s['sha256'] or (base/name).stat().st_size!=s['bytes']:raise ValueError('Qualified publication member differs')
    catalog=read(base/'programs.json');parent=catalog['accepted'];arrays=read(base/'array-manifest.json');serving=read(base/'final-serving.json')
    if read(base/'report.json')['selected']!='parent' or parent['programSHA256']!=catalog['parent']['programSHA256'] or [d['atoms'] for d in parent['program']['dictionaries']]!=WORK['atomCounts']:raise ValueError('Accepted fixed parent differs')
    fields={n:arrays['fit0-'+n] for n in ('h','z','target','zt','features','weights','edges','pairWeights','gamma','scaleVector','gradient')}
    fields['normGamma']=arrays['gamma']
    for case in previous['discovery']+previous['transfer']:
        for method in ('parent','target'):
            for port in ('H','Z'):fields[method+'-'+case+'-'+port]=arrays[method+'-'+case+'-'+port]
    h0={r['id']:r['H0'] for r in serving['rows'] if r['id'] in previous['discovery']}
    lexical=previous['lexicalDirectory'];assets={p:s for p,s in previous['servingAssets'].items() if p.startswith(lexical+'/')}
    folder=Path(parent['directory']);assets.update({str(folder/name):digest for name,digest in parent['files'].items()})
    inventory={x['path']:x for x in pub['freshRemoteFiles']};remote={}
    for item in [*fields.values(),*h0.values()]:
        match=inventory[item['path']]
        if any(item[k]!=match[k] for k in ('sha256','bytes')):raise ValueError('Selected numeric custody differs')
        remote[item['path']]={k:match[k] for k in ('sha256','bytes')}
    for path,digest in assets.items():
        if inventory[path]['sha256']!=digest:raise ValueError('Serving asset custody differs')
        remote[path]={k:inventory[path][k] for k in ('sha256','bytes')}
    family=HERE.parent/'docs/plans/parent-preserving-writer-step.md'
    if sha(family)!=FAMILY_SHA:raise ValueError('Selected family changed')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for name in OWNED:shutil.copyfile(HERE/name,payload/name)
    exported=payload/'saturn';exported.mkdir()
    for name in ('titan.py','tensor_hash.py'):shutil.copyfile(HERE.parents[1]/'saturn/src/saturn'/name,exported/name)
    shutil.copyfile(HERE/'contextual_writer_saturn_init.py',exported/'__init__.py')
    for name in PARENTS:shutil.copyfile(base/name,payload/('parent-'+name))
    for path,name in [(family,'family-plan.md'),(HERE.parent/'docs/plans/cauchy-writer-assay.md','frozen-plan.md')]:shutil.copyfile(path,payload/name)
    retained=['request.json','family-plan.md','frozen-plan.md']+['parent-'+n for n in PARENTS]
    req={'schema':'tce-cauchy-writer-v1','parent':{'jobID':old['job_id'],'custody':custody,'publicationSHA256':PUB_SHA,'program':parent},
         'fields':fields,'ownH0':h0,'remoteInputs':remote,'servingAssets':assets,'lexicalDirectory':lexical,'cases':previous['cases'],'discovery':previous['discovery'],'transfer':previous['transfer'],'novelNoTarget':previous['novelNoTarget'],
         'numericalProfile':previous['numericalProfile'],'torchVersion':previous['torchVersion'],'rmsNormEpsilon':previous['rmsNormEpsilon'],'policy':POLICY,'eta':ETA,'physicalWork':WORK,'retainedMetadata':retained,
         'sourceHashes':{str(p.relative_to(payload)):sha(p) for p in payload.rglob('*') if p.is_file()},'device':'cuda','backend':'CPU1-fixed-route-Cauchy;CUDA-source-RMS-gamma-only;stdlib-serving','dtype':'bfloat16','needs':{'cuda':True},'timeoutSeconds':900,
         'scheduling_contract':one_lease_scheduling_contract(model_loads=0,model_free=True,logical_branches=3,logical_operations=7,logical_checkpoint_replays=0,extra={'modelLoadScope':'No full-model/shard load;model_free is scheduler load classification only. Saved original gamma5120B and at most six original normalization/gamma evaluations remain paid neural arithmetic.','branchIsolation':'One proposal;discovery finite feedback;immutable selection/serving seal;known-transfer diagnostic only;no child leases.'}),
         'protocol':'One joint constant ray,no route/dose sweep. Reuse actual110 parent/native states. At most6newnorms,zeroaffines. Fresh parent/candidate/accepted serving before knowntransfer numeric release. Retain rejectedcandidate,allstrata;onelease,noretries,image,capacityexpansion.'}
    req['requestSHA256']=identity(req);save(root/'request.json',req);save(payload/'request.json',req)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':sorted([p.name for p in payload.iterdir()]+['preflight.v1.json'])})
    return {'root':str(root),'requestSHA256':req['requestSHA256'],'sourceMembers':len(req['sourceHashes']),'remoteFiles':len(remote),'physicalWork':WORK}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();req=read(root/'request.json');verify(root/'payload',req)
    if (root/'job.json').exists():raise ValueError('Already submitted;inspect existing job handle')
    cmd=command()[:-5]+['python','-u','cauchy_writer_study.py','execute','.'];i=cmd.index('uv');cmd[i:i]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=req,experiment='tce-cauchy-writer-v1',task='analysis',model=None,payload=root/'payload',pin='beast',gpu=True,needs=req['needs'],mrun_config={k:req[k] for k in ('schema','requestSHA256','device','backend','dtype','scheduling_contract')},timeout_s=900,detach=True,retry_on_kill=False,preflight=True,
        note='One joint23atom fixed-route Cauchy step/fullJ on2040rows,1017edges;CPU1,eta1/16,onefinitecandidate,exactparentrestore. Atmost6CUDA sourceRMSunits/0affines,5120Bgamma,no model/shardload. One fresh stdlib child;knowntransfer afterselectionseal. Expected<5min/900scap;onehostlocallease,intentfirst sizing,noretry,sweep or image.')
    value={'jobID':str(job),'requestSHA256':req['requestSHA256']};save(root/'job.json',value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','execute','collect']);parser.add_argument('root');args=parser.parse_args()
    if args.action=='execute':
        from cauchy_writer_worker import execute
        result=execute(args.root)
    elif args.action=='collect':
        from response_study import collect
        result=collect(args.root)
    else:result=globals()[args.action](args.root)
    if result is not None:print(json.dumps(result))
