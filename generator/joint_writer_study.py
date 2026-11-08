"""One bounded joint two-slot writer construction with finite consumer feedback."""
import argparse
import json
from pathlib import Path
import shutil
from contextual_writer_program import BUDGET,FINITE,read,save,sha
from coupled_normalization_study import OWNED as PREVIOUS_OWNED,identity,verify
from joint_writer_math import POLICY,DOSES

HERE=Path(__file__).resolve().parent
OWNED=PREVIOUS_OWNED+['joint_writer_math.py','joint_writer_study.py','joint_writer_worker.py','publish_joint_writer.py','test_joint_writer.py']
PARENTS=['publication.json','request.json','report.json','programs.json','reuse.json','parent-inputs.json','parent-lexical-manifest.json','final-serving.json']
WORK={'proposalFitAttempts':2,'maximumCandidatePrograms':4,'slotOrder':[0,1],'doses':list(DOSES),'maximumNormalizationUnits':30,'originalLinears':0,'nominalMatrixMACs':0,
      'QKVMatricesMaterialized':0,'discoveryHistories':4,'discoveryLearnedRows':2040,'freshServingChildren':2,'knownTransferHistories':2,'novelUnscoredHistories':1,'images':0}
FAMILY_SHA='abef41111855f8c620b5d3ab40ea7395ccbfc5d32e3e48d25a5922d04edb31b5'


def stage(root):
    from mrun.client.api import Api
    from mdb.job_support import one_lease_scheduling_contract
    base=HERE.parent/'docs/results/contrast-writer';pub=read(base/'publication.json');old=read(base/'scheduler-job.json');live=Api().json('GET','/api/jobs/'+old['job_id'])
    custody=old['payload_custody'];phases=[custody[k] for k in ('declared','sealed','executed')]
    if sha(base/'publication.json')!='0a3a231312dc619a3266a7b15b6e35640b95616575765176c9fed10f041b5d54' or pub['jobID']!=old['job_id'] or live['state']!='succeeded' or live['result']!=old['result'] or live['payload_custody']!=custody or not custody['required'] or not live.get('admission'):raise ValueError('Qualified104 authority differs')
    if len({(p['sha256'],p['size_bytes']) for p in phases})!=1 or phases[-1]['host']!='beast':raise ValueError('Qualified104 payload differs')
    for name,item in pub['files'].items():
        if sha(base/name)!=item['sha256'] or (base/name).stat().st_size!=item['bytes']:raise ValueError('Complete104 publication differs')
    previous=read(base/'request.json');catalog=read(base/'programs.json');reuse=read(base/'reuse.json');source=read(base/'parent-inputs.json');inventory={x['path']:x for x in pub['freshRemoteFiles']}
    parent=catalog['accepted'];folder=Path(parent['directory']);lexical=reuse['lexicalDirectory'];assets={k:v for k,v in reuse['servingAssets'].items() if k.startswith(lexical+'/')}
    assets[str(folder/'program.json')]=parent['programSHA256'];assets[str(folder/'prefix.bf16')]=parent['program']['prefixSHA256']
    for item in parent['program']['dictionaries']:assets[str(folder/item['path'])]=item['sha256']
    if not assets or any(inventory[k]['sha256']!=v for k,v in assets.items()):raise ValueError('Selected lexical/parent program custody differs')
    normbase=HERE.parent/'docs/results/coupled-normalization';normreq=read(normbase/'request.json');normpub=read(normbase/'publication.json')
    if sha(normbase/'publication.json')!='8bb04982247fdcaeab39b4d98565851d4988bde2991da23660570da8a89acce4':raise ValueError('109 publication differs')
    if sha(normbase/'request.json')!=normpub['files']['request.json']['sha256'] or (normbase/'request.json').stat().st_size!=normpub['files']['request.json']['bytes']:raise ValueError('109 request publication join differs')
    gamma=normreq['gammaParameter'];asset=normreq['selectedParameterAsset']
    if gamma not in source['parameters'] or not any(x['path']==asset['path'] and x['sha256']==asset['sha256'] for x in normpub['freshRemoteFiles']):raise ValueError('Gamma custody differs')
    family=HERE.parent/'docs/plans/joint-baseline-response-writer.md'
    if sha(family)!=FAMILY_SHA or parent['program']['budget']!=BUDGET:raise ValueError('Selected family/budget changed')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for name in OWNED:shutil.copyfile(HERE/name,payload/name)
    exported=payload/'saturn';exported.mkdir()
    for name in ('titan.py','tensor_hash.py'):shutil.copyfile(HERE.parents[1]/'saturn/src/saturn'/name,exported/name)
    shutil.copyfile(HERE/'contextual_writer_saturn_init.py',exported/'__init__.py')
    for name in PARENTS:shutil.copyfile(base/name,payload/('parent-'+name))
    for path,name in [(family,'family-plan.md'),(HERE.parent/'docs/plans/joint-writer-assay.md','frozen-plan.md'),(normbase/'publication.json','normalization109-publication.json'),(normbase/'request.json','normalization109-request.json')]:shutil.copyfile(path,payload/name)
    retained=['request.json','family-plan.md','frozen-plan.md','normalization109-publication.json','normalization109-request.json']+['parent-'+n for n in PARENTS]
    contract=one_lease_scheduling_contract(model_loads=1,logical_branches=5,logical_operations=7,logical_checkpoint_replays=0,
        extra={'modelLoadScope':'Gammaonly5120B;fullshardauthenticated,noQKVmatrices.','branchIsolation':'Two sequential slot proposal families,four finite H/Z candidate futures,exact parent restore;two fresh stdlib serving children inside one lease.'})
    req={'schema':'tce-joint-writer-v1','cases':previous['cases'],'discovery':['A','B','00','10'],'transfer':['01','11'],'novelNoTarget':['novel'],
         'parent':{'jobID':old['job_id'],'custody':custody,'publicationSHA256':sha(base/'publication.json'),'program':parent},'lexicalDirectory':lexical,'servingAssets':assets,
         'gammaParameter':gamma,'selectedParameterAsset':asset,'numericalProfile':source['numericalProfile'],'torchVersion':source['torchVersion'],'rmsNormEpsilon':source['rmsNormEpsilon'],
         'budget':BUDGET,'finitePolicy':FINITE,'objectivePolicy':POLICY,'physicalWork':WORK,'retainedMetadata':retained,
         'sourceHashes':{str(p.relative_to(payload)):sha(p) for p in payload.rglob('*') if p.is_file()},'device':'cuda','backend':'CPU1-paired-proposal-and-tree;CUDA-original-RMS-gamma-only;stdlib-serving',
         'dtype':'bfloat16','batch_size':1,'seq_len':512,'needs':{'cuda':True},'scheduling_contract':contract,'timeoutSeconds':900,
         'protocol':'Two replaceable small slots;one0then1 sweep,twodoses1/.5 each. Four finite H/Z objectives;parentfirstties,allfutures retained,transfer numeric targets after fresh servingseal. Originalnormalization paid offline,never serving. No QKV,attention,MLP,image,newmodel,extrasweep/retry.'}
    req['requestSHA256']=identity(req);save(root/'request.json',req);save(payload/'request.json',req)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':sorted([p.name for p in payload.iterdir()]+['preflight.v1.json'])})
    return {'root':str(root),'requestSHA256':req['requestSHA256'],'physicalWork':WORK}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();req=read(root/'request.json');verify(root/'payload',req)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect existing handle')
    cmd=command()[:-5]+['python','-u','joint_writer_study.py','execute','.'];i=cmd.index('uv');cmd[i:i]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=req,experiment='tce-joint-writer-v1',task='analysis',model=None,payload=root/'payload',pin='beast',gpu=True,needs=req['needs'],
        mrun_config={k:req[k] for k in ('schema','requestSHA256','device','backend','dtype','batch_size','seq_len','scheduling_contract')},timeout_s=900,detach=True,retry_on_kill=False,preflight=True,
        note='Joint inputowned2slot/depth4 writer:one slot0then1 sweep,2paired treefits/4candidate programs;balanced finite H/Z endpoints+responses on2040discovery rows. Gammaonly5120B,4.97GBshard+lexicalauth,max30full512RMSunits/0affineMAC. Fresh stdlib serving before knowntransfer targets. CPU1/no childleases. Expected<10min/900scap;oneCUDAhostlocallease,intentfirst sizing,no retry/extra sweep/image.')
    value={'jobID':str(job),'requestSHA256':req['requestSHA256']};save(root/'job.json',value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','execute','collect']);parser.add_argument('root');args=parser.parse_args()
    if args.action=='execute':
        from joint_writer_worker import execute
        result=execute(args.root)
    elif args.action=='collect':
        from response_study import collect
        result=collect(args.root)
    else:result=globals()[args.action](args.root)
    if result is not None:print(json.dumps(result))
