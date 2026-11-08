"""One host-local lease for six source-policy normalization reconstructions."""
import argparse
import json
from pathlib import Path
import shutil
from contextual_writer_program import read,save,sha
from finite_consumer_crossover_study_v2 import OWNED as PREVIOUS_OWNED,identity,verify
from coupled_normalization import STATES,EDGES,CONTRASTS,SCORING,SOURCE_POLICY,EXPLANATORY_POLICY

HERE=Path(__file__).resolve().parent
OWNED=PREVIOUS_OWNED+['coupled_normalization.py','coupled_normalization_study.py','coupled_normalization_worker.py','publish_coupled_normalization.py','test_coupled_normalization.py']
PARENTS=['request.json','publication.json','report.json','array-manifest.json','typed-construction.json','analysis.json','consumer-fields.json','typed-consumers.json','qualified-parent-inputs.json']
WORK={'normalizationActs':6,'analysisActs':1,'normalizedRows':3072,'coordinatesPerPointwiseStage':7864320,'rsqrtCallsByRow':3072,
      'originalLinears':0,'nominalMatrixMACs':0,'QKVMatricesMaterialized':0,'fullOperandFields':6,'gammaFields':1,'analysisRowFields':79,
      'retainedNumericFields':86,'comparisonRows':1024,'intervenedRows':2,'unmodifiedRowControlRows':1022,'fullEncoderEvaluations':0,'images':0}
FAMILY_SHA='c22e3dfb640a472ef8d6267a5dd13dcb258d86d635bddda82e8f84b3d14fae8f'
PARENT_PUBLICATION_SHA='81f4fb4078446d9ddeb510ac0115353aeeca701cd03f0b08d13786c64e8fc6ee'


def stage(root):
    from mrun.client.api import Api
    from mdb.job_support import one_lease_scheduling_contract
    base=HERE.parent/'docs/results/finite-consumer-crossover';pub=read(base/'publication.json');old=read(base/'scheduler-job.json')
    live=Api().json('GET','/api/jobs/'+old['job_id']);custody=old['payload_custody'];phases=[custody[k] for k in ('declared','sealed','executed')]
    if sha(base/'publication.json')!=PARENT_PUBLICATION_SHA or pub['jobID']!=old['job_id'] or live['state']!='succeeded' or live['result']!=old['result'] or live['payload_custody']!=custody or not custody['required'] or not live.get('admission'):raise ValueError('Qualified107 authority differs')
    if len({(p['sha256'],p['size_bytes']) for p in phases})!=1 or phases[-1]['host']!='beast':raise ValueError('Qualified107 payload differs')
    for name,item in pub['files'].items():
        if sha(base/name)!=item['sha256'] or (base/name).stat().st_size!=item['bytes']:raise ValueError('Complete107 publication differs')
    previous=read(base/'request.json');arrays=read(base/'array-manifest.json');source=read(base/'qualified-parent-inputs.json')
    if previous['position']!=9 or previous['cases']!=['00','10'] or source['parameters']!=previous['parameters'] or source['numericalProfile']!=previous['numericalProfile']:raise ValueError('Qualified107 family differs')
    endpoints={**previous['endpoints'],**{k:arrays[k] for k in ('Xno','Xon')}}
    if set(endpoints)!=set(STATES):raise ValueError('Six actual state domain differs')
    inventory={p['path']:p for p in pub['freshRemoteFiles']};remote={}
    for name,item in endpoints.items():
        if item['shape']!=[1,512,2560] or item['storedDtype']!='<u2' or item['format']!=('npy' if name in ('a','b') else 'raw'):raise ValueError('Qualified full BF16 ABI differs')
        witness=inventory[item['path']]
        if any(witness[k]!=item[k] for k in ('sha256','bytes')):raise ValueError('State lacks107 fresh byte custody')
        remote[item['path']]={k:item[k] for k in ('sha256','bytes')}
    gamma=next(p for p in previous['parameters'] if p['name']=='model.layers.1.input_layernorm.weight')
    asset=next(p for p in previous['selectedParameterAssets'] if p['path']==gamma['file'])
    if any(inventory[asset['path']][k]!=asset[k] for k in ('sha256','bytes')) or gamma['bytes']!=5120:raise ValueError('Selected gamma/shard custody differs')
    family=HERE.parent/'docs/plans/coupled-baseline-response-state.md'
    if sha(family)!=FAMILY_SHA:raise ValueError('Selected108 contract changed')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for name in OWNED:shutil.copyfile(HERE/name,payload/name)
    exported=payload/'saturn';exported.mkdir()
    for name in ('titan.py','tensor_hash.py'):shutil.copyfile(HERE.parents[1]/'saturn/src/saturn'/name,exported/name)
    shutil.copyfile(HERE/'contextual_writer_saturn_init.py',exported/'__init__.py')
    for name in PARENTS:shutil.copyfile(base/name,payload/('parent-'+name))
    for source_path,name in [(family,'family-plan.md'),(HERE.parent/'docs/plans/coupled-normalization-assay.md','frozen-plan.md'),
                             (HERE.parent/'docs/results/research-review-cycle108/advisor-survey.json','review108-survey.json'),
                             (HERE.parent/'docs/results/research-review-cycle108/public-scalar-baseline-geometry.json','review108-geometry.json'),
                             (HERE.parent/'docs/results/research-review-cycle108/public-scalar-common-error.json','review108-common-error.json')]:shutil.copyfile(source_path,payload/name)
    retained=['request.json','family-plan.md','frozen-plan.md','review108-survey.json','review108-geometry.json','review108-common-error.json']+['parent-'+n for n in PARENTS]
    contract=one_lease_scheduling_contract(model_loads=1,logical_branches=6,logical_operations=7,logical_checkpoint_replays=0,
        extra={'modelLoadScope':'Only5120B original gamma; full shard authenticated, no QKV tensor materialization.',
               'branchIsolation':'Six source-policy normalization Acts and one FP64 explanatory Act; no child leases/retries/expansion.'})
    req={'schema':'tce-coupled-normalization-v1','position':9,'cases':['00','10'],'states':list(STATES),'edges':EDGES,'contrasts':CONTRASTS,'endpoints':endpoints,
         'parent':{'jobID':old['job_id'],'custody':custody,'publicationSHA256':PARENT_PUBLICATION_SHA},'remoteInputs':remote,'gammaParameter':gamma,'selectedParameterAsset':asset,
         'numericalProfile':previous['numericalProfile'],'torchVersion':previous['torchVersion'],'rmsNormEpsilon':previous['rmsNormEpsilon'],
         'sourcePolicy':SOURCE_POLICY,'explanatoryPolicy':EXPLANATORY_POLICY,'scoring':SCORING,'physicalWork':WORK,'retainedMetadata':retained,
         'sourceHashes':{str(p.relative_to(payload)):sha(p) for p in payload.rglob('*') if p.is_file()},'device':'cuda','backend':'original-RMS-gamma-only;CPU-FP64-explanation',
         'dtype':'bfloat16','batch_size':1,'seq_len':512,'needs':{'cuda':True},'scheduling_contract':contract,'timeoutSeconds':900,
         'protocol':'Offline same six actual107 states. Six paid original norm/gamma reconstructions,zero affineMACs. Source-policy reconstruction,not parity with unavailable107 intermediates. No fit,newprompt,QKV,attention,MLP,suffix,image,retry or expansion.'}
    req['requestSHA256']=identity(req);save(root/'request.json',req);save(payload/'request.json',req)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':sorted([p.name for p in payload.iterdir()]+['preflight.v1.json'])})
    return {'root':str(root),'requestSHA256':req['requestSHA256'],'physicalWork':WORK}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();req=read(root/'request.json');verify(root/'payload',req)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect existing handle')
    cmd=command()[:-5]+['python','-u','coupled_normalization_study.py','execute','.'];i=cmd.index('uv')
    cmd[i:i]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=req,experiment='tce-coupled-normalization-v1',task='analysis',model=None,payload=root/'payload',pin='beast',gpu=True,needs=req['needs'],
        mrun_config={k:req[k] for k in ('schema','requestSHA256','device','backend','dtype','batch_size','seq_len','scheduling_contract')},timeout_s=900,detach=True,retry_on_kill=False,preflight=True,
        note='Six stored107 full512x2560 BF16 states:6 originalRMS/gamma-only Acts,0affineMAC/QKV matrices,1CPUFP64 direction/denominator/finite analysis. Load5120B gamma after4.97GBshard authentication;CPU1. 1024row comparisons/1022unchanged controls. No fit,prompt,suffix,image,retry/expansion. One hostlocal CUDA lease,intentfirst sizing;expected<5min/900scap.')
    value={'jobID':str(job),'requestSHA256':req['requestSHA256']};save(root/'job.json',value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','execute','collect']);parser.add_argument('root');args=parser.parse_args()
    if args.action=='execute':
        from coupled_normalization_worker import execute
        result=execute(args.root)
    elif args.action=='collect':
        from response_study import collect
        result=collect(args.root)
    else:result=globals()[args.action](args.root)
    if result is not None:print(json.dumps(result))
