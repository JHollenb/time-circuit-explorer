"""One guarded lease: deferred-value actor seals/exits before reference scoring."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read, save, sha, identity
from flux_clock_study import verify
from image_dual_study import OWNED as PREVIOUS_OWNED, WHEELS

HERE=Path(__file__).resolve().parent
OWNED=['deferred_values.py','deferred_value_worker.py','deferred_value_analysis.py','deferred_value_study.py']+PREVIOUS_OWNED


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from projected_complement import fields
    from deferred_value_worker import PARAMETERS
    from image_dual_analysis import NAMES
    parent,collection,job,previous=checked(HERE.parent/'artifacts/image-dual-v1')
    older,oc,oj,op=checked(HERE.parent/'artifacts/propagated-context-v1')
    api=Api()
    for c,j in [(collection,job),(oc,oj)]:
        live=api.json('GET','/api/jobs/'+j['job_id'])
        if live['state']!='succeeded' or live['payload_custody']!=c['custody']:raise ValueError('Parent custody changed')
    p=parent/'result'
    if sha(p/'report.json')!='84dc610d01f1fa98adfbf209fe248687a5abebbeb1eb2ccaa86db742f43b78a8':raise ValueError('Frozen parent differs')
    compiler,actor,s=[read(p/n) for n in ['compiler.json','actor.json','inputs.json']]
    for n in ['compiler','actor']:
        if read(p/(n+'-seal.json'))[n+'SHA256']!=sha(p/(n+'.json')):raise ValueError('Parent seal differs')
    targets=read(p/'targets.json');targets['retained']={n:{**actor['source']['fields'],**actor['next_query']['fields']}[n] for n in NAMES}
    reuse={n:compiler['compile']['fields'][n] for n in ['duals','pDenominator','groupIDs']}
    receipt=read(parent/'retained-array-verification.json')
    for x in fields({'targets':targets,'reuse':reuse,'selected':s['selected'],'textKeys':s['textKeySlice'],'banks':s['banks']}):
        if receipt['files'][x['path']]['sha256']!=x['sha256']:raise ValueError('Parent operand byte custody absent')
    denied={x['path'] for x in fields(targets)}|{x['path'] for x in fields(actor)}
    denied|={x['path'] for x in fields(read(p/'reference.json'))}
    denied|={s['banks'][n][k]['path'] for n in ['P','Q'] for k in ['K','V']}|{s['selected']['V']['path']}
    denied|={compiler['compile']['fields']['rawP']['path']}
    remote=job['result']['diagnostics']['cwd']+'/response-result/'
    s.update(reuse=reuse,compilerSealSHA256=sha(p/'compiler-seal.json'),controlModLN=actor['modln']['fields']['x'],
        matchedControlFields={n:actor['read']['fields'][n] for n in ['imageNumerator','textLogits','textK_hybrid','logits','probability','peak','partition','H_FP64']},
        deferredParentJobID=job['job_id'],deferredParentCustody=collection['custody'],
        deferredParentRecords={n:{'path':remote+n,'sha256':sha(p/n)} for n in ['compiler.json','compiler-seal.json','actor.json','actor-seal.json','reference.json','reference-seal.json','report.json','request.json','inputs.json','targets.json']},
        deniedNumericalInputs=sorted(denied))
    artifacts=read(older/'result/parameter-artifacts.json')
    if read(older/'result/inputs.json')['imageAssets']!=s['imageAssets']:raise ValueError('Original V map/model source differs')
    artifacts['artifacts']=[x for x in artifacts['artifacts'] if x['source']['tensor'] in PARAMETERS]
    if {x['source']['tensor'] for x in artifacts['artifacts']}!=set(PARAMETERS):raise ValueError('Required original maps incomplete')
    s['originalParameterArtifacts']=artifacts['artifacts']
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for n in OWNED:
        if n in previous['ownedSources'] and sha(HERE/n)!=previous['sourceHashes'][n]:raise ValueError('Frozen helper changed')
        shutil.copyfile(HERE/n,payload/n)
    for n in WHEELS:shutil.copyfile(parent/'payload'/n,payload/n)
    shutil.copyfile(p/'retrieve_weights.py',payload/'retrieve_weights.py')
    shutil.copyfile(HERE.parent/'docs/plans/deferred-value-projection.md',payload/'frozen-plan.md')
    save(payload/'inputs.json',s);save(payload/'targets.json',targets);save(payload/'parameter-artifacts.json',artifacts)
    preview=_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda')
    save(payload/'resource-basis.json',{'intentFirstPreview':preview,'measuredParent':job['result'],
        'geometry':{'distinctBF16ParameterBytes':245367040,'reusedDualFP64Bytes':153354240,
            'twoFP64WeightedInputsBytes':4718592,'newFullValueBanks':0},
        'exception':'Selected7tensors,one actor not fullmodel. Prior58:1712.7MBRSS568MBdevice92.2s with compiler/ref;60same5consumer:1643RSS568device56.48s. New2Vmaps37.75MBBF16/150.99MBFP64plusfew3MBnativefields. RAM4096VRAM1024CPU4disk3GB,120sexpected600cap. No fullvalue projection or extraMLP; postseal NumPy AV18.874MMAC.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=9,logical_checkpoint_replays=0,
        conditioner_policy='Paid authenticated Qimage/text/Ptext states, retainedPnorm/queryduals,QtextKprefix,selectedPQK and originalV/consumer maps.',
        extra={'modelLoadScope':'Actor2V+5consumer tensors; no full transformer.',
            'targetIsolation':'All full/selected/hybrid V and prior outputs/probabilities denied before actorseal/exit; sameallocation actual own H→MLP→T→nextQ.'})
    r={'schema':'tce-deferred-value-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in payload.iterdir()},
        'scheduling_contract':contract,'model':'flux2-klein-4b-distilled','backend':'diffusers','device':'cuda','dtype':'bfloat16',
        'batch_size':1,'seq_len':1536,'height':512,'width':512,'needs':{'cuda':True},'timeoutSeconds':600,
        'case':previous['case'],'support':previous['support'],'logicalStep':2,'sourceJoint':1,'nextJoint':2,
        'arithmeticSubplans':{'actor':'nativeQimage/text/PtextModLN→Ptextsplice→same58allocation→CPUFP64weightedinput/Vheadmaps→ownfiniteMLP/nextQ',
            'postSeal':'One native-bankAV using actualactor probabilities then saved response/telescope diagnostics'},
        'plannedMatrixMACs':{'actor':1855979520,'postSealBankAV':18874368,'total':1874853888},
        'protocol':'One deferred-value primary with same retainedPnorm/allocation. No full/selectedV input beforeseal, no fullVformation/newcompiler/referencecampaign/fit/gain/normrepair/extraconsumer/suffix/image. Realidentity, finite differences measured; fewdemandedqueries only.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(payload/'request.json',r)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in payload.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same handle')
    cmd=command()[:-5]
    for n in WHEELS:cmd+=['--with','./'+n]
    cmd+=['python','-u','deferred_value_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-deferred-value-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','backend','device','dtype','batch_size','seq_len','height','width','scheduling_contract','arithmeticSubplans']},
        ram_mb=4096,vram_mb=1024,cpu_threads=4,disk_gb=3,est_wall_s=120,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,
        note='DeferredV spherecube2027step2joint1→2;4demandedqueries. NativeBF16Qimage/text/PtextModLN→splice→same58duals/retainedPnormallocation→weightedinputs452.985MMAC+Vheadmaps75.497M→ownnativeMLP/nextQ. Actor1.856B,postsealnativebankAV18.874M,total1.875BMAC. No full/selectedV/oldprobabilities beforeseal; no newcompiler/fullV/refcampaign/fit/gain/normrepair/suffix/image. RAM4096VRAM1024CPU4disk3GB,120sexpected600cap,noretry,selected7tensors245MB. Paid native states/maps/textK/duals/Pnorm.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);started=time.perf_counter();tick=started
    subprocess.run([sys.executable,'-u',str(root/'deferred_value_worker.py'),str(root)],cwd=root,check=True)
    actor_seconds=time.perf_counter()-tick;print('DEFERRED_VALUE_ACTOR_EXITED',flush=True)
    from deferred_value_analysis import execute as analyze
    tick=time.perf_counter();report=analyze(root);analysis_seconds=time.perf_counter()-tick;out=root/'response-result'
    for n in ['request.json','inputs.json','targets.json','parameter-artifacts.json','retrieve_weights.py','resource-basis.json','frozen-plan.md']:
        shutil.copyfile(root/n,out/n)
    save(out/'report.json',{'schema':'tce-deferred-value-result-v1','requestSHA256':r['requestSHA256'],**report,
        'phases':[{'phase':'actor','secondsIncludingFreshProcess':actor_seconds},{'phase':'postSealAnalysis','seconds':analysis_seconds}],
        'workerWallSeconds':time.perf_counter()-started,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
