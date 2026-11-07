"""One guarded source-state reader, sealed before historical comparison."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read,save,sha,identity
from flux_clock_study import verify
from deferred_value_study import OWNED as PREVIOUS_OWNED,WHEELS

HERE=Path(__file__).resolve().parent
OWNED=['source_reader.py','source_reader_worker.py','source_reader_analysis.py','source_reader_study.py']+PREVIOUS_OWNED


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from projected_complement import fields
    from source_reader_worker import PARAMETERS
    from image_dual_analysis import NAMES
    parent,collection,job,previous=checked(HERE.parent/'artifacts/deferred-value-v1')
    older,oc,oj,op=checked(HERE.parent/'artifacts/propagated-context-v1')
    api=Api()
    for c,j in [(collection,job),(oc,oj)]:
        live=api.json('GET','/api/jobs/'+j['job_id'])
        if live['state']!='succeeded' or live['payload_custody']!=c['custody']:raise ValueError('Parent custody changed')
    p=parent/'result'
    if sha(p/'report.json')!='80b550ce6fa94e2fef540ae0073a82c322777cc7078a8a5fb4d74282f0f3bed1':raise ValueError('Frozen parent differs')
    actor,s,targets=[read(p/n) for n in ['actor.json','inputs.json','targets.json']]
    if read(p/'actor-seal.json')['actorSHA256']!=sha(p/'actor.json'):raise ValueError('Parent seal differs')
    targets['assisted']={n:{**actor['source']['fields'],**actor['next_query']['fields']}[n] for n in NAMES}
    receipt=read(parent/'retained-array-verification.json')
    for x in fields({'targets':targets,'selected':s['selected'],'banks':s['banks']}):
        if receipt['files'][x['path']]['sha256']!=x['sha256']:raise ValueError('Parent operand custody absent')
    denied=set(s['deniedNumericalInputs'])|{x['path'] for x in fields({'actor':actor,'report':read(p/'report.json'),
        'targets':targets,'selected':s['selected'],'banks':s['banks'],'reuse':s['reuse'],'textKeySlice':s['textKeySlice']})}
    remote=job['result']['diagnostics']['cwd']+'/response-result/'
    s.update(sourceReaderParentJobID=job['job_id'],sourceReaderParentCustody=collection['custody'],
        sourceReaderParentRecords={n:{'path':remote+n,'sha256':sha(p/n)} for n in ['actor.json','actor-seal.json','actor-execution.json','report.json','request.json','inputs.json','targets.json']},
        deniedNumericalInputs=sorted(denied))
    artifacts=read(older/'result/parameter-artifacts.json')
    if read(older/'result/inputs.json')['imageAssets']!=s['imageAssets']:raise ValueError('Original parameter/model source differs')
    artifacts['artifacts']=[x for x in artifacts['artifacts'] if x['source']['tensor'] in PARAMETERS]
    if {x['source']['tensor'] for x in artifacts['artifacts']}!=set(PARAMETERS):raise ValueError('Required original maps incomplete')
    s['originalParameterArtifacts']=artifacts['artifacts']
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for n in OWNED:
        if n in previous['ownedSources'] and sha(HERE/n)!=previous['sourceHashes'][n]:raise ValueError('Frozen helper changed')
        shutil.copyfile(HERE/n,payload/n)
    for n in WHEELS:shutil.copyfile(parent/'payload'/n,payload/n)
    shutil.copyfile(p/'retrieve_weights.py',payload/'retrieve_weights.py')
    shutil.copyfile(HERE.parent/'docs/plans/source-input-deferred-reader.md',payload/'frozen-plan.md')
    save(payload/'inputs.json',s);save(payload/'targets.json',targets);save(payload/'parameter-artifacts.json',artifacts)
    preview=_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda')
    save(payload/'resource-basis.json',{'intentFirstPreview':preview,'measuredParent':job['result'],
        'geometry':{'distinctBF16ParameterBytes':301990912,'fullKeyBF16Bytes':9437184,'queryDualTableBytes':0,'newFullValueBanks':0},
        'exception':'Selected13tensors,one actor not fullmodel. Measured62:1868.1MBRSS626MBdevice41.23s. New K/Q maps+norms56.624MBBF16/nativeK intermediates<50MB; removes153MBFP64duals. RAM4096VRAM1024CPU4disk3GB,120sexpected600cap. No extra consumer/fullV/referencecampaign.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=11,logical_checkpoint_replays=0,
        conditioner_policy='Authenticated Qimage/text/context and Ptext/context source cuts only; original13maps. Prefix/clockcontext production paid.',
        extra={'modelLoadScope':'Native Q/K formation2Kmaps/2norms+Qmap/norm,2Vmaps,5consumer tensors; no full transformer.',
            'targetIsolation':'All archived selectedQ/K/V/R/mod/rope,banks,duals,norms,oldfields denied before actor seal/exit. Own source→QK→allocation→deferredV→MLP→nextQ.'})
    r={'schema':'tce-source-reader-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in payload.iterdir()},
        'scheduling_contract':contract,'model':'flux2-klein-4b-distilled','backend':'diffusers','device':'cuda','dtype':'bfloat16',
        'batch_size':1,'seq_len':1536,'height':512,'width':512,'needs':{'cuda':True},'timeoutSeconds':600,
        'case':previous['case'],'support':previous['support'],'logicalStep':2,'sourceJoint':1,'nextJoint':2,
        'arithmeticSubplans':{'actor':'Sourceonly nativeModLN/selectedQ/fullhybridK/RMS/RoPE→CPUFP64QK/softmax/deferredV→ownnativeMLP/nextQ',
            'postSeal':'Saved source/operand/consumer comparisons only, no new referenceAV or learned maps'},
        'plannedMatrixMACs':{'actor':15495856128,'postSealAnalysis':0,'total':15495856128},
        'protocol':'One source-state interface primary; no archived selectedoperands/nativebanks/duals/Pnorm beforeseal. Pay fullK/nativeQ,freeze deferredV/consumer. Preserve finite mismatches, no new fullV/refcampaign/fit/normapproximation/extraconsumer/suffix/image.'}
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
    cmd+=['python','-u','source_reader_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-source-reader-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','backend','device','dtype','batch_size','seq_len','height','width','scheduling_contract','arithmeticSubplans']},
        ram_mb=4096,vram_mb=1024,cpu_threads=4,disk_gb=3,est_wall_s=120,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,
        note='Source-state fourqueryreader spherecube2027step2joint1→2. Qimage/text+Ptext/context only, own selectedPquery/fullhybridK nativeBF16; liveCPUFP64QK/softmax→deferredV→ownMLP/T/nextQ. K14.496B+Q37.749M+QK18.874M+V528.482M+consumer415.236M,total15.496BMAC. No archivedselectedoperands/banks/duals/Pnorm/oldoutputs beforeseal, no fullV/refcampaign/fit/suffix/image. Selected13tensors302MB, measured62:1868RSS626device41s;RAM4096VRAM1024CPU4disk3GB,120expected600cap,noretry. Nativeprefix/clockcontext/maps paid.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);started=time.perf_counter();tick=started
    subprocess.run([sys.executable,'-u',str(root/'source_reader_worker.py'),str(root)],cwd=root,check=True)
    actor_seconds=time.perf_counter()-tick;print('SOURCE_READER_ACTOR_EXITED',flush=True)
    from source_reader_analysis import execute as analyze
    tick=time.perf_counter();report=analyze(root);analysis_seconds=time.perf_counter()-tick;out=root/'response-result'
    for n in ['request.json','inputs.json','targets.json','parameter-artifacts.json','retrieve_weights.py','resource-basis.json','frozen-plan.md']:
        shutil.copyfile(root/n,out/n)
    save(out/'report.json',{'schema':'tce-source-reader-result-v1','requestSHA256':r['requestSHA256'],**report,
        'phases':[{'phase':'actor','secondsIncludingFreshProcess':actor_seconds},{'phase':'postSealAnalysis','seconds':analysis_seconds}],
        'workerWallSeconds':time.perf_counter()-started,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
