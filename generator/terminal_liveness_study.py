"""Freeze one physical terminal-output reduction, then compare saved targets."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from encoder_atoms import read,save,sha,identity
from flux_clock_study import verify
from terminal_liveness import PARAMETERS

HERE=Path(__file__).resolve().parent
OWNED=['terminal_liveness.py','terminal_liveness_worker.py','terminal_liveness_analysis.py','terminal_liveness_study.py',
       'terminal_capture_guard_v2.py','encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py',
       'complete_masked_study.py','masked_kv_study.py','projected_complement.py','sparse_text_kv.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    parent,collection,job,old=checked(HERE.parent/'artifacts/terminal-native-capture-v2');pub=HERE.parent/'docs/results/terminal-native-capture'
    for name,digest in read(pub/'files.json').items():
        if sha(pub/name)!=digest:raise ValueError('Published native capture differs')
    capture=read(pub/'capture.json');source=read(pub/'state-cut-records.json')['source'];s0=read(pub/'inputs.json');execution=read(pub/'execution.json')
    if source['cutID']!=capture['source']['cutID'] or source['manifest']['execution']['requestSHA256']!=old['requestSHA256']:raise ValueError('Capture source/request differs')
    allowed={source['cutFile']['path']}|{n for page in source['pages'].values() for n in page['files']}
    target=read(pub/'state-cut-records.json')['target'];denied={target['cutFile']['path']}|{n for page in target['pages'].values() for n in page['files']}
    denied|={x['path'] for x in capture['targetFields'].values()};denied-=allowed
    root=Path(root).resolve();root.mkdir(exist_ok=False,parents=True);payload=root/'payload';payload.mkdir()
    for name in OWNED:
        if name in old['ownedSources'] and sha(HERE/name)!=old['sourceHashes'][name]:raise ValueError('Inherited helper changed:'+name)
        shutil.copyfile(HERE/name,payload/name)
    for name in WHEELS:shutil.copyfile(parent/'payload'/name,payload/name)
    shutil.copyfile(HERE.parent/'docs/plans/terminal-output-liveness.md',payload/'frozen-plan.md')
    shutil.copyfile(HERE.parent/'docs/plans/terminal-liveness-execution-intake.md',payload/'execution-intake.md')
    s={k:s0[k] for k in ['modelRoot','nativeSources','imageAssets','versions','numericalProfile','modelConfig']}
    recipes=[x for x in s0['originalParameterArtifacts'] if x['source']['tensor'] in PARAMETERS]
    if len(recipes)!=6:raise ValueError('Six original terminal/head tensors absent')
    s.update(sourceCut=source,sourceCombinedPayloadSHA256=capture['sourceFields']['combined']['payloadSHA256'],
        nativeHeadInputABI=execution['typedActs'][-1]['sourceFrame']['slots']['image']['abi'],originalParameterArtifacts=recipes,
        deniedNumericalInputs=sorted(denied|set(s0['deniedNumericalInputs'])),parentJobID=job['job_id'],parentCustody=collection['custody'],
        parentRecords={n:sha(pub/n) for n in ['files.json','capture.json','capture-seal.json','execution.json','report.json','retained-array-verification.json','state-cut-records.json']})
    save(payload/'inputs.json',s);save(payload/'parameter-artifacts.json',{'artifacts':recipes})
    save(payload/'targets.json',{'fields':{n:capture['targetFields'][n] for n in ['image','velocity']},'sourceCombined':capture['sourceFields']['combined'],
        'sourceCutID':source['cutID'],'targetCutID':target['cutID'],'scope':'Saved targets; first numerical opens only in fresh analysis child after candidate seal and producer exit. No new native reference.'})
    save(payload/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',Api(),seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredParent':{k:job['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'originalParameterBytes':283902464,
        'physicalSliceBytes':169869312,'residentNativeBlocks':0,'physicalAffines':5,'attentionCalls':1,
        'exception':'Parent streams491MBmax nativeblock at1999MBRSS1196MBdevice. This singleterminal+head loads284MBoriginal plus170MBphysical maps, smaller activations.4096MBRAM2048MBVRAM4CPU1GBdisk,60sexpected240cap,no retries. Fresh saved-only analysis child; no co-resident full model or reference.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=4,logical_checkpoint_replays=0,
        conditioner_policy='Authenticated complete13slot native terminalsource; sourceprefix/clock/capture/failedpredecessor remainpaid.',
        extra={'modelLoadScope':'Sixoriginaltensors only, physicalKV/QGU maps, nativefinalhead; no fullblock/model construction.',
               'purpose':'Tiny suppliedtoy mechanics, physicalterminalimage producer, ownvelocityconsumer, freshsavedtargetanalysis. No fullnative/reference/image.'})
    r={'schema':'tce-terminal-liveness-request-v1','ownedSources':OWNED,'sourceHashes':{p.name:sha(p) for p in payload.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'needs':{'cuda':True},'timeoutSeconds':240,
        'weightRowIntervals':{'KV':[[3072,9216]],'QGU':[[0,3072],[9216,27648]]},'candidateAffineMACs':135712997376,'candidateAttentionMACs':9663676416,
        'nativeBlockAndHeadMatrixMACs':203358732288,'avoidedMatrixMACs':57982058496,
        'protocol':'One allcurrentKV+imageQ/GU/output producer with physical originalweight slicing, ownimage→originalvelocityhead. Seal and producerexit before fresh savedtargetanalysis. No textQ/MLP/output allocation, fullnative/reference, prefix, encoder,clocknetwork,scheduler,VAE,image,fit or sweep.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(payload/'request.json',r)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'root':str(root),'requestSHA256':r['requestSHA256'],'sourceCutID':source['cutID']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted: inspect same job')
    cmd=command()[:-5]
    for name in WHEELS:cmd+=['--with','./'+name]
    cmd+=['python','-u','terminal_liveness_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='Physicalterminal19:all1536KV,image1024Q/GU/output,ownimage→originalhead.5affines135.713B+attention9.664BMAC;57.982Bremoved vs nativeblock/head.6originaltensors284MB+170MBslices; no fullmodel/prefix/reference/encoder/clocknet/scheduler/VAE/image/fit. Exact13slot sourcepageallowlist; targetdenied untilseal+exit, freshsavedanalysis. Parent1999RSS1196device;4096RAM2048VRAM4CPU1GBdisk,60sexpect240cap,noretries. Tinytoy first; noqualityabort.'
    if len(note)>500:raise ValueError('Queue note too long')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-terminal-output-liveness-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','height','width','scheduling_contract']},
        ram_mb=4096,vram_mb=2048,cpu_threads=4,disk_gb=1,est_wall_s=60,timeout_s=240,detach=True,retry_on_kill=False,preflight=True,note=note)
    result={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',result);return result


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r)
    for name in ['terminal_liveness_worker.py','terminal_liveness_analysis.py']:
        subprocess.run([sys.executable,'-u',str(root/name),str(root)],cwd=root,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        result=collect(a.root)
    else:result=globals()[a.action](a.root)
    print(json.dumps(result))
