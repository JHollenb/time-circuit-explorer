"""One four-clock own-state terminal rewrite, no new reference trajectory."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from encoder_atoms import read,save,sha,identity
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['terminal_attention_ledger.py','terminal_trajectory.py','terminal_trajectory_worker_v2.py','terminal_trajectory_ledger.py','terminal_trajectory_analysis.py','terminal_trajectory_study_v2.py',
    'terminal_liveness.py','terminal_capture_guard_v2.py','encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','masked_kv_study.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from publish_complete_masked import checked as native_checked
    parent,pc,pj,pr=checked(HERE.parent/'artifacts/terminal-liveness-v1');old,nc,nj,nr=native_checked(HERE.parent/'artifacts/complete-masked-image-v1')
    intake_path=HERE.parent/'artifacts/terminal-trajectory-intake-v1/intake.json';i=read(intake_path);api=Api()
    if i['intakeSourceSHA256']!=sha(HERE/'terminal_trajectory_intake.py') or i['nativeCustody']!=nc['custody'] or i['nativeJobID']!=nj['job_id']:raise ValueError('Qualified comparator intake differs')
    for job_id,custody in [(i['nativeJobID'],i['nativeCustody']),(i['noiseJobID'],i['noiseCustody'])]:
        current=api.json('GET','/api/jobs/'+job_id)
        if current['state']!='succeeded' or current['payload_custody']!=custody:raise ValueError('Native source/noise custody differs')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    for name in OWNED:
        if name in pr['ownedSources'] and sha(HERE/name)!=pr['sourceHashes'][name]:raise ValueError('Frozen terminal/helper changed:'+name)
        shutil.copyfile(HERE/name,p/name)
    for name in WHEELS:shutil.copyfile(parent/'payload'/name,p/name)
    shutil.copyfile(intake_path,p/'intake.json');shutil.copyfile(HERE.parent/'docs/plans/terminal-trajectory-v2.md',p/'frozen-plan.md')
    s={k:i[k] for k in ['modelRoot','E','noise','schedule','nativeSources','imageAssets','versions','numericalProfile','attentionProcessors']}
    denied=[x['path'] for x in i['targets']['fields'].values()]+[i['targets']['image']['path']]
    s.update(failedPriorJobID='job-f5d33304c764',failedPriorRequestSHA256='2ee8f6d7e5d8ae97b1ffde307497b3fc9961a4570e5e90a20d70a0952a2fc978',deniedNumericalInputs=denied,nativeJobID=i['nativeJobID'],nativeCustody=i['nativeCustody'],noiseJobID=i['noiseJobID'],noiseCustody=i['noiseCustody'],
        parentLivenessJobID=pj['job_id'],parentLivenessCustody=pc['custody'],sourceRecords=i['sourceRecords'],intakeSHA256=sha(intake_path),case='A',
        prompt=nr['prompts'][0]['prompt'],frozenTerminalSourceSHA256=sha(HERE/'terminal_liveness.py'))
    save(p/'inputs.json',s);save(p/'targets.json',i['targets'])
    save(p/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'nativeImageParentJobID':nj['job_id'],'nativeImageParentResult':{k:nj['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'nativeImageParentReservation':nj['reservation'],
        'physicalMapBytesAdded':169869312,'scope':'One whole image model+VAE, four own denoisers; not terminal-only memory scope.',
        'exception':'Measured nativefullimage sameB1/512 parent7926MBRSS9148MBdevice. Fulloriginaltransformer/VAE plus170MBphysicalmaps; reuse14336MBRAM13000MBVRAM4CPU3GBdisk.120sexpected600cap,noretries. No encoder, native shadow or co-resident second model; analysis freshCPU child.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=10,logical_checkpoint_replays=0,
        conditioner_policy='Authenticated saved native E/initialnoise/IDs/schedule, historical encoding paid; no newencoder or noisedraw.',
        extra={'purpose':'Tinytwo-clock CPUfixture;four ownphysicalrewritten denoisers/four originalcommits/one originalVAE, thenfreshsavedtargetanalysis.',
               'modelLoadScope':'One fulloriginalimage transformer/VAE with fixed physicalterminalmaps. Alllive learnedwork charged; no shadowreference.'})
    r={'schema':'tce-terminal-trajectory-request-v2','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'steps':4,'needs':{'cuda':True},'timeoutSeconds':600,
        'expectedCalls':{'denoiser':4,'joint':20,'singlePrefix':76,'reducedTerminal':4,'scheduler':4,'VAE':1},'prospectiveAvoidedTerminalMatrixMACs':231928233984,
        'protocol':'One ownfourclock trajectory from supplied E/noise; onlyfinalsingleblock physicalimage-only rewrite. Ownvelocity→originalscheduler→nextownlatent, no native refresh. One originalVAEdecode, seal+producerexit before savedreference compare. No newnative reference/encoder/noise/fit/sweep or imageinheritance.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'root':str(root),'requestSHA256':r['requestSHA256'],'nativeComparatorJobID':nj['job_id']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same handle')
    cmd=command()[:-5]
    for name in WHEELS:cmd+=['--with','./'+name]
    cmd+=['python','-u','terminal_trajectory_study_v2.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='Own4clock image:fixed physicalterminalrewrite,allcurrentKV+imageQ/MLP/output,ownvelocity→4nativecommits→nextownlatent,1VAE.20joint76singleprefix4reducedterminal;440denoiserlinears100attention,231.928BMACremoved;allother learnedwork paid. SavedE/noise/oneexistingnativecaseA target;no newencoder/reference/noise/fit/sweep. Source-only guard,seal+exit beforeanalysis. Fullimageparent7926RSS9148device+170MBmaps;14336RAM13000VRAM4CPU3GBdisk120sexpect600cap,noretries.'
    if len(note)>500:raise ValueError('Queue note too long')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-terminal-own-trajectory-v2',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','height','width','steps','scheduling_contract']},
        ram_mb=14336,vram_mb=13000,cpu_threads=4,disk_gb=3,est_wall_s=120,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r)
    for name in ['terminal_trajectory_worker_v2.py','terminal_trajectory_analysis.py']:
        subprocess.run([sys.executable,'-u',str(root/name),str(root)],cwd=root,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
