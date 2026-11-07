"""One image-blind text compiler and own image actor against existing history."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from encoder_atoms import read,save,sha,identity
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['prompt_program.py','prompt_program_compiler.py','prompt_program_actor.py','prompt_program_ledger.py','prompt_program_analysis.py','prompt_program_study.py','test_prompt_program.py',
    'terminal_attention_ledger.py','terminal_capture_guard_v2.py','encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','masked_kv_study.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from publish_complete_masked import checked as native_checked
    parent,pc,pj,pr=checked(HERE.parent/'artifacts/fixed-text-memory-v1');old,nc,nj,nr=native_checked(HERE.parent/'artifacts/complete-masked-image-v1')
    intake_path=HERE.parent/'artifacts/terminal-trajectory-intake-v1/intake.json';i=read(intake_path);api=Api()
    if sha(intake_path)!=pr['sourceHashes']['intake.json'] or i['intakeSourceSHA256']!=sha(HERE/'terminal_trajectory_intake.py') or i['nativeCustody']!=nc['custody'] or i['nativeJobID']!=nj['job_id']:raise ValueError('Qualified comparator intake differs')
    for job_id,custody in [(i['nativeJobID'],i['nativeCustody']),(i['noiseJobID'],i['noiseCustody'])]:
        current=api.json('GET','/api/jobs/'+job_id)
        if current['state']!='succeeded' or current['payload_custody']!=custody:raise ValueError('Native source/noise custody differs')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    for name in OWNED:
        if name in pr['ownedSources'] and sha(HERE/name)!=pr['sourceHashes'][name]:raise ValueError('Frozen terminal/helper changed:'+name)
        shutil.copyfile(HERE/name,p/name)
    for name in WHEELS:shutil.copyfile(parent/'payload'/name,p/name)
    shutil.copyfile(intake_path,p/'intake.json');shutil.copyfile(HERE.parent/'docs/plans/prompt-only-text-compiler.md',p/'frozen-plan.md')
    s={k:i[k] for k in ['modelRoot','E','noise','schedule','nativeSources','imageAssets','versions','numericalProfile','attentionProcessors']}
    prior_candidate=read(parent/'result/candidate.json')
    denied=sorted(set([x['path'] for x in i['targets']['fields'].values()]+[i['targets']['image']['path']]+[x['path'] for x in prior_candidate['fields'].values()]+[prior_candidate['image']['path']]))
    s.update(deniedNumericalInputs=denied,nativeJobID=i['nativeJobID'],nativeCustody=i['nativeCustody'],noiseJobID=i['noiseJobID'],noiseCustody=i['noiseCustody'],
        priorTrajectoryJobID=pj['job_id'],priorTrajectoryCustody=pc['custody'],priorTrajectoryReportSHA256=sha(parent/'result/report.json'),sourceRecords=i['sourceRecords'],intakeSHA256=sha(intake_path),case='A',
        prompt=nr['prompts'][0]['prompt'],frozenLawSourceSHA256=sha(HERE/'prompt_program.py'))
    save(p/'inputs.json',s);save(p/'targets.json',i['targets'])
    save(p/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'nativeImageParentJobID':nj['job_id'],'nativeImageParentResult':{k:nj['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'nativeImageParentReservation':nj['reservation'],
        'maxActivePhysicalSliceBytes':37748736,'logicalCacheBytes':629145600,'perClockCacheGPUBytes':157286400,'retainedTextStateBytes':314572800,'activeMapScope':'Logical one-mapset callback bound, not independent GPU residency measurement','scope':'Sequential compiler full model then fresh actor full model+VAE; <=oneclock157MBKV outputs live in compiler/actor plus small current state. ~950MBcompiler/150MBactor diagnostic arrays on disk. Model weights never co-resident across children.',
        'priorTrajectoryJobID':pj['job_id'],'priorTrajectoryResult':{k:pj['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'exception':'Cycle67 sameB1/512 measured8290.5MBRSS9116MBdevice,119.85s. Compiler then actor sequential, one37.7MBterminalslice plus157MBcurrentclockcache, ~1.2GBretainedarrays. Reuse14336RAM13000VRAM4CPU3GBdisk;180sexpected600cap,noretries. Models never co-resident; noencoder/shadow/newreference.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=14,logical_checkpoint_replays=0,
        conditioner_policy='Authenticated saved native E/initialnoise/IDs/schedule, historical encoding paid; no newencoder or noisedraw.',
        extra={'purpose':'TinyCPUfixture;4textcompilerclocks then freshactor4owndenoisers/4commits/1VAE then freshsavedtargetanalysis.',
               'modelLoadScope':'Compiler full originaltransformer then fresh actor fulltransformer/VAE. One37.7MBcompilerKVslice, currentclock157MBKVcache; all learnedwork paid.'})
    r={'schema':'tce-prompt-program-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'steps':4,'needs':{'cuda':True},'timeoutSeconds':600,
        'expectedCalls':{'denoiser':4,'reducedJoint':20,'reducedSingle':80,'scheduler':4,'VAE':1},'prospectiveCompilerMatrixMACs':6223407611904,'prospectiveKVCacheBytes':629145600,'prospectiveCompilerPlusOneImageMACs':21062568902656,'prospectiveAvoidedMatrixMACs':541165879296,
        'protocol':'Compiler startsC(E) independentlyperclock;originaltextnorm/projection/gates/MLP withtext-onlyattention. CacheincomingnormalizedrotatedK/originalV at100layers,96textupdates,lasttextKVonly. Freshactorreadsowncache+ownimageKV,4owncommits+1VAE. Compilerinumericalallowlistexcludesnoise/imageIDs/nativehiddenstates. Seal+exitbeforeactor;actorseal+exitbeforesavedcomparison. No oracle/fit/newreference/sweep;architectural+finitegeometrycandidate,alllearnedworkpaid.'}
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
    cmd+=['python','-u','prompt_program_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='Text-only depth program:4clocks/100KV/96updates, incomingK/Vsealed;freshactor4ownsteps+1VAE. Compiler6.223TMAC/629MBcache;21.063Tfirstimageprospective. No image/noise incompiler, no shadow/ref/fit/sweep. Sequentialmodels;157MBcurrentclockKV,one37.7MBslice. Prior8291MBRSS/9116MBdevice.14336RAM/13000VRAM4CPU3GBdisk,180sexpected600cap,noretries. Seal+childexitbeforeactor/analysis.'
    if len(note)>500:raise ValueError('Queue note too long')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-prompt-program-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','height','width','steps','scheduling_contract']},
        ram_mb=14336,vram_mb=13000,cpu_threads=4,disk_gb=3,est_wall_s=180,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r)
    for name in ['prompt_program_compiler.py','prompt_program_actor.py','prompt_program_analysis.py']:
        subprocess.run([sys.executable,'-u',str(root/name),str(root)],cwd=root,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
