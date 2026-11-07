"""One live image-partition-moments trajectory against the existing comparator."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from encoder_atoms import read,save,sha,identity
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['gate_first_trajectory.py','gate_first_mlp.py','image_partition_moments.py','gate_trajectory_worker.py','gate_trajectory_ledger.py','gate_trajectory_analysis.py','gate_trajectory_study.py','test_gate_first_trajectory.py',
    'fixed_text_memory.py','terminal_attention_ledger.py','terminal_liveness.py','terminal_capture_guard_v2.py','encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','masked_kv_study.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from publish_complete_masked import checked as native_checked
    parent,pc,pj,pr=checked(HERE.parent/'artifacts/image-partition-moments-v1');old,nc,nj,nr=native_checked(HERE.parent/'artifacts/complete-masked-image-v1')
    intake_path=HERE.parent/'artifacts/terminal-trajectory-intake-v1/intake.json';i=read(intake_path);api=Api()
    if sha(intake_path)!=pr['sourceHashes']['intake.json'] or i['intakeSourceSHA256']!=sha(HERE/'terminal_trajectory_intake.py') or i['nativeCustody']!=nc['custody'] or i['nativeJobID']!=nj['job_id']:raise ValueError('Qualified comparator intake differs')
    for job_id,custody in [(i['nativeJobID'],i['nativeCustody']),(i['noiseJobID'],i['noiseCustody'])]:
        current=api.json('GET','/api/jobs/'+job_id)
        if current['state']!='succeeded' or current['payload_custody']!=custody:raise ValueError('Native source/noise custody differs')
    frozen71=read(HERE.parent/'artifacts/gate-first-terminal-v1/request.json')
    for name in OWNED:
        if name in frozen71['ownedSources'] and sha(HERE/name)!=frozen71['sourceHashes'][name]:raise ValueError('Frozen cycle71 helper changed:'+name)
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    for name in OWNED:
        if name in pr['ownedSources'] and sha(HERE/name)!=pr['sourceHashes'][name]:raise ValueError('Frozen terminal/helper changed:'+name)
        shutil.copyfile(HERE/name,p/name)
    for name in WHEELS:shutil.copyfile(parent/'payload'/name,p/name)
    shutil.copyfile(intake_path,p/'intake.json');shutil.copyfile(HERE.parent/'docs/plans/gate-first-single-trajectory.md',p/'frozen-plan.md')
    s={k:i[k] for k in ['modelRoot','E','noise','schedule','nativeSources','imageAssets','versions','numericalProfile','attentionProcessors']}
    prior_candidate=read(parent/'result/candidate.json')
    denied=sorted(set([x['path'] for x in i['targets']['fields'].values()]+[i['targets']['image']['path']]+[x['path'] for x in prior_candidate['fields'].values()]+[prior_candidate['image']['path']]))
    s.update(deniedNumericalInputs=denied,nativeJobID=i['nativeJobID'],nativeCustody=i['nativeCustody'],noiseJobID=i['noiseJobID'],noiseCustody=i['noiseCustody'],
        priorTrajectoryJobID=pj['job_id'],priorTrajectoryCustody=pc['custody'],priorTrajectoryReportSHA256=sha(parent/'result/report.json'),priorTrajectoryReportPath=pj['result']['diagnostics']['cwd']+'/response-result/report.json',sourceRecords=i['sourceRecords'],intakeSHA256=sha(intake_path),case='A',
        prompt=nr['prompts'][0]['prompt'],frozenTerminalSourceSHA256=sha(HERE/'terminal_liveness.py'))
    save(p/'inputs.json',s);save(p/'targets.json',i['targets'])
    hosts=api.json('GET','/api/hosts');host=next(x for x in hosts if x['name']=='beast')
    if 14336*1.1+host['admission']['ram_margin_mb']>=host['ram_total_mb']:raise ValueError('Physical RAM guard envelope impossible')
    save(p/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'priorTrajectoryJobID':pj['job_id'],'priorTrajectoryResult':{k:pj['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'priorTrajectoryReservation':pj['reservation'],
        'liveHost':{k:host[k] for k in ['name','active_jobs','admission','ram_total_mb','vram_total_mb','telemetry']},
        'maxActivePhysicalMapBytes':113246208,'sourceNormFP32Bytes':1474560,'maximumTwoBankNormTemporaryBytes':226492416,'oneFullGateBF16Bytes':28311552,
        'selectorFP32SquareAndWeightedBytes':113246208,'currentClockCapturedBaseBytes':351043584,'selectorRetentionBytesUpperBoundPerClock':5000000,
        'scope':'Original fullmodel/VAE, one113MBbase OR47MBselectedmapset at a time, compact norms/80selectionvectors; no fullgate capture. Full70history8308.8MBRSS9476MBdevice, <200MBnewselector/norm workspace. Original14336RAM13000VRAM4CPU3GBdisk envelope retained from measured full trajectory. 240sexpected600cap, no retries; no secondmodel/reference. Logical bytes not simultaneous measuredGPUresidency.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=11,logical_checkpoint_replays=0,
        conditioner_policy='Authenticated saved native E/initialnoise/IDs/schedule, historical encoding paid; no newencoder or noisedraw.',
        extra={'purpose':'TinyCPUfixture;sourceNorms once;4own gatefirstdenoisers/4originalcommits/1originalVAE,thenfreshsavedtarget/selectoranalysis.',
               'modelLoadScope':'One fulloriginaltransformer/VAE;80quarter selections,520affines,100imageSDPA,480FP32momentproducts. Ephemeral113MBbase/47MBselectedmaps;351MBclock outputs+compactvectors. All learned maps anddecoderpaid.'})
    r={'schema':'tce-gate-first-trajectory-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'steps':4,'needs':{'cuda':True},'timeoutSeconds':600,
        'expectedCalls':{'denoiser':4,'joint':20,'single':76,'terminal':4,'scheduler':4,'VAE':1},'prospectiveFirstImageMACs':15989289779200,'prospectiveAvoidedMatrixMACs':5614445002752,
        'protocol':'Current BF16 fullG, FP32 source norm/current gate energy score, stable sourceindex ties, fixed2304/9216commonchannels. Nonterminal1536rows QKVG then selectedU/down; terminal1024imageQG/U/down with1536KV. Owncoupledtext/image and cycle70momentfeedback,4owncommits1originalVAE. Source norms once,80supports,ephemeralphysicalmaps. Seal+exit beforeexistingnative/prior70compare. No hiddenrefresh/shadow/reference/fit/sweep/balancing/fallback/zero-neuralclaim.'}
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
    cmd+=['python','-u','gate_trajectory_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='Gatefirstquarter all20singles/4ownclocks:80supports,520affines,100imageSDPA,480momentproducts,4commits1VAE.15.989TMACprospective. Source norms once,113MBbase OR47MBselectedmaps,351MBclockoutputs+compactvectors;fullgatesnotcaptured. Prior70:8309MBRSS9476MBdevice.14336RAM13000VRAM4CPU3GBdisk240sexpected600cap,noretries. SavedE/noise, nohiddenrefresh/shadow/ref/fit/sweep. Seal+exitbeforeanalysis.'
    if len(note)>500:raise ValueError('Queue note too long')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-gate-first-trajectory-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','height','width','steps','scheduling_contract']},
        ram_mb=14336,vram_mb=13000,cpu_threads=4,disk_gb=3,est_wall_s=240,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r)
    for name in ['gate_trajectory_worker.py','gate_trajectory_analysis.py']:
        subprocess.run([sys.executable,'-u',str(root/name),str(root)],cwd=root,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
