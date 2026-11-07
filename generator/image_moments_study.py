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
OWNED=['image_partition_moments.py','image_moments_worker.py','image_moments_ledger.py','image_moments_analysis.py','image_moments_study.py','test_image_partition_moments.py',
    'fixed_text_memory.py','terminal_attention_ledger.py','terminal_liveness.py','terminal_capture_guard_v2.py','encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','masked_kv_study.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from publish_complete_masked import checked as native_checked
    parent,pc,pj,pr=checked(HERE.parent/'artifacts/partition-feedback-v1');old,nc,nj,nr=native_checked(HERE.parent/'artifacts/complete-masked-image-v1')
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
    shutil.copyfile(intake_path,p/'intake.json');shutil.copyfile(HERE.parent/'docs/plans/image-partition-moments.md',p/'frozen-plan.md')
    s={k:i[k] for k in ['modelRoot','E','noise','schedule','nativeSources','imageAssets','versions','numericalProfile','attentionProcessors']}
    prior_candidate=read(parent/'result/candidate.json')
    denied=sorted(set([x['path'] for x in i['targets']['fields'].values()]+[i['targets']['image']['path']]+[x['path'] for x in prior_candidate['fields'].values()]+[prior_candidate['image']['path']]))
    s.update(deniedNumericalInputs=denied,nativeJobID=i['nativeJobID'],nativeCustody=i['nativeCustody'],noiseJobID=i['noiseJobID'],noiseCustody=i['noiseCustody'],
        priorTrajectoryJobID=pj['job_id'],priorTrajectoryCustody=pc['custody'],priorTrajectoryReportSHA256=sha(parent/'result/report.json'),priorTrajectoryReportPath=pj['result']['diagnostics']['cwd']+'/response-result/report.json',sourceRecords=i['sourceRecords'],intakeSHA256=sha(intake_path),case='A',
        prompt=nr['prompts'][0]['prompt'],frozenTerminalSourceSHA256=sha(HERE/'terminal_liveness.py'))
    save(p/'inputs.json',s);save(p/'targets.json',i['targets'])
    save(p/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'nativeImageParentJobID':nj['job_id'],'nativeImageParentResult':{k:nj['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'nativeImageParentReservation':nj['reservation'],
        'maxActivePhysicalSliceBytes':169869312,'oneTextScoreFP32Bytes':25165824,'scoreAndSoftmaxLogicalBytes':50331648,'capturedBlockOutputsPerClockBytes':351043584,'perBlockMeanCovarianceBytes':1585152,'activeMapScope':'Logical one-mapset callback bound, not independent GPU residency measurement','scope':'One full image model/VAE; each clock retains351MB typed outputs including moments. One25.2MB TTscore plussoftmax/centeredkey/FP32workspace during liveblocks, released before terminal170MBmapset. ~1.55GBnewarrays,3GBdisk. No compiler/secondmodel.',
        'priorTrajectoryJobID':pj['job_id'],'priorTrajectoryResult':{k:pj['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'exception':'Cycle69sameB1/512 sampled8442.8MBRSS9416MBdevice;actor CUDAallocated8.62GB,reserved9.54GB. New351MBcurrentclockoutputs,25.2MBTTscore+softmax/centeredkey/FP32workspace,one170MBterminalmap. Same14336MBRAM13000MBVRAM4CPU3GBdisk envelope,180sexpected600cap,noretries. No compiler/secondmodel/shadow/reference.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=10,logical_checkpoint_replays=0,
        conditioner_policy='Authenticated saved native E/initialnoise/IDs/schedule, historical encoding paid; no newencoder or noisedraw.',
        extra={'purpose':'TinyCPUfixture;4ownmomentdenoisers/4originalcommits/1originalVAE,thenfreshsavedtarget/momentanalysis.',
               'modelLoadScope':'One fulloriginaltransformer/VAE;96each covariance/querycov/querymean/TTscores/TTAV=480FP32products,100imageSDPA;one170MBterminalmapset,351MBclockoutputs. Allmaps/decoderpaid,no shadowreference.'})
    r={'schema':'tce-image-partition-moments-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'steps':4,'needs':{'cuda':True},'timeoutSeconds':600,
        'expectedCalls':{'denoiser':4,'joint':20,'single':76,'terminal':4,'scheduler':4,'VAE':1},'prospectiveFirstImageMACs':21120701956096,'prospectiveAvoidedMatrixMACs':483032825856,
        'protocol':'Owncoupledtext/imageevolution;postRoPEpopulationmean/fullcovariance→logNI+qmean/sqrtD+qquad/2D;TTFP32scores→postscale→logsumexp/conditionalAV→ownalpha→BF16textread,96liveupdates. Nativefullbankimageattention,originaljointsequential/singleparallelwrites,lasttextKVonly.4owncommits+1VAE. Seal+exitbeforeoldcompare. Alloriginalmaps/keys/decoderpaid;no fulltextimagegrid/compiler/nativehiddenstates/ref/fit/sweep/zero-neuralclaim.'}
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
    cmd+=['python','-u','image_moments_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='Livekeymoments:4ownclocks,440affines/100imageSDPA/480FP32products,4commits+1VAE. FullpostRoPEpopulationcov;TTconditionalAVtimesalpha.21.121TMACprospective. SavedE/noise,no fulltextimagegrid/shadow/ref/fit/sweep.351MBclockoutputs+25.2MBTTscore/softmax;170MBterminalmap. Prior8443MBRSS/9416MBdevice;14336RAM13000VRAM4CPU3GBdisk,180sexpected600cap,noretries. Seal+exitbeforeanalysis.'
    if len(note)>500:raise ValueError('Queue note too long')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-image-partition-moments-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','height','width','steps','scheduling_contract']},
        ram_mb=14336,vram_mb=13000,cpu_threads=4,disk_gb=3,est_wall_s=180,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r)
    for name in ['image_moments_worker.py','image_moments_analysis.py']:
        subprocess.run([sys.executable,'-u',str(root/name),str(root)],cwd=root,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
