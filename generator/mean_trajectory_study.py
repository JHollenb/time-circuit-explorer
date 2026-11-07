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
OWNED=['mean_up_trajectory.py','gate_first_trajectory.py','gate_first_mlp.py','mean_up_completion.py','image_partition_moments.py','mean_trajectory_worker.py','mean_trajectory_ledger.py','mean_trajectory_analysis.py','mean_trajectory_study.py','test_mean_up_trajectory.py',
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
    second,sc,sj,sr=checked(HERE.parent/'artifacts/gate-first-trajectory-v1')
    local,lc,lj,lr=checked(HERE.parent/'artifacts/current-stream-mean-up-v1')
    for rr in [sr,lr]:
        for name in OWNED:
            if name in rr['ownedSources'] and sha(HERE/name)!=rr['sourceHashes'][name]:raise ValueError('Frozen reused source changed:'+name)
    frozen71=read(HERE.parent/'artifacts/gate-first-terminal-v1/request.json')
    for name in OWNED:
        if name in frozen71['ownedSources'] and sha(HERE/name)!=frozen71['sourceHashes'][name]:raise ValueError('Frozen cycle71 helper changed:'+name)
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    for name in OWNED:
        if name in pr['ownedSources'] and sha(HERE/name)!=pr['sourceHashes'][name]:raise ValueError('Frozen terminal/helper changed:'+name)
        shutil.copyfile(HERE/name,p/name)
    for name in WHEELS:shutil.copyfile(parent/'payload'/name,p/name)
    shutil.copyfile(intake_path,p/'intake.json');shutil.copyfile(HERE.parent/'docs/plans/mean-up-own-trajectory.md',p/'frozen-plan.md')
    s={k:i[k] for k in ['modelRoot','E','noise','schedule','nativeSources','imageAssets','versions','numericalProfile','attentionProcessors']}
    prior_candidate=read(parent/'result/candidate.json')
    comparisons={'moment70':{'jobID':pj['job_id'],'custody':pc['custody'],'candidatePath':pj['result']['diagnostics']['cwd']+'/response-result/candidate.json','candidateSHA256':sha(parent/'result/candidate.json'),'reportPath':pj['result']['diagnostics']['cwd']+'/response-result/report.json','reportSHA256':sha(parent/'result/report.json')},'quarter72':{'jobID':sj['job_id'],'custody':sc['custody'],'candidatePath':sj['result']['diagnostics']['cwd']+'/response-result/candidate.json','candidateSHA256':sha(second/'result/candidate.json'),'reportPath':sj['result']['diagnostics']['cwd']+'/response-result/report.json','reportSHA256':sha(second/'result/report.json')}}
    comparison_names=['velocity'+str(k) for k in range(4)]+['latent'+str(k+1) for k in range(4)]+['pngRGB']
    for label,directory in [('moment70',parent),('quarter72',second)]:
        record=comparisons[label];previous=read(directory/'result/candidate.json')
        record['fields']={name:previous['fields'][name] for name in comparison_names}
        record['sourceFiles']={'candidate':{'path':record['candidatePath'],'sha256':record['candidateSHA256']},'report':{'path':record['reportPath'],'sha256':record['reportSHA256']}}
    save(p/'comparators.json',comparisons)
    denied=sorted(set([x['path'] for x in read(second/'result/candidate.json')['fields'].values()]+[read(second/'result/candidate.json')['image']['path']]+[x['path'] for x in read(local/'result/candidate.json')['fields'].values()]+[x['path'] for x in i['targets']['fields'].values()]+[i['targets']['image']['path']]+[x['path'] for x in prior_candidate['fields'].values()]+[prior_candidate['image']['path']]))
    s.update(deniedNumericalInputs=denied,nativeJobID=i['nativeJobID'],nativeCustody=i['nativeCustody'],noiseJobID=i['noiseJobID'],noiseCustody=i['noiseCustody'],
        priorTrajectoryJobID=pj['job_id'],priorTrajectoryCustody=pc['custody'],priorTrajectoryReportSHA256=sha(parent/'result/report.json'),priorTrajectoryReportPath=pj['result']['diagnostics']['cwd']+'/response-result/report.json',sourceRecords=i['sourceRecords'],intakeSHA256=sha(intake_path),case='A',
        prompt=nr['prompts'][0]['prompt'],frozenTerminalSourceSHA256=sha(HERE/'terminal_liveness.py'))
    save(p/'inputs.json',s);save(p/'targets.json',i['targets'])
    hosts=api.json('GET','/api/hosts');host=next(x for x in hosts if x['name']=='beast')
    if 14336*1.1+host['admission']['ram_margin_mb']>=host['ram_total_mb']:raise ValueError('Physical RAM guard envelope impossible')
    save(p/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'priorTrajectoryJobID':pj['job_id'],'priorTrajectoryResult':{k:pj['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'priorTrajectoryReservation':pj['reservation'],
        'liveHost':{k:host[k] for k in ['name','active_jobs','admission','ram_total_mb','vram_total_mb','telemetry']},
        'maxActivePhysicalMapBytes':113246208,'additionalRetainedArrayBytes':8729346048,'activationAndProductBytes':4454350848,
        'additionalRetainedBytesByField':{'normalized':742391808,'activation':2227175424,'upRaw':556793856,'completedUp':2227175424,'product':2227175424,'streamMeanFP32':1916928,'streamMean':958464,'anchorUp':2875392,'preResidualUpdate':742391808,'gate':491520},
        'completionActs':80,'completionOutputABIs':1668,'anchors':156,'denoiserAffines':600,
        'scope':'Measured full70/72 trajectory reservation14336RAM13000VRAM4CPU. Additional8.729GBcompletebanks streamed/fsyncednearcompute, no growingGPUhistory; base outputs1.56GB plus differences/metadata under16GBdisk. One physicalmapset at a time113MBmax. Expected360s cap840s no retries; one model/no new reference. Logical sizes are not measured simultaneous residency.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=91,logical_checkpoint_replays=0,
        conditioner_policy='Authenticated saved native E/initialnoise/IDs/schedule, historical encoding paid; no newencoder or noisedraw.',
        extra={'purpose':'TinyCPUfixture;sourceNorms once;80nestedmeancompletions/4owndenoisers/4originalcommits/1originalVAE,thenfreshsavedtarget/selectoranalysis.',
               'modelLoadScope':'One originaltransformer/VAE,80mean-Up completions,156anchorrows,600affines,100imageSDPA,480FP32momentproducts. Fullbanks streamednearcompute,8.729GBadditional retention. Original learnedmaps/decoderpaid.'})
    r={'schema':'tce-mean-up-trajectory-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'steps':4,'needs':{'cuda':True},'timeoutSeconds':840,
        'expectedCalls':{'denoiser':4,'joint':20,'single':76,'terminal':4,'scheduler':4,'VAE':1},'prospectiveFirstImageMACs':18559412469760,'prospectiveAvoidedMatrixMACs':3044322312192,
        'protocol':'Frozen74meanlaw all20singles/4ownclocks: current1536Z twoFP32means→BF16→originalU; terminal1024imageZ oneanchor/fullKV; currentquarter overwrite/allcurrentgates/fulloriginalconcatoutput. No fullcurrentUprojection. Original5joints/70momenttext/clocks/head/VAE unchanged.156anchorrows600affines90Acts3139ABIs. Fullactivation/product/completedUp/currentselectedUp/Z/update/means/anchors/gate streamednearcompute. SavedE/noise and originalmaps paid. Seal+exit beforeexistingnative/70/72comparison. No refresh/newreference/fit/sweep/zero-neuralclaim.'}
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
    cmd+=['python','-u','mean_trajectory_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='Frozen74 mean-Up through20singles/4ownclocks:80supports156anchors600affines100imageSDPA480moments,4commits1VAE.18.559TMAC. Fullbanks streamed/fsyncednearcompute8.729GBadditional;113MBmaxmapset. Prior70/72~8.3GBRSS9.5GBdevice.14336RAM13000VRAM4CPU16GBdisk360sexpected840cap,noretries. SavedE/noise/originalmaps paid;no refresh/ref/fit/sweep. Seal+exit beforeexistingcomparisons.'
    if len(note)>500:raise ValueError('Queue note too long')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-mean-up-trajectory-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','height','width','steps','scheduling_contract']},
        ram_mb=14336,vram_mb=13000,cpu_threads=4,disk_gb=16,est_wall_s=360,timeout_s=840,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r)
    for name in ['mean_trajectory_worker.py','mean_trajectory_analysis.py']:
        subprocess.run([sys.executable,'-u',str(root/name),str(root)],cwd=root,check=True)


def collect(root):
    from mrun.client.api import Api
    from run import inventory_code
    root=Path(root).resolve();job=Api().json('GET','/api/jobs/'+read(root/'job.json')['jobID']);result=job.get('result') or {};custody=job.get('payload_custody') or {}
    if job['state']!='succeeded' or result.get('returncode')!=0:raise ValueError('Job not successfully terminal: '+job['state'])
    if job.get('config',{}).get('requestSHA256')!=read(root/'request.json')['requestSHA256']:raise ValueError('Request binding differs')
    if not job.get('admission') or custody.get('required') is not True:raise ValueError('Guarded custody absent')
    phases=[custody.get(k) or {} for k in ['declared','sealed','executed']]
    if not phases[0].get('sha256') or any(any(p.get(k)!=phases[0].get(k) for k in ['sha256','size_bytes']) for p in phases) or phases[2].get('host')!='beast':raise ValueError('Payload custody differs')
    cwd=result['diagnostics']['cwd']
    if cwd!='/mnt/hdd8tb/mrun-work/'+job['job_id']:raise ValueError('Unexpected worker path')
    remote=cwd+'/response-result'
    # The aggregate budget already bounds every member. A separate32MiB cap
    # refused valid larger observation streams within the128MiB total budget.
    code=inventory_code(remote).replace('size>8<<20 or total>32<<20','total>256<<20').replace('len(files)>50','len(files)>200')
    def inventory():return json.loads(subprocess.run(['ssh','-T','beast','python3','-'],input=code,text=True,capture_output=True,check=True,timeout=30).stdout)
    before=inventory();local=root/'result';local.mkdir(exist_ok=True)
    subprocess.run(['rsync','-a','--protect-args','beast:'+remote+'/',str(local)+'/'],check=True,capture_output=True,timeout=90)
    after=inventory();actual={str(p.relative_to(local)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in local.rglob('*') if p.is_file()}
    if before!=after or actual!=before['files']:raise ValueError('Remote/local result inventory differs')
    if read(local/'report.json')['requestSHA256']!=read(root/'request.json')['requestSHA256']:raise ValueError('Collected report request differs')
    save(root/'collection.json',{'jobID':job['job_id'],'custody':custody,'result':result,'inventory':before});save(root/'scheduler-job.json',job)
    return {'result':str(local),'jobID':job['job_id']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
