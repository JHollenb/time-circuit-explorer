"""One fixed saved own-state diagnostic, admitted through guarded mrun."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from encoder_atoms import read,save,sha,identity
from flux_clock_study import verify
from gated_covariance_worker import PARAMETERS
from gated_covariance import SOURCE_NAMES,abis

HERE=Path(__file__).resolve().parent
OWNED=['gated_covariance.py','gated_covariance_worker.py','gated_covariance_ledger.py','gated_covariance_analysis.py','gated_covariance_study.py','test_gated_covariance.py',
    'encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','masked_kv_study.py','terminal_capture_guard_v2.py','projected_complement.py','sparse_text_kv.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    import math
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from publish_mean_trajectory import audit
    parent,col,job,pr=checked(HERE.parent/'artifacts/mean-up-trajectory-v1');p=parent/'result';s=read(p/'inputs.json');c=read(p/'candidate.json');e=read(p/'execution.json');v=read(parent/'retained-array-verification.json')
    audit(HERE.parent/'docs/results/mean-up-trajectory')
    record=next(x for x in e['completionRecords'] if x['step']==3 and x['index']==0);act=next(x for x in e['completionActs'] if x['stage']=='completion3_0')
    if act['outputFrame']['frame_id']!=record['typedOutputFrameID'] or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Own parent transaction/seal differs')
    source={name:record['fields'][name] for name in SOURCE_NAMES}
    for item in source.values():
        if v['files'][item['path']]['sha256']!=item['sha256'] or v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Own source byte witness absent')
    api=Api();live=api.json('GET','/api/jobs/'+job['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=col['custody']:raise ValueError('Own parent custody differs')
    for name in OWNED:
        if name in pr['ownedSources'] and sha(HERE/name)!=pr['sourceHashes'][name]:raise ValueError('Frozen reused parent helper changed:'+name)
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for name in OWNED:shutil.copyfile(HERE/name,payload/name)
    for name in WHEELS:shutil.copyfile(parent/'payload'/name,payload/name)
    shutil.copyfile(HERE.parent/'docs/plans/own-state-gated-covariance.md',payload/'frozen-plan.md')
    config={n:s[n] for n in ['modelRoot','nativeSources','versions','numericalProfile']};originals={x['name']:x for x in e['loadedTransformerParameters']}
    config.update(sourceFields=source,originalParameters=[originals[n] for n in PARAMETERS],imageAssets={n:s['imageAssets'][n] for n in ['transformer/config.json','transformer/diffusion_pytorch_model.safetensors']},
        parentJobID=job['job_id'],parentCustody=col['custody'],parentRoot=job['result']['diagnostics']['cwd'],parentSealSHA256=sha(p/'candidate-seal.json'),completionOutputFrameID=record['typedOutputFrameID'],
        parentRecords={n:sha(p/n) for n in ['request.json','candidate.json','execution.json','candidate-seal.json','report.json']},parentCompletionRecord=record,parentCompletionAct=act,
        deniedNumericalInputs=sorted(({x['path'] for x in c['fields'].values()}|{c['image']['path']}|set(s['deniedNumericalInputs']))-{x['path'] for x in source.values()}))
    save(payload/'inputs.json',config);host=next(x for x in api.json('GET','/api/hosts') if x['name']=='beast')
    logical=sum(math.prod(x['shape'])*8 for x in abis().values());history=read(HERE.parent/'artifacts/matched-single-write-v1/scheduler-job.json')
    save(payload/'resource-basis.json',{'intentFirstOriginalStoragePreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredLocalHistoryJobID':history['job_id'],'measuredLocalHistory':{k:history['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'liveHost':{k:host[k] for k in ['name','active_jobs','admission','telemetry','ram_total_mb','vram_total_mb']},
        'originalParameterBytes':sum(x['bytes'] for x in config['originalParameters']),'sourcePayloadBytes':sum(math.prod(x['shape'])*(4 if x['logicalDtype']=='torch.float32' else 8 if x['logicalDtype']=='torch.int64' else 2) for x in source.values()),
        'outputPayloadBytes':logical,'outputBytesByField':{n:math.prod(x['shape'])*8 for n,x in abis().items()},'additionalLiveFP64ActivationAndTemporaryBytesUpperBound':452984832,
        'scope':'No fullmodel object.2originalBF16params245MB+~100MBsource+1.803GBretainedFP64outputs, <=453MBextraactivation/reduction/temp bound; CUDAcontext/workspace additional. Measured73local2050.5MBRSS1508device. Declared4096RAM4096VRAM4CPU3GBdisk120expected300cap,noretries; fullmodelfallbackplanning inapplicable to3explicitprojections. Exact output sizes are logical,not observedpeak.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=2,logical_checkpoint_replays=0,conditioner_policy='One authenticated saved ownclock3single0; originalgeneration/E/noise/maps/capture paid.',extra={'purpose':'FP64fullUp/2Down gatedcovariance diagnostic thenpostsealownscalars. No newwholemodelreference/head/image/fit/sweep.','modelLoadScope':'2originalparams plus sourceviews,28outputABIs,1803313368Boutputpayload;3FP64projections130459631616MAC.'})
    r={'schema':'tce-gated-covariance-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in payload.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'torch','dtype':'float64','batch_size':1,'seq_len':1536,'needs':{'cuda':True},'timeoutSeconds':300,'prospectiveMatrixMACs':130459631616,
        'fixedAddress':{'clock':3,'singleBlock':0},'protocol':'One fixedownstate:actualZ/A/currentS/completedUp/product/gate, sourceU/D. PaidFP64Up plus2Down; YQ=YE-YR combinedpolicy/reassociation. Constant/centered/Gram/rowoutcomes,productcovarianceidentity,oraclesuppliedconstantrepresentationflooronly. No nativefiniteconcatblockerror,newcontinuation/head/image/fit/sweep/zero-neuralclaim.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(payload/'request.json',r);save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in payload.iterdir()]+['preflight.v1.json']})
    return {'root':str(root),'requestSHA256':r['requestSHA256'],'parentJobID':job['job_id']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same handle')
    cmd=command()[:-5]
    for name in WHEELS:cmd+=['--with','./'+name]
    cmd+=['python','-u','gated_covariance_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='One savedownclock3single0:FP64fullUp+2Down130.460BMAC; currentA/S/gate supplied,2originalparams245MB.28ABIs1.803GBoutputs retainednearcompute;~100MBinputs<=453MBtemps, no fullmodel.4096RAM4096VRAM4CPU3GBdisk120expected300cap,noretries. Constant/centered/combinedpolicy-reassociation andoraclefloor;no correction/head/image/fit/sweep/nativeconcaterrorclaim.'
    if len(note)>500:raise ValueError('Queue note too long')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-own-state-gated-covariance-v1',model=r['model'],task='analysis',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','scheduling_contract']},ram_mb=4096,vram_mb=4096,cpu_threads=4,disk_gb=3,est_wall_s=120,timeout_s=300,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();verify(root,read(root/'request.json'));(root/'response-result').mkdir()
    for name in ['gated_covariance_worker.py','gated_covariance_analysis.py']:subprocess.run([sys.executable,'-u',str(root/name),str(root)],check=True,cwd=root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
