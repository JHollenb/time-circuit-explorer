"""Stage one two-call first-text-write/complete-image-carrier experiment."""
import argparse
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha,identity
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['text_crossing.py','text_crossing_worker.py','text_crossing_study.py','carrier_crossing.py','native_reader_corpus.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','encoder_atoms.py','paired_fields.py','masked_kv_study.py','source_response.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from publish_source_native import checked
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    parent,c,j,old=checked(HERE.parent/'artifacts/carrier-crossing-v1');api=Api();live=api.json('GET','/api/jobs/'+j['job_id']);p0=parent/'result';report=read(p0/'report.json');e=read(p0/'execution.json');seal=read(p0/'velocity-seal.json');s=read(p0/'inputs.json')
    if live['state']!='succeeded' or live['payload_custody']!=c['custody'] or report['velocitySealSHA256']!=sha(p0/'velocity-seal.json') or read(parent/'retained-array-verification.json')['velocitySealSHA256']!=sha(p0/'velocity-seal.json'):raise ValueError('Parent custody/release differs')
    if e['textWriteFields']['R0']['shape']!=[1,512,3072] or e['textWriteFields']['R0']['logicalDtype']!='torch.bfloat16':raise ValueError('Complete text-write ABI differs')
    root=Path(root).resolve();root.mkdir(exist_ok=False,parents=True);p=root/'payload';p.mkdir()
    for n in OWNED:
        if n in old['ownedSources'] and sha(HERE/n)!=old['sourceHashes'][n]:raise ValueError('Qualified helper changed: '+n)
        shutil.copyfile(HERE/n,p/n)
    for n in WHEELS:shutil.copyfile(parent/'payload'/n,p/n)
    shutil.copyfile(HERE.parent/'docs/plans/carrier-text-write-crossing.md',p/'frozen-plan.md');base=j['result']['diagnostics']['cwd']
    s.update(receivers={'R1':s['receivers']['R1']},textDonor=e['textWriteFields']['R0'],parameterMapIdentity=e['parameterMapIdentity'],expectedNativeFields={k:e[k]['R1'] for k in ['nativeH0Fields','textHeadFields','textWriteFields']},expectedReceiverPrefix=e['receiverPrefixHashes']['R1'],
        protectedRoots=[base,*s['protectedRoots']],deniedCanaries=[*s['deniedCanaries'],seal['fields']['R1A']['path']],parentRecords={n:{'path':base+'/response-result/'+n,'sha256':sha(p0/n)} for n in ['inputs.json','report.json','execution.json','velocity-seal.json']})
    save(p/'inputs.json',s);save(p/'targets.json',{'parentJobID':j['job_id'],'velocitySealSHA256':sha(p0/'velocity-seal.json'),'fields':seal['fields']})
    save(p/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredCompatibleJobID':j['job_id'],'measuredCompatibleResult':c['result'],'measuredCompatibleReservation':j['reservation'],
        'exception':'Generic alias lacks exact artifact/config and previews max_batch64, not this scalarB1 native hook path. Same169tensors/B1/1536 originalclock geometry measured7546.3MBRSS/8120MBVRAM in46. Two calls instead of4, extra3.15MB textdonor plus small replacement backing. Reduce reservation to10240MBRAM/10240MBVRAM CPU4 with measured headroom;600scap60sestimate. Beast-local immutable inputs require pin.'})
    contract=one_lease_scheduling_contract(logical_branches=2,logical_operations=2,logical_checkpoint_replays=0,conditioner_policy='Reuse exact source-bound conditioning and original clock; no encoder.',
        extra={'modelLoadScope':'One resident original169-tensor transformer,2scalarB1calls. Each firsttextoutput and imageprojectioninput replaced by exact declared payloads; both native reads/projections paid.'})
    r={'schema':'tce-text-crossing-request-v1','parentJobID':j['job_id'],'parentRequestSHA256':old['requestSHA256'],'ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16','needs':{'cuda':True},'batch_size':1,'seq_len':1536,'height':512,'width':512,'timeoutSeconds':600,
        'protocol':'Two R1 calls with complete H_A/H_B image clamps and archived R0 firsttextWrite output clamp. Seal2velocities before opening4oldvelocities. Signed orderedI=(d10-d0)+(d1-d10),all/944queryrecords. All nativeworkpaid,no fit/newdonor/encoder/VAE/image/noise/scheduler.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r);save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256'],'parentJobID':j['job_id']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same job')
    cmd=command()[:-5]
    for n in WHEELS:cmd+=['--with','./'+n]
    cmd+=['python','-u','text_crossing_study.py','execute','.'];i=cmd.index('uv');cmd[i:i]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-text-crossing-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','height','width','scheduling_contract']},
        ram_mb=10240,vram_mb=10240,cpu_threads=4,disk_gb=1,est_wall_s=60,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,
        note='One169-tensor native resident model;2scalar B1/1536 R1calls. Same H_A/H_B imageinput clamps,firsttextoutput fixedR0T0. Seal2velocities beforeoldtargetread ->signedorderedinteraction.218affines/50attention,9.448TaffineMAC;bothdiscardedreads/clockpaid. No encoder/newdonor/fit/VAE/image/noise/scheduler. Measured46basis10240RAM/VRAM CPU4;Beastlocal,60sexpect600scap,noretry.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='execute':
        from text_crossing_worker import execute
        value=execute(a.run)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
