"""Stage, execute and collect one four-call native carrier/receiver crossing."""
import argparse
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha,identity
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['carrier_crossing.py','carrier_crossing_worker.py','carrier_crossing_study.py','native_reader_corpus.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','encoder_atoms.py','paired_fields.py','masked_kv_study.py','source_response.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    parent=HERE.parent/'artifacts/source-native-reader-v1';c,j,old=checked_collection(parent,current_workers=False);api=Api();live=api.json('GET','/api/jobs/'+j['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody']:raise ValueError('Native donor parent differs')
    p0=parent/'result';im=read(p0/'images.json');source=read(p0/'inputs.json');corp=read(p0/'corpus.json');report=read(p0/'report.json')
    if report['candidateSealSHA256']!=sha(p0/'candidate-seal.json') or read(parent/'retained-array-verification.json')['candidateSealSHA256']!=sha(p0/'candidate-seal.json'):raise ValueError('Parent release/full bytes differ')
    reads={x['id']:x for x in im['reads']};records={x['id']:x for x in corp['records']};receivers={key:{'id':name,'z':records[name]['z']} for key,name in [('R0','s60672.base.canonical'),('R1','s60673.base.canonical')]}
    donors={key:reads[name]['H0'] for key,name in [('A','s60672.base.canonical'),('B','s60672.base.far_bottom_right')]}
    if any(x['shape']!=[1,1024,3072] or x['logicalDtype']!='torch.bfloat16' for x in donors.values()):raise ValueError('Complete carrier ABI differs')
    root=Path(root).resolve();root.mkdir(exist_ok=False,parents=True);p=root/'payload';p.mkdir()
    for n in OWNED:
        if n in old['ownedSources'] and sha(HERE/n)!=old['sourceHashes'][n]:raise ValueError('Qualified helper changed: '+n)
        shutil.copyfile(HERE/n,p/n)
    for n in WHEELS:shutil.copyfile(parent/'payload'/n,p/n)
    shutil.copyfile(HERE.parent/'docs/plans/carrier-receiver-crossing.md',p/'frozen-plan.md')
    common={k:corp['conditioningPages']['base.'+k] for k in ['conditioning','img_ids','txt_ids']};files={name:corp['sourceFiles'][name] for x in common.values() for name in x['files']};base=j['result']['diagnostics']['cwd']
    assets={k:v for k,v in source['imageAssets'].items() if k.startswith('transformer/')}
    s={'modelRoot':source['modelRoot'],'versions':source['versions'],'nativeSources':source['nativeSources'],'imageAssets':assets,'numericalProfile':source['numericalProfile'],'attentionProcessors':source['attentionProcessors'],
        'receivers':receivers,'donors':donors,'common':common,'commonFiles':files,'clock':source['clockTable']['rows'][3]['clockOperands']['transformer_argument'],
        'protectedRoots':[base,*corp['sourceRoots'],*source['targetRoots'],source['modelRoot']],
        'deniedCanaries':[records['s60673.base.canonical']['target']['path'],str(Path(source['modelRoot'])/next(k for k in source['imageAssets'] if k.startswith('vae/') and k.endswith('.safetensors')))],
        'parentRecords':{n:{'path':base+'/response-result/'+n,'sha256':sha(p0/n)} for n in ['images.json','inputs.json','corpus.json','candidate-seal.json','report.json']}}
    save(p/'inputs.json',s)
    full=HERE.parent/'artifacts/two-clock-v1';fc,fj,fr=checked_collection(full,current_workers=False)
    basis={'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredCompatibleJobID':fj['job_id'],'measuredCompatibleResult':fc['result'],'measuredCompatibleReservation':fj['reservation'],
        'exception':'Generic alias preview has no exact artifact/config and assumes max_batch64 with~40GBVRAM, inapplicable to scalar B1/1536 native calls. Reuse measured same-geometry full-native parent14336RAM/13000VRAM CPU4. Current169-tensor original clock adds341.31MB relative to compiled-clock parent, while VAE is omitted; retain established headroom.4nativecalls/no encoder/VAE, allinputs Beast-local. Actual peaks recorded for future sizing.'}
    save(p/'resource-basis.json',basis)
    contract=one_lease_scheduling_contract(logical_branches=4,logical_operations=4,logical_checkpoint_replays=0,conditioner_policy='Reuse exact source-bound base conditioning and clock; no new encoder.',
        extra={'modelLoadScope':'One resident full original169-tensor transformer with4scalar B1 native calls and one declared typed image-projection-input clamp. No compiled clock replacement.'})
    r={'schema':'tce-carrier-crossing-request-v1','parentJobID':j['job_id'],'parentRequestSHA256':old['requestSHA256'],'ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16','needs':{'cuda':True},'batch_size':1,'seq_len':1536,'height':512,'width':512,'timeoutSeconds':600,
        'protocol':'One 2x2 complete H0A/H0B clamp at firstjoint image to_out[0], same donors across two canonical clock3 receivers.4ordinary native calls with all169tensors/originalclock;recipientH0 discardedbutpaid. Seal4velocities before factorial scoring. No newdonors/encoder/VAE/images/scheduler/noise/fit or naturalparity claim.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r);save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256'],'donorParentJobID':j['job_id']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same job')
    cmd=command()[:-5]
    for n in WHEELS:cmd+=['--with','./'+n]
    cmd+=['python','-u','carrier_crossing_study.py','execute','.'];i=cmd.index('uv');cmd[i:i]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-carrier-crossing-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','height','width','scheduling_contract']},
        ram_mb=14336,vram_mb=13000,cpu_threads=4,disk_gb=1,est_wall_s=90,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,
        note='One resident169-tensor native transformer;4scalar B1/1536 calls,2receivers x identical2donorH0. Typed firstjointimage to_out[0] wholeinput clamp,4velocitiesseal ->factorial. RecipientH0/text/clock paid:436affines/100attention,18.896TaffineMAC. No encoder/donorformation/VAE/images/noise/fit. Measured14336RAM13000VRAM CPU4 exception;genericB64previewinapplicable. Beastlocal;90sexpected600scap,noretries.')
    v={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',v);return v


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='execute':
        from carrier_crossing_worker import execute
        v=execute(a.run)
    elif a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(json.dumps(v))
