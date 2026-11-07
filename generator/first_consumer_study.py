"""Stage one four-matrix/four-cell image-local consumer assay."""
import argparse
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha,identity
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['first_consumer.py','first_consumer_analysis.py','first_consumer_worker.py','first_consumer_study.py','native_reader_corpus.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','encoder_atoms.py','paired_fields.py','masked_kv_study.py','source_response.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from first_consumer import SELECTED
    from publish_source_native import checked
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    parent,c,j,old=checked(HERE.parent/'artifacts/text-crossing-v1');source,sc,sj,sr=checked(HERE.parent/'artifacts/carrier-crossing-v1');api=Api()
    for job,custody in [(j,c['custody']),(sj,sc['custody'])]:
        live=api.json('GET','/api/jobs/'+job['job_id'])
        if live['state']!='succeeded' or live['payload_custody']!=custody:raise ValueError('Parent live custody differs')
    p0=source/'result';s0=read(p0/'inputs.json');e=read(p0/'execution.json');seal=read(p0/'velocity-seal.json');old_clock=HERE.parent/'artifacts/source-native-reader-v1/result/inputs.json'
    if sha(old_clock)!=s0['parentRecords']['inputs.json']['sha256'] or read(source/'retained-array-verification.json')['velocitySealSHA256']!=sha(p0/'velocity-seal.json') or read(parent/'retained-array-verification.json')['velocitySealSHA256']!=sha(parent/'result/velocity-seal.json'):raise ValueError('Original release/clock lineage differs')
    modulation=read(old_clock)['clockTable']['rows'][3]['fields']['double_img'];geometries=[{'shape':x['shape'],'stride':x['consumedStride'],'storageOffset':x['consumedStorageOffset']} for x in seal['writes']]
    if any(x!=geometries[0] for x in geometries) or any(x['temb_mod_img']!=modulation['payloadSHA256'] for x in e['receiverPrefixHashes'].values()):raise ValueError('Qualified projection/clock geometry differs')
    selected=[x for x in e['loadedParameters'] if x['name'] in SELECTED]
    if len(selected)!=4 or sum(x['bytes'] for x in selected)!=189530112:raise ValueError('Four-matrix admission differs')
    root=Path(root).resolve();root.mkdir(exist_ok=False,parents=True);p=root/'payload';p.mkdir()
    for n in OWNED:
        if n in old['ownedSources'] and sha(HERE/n)!=old['sourceHashes'][n]:raise ValueError('Qualified helper changed: '+n)
        shutil.copyfile(HERE/n,p/n)
    for n in WHEELS:shutil.copyfile(parent/'payload'/n,p/n)
    shutil.copyfile(HERE.parent/'docs/plans/first-image-consumer.md',p/'frozen-plan.md');base=j['result']['diagnostics']['cwd'];sb=sj['result']['diagnostics']['cwd']
    s={k:s0[k] for k in ['modelRoot','versions','nativeSources','imageAssets','numericalProfile','receivers','donors']};s.update(modulation=modulation,selectedParameters=selected,expectedPrefixes=e['receiverPrefixHashes'],carrierGeometry=geometries[0],
        protectedRoots=[base,sb,*s0['protectedRoots']],deniedCanaries=[*s0['deniedCanaries'],read(parent/'result/velocity-seal.json')['fields']['A']['path']],
        parentRecords={**{'crossing.'+n:{'path':sb+'/response-result/'+n,'sha256':sha(p0/n)} for n in ['inputs.json','execution.json','report.json','velocity-seal.json']},
            **{'text.'+n:{'path':base+'/response-result/'+n,'sha256':sha(parent/'result'/n)} for n in ['inputs.json','report.json','velocity-seal.json']},'sourceClockInputs':s0['parentRecords']['inputs.json']})
    save(p/'inputs.json',s)
    history=read(HERE.parent/'artifacts/source-native-reader-v1/collection.json')
    save(p/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1024],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredComponentJobID':history['jobID'],'measuredComponentResult':history['result'],
        'workingSet':{'selectedParameterBytes':189530112,'maxFourFP64OperandBytes':4*1024*18432*8,'maxNativeFusedBytesPerCell':1024*18432*2,'streamPerStage':True,'expectedRetainedBytesApprox':3400000000},
        'exception':'Generic full-alias/B64 planning cannot represent four selected matrices/12scalar affines, no attention/fullTransformer/VAE. Compatible partial-reader job1892 measured3867.7MBRSS/1802MBVRAM with142MBselectedparams plus attention/VAE. Current190MBparams/noattention/VAE, stream stage FP64 diagnostics; reserve6144MBRAM/4096MBVRAM CPU4/disk8GB. Actualpeaks replace this basis for future matching runs. Beast-local immutable operands require pin.'})
    contract=one_lease_scheduling_contract(logical_branches=4,logical_operations=4,logical_checkpoint_replays=0,conditioner_policy='Supplied complete H0/receiver states and qualified clock3 modulation; no encoder or learned clock.',
        extra={'modelLoadScope':'One four-matrix selected local module.2B1inputprojections+2B1sharedcarrierprojections+4originalMLPpaths.12affines;no full Transformer/attention/text/suffix. Seal all native boundaries before FP64 diagnostic factorization.'})
    r={'schema':'tce-first-consumer-request-v1','parentJobID':j['job_id'],'parentRequestSHA256':old['requestSHA256'],'sourceCrossingJobID':sj['job_id'],'ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16','needs':{'cuda':True},'batch_size':1,'seq_len':1024,'height':512,'width':512,'timeoutSeconds':600,
        'protocol':'Four-cell local first image consumer only. Selected4originalmatrices190MB/12affines368.025BMAC, nativeBF16orderedcaptures;seal then finiteLN/product diagnostics andFP64SiLUreference. No fullTransformer/attention/text/clocknetwork/encoder/fit/noise/VAE/image/scheduler/oldvelocitytargets.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r);save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same job')
    cmd=command()[:-5]
    for n in WHEELS:cmd+=['--with','./'+n]
    cmd+=['python','-u','first_consumer_study.py','execute','.'];i=cmd.index('uv');cmd[i:i]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-first-consumer-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','height','width','scheduling_contract']},ram_mb=6144,vram_mb=4096,cpu_threads=4,disk_gb=8,est_wall_s=90,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,
        note='One4matrix localimage consumer:2B1input+2sharedcarrier projections,4MLPpaths=12affines368.025BMAC. Allnativefieldsseal ->LN/product finite identities+FP64SiLU. No fullTransformer/attention/text/clocknet/encoder/fit/VAE/image/oldvelocityread. Selected190MB;streamstages~3.4GBretained. Partialreader historybasis6144RAM4096VRAM CPU4disk8GB. Beastlocal90sexpect600scap,noretry.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='execute':
        from first_consumer_worker import execute
        value=execute(a.run)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
