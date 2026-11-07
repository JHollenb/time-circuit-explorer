"""One guarded lease: isolated producer exits, then saved-field analysis."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read,save,sha,identity
from flux_clock_study import verify
from mean_gate import SOURCE_NAMES,COMPARISON_NAMES,abis
from mean_gate_analysis import analysis_abis

HERE=Path(__file__).resolve().parent
OWNED=['mean_gate.py','mean_gate_worker.py','mean_gate_ledger.py','mean_gate_analysis.py','mean_gate_study.py','test_mean_gate.py',
    'encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','masked_kv_study.py','terminal_capture_guard_v2.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from publish_gated_covariance import audit
    parent,col,job,pr=checked(HERE.parent/'artifacts/own-state-gated-covariance-v1');p=parent/'result';s=read(p/'inputs.json');c=read(p/'candidate.json');e=read(p/'execution.json');v=read(parent/'retained-array-verification.json')
    audit(HERE.parent/'docs/results/own-state-gated-covariance')
    if v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or len(e['typedActs'])!=1: raise ValueError('Authenticated parent release differs')
    source={n:(s['sourceFields'][n] if n in s['sourceFields'] else c['fields'][n]) for n in SOURCE_NAMES}
    comparison={n:c['fields'][n] for n in COMPARISON_NAMES}
    for item in [*source.values(),*comparison.values()]:
        if v['files'][item['path']]['sha256']!=item['sha256'] or v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']: raise ValueError('Parent byte witness absent')
    api=Api();live=api.json('GET','/api/jobs/'+job['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=col['custody']: raise ValueError('Parent scheduler custody differs')
    for n in OWNED:
        if n in pr['ownedSources'] and sha(HERE/n)!=pr['sourceHashes'][n]: raise ValueError('Frozen reused helper differs:'+n)
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for n in OWNED: shutil.copyfile(HERE/n,payload/n)
    for n in WHEELS: shutil.copyfile(parent/'payload'/n,payload/n)
    shutil.copyfile(HERE.parent/'docs/plans/mean-gate-variable-gate.md',payload/'frozen-plan.md')
    config={'sourceFields':source,'comparisonFields':comparison,'parentJobID':job['job_id'],'parentCustody':col['custody'],
        'parentRoot':job['result']['diagnostics']['cwd'],'sourceRoots':[job['result']['diagnostics']['cwd'],s['parentRoot']],
        'parentSealSHA256':sha(p/'candidate-seal.json'),'parentOutputFrameID':e['typedActs'][0]['outputFrame']['frame_id'],
        'parentRecords':{n:sha(p/n) for n in ['candidate.json','execution.json','candidate-seal.json','report.json','inputs.json']},
        'numericalProfile':s['numericalProfile'],'torchVersion':s['versions']['torch'],
        'originalDLineage':{'promotedDescriptor':c['fields']['weightD'],'sourceView':e['sourceViews']['D'],'originalParameterWitness':v['originalParameterWitnesses'][e['sourceViews']['D']['sourceParameter']]},
        'deniedNumericalInputs':sorted((set(s['deniedNumericalInputs'])|{x['path'] for x in c['fields'].values()}|{x['path'] for x in s['sourceFields'].values()})-{x['path'] for x in source.values()})}
    save(payload/'inputs.json',config);host=next(x for x in api.json('GET','/api/hosts') if x['name']=='beast')
    producer=sum(math.prod(x['shape'])*8 for x in abis().values());analysis=sum(math.prod(x['shape'])*8 for x in analysis_abis().values())
    save(payload/'resource-basis.json',{'intentFirstFullModelPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredParentJobID':job['job_id'],'measuredParent':{k:job['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},
        'liveHost':{k:host[k] for k in ['name','active_jobs','admission','telemetry','ram_total_mb','vram_total_mb']},
        'sourceFileBytes':sum(x['bytes'] for x in source.values()),'producerOutputLogicalBytes':producer,'analysisOutputLogicalBytes':analysis,
        'producerBytesByField':{n:math.prod(x['shape'])*8 for n,x in abis().items()},'analysisBytesByField':{n:math.prod(x['shape'])*8 for n,x in analysis_abis().items()},
        'producerExtraFullMLPTemporaryBudgetBytes':4*1536*9216*8,'analysisExtraLogicalBudgetBytes':2<<30,
        'scope':'Saved6fields include newly loaded226492416B promoted original Down map; zero original checkpoint tensors/full model loaded. Producer input bytes plus six outputs and conservative four full-MLP temporary budget; CUDA context/workspace additional. Analysis releases CUDA process and loads source without D, produced fields, four comparisons, signed arrays/centering. 2GiB extra analysis budget is conservative accounting, not measured peak. Parent51.3s1836.8MBRSS2926device included3projections/1.803GBoutputs. Smaller one-Down run:4096RAM3072VRAM4CPU2GBdisk90expected240cap,no retries. Fullmodel planner is inapplicable to saved explicit contraction.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=2,logical_checkpoint_replays=0,
        conditioner_policy='One paid ownclock3single0, savedA/Ustar/meanU/S/D/g; historical generation and original learned work remain paid.',
        extra={'purpose':'One mean-gate Down thenpost-exit saved-field decomposition. No new Up/head/image/fit/sweep.','modelLoadScope':'Loads226492416B authenticated promoted original Down from76; zero checkpoint tensors.6producerABIs plus15analysisABIs.'})
    r={'schema':'tce-mean-gate-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in payload.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'torch','dtype':'float64','batch_size':1,'seq_len':1536,'needs':{'cuda':True},
        'timeoutSeconds':240,'prospectiveMatrixMACs':43486543872,'fixedAddress':{'clock':3,'singleBlock':0},
        'protocol':'Producer derivesmeanA,L=omitted*meanA*(Ustar-meanU),oneFP64Down/gating/means; seal+exit before YR/YQ/YE/Rproduct. Bdelta=YR-YL subtraction-defined; full and within-stream-centered residuals/Gram/signed rows. No oracle mean supplied to producer, new Up/block/head/image/fit/sweep/zero-neural claim.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(payload/'request.json',r);save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in payload.iterdir()]+['preflight.v1.json']})
    return {'root':str(root),'requestSHA256':r['requestSHA256'],'parentJobID':job['job_id'],'producerOutputLogicalBytes':producer,'analysisOutputLogicalBytes':analysis}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists(): raise ValueError('Already submitted; inspect same handle')
    cmd=command()[:-5]
    for n in WHEELS: cmd+=['--with','./'+n]
    cmd+=['python','-u','mean_gate_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='One ownclock3single0 mean-gate Down43.487BMAC;6savedfields include226MB originalD,zero checkpoint tensors/fullmodel.6producer+15analysisABIs,~605MBnewarrays nearcompute.Producerseal/exitbeforeoldYR/Q/E. Full/streamcentered residuals/signedGram/rows;changingterm subtraction-derived.4096RAM3072VRAM4CPU2GBdisk90expected240cap,noretries. No newUp/head/image/fit/sweep/strictzero-neuralclaim.'
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-mean-gate-v1',model=r['model'],task='analysis',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','scheduling_contract']},
        ram_mb=4096,vram_mb=3072,cpu_threads=4,disk_gb=2,est_wall_s=90,timeout_s=240,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();verify(root,read(root/'request.json'));(root/'response-result').mkdir()
    started=time.monotonic()
    with subprocess.Popen([sys.executable,'-u',str(root/'mean_gate_worker.py'),str(root)],cwd=root) as producer:
        code=producer.wait()
        if code: raise subprocess.CalledProcessError(code,producer.args)
        save(root/'response-result/producer-exit.json',{'schema':'tce-mean-gate-producer-exit-v1','producerPID':producer.pid,'returncode':code,
            'requestSHA256':read(root/'request.json')['requestSHA256'],'candidateSealSHA256':sha(root/'response-result/candidate-seal.json'),
            'producerProcessSeconds':time.monotonic()-started,'scope':'Parent observed successful producer process exit before launching comparison analysis.'})
    subprocess.run([sys.executable,'-u',str(root/'mean_gate_analysis.py'),str(root)],check=True,cwd=root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else: value=globals()[a.action](a.root)
    print(json.dumps(value))
