"""One guarded quadratic producer and post-exit saved-field analysis."""
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
from odd_gate import SOURCE_NAMES,COMPARISON_NAMES,PARAMETER,abis
from odd_gate_analysis import analysis_abis

HERE=Path(__file__).resolve().parent
OWNED=['odd_gate.py','odd_gate_worker.py','odd_gate_ledger.py','odd_gate_analysis.py','odd_gate_study.py','test_odd_gate.py',
    'encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','masked_kv_study.py','terminal_capture_guard_v2.py','projected_complement.py','sparse_text_kv.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from publish_mean_gate import audit
    parent,col,job,pr=checked(HERE.parent/'artifacts/mean-gate-v1');p=parent/'result';s=read(p/'inputs.json');c=read(p/'candidate.json');e=read(p/'execution.json');a=read(p/'analysis-fields.json');v=read(parent/'retained-array-verification.json')
    cov,ccol,cjob,cpr=checked(HERE.parent/'artifacts/own-state-gated-covariance-v1');cp=cov/'result';cs=read(cp/'inputs.json');ce=read(cp/'execution.json');cv=read(cov/'retained-array-verification.json')
    audit(HERE.parent/'docs/results/mean-gate')
    source={n:s['sourceFields'][n] for n in SOURCE_NAMES if n not in ['normalized','linearOutput']};source.update(normalized=cs['sourceFields']['normalized'],linearOutput=c['fields']['linearOutput'])
    comparison={n:s['comparisonFields'][n] for n in ['centeredOutput','policyOutput','totalOutput']};comparison.update(activation=s['sourceFields']['activation'],activationMean=c['fields']['activationMean'],changingProduct=a['fields']['changingProduct'])
    for item in [*source.values(),*comparison.values()]:
        witness=v if item['path'] in v['files'] else cv
        if witness['files'][item['path']]['sha256']!=item['sha256'] or witness['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Parent byte witness absent')
    api=Api()
    for j,co in [(job,col),(cjob,ccol)]:
        live=api.json('GET','/api/jobs/'+j['job_id'])
        if live['state']!='succeeded' or live['payload_custody']!=co['custody']:raise ValueError('Parent scheduler custody differs')
    for n in OWNED:
        for old in [pr,cpr]:
            if n in old['ownedSources'] and sha(HERE/n)!=old['sourceHashes'][n]:raise ValueError('Frozen reused helper differs:'+n)
    reviews={n:sha(HERE.parent/n) for n in ['docs/results/research-review-cycle78/advisor-survey.json','docs/results/storage-audit-cycle78/publication.json']}
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for n in OWNED:shutil.copyfile(HERE/n,payload/n)
    for n in WHEELS:shutil.copyfile(parent/'payload'/n,payload/n)
    shutil.copyfile(HERE.parent/'docs/plans/odd-gate-quadratic.md',payload/'frozen-plan.md')
    config={n:cs[n] for n in ['modelRoot','nativeSources','versions','numericalProfile','imageAssets']}
    config.update(sourceFields=source,comparisonFields=comparison,parentJobID=job['job_id'],parentCustody=col['custody'],parentRoot=job['result']['diagnostics']['cwd'],
        parentSealSHA256=sha(p/'candidate-seal.json'),parentOutputFrameID=e['typedActs'][0]['outputFrame']['frame_id'],
        parentRecords={n:sha(p/n) for n in ['candidate.json','execution.json','candidate-seal.json','report.json','inputs.json','analysis-fields.json','producer-exit.json']},
        covarianceJobID=cjob['job_id'],covarianceCustody=ccol['custody'],covarianceRoot=cjob['result']['diagnostics']['cwd'],covarianceSealSHA256=sha(cp/'candidate-seal.json'),
        covarianceRecords={n:sha(cp/n) for n in ['inputs.json','candidate.json','execution.json','candidate-seal.json']},originalParameters=[next(x for x in ce['loadedParameters'] if x['name']==PARAMETER)],
        originalDLineage=s['originalDLineage'],sourceRoots=sorted(set(s['sourceRoots']+[job['result']['diagnostics']['cwd']])),preStageReviewSnapshots=reviews,
        deniedNumericalInputs=sorted((set(s['deniedNumericalInputs'])|{x['path'] for x in comparison.values()}|{x['path'] for x in c['fields'].values()}|{x['path'] for x in a['fields'].values()})-{x['path'] for x in source.values()}))
    save(payload/'inputs.json',config);host=next(x for x in api.json('GET','/api/hosts') if x['name']=='beast')
    producer=sum(math.prod(x['shape'])*8 for x in abis().values());analysis=sum(math.prod(x['shape'])*8 for x in analysis_abis().values())
    save(payload/'resource-basis.json',{'intentFirstFullModelPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredParentJobID':job['job_id'],'measuredParent':{k:job['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},
        'measuredCovarianceJobID':cjob['job_id'],'measuredCovariance':{k:cjob['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},
        'liveHost':{k:host[k] for k in ['name','active_jobs','admission','telemetry','ram_total_mb','vram_total_mb']},
        'sourceFileBytes':sum(x['bytes'] for x in source.values()),'originalCheckpointTensorBytes':sum(x['bytes'] for x in config['originalParameters']),
        'producerOutputLogicalBytes':producer,'analysisOutputLogicalBytes':analysis,'producerBytesByField':{n:math.prod(x['shape'])*8 for n,x in abis().items()},'analysisBytesByField':{n:math.prod(x['shape'])*8 for n,x in analysis_abis().items()},
        'producerExtraFullMLPTemporaryBudgetBytes':4*1536*9216*8,'analysisExtraLogicalBudgetBytes':3<<30,
        'scope':'One169.9MBoriginalfusedBF16tensor plusphysical56.6MBGview; promotedGmap/Z retained, originalD226.5MB andUp/L/history supplied. Produceroutputs+inputs+fourfullMLPtemporaries budget; CUDAcontext/workspace extra. Post-exit analysis skipsD/Z/weightG, loads actualA and product/output comparisons with3GiBextra logical budget, not a measuredpeak. Parent77actual27.94s2043.6MBRSS1104device,76actual51.3s1836.8RSS2926device.6144RAM4096VRAM4CPU4GBdisk120expected360cap,no retries. Fullmodel planner inapplicable to two explicit projections.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=2,logical_checkpoint_replays=0,conditioner_policy='Paid ownclock3single0 Z/Ustar/L/S/g/originalmaps; no raw-prompt serving.',
        extra={'purpose':'OneFP64gate+Down86.973BMAC quadratic producer,seals/exitsbeforeactualA/errors;post-releaseH/Fproductspaceonly.','modelLoadScope':'Oneoriginalfusedtensor/physicalGslice plus savedoriginalD; no fullmodel/Up/head/image/fit/sweep.'})
    r={'schema':'tce-odd-gate-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in payload.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'torch','dtype':'float64','batch_size':1,'seq_len':1536,'needs':{'cuda':True},'timeoutSeconds':360,'prospectiveMatrixMACs':86973087744,'fixedAddress':{'clock':3,'singleBlock':0},
        'protocol':'FreshFP64G*,T=.5omittedδG*δU,oneDown,inheritedYL;seal g(YL+YT)andexitbeforeactualA/oldYR/Q/E. EvaluatedH*/A*/Fproductspaceafterreleaseonly;outputremainder subtraction-defined. Full/centered/Gram/signedrows;notnativeG/finiteblock/image/economicalcompiler/zero-neuralclaim.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(payload/'request.json',r);save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in payload.iterdir()]+['preflight.v1.json']})
    return {'root':str(root),'requestSHA256':r['requestSHA256'],'producerOutputLogicalBytes':producer,'analysisOutputLogicalBytes':analysis}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same handle')
    cmd=command()[:-5]
    for n in WHEELS:cmd+=['--with','./'+n]
    cmd+=['python','-u','odd_gate_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='Oneownclock3single0 FP64gate+Down86.973BMAC,oneoriginalfused169.9MBtensor/physicalGslice,saved226.5MBoriginalD/inheritedUp/Lpaid.11producer+14analysisABIs.Producerseal/exitbeforeactualA/errors;H/Fproductspaceonly,remainder subtraction-derived.6144RAM4096VRAM4CPU4GBdisk120expected360cap,noretries. No Up/head/image/fit/sweep/nativeG/zero-neuralclaim.'
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-odd-gate-v1',model=r['model'],task='analysis',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','scheduling_contract']},ram_mb=6144,vram_mb=4096,cpu_threads=4,disk_gb=4,est_wall_s=120,timeout_s=360,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();verify(root,read(root/'request.json'));(root/'response-result').mkdir();started=time.monotonic()
    with subprocess.Popen([sys.executable,'-u',str(root/'odd_gate_worker.py'),str(root)],cwd=root) as producer:
        code=producer.wait()
        if code:raise subprocess.CalledProcessError(code,producer.args)
        save(root/'response-result/producer-exit.json',{'schema':'tce-odd-gate-producer-exit-v1','producerPID':producer.pid,'returncode':code,'requestSHA256':read(root/'request.json')['requestSHA256'],
            'candidateSealSHA256':sha(root/'response-result/candidate-seal.json'),'producerProcessSeconds':time.monotonic()-started,'scope':'Parent observed successful producer exit before comparison analysis launch.'})
    subprocess.run([sys.executable,'-u',str(root/'odd_gate_analysis.py'),str(root)],check=True,cwd=root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
