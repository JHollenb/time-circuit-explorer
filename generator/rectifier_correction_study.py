"""One finite seven-input producer, process exit, then post-release comparison."""
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
from rectifier_correction import SOURCE_NAMES,COMPARISON_NAMES,abis
from rectifier_correction_analysis import analysis_abis

HERE=Path(__file__).resolve().parent
OWNED=['rectifier_correction.py','rectifier_correction_worker.py','rectifier_correction_ledger.py','rectifier_correction_analysis.py','rectifier_correction_study.py','test_rectifier_correction.py',
    'encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','masked_kv_study.py','terminal_capture_guard_v2.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def source_bindings(p):
    s=read(p/'inputs.json');c=read(p/'candidate.json')
    source={n:s['sourceFields'][n] for n in SOURCE_NAMES if n!='gateReference'};source['gateReference']=c['fields']['gateReference']
    comparison={n:s['comparisonFields'][n] for n in COMPARISON_NAMES if n!='quadraticForecast'};comparison['quadraticForecast']=c['fields']['forecast']
    return source,comparison


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from publish_odd_gate import audit
    parent,col,job,pr=checked(HERE.parent/'artifacts/odd-gate-v1');p=parent/'result';s=read(p/'inputs.json');e=read(p/'execution.json');v=read(parent/'retained-array-verification.json');audit(HERE.parent/'docs/results/odd-gate')
    source,comparison=source_bindings(p)
    for item in [*source.values(),*comparison.values()]:
        if v['files'][item['path']]['sha256']!=item['sha256'] or v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Parent full-byte witness absent')
    api=Api();live=api.json('GET','/api/jobs/'+job['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=col['custody']:raise ValueError('Parent scheduler custody differs')
    for n in OWNED:
        if n in pr['ownedSources'] and sha(HERE/n)!=pr['sourceHashes'][n]:raise ValueError('Frozen reused helper differs:'+n)
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for n in OWNED:shutil.copyfile(HERE/n,payload/n)
    for n in WHEELS:shutil.copyfile(parent/'payload'/n,payload/n)
    shutil.copyfile(HERE.parent/'docs/plans/combined-rectifier-correction.md',payload/'frozen-plan.md')
    config={'sourceFields':source,'comparisonFields':comparison,'parentJobID':job['job_id'],'parentCustody':col['custody'],'parentRoot':job['result']['diagnostics']['cwd'],
        'parentSealSHA256':sha(p/'candidate-seal.json'),'parentOutputFrameID':e['typedActs'][0]['outputFrame']['frame_id'],
        'parentRecords':{n:sha(p/n) for n in ['inputs.json','candidate.json','candidate-seal.json','execution.json','analysis-fields.json','report.json','producer-exit.json']},
        'sourceRoots':sorted(set(s['sourceRoots']+[job['result']['diagnostics']['cwd']])),'numericalProfile':s['numericalProfile'],'torchVersion':s['versions']['torch'],'originalDLineage':s['originalDLineage'],
        'preStageReviewSnapshots':{n:sha(HERE.parent/n) for n in ['docs/results/gated-even-bound-evidence/advisor.json','docs/results/research-review-cycle78/advisor-survey.json','docs/results/storage-audit-cycle78/publication.json']},
        'deniedNumericalInputs':sorted((set(s['deniedNumericalInputs'])|{x['path'] for x in comparison.values()}|{x['path'] for x in read(p/'candidate.json')['fields'].values()}|{x['path'] for x in read(p/'analysis-fields.json')['fields'].values()})-{x['path'] for x in source.values()})}
    save(payload/'inputs.json',config);host=next(x for x in api.json('GET','/api/hosts') if x['name']=='beast')
    producer=sum(math.prod(x['shape'])*8 for x in abis().values());analysis=sum(math.prod(x['shape'])*8 for x in analysis_abis().values())
    save(payload/'resource-basis.json',{'intentFirstFullModelPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredParent':{k:job['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'measuredParentJobID':job['job_id'],'liveHost':{k:host[k] for k in ['name','active_jobs','admission','telemetry','ram_total_mb','vram_total_mb']},
        'sourceFileBytes':sum(x['bytes'] for x in source.values()),'comparisonFileBytes':sum(x['bytes'] for x in comparison.values()),'producerOutputLogicalBytes':producer,'analysisOutputLogicalBytes':analysis,
        'producerBytesByField':{n:math.prod(x['shape'])*8 for n,x in abis().items()},'analysisBytesByField':{n:math.prod(x['shape'])*8 for n,x in analysis_abis().items()},
        'producerExtraFullMLPTemporaryBudgetBytes':4*1536*9216*8,'analysisExtraLogicalBudgetBytes':2<<30,
        'scope':'7savedfields~491MB includeGstar/fullUp/YL/originalD226492416B. No checkpoint/model/G/Up projection.9producerABIs and12analysisABIs,full products/residuals retainednearcompute. Parent78twoFP64projections43.15s2894.6MBRSS1822device;77oneDown27.94s2043.6MBRSS1104device. ExtraGsource/rectifier/product/residual arrays vs77 accounted.6144RAM3072VRAM4CPU3GBdisk90expected300cap,no retries. Fullmodel sizing inapplicable to saved explicit oneDown.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=2,logical_checkpoint_replays=0,conditioner_policy='Paid Gstar/Up/L/means/support/D/g at ownclock3single0; historical admittedAmean insideYL remainspaid.',
        extra={'purpose':'One combinedrectifier/Down then post-exit full/centered/signed analysis. No oldF reuse, H/tailDown, fit, image.','modelLoadScope':'7savedfields,one226MBoriginalD;zero checkpoint tensor/G/Up loads or projections.9producer12analysisABIs.'})
    r={'schema':'tce-rectifier-correction-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in payload.iterdir()},'scheduling_contract':contract,'model':'flux2-klein-4b-distilled','device':'cuda','backend':'torch','dtype':'float64','batch_size':1,'seq_len':1536,'needs':{'cuda':True},'timeoutSeconds':300,'prospectiveMatrixMACs':43486543872,'fixedAddress':{'clock':3,'singleBlock':0},
        'protocol':'r=max(Gstar,0),Pplus=omitted*deltaRectifier*deltaU,oneFP64Down,fullg(YL+Yplus);seal/exitbeforeA/B/YR/Q/E/oldforecast. PostreleaseA-minusr/newexpressionproductbookkeeping,subtractiondefinedoutputremainder. Zero/L/oldL+T/newfull/centered/mean/signed/everyrow;no tailactivation/HDown/fit/image/economicalzero-neuralclaim.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(payload/'request.json',r);save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in payload.iterdir()]+['preflight.v1.json']})
    return {'root':str(root),'requestSHA256':r['requestSHA256'],'producerOutputLogicalBytes':producer,'analysisOutputLogicalBytes':analysis}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same handle')
    cmd=command()[:-5]
    for n in WHEELS:cmd+=['--with','./'+n]
    cmd+=['python','-u','rectifier_correction_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-rectifier-correction-v1',model=r['model'],task='analysis',payload=root/'payload',pin='beast',needs=r['needs'],mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','scheduling_contract']},
        ram_mb=6144,vram_mb=3072,cpu_threads=4,disk_gb=3,est_wall_s=90,timeout_s=300,detach=True,retry_on_kill=False,preflight=True,
        note='Oneownclock3single0 rectifier14.156Melements+Down43.487BMAC,7savedfieldsincludingGstar/fullUp/YL/D226MBpaid,no checkpoint/G/Up/head/image.9producer12analysisABIs. Seal/exitbeforeA/B/errors,oldFnotnewresidual; full/centered/mean/signedallrows.6144RAM3072VRAM4CPU3GBdisk90expected300cap,noretries;onefinitelease.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();verify(root,read(root/'request.json'));(root/'response-result').mkdir();started=time.monotonic()
    with subprocess.Popen([sys.executable,'-u',str(root/'rectifier_correction_worker.py'),str(root)],cwd=root) as producer:
        code=producer.wait()
        if code:raise subprocess.CalledProcessError(code,producer.args)
        save(root/'response-result/producer-exit.json',{'schema':'tce-rectifier-correction-producer-exit-v1','producerPID':producer.pid,'returncode':code,'requestSHA256':read(root/'request.json')['requestSHA256'],'candidateSealSHA256':sha(root/'response-result/candidate-seal.json'),'producerProcessSeconds':time.monotonic()-started,'scope':'Parent observed successful producer exit before starting comparison process.'})
    subprocess.run([sys.executable,'-u',str(root/'rectifier_correction_analysis.py'),str(root)],check=True,cwd=root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
