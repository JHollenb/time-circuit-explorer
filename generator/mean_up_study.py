"""One sealed current-stream mean-Up candidate, then immutable73 comparisons."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from encoder_atoms import read,save,sha,identity
from flux_clock_study import verify
from matched_single_study import OWNED as PRIOR,WHEELS

HERE=Path(__file__).resolve().parent
OWNED=['mean_up_completion.py','mean_up_worker.py','mean_up_ledger.py','mean_up_analysis.py','mean_up_study.py','test_mean_up_completion.py',*PRIOR]


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    parent,col,job,pr=checked(HERE.parent/'artifacts/matched-single-write-v1');p=parent/'result'
    s=read(p/'inputs.json');c=read(p/'candidate.json');a=read(p/'anatomy.json');v=read(parent/'retained-array-verification.json');seal=read(p/'candidate-seal.json')
    if v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or seal['candidateSHA256']!=sha(p/'candidate.json') or seal['executionSHA256']!=sha(p/'execution.json'):raise ValueError('Parent release/bytes differ')
    for item in [*s['sourceFields'].values(),*c['fields'].values()]:
        if v['files'][item['path']]['sha256']!=item['sha256'] or v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Required parent byte witness absent')
    for n in PRIOR:
        if sha(HERE/n)!=pr['sourceHashes'][n]:raise ValueError('Frozen parent source changed:'+n)
    api=Api()
    for previous in s['parents'].values():
        live=api.json('GET','/api/jobs/'+previous['jobID'])
        if live['state']!='succeeded' or live['payload_custody']!=previous['custody']:raise ValueError('Historical parent custody differs')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for n in OWNED:shutil.copyfile(HERE/n,payload/n)
    for n in WHEELS:shutil.copyfile(parent/'payload'/n,payload/n)
    shutil.copyfile(HERE.parent/'docs/plans/current-stream-mean-up.md',payload/'frozen-plan.md')
    config={n:s[n] for n in ['modelRoot','nativeSources','versions','numericalProfile','imageAssets','sourceFields','originalParameters']}
    config['parents']={**s['parents'],'matched73':{'jobID':job['job_id'],'custody':col['custody'],'root':job['result']['diagnostics']['cwd']}}
    config['parentSeals']={**s['parentSeals'],'matched73':sha(p/'candidate-seal.json')}
    config['parentRecords']={**s['parentRecords'],'matched73':{n:sha(p/n) for n in ['request.json','candidate.json','execution.json','candidate-seal.json','report.json']}}
    denied=set(s['deniedNumericalInputs'])|{x['path'] for x in [*c['fields'].values(),*a['fields'].values(),*read(p/'differences.json').values()]}
    config['deniedNumericalInputs']=sorted(denied-{x['path'] for x in s['sourceFields'].values()})
    save(payload/'inputs.json',config);save(payload/'targets.json',{'fields':c['fields'],'scope':'Immutable73sparse/dense actualfields; numeric access onlyafter currentcandidate sealed/exited.'})
    host=next(x for x in api.json('GET','/api/hosts') if x['name']=='beast')
    save(payload/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredMatchedHistory':{k:job['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},
        'liveHost':{k:host[k] for k in ['name','active_jobs','admission','telemetry','ram_total_mb','vram_total_mb']},
        'originalParameterBytes':245367296,'maxOnePhysicalMapBytes':113246208,'oneFullBF16MLPBankBytes':28311552,'oneFullFP64ComparisonBankBytes':113246208,
        'scope':'One local source actor, original4params, fullgate/completedUp/product capture pluscompact2rowanchors; no fullmodel. Prior73sampled2050.5MBRSS1508device includes heavierFP64diagnostic. Serialpostexit CPUcomparisons load few113MBbanks at a time. Measured-local exception4096RAM2048VRAM4CPU1GBdisk90expected180cap,noretries. Planning wholemodel/B64fallback is inapplicable tooneblock.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=2,logical_checkpoint_replays=0,
        conditioner_policy='Authenticated73common clock0single0input pluspaid72sourceNormcache; historicalprefix/conditioner/noise/norms paid.',
        extra={'purpose':'One currentstreammeanUpcandidate seal/exit thenimmutable73 comparisons. No newdense/head/image/fit/sweep.',
               'modelLoadScope':'4single0params245MB;4affines1imageSDPA5momentproducts167763247104MAC. Streamreductions/gates/rank/gathers/hash/copy/I/Oextra.'})
    r={'schema':'tce-mean-up-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in payload.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'diffusers','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'needs':{'cuda':True},'timeoutSeconds':180,
        'prospectiveAffineMACs':155883405312,'prospectiveMatrixMACs':167763247104,'selectedChannels':2304,
        'protocol':'ActualZ FP32means pertext/image ->BF16 ->original U2rowanchors; ownquarterselectedUp overwrite; fullcurrentgates/product/concatout/residual. No fullcurrentUpbank/targetanchor beforeseal+exit; historical73comparisononlyafter. Source learnedmaps/state/history paid;strictgoalopen.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(payload/'request.json',r);save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in payload.iterdir()]+['preflight.v1.json']})
    return {'root':str(root),'requestSHA256':r['requestSHA256'],'parent':job['job_id']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same handle')
    cmd=command()[:-5]
    for n in WHEELS:cmd+=['--with','./'+n]
    cmd+=['python','-u','mean_up_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='One commoninput mean-Up:FP32text/imageZmeans->BF16 originalU2rows;ownquarterUp overwrite,allgates/fullout.4affines1imageSDPA5momentproducts167.763BMAC;4params245MB. Seal+exit before73comparators;no fullcurrentUpbank/ref/head/image/fit/sweep. Reductions/ranking/maps/copy/hashpaid. Measuredlocal4096RAM2048VRAM4CPU1GBdisk90expected180cap,noretries;strictzero-neuralopen.'
    if len(note)>500:raise ValueError('Queue note too long')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-current-stream-mean-up-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','height','width','scheduling_contract']},
        ram_mb=4096,vram_mb=2048,cpu_threads=4,disk_gb=1,est_wall_s=90,timeout_s=180,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();verify(root,read(root/'request.json'));(root/'response-result').mkdir()
    subprocess.run([sys.executable,'-u',str(root/'mean_up_worker.py'),str(root)],check=True,cwd=root)
    if not (root/'response-result/candidate-seal.json').exists():raise ValueError('No candidate release')
    subprocess.run([sys.executable,'-u',str(root/'mean_up_analysis.py'),str(root)],check=True,cwd=root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
