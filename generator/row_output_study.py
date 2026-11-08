"""One finite guarded output cut with post-exit82/73 target comparison."""
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
from row_output import SOURCE_NAMES,abis,work
from row_output_analysis import analysis_abis

HERE=Path(__file__).resolve().parent
OWNED=['row_output.py','row_output_worker.py','row_output_ledger.py','row_output_analysis.py','row_output_study.py','test_row_output.py',
    'gate_up_basis_ledger.py','gate_up_basis_analysis.py','encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','masked_kv_study.py','terminal_capture_guard_v2.py','projected_complement.py','sparse_text_kv.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def logical_bytes(item):return math.prod(item['shape'])*(2 if item['dtype']=='torch.bfloat16' else 8)


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from projected_complement import fields
    from publish_source_native import checked
    from publish_gate_up_basis import audit
    parent,col,job,pr=checked(HERE.parent/'artifacts/current-state-gate-up-v1');p=parent/'result';s=read(p/'inputs.json');c=read(p/'candidate.json');v=read(parent/'retained-array-verification.json');audit(HERE.parent/'docs/results/current-state-gate-up')
    source={name:(c['fields'][name] if name in ['coordinates','product'] else s['sourceFields'][name]) for name in SOURCE_NAMES}
    comparison={'82':{name:c['fields'][name] for name in ['preResidualUpdate','exit']},'73':{name:s['comparisonFields'][name] for name in ['preResidualUpdate','exit']}}
    for item in [*source.values(),*list(fields(comparison))]:
        if v['files'][item['path']]['sha256']!=item['sha256'] or v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Parent byte witness absent')
    api=Api();live=api.json('GET','/api/jobs/'+job['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=col['custody']:raise ValueError('Parent current custody differs')
    for name in OWNED:
        if name in pr['ownedSources'] and sha(HERE/name)!=pr['sourceHashes'][name]:raise ValueError('Frozen helper differs:'+name)
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for name in OWNED:shutil.copyfile(HERE/name,payload/name)
    for name in WHEELS:shutil.copyfile(parent/'payload'/name,payload/name)
    for name,dest in [('docs/plans/current-row-output-projection.md','frozen-plan.md'),('docs/plans/current-row-output-implementation.md','implementation-notes.md')]:shutil.copyfile(HERE.parent/name,payload/dest)
    denied=set(s['deniedNumericalInputs'])|{item['path'] for item in fields(c)}|{item['path'] for item in fields(read(p/'analysis-fields.json'))}|{item['path'] for item in fields(s['sourceFields'])}
    config={key:s[key] for key in ['modelRoot','nativeSources','versions','imageAssets','numericalProfile']}
    config.update(sourceFields=source,comparisonFields=comparison,originalParameters=[s['originalParameters'][1]],parentJobID=job['job_id'],parentCustody=col['custody'],parentRoot=job['result']['diagnostics']['cwd'],
        parentCandidateSealSHA256=sha(p/'candidate-seal.json'),parentByteVerificationSHA256=sha(parent/'retained-array-verification.json'),
        parentRecords={name:sha(p/name) for name in ['request.json','inputs.json','candidate.json','execution.json','candidate-seal.json','producer-exit.json','report.json']},
        sourceRoots=sorted({job['result']['diagnostics']['cwd'],*s['sourceRoots']}),deniedNumericalInputs=sorted(denied-{item['path'] for item in source.values()}),
        preStageReviewSnapshots={name:sha(HERE.parent/name) for name in ['docs/results/current-state-gate-up-evidence/advisor.json','docs/results/research-review-cycle81/advisor-survey.json','docs/results/storage-audit-cycle81/publication.json']})
    save(payload/'inputs.json',config);host=next(item for item in api.json('GET','/api/hosts') if item['name']=='beast')
    save(payload/'resource-basis.json',{'intentFirstFullModelPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredParentJobID':job['job_id'],'measuredParent':{key:job['result'][key] for key in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'liveHost':{key:host[key] for key in ['name','active_jobs','admission','telemetry','ram_total_mb','vram_total_mb']},
        'sourceFileBytes':sum(item['bytes'] for item in source.values()),'comparisonFileBytes':sum(item['bytes'] for item in fields(comparison)),
        'originalLoadedParameterBytes':75497472,'promotedOutFP64Bytes':3072*12288*8,'joinedBF16Bytes':1536*12288*2,'joinedFP64Bytes':1536*12288*8,
        'producerOutputLogicalBytes':sum(map(logical_bytes,abis().values())),'analysisOutputLogicalBytes':sum(map(logical_bytes,analysis_abis().values())),
        'producerBytesByField':{name:logical_bytes(item) for name,item in abis().items()},'analysisBytesByField':{name:logical_bytes(item) for name,item in analysis_abis().items()},
        'extraProducerTemporaryBudgetBytes':512<<20,'analysisExtraLogicalBudgetBytes':1<<30,
        'scope':'One75.5MBoriginalout/302MBpromotedweight,6sourcefields~49MB,38MBjoinedBF16/151MBjoinedFP64,produceroutput/basis/QR/reconstruction/hash/copy budget explicit. Parent82peak2079.3MBRSS1998device withlarger453MBGU/430MBoutputs;4096RAM3072VRAM4CPU3GBdisk90expected300cap. Nofullmodel/GUprojection/source/attention/encoder/VAE. Fullmodel fallback sizing inapplicable; actual selected one-output-map science. Postexit4smallQRdiagnosticproducts and FP64NumPy targetaccounts under1GiBextra.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=2,logical_checkpoint_replays=0,conditioner_policy='Paid82current-inputC/ownproduct and73read/gate/residual. Originalout plus82GUformation/upstreampaid; no rawprompt producer.',
        extra={'purpose':'Same-coordinate completewrite projection, seal/exit then82primary73secondary full/centered/signed everyrow. No rank/seed/outputfit/reference/continuation/image.',
               'modelLoadScope':'Oneoriginalout75497472B;6FP64products12.684BMAC+4postexitdiagnosticproducts50.332MMAC,2QR95.071MFLOPseparate. Historical82GU19.327Bremainscomposedcost.'})
    costs=work();request={'schema':'tce-current-row-output-request-v1','ownedSources':OWNED,'sourceHashes':{item.name:sha(item) for item in payload.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'torch','dtype':'float64','batch_size':1,'seq_len':1536,'needs':{'cuda':True},'timeoutSeconds':300,
        'prospectiveCandidateMACs':costs['candidateMACs'],'prospectiveDiagnosticMACs':costs['diagnosticMACs'],'prospectiveAssayMACs':costs['assayMACs'],'fixedAddress':{'clock':0,'singleBlock':0},
        'protocol':'6paidfields+originalout;canonicalFP64rowQR(C),U.TJ->originalWout->Ucompleteupdate,onefinalFP64->FP32->BF16,separateBF16gate/add. ActualJBF16joinhashbeforepromotion. Seal/process exitbefore82update/exits and73targets.4postexitUR/Gramproducts50.332MMACseparate;primary82secondary73full/centered/everyrow/signederroridentity. NoJrecon/fulloutreference/outputfit/rank/seed/GU/source/attention/continuation/image/zero-neuralclaim.'}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(payload/'request.json',request);save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[item.name for item in payload.iterdir()]+['preflight.v1.json']})
    return {'root':str(root),'requestSHA256':request['requestSHA256'],'candidateMACs':costs['candidateMACs'],'diagnosticMACs':costs['diagnosticMACs']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same handle')
    cmd=command()[:-5]
    for name in WHEELS:cmd+=['--with','./'+name]
    cmd+=['python','-u','row_output_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-current-row-output-v1',model=r['model'],task='analysis',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={key:r[key] for key in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','scheduling_contract']},ram_mb=4096,vram_mb=3072,cpu_threads=4,disk_gb=3,est_wall_s=90,timeout_s=300,detach=True,retry_on_kill=False,preflight=True,
        note='Source82clock0single0 C/product+paid73read/gate/residual/originalout75MB.6FP64products12.684BMAC/same128rowQR2/95.071MFLOPs;onefinaldoublecast/nativeBF16gate/add.4postexitUR/Gramdiagnostics50.332MMAC;82primary73secondary/fullcentered/signedall1536rows. NoJrecon/outputfit/rankseed/GU/reference/image.4096RAM3072VRAM4CPU3GBdisk90expected300cap,noretry,onelease;all source/history/maps paid.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();verify(root,read(root/'request.json'));(root/'response-result').mkdir();started=time.monotonic()
    with subprocess.Popen([sys.executable,'-u',str(root/'row_output_worker.py'),str(root)],cwd=root) as producer:
        code=producer.wait()
        if code:raise subprocess.CalledProcessError(code,producer.args)
        save(root/'response-result/producer-exit.json',{'schema':'tce-current-row-output-producer-exit-v1','producerPID':producer.pid,'returncode':code,'requestSHA256':read(root/'request.json')['requestSHA256'],
            'candidateSealSHA256':sha(root/'response-result/candidate-seal.json'),'producerProcessSeconds':time.monotonic()-started,'scope':'Parent observed successful producer exit before launching82/73 target analyzer.'})
    subprocess.run([sys.executable,'-u',str(root/'row_output_analysis.py'),str(root)],check=True,cwd=root)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','execute','collect']);parser.add_argument('root');args=parser.parse_args()
    if args.action=='collect':
        from response_study import collect
        value=collect(args.root)
    else:value=globals()[args.action](args.root)
    print(json.dumps(value))
