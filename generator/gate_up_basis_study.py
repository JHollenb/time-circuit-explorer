"""One guarded current-state producer lease; exit before saved native comparison."""
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
from gate_up_basis import SOURCE_NAMES,COMPARISON_NAMES,abis,work
from gate_up_basis_analysis import analysis_abis

HERE=Path(__file__).resolve().parent
OWNED=['gate_up_basis.py','gate_up_basis_worker.py','gate_up_basis_ledger.py','gate_up_basis_analysis.py','gate_up_basis_study.py','test_gate_up_basis.py',
    'encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','masked_kv_study.py','terminal_capture_guard_v2.py','projected_complement.py','sparse_text_kv.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def logical_bytes(abi):
    return math.prod(abi['shape'])*(1 if abi['dtype']=='torch.int8' else 2 if abi['dtype']=='torch.bfloat16' else 8)


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from projected_complement import fields
    from publish_source_native import checked
    from publish_matched_single import audit
    parent,col,job,pr=checked(HERE.parent/'artifacts/matched-single-write-v1');p=parent/'result';s=read(p/'inputs.json');dense=read(p/'replay-dense.json');v=read(parent/'retained-array-verification.json')
    audit(HERE.parent/'docs/results/matched-single-write')
    source={n:(s['sourceFields'][n] if n in ['text','image'] else dense['fields'][n]) for n in SOURCE_NAMES}
    comparison={n:dense['fields'][n] for n in COMPARISON_NAMES}
    for item in [*source.values(),*comparison.values()]:
        if v['files'][item['path']]['sha256']!=item['sha256'] or v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Parent full-byte witness absent')
    api=Api();live=api.json('GET','/api/jobs/'+job['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=col['custody']:raise ValueError('Parent authoritative custody differs')
    for n in OWNED:
        if n in pr['ownedSources'] and sha(HERE/n)!=pr['sourceHashes'][n]:raise ValueError('Frozen reused helper differs:'+n)
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for n in OWNED:shutil.copyfile(HERE/n,payload/n)
    for n in WHEELS:shutil.copyfile(parent/'payload'/n,payload/n)
    for n,dest in [('docs/plans/current-state-gate-up-basis.md','frozen-plan.md'),('docs/plans/current-state-gate-up-implementation.md','implementation-notes.md')]:shutil.copyfile(HERE.parent/n,payload/dest)
    denied=set(s['deniedNumericalInputs'])|{x['path'] for x in fields(read(p/'candidate.json'))}|{x['path'] for x in fields(read(p/'anatomy.json'))}
    denied|={x['path'] for x in fields(read(p/'targets.json'))};denied|={x['path'] for x in fields(s['sourceFields'])}
    config={k:s[k] for k in ['modelRoot','nativeSources','versions','imageAssets','numericalProfile']}
    config.update(sourceFields=source,comparisonFields=comparison,originalParameters=s['originalParameters'][:2],parentJobID=job['job_id'],parentCustody=col['custody'],parentRoot=job['result']['diagnostics']['cwd'],
        parentDenseSealSHA256=sha(p/'seal-dense.json'),parentCandidateSealSHA256=sha(p/'candidate-seal.json'),parentByteVerificationSHA256=sha(parent/'retained-array-verification.json'),
        parentRecords={n:sha(p/n) for n in ['inputs.json','replay-dense.json','execution-dense.json','seal-dense.json','candidate-seal.json','report.json']},
        sourceRoots=sorted({job['result']['diagnostics']['cwd'],*(str(Path(x['path']).parents[2]) for x in source.values())}),
        deniedNumericalInputs=sorted(denied-{x['path'] for x in source.values()}),
        preStageReviewSnapshots={n:sha(HERE.parent/n) for n in ['docs/results/rectifier-sign-support-evidence/advisor.json','docs/results/research-review-cycle81/advisor-survey.json','docs/results/storage-audit-cycle81/publication.json']})
    save(payload/'inputs.json',config);host=next(x for x in api.json('GET','/api/hosts') if x['name']=='beast')
    save(payload/'resource-basis.json',{'intentFirstFullModelPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredParentJobID':job['job_id'],'measuredParent':{k:job['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},
        'liveHost':{k:host[k] for k in ['name','active_jobs','admission','telemetry','ram_total_mb','vram_total_mb']},
        'sourceFileBytes':sum(x['bytes'] for x in source.values()),'comparisonFileBytes':sum(x['bytes'] for x in comparison.values()),
        'originalLoadedParameterBytes':sum(x['bytes'] for x in config['originalParameters']),'promotedGUWeightFP64Bytes':18432*3072*8,
        'initializedBF16BackingBytes':1536*27648*2,'producerOutputLogicalBytes':sum(map(logical_bytes,abis().values())),
        'analysisOutputLogicalBytes':sum(map(logical_bytes,analysis_abis().values())),
        'producerBytesByField':{n:logical_bytes(x) for n,x in abis().items()},'analysisBytesByField':{n:logical_bytes(x) for n,x in analysis_abis().items()},
        'extraProducerTemporaryBudgetBytes':1<<30,'analysisExtraLogicalBudgetBytes':2<<30,
        'scope':'Selected two245MBoriginalparameters/5sourcefields28MB,453MBFP64GU,430MBlogicalproduceroutputs plus226MBreconstruction intermediates,85MBinitializedbacking,FP32/BF16admission/QR/library/hash staging under extra1GiB. Parent73measured2051MBRSS/1508MBdevice;6144RAM4096VRAM4CPU4GBdisk120expected360cap. Fullmodel preview is inapplicable to selected oneblock consumer; nofullmodel/QKV/attention/source/encoder/VAE execution. Postexit NumPy target/scalar analysis~2GiBtemporarybudget,4diagnosticCUDAproducts.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=2,logical_checkpoint_replays=0,
        conditioner_policy='Paid source73 clock0single0 completeZ/read/gate/text/image, original GU/output maps; no rawprompt source producer.',
        extra={'purpose':'One current-state rank128 completeGU producer and native local consumer, seal/exit then saved targets and exactinput/Gram diagnostics. No sweep/fit/newreference/image.',
               'modelLoadScope':'Two original parameters245366784B; source formation/QKV/attention not reexecuted.77.309BcandidateMAC+0.705BpostexitdiagnosticMAC,QR/scalar/dataworkadditional.'})
    costs=work();request={'schema':'tce-current-state-gate-up-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in payload.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'torch','dtype':'float64','batch_size':1,'seq_len':1536,'needs':{'cuda':True},'timeoutSeconds':360,
        'prospectiveCandidateMACs':costs['candidateMACs'],'prospectiveDiagnosticMACs':costs['diagnosticMACs'],'prospectiveAssayMACs':costs['assayMACs'],'fixedAddress':{'clock':0,'singleBlock':0},
        'protocol':'5sourceonly/2originalmaps,allcurrentrows2PCG64rank128signsketches,canonicalFP64HouseholderQR,completeGUreconstruction->FP32->BF16nativeview/nativeFlux2SwiGLU/fullout/gate/residual. Seal+process exit before6denseoldtargets. ExactpostexitCQt/QGramchargedseparately,allfull/centered/everyrow/actualexitincrement. No originalGUprojection, source refresh, fit/ranksweep/seedchange/reference/image/zero-neuralclaim.'}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(payload/'request.json',request)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in payload.iterdir()]+['preflight.v1.json']})
    return {'root':str(root),'requestSHA256':request['requestSHA256'],'candidateMACs':costs['candidateMACs'],'diagnosticMACs':costs['diagnosticMACs']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect the same handle')
    cmd=command()[:-5]
    for n in WHEELS:cmd+=['--with','./'+n]
    cmd+=['python','-u','gate_up_basis_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-current-state-gate-up-v1',model=r['model'],task='analysis',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','scheduling_contract']},
        ram_mb=6144,vram_mb=4096,cpu_threads=4,disk_gb=4,est_wall_s=120,timeout_s=360,detach=True,retry_on_kill=False,preflight=True,
        note='Source73clock0single0only:5paidfields2originalparameters245MB,rank128allrowsFP64Householdercurrentbasis/completeGU/nativeSwiGLU/fullout/gate/residual.77.309BcandidateMAC,0.705BpostexitdiagnosticMAC,QR397MFLOP/scalar/IOnotfree.430MBproducer+39MBanalysisarraysnearcompute. Seal/exitbefore6denseoldtargets;noQKV/attention/source/reference/image/sweep.6144RAM4096VRAM4CPU4GBdisk120expected360cap,noretry,onelease.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();verify(root,read(root/'request.json'));(root/'response-result').mkdir();started=time.monotonic()
    with subprocess.Popen([sys.executable,'-u',str(root/'gate_up_basis_worker.py'),str(root)],cwd=root) as producer:
        code=producer.wait()
        if code:raise subprocess.CalledProcessError(code,producer.args)
        save(root/'response-result/producer-exit.json',{'schema':'tce-current-state-gate-up-producer-exit-v1','producerPID':producer.pid,'returncode':code,'requestSHA256':read(root/'request.json')['requestSHA256'],
            'candidateSealSHA256':sha(root/'response-result/candidate-seal.json'),'producerProcessSeconds':time.monotonic()-started,'scope':'Parent observed successful exit before starting saved-target analysis process.'})
    subprocess.run([sys.executable,'-u',str(root/'gate_up_basis_analysis.py'),str(root)],check=True,cwd=root)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','execute','collect']);parser.add_argument('root');args=parser.parse_args()
    if args.action=='collect':
        from response_study import collect
        value=collect(args.root)
    else:value=globals()[args.action](args.root)
    print(json.dumps(value))
