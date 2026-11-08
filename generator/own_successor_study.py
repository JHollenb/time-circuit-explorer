"""One guarded own-state successor, then separate post-exit comparison."""
import argparse
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read,save,sha,identity
from own_successor import SOURCE_NAMES,PARAMETERS,abis,work
from own_successor_analysis import analysis_abis
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['own_successor.py','own_successor_worker.py','own_successor_ledger.py','own_successor_analysis.py','own_successor_study.py','test_own_successor.py',
       'gate_up_basis.py','receiver_row_output.py','row_output.py','gate_up_basis_ledger.py','gate_up_basis_analysis.py',
       'image_partition_moments.py','fixed_text_memory.py','terminal_liveness.py','terminal_attention_ledger.py',
       'paired_fields.py','encoder_atoms.py','flux_clock.py','flux_clock_study.py','complete_masked_study.py','masked_kv_study.py',
       'terminal_capture_guard_v2.py','projected_complement.py','sparse_text_kv.py']
WHEELS=['mdb_runtime-0.1.0-py3-none-any.whl','saturn_pub-0.4.1-py3-none-any.whl','mrun_pub-0.1.2-py3-none-any.whl']


def logical_bytes(item):
    sizes={'torch.bfloat16':2,'torch.float64':8,'float64':8,'torch.float32':4,'torch.int8':1}
    return math.prod(item['shape'])*sizes[item['dtype']]


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from projected_complement import fields
    api=Api();intake_path=HERE.parent/'artifacts/own-successor-intake-cycle87.json';i=read(intake_path)
    if i['verifierSHA256']!=sha(HERE/'own_successor_intake.py') or i['byteTemplateSHA256']!=sha(HERE/'publish_sparse_text_kv.py'):raise ValueError('Fresh intake source differs')
    for parent in i['parents'].values():
        actual=api.json('GET','/api/jobs/'+parent['jobID'])
        if actual['state']!='succeeded' or actual['payload_custody']!=parent['custody'] or actual['result']['diagnostics']['cwd']!=parent['root']:raise ValueError('Original custody differs')
    survey=read(HERE.parent/'docs/results/research-review-cycle87/advisor-survey.json')
    storage=read(HERE.parent/'docs/results/storage-audit-cycle87/publication.json')
    if survey['cycle']!=87 or not survey['newFullNewestPostSurvey'] or storage['cycle']!=87:raise ValueError('Scheduled reviews incomplete')
    previous=HERE.parent/'artifacts/receiver-weighted-row-output-v1';p86=previous/'result';s86=read(p86/'inputs.json')
    for key in ['nativeSources','versions','numericalProfile']:
        if i['sourceConfig'][key]!=s86[key]:raise ValueError('Source/runtime profile differs')
    for item in [*i['sourceFields'].values(),*list(fields(i['comparisonFields']))]:
        if i['files'][item['path']]['sha256']!=item['sha256'] or i['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Complete intake witness missing')
    for old in [read(p86/'request.json'),read(HERE.parent/'artifacts/current-state-gate-up-v1/result/request.json'),read(HERE.parent/'artifacts/image-partition-moments-v1/result/request.json')]:
        for name in OWNED:
            if name in old['ownedSources'] and sha(HERE/name)!=old['sourceHashes'][name]:raise ValueError('Reused frozen helper changed:'+name)
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for name in OWNED:shutil.copyfile(HERE/name,payload/name)
    for name in WHEELS:shutil.copyfile(previous/'payload'/name,payload/name)
    for source,target in [('docs/plans/receiver-own-state-successor.md','frozen-plan.md'),('docs/plans/own-successor-implementation.md','implementation-notes.md')]:shutil.copyfile(HERE.parent/source,payload/target)
    shutil.copyfile(intake_path,payload/'intake.json')
    denied=set(s86['deniedNumericalInputs'])
    for directory in [p86,HERE.parent/'artifacts/image-partition-moments-v1/result']:
        denied|={x['path'] for x in fields(read(directory/'candidate.json'))}
        if (directory/'analysis-fields.json').exists():denied|={x['path'] for x in fields(read(directory/'analysis-fields.json'))}
    config={key:i['sourceConfig'][key] for key in ['nativeSources','versions','numericalProfile']}
    config.update(modelRoot=i['modelRoot'],imageAssets={n:h for n,h in i['sourceConfig']['imageAssets'].items() if n.startswith('transformer/')},
        sourceFields=i['sourceFields'],comparisonFields=i['comparisonFields'],originalParameters=i['originalParameters'],
        parents=i['parents'],metadataFiles=i['metadataFiles'],intakeSHA256=sha(intake_path),
        sourceRoots=[x['root'] for x in i['parents'].values()],
        deniedNumericalInputs=sorted(denied-{x['path'] for x in i['sourceFields'].values()}),
        preStageReviewSnapshots={n:sha(HERE.parent/n) for n in ['docs/results/research-review-cycle87/advisor-survey.json','docs/results/storage-audit-cycle87/publication.json','docs/results/receiver-weighted-row-output-evidence/advisor.json']})
    save(payload/'inputs.json',config)
    host=next(x for x in api.json('GET','/api/hosts') if x['name']=='beast')
    save(payload/'resource-basis.json',{'intentFirstFullModelPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'closestParentMeasurements':{'GU82':read(HERE.parent/'docs/results/current-state-gate-up/scheduler.json')['result'],
                                    'output86':read(HERE.parent/'docs/results/receiver-weighted-row-output/scheduler.json')['result']},
        'liveHost':{k:host[k] for k in ['name','active_jobs','admission','telemetry','ram_total_mb','vram_total_mb']},
        'originalParameterBF16Bytes':245367296,'physicalQKVBF16Bytes':9216*3072*2,'promotedGUFP64Bytes':18432*3072*8,
        'promotedOutFP64Bytes':3072*12288*8,'onePaddedFusedBackingBytes':1536*27648*2,
        'producerOutputLogicalBytes':sum(map(logical_bytes,abis().values())),'analysisOutputLogicalBytes':sum(map(logical_bytes,analysis_abis().values())),
        'producerBytesByField':{n:logical_bytes(a) for n,a in abis().items()},'analysisBytesByField':{n:logical_bytes(a) for n,a in analysis_abis().items()},
        'producerExtraTemporaryBudgetBytes':1<<30,'analysisExtraTemporaryBudgetBytes':1<<30,
        'scope':'Combined single1 original4params245MB, physicalQKV56.6MB, separate padded fused buffers84.9MB each, promotedGU453MB freed before302MB out, retained FP64GU226MB and current attention/bases. Prior82RSS2079/1998device and86RSS1823/1566device are components,not combined measured peak. New allocation-derived6144RAM4096VRAM4CPU4GBdisk120estimate360cap; actualadmission/peaks authoritative. No fullmodel/encoder/VAE load.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=2,logical_checkpoint_replays=0,
        conditioner_policy='Actual86text/image plus paid70clock/RoPE. All current normalized/QKV/read/GU/product/outputbases re-formed; original single1maps paid.',
        extra={'purpose':'One clock0single1 ownstate transition, candidate seal/process exit before paid70 successor comparison; fullcentered D+W/all1536rows.',
               'modelLoadScope':'4originaltensors245MB only. One slicedBF16QKV,17FP64products,5FP32moments,imageSDPA,4QR;94.625Bcandidate+.755BdiagnosticMAC,QR492.132MFLOPs separate. No hidden fullGU/out ornewreference/image/fit/sweep.'})
    costs=work();r={'schema':'tce-own-successor-request-v1','ownedSources':OWNED,'sourceHashes':{p.name:sha(p) for p in payload.iterdir()},'scheduling_contract':contract,
        'model':'flux2-klein-4b-distilled','device':'cuda','backend':'torch','dtype':'mixed-bfloat16-float32-float64','batch_size':1,'seq_len':1536,'needs':{'cuda':True},'timeoutSeconds':360,
        'prospectiveCandidateMACs':costs['candidateMACs'],'prospectiveDiagnosticMACs':costs['diagnosticMACs'],'prospectiveAssayMACs':costs['assayMACs'],'fixedAddress':{'clock':0,'singleBlock':1},
        'protocol':'Own86incoming->native norm/mod->physicalBF16QKV rows0:9216 (kernelparityunclaimed),nativeRMS/RoPE,currentimageSDPA/textmoments;frozen82operand-onlyGU,admittednativeproduct;frozen86receiveroutputusingoriginalsingle1map;nativegate/add. Re-formallbases/V. No archivednormalized/read/product/bases/targets untilseal+observedexit;then D+W signed/fullcentered/everyrow and8optionaldiagnosticproducts. All learnedmaps/clocks paid;noimage/compactclosure/zero-neuralclaim.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(payload/'request.json',r)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'root':str(root),'requestSHA256':r['requestSHA256'],'candidateMACs':costs['candidateMACs'],'diagnosticMACs':costs['diagnosticMACs']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same handle')
    cmd=command()[:-5]
    for name in WHEELS:cmd+=['--with','./'+name]
    cmd+=['python','-u','own_successor_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='Own86text/image->single1 norm/slicedQKV/RMS/RoPE/liveattention/moments/82operand-onlyGU/86output/nativegate-add.4params245MB,94.625Bcandidate+.755BdiagnosticMAC,4QR492M FLOPs. Seal+exit before70D+W fullcentered/all1536rows. One guardedCUDA lease,6144RAM4096VRAM4CPU4GBdisk120est360cap,noretry. Stop afteroneblock;noarchivedoperands/fullGUorout/reference/image;learnedmaps paid.'
    if len(note)>500:raise ValueError('Queue note exceeds frontend limit')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-own-successor-v1',model=r['model'],task='analysis',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','scheduling_contract']},
        ram_mb=6144,vram_mb=4096,cpu_threads=4,disk_gb=4,est_wall_s=120,timeout_s=360,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);(root/'response-result').mkdir();tick=time.monotonic()
    with subprocess.Popen([sys.executable,'-u',str(root/'own_successor_worker.py'),str(root)],cwd=root) as producer:
        code=producer.wait()
        if code:raise subprocess.CalledProcessError(code,producer.args)
        save(root/'response-result/producer-exit.json',{'schema':'tce-own-successor-producer-exit-v1','requestSHA256':r['requestSHA256'],
            'producerPID':producer.pid,'returncode':code,'candidateSealSHA256':sha(root/'response-result/candidate-seal.json'),
            'producerProcessSeconds':time.monotonic()-tick,'scope':'Parent observed successful producer exit before target analyzer starts.'})
    subprocess.run([sys.executable,'-u',str(root/'own_successor_analysis.py'),str(root)],check=True,cwd=root)


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','execute','collect']);parser.add_argument('root');a=parser.parse_args()
    if a.action=='collect':
        from response_study import collect
        result=collect(a.root)
    else:result=globals()[a.action](a.root)
    print(json.dumps(result))
