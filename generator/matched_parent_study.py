"""One guarded matched-parent successor; all comparisons follow seal and exit."""
import argparse
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read,save,sha,identity
from own_successor_study import OWNED as PREVIOUS_OWNED,WHEELS,logical_bytes
from matched_parent import abis,work
from matched_parent_analysis import analysis_abis
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=PREVIOUS_OWNED+['matched_parent.py','matched_parent_worker.py','matched_parent_ledger.py',
                     'matched_parent_analysis.py','matched_parent_study.py','test_matched_parent.py']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from projected_complement import fields
    api=Api();intake_path=HERE.parent/'artifacts/matched-parent-intake-cycle88.json';i=read(intake_path)
    if i['verifierSHA256']!=sha(HERE/'matched_parent_intake.py') or i['byteTemplateSHA256']!=sha(HERE/'own_successor_intake.py'):
        raise ValueError('Fresh intake sources differ')
    for parent in i['parents'].values():
        actual=api.json('GET','/api/jobs/'+parent['jobID'])
        if actual['state']!='succeeded' or actual['payload_custody']!=parent['custody'] or actual['result']['diagnostics']['cwd']!=parent['root']:
            raise ValueError('Original custody differs')
    review=read(HERE.parent/'docs/results/matched-parent-evidence/advisor-prestage.json')
    if review['decision']!='approved-exact-snapshots':raise ValueError('Exact pre-stage review missing')
    for name,digest in review['sourceHashes'].items():
        if sha(HERE.parent/name)!=digest:raise ValueError('Reviewed source differs:'+name)
    for item in [*i['sourceFields'].values(),*list(fields(i['comparisonFields']))]:
        if i['files'][item['path']]['sha256']!=item['sha256'] or i['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:
            raise ValueError('Complete intake witness missing')
    previous=HERE.parent/'artifacts/own-successor-v1';p87=previous/'result'
    old=read(p87/'request.json');s87=read(p87/'inputs.json')
    for name in PREVIOUS_OWNED:
        if sha(HERE/name)!=old['sourceHashes'][name]:raise ValueError('Frozen87 helper differs:'+name)
    if i['sourceConfig']!={k:s87[k] for k in ['nativeSources','versions','imageAssets','numericalProfile']}:
        raise ValueError('Frozen source/runtime profile differs')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for name in OWNED:shutil.copyfile(HERE/name,payload/name)
    for name in WHEELS:shutil.copyfile(previous/'payload'/name,payload/name)
    for source,target in [('docs/plans/matched-parent-successor.md','frozen-plan.md'),('docs/plans/matched-parent-implementation.md','implementation-notes.md')]:
        shutil.copyfile(HERE.parent/source,payload/target)
    shutil.copyfile(intake_path,payload/'intake.json')
    denied=set(s87['deniedNumericalInputs'])
    for directory in [p87,HERE.parent/'artifacts/receiver-weighted-row-output-v1/result']:
        for name in ['candidate.json','analysis-fields.json']:
            denied|={x['path'] for x in fields(read(directory/name))}
    denied|={x['path'] for x in fields(i['comparisonFields'])}
    config={k:i['sourceConfig'][k] for k in ['nativeSources','versions','imageAssets','numericalProfile']}
    config.update(modelRoot=i['modelRoot'],sourceFields=i['sourceFields'],comparisonFields=i['comparisonFields'],
        originalParameters=i['originalParameters'],parents=i['parents'],metadataFiles=i['metadataFiles'],
        intakeSHA256=sha(intake_path),sourceRoots=[x['root'] for x in i['parents'].values()],
        deniedNumericalInputs=sorted(denied-{x['path'] for x in i['sourceFields'].values()}),
        frozenProgramSHA256=sha(HERE/'own_successor.py'),previousRequestSHA256=old['requestSHA256'],
        preStageReviewSnapshots={n:sha(HERE.parent/n) for n in [
            'docs/results/research-review-cycle87/advisor-survey.json',
            'docs/results/storage-audit-cycle87/publication.json','docs/results/matched-parent-evidence/advisor-prestage.json']})
    save(payload/'inputs.json',config)
    host=next(x for x in api.json('GET','/api/hosts') if x['name']=='beast')
    save(payload/'resource-basis.json',{
        'intentFirstFullModelPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'closestWholeProducerMeasurement':read(HERE.parent/'docs/results/own-successor/scheduler.json')['result'],
        'liveHost':{k:host[k] for k in ['name','active_jobs','admission','telemetry','ram_total_mb','vram_total_mb']},
        'originalParameterBF16Bytes':245367296,'physicalQKVBF16Bytes':9216*3072*2,
        'producerOutputLogicalBytes':sum(map(logical_bytes,abis().values())),
        'analysisOutputLogicalBytes':sum(map(logical_bytes,analysis_abis().values())),
        'analysisFiveDecodedCornersBytes':5*1536*3072*8,
        'analysisSixVectorsBytes':6*1536*3072*8,'analysisSixCenteredVectorsBytes':6*1536*3072*8,
        'extraProducerTemporaryAllowanceBytes':1<<30,'extraAnalyzerTemporaryAllowanceBytes':1<<30,
        'scope':'Same measured87 producer2717.2MB RSS/2554sampleddevice,not future peak guarantee. NewCPU scalar analyzer adds decoded5corners/sixvectors/centeredcopies and4retainedfields. GPU producer unchanged608MBlogical outputs/245MBmaps/56.6MBphysicalQKV/twopadded85MBbuffers/promotedGU453MBfreedbeforeout302MB. Allocation-derived6144RAM4096VRAM4CPU4GBdisk120est360cap retained;actualadmission/peaks authoritative. Fullmodel fit inapplicable.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=2,logical_checkpoint_replays=0,
        conditioner_policy='Authenticated70/73parentinput plus commonpaid70clock/RoPE;all current operands/bases reformed with exactfrozen87numericalproducer.',
        extra={'purpose':'One matched-parent missing corner;seal+successfulproducerexit before own86/87/parent70comparisons.',
               'modelLoadScope':'4originalsingle1tensors245MB.94.625BcandidateMAC,4QR492.132MFLOPs;no8factorizationproducts. PostexitCPUFP64scalar/data accounts only,no dense reference/image/fit/sweep.'})
    costs=work()
    r={'schema':'tce-matched-parent-request-v1','ownedSources':OWNED,'sourceHashes':{p.name:sha(p) for p in payload.iterdir()},
       'scheduling_contract':contract,'model':'flux2-klein-4b-distilled','device':'cuda','backend':'torch',
       'dtype':'mixed-bfloat16-float32-float64','batch_size':1,'seq_len':1536,'needs':{'cuda':True},'timeoutSeconds':360,
       'prospectiveCandidateMACs':costs['candidateMACs'],'prospectiveDiagnosticMACs':0,'fixedAddress':{'clock':0,'singleBlock':1},
       'protocol':'Samefrozen87reducedprogram on parent70/73incoming. Everyoperand/basisfresh,samerank/seeds/slicing/rounding.9inputs45fields,sealthenexit before5corneranalysis. Eout=E0+T,W=E0+(T-D),fullcentered/allrows/undefinedratios. Tnotnativepropagation,E0notcompression-only;originalmaps/clocks/historicalworkpaid.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(payload/'request.json',r)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'root':str(root),'requestSHA256':r['requestSHA256'],'candidateMACs':costs['candidateMACs'],'diagnosticMACs':0}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted;inspect same handle')
    cmd=command()[:-5]
    for name in WHEELS:cmd+=['--with','./'+name]
    cmd+=['python','-u','matched_parent_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='Same frozen87single1 on70/73parentinput;alloperands/basesfresh.4maps245MB,94.625BMAC,4QR492M FLOPs. Seal+exitbeforeown86/87/parent70fivecorner E0/T/U/D/W/out fullcentered/all1536rows. CPUscalaranalysis,no8factorizationproducts. One guardedCUDAlease,6144RAM4096VRAM4CPU4GBdisk120est360cap,noretry. Stopafteroneblock;no dense reference/image/fit/sweep;learnedmaps paid.'
    if len(note)>500:raise ValueError('Queue note too long')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-matched-parent-v1',model=r['model'],task='analysis',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','scheduling_contract']},
        ram_mb=6144,vram_mb=4096,cpu_threads=4,disk_gb=4,est_wall_s=120,timeout_s=360,detach=True,retry_on_kill=False,preflight=True,note=note)
    v={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',v);return v


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);(root/'response-result').mkdir();tick=time.monotonic()
    with subprocess.Popen([sys.executable,'-u',str(root/'matched_parent_worker.py'),str(root)],cwd=root) as producer:
        code=producer.wait()
        if code:raise subprocess.CalledProcessError(code,producer.args)
        save(root/'response-result/producer-exit.json',{'schema':'tce-matched-parent-producer-exit-v1','requestSHA256':r['requestSHA256'],
            'producerPID':producer.pid,'returncode':code,'candidateSealSHA256':sha(root/'response-result/candidate-seal.json'),
            'producerProcessSeconds':time.monotonic()-tick,'scope':'Parent observed successful producer exit before five-corner analyzer starts.'})
    subprocess.run([sys.executable,'-u',str(root/'matched_parent_analysis.py'),str(root)],check=True,cwd=root)


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
