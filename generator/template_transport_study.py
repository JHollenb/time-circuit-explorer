"""One guarded matched pair; reviews/intake/source freeze precede any submission."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read,save,sha,identity
from dense_corner_study import OWNED as PREVIOUS_OWNED
from own_successor_study import WHEELS,logical_bytes
from template_transport import abis,work
from template_transport_analysis import analysis_abis
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=PREVIOUS_OWNED+['template_transport.py','template_transport_worker.py','template_transport_ledger.py',
                     'template_transport_analysis.py','template_transport_study.py','test_template_transport.py']


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from projected_complement import fields
    from template_transport_intake import validate_metadata
    api=Api();intake=HERE.parent/'artifacts/template-transport-intake-cycle90.json';i=read(intake)
    if i['verifierSHA256']!=sha(HERE/'template_transport_intake.py') or i['byteTemplateSHA256']!=sha(HERE/'own_successor_intake.py'):
        raise ValueError('Fresh intake source differs')
    validate_metadata(i)
    for parent in i['parents'].values():
        actual=api.json('GET','/api/jobs/'+parent['jobID'])
        if actual['state']!='succeeded' or actual['payload_custody']!=parent['custody'] or actual['result']['diagnostics']['cwd']!=parent['root']:
            raise ValueError('Original successful identity/custody differs')
    survey=read(HERE.parent/'docs/results/research-review-cycle90/advisor-survey.json')
    storage=read(HERE.parent/'docs/results/storage-audit-cycle90/publication.json')
    if survey['cycle']!=90 or not survey['newFullNewestPostSurvey'] or len(survey['reviewedWorkingSnapshots'])!=55 or storage['cycle']!=90 or storage['totalRestorePointers']!=93:
        raise ValueError('Scheduled full90 reviews incomplete')
    review=read(HERE.parent/'docs/results/template-transport-evidence/advisor-prestage.json')
    if review['decision']!='approved-exact-snapshots':raise ValueError('Exact pre-stage review missing')
    for name,digest in review['sourceHashes'].items():
        if sha(HERE.parent/name)!=digest:raise ValueError('Reviewed source differs:'+name)
    for item in [*i['sourceFields'].values(),*i['templateFields'].values(),*list(fields(i['comparisonFields']))]:
        if i['files'][item['path']]['sha256']!=item['sha256'] or i['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:
            raise ValueError('Complete fresh intake witness missing')
    previous=HERE.parent/'artifacts/dense-corner-v1';old=read(previous/'result/request.json');s89=read(previous/'result/inputs.json')
    for name in PREVIOUS_OWNED:
        if sha(HERE/name)!=old['sourceHashes'][name]:raise ValueError('Frozen predecessor helper changed:'+name)
    if i['sourceConfig']!={k:s89[k] for k in ['nativeSources','versions','imageAssets','numericalProfile']}:
        raise ValueError('Original runtime/source profile differs')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for name in OWNED:shutil.copyfile(HERE/name,payload/name)
    for name in WHEELS:shutil.copyfile(previous/'payload'/name,payload/name)
    for source,target in [('docs/plans/frozen-template-matched-pair.md','frozen-plan.md'),('docs/plans/template-transport-implementation.md','implementation-notes.md')]:
        shutil.copyfile(HERE.parent/source,payload/target)
    shutil.copyfile(intake,payload/'intake.json')
    denied=set(s89['deniedNumericalInputs'])
    for directory in [previous/'result',HERE.parent/'artifacts/matched-parent-v1/result']:
        for name in ['candidate.json','analysis-fields.json']:denied|={x['path'] for x in fields(read(directory/name))}
    denied|={x['path'] for x in fields(i['comparisonFields'])}
    allowed={x['path'] for x in [*i['sourceFields'].values(),*i['templateFields'].values(),*i['comparisonFields']['parentIncoming'].values()]}
    config={k:i['sourceConfig'][k] for k in ['nativeSources','versions','imageAssets','numericalProfile']}
    config.update(modelRoot=i['modelRoot'],sourceFields=i['sourceFields'],comparisonFields=i['comparisonFields'],templateFields=i['templateFields'],
        originalParameters=i['originalParameters'],parents=i['parents'],metadataFiles=i['metadataFiles'],intakeSHA256=sha(intake),
        sourceRoots=[x['root'] for x in i['parents'].values()],deniedNumericalInputs=sorted(denied-allowed),
        preStageReviewSnapshots={n:sha(HERE.parent/n) for n in ['docs/results/research-review-cycle90/advisor-survey.json',
            'docs/results/storage-audit-cycle90/publication.json','docs/results/template-transport-evidence/advisor-prestage.json']})
    save(payload/'inputs.json',config)
    host=next(x for x in api.json('GET','/api/hosts') if x['name']=='beast')
    save(payload/'resource-basis.json',{
        'intentFirstFullModelPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'closestDynamicProducerMeasurement':read(HERE.parent/'docs/results/matched-parent/scheduler.json')['result'],
        'latestDenseProducerMeasurement':read(HERE.parent/'docs/results/dense-corner/scheduler.json')['result'],
        'liveHost':{k:host[k] for k in ['name','active_jobs','admission','telemetry','ram_total_mb','vram_total_mb']},
        'originalParameterBF16Bytes':245367296,'templatePayloadBytes':45613056,'physicalQKVBF16Bytes':56623104,
        'oneArmOutputLogicalBytes':sum(map(logical_bytes,abis().values())),
        'bothArmsOutputLogicalBytes':2*sum(map(logical_bytes,abis().values())),
        'analysisOutputLogicalBytes':sum(map(logical_bytes,analysis_abis().values())),
        'analysisEightDecodedCornersBytes':8*1536*3072*8,'analysisThirteenVectorsBytes':13*1536*3072*8,
        'analysisThirteenCenteredVectorsBytes':13*1536*3072*8,'extraProducerTemporaryAllowanceBytes':1<<30,'extraAnalyzerTemporaryAllowanceBytes':1<<30,
        'streaming':'One original parameter/template load;two equal-shape arms sequentially to retain the same finite call geometry. Persist/release first complete outputs before second. No parity-qualified batched alternative assumed.',
        'scope':'Allocation-based partial-model exception. Prior88onearm2826.8MBRSS/2554sampleddevice;89dense2286.5/1164,neither guarantees newpeak. No453MB promotedGU/newQR/sketch;originalout302MB and one currentarm live. CPU8corners+13vectors+13centered+1GiB.6144RAM4096VRAM4CPU4GBdisk90estimate360cap;guard/admission/actualpeaks authoritative. Fullmodel preview inapplicable.'})
    contract=one_lease_scheduling_contract(logical_branches=2,logical_operations=3,logical_checkpoint_replays=0,
        conditioner_policy='Actual parent70/73 andown86incoming/commonpaid70clocks;one immutable parent88GU/outputtemplate,alloperandsfresh.',
        extra={'purpose':'One matched-template pair,sealbothevaluations+observedexit before89/88/87/70 comparisons.',
               'modelLoadScope':'4originalsingle1tensors245MB+45.6MBtemplates.144.556BMAC,20FP64/2BF16/10FP32/2SDPA,noQR/compile. ScalarCPUeightcorneraccounts,no fit/image/prefix/head/sweep.'})
    costs=work();r={'schema':'tce-template-transport-request-v1','ownedSources':OWNED,'sourceHashes':{p.name:sha(p) for p in payload.iterdir()},
        'scheduling_contract':contract,'model':'flux2-klein-4b-distilled','device':'cuda','backend':'torch','dtype':'mixed-bfloat16-float32-float64',
        'batch_size':1,'seq_len':1536,'needs':{'cuda':True},'timeoutSeconds':360,'prospectiveCandidateMACs':costs['candidateMACs'],
        'prospectiveDiagnosticMACs':0,'fixedAddress':{'clock':0,'singleBlock':1},
        'protocol':'Same explicitly reloaded template,parent+owninputs;12input33output typedActs each. Sealbothandexit beforecomparisons. A0/A1 andB=A1-A0 retained,nonzeroA0 notqualitygate.144555638784matrixMAC,alllearned/source/clock/templatehistorypaid.'}
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
    cmd+=['python','-u','template_transport_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='One frozen88template onparent70/73+own86inputs.4maps245MB+45.6MBtemplates;144.556BMAC,20FP64/2BF16/10FP32/2SDPA,noQR/compile. Persist/releaseeacharm,sealboth+exitbefore8cornercomparison;nonzeroA0retained. OneguardedCUDAlease,6144RAM4096VRAM4CPU4GBdisk90est360cap,noretry. Stopafter2blocks;no prefix/head/image/fit/sweep;learnedmaps/historypaid.'
    if len(note)>500:raise ValueError('Queue note too long')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-template-transport-v1',model=r['model'],task='analysis',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','scheduling_contract']},
        ram_mb=6144,vram_mb=4096,cpu_threads=4,disk_gb=4,est_wall_s=90,timeout_s=360,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);(root/'response-result').mkdir();tick=time.monotonic()
    with subprocess.Popen([sys.executable,'-u',str(root/'template_transport_worker.py'),str(root)],cwd=root) as producer:
        code=producer.wait()
        if code:raise subprocess.CalledProcessError(code,producer.args)
        save(root/'response-result/producer-exit.json',{'schema':'tce-template-transport-producer-exit-v1','requestSHA256':r['requestSHA256'],
            'producerPID':producer.pid,'returncode':code,'candidateSealSHA256':sha(root/'response-result/candidate-seal.json'),
            'producerProcessSeconds':time.monotonic()-tick,'scope':'Parent observed successful two-arm producer exit before eight-corner analyzer starts.'})
    subprocess.run([sys.executable,'-u',str(root/'template_transport_analysis.py'),str(root)],check=True,cwd=root)


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
