"""One guarded original70 dense comparator at the existing own-state input."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read,save,sha,identity
from matched_parent_study import OWNED as PREVIOUS_OWNED
from own_successor_study import WHEELS,logical_bytes
from dense_corner import abis,work
from dense_corner_analysis import analysis_abis
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=PREVIOUS_OWNED+['dense_corner.py','dense_corner_worker.py','dense_corner_ledger.py',
                     'dense_corner_analysis.py','dense_corner_study.py','test_dense_corner.py']
DIRECTORIES={'70':'image-partition-moments-v1','73':'matched-single-write-v1','86':'receiver-weighted-row-output-v1','87':'own-successor-v1','88':'matched-parent-v1'}


def original_metadata(i,tag,name):
    path=HERE.parent/'artifacts'/DIRECTORIES[tag]/'result'/name
    remote=str(Path(i['parents'][tag]['root'])/'response-result'/name)
    if sha(path)!=i['metadataFiles'][remote]:raise ValueError('Authenticated corner metadata differs')
    return read(path)


def validate_corner_metadata(i):
    s87=original_metadata(i,'87','inputs.json');s88=original_metadata(i,'88','inputs.json')
    c86=original_metadata(i,'86','candidate.json')['fields'];c87=original_metadata(i,'87','candidate.json')['fields']
    c88=original_metadata(i,'88','candidate.json')['fields'];r70=original_metadata(i,'70','request.json')
    for n in ['text','image']:
        if i['sourceFields'][n]!=c86[n+'Out'] or s87['sourceFields'][n]!=i['sourceFields'][n] or s88['comparisonFields']['ownIncoming'][n]!=i['sourceFields'][n]:
            raise ValueError('Own input corner differs')
        if s88['sourceFields'][n]!=i['comparisonFields']['parentIncoming'][n] or c87[n+'Out']!=i['comparisonFields']['ownOutgoing'][n] or c88[n+'Out']!=i['comparisonFields']['matchedOutgoing'][n]:
            raise ValueError('Parent/reduced output corner differs')
        if s88['comparisonFields']['parentOutgoing'][n]!=i['comparisonFields']['parentOutgoing'][n] or s88['comparisonFields']['ownOutgoing'][n]!=i['comparisonFields']['ownOutgoing'][n]:
            raise ValueError('Existing comparison corner differs')
    for n in ['mod','cosine','sine']:
        if i['sourceFields'][n]!=s87['sourceFields'][n] or i['sourceFields'][n]!=s88['sourceFields'][n]:
            raise ValueError('Common native clock/RoPE differs')
    if i['originalParameters']!=s87['originalParameters'] or i['originalParameters']!=s88['originalParameters']:
        raise ValueError('Original map lineage differs')
    if i['frozenConsumerSHA256']!=r70['sourceHashes']['image_partition_moments.py'] or i['frozenConsumerSHA256']!=sha(HERE/'image_partition_moments.py'):
        raise ValueError('Original70 consumer differs')


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from projected_complement import fields
    api=Api();intake_path=HERE.parent/'artifacts/dense-corner-intake-cycle89.json';i=read(intake_path)
    if i['verifierSHA256']!=sha(HERE/'dense_corner_intake.py') or i['byteTemplateSHA256']!=sha(HERE/'own_successor_intake.py'):
        raise ValueError('Fresh intake source differs')
    validate_corner_metadata(i)
    for parent in i['parents'].values():
        actual=api.json('GET','/api/jobs/'+parent['jobID'])
        if actual['state']!='succeeded' or actual['payload_custody']!=parent['custody'] or actual['result']['diagnostics']['cwd']!=parent['root']:
            raise ValueError('Original custody differs')
    review=read(HERE.parent/'docs/results/dense-corner-evidence/advisor-prestage.json')
    if review['decision']!='approved-exact-snapshots':raise ValueError('Exact pre-stage review absent')
    for name,digest in review['sourceHashes'].items():
        if sha(HERE.parent/name)!=digest:raise ValueError('Reviewed source changed:'+name)
    for item in [*i['sourceFields'].values(),*list(fields(i['comparisonFields']))]:
        if i['files'][item['path']]['sha256']!=item['sha256'] or i['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:
            raise ValueError('Complete intake witness missing')
    previous=HERE.parent/'artifacts/matched-parent-v1';p88=previous/'result';old=read(p88/'request.json');s88=read(p88/'inputs.json')
    for name in PREVIOUS_OWNED:
        if sha(HERE/name)!=old['sourceHashes'][name]:raise ValueError('Frozen88 helper differs:'+name)
    if i['sourceConfig']!={k:s88[k] for k in ['nativeSources','versions','imageAssets','numericalProfile']}:
        raise ValueError('Original runtime/profile differs')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for name in OWNED:shutil.copyfile(HERE/name,payload/name)
    for name in WHEELS:shutil.copyfile(previous/'payload'/name,payload/name)
    for source,target in [('docs/plans/dense-own-state-corner.md','frozen-plan.md'),('docs/plans/dense-corner-implementation.md','implementation-notes.md')]:
        shutil.copyfile(HERE.parent/source,payload/target)
    shutil.copyfile(intake_path,payload/'intake.json')
    denied=set(s88['deniedNumericalInputs'])
    for tag in ['86','87','88']:
        directory=HERE.parent/'artifacts'/DIRECTORIES[tag]/'result'
        for name in ['candidate.json','analysis-fields.json']:denied|={x['path'] for x in fields(read(directory/name))}
    denied|={x['path'] for x in fields(i['comparisonFields'])}
    config={k:i['sourceConfig'][k] for k in ['nativeSources','versions','imageAssets','numericalProfile']}
    config.update(modelRoot=i['modelRoot'],sourceFields=i['sourceFields'],comparisonFields=i['comparisonFields'],
        originalParameters=i['originalParameters'],parents=i['parents'],metadataFiles=i['metadataFiles'],
        intakeSHA256=sha(intake_path),sourceRoots=[x['root'] for x in i['parents'].values()],
        deniedNumericalInputs=sorted(denied-{x['path'] for x in i['sourceFields'].values()}),
        frozenConsumerSHA256=i['frozenConsumerSHA256'],
        preStageReviewSnapshots={n:sha(HERE.parent/n) for n in ['docs/results/research-review-cycle87/advisor-survey.json','docs/results/storage-audit-cycle87/publication.json','docs/results/dense-corner-evidence/advisor-prestage.json']})
    save(payload/'inputs.json',config)
    host=next(x for x in api.json('GET','/api/hosts') if x['name']=='beast')
    save(payload/'resource-basis.json',{
        'intentFirstFullModelPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'priorReducedMeasurementNotNativePeak':read(HERE.parent/'docs/results/matched-parent/scheduler.json')['result'],
        'liveHost':{k:host[k] for k in ['name','active_jobs','admission','telemetry','ram_total_mb','vram_total_mb']},
        'originalParameterBF16Bytes':245367296,'fullFusedBF16Bytes':1536*27648*2,
        'producerOutputLogicalBytes':sum(map(logical_bytes,abis().values())),
        'analysisOutputLogicalBytes':sum(map(logical_bytes,analysis_abis().values())),
        'producerBytesByField':{k:logical_bytes(a) for k,a in abis().items()},
        'analysisSixDecodedCornersBytes':6*1536*3072*8,'analysisTenVectorsBytes':10*1536*3072*8,
        'analysisTenCenteredVectorsBytes':10*1536*3072*8,
        'producerExtraTemporaryAllowanceBytes':1<<30,'analyzerExtraTemporaryAllowanceBytes':1<<30,
        'scope':'New allocation-derived4096RAM3072VRAM4CPU3GBdisk90est300cap. Fourmaps245MB plus original85MBfused backing,GU/rawQKVviews shareit,activation28MB/product28MB/J38MB/normalized9MB/update9MB/exit9MB/rotary18MB/image6MB/moments2MB/source15MB,CUDAcontext/workspace/copies/hashes extra1GiB. Retained logicalfields includealiases,not unique GPU storage. CPU arraypublication max85MB raw+decoded/copy buffers;analyzer6decoded226MB/10vectors377MB/10centered377MB+1GiBtemp. No FP64learned map promotions/QR. Prior2826.8RSS/2554device is reducedproducer,not nativepeak guarantee. Generic fullmodel fit inapplicable;actualadmission/peaks authoritative.'})
    contract=one_lease_scheduling_contract(logical_branches=1,logical_operations=2,logical_checkpoint_replays=0,
        conditioner_policy='Admittedactual86text/image pluscommonpaid70clock/RoPE. Immutable70singlefunction/fulloriginalfused/output/nativecomponentgeometry.',
        extra={'purpose':'One missingdense-comparatorcorner;seal+successfulexit before allcomparison-only corners. Six-corner TN/E1/Q/V fullcentered/allrows.',
               'modelLoadScope':'4originalsingle1tensors245MB.2BF16affines+5FP32moments+imageSDPA=200321531904matrixMAC;noQR/FP64candidate/newprefix/head/image/fit/sweep. PostexitCPUscalar/dataanalysisadditional.'})
    costs=work();r={'schema':'tce-dense-corner-request-v1','ownedSources':OWNED,'sourceHashes':{p.name:sha(p) for p in payload.iterdir()},
        'scheduling_contract':contract,'model':'flux2-klein-4b-distilled','device':'cuda','backend':'torch',
        'dtype':'mixed-bfloat16-float32','batch_size':1,'seq_len':1536,'needs':{'cuda':True},'timeoutSeconds':300,
        'prospectiveCandidateMACs':costs['candidateMACs'],'prospectiveDiagnosticMACs':0,'fixedAddress':{'clock':0,'singleBlock':1},
        'protocol':'F70(xhat),immutable70singleconsumer/fullfusedgeometry/originalmaps with observinghooks only. Nativecapturedfields fingerprintedimmediatelyandatretention.9inputs27outputs;seal/exitbeforeTN/E1/Q= T-TN=E1-E0/V accounts,allfullcentered/rows. Declaredmomentcomparator,not independentfullattention or zero-neuralproducer.'}
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
    cmd+=['python','-u','dense_corner_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    note='One original70fullfused single1 onactual86input:4maps245MB,200.322BmatrixMAC,2BF16affines/5FP32moments/imageSDPA,noQR. Seal+exitbefore sixcorner TN/E1/Q/V allfullcentered1536rows;CPUscalarworkadditional. One guardedCUDAlease,4096RAM3072VRAM4CPU3GBdisk90est300cap,noretry. Stopafteroneblock;no prefix/head/image/fit/sweep;learnedmaps/state/clocks paid.'
    if len(note)>500:raise ValueError('Queue note too long')
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-dense-corner-v1',model=r['model'],task='analysis',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','device','backend','dtype','batch_size','seq_len','scheduling_contract']},
        ram_mb=4096,vram_mb=3072,cpu_threads=4,disk_gb=3,est_wall_s=90,timeout_s=300,detach=True,retry_on_kill=False,preflight=True,note=note)
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);(root/'response-result').mkdir();tick=time.monotonic()
    with subprocess.Popen([sys.executable,'-u',str(root/'dense_corner_worker.py'),str(root)],cwd=root) as producer:
        code=producer.wait()
        if code:raise subprocess.CalledProcessError(code,producer.args)
        save(root/'response-result/producer-exit.json',{'schema':'tce-dense-corner-producer-exit-v1','requestSHA256':r['requestSHA256'],
            'producerPID':producer.pid,'returncode':code,'candidateSealSHA256':sha(root/'response-result/candidate-seal.json'),
            'producerProcessSeconds':time.monotonic()-tick,'scope':'Parent observed successful producer exit before six-corner analyzer starts.'})
    subprocess.run([sys.executable,'-u',str(root/'dense_corner_analysis.py'),str(root)],check=True,cwd=root)


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
