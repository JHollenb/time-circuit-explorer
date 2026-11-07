"""Prespecified stale-key response test from two missing complementary corners."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import read, save, sha, identity
from flux_clock_study import verify
from propagated_context_study import OWNED as PREVIOUS_OWNED, WHEELS

HERE = Path(__file__).resolve().parent
OWNED = ['stale_key_context_worker.py', 'stale_key_context_analysis.py', 'stale_key_context_study.py'] + PREVIOUS_OWNED


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from mdb.job_support import one_lease_scheduling_contract
    from publish_source_native import checked
    from projected_complement import fields
    from propagated_context import RETURN_PARAMETERS, NEXT_PARAMETERS
    from stale_key_context_worker import CORNERS
    from stale_key_context_analysis import NAMES
    parent, collection, job, previous = checked(HERE.parent / 'artifacts/propagated-context-v1')
    api = Api();live = api.json('GET', '/api/jobs/' + job['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != collection['custody']:
        raise ValueError('Parent scheduler custody changed')
    p = parent / 'result'
    expected = {'report.json': '5172e93bb0e21bfb0732b216ac41b17333bb72b9636b2dabe75fdee8b81d23a5',
                'primary-seal.json': 'd6b329bad47396198545b11a02fbd89fbd1444c6a40cb039731db1c610bb0054',
                'control-seal.json': '90f1a62641b2df71663f4f2f64fcd50b403635ee32a153b627b8ad57e9fbd181'}
    if any(sha(p/n) != digest for n, digest in expected.items()):
        raise ValueError('Frozen parent result differs')
    bank, primary, control, old = [read(p/n) for n in ['banks.json', 'primary.json', 'control.json', 'inputs.json']]
    for n in ['banks', 'primary', 'control']:
        if read(p/(n+'-seal.json'))[n+'SHA256'] != sha(p/(n+'.json')):
            raise ValueError('Complete parent phase seal differs')
    byte_receipt = read(parent / 'retained-array-verification.json')
    selected = bank['selected']['fields'];banks = {role: bank['banks'][role]['fields'] for role in ['P', 'Q']}
    targets = {'diagonals': {corner: {n: {**arm['source']['fields'], **arm['nextQuery']['fields']}[n] for n in NAMES}
                              for corner, arm in [('PP', control), ('QQ', primary)]}}
    for x in fields({'selected': selected, 'banks': banks, 'targets': targets}):
        if byte_receipt['files'][x['path']]['sha256'] != x['sha256']:
            raise ValueError('Full parent byte custody absent')
    remote = job['result']['diagnostics']['cwd'] + '/response-result/'
    parent_records = {n: {'path': remote+n, 'sha256': sha(p/n)} for n in ['banks.json', 'banks-seal.json', 'primary.json', 'primary-seal.json', 'control.json', 'control-seal.json', 'request.json', 'report.json']}
    artifacts = read(p / 'parameter-artifacts.json');required = set(RETURN_PARAMETERS + NEXT_PARAMETERS)
    artifacts['artifacts'] = [x for x in artifacts['artifacts'] if x['source']['tensor'] in required]
    if {x['source']['tensor'] for x in artifacts['artifacts']} != required:
        raise ValueError('Selected original map recipes incomplete')
    s = {k: old[k] for k in ['modelRoot', 'versions', 'nativeSources', 'numericalProfile', 'imageAssets']}
    s.update(selected=selected, banks=banks, parentRecords=parent_records, parentBankSealSHA256=sha(p/'banks-seal.json'),
             parentJobID=job['job_id'], parentCustody=live['payload_custody'], parentByteReceiptSHA256=sha(parent/'retained-array-verification.json'),
             originalParameterArtifacts=artifacts['artifacts'], deniedTargets=sorted({x['path'] for x in fields(targets)}))
    root = Path(root).resolve();root.mkdir(parents=True, exist_ok=False);payload = root/'payload';payload.mkdir()
    for n in OWNED:
        if n in previous['ownedSources'] and sha(HERE/n) != previous['sourceHashes'][n]:
            raise ValueError('Frozen runtime helper changed')
        shutil.copyfile(HERE/n, payload/n)
    for n in WHEELS:
        shutil.copyfile(parent/'payload'/n, payload/n)
    shutil.copyfile(p/'retrieve_weights.py', payload/'retrieve_weights.py')
    shutil.copyfile(HERE.parent/'docs/plans/stale-key-context-response.md', payload/'frozen-plan.md')
    save(payload/'inputs.json', s);save(payload/'targets.json', targets);save(payload/'parameter-artifacts.json', artifacts)
    preview = _plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda')
    save(payload/'resource-basis.json', {'intentFirstPreview': preview, 'measuredParent': live['result'],
         'geometry': {'perActorBF16ParameterBytes': 207618304, 'perActorComplementKVBytes': 18874368, 'newBankProjections': 0},
         'exception': 'Two sequential selected actors, not a full-model load. Parent1752.8MBRSS586MBdevice; each actor279478784Ballocated/285212672Breserved CUDA. Use4096RAM1024VRAMCPU4disk3GB,120sexpected600cap. No bankproducer/cache/fullmodel/suffix/image.'})
    contract = one_lease_scheduling_contract(logical_branches=2, logical_operations=10, logical_checkpoint_replays=0,
        conditioner_policy='Paid native source states, sealed fresh parent K/V, identical shared selected P operands, original maps and post-seal diagonal targets.',
        extra={'modelLoadScope':'Five original tensors per actor; no bank/fullmodel formation.',
               'targetIsolation':'Both diagonals denied until both new actors seal/exit; diagnostic denies primary arrays. Primary also denies Q-key bank and P-value bank.'})
    r = {'schema':'tce-stale-key-context-request-v1','ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in payload.iterdir()},
         'scheduling_contract':contract,'model':'flux2-klein-4b-distilled','backend':'diffusers','device':'cuda','dtype':'bfloat16',
         'batch_size':1,'seq_len':1536,'height':512,'width':512,'needs':{'cuda':True},'timeoutSeconds':600,
         'case':previous['case'],'support':previous['support'],'logicalStep':2,'sourceJoint':1,'nextJoint':2,
         'corners':{n:list(v) for n,v in CORNERS.items()},'arithmeticSubplans':{'primary':'K_P/V_Q-original-ownMLP-ownnextQ','diagnostic':'K_Q/V_P-same-finite-geometry'},
         'protocol':'Prespecified stale-key primaryPQ anddiagnosticQP; twoownMLP/nextqueryactors905,969,664MAC. Bothsealexit beforePP/QQtargets; score actualresponses vsQQ-PP andfourcornerinteraction. No newbanks/fit/reference/suffix/image.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(payload/'request.json',r)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in payload.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same handle')
    cmd=command()[:-5]
    for n in WHEELS:cmd+=['--with','./'+n]
    cmd+=['python','-u','stale_key_context_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-stale-key-context-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs=r['needs'],
        mrun_config={k:r[k] for k in ['schema','requestSHA256','model','backend','device','dtype','batch_size','seq_len','height','width','scheduling_contract','arithmeticSubplans']},
        ram_mb=4096,vram_mb=1024,cpu_threads=4,disk_gb=3,est_wall_s=120,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,
        note='Stale-key response spherecube2027 step2 joint1:primaryKP/VQ thenpostsealdiagnosticKQ/VP;sharedPselected. Two4rowattention/ownMLP/ownjoint2queryactors8affines2SDPA905.970MMAC. BothsealexitbeforeoldPP/QQoutputs. Score response vsQQ-PP,zero,perheadandinteraction. No bankformation/fit/fullreference/suffix/VAE/image. Paidbanks/baselines/maps. Parent1753RSS586device;each279MBallocated. CPU4RAM4096VRAM1024disk3GB,120sexpected600cap,noretry.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    root=Path(root).resolve();r,s=read(root/'request.json'),read(root/'inputs.json');verify(root,r)
    for item in s['parentRecords'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Parent metadata changed')
    started=time.perf_counter();phases=[]
    for phase in ['primary','diagnostic']:
        tick=time.perf_counter();subprocess.run([sys.executable,'-u',str(root/'stale_key_context_worker.py'),phase,str(root)],cwd=root,check=True)
        phases.append({'phase':phase,'wallSecondsIncludingFreshProcess':time.perf_counter()-tick});print('STALE_KEY_CONTEXT_'+phase.upper()+'_EXITED',flush=True)
    from stale_key_context_analysis import execute as analyze
    report=analyze(root);out=root/'response-result'
    for n in ['request.json','inputs.json','targets.json','parameter-artifacts.json','retrieve_weights.py','resource-basis.json','frozen-plan.md']:
        shutil.copyfile(root/n,out/n)
    save(out/'report.json',{'schema':'tce-stale-key-context-result-v1','requestSHA256':r['requestSHA256'],**report,'phases':phases,
         'workerWallSeconds':time.perf_counter()-started,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
         'limits':['One known fixed-query source pair. Parentbanks/diagonals/nativeprefixes and originalmaps remain paid.',
                   'No freshQvalue producer or avoided-key serving measurement; two new actors reuse sealedbanks.',
                   'No rawprompt completion,full ownstate diffusion,VAE/image or arbitrary-prompt zero-neural capability.']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','execute','collect']);parser.add_argument('run');args=parser.parse_args()
    if args.action=='collect':
        from response_study import collect
        value=collect(args.run)
    else:value=globals()[args.action](args.run)
    if value is not None:print(json.dumps(value))
