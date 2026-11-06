"""Fresh raw-prompt factorial; candidate own trajectories seal before references."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import np, read, save, sha, identity
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
CASES=['00','10','01','11']
PROMPTS=[{'id':c,'prompt':f'A studio photograph of a {pitcher} ceramic pitcher to the {side} of a {can} metal watering can on a white table, soft lighting.'}
    for c,pitcher,can,side in [('00','orange','teal','left'),('10','teal','orange','left'),('01','orange','teal','right'),('11','teal','orange','right')]]
ENCODER_PORTS=['H'+str(i) for i in range(28)]+['E']
IMAGE_PORTS=['C'+str(i) for i in range(4)]+['velocity'+str(i) for i in range(4)]+['latent'+str(i) for i in range(1,5)]+['vaeInput','decodedRaw','rgbFloat','pngRGB']
OWNED=['fresh_square_study.py','fresh_square_encoder.py','fresh_square_images.py','uncentered_reader.py','uncentered_reader_study.py',
    'encoder_atoms.py','paired_fields.py','masked_kv_study.py','conditioning_demand.py','complete_masked_study.py','flux_clock.py','flux_clock_study.py']


def inherited_clock(root,source):
    p=Path(root)/'clock-table.json'
    if sha(p)!=source['clockTableSHA256']:raise ValueError('Clock bytes changed')
    table=read(p)
    if table['requestSHA256']!=source['clockParentRequestSHA256'] or table['contract']['checkpointSHA256']!=source['imageAssets']['transformer/diffusion_pytorch_model.safetensors']:raise ValueError('Clock identity differs')
    return table


def reference_paths(root):
    root=Path(root)
    return [root/n for n in ['encoder-full.json','images-full.json','full-seal.json']]+[root/('full-'+c+'.png') for c in CASES]+[root/'arrays/full'/c/(n+'.npy') for c in CASES for n in ENCODER_PORTS+IMAGE_PORTS+['fullHead']]


def candidate_guard(root,lane,phase):
    if lane=='full':
        root=Path(root)
        if not (root/'candidate-seal.json').is_file():raise ValueError('Candidate trajectories must seal before references')
        s=read(root/'candidate-seal.json');r=read(root/'request.json')
        if s['lane']!='candidate' or s['requestSHA256']!=r['requestSHA256']:raise ValueError('Candidate seal belongs to another request')
        for key,name in [('readerSHA256','reader-seal.json'),('encoderSHA256','encoder-candidate.json'),('imagesSHA256','images-candidate.json'),('noiseSealSHA256','noise-seal.json'),('clockTableSHA256','clock-table.json')]:
            if s[key]!=sha(root/name):raise ValueError('Sealed candidate record changed: '+name)
        return
    from flux_clock import deny_targets
    paths=reference_paths(root);deny_targets(paths)
    try:open(paths[0],'rb')
    except PermissionError:pass
    else:raise RuntimeError('Reference exclusion inactive')
    save(Path(root)/('guard-'+phase+'.json'),{'deniedReferencePaths':[str(p) for p in paths],'blockedOpenCanary':True})


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from publish_context_statistics import checked_collection
    parent=HERE.parent/'artifacts/uncentered-reader-v1';c,j,prior=checked_collection(parent,current_workers=False);api=Api();live=api.json('GET','/api/jobs/'+j['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody']:raise ValueError('Qualified frozen reader parent differs')
    target=read(parent/'result/targets.json');source=target['parentSource'];previous=read(parent/'result/inputs.json')
    actual=HERE.parent/'artifacts/selected-reader-image-v1';ac,aj,ar=checked_collection(actual,current_workers=False)
    al=api.json('GET','/api/jobs/'+aj['job_id'])
    if al['state']!='succeeded' or al['payload_custody']!=ac['custody']:raise ValueError('Qualified consumer parent differs')
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    for name in OWNED:
        if name in prior['sourceHashes'] and sha(HERE/name)!=prior['sourceHashes'][name]:raise ValueError('Frozen reader/shared source changed: '+name)
        shutil.copyfile(HERE/name,p/name)
    for name in ['flux_functional_qwen_prefix.py','qwen_h18_state_history.py']:
        path=actual/'payload'/name
        if sha(path)!=ar['sourceHashes'][name]:raise ValueError('Qualified graph/history changed')
        shutil.copyfile(path,p/name)
    shutil.copyfile(actual/'result/clock-table.json',p/'clock-table.json');shutil.copyfile(HERE.parent/'docs/plans/fresh-prompt-square.md',p/'frozen-plan.md')
    keep=['lexicalSource','qwenShards','encoderNumericalProfile','encoderVersions','sdpaSourceSHA256','modelRoot','imageAssets','nativeSources',
        'versions','numericalProfile','attentionProcessors','schedule','clockTableSHA256','clockParentRequestSHA256','clockParentRoot']
    inputs={k:source[k] for k in keep};inputs.update(lexicalStore=previous['lexicalStore'],readerGeometry=previous['readerGeometry'],excludedSources=previous['excludedSources'],
        forbiddenTargets=list(previous['forbiddenTargets']),
        readerParentRoot=j['result']['diagnostics']['cwd'],readerParentRequestSHA256=prior['requestSHA256'],consumerParentRoot=aj['result']['diagnostics']['cwd'],consumerParentRequestSHA256=ar['requestSHA256'])
    # No prior contextual fields, conditioning or image is a serving operand.
    old_reader=read(parent/'result/reader-seal.json')
    inputs['forbiddenTargets'] += [x['path'] for x in [*old_reader['fields'].values(),*old_reader['h0Fields'].values(),*old_reader['packets'],*old_reader['states'],*old_reader['lexicalRows']]]
    for g in [target['encoder']['fields'],target['images']['fields']]:
        inputs['forbiddenTargets'] += [x['path'] for fs in g.values() for x in fs.values()]
    inputs['forbiddenTargets'] += [x['path'] for x in target['images']['images'].values()]
    save(p/'inputs.json',inputs)
    preview=_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda')
    save(p/'resource-basis.json',{'intentFirstPreview':preview,'measuredConsumerJobID':aj['job_id'],'measuredResult':ac['result'],
        'exception':'Generic fallback/B64 sizing does not describe serialB1/512. Reuse measured14336MBRAM/13000MBVRAM envelope. Four cases increase retained fields, not batch. CPU1encoder/4image, separate children, no model co-residency. Expected600s,1200scap, no retries.'})
    request={'schema':'tce-fresh-prompt-square-request-v1','prompts':PROMPTS,'seed':271828,'manifestSHA256':prior['manifestSHA256'],'readerPolicy':prior['readerPolicy'],
        'ownedSources':OWNED,'frozenReaderSHA256':prior['sourceHashes']['uncentered_reader.py'],'parentRequestSHA256':prior['requestSHA256'],
        'sourceHashes':{f.name:sha(f) for f in p.iterdir()},'model':'flux2-klein-4b-distilled','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'steps':4,'timeoutSeconds':1200,
        'scope':'Fresh fixed compositional coverage. Four own top4 candidates seal before four full-attention references. Both use original learned encoder suffix/denoiser/VAE and qualified compiled clocks; no zero-neural or arbitrary-prompt certificate.',
        'protocol':'One fresh native noise draw at seed271828, shared exactly. No fitting, support/precision sweep, length matching, prompt replacement, hidden-state refresh or target transplantation. Preserve endpoints/four edges/mixed interaction, active/padding and semantic countertrends.'}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from run import command
    from mrun.client.submit import launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted')
    cmd=command()[:-5]+['python','-u','fresh_square_study.py','execute','.'];i=cmd.index('uv');cmd[i:i]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=launch(cmd,experiment='tce-fresh-prompt-square-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs={'cuda':True},
        config={k:r[k] for k in ['schema','requestSHA256','model','backend','device','dtype','batch_size','seq_len','height','width']},
        ram_mb=14336,vram_mb=13000,cpu_threads=4,disk_gb=3,est_wall_s=600,timeout_s=1200,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='Fresh2x2 orange/teal pitcher/wateringcan left/right,seed271828. Frozen plain top4/own4encoder+images seal before4nativefullattention refs.212nativeencoderblocks/1500linears,32denoisers/8VAE paid.1newsharednoise,compiledclocks. SerialB1/512 separatechildren CPU1encoder/4image. No fit/sweep/refresh/replacement.14336RAM13000VRAM measured0021bd0429e3 exception vsbadgenericB64.600sexpected1200scap,noretries.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def noise(root):
    import torch,diffusers
    from diffusers import Flux2KleinPipeline
    from complete_masked_study import profile
    from paired_fields import retain,payload
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);source=read(root/'inputs.json');numerical=profile(torch,4)
    if numerical!=source['numericalProfile'] or {'torch':torch.__version__,'diffusers':diffusers.__version__}!=source['versions']:raise ValueError('Noise numerical/framework contract differs')
    for name,digest in source['nativeSources'].items():
        if sha(Path(diffusers.__file__).parent/name)!=digest:raise ValueError('Noise implementation differs')
    pipe=Flux2KleinPipeline(scheduler=None,vae=None,text_encoder=None,tokenizer=None,transformer=None,is_distilled=True)
    with torch.inference_mode():
        packed,ids=pipe.prepare_latents(1,32,512,512,torch.bfloat16,torch.device('cuda'),torch.Generator(device='cpu').manual_seed(r['seed']))
        raw=pipe._unpack_latents_with_ids(packed,ids,32,32)
        if payload(pipe._pack_latents(raw))!=payload(packed):raise ValueError('Native noise unpack/repack does not close')
        save(root/'noise-seal.json',{'schema':'tce-fresh-square-noise-v1','requestSHA256':r['requestSHA256'],'seed':r['seed'],'newNoiseDraws':1,
            'raw':retain(root/'arrays/shared','rawNoise',raw),'packed':retain(root/'arrays/shared','packedNoise',packed),'ids':retain(root/'arrays/shared','imageIDs',ids),'numericalProfile':numerical})


def seal(root,lane):
    from paired_fields import load
    enc=read(root/('encoder-'+lane+'.json'));im=read(root/('images-'+lane+'.json'));reader=read(root/'reader-seal.json')
    if set(enc['fields'])!=set(CASES) or set(im['fields'])!=set(CASES):raise ValueError('Square corners incomplete')
    for c in CASES:
        if set(enc['fields'][c])!=set(ENCODER_PORTS) or set(im['fields'][c])!=set(IMAGE_PORTS):raise ValueError('Full consumer field coverage differs')
        for fs in [enc['fields'][c],im['fields'][c]]:
            for f in fs.values():load(f)
        if im['actualEFields'][c]!=enc['fields'][c]['E'] or im['schedules'][c]!=read(root/'inputs.json')['schedule']:raise ValueError('Actual conditioning/schedule join differs')
        if sha(im['images'][c]['path'])!=im['images'][c]['sha256']:raise ValueError('Original image changed')
    if enc['readerSealSHA256']!=sha(root/'reader-seal.json') or enc['lexicalSealSHA256']!=reader['lexicalSealSHA256'] or im['encoderSHA256']!=sha(root/('encoder-'+lane+'.json')) or im['tableSHA256']!=sha(root/'clock-table.json') or im['noiseSealSHA256']!=sha(root/'noise-seal.json'):raise ValueError('Consumer source chain differs')
    if lane=='full' and (enc['referenceReleasedAfterSealSHA256']!=sha(root/'candidate-seal.json') or im['referenceReleasedAfterSealSHA256']!=sha(root/'candidate-seal.json')):raise ValueError('References released under another candidate seal')
    for f in [*reader['fields'].values(),*reader['h0Fields'].values()]:load(f)
    for f in reader['packets']+reader['states']+reader['lexicalRows']:
        if sha(f['path'])!=f['sha256']:raise ValueError('Candidate packet changed')
    value={'schema':'tce-fresh-square-seal-v1','lane':lane,'requestSHA256':read(root/'request.json')['requestSHA256'],
        'readerSHA256':sha(root/'reader-seal.json'),'encoderSHA256':sha(root/('encoder-'+lane+'.json')),'imagesSHA256':sha(root/('images-'+lane+'.json')),
        'noiseSealSHA256':sha(root/'noise-seal.json'),'clockTableSHA256':sha(root/'clock-table.json'),'encoderFields':enc['fields'],'imageFields':im['fields'],'images':im['images']}
    save(root/(lane+'-seal.json'),value)


def factorial(fields):
    f={c:np.asarray(fields[c],dtype=np.float64) for c in CASES}
    if len({x.shape for x in f.values()})!=1:raise ValueError('Factorial geometry differs')
    return {'color-left':f['10']-f['00'],'color-right':f['11']-f['01'],'relation-base':f['01']-f['00'],'relation-swapped':f['11']-f['10'],
        'mixed':f['11']-f['10']-f['01']+f['00']}


def score(root,out,phases,started):
    from paired_fields import load,metrics,retain
    enc={l:read(root/('encoder-'+l+'.json')) for l in ['candidate','full']};images={l:read(root/('images-'+l+'.json')) for l in enc}
    masks={}
    for row in read(root/'reader-seal.json')['lexicalRows']:
        with np.load(row['path'],allow_pickle=False) as f:masks[row['id']]=f['mask'].astype(bool)
    accounts=[];responses=[];diffs={};observations=0;seal_sha=sha(root/'candidate-seal.json')
    with (out/'observations.jsonl').open('x') as stream:
        for port in ENCODER_PORTS[1:]+IMAGE_PORTS:
            is_encoder=port in ENCODER_PORTS;group=enc if is_encoder else images
            values={l:{c:load(group[l]['fields'][c][port]) for c in CASES} for l in group}
            for c in CASES:
                a,b=values['candidate'][c],values['full'][c];accounts.append({'case':c,'port':port,'region':'all',**metrics(a,b)})
                if is_encoder:
                    for region,use in [('active',masks[c]),('padding',~masks[c])]:accounts.append({'case':c,'port':port,'region':region,**metrics(a.reshape(512,-1)[use],b.reshape(512,-1)[use])})
                if is_encoder or port in ['rgbFloat','pngRGB']:
                    for i,(ar,br) in enumerate(zip(a.reshape(512,-1),b.reshape(512,-1))):
                        stream.write(json.dumps({'kind':'endpoint','case':c,'port':port,'position':i,'attended':bool(masks[c][i]) if is_encoder else None,**metrics(ar,br)},allow_nan=False)+'\n');observations+=1
            transforms={l:factorial(v) for l,v in values.items()}
            for kind in transforms['candidate']:
                a,b=transforms['candidate'][kind],transforms['full'][kind];row={'kind':kind,'port':port,'region':'all',**metrics(a,b)};row['nativeResponseExactlyZero']=row['referenceRms']==0;responses.append(row)
                if is_encoder:
                    stack=np.stack(list(masks.values()));regions={'active-all-corners':stack.all(0),'padding-all-corners':~stack.any(0),'mixed-attendance':stack.any(0)&~stack.all(0)}
                    for region,use in regions.items():
                        if use.any():responses.append({'kind':kind,'port':port,'region':region,**metrics(a.reshape(512,-1)[use],b.reshape(512,-1)[use])})
                if port in ['E','latent4','rgbFloat','pngRGB']:
                    for label,v in [('candidate',a),('full',b),('error',a-b)]:diffs[kind+'.'+port+'.'+label]=retain(root/'arrays/responses',kind+'.'+port+'.'+label,v)
    if sha(root/'candidate-seal.json')!=seal_sha:raise ValueError('Sealed candidates changed')
    for lane,im in images.items():
        for c,f in im['images'].items():shutil.copyfile(f['path'],out/(lane+'-'+c+'.png'))
    for name in ['request.json','inputs.json','resource-basis.json','frozen-plan.md','clock-table.json','noise-seal.json','reader-seal.json','encoder-candidate.json','images-candidate.json','encoder-full.json','images-full.json','candidate-seal.json','full-seal.json','guard-reader.json','guard-encoder.json','guard-images.json']:shutil.copyfile(root/name,out/name)
    save(out/'responses.json',{'fields':diffs})
    calls=[x for e in enc.values() for x in e['primitives']];im_calls=[x for im in images.values() for x in im['linears']]
    save(out/'report.json',{'schema':'tce-fresh-square-result-v1','requestSHA256':read(root/'request.json')['requestSHA256'],'candidateSealSHA256':seal_sha,'fullSealSHA256':sha(root/'full-seal.json'),
        'accounts':accounts,'factorialResponses':responses,'observations':observations,'phases':phases,'workerSecondsAfterImports':time.perf_counter()-started,
        'work':{'newImages':8,'newCandidateImages':4,'newReferenceImages':4,'newNoiseDraws':1,'nativeEncoderBlocks':sum(e['nativeComponentBlocks'] for e in enc.values()),
            'encoderSDPA':sum(x['kind']=='SDPA' for x in calls),'encoderLinears':sum(x['kind']=='linear' for x in calls),'encoderNominalLinearMACs':sum(x['nominalMACs'] for x in calls if x['kind']=='linear'),
            'nativeDenoiserCalls':sum(x['denoiser'] for im in images.values() for x in im['counts'].values()),'nativeVAEDecodes':sum(x['VAE'] for im in images.values() for x in im['counts'].values()),
            'nativeTransformerLinears':sum(x['parameter'].startswith('transformer.') for x in im_calls),'nativeTransformerNominalLinearMACs':sum(x['nominalMACs'] for x in im_calls if x['parameter'].startswith('transformer.')),
            'onlineClockGathers':sum(len(im['clockCalls']) for im in images.values()),'onlineClockAffines':0,'firstReaderCalls':4},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limitations':['One fixed2x2square/oneseed; four dependent corners are not arbitrary-prompt coverage. Preserve native semantic failures and evaluator disagreement.',
            'Both lanes retain full learned contextual encoder/denoiser/VAE. Native full-attention reference begins from exact compiled lexical H0 and uses previously qualified compiled clocks. No zero-neural or matched-speedup claim.',
            'Source-only reader was frozen before fresh endpoints; old zero/uniform controls remain development evidence, not rerun controls on this square.']})


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);started=time.perf_counter();out=root/'response-result';out.mkdir();phases=[]
    for action,lane in [('noise',None),('reader',None),('encoder','candidate'),('images','candidate'),('encoder','full'),('images','full')]:
        if (action,lane)==('encoder','full'):seal(root,'candidate');print('FRESH_SQUARE_CANDIDATES_SEALED',sha(root/'candidate-seal.json'),flush=True)
        tick=time.perf_counter();cmd=[sys.executable,'-u',str(root/'fresh_square_study.py'),action,str(root)]+(['--lane',lane] if lane else [])
        subprocess.run(cmd,check=True,timeout=900);phases.append({'action':action,'lane':lane,'wallSecondsIncludingFreshProcess':time.perf_counter()-tick})
    seal(root,'full');score(root,out,phases,started);print('FRESH_SQUARE_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','noise','reader','encoder','images','collect']);p.add_argument('run');p.add_argument('--lane',choices=['candidate','full']);a=p.parse_args()
    if a.action=='reader':
        candidate_guard(Path(a.run).resolve(),'candidate','reader');import uncentered_reader;v=uncentered_reader.execute(a.run)
    elif a.action in ['encoder','images']:v=__import__('fresh_square_'+a.action).execute(a.run,a.lane)
    elif a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(json.dumps(v))
