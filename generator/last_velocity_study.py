"""Own-prefix last-velocity assay; candidates seal before saved targets."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import np, read, save, sha, identity
from flux_clock_study import verify

HERE = Path(__file__).resolve().parent
OWNED = ['last_velocity_study.py','last_velocity.py','last_velocity_decode.py',
    'encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py',
    'complete_masked_study.py','masked_kv_study.py','retained_queries.py']
POLICIES = ['secant','hold']


def stage(root):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    parent = HERE.parent / 'artifacts/retained-query-v1'
    c,j,old = checked_collection(parent, current_workers=False)
    live = Api().json('GET','/api/jobs/'+j['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != c['custody']:
        raise ValueError('Own-prefix parent is not authoritative')
    f = parent / 'result'; s = read(f/'inputs.json'); im = read(f/'images-candidate.json')
    root = Path(root).resolve(); root.mkdir(exist_ok=False, parents=True); p = root/'payload'; p.mkdir()
    for n in OWNED:
        if n in old['ownedSources'] and sha(HERE/n) != old['sourceHashes'][n]:
            raise ValueError('Qualified shared source differs')
        shutil.copyfile(HERE/n,p/n)
    shutil.copyfile(HERE.parent/'docs/plans/last-velocity-forecast.md',p/'frozen-plan.md')
    prefix = {case: {n: im['fields'][case][n] for n in ['latent3','velocity1','velocity2']} for case in ['00','10']}
    allowed = {v['path'] for fs in prefix.values() for v in fs.values()}
    ids = read(f/'noise-seal.json')['ids']; allowed.add(ids['path'])
    forbidden = s['forbiddenTargets'] + [v['path'] for fs in im['fields'].values() for v in fs.values()] + [v['path'] for v in im['images'].values()]
    targets = {'own': {'fields': im['fields'], 'images': im['images']},
        'full': read(f/'targets.json')['fullImages'],
        'sourceFiles': {n: {'path':j['result']['diagnostics']['cwd']+'/'+n, 'sha256':sha(f/n)} for n in ['images-candidate.json','candidate-seal.json']}}
    source = {'parentRoot': j['result']['diagnostics']['cwd'], 'parentJobID':j['job_id'],
        'parentRequestSHA256':old['requestSHA256'],'parentReportSHA256':sha(f/'report.json'),
        'parentSealSHA256':sha(f/'candidate-seal.json'),'prefix':prefix,'ids':ids,
        'modelRoot':s['modelRoot'],'versions':s['versions'],'nativeSources':s['nativeSources'],
        'numericalProfile':s['numericalProfile'],'schedule':s['schedule'],
        'decoderAssets':{n:h for n,h in s['imageAssets'].items() if n.startswith('vae/') or n=='scheduler/scheduler_config.json'},
        'forbiddenTargets':sorted(set(forbidden)-allowed)}
    save(p/'inputs.json',source); save(p/'targets.json',targets)
    save(p/'resource-basis.json',{'policy':'Intent-first VAE-only workload, automatic RAM/VRAM sizing, CUDA required; no explicit memory reservation.',
        'component':'Native VAE only, four serial B1/512 decodes; no transformer/encoder construction or weights.',
        'historyIdentity':'tce-last-velocity-v1 / vae-decode; source FLUX checkpoint identified by decoder asset hashes.',
        'placement':'Own-prefix arrays and qualified VAE store are host-local on Beast; single guarded finite lease, no retries.'})
    r = {'schema':'tce-last-velocity-request-v1','specimen':'flux2-klein-4b-distilled','component':'vae-only',
        'policies':POLICIES,'prompts':old['prompts'],'seed':old['seed'],'parentRequestSHA256':old['requestSHA256'],
        'ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},
        'backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'seq_len':1024,'height':512,'width':512,
        'task_family':'vae-decode','timeoutSeconds':300,
        'protocol':'Framework-free FP32secant/hold forecasts from authenticated own Z3/v1/v2. Four native finite commits/VAE images seal before target release. Two postseal scheduler-only exact parent replays. No new encoder/denoiser/reference/noise/fitting.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from run import command
    from mrun.client.submit import launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect the existing job')
    cmd=command()[:-5]+['python','-u','last_velocity_study.py','execute','.']
    i=cmd.index('uv');cmd[i:i]=['OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=launch(cmd,experiment='tce-last-velocity-v1',task='vae-decode',payload=root/'payload',pin='beast',needs={'cuda':True},
        config={k:r[k] for k in ['schema','requestSHA256','specimen','component','backend','device','dtype','batch_size','seq_len','height','width','task_family']},
        cpu_threads=4,disk_gb=1,est_wall_s=90,timeout_s=300,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='VAE-only:4serial B1/512 decodes of fixed own-prefix lastvelocity secant vs hold. No encoder/denoiser weights or calls. Framework-free forecaster child, guarded finite scheduler/VAE child; all4images seal before savedtargets and2scheduler-only exact parent replays. Six historical prefix denoisers remain charged. Automatic RAM/VRAM sizing; host-local inputs Beast;90sexpected300scap,noretries.')
    result={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',result);return result


def seal(root):
    from paired_fields import load
    r=read(root/'request.json');s=read(root/'inputs.json');f=read(root/'forecast-seal.json');im=read(root/'images.json')
    keys={policy+'-'+case for policy in r['policies'] for case in s['prefix']}
    if set(f['fields'])!=keys or set(im['fields'])!=keys or set(im['images'])!=keys:
        raise ValueError('Required candidate coverage differs')
    if im['forecastSealSHA256']!=sha(root/'forecast-seal.json') or im['requestSHA256']!=r['requestSHA256']:
        raise ValueError('Actual forecast consumer differs')
    for key in keys:
        if set(f['fields'][key])!={'forecastFP32','velocity3'} or set(im['fields'][key])!={'latent4','vaeInput','decodedRaw','rgbFloat','pngRGB'}:
            raise ValueError('Required candidate fields differ')
        for fs in [f['fields'][key],im['fields'][key]]:
            for item in fs.values():load(item)
        actual=next(a for a in im['actual'] if a['policy']+'-'+a['case']==key)
        case=key.split('-')[1]
        if actual['Z3PayloadSHA256']!=s['prefix'][case]['latent3']['payloadSHA256'] or actual['velocityPayloadSHA256']!=f['fields'][key]['velocity3']['payloadSHA256']:
            raise ValueError('Actual scheduler input identity differs')
        if sha(im['images'][key]['path'])!=im['images'][key]['sha256']:
            raise ValueError('Candidate image differs')
    save(root/'candidate-seal.json',{'schema':'tce-last-velocity-candidate-seal-v1','requestSHA256':r['requestSHA256'],
        'forecastSealSHA256':sha(root/'forecast-seal.json'),'imagesSHA256':sha(root/'images.json'),'fields':im['fields'],'images':im['images']})


def score(root,out,phases,started):
    import json
    import torch
    import diffusers
    from diffusers import FlowMatchEulerDiscreteScheduler
    from last_velocity_decode import scheduler,checked_assets
    from paired_fields import load,payload,metrics,paired,retain
    from complete_masked_study import profile
    from retained_queries import energy_account
    r=read(root/'request.json');s=read(root/'inputs.json');f=read(root/'forecast-seal.json');im=read(root/'images.json')
    seal_sha=sha(root/'candidate-seal.json');targets=read(root/'targets.json')
    profile(torch,4);checked_assets(s,diffusers)
    for item in targets['sourceFiles'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Parent source metadata changed')
    replays=[]
    with torch.inference_mode():
        for case in s['prefix']:
            sch=scheduler(s,torch,FlowMatchEulerDiscreteScheduler)
            z=load(s['prefix'][case]['latent3'],'cuda');v=load(targets['own']['fields'][case]['velocity3'],'cuda')
            replay=sch.step(v,sch.timesteps[3],z,return_dict=False)[0]
            if payload(replay)!=targets['own']['fields'][case]['latent4']['payloadSHA256']:
                raise ValueError('Pinned finite scheduler does not reproduce parent exactly')
            replays.append({'case':case,'replayedZ4PayloadSHA256':payload(replay),'bitExact':True})
    accounts=[];responses=[];pixel_terms=[];finite_terms=[];diagnostic={};observations=0
    values={};gap=float(s['schedule']['sigmas'][4]-s['schedule']['sigmas'][3])
    with (out/'observations.jsonl').open('x') as stream:
        for policy in r['policies']:
            values[policy]={}
            for case in s['prefix']:
                key=policy+'-'+case;fs={**f['fields'][key],**im['fields'][key]};values[policy][case]={}
                for port in ['velocity3','latent4','vaeInput','decodedRaw','rgbFloat','pngRGB']:
                    a=load(fs[port]);values[policy][case][port]=a
                    for ref in ['own','full']:
                        if ref=='full' and port not in ['latent4','pngRGB']:continue
                        b=load(targets[ref]['fields'][case][port]);accounts.append({'policy':policy,'case':case,'port':port,'reference':ref,**metrics(a,b)})
                        if ref=='own' and port in ['velocity3','latent4','pngRGB']:
                            aa=a.reshape(512 if port=='pngRGB' else a.shape[1],-1);bb=b.reshape(aa.shape)
                            for j,(ar,br) in enumerate(zip(aa,bb)):
                                stream.write(json.dumps({'policy':policy,'case':case,'port':port,'position':j,'reference':ref,**metrics(ar,br)},allow_nan=False)+'\n');observations+=1
                own=targets['own']['fields'][case];vhat=values[policy][case]['velocity3'];v3=load(own['velocity3'])
                actual=values[policy][case]['latent4']-load(own['latent4']);smooth=gap*(vhat-v3);finite=actual-smooth
                diagnostic[key]={n:retain(root/'arrays/diagnostic'/key,n,x) for n,x in [('smoothLastStepDefect',smooth),('finiteCommitRemainder',finite),('actualLastStepDefect',actual)]}
                finite_terms.append({'policy':policy,'case':case,'firstTerm':'smoothLastStepDefect','secondTerm':'finiteCommitRemainder',**energy_account(smooth,finite,actual)})
                a=values[policy][case]['pngRGB'];b=load(own['pngRGB']);c=load(targets['full']['fields'][case]['pngRGB'])
                pixel_terms.append({'policy':policy,'case':case,'firstTerm':'newLastCallReplacementError','secondTerm':'inheritedEncoderImageError',**energy_account(a-b,b-c,a-c)})
            for port in ['velocity3','latent4','pngRGB']:
                a,b=[values[policy][case][port] for case in ['00','10']]
                for ref in ['own','full']:
                    if port=='velocity3' and ref=='full':continue
                    c,d=[load(targets[ref]['fields'][case][port]) for case in ['00','10']]
                    responses.append({'policy':policy,'port':port,'reference':ref,**paired(a,b,c,d)})
    if sha(root/'candidate-seal.json')!=seal_sha:raise ValueError('Candidate seal changed')
    for key,item in im['images'].items():shutil.copyfile(item['path'],out/(key+'.png'))
    for ref in ['own','full']:
        for case,item in targets[ref]['images'].items():
            if sha(item['path'])!=item['sha256']:raise ValueError('Original reference PNG changed')
            shutil.copyfile(item['path'],out/(ref+'-'+case+'.png'))
    for n in ['request.json','inputs.json','targets.json','resource-basis.json','frozen-plan.md','forecast-seal.json','images.json','candidate-seal.json']:
        shutil.copyfile(root/n,out/n)
    save(out/'diagnostic-fields.json',diagnostic)
    save(out/'report.json',{'schema':'tce-last-velocity-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal_sha,
        'accounts':accounts,'pairedColorResponse':responses,'finiteCommitDecomposition':finite_terms,'pixelErrorInteraction':pixel_terms,
        'nativeSchedulerReplays':replays,'observations':observations,'phases':phases,'workerSecondsAfterImports':time.perf_counter()-started,
        'work':{'newEncoderEvaluations':0,'newDenoiserEvaluations':0,'candidateFiniteSchedulerSteps':4,'postsealNativeSchedulerReplays':2,'nativeVAEDecodes':4,'newImages':4,'newReferenceImages':0,'newNoiseDraws':0,
            'VAEConvLinearCalls':len(im['operations']),'VAENominalConvLinearMACs':sum(x['nominalMACs'] for x in im['operations']),
            'chargedHistoricalPrefixDenoisers':6,'chargedHistoricalPrefixTransformerLinears':618,'chargedHistoricalPrefixNominalAffineMACs':28343562928128,
            'prospectiveDenoiserCallsAvoidedPair':2,'prospectiveTransformerLinearsAvoidedPair':206,'prospectiveNominalAffineMACsAvoidedPair':9447854309376},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limitations':['Known own-prefix replay, not fresh arbitrary-prompt execution or zero-neural whole generation.','Three original denoisers per image and original encoder remain paid prefix inputs; four VAE decodes are new learned work.','Fixed demanding extrapolation versus hold, no fitting or sweep. Prospective25%denoiser reduction is not measured speedup.']})


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);started=time.perf_counter();out=root/'response-result';out.mkdir();phases=[]
    for action in ['forecast','decode']:
        tick=time.perf_counter();subprocess.run([sys.executable,'-u',str(root/'last_velocity_study.py'),action,str(root)],check=True,timeout=240)
        phases.append({'action':action,'wallSecondsIncludingFreshProcess':time.perf_counter()-tick})
    seal(root);print('LAST_VELOCITY_ALL_IMAGES_SEALED',sha(root/'candidate-seal.json'),flush=True)
    score(root,out,phases,started);print('LAST_VELOCITY_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','forecast','decode','collect']);p.add_argument('run');a=p.parse_args()
    if a.action in ['forecast','decode']:v=__import__('last_velocity' if a.action=='forecast' else 'last_velocity_decode').execute(a.run)
    elif a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(__import__('json').dumps(v))
