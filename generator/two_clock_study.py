"""Frozen two-clock omission/recovery factorial on an authenticated own prefix."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import np,read,save,sha,identity
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['two_clock_study.py','two_clock_images.py','two_clock.py','last_velocity_decode.py',
    'encoder_atoms.py','paired_fields.py','flux_clock.py','flux_clock_study.py',
    'complete_masked_study.py','masked_kv_study.py','retained_queries.py']
PORTS=['latent3','velocity3','latent4','vaeInput','decodedRaw','rgbFloat','pngRGB']


def qualify(nn,enc,forecast,nh,own_source):
    joins=[]
    for c in ['00','10']:
        held=forecast['fields']['hold-'+c]['velocity3']
        expected={n:nn['fields'][c][n] for n in ['latent3','velocity1','velocity2']}
        if forecast['prefix'][c]!=expected or held['payloadSHA256']!=expected['velocity2']['payloadSHA256']:
            raise ValueError('Historical hold does not bind the same own prefix')
        actual=next(x for x in nh['actual'] if x['policy']=='hold' and x['case']==c)
        if actual['Z3PayloadSHA256']!=expected['latent3']['payloadSHA256'] or actual['velocityPayloadSHA256']!=held['payloadSHA256']:
            raise ValueError('Historical hold actual input differs')
        e=enc['fields'][c]['E']
        if nn['actualEFields'][c]!=e:raise ValueError('Native corner used another E')
        for step in [2,3]:
            x=next(x for x in nn['actual'] if x['kind']=='denoiser-input' and x['case']==c and x['step']==step)
            if x['latentPayloadSHA256']!=nn['fields'][c]['latent'+str(step)]['payloadSHA256'] or x['EPayloadSHA256']!=e['payloadSHA256']:
                raise ValueError('Native corner prefix/interface differs')
        if own_source['prefix'][c]!=expected:raise ValueError('Saved hold request differs')
        joins.append({'case':c,'Z2':nn['fields'][c]['latent2'],'v1':nn['fields'][c]['velocity1'],
            'E':e,'NNZ3':expected['latent3'],'NHVelocityPayloadSHA256':held['payloadSHA256'],
            'NHZ4':nh['fields']['hold-'+c]['latent4']})
    return joins


def stage(root):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    api=Api();parent=HERE.parent/'artifacts/retained-query-v1';hold=HERE.parent/'artifacts/last-velocity-v1'
    c,j,old=checked_collection(parent,current_workers=False);hc,hj,hr=checked_collection(hold,current_workers=False)
    for receipt,job in [(c,j),(hc,hj)]:
        fresh=api.json('GET','/api/jobs/'+job['job_id'])
        if fresh['state']!='succeeded' or fresh['payload_custody']!=receipt['custody']:raise ValueError('Authoritative corner differs')
    p0=parent/'result';h0=hold/'result';s=read(p0/'inputs.json');nn=read(p0/'images-candidate.json');enc=read(p0/'encoder-candidate.json')
    nh=read(h0/'images.json');forecast=read(h0/'forecast-seal.json');hs=read(h0/'inputs.json')
    if hs['parentRequestSHA256']!=old['requestSHA256'] or hs['parentReportSHA256']!=sha(p0/'report.json') or hs['parentSealSHA256']!=sha(p0/'candidate-seal.json'):
        raise ValueError('Hold parent lineage differs')
    if hs['numericalProfile']!=s['numericalProfile'] or hs['nativeSources']!=s['nativeSources'] or hs['schedule']!=s['schedule'] or hs['versions']!=s['versions'] or any(s['imageAssets'].get(n)!=h for n,h in hs['decoderAssets'].items()):
        raise ValueError('Historical consumer contracts differ')
    bindings=qualify(nn,enc,forecast,nh,hs)
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    for n in OWNED:
        for previous in [old,hr]:
            if n in previous['ownedSources'] and sha(HERE/n)!=previous['sourceHashes'][n]:raise ValueError('Shared qualified source changed')
        shutil.copyfile(HERE/n,p/n)
    shutil.copyfile(HERE.parent/'docs/plans/two-clock-velocity-factorial.md',p/'frozen-plan.md')
    table=read(p0/'clock-table.json');ids=read(p0/'noise-seal.json')['ids']
    prefix={c:{'latent2':nn['fields'][c]['latent2'],'velocity1':nn['fields'][c]['velocity1'],'E':enc['fields'][c]['E']} for c in ['00','10']}
    target={'NN':{'fields':nn['fields'],'images':nn['images']},
        'NH':{'fields':{c:{**nh['fields']['hold-'+c],'latent3':nn['fields'][c]['latent3'],'velocity3':forecast['fields']['hold-'+c]['velocity3']} for c in ['00','10']},
              'images':{c:nh['images']['hold-'+c] for c in ['00','10']}},
        'full':read(p0/'targets.json')['fullImages'],'sourceFiles':{}}
    for label,job,local,names in [('native',j,p0,['images-candidate.json','candidate-seal.json','encoder-candidate.json']),
                                 ('hold',hj,h0,['forecast-seal.json','images.json','candidate-seal.json'])]:
        for n in names:target['sourceFiles'][label+'-'+n]={'path':job['result']['diagnostics']['cwd']+'/'+n,'sha256':sha(local/n)}
    allowed={v['path'] for fs in prefix.values() for v in fs.values()}|{ids['path']}
    forbidden=set(s['forbiddenTargets'])
    for ref in ['NN','NH','full']:
        forbidden.update(v['path'] for fs in target[ref]['fields'].values() for v in fs.values())
        forbidden.update(v['path'] for v in target[ref]['images'].values())
    source=dict(s);source.update(prefix=prefix,ids=ids,clockTable=table,
        nativeParentRoot=j['result']['diagnostics']['cwd'],holdParentRoot=hj['result']['diagnostics']['cwd'],
        parentRequestSHA256=old['requestSHA256'],holdRequestSHA256=hr['requestSHA256'],
        targetRoots=sorted({j['result']['diagnostics']['cwd'],hj['result']['diagnostics']['cwd'],s['parentRoot']}),
        forbiddenTargets=sorted(forbidden-allowed),guardCanary=nn['fields']['00']['velocity2']['path'])
    save(p/'inputs.json',source);save(p/'targets.json',target);save(p/'inherited-bindings.json',{'nativeRequestSHA256':old['requestSHA256'],'holdRequestSHA256':hr['requestSHA256'],'joins':bindings,'scope':'Complete known-input/source/profile/schedule/clock/image consumer binding; not fresh prompt evidence.'})
    save(p/'resource-basis.json',{'intentFirstPreview':_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda'),
        'measuredCompatibleJobID':j['job_id'],'measuredCompatibleResult':c['result'],'measuredCompatibleReservation':j['reservation'],
        'exception':'Generic full-model/fallback envelope is inapplicable to this B1 serial same-shape consumer. Reuse measured14336MBRAM/13000MBVRAM CPU4; prior peak8848MBVRAM. Two native calls/four VAE decodes, unchanged parameter residency/1536tokens. Guard max14300MB fits15557MB physical-minus-margin; verify admission. No encoder, retries or duplicate reference run.'})
    r={'schema':'tce-two-clock-request-v1','policies':['HH','HN'],'prompts':old['prompts'],'seed':old['seed'],
        'parentRequestSHA256':old['requestSHA256'],'holdRequestSHA256':hr['requestSHA256'],'ownedSources':OWNED,
        'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'model':'flux2-klein-4b-distilled','backend':'diffusers',
        'device':'cuda','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'steps':4,'timeoutSeconds':600,
        'protocol':'Actual own Z2/v1/E; shared U2(Z2,v1). HH holds v1 again; HN nativeclock3 consumes actual emittedZ3H. Fourimages seal before inheritedNN/NH/full targets;4postseal scheduler-only exact native/hold replays. No fit/newref/noise.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from run import command
    from mrun.client.submit import launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect the existing job')
    cmd=command()[:-5]+['python','-u','two_clock_study.py','execute','.'];i=cmd.index('uv')
    cmd[i:i]=['OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=launch(cmd,experiment='tce-two-clock-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs={'cuda':True},
        config={k:r[k] for k in ['schema','requestSHA256','model','backend','device','dtype','batch_size','seq_len','height','width']},
        ram_mb=14336,vram_mb=13000,cpu_threads=4,disk_gb=1,est_wall_s=120,timeout_s=600,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='OwnZ2/v1/E two-clock HH/HN factorial.2shared held-clock2 commits;HHholds v1again,HN actualnativeclock3 on ownZ3H.2native denoisers/206affines/4VAE/fourimages seal before inheritedNN/NH/full;4postseal scheduler-only exact replays.4historicalprefixdenoiserscharged. No encoder/newrefs/noise/fit. CPU4 serialB1/1536 measured14336RAM13000VRAM compatible exception vsbadgeneric envelope;120sexpected600scap,noretries.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def seal(root):
    from paired_fields import load
    r=read(root/'request.json');s=read(root/'inputs.json');im=read(root/'images.json');keys={p+'-'+c for p in r['policies'] for c in s['prefix']}
    if set(im['fields'])!=keys or set(im['images'])!=keys or im['requestSHA256']!=r['requestSHA256']:raise ValueError('Required candidate coverage differs')
    for key in keys:
        policy,c=key.split('-');expected=set(PORTS)|({'C3'} if policy=='HN' else set())
        if set(im['fields'][key])!=expected or im['fields'][key]['latent3']!=im['common'][c]['latent3']:
            raise ValueError('Branch does not share exact emitted parent')
        for item in im['fields'][key].values():load(item)
        if policy=='HH' and im['fields'][key]['velocity3']['payloadSHA256']!=s['prefix'][c]['velocity1']['payloadSHA256']:
            raise ValueError('HH did not hold v1 again')
        if sha(im['images'][key]['path'])!=im['images'][key]['sha256']:raise ValueError('Original PNG differs')
        shared=next(x for x in im['actual'] if x['case']==c and x['kind']=='shared-commit')
        if shared['incomingPayloadSHA256']!=s['prefix'][c]['latent2']['payloadSHA256'] or shared['velocityPayloadSHA256']!=s['prefix'][c]['velocity1']['payloadSHA256'] or shared['outgoingPayloadSHA256']!=im['common'][c]['latent3']['payloadSHA256']:
            raise ValueError('Shared actual input/output differs')
        if policy=='HN':
            actual=next(x for x in im['actual'] if x['case']==c and x['kind']=='native-input')
            if actual['latentPayloadSHA256']!=im['common'][c]['latent3']['payloadSHA256'] or actual['EPayloadSHA256']!=s['prefix'][c]['E']['payloadSHA256']:
                raise ValueError('Actual native read did not consume own state')
        final=next(x for x in im['actual'] if x['case']==c and x['policy']==policy and x['kind']=='final-commit')
        if final['incomingPayloadSHA256']!=im['common'][c]['latent3']['payloadSHA256'] or final['velocityPayloadSHA256']!=im['fields'][key]['velocity3']['payloadSHA256'] or final['outgoingPayloadSHA256']!=im['fields'][key]['latent4']['payloadSHA256']:
            raise ValueError('Actual final commit differs')
    save(root/'candidate-seal.json',{'schema':'tce-two-clock-seal-v1','requestSHA256':r['requestSHA256'],
        'imagesSHA256':sha(root/'images.json'),'common':im['common'],'fields':im['fields'],'images':im['images']})


def score(root,out,phases,started):
    import json
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from last_velocity_decode import scheduler
    from complete_masked_study import profile
    from paired_fields import load,payload,retain,metrics,paired
    from retained_queries import energy_account
    from two_clock import interaction,finite_commit
    r=read(root/'request.json');s=read(root/'inputs.json');im=read(root/'images.json');seal_sha=sha(root/'candidate-seal.json');target=read(root/'targets.json');profile(torch,4)
    for x in target['sourceFiles'].values():
        if sha(x['path'])!=x['sha256']:raise ValueError('Historical source changed')
    replays=[]
    with torch.inference_mode():
        for c in s['prefix']:
            z2=load(s['prefix'][c]['latent2'],'cuda');v2=load(target['NN']['fields'][c]['velocity2'],'cuda');sch=scheduler(s,torch,FlowMatchEulerDiscreteScheduler);sch.set_begin_index(2)
            z3=sch.step(v2,sch.timesteps[2],z2,return_dict=False)[0]
            if payload(z3)!=target['NN']['fields'][c]['latent3']['payloadSHA256']:raise ValueError('Native clock2 finite replay differs')
            sch=scheduler(s,torch,FlowMatchEulerDiscreteScheduler);z4=sch.step(v2,sch.timesteps[3],z3,return_dict=False)[0]
            if payload(z4)!=target['NH']['fields'][c]['latent4']['payloadSHA256']:raise ValueError('Historical hold finite replay differs')
            replays.append({'case':c,'nativeZ3PayloadSHA256':payload(z3),'holdZ4PayloadSHA256':payload(z4),'bitExact':True})
    fs={p:{c:(im['fields'][p+'-'+c] if p in ['HH','HN'] else target[p]['fields'][c]) for c in ['00','10']} for p in ['NN','NH','HN','HH']}
    values={p:{c:{port:load(fs[p][c][port]) for port in PORTS} for c in ['00','10']} for p in fs}
    accounts=[];responses=[];contrasts=[];energy=[];interaction_fields={};finite_fields={};finite_accounts=[];identity=[];observations=0;gap2=s['schedule']['sigmas'][3]-s['schedule']['sigmas'][2];gap3=s['schedule']['sigmas'][4]-s['schedule']['sigmas'][3]
    with (out/'observations.jsonl').open('x') as stream:
        for p in fs:
            for c in ['00','10']:
                for port in ['velocity3','latent4','pngRGB']:
                    a=values[p][c][port]
                    for ref in ['NN','full']:
                        b=values['NN'][c][port] if ref=='NN' else load(target['full']['fields'][c][port])
                        accounts.append({'policy':p,'case':c,'port':port,'reference':ref,**metrics(a,b)})
                    if p in ['HH','HN']:
                        b=values['NN'][c][port];aa=a.reshape(512 if port=='pngRGB' else a.shape[1],-1);bb=b.reshape(aa.shape)
                        for j,(ar,br) in enumerate(zip(aa,bb)):
                            stream.write(json.dumps({'policy':p,'case':c,'port':port,'position':j,**metrics(ar,br)},allow_nan=False)+'\n');observations+=1
                if p!='NN':
                    a=values[p][c]['pngRGB'];b=values['NN'][c]['pngRGB'];full=load(target['full']['fields'][c]['pngRGB'])
                    energy.append({'policy':p,'case':c,'firstTerm':'policyReplacementError','secondTerm':'inheritedEncoderImageError',**energy_account(a-b,b-full,a-full)})
            for port in ['velocity3','latent4','pngRGB']:
                a,b=[values[p][c][port] for c in ['00','10']]
                for ref in ['NN','full']:
                    c,d=[values['NN'][case][port] if ref=='NN' else load(target['full']['fields'][case][port]) for case in ['00','10']]
                    responses.append({'policy':p,'port':port,'reference':ref,**paired(a,b,c,d)})
        for c in ['00','10']:
            interaction_fields[c]={};finite_fields[c]={};z2=load(s['prefix'][c]['latent2']);v1=load(s['prefix'][c]['velocity1'])
            for kind,z3,v in [('N',values['NN'][c]['latent3'],load(target['NN']['fields'][c]['velocity2'])),('H',values['HH'][c]['latent3'],v1)]:
                eps=finite_commit(z2,z3,v,gap2);n='clock2'+kind;finite_fields[c][n]=retain(root/'arrays/finite'/c,n,eps)
                finite_accounts.append({'case':c,'commit':n,'rms':float(np.sqrt(np.mean(eps*eps))),'energy':float(np.sum(eps*eps))})
            eps3={}
            for p in fs:
                v=values[p][c];eps=finite_commit(v['latent3'],v['latent4'],v['velocity3'],gap3);n='clock3'+p;eps3[p]=eps;finite_fields[c][n]=retain(root/'arrays/finite'/c,n,eps)
                finite_accounts.append({'case':c,'commit':n,'rms':float(np.sqrt(np.mean(eps*eps))),'energy':float(np.sum(eps*eps))})
            js={}
            for port in PORTS:
                v=[values[p][c][port] for p in ['NN','NH','HN','HH']];j=interaction(*v);js[port]=j
                if port=='latent3' and np.any(j):raise ValueError('Shared-parent cancellation broken')
                interaction_fields[c][port]=retain(root/'arrays/interaction'/c,port,j)
                contrasts.append({'case':c,'port':port,'interpretation':'Held-history late read compared with native-history late read; discrepancy is minus policy interaction.',**metrics(v[2]-v[3],v[0]-v[1])})
            finite_j=interaction(*[eps3[p] for p in ['NN','NH','HN','HH']]);finite_fields[c]['finalInteraction']=retain(root/'arrays/finite'/c,'finalInteraction',finite_j)
            identity.append({'case':c,'firstTerm':'sigmaGapTimesVelocityInteraction','secondTerm':'finiteCommitInteraction',**energy_account(gap3*js['velocity3'],finite_j,js['latent4'])})
    if sha(root/'candidate-seal.json')!=seal_sha:raise ValueError('Candidate seal changed')
    for key,x in im['images'].items():shutil.copyfile(x['path'],out/(key+'.png'))
    for ref in ['NN','NH','full']:
        for c,x in target[ref]['images'].items():
            if sha(x['path'])!=x['sha256']:raise ValueError('Inherited original PNG changed')
            shutil.copyfile(x['path'],out/(ref+'-'+c+'.png'))
    for n in ['request.json','inputs.json','targets.json','inherited-bindings.json','resource-basis.json','frozen-plan.md','images.json','candidate-seal.json']:shutil.copyfile(root/n,out/n)
    save(out/'interaction-fields.json',interaction_fields);save(out/'finite-fields.json',finite_fields)
    save(out/'report.json',{'schema':'tce-two-clock-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal_sha,
        'accounts':accounts,'pairedColorResponse':responses,'lateReadContrast':contrasts,'pixelErrorInteraction':energy,'finiteCommits':finite_accounts,
        'latentInteractionIdentity':identity,'nativeSchedulerReplays':replays,'observations':observations,'phases':phases,'workerSecondsAfterImports':time.perf_counter()-started,
        'work':{'newEncoderEvaluations':0,'nativeDenoisers':sum(x['denoiser'] for x in im['counts'].values()),'nativeTransformerLinears':sum(x['parameter'].startswith('transformer.') for x in im['linears']),
            'nativeTransformerNominalAffineMACs':sum(x['nominalMACs'] for x in im['linears'] if x['parameter'].startswith('transformer.')),
            'nativeVAEDecodes':4,'VAEConvLinearCalls':len(im['VAEOperations']),'VAENominalConvLinearMACs':sum(x['nominalMACs'] for x in im['VAEOperations']),
            'sharedClock2Commits':2,'finalClock3Commits':4,'postsealSchedulerReplayCommits':4,'newImages':4,'newReferenceImages':0,'newNoiseDraws':0,'newClockCompilations':0,
            'chargedHistoricalPrefixDenoisers':4,'chargedHistoricalPrefixNominalAffineMACs':18895708618752,'prospectiveCallsPerImage':{'NN':4,'NH':3,'HN':3,'HH':2}},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limitations':['Known two-context own-prefix replay with original weights/encoding/earliercalls/VAE paid; not arbitrary-prompt zero-neural serving.','NN/NH inherited only through complete authenticated source/profile/clock/input identity. No new reference campaign or matched-speedup measurement.','Factorial is a policy interaction: held velocity depends on history. Not isolated denoiser curvature.']})


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);started=time.perf_counter();out=root/'response-result';out.mkdir();tick=time.perf_counter()
    subprocess.run([sys.executable,'-u',str(root/'two_clock_study.py'),'images',str(root)],check=True,timeout=480)
    phases=[{'action':'images','wallSecondsIncludingFreshProcess':time.perf_counter()-tick}]
    seal(root);print('TWO_CLOCK_ALL_CANDIDATES_SEALED',sha(root/'candidate-seal.json'),flush=True)
    score(root,out,phases,started);print('TWO_CLOCK_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','images','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='images':
        from two_clock_images import execute
        v=execute(a.run)
    elif a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(__import__('json').dumps(v))
