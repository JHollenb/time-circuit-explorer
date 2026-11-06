"""Source-pad tangent versus constant write, own images, postseal diagnostics."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import np,read,save,sha,identity
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
ENCODER_PORTS=['H'+str(i) for i in range(28)]+['E']
IMAGE_PORTS=['C'+str(i) for i in range(4)]+['velocity'+str(i) for i in range(4)]+['latent'+str(i) for i in range(1,5)]+['vaeInput','decodedRaw','rgbFloat','pngRGB']
OWNED=['padding_mlp_study.py','padding_mlp.py','padding_mlp_compile.py','padding_mlp_encoder.py','padding_mlp_images.py','padding_mlp_diagnostic.py',
    'uncentered_reader.py','uncentered_reader_study.py','encoder_atoms.py','paired_fields.py','masked_kv_study.py','conditioning_demand.py','complete_masked_study.py','flux_clock.py','flux_clock_study.py']


def inherited_clock(root,source):
    p=Path(root)/'clock-table.json'
    if sha(p)!=source['clockTableSHA256']:raise ValueError('Clock changed')
    t=read(p)
    if t['requestSHA256']!=source['clockParentRequestSHA256']:raise ValueError('Clock request differs')
    return t


def install_guard(root):
    from flux_clock import deny_targets
    root=Path(root);source=read(root/'inputs.json');deny_targets([root/'targets.json',*source['forbiddenTargets']])
    try:open(root/'targets.json','rb')
    except PermissionError:return True
    raise RuntimeError('Target exclusion inactive')


def stage(root):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    parent=HERE.parent/'artifacts/fresh-prompt-square-v1';c,j,old=checked_collection(parent,current_workers=False);api=Api();live=api.json('GET','/api/jobs/'+j['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody']:raise ValueError('Qualified square differs')
    f=parent/'result';source=read(f/'inputs.json');old_reader=read(f/'reader-seal.json');root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    for name in OWNED:
        if name in old['sourceHashes'] and sha(HERE/name)!=old['sourceHashes'][name]:raise ValueError('Shared qualified source changed: '+name)
        shutil.copyfile(HERE/name,p/name)
    for name in ['flux_functional_qwen_prefix.py','qwen_h18_state_history.py']:
        if sha(parent/'payload'/name)!=old['sourceHashes'][name]:raise ValueError('Graph/history differs')
        shutil.copyfile(parent/'payload'/name,p/name)
    for name in ['clock-table.json','noise-seal.json']:shutil.copyfile(f/name,p/name)
    shutil.copyfile(HERE.parent/'docs/plans/padding-mlp-tangent.md',p/'frozen-plan.md')
    source.update(noiseSealSHA256=sha(f/'noise-seal.json'),noiseParentRequestSHA256=old['requestSHA256'],parentRoot=j['result']['diagnostics']['cwd'],parentRequestSHA256=old['requestSHA256'])
    source['forbiddenTargets'] += [x['path'] for x in [*old_reader['fields'].values(),*old_reader['h0Fields'].values(),*old_reader['packets'],*old_reader['states'],*old_reader['lexicalRows']]]
    target={'reader':old_reader,'sourceFiles':{n:{'path':source['parentRoot']+'/'+n,'sha256':sha(f/n)} for n in ['reader-seal.json','encoder-candidate.json','images-candidate.json','encoder-full.json','images-full.json','candidate-seal.json','full-seal.json']}}
    for lane,label in [('candidate','top4'),('full','full')]:
        for kind in ['encoder','images']:
            value=read(f/(kind+'-'+lane+'.json'));target[label+kind.title()]=value
            source['forbiddenTargets'] += [x['path'] for fs in value['fields'].values() for x in fs.values()]
            if kind=='images':source['forbiddenTargets'] += [x['path'] for x in value['images'].values()]
    source['forbiddenTargets']=sorted(set(source['forbiddenTargets']));save(p/'inputs.json',source);save(p/'targets.json',target)
    preview=_plans_for_hosts('flux2-klein-4b-distilled','forward',api,seq_lens=[1536],backend='diffusers',dtype='bfloat16',device='cuda')
    save(p/'resource-basis.json',{'intentFirstPreview':preview,'measuredParentJobID':j['job_id'],'measuredParentResult':c['result'],
        'exception':'GenericB64/fallback shape is inapplicable. Reuse measured serialB1/51214336MBRAM13000MBVRAM envelope; compiler/reader/encoder/images/diagnostic separatechildren, imageweights not co-resident with J/compiler. Fourimages,300sexpected900scap,noretries.'})
    prompts=[{**q,'id':policy+'-'+q['id'],'case':q['id'],'policy':policy} for policy in ['constant','tangent'] for q in old['prompts'] if q['id'] in ['00','10']]
    r={'schema':'tce-padding-mlp-request-v1','prompts':prompts,'seed':old['seed'],'manifestSHA256':old['manifestSHA256'],'readerPolicy':old['readerPolicy'],'frozenReaderSHA256':old['frozenReaderSHA256'],'parentRequestSHA256':old['requestSHA256'],
        'ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'model':'flux2-klein-4b-distilled','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'seq_len':1536,'height':512,'width':512,'steps':4,'timeoutSeconds':900,
        'scope':'Source-only full-input smooth tangent versus native source-anchor constant for block0padding MLP. Active37rows share exact original write; all512rows continue. Reduced learned arithmetic, no strict zero-neural or speedup claim.',
        'protocol':'Compile FP64smooth sourceJ/emittedFP32 at decoded nativeRMS(originalpadembedding); nativeb at repeated512rows. Own raw firstreader/residual/nativeRMS. Runtime FP32delta@J.T+nativeb thenBF16.4ownimages seal before2paid nativeMLP diagnostics and savedtop4/fulltarget scoring. No fit/rankcut/newnoise/nativeimage campaign/targetrefresh.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r);save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from run import command
    from mrun.client.submit import launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted')
    cmd=command()[:-5]+['python','-u','padding_mlp_study.py','execute','.'];i=cmd.index('uv');cmd[i:i]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job=launch(cmd,experiment='tce-padding-mlp-tangent-v1',model=r['model'],task='forward',payload=root/'payload',pin='beast',needs={'cuda':True},config={k:r[k] for k in ['schema','requestSHA256','model','backend','device','dtype','batch_size','seq_len','height','width']},
        ram_mb=14336,vram_mb=13000,cpu_threads=4,disk_gb=3,est_wall_s=300,timeout_s=900,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='Sourcepad nativeb/smoothFP64J->FP32; constant/tangent paddingMLP0 on00/10. Ownrawreader/nativeRMS,shared37activewrite,all512rows.4images/104nativeencoderblocks/16denoisers/4VAE paid. Seal before2nativeMLP diagnostic+savedtop4/fulltargets. No rankfit/refresh/newnoise/refs. Compiler127.5BMAC+anchor,liveJlearned. SerialchildrenCPU1/4 B1/512,14336RAM13000VRAM measuredb69f30eff20a exception vsbadB64plan.300sexpected900scap,noretries.')
    v={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',v);return v


def seal(root):
    from paired_fields import load
    r=read(root/'request.json');enc=read(root/'encoder-candidate.json');im=read(root/'images-candidate.json');reader=read(root/'reader-seal.json');ids=[q['id'] for q in r['prompts']]
    if set(enc['fields'])!=set(ids) or set(im['fields'])!=set(ids):raise ValueError('Candidate coverage differs')
    for c in ids:
        if set(enc['fields'][c])!=set(ENCODER_PORTS) or set(im['fields'][c])!=set(IMAGE_PORTS):raise ValueError('Complete field coverage differs')
        for fs in [enc['fields'][c],im['fields'][c],enc['firstMLPFields'][c]]:
            for x in fs.values():load(x)
        if im['actualEFields'][c]!=enc['fields'][c]['E'] or im['schedules'][c]!=read(root/'inputs.json')['schedule']:raise ValueError('Actual E/schedule differs')
        if sha(im['images'][c]['path'])!=im['images'][c]['sha256']:raise ValueError('Original image changed')
    if enc['compileReceiptSHA256']!=sha(root/'compile-receipt.json') or enc['readerSealSHA256']!=sha(root/'reader-seal.json') or enc['lexicalSealSHA256']!=reader['lexicalSealSHA256'] or im['encoderSHA256']!=sha(root/'encoder-candidate.json'):raise ValueError('Consumer chain differs')
    save(root/'candidate-seal.json',{'schema':'tce-padding-mlp-candidates-v1','requestSHA256':r['requestSHA256'],'compileReceiptSHA256':sha(root/'compile-receipt.json'),'readerSHA256':sha(root/'reader-seal.json'),'encoderSHA256':sha(root/'encoder-candidate.json'),
        'imagesSHA256':sha(root/'images-candidate.json'),'clockTableSHA256':sha(root/'clock-table.json'),'noiseSealSHA256':sha(root/'noise-seal.json'),'encoderFields':enc['fields'],'imageFields':im['fields'],'images':im['images']})


def score(root,out,phases,started):
    from paired_fields import load,metrics,paired,retain
    r=read(root/'request.json');enc=read(root/'encoder-candidate.json');im=read(root/'images-candidate.json');target=read(root/'targets.json');diag=read(root/'native-write-diagnostic.json');seal_sha=sha(root/'candidate-seal.json');accounts=[];pairs=[];writes=[];drift=[];diffs={};observations=0
    for x in target['sourceFiles'].values():
        if sha(x['path'])!=x['sha256']:raise ValueError('Frozen target metadata differs')
    with (out/'observations.jsonl').open('x') as stream:
        for q in r['prompts']:
            c=q['id'];base=q['case'];mask=load(enc['firstMLPFields'][c]['mask']).astype(bool);use={'all':np.ones(512,dtype=bool),'active':mask,'padding':~mask}
            for port in ENCODER_PORTS[1:]+IMAGE_PORTS:
                is_enc=port in ENCODER_PORTS;a=load((enc if is_enc else im)['fields'][c][port])
                for reference in ['top4','full']:
                    fs=target[reference+('Encoder' if is_enc else 'Images')]['fields'][base];b=load(fs[port]);accounts.append({'id':c,'policy':q['policy'],'case':base,'reference':reference,'port':port,'region':'all',**metrics(a,b)})
                    if is_enc:
                        for region in ['active','padding']:accounts.append({'id':c,'policy':q['policy'],'case':base,'reference':reference,'port':port,'region':region,**metrics(a.reshape(512,-1)[use[region]],b.reshape(512,-1)[use[region]])})
                    if is_enc or port in ['rgbFloat','pngRGB']:
                        for i,(ar,br) in enumerate(zip(a.reshape(512,-1),b.reshape(512,-1))):stream.write(json.dumps({'id':c,'policy':q['policy'],'case':base,'reference':reference,'port':port,'position':i,'attended':bool(mask[i]) if is_enc else None,**metrics(ar,br)},allow_nan=False)+'\n');observations+=1
                    if reference=='top4' and port in ['E','latent4','pngRGB']:diffs[c+'.'+port]=retain(root/'arrays/differences',c+'.'+port,a-b)
            native=load(diag['fields'][base]['write']);own=load(enc['firstMLPFields'][c]['write'])
            for region,selection in use.items():writes.append({'id':c,'policy':q['policy'],'region':region,**metrics(own.reshape(512,-1)[selection],native.reshape(512,-1)[selection])})
            drift.append({'id':c,'activeH1':metrics(load(enc['fields'][c]['H1']).reshape(512,-1)[mask],load(target['top4Encoder']['fields'][base]['H1']).reshape(512,-1)[mask]),
                'diagnosticH1VsTop4':metrics(load(diag['fields'][base]['H1']),load(target['top4Encoder']['fields'][base]['H1']))})
        for policy in ['constant','tangent']:
            for port in ['H1','E','velocity0','latent4','pngRGB']:
                is_enc=port in ENCODER_PORTS;own=(enc if is_enc else im)['fields'];a,b=[load(own[policy+'-'+c][port]) for c in ['00','10']]
                for reference in ['top4','full']:
                    fs=target[reference+('Encoder' if is_enc else 'Images')]['fields'];ra,rb=[load(fs[c][port]) for c in ['00','10']];pairs.append({'policy':policy,'reference':reference,'port':port,**paired(a,b,ra,rb)})
                if port in ['E','latent4','pngRGB']:diffs['color.'+policy+'.'+port]=retain(root/'arrays/differences','color.'+policy+'.'+port,b-a)
    if sha(root/'candidate-seal.json')!=seal_sha:raise ValueError('Candidate seal changed')
    for q in r['prompts']:shutil.copyfile(im['images'][q['id']]['path'],out/(q['id']+'.png'))
    for ref in ['top4','full']:
        for c in ['00','10']:
            x=target[ref+'Images']['images'][c]
            if sha(x['path'])!=x['sha256']:raise ValueError('Saved original image differs')
            shutil.copyfile(x['path'],out/(ref+'-'+c+'.png'))
    for n in ['request.json','inputs.json','targets.json','resource-basis.json','frozen-plan.md','clock-table.json','noise-seal.json','compile-receipt.json','reader-seal.json','encoder-candidate.json','images-candidate.json','candidate-seal.json','native-write-diagnostic.json']:shutil.copyfile(root/n,out/n)
    save(out/'differences.json',diffs);calls=enc['primitives'];image_calls=im['linears']
    save(out/'report.json',{'schema':'tce-padding-mlp-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal_sha,'accounts':accounts,'pairedColorResponse':pairs,'directMLPWrite':writes,'activeCompactGEMMDrift':drift,'observations':observations,'phases':phases,'workerSecondsAfterImports':time.perf_counter()-started,
        'work':{'newImages':4,'newReferenceImages':0,'newNoiseDraws':0,'nativeEncoderBlocks':enc['nativeComponentBlocks'],'encoderSDPA':sum(x['kind']=='SDPA' for x in calls),'encoderLinears':sum(x['kind']=='linear' for x in calls),'encoderNominalLinearMACs':sum(x['nominalMACs'] for x in calls if x['kind']=='linear'),
            'tangentGEMMs':len(enc['tangentCalls']),'tangentMACs':sum(x['nominalMACs'] for x in enc['tangentCalls']),'sharedActiveWriteReuses':len(enc['sharedActiveWriteReads']),
            'nativeDenoisers':sum(x['denoiser'] for x in im['counts'].values()),'nativeVAE':sum(x['VAE'] for x in im['counts'].values()),'nativeTransformerLinears':sum(x['parameter'].startswith('transformer.') for x in image_calls),'nativeTransformerNominalLinearMACs':sum(x['nominalMACs'] for x in image_calls if x['parameter'].startswith('transformer.')),
            'offlineCompiler':read(root/'compile-receipt.json')['work'],'paidDiagnostic':diag['work']},'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limitations':['Two known development prompts/oneseed; source-only anchor and J, no fresh arbitrary-prompt or strictzero-neural certificate.','J remains learned dense arithmetic; original weights remain for active rows and all later encoder/denoiser/VAE work. Native diagnostic is paid postseal, not serving.','Native BF16 anchor differs from smooth FP64 surrogate; compact37active-row GEMM drift and emission/rounding remain explicit. No matched speedup.']})


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);started=time.perf_counter();out=root/'response-result';out.mkdir();phases=[]
    for action in ['compile','reader','encoder','images']:
        tick=time.perf_counter();subprocess.run([sys.executable,'-u',str(root/'padding_mlp_study.py'),action,str(root)],check=True,timeout=780);phases.append({'action':action,'wallSecondsIncludingFreshProcess':time.perf_counter()-tick})
    seal(root);print('PADDING_MLP_CANDIDATES_SEALED',sha(root/'candidate-seal.json'),flush=True)
    tick=time.perf_counter();subprocess.run([sys.executable,'-u',str(root/'padding_mlp_study.py'),'diagnostic',str(root)],check=True,timeout=180);phases.append({'action':'diagnostic','wallSecondsIncludingFreshProcess':time.perf_counter()-tick})
    score(root,out,phases,started);print('PADDING_MLP_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','compile','reader','encoder','images','diagnostic','collect']);p.add_argument('run');a=p.parse_args()
    if a.action in ['compile','encoder','images','diagnostic']:v=__import__('padding_mlp_'+a.action).execute(a.run)
    elif a.action=='reader':
        install_guard(Path(a.run).resolve());import uncentered_reader
        v=uncentered_reader.execute(a.run)
    elif a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(json.dumps(v))
