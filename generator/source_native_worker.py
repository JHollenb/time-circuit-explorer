"""One resident selective reader, isolated fit/writer actors, then two images."""
from pathlib import Path
import os
import subprocess
import sys
import time
from encoder_atoms import np,read,save,sha,identity
from paired_fields import load,retain,payload


def guard(root,s,foreign):
    import os
    phase={'serving':False};common=set();training=set();serving=set()
    for row in s['clockTable']['rows']:
        common.update(x['path'] for x in [*row['fields'].values(),*row['clockOperands'].values()])
    common.update(str(Path(s['modelRoot'])/n) for n in s['imageAssets']);common.add(s['ids']['path'])
    for item in foreign['conditioningPages'].values():common.update(item['files'])
    for row in foreign['records']:
        ports=['z','previous']+(['target'] if row['population']=='discovery' else [])
        (training if row['population']=='discovery' else serving).update(row[n]['path'] for n in ports)
    serving.update(x['path'] for fs in s['prefix'].values() for x in fs.values())
    protected=[Path(p).resolve() for p in [*foreign['sourceRoots'],*s['targetRoots'],s['modelRoot']]]
    common={str(Path(p).resolve()) for p in common};training={str(Path(p).resolve()) for p in training};serving={str(Path(p).resolve()) for p in serving}
    target=str((root/'targets.json').resolve())
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve();name=str(p);allowed=common|training|(serving if phase['serving'] else set())
        if name==target or (any(p.is_relative_to(x) for x in protected) and name not in allowed):
            raise PermissionError('Undeclared or unsealed future unavailable in producer')
        if p.suffix in ('.safetensors','.pt','.pth') and name not in allowed:raise PermissionError('Undeclared model bytes')
    sys.addaudithook(audit)
    for p in [root/'targets.json',s['guardCanary'],s['prefix']['00']['latent3']['path']]:
        try:open(p,'rb')
        except PermissionError:continue
        raise RuntimeError('Producer access phase guard inactive')
    return phase


def execute(root):
    from flux_clock_study import verify,source_checks,assets_unchanged
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json');foreign=read(root/'corpus.json')
    access=guard(root,s,foreign)
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from diffusers import AutoencoderKLFlux2,Flux2KleinPipeline,FlowMatchEulerDiscreteScheduler
    from mdb.titan import Frame,ActSpec,ActRegistry,ProvidedFingerprint
    from source_native_reader import load as load_reader,form,PARAMETERS
    from native_reader_corpus import load as historical
    from complete_masked_study import profile
    from flux_clock import snapshot
    from last_velocity_decode import scheduler
    started=time.perf_counter();cpu=time.process_time();numerical=profile(torch,4)
    if numerical!=s['numericalProfile']:raise ValueError('Qualified numerical profile differs')
    model,assets=source_checks(s,diffusers);config=read(model/'transformer/config.json')
    if config['eps']!=1e-6 or config['num_attention_heads']!=24:raise ValueError('Source reader configuration differs')
    weights,norms,pos,loaded=load_reader(model/'transformer/diffusion_pytorch_model.safetensors',config)
    before={n:(v.data_ptr(),v._version,payload(v)) for n,v in weights.items()}
    parameter_hash=identity({x['name']:x['payloadSHA256'] for x in loaded})
    clock=s['clockTable']['rows'][3];mods={n:load(clock['fields'][n],'cuda') for n in ['double_img','double_txt']}
    pages={n:historical(x,'cuda') for n,x in foreign['conditioningPages'].items()}
    if any(payload(pages[role+'.sigmas'])!=s['schedule']['sigmasPayloadSHA256'] or payload(pages[role+'.timesteps'])!=s['schedule']['timestepsPayloadSHA256'] for role in ['base','shape']):raise ValueError('Historical clock differs')
    vae=AutoencoderKLFlux2.from_pretrained(str(model/'vae'),torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True,device_map={'':'cuda'})
    vae.eval().requires_grad_(False);pipe=Flux2KleinPipeline(scheduler=None,vae=vae,transformer=None,text_encoder=None,tokenizer=None,is_distilled=True)
    def vae_census():return {n:(v.data_ptr(),v._version,list(v.shape),str(v.dtype)) for n,v in [*vae.named_parameters(),*vae.named_buffers()]}
    vae_before=vae_census();vae_buffers={n:payload(v) for n,v in vae.named_buffers()}
    phase={};operations=[];attention=[];frames=[];read_rows=[];vae_ops=[];vae_hooks=[]
    def vae_hook(name):
        def hook(module,args,y):
            vae_ops.append({**phase,'module':name,'kind':type(module).__name__,'inputShape':list(args[0].shape),'outputShape':list(y.shape),
                'nominalMACs':y.numel()*(module.weight.numel()//module.weight.shape[0])})
        return hook
    for n,m in vae.named_modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):vae_hooks.append(m.register_forward_hook(vae_hook(n)))
    registry=ActRegistry();spec=ActSpec('tce.first_joint_image_read','v1',reads=('Z','E','imgIDs','txtIDs','modImage','modText','weights'),writes=('H0',),
        write_kinds={'H0':'tensor'},write_abis={'H0':{'shape':[1,1024,3072],'dtype':'torch.bfloat16','layout':'native dispatch output flattened heads'}},
        numerical_contract='source-qualified-BF16-first-joint-image-read')
    def operation(values):
        if values['weights'] is not weights or snapshot(torch)!=numerical:raise ValueError('Native reader binding differs')
        for n,v in weights.items():
            if (v.data_ptr(),v._version)!=before[n][:2]:raise ValueError('Source parameter mutable closure changed')
        tick=time.perf_counter()
        h=form(weights,values['Z'],values['E'],values['imgIDs'],values['txtIDs'],values['modImage'],values['modText'],norms,pos,native,24,operations,attention,phase)
        torch.cuda.synchronize()
        return {'H0':h},{'rawReaderWallSeconds':time.perf_counter()-tick,'originalAffineCalls':7,'originalAffineMACs':51136954368,'targetAccess':False}
    registry.register(spec,'selected-original-parameters',operation)
    def read_one(label,pop,z,n,e,ids,txt):
        phase.clear();phase.update(id=label,population=pop)
        values={'Z':z,'E':e,'imgIDs':ids,'txtIDs':txt,'modImage':mods['double_img'],'modText':mods['double_txt'],'weights':weights}
        fp={k:ProvidedFingerprint.content_sha256(parameter_hash if k=='weights' else payload(v)) for k,v in values.items()}
        abis={k:({'parameters':loaded,'mapIdentity':parameter_hash} if k=='weights' else {'shape':list(v.shape),'dtype':str(v.dtype),'stride':list(v.stride())}) for k,v in values.items()}
        frame=Frame.from_values(values,kinds={k:'parameter-map' if k=='weights' else 'tensor' for k in values},abis=abis,slot_fingerprints=fp,program_fingerprint=r['requestSHA256'],
            metadata={'id':label,'population':pop,'clock':3,'fingerprintConvention':'raw contiguous logical tensor bytes; weights map of verified source payloads'})
        output,evidence=registry.run(spec.identity,'selected-original-parameters',frame)
        fields={k:retain(root/'arrays/reads'/label,k,v) for k,v in [('z',z),('previous',n),('H0',output.values['H0'])]}
        row={'id':label,'population':pop,**fields,'frame':frame.to_dict(),'outputFrame':output.to_dict(),'evidence':evidence.to_dict()}
        frames.append(row);read_rows.append(row);return row
    discovery=[x for x in foreign['records'] if x['population']=='discovery'];features=[];targets=[]
    with torch.inference_mode():
        for row in discovery:
            z=historical(row['z'],'cuda');n=historical(row['previous'],'cuda');role=row['conditioning']
            fields=read_one(row['id'],'discovery',z,n,pages[role+'.conditioning'],pages[role+'.img_ids'],pages[role+'.txt_ids'])
            features.append(np.concatenate([load(fields[k]) for k in ['z','previous','H0']],-1).reshape(1024,3328).astype(np.float32))
            targets.append((historical(row['target']).double().numpy()-n.double().cpu().numpy()).reshape(1024,128))
    index={x['id']:i for i,x in enumerate(discovery)};pairs=np.asarray([[index[a],index[b]] for a,b in foreign['discoveryPairs']],dtype=np.int64)
    affected=np.zeros((32,32),bool);affected[-8:,-8:]=True;halo=affected.copy();halo[1:]|=affected[:-1];halo[:-1]|=affected[1:];halo[:,1:]|=affected[:,:-1];halo[:,:-1]|=affected[:,1:];mask=~halo.ravel()
    for a,b in pairs:
        if not np.array_equal(features[a][mask,:256],features[b][mask,:256]):raise ValueError('Declared unchanged-query operands differ')
    packet=root/'discovery.npz';np.savez(packet,features=np.stack(features),targets=np.stack(targets),pairs=pairs,mask=mask)
    packet_record={'path':str(packet),'sha256':sha(packet),'bytes':packet.stat().st_size}
    del features,targets
    quarantine={'protectedRoots':[*foreign['sourceRoots'],*s['targetRoots'],s['modelRoot']],
        'deniedCanaries':[str(root/'targets.json'),s['guardCanary'],foreign['records'][-1]['target']['path']]}
    fit_spec={'packet':packet_record,'output':str(root/'fit'),**quarantine};save(root/'fit-spec.json',fit_spec)
    zero=root/'baseline-writer.npy';np.save(zero,np.zeros((3329,128),np.float64),allow_pickle=False)
    baseline=Frame.from_values({'writer':{'path':str(zero),'sha256':sha(zero)}},program_fingerprint=r['requestSHA256'])
    fit_act=ActSpec('tce.fit_velocity_writer','v1',reads=('discoveryPacket',),writes=('writer',),numerical_contract='discovery-only-FP64-ridge-and-paired-response')
    fit_registry=ActRegistry()
    def proposed(values):
        if values['discoveryPacket']!=packet_record:raise ValueError('Fit packet identity differs')
        subprocess.run([sys.executable,'-u',str(root/'source_native_writer.py'),'fit',str(root/'fit-spec.json')],check=True,timeout=300,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        return {'writer':{'path':str(root/'fit/fit.json'),'sha256':sha(root/'fit/fit.json')}},{'discoveryOnly':True,'epochs':0,'gain':1}
    fit_registry.register(fit_act,'one-closed-form-proposal',proposed)
    source=Frame.from_values({'discoveryPacket':packet_record,'writer':baseline.values['writer']},program_fingerprint=r['requestSHA256'])
    fitted,fit_evidence=fit_registry.run(fit_act.identity,'one-closed-form-proposal',source)
    restored=source
    if restored.fingerprint!=source.fingerprint or sha(zero)!=baseline.values['writer']['sha256']:raise ValueError('Baseline restoration differs')
    fit_info=read(root/'fit/fit.json')
    for item in fit_info['fields'].values():load(item)
    save(root/'coefficient-seal.json',{'schema':'tce-source-native-coefficient-seal-v1','fitReceiptSHA256':sha(root/'fit/fit.json'),'fields':fit_info['fields'],'gain':1,
        'discoveryPacket':packet_record,'baselineWriter':baseline.values['writer'],'sourceFrame':source.to_dict(),'fittedFrame':fitted.to_dict(),'restoredFrame':restored.to_dict(),'fitEvidence':fit_evidence.to_dict(),
        'rollbackScope':'Reselect the immutable baseline Frame; not restoration of mutated solver/model state. No optimizer/RNG/epoch state exists. Original reader parameters remain separately fingerprinted.'})
    print('SOURCE_NATIVE_COEFFICIENTS_SEALED',sha(root/'coefficient-seal.json'),flush=True)
    access['serving']=True;serving=[];main_common={}
    with torch.inference_mode():
        for row in foreign['records']:
            if row['population']=='discovery':continue
            role=row['conditioning'];fields=read_one(row['id'],row['population'],historical(row['z'],'cuda'),historical(row['previous'],'cuda'),pages[role+'.conditioning'],pages[role+'.img_ids'],pages[role+'.txt_ids'])
            serving.append({k:fields[k] for k in ['id','population','z','previous','H0']}|{'output':str(root/'arrays/writer'/row['id'])})
        for c,fs in s['prefix'].items():
            z=load(fs['latent3'],'cuda');n=load(fs['velocity1'],'cuda');e=load(fs['E'],'cuda');ids=load(s['ids'],'cuda');txt=pipe._prepare_text_ids(e).to('cuda')
            fields=read_one(c,'main',z,n,e,ids,txt);main_common[c]={'txtIDs':retain(root/'arrays/common'/c,'txtIDs',txt)}
            serving.append({k:fields[k] for k in ['id','population','z','previous','H0']}|{'output':str(root/'arrays/writer'/c)})
    save(root/'serve-spec.json',{'coefficientSeal':str(root/'coefficient-seal.json'),'fitReceipt':str(root/'fit/fit.json'),'inputs':serving,'receipt':str(root/'serving.json'),**quarantine})
    subprocess.run([sys.executable,'-u',str(root/'source_native_writer.py'),'serve',str(root/'serve-spec.json')],check=True,timeout=180,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    serving_receipt=read(root/'serving.json');bindings=[];binding_registry=ActRegistry()
    binding_spec=ActSpec('tce.bind_emitted_velocity','v1',reads=('Z','previous','H0','coefficients','emitted'),writes=('velocity','rawVelocity'),
        write_kinds={'velocity':'tensor-artifact','rawVelocity':'tensor-artifact'},
        write_abis={'velocity':{'shape':[1,1024,128],'dtype':'torch.bfloat16'},'rawVelocity':{'shape':[1,1024,128],'dtype':'float64'}},
        numerical_contract='qualified-file-handoff-after-isolated-FP64-writer; not another writer evaluation')
    def bind(values):
        row=values['emitted']
        if values['coefficients']!=sha(root/'coefficient-seal.json') or {k:values[n] for k,n in [('z','Z'),('previous','previous'),('H0','H0')]}!=row['inputFields']:raise ValueError('Writer input/coefficient lineage differs')
        for name in ['velocity','rawVelocity']:load(row[name])
        return {name:row[name] for name in ['velocity','rawVelocity']},{'servingReceiptSHA256':sha(root/'serving.json'),'method':row['method'],'additionalWriterProducts':0}
    binding_registry.register(binding_spec,'sealed-actor-output-qualification',bind)
    for row in serving_receipt['rows']:
        vals={'Z':row['inputFields']['z'],'previous':row['inputFields']['previous'],'H0':row['inputFields']['H0'],
            'coefficients':sha(root/'coefficient-seal.json'),'emitted':row}
        abis={n:{'shape':v['shape'],'dtype':v['logicalDtype'],'storedDtype':v['storedDtype']} for n,v in vals.items() if n in ['Z','previous','H0']}
        frame=Frame.from_values(vals,kinds={n:'tensor-artifact' if n in abis else 'receipt' for n in vals},abis=abis,program_fingerprint=r['requestSHA256'],metadata={'id':row['id'],'method':row['method']})
        output,evidence=binding_registry.run(binding_spec.identity,'sealed-actor-output-qualification',frame)
        bindings.append({'id':row['id'],'method':row['method'],'inputFrame':frame.to_dict(),'outputFrame':output.to_dict(),'evidence':evidence.to_dict()})
    actual=[];fields={};images={};timings=[]
    with torch.inference_mode():
        for c,fs in s['prefix'].items():
            phase.clear();phase.update(id=c,population='main')
            writer=next(x for x in read(root/'serving.json')['rows'] if x['id']==c and x['method']=='candidate')
            z3=load(fs['latent3'],'cuda');velocity=load(writer['velocity'],'cuda');f={'latent3':fs['latent3'],'velocity3':writer['velocity'],'rawVelocity':writer['rawVelocity']};fields[c]=f
            final=scheduler(s,torch,FlowMatchEulerDiscreteScheduler);z4=final.step(velocity,final.timesteps[3],z3,return_dict=False)[0];f['latent4']=retain(root/'arrays/images'/c,'latent4',z4)
            actual.append({'case':c,'kind':'final-commit','incomingPayloadSHA256':payload(z3),'velocityPayloadSHA256':payload(velocity),'outgoingPayloadSHA256':payload(z4),'finalStepIndex':final.step_index})
            z=pipe._unpack_latents_with_ids(z4,load(s['ids'],'cuda'),32,32);mean=vae.bn.running_mean.view(1,-1,1,1).to(z.device,z.dtype)
            std=torch.sqrt(vae.bn.running_var.view(1,-1,1,1)+vae.config.batch_norm_eps).to(z.device,z.dtype)
            z=pipe._unpatchify_latents(z*std+mean);f['vaeInput']=retain(root/'arrays/images'/c,'vaeInput',z)
            torch.cuda.synchronize();tick=time.perf_counter();decoded=vae.decode(z,return_dict=False)[0];torch.cuda.synchronize();timings.append({'case':c,'kind':'VAE','wallSeconds':time.perf_counter()-tick})
            f['decodedRaw']=retain(root/'arrays/images'/c,'decodedRaw',decoded);rgb=pipe.image_processor.postprocess(decoded,output_type='np');png=pipe.image_processor.numpy_to_pil(rgb)[0]
            file=root/('reader-'+c+'.png');png.save(file);f['rgbFloat']=retain(root/'arrays/images'/c,'rgbFloat',rgb);f['pngRGB']=retain(root/'arrays/images'/c,'pngRGB',np.asarray(png))
            images[c]={'path':str(file),'sha256':sha(file),'bytes':file.stat().st_size}
    for hook in vae_hooks:hook.remove()
    if vae_census()!=vae_before or vae_buffers!={n:payload(v) for n,v in vae.named_buffers()}:raise ValueError('VAE mutable closure changed')
    if before!={n:(v.data_ptr(),v._version,payload(v)) for n,v in weights.items()} or sha(root/'fit/fit.json')!=read(root/'coefficient-seal.json')['fitReceiptSHA256']:raise ValueError('Sealed parameters changed')
    if len(read_rows)!=46 or len(operations)!=322 or len(attention)!=46 or len(vae_ops)!=80:raise ValueError('Declared physical work differs')
    assets_unchanged(assets)
    save(root/'images.json',{'schema':'tce-source-native-images-v1','requestSHA256':r['requestSHA256'],'fields':fields,'images':images,'common':main_common,'reads':read_rows,
        'actual':actual,'writerBindings':bindings,'linears':operations,'attentionCalls':attention,'VAEOperations':vae_ops,'loadedParameters':loaded,'assets':assets,'numericalProfile':numerical,
        'coefficientSealSHA256':sha(root/'coefficient-seal.json'),'servingSHA256':sha(root/'serving.json'),'fitReceiptSHA256':sha(root/'fit/fit.json'),
        'timings':timings,'workerSecondsAfterImports':time.perf_counter()-started,'cpuSecondsAfterImports':time.process_time()-cpu,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),
        'scope':'46source-nativeimageattentionreads,one discovery-onlyclosedformfit,30isolatedservedwriteroutputs,2originalVAEdecodes. Fullnativeencoder/denoiser calls0; all7originalaffines/read andfittedmap remainpaid.'})
