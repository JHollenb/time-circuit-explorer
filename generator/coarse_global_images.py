"""Source-defined coarse conditional read and full-resolution innovation commit."""
from pathlib import Path
import time
from encoder_atoms import np,read,save,sha
from paired_fields import load,payload,retain
from flux_clock import header,partition,selected_load,install_tables,snapshot,CLOCK_PREFIXES


def execute(root):
    from flux_clock_study import verify,source_checks,assets_unchanged
    from two_clock import install_guard
    from coarse_global import geometry,mask,pool_torch,lift_torch,check
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json');install_guard(root)
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native_module
    from diffusers import Flux2KleinPipeline,Flux2Transformer2DModel,AutoencoderKLFlux2,FlowMatchEulerDiscreteScheduler
    from complete_masked_study import profile
    from last_velocity_decode import scheduler
    tick=time.perf_counter();cpu=time.process_time();numerical=profile(torch,4)
    if numerical!=s['numericalProfile']:raise ValueError('Qualified native profile differs')
    model,assets=source_checks(s,diffusers);table=s['clockTable'];phase={};clock_calls=[]
    if table['contract']['numericalProfile']!=numerical or table['contract']['schedule']!=s['schedule'] or table['contract']['checkpointSHA256']!=s['imageAssets']['transformer/diffusion_pytorch_model.safetensors']:
        raise ValueError('Clock/source binding differs')
    config=read(model/'transformer/config.json');entries,h=header(model/'transformer/diffusion_pytorch_model.safetensors');removed,kept=partition(entries)
    if h!=table['header'] or removed!=table['removedParameterNames'] or kept!=table['remainingParameterNames']:raise ValueError('Parameter partition differs')
    with torch.device('meta'):transformer=Flux2Transformer2DModel.from_config(config)
    install_tables(transformer,table,'cuda',phase,clock_calls)
    reads=selected_load(transformer,model/'transformer/diffusion_pytorch_model.safetensors',kept,entries,'cuda')
    if any(n.startswith(CLOCK_PREFIXES) for n,_ in transformer.named_parameters()):raise ValueError('Compiled clock parameters loaded')
    vae=AutoencoderKLFlux2.from_pretrained(str(model/'vae'),torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True,device_map={'':'cuda'})
    transformer.eval().requires_grad_(False);vae.eval().requires_grad_(False)
    pipe=Flux2KleinPipeline(scheduler=None,vae=vae,transformer=transformer,text_encoder=None,tokenizer=None,is_distilled=True)
    def processors():return sorted({type(m.processor).__name__ for m in transformer.modules() if hasattr(m,'processor')})
    if processors()!=s['attentionProcessors'] or transformer.is_cache_enabled or getattr(transformer,'_hf_hook',None) is not None:raise ValueError('Attention/cache mode differs')
    def census():
        return {prefix+'.'+n:[v.data_ptr(),v._version,list(v.shape),str(v.dtype)] for prefix,m in [('transformer',transformer),('vae',vae)] for n,v in [*m.named_parameters(),*m.named_buffers()]}
    before=census();constants={n:payload(v) for n,v in transformer.named_buffers()};ids=load(s['ids'],'cuda')
    coarse_ids=torch.from_numpy(geometry(load(s['ids']))).to('cuda');key_mask=torch.from_numpy(mask()).to('cuda')
    common={};fields={};images={};actual=[];timings=[];linears=[];vae_ops=[];counts={};hooks=[];expected={};attention_calls=[]
    names={p.data_ptr():prefix+'.'+n for prefix,m in [('transformer',transformer),('vae',vae)] for n,p in m.named_parameters()}
    old_linear=torch.nn.functional.linear;old_dispatch=native_module.dispatch_attention_fn
    def linear(x,weight,bias=None):
        n=names.get(weight.data_ptr())
        if n is None:raise ValueError('Unidentified affine operand')
        linears.append({**phase,'parameter':n,'inputShape':list(x.shape),'weightShape':list(weight.shape),'nominalMACs':x.numel()//x.shape[-1]*weight.numel()})
        return old_linear(x,weight,bias)
    def dispatch(q,k,v,*args,**kwargs):
        m=kwargs.get('attn_mask')
        if list(q.shape)!=[1,768,24,128] or k.shape!=q.shape or v.shape!=q.shape or m is None or payload(m)!=payload(key_mask):
            raise ValueError('Actual attention geometry/multiplicity differs')
        attention_calls.append({**phase,'queryShape':list(q.shape),'keyShape':list(k.shape),'maskPayloadSHA256':payload(m)})
        return old_dispatch(q,k,v,*args,**kwargs)
    def native_pre(module,args,kwargs):
        if phase['policy']!='coarse' or phase['step']!=3:raise ValueError('Unexpected coarse denoiser')
        check(kwargs,snapshot(torch),table['contract'],expected)
        actual.append({**phase,'kind':'coarse-input',**{n+'PayloadSHA256':payload(kwargs[n]) for n in ['hidden_states','encoder_hidden_states','timestep','img_ids','txt_ids']},
            'maskPayloadSHA256':payload(kwargs['joint_attention_kwargs']['attention_mask'])})
        counts[phase['case']]['denoiser']+=1
    def context_pre(module,args):
        if payload(args[0])!=s['prefix'][phase['case']]['E']['payloadSHA256']:raise ValueError('Wrong actual context')
        counts[phase['case']]['context']+=1
    def context_post(module,args,y):fields[phase['case']]['C3']=retain(root/'arrays/images'/phase['case'],'C3',y)
    def block(kind):
        def hook(module,args):counts[phase['case']][kind]+=1
        return hook
    def vae_op(name):
        def hook(module,args,y):
            vae_ops.append({**phase,'module':name,'kind':type(module).__name__,'inputShape':list(args[0].shape),'outputShape':list(y.shape),'nominalMACs':y.numel()*(module.weight.numel()//module.weight.shape[0])})
        return hook
    hooks.extend([transformer.register_forward_pre_hook(native_pre,with_kwargs=True),transformer.context_embedder.register_forward_pre_hook(context_pre),transformer.context_embedder.register_forward_hook(context_post)])
    for m in transformer.transformer_blocks:hooks.append(m.register_forward_pre_hook(block('joint')))
    for m in transformer.single_transformer_blocks:hooks.append(m.register_forward_pre_hook(block('single')))
    for n,m in vae.named_modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):hooks.append(m.register_forward_hook(vae_op(n)))
    setup=time.perf_counter()-tick;torch.nn.functional.linear=linear;native_module.dispatch_attention_fn=dispatch
    try:
        with torch.inference_mode():
            for c,fs in s['prefix'].items():
                counts[c]={'denoiser':0,'context':0,'joint':0,'single':0,'VAE':0};phase.update(case=c,policy='coarse',step=3)
                z3=load(fs['latent3'],'cuda');v1=load(fs['velocity1'],'cuda');e=load(fs['E'],'cuda')
                common[c]={'latent3':fs['latent3'],'txtIDs':retain(root/'arrays/common'/c,'txtIDs',pipe._prepare_text_ids(e).to('cuda'))}
                fields[c]={'latent3':fs['latent3']};f=fields[c];pooled=pool_torch(z3);pv1=pool_torch(v1);zc=pooled.to(torch.bfloat16);text_ids=load(common[c]['txtIDs'],'cuda')
                t=load(table['rows'][3]['clockOperands']['transformer_argument'],'cuda')
                for n,a in [('pooledLatentRaw',pooled),('pooledLatent',zc),('pooledHeldVelocity',pv1),('coarseIDs',coarse_ids),('keyLogweights',key_mask)]:f[n]=retain(root/'arrays/images'/c,n,a)
                expected.update({n:payload(a) for n,a in [('hidden_states',zc),('encoder_hidden_states',e),('img_ids',coarse_ids),('txt_ids',text_ids),('timestep',t),('attention_mask',key_mask)]})
                torch.cuda.synchronize();start=time.perf_counter()
                with transformer.cache_context('cond'):
                    vc=transformer(hidden_states=zc,timestep=t,guidance=None,encoder_hidden_states=e,txt_ids=text_ids,img_ids=coarse_ids,
                        joint_attention_kwargs={'attention_mask':key_mask},return_dict=False)[0]
                torch.cuda.synchronize();timings.append({**phase,'kind':'coarseDenoiser','wallSeconds':time.perf_counter()-start})
                innovation=lift_torch(vc.float()-pv1);raw=v1.float()+innovation;velocity=raw.to(torch.bfloat16)
                for n,a in [('coarseVelocity',vc),('liftedInnovation',innovation),('rawVelocity',raw),('velocity3',velocity)]:f[n]=retain(root/'arrays/images'/c,n,a)
                final=scheduler(s,torch,FlowMatchEulerDiscreteScheduler);z4=final.step(velocity,final.timesteps[3],z3,return_dict=False)[0];f['latent4']=retain(root/'arrays/images'/c,'latent4',z4)
                actual.append({**phase,'kind':'final-commit','incomingPayloadSHA256':payload(z3),'velocityPayloadSHA256':payload(velocity),'outgoingPayloadSHA256':payload(z4),'finalStepIndex':final.step_index})
                z=pipe._unpack_latents_with_ids(z4,ids,32,32);mean=vae.bn.running_mean.view(1,-1,1,1).to(z.device,z.dtype)
                std=torch.sqrt(vae.bn.running_var.view(1,-1,1,1)+vae.config.batch_norm_eps).to(z.device,z.dtype)
                z=pipe._unpatchify_latents(z*std+mean);f['vaeInput']=retain(root/'arrays/images'/c,'vaeInput',z)
                torch.cuda.synchronize();start=time.perf_counter();decoded=vae.decode(z,return_dict=False)[0];torch.cuda.synchronize()
                counts[c]['VAE']+=1;timings.append({**phase,'kind':'VAE','wallSeconds':time.perf_counter()-start});f['decodedRaw']=retain(root/'arrays/images'/c,'decodedRaw',decoded)
                rgb=pipe.image_processor.postprocess(decoded,output_type='np');png=pipe.image_processor.numpy_to_pil(rgb)[0];path=root/('coarse-'+c+'.png');png.save(path)
                f['rgbFloat']=retain(root/'arrays/images'/c,'rgbFloat',rgb);f['pngRGB']=retain(root/'arrays/images'/c,'pngRGB',np.asarray(png));images[c]={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
                if payload(z3)!=fs['latent3']['payloadSHA256'] or final.step_index!=4 or rgb.shape!=(1,512,512,3):raise ValueError('Actual final state differs')
                print('COARSE_GLOBAL_IMAGE',c,counts[c],flush=True)
    finally:
        torch.nn.functional.linear=old_linear;native_module.dispatch_attention_fn=old_dispatch
        for hook in hooks:hook.remove()
    if census()!=before or constants!={n:payload(v) for n,v in transformer.named_buffers()} or processors()!=s['attentionProcessors']:raise ValueError('Consumer mutable closure changed')
    if any(x!={'denoiser':1,'context':1,'joint':5,'single':20,'VAE':1} for x in counts.values()) or len(clock_calls)!=10 or len(attention_calls)!=50:raise ValueError('Actual work instrument differs')
    assets_unchanged(assets)
    save(root/'images.json',{'schema':'tce-coarse-global-images-v1','requestSHA256':r['requestSHA256'],'common':common,'fields':fields,'images':images,'actual':actual,
        'counts':counts,'timings':timings,'linears':linears,'VAEOperations':vae_ops,'attentionCalls':attention_calls,'clockCalls':clock_calls,'loadedParameters':reads,
        'removedParameters':removed,'assets':assets,'numericalProfile':numerical,'attentionProcessors':processors(),'remainingParameterBytesLoaded':sum(x['bytes'] for x in reads),
        'constantBufferPayloads':constants,'setupSecondsAfterImports':setup,'workerSecondsAfterImports':time.perf_counter()-tick,'cpuSecondsAfterImports':time.process_time()-cpu,
        'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'scope':'Source-defined256image/512text coarseglobal read;2originalVAEdecodes. Original learnedmaps/prefixes remain paid.'})
