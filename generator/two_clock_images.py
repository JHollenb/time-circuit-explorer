"""One shared omitted-write state, then HH and native-on-own-state HN."""
from pathlib import Path
import time
from encoder_atoms import np,read,save,sha
from paired_fields import load,payload,retain
from flux_clock import header,partition,selected_load,install_tables,check_mode,snapshot,CLOCK_PREFIXES


def execute(root):
    from flux_clock_study import verify,source_checks,assets_unchanged
    from two_clock import install_guard
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json')
    install_guard(root)
    import torch
    import diffusers
    from diffusers import Flux2KleinPipeline,Flux2Transformer2DModel,AutoencoderKLFlux2,FlowMatchEulerDiscreteScheduler
    from complete_masked_study import profile
    from last_velocity_decode import scheduler
    tick=time.perf_counter();cpu=time.process_time();numerical=profile(torch,4)
    if numerical!=s['numericalProfile']:raise ValueError('Qualified native profile differs')
    model,assets=source_checks(s,diffusers);table=s['clockTable'];phase={};clock_calls=[]
    if table['contract']['numericalProfile']!=numerical or table['contract']['schedule']!=s['schedule'] or table['contract']['checkpointSHA256']!=s['imageAssets']['transformer/diffusion_pytorch_model.safetensors']:
        raise ValueError('Clock/source binding differs')
    config=read(model/'transformer/config.json');entries,h=header(model/'transformer/diffusion_pytorch_model.safetensors');removed,kept=partition(entries)
    if h!=table['header'] or removed!=table['removedParameterNames'] or kept!=table['remainingParameterNames']:
        raise ValueError('Original parameter partition differs')
    with torch.device('meta'):transformer=Flux2Transformer2DModel.from_config(config)
    install_tables(transformer,table,'cuda',phase,clock_calls)
    reads=selected_load(transformer,model/'transformer/diffusion_pytorch_model.safetensors',kept,entries,'cuda')
    if any(n.startswith(CLOCK_PREFIXES) for n,_ in transformer.named_parameters()):raise ValueError('Compiled clock parameters loaded')
    vae=AutoencoderKLFlux2.from_pretrained(str(model/'vae'),torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True,device_map={'':'cuda'})
    transformer.eval().requires_grad_(False);vae.eval().requires_grad_(False)
    pipe=Flux2KleinPipeline(scheduler=None,vae=vae,transformer=transformer,text_encoder=None,tokenizer=None,is_distilled=True)
    def processors():return sorted({type(m.processor).__name__ for m in transformer.modules() if hasattr(m,'processor')})
    if processors()!=s['attentionProcessors'] or transformer.is_cache_enabled or getattr(transformer,'_hf_hook',None) is not None:
        raise ValueError('Native attention/cache mode differs')
    def census():
        return {prefix+'.'+n:[v.data_ptr(),v._version,list(v.shape),str(v.dtype)] for prefix,m in [('transformer',transformer),('vae',vae)] for n,v in [*m.named_parameters(),*m.named_buffers()]}
    before=census();constants={n:payload(v) for n,v in transformer.named_buffers()}
    ids=load(s['ids'],'cuda');common={};fields={};images={};actual=[];timings=[];linears=[];vae_ops=[];counts={};hooks=[]
    names={p.data_ptr():prefix+'.'+n for prefix,m in [('transformer',transformer),('vae',vae)] for n,p in m.named_parameters()}
    old_linear=torch.nn.functional.linear
    def linear(x,weight,bias=None):
        n=names.get(weight.data_ptr())
        if n is None:raise ValueError('Unidentified affine operand')
        linears.append({**phase,'parameter':n,'inputShape':list(x.shape),'weightShape':list(weight.shape),'nominalMACs':x.numel()//x.shape[-1]*weight.numel()})
        return old_linear(x,weight,bias)
    def native_pre(module,args,kwargs):
        if phase['policy']!='HN' or phase['step']!=3:raise ValueError('Unexpected learned denoiser call')
        check_mode(kwargs,snapshot(torch),table['contract'])
        c=phase['case'];f=s['prefix'][c]
        if payload(kwargs['hidden_states'])!=common[c]['latent3']['payloadSHA256'] or payload(kwargs['encoder_hidden_states'])!=f['E']['payloadSHA256']:
            raise ValueError('Native reader received another state/context')
        clock=table['rows'][3]['clockOperands']['transformer_argument']
        if payload(kwargs['timestep'])!=clock['payloadSHA256'] or payload(kwargs['img_ids'])!=s['ids']['payloadSHA256'] or payload(kwargs['txt_ids'])!=common[c]['txtIDs']['payloadSHA256']:
            raise ValueError('Actual native clock/positions differ')
        actual.append({**phase,'kind':'native-input','latentPayloadSHA256':payload(kwargs['hidden_states']),
            'EPayloadSHA256':payload(kwargs['encoder_hidden_states']),'clockArgumentPayloadSHA256':payload(kwargs['timestep']),
            'imgIDsPayloadSHA256':payload(kwargs['img_ids']),'txtIDsPayloadSHA256':payload(kwargs['txt_ids'])})
        counts[c]['denoiser']+=1
    def context_pre(module,args):
        if payload(args[0])!=s['prefix'][phase['case']]['E']['payloadSHA256']:raise ValueError('Wrong actual context read')
        counts[phase['case']]['context']+=1
    def context_post(module,args,y):fields['HN-'+phase['case']]['C3']=retain(root/'arrays/images'/('HN-'+phase['case']),'C3',y)
    def block(kind):
        def hook(module,args):counts[phase['case']][kind]+=1
        return hook
    def vae_op(name):
        def hook(module,args,y):
            vae_ops.append({**phase,'module':name,'kind':type(module).__name__,'inputShape':list(args[0].shape),
                'outputShape':list(y.shape),'nominalMACs':y.numel()*(module.weight.numel()//module.weight.shape[0])})
        return hook
    hooks.extend([transformer.register_forward_pre_hook(native_pre,with_kwargs=True),
        transformer.context_embedder.register_forward_pre_hook(context_pre),transformer.context_embedder.register_forward_hook(context_post)])
    for m in transformer.transformer_blocks:hooks.append(m.register_forward_pre_hook(block('joint')))
    for m in transformer.single_transformer_blocks:hooks.append(m.register_forward_pre_hook(block('single')))
    for n,m in vae.named_modules():
        if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):hooks.append(m.register_forward_hook(vae_op(n)))
    setup=time.perf_counter()-tick;torch.nn.functional.linear=linear
    try:
        with torch.inference_mode():
            for c,fs in s['prefix'].items():
                counts[c]={'denoiser':0,'context':0,'joint':0,'single':0,'VAE':0};phase.update(case=c,policy='shared',step=2)
                z2=load(fs['latent2'],'cuda');v1=load(fs['velocity1'],'cuda');e=load(fs['E'],'cuda')
                sch=scheduler(s,torch,FlowMatchEulerDiscreteScheduler);sch.set_begin_index(2)
                torch.cuda.synchronize();start=time.perf_counter();z3=sch.step(v1,sch.timesteps[2],z2,return_dict=False)[0];torch.cuda.synchronize()
                timings.append({**phase,'kind':'sharedCommit','wallSeconds':time.perf_counter()-start})
                common[c]={'latent3':retain(root/'arrays/common'/c,'latent3',z3),
                    'txtIDs':retain(root/'arrays/common'/c,'txtIDs',pipe._prepare_text_ids(e).to('cuda'))}
                actual.append({**phase,'kind':'shared-commit','incomingPayloadSHA256':payload(z2),'velocityPayloadSHA256':payload(v1),'outgoingPayloadSHA256':payload(z3),'finalStepIndex':sch.step_index})
                for policy in r['policies']:
                    key=policy+'-'+c;phase.update(policy=policy,step=3);fields[key]={'latent3':common[c]['latent3']}
                    if policy=='HH':velocity=v1
                    elif policy=='HN':
                        t=load(table['rows'][3]['clockOperands']['transformer_argument'],'cuda');text_ids=load(common[c]['txtIDs'],'cuda')
                        torch.cuda.synchronize();start=time.perf_counter()
                        with transformer.cache_context('cond'):
                            velocity=transformer(hidden_states=z3.to(transformer.dtype),timestep=t,guidance=None,
                                encoder_hidden_states=e,txt_ids=text_ids,img_ids=ids,joint_attention_kwargs=None,return_dict=False)[0]
                        velocity=velocity[:,:z3.size(1):];torch.cuda.synchronize()
                        timings.append({**phase,'kind':'nativeDenoiser','wallSeconds':time.perf_counter()-start})
                    else:raise ValueError('Unknown policy')
                    fields[key]['velocity3']=retain(root/'arrays/images'/key,'velocity3',velocity)
                    final=scheduler(s,torch,FlowMatchEulerDiscreteScheduler)
                    z4=final.step(velocity,final.timesteps[3],z3,return_dict=False)[0]
                    fields[key]['latent4']=retain(root/'arrays/images'/key,'latent4',z4)
                    actual.append({**phase,'kind':'final-commit','incomingPayloadSHA256':payload(z3),'velocityPayloadSHA256':payload(velocity),'outgoingPayloadSHA256':payload(z4),'finalStepIndex':final.step_index})
                    z=pipe._unpack_latents_with_ids(z4,ids,32,32)
                    mean=vae.bn.running_mean.view(1,-1,1,1).to(z.device,z.dtype)
                    std=torch.sqrt(vae.bn.running_var.view(1,-1,1,1)+vae.config.batch_norm_eps).to(z.device,z.dtype)
                    z=pipe._unpatchify_latents(z*std+mean);fields[key]['vaeInput']=retain(root/'arrays/images'/key,'vaeInput',z)
                    torch.cuda.synchronize();start=time.perf_counter();decoded=vae.decode(z,return_dict=False)[0];torch.cuda.synchronize()
                    counts[c]['VAE']+=1;timings.append({**phase,'kind':'VAE','wallSeconds':time.perf_counter()-start})
                    fields[key]['decodedRaw']=retain(root/'arrays/images'/key,'decodedRaw',decoded)
                    rgb=pipe.image_processor.postprocess(decoded,output_type='np');png=pipe.image_processor.numpy_to_pil(rgb)[0];path=root/(key+'.png');png.save(path)
                    fields[key]['rgbFloat']=retain(root/'arrays/images'/key,'rgbFloat',rgb);fields[key]['pngRGB']=retain(root/'arrays/images'/key,'pngRGB',np.asarray(png))
                    images[key]={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
                    if payload(z3)!=common[c]['latent3']['payloadSHA256'] or final.step_index!=4 or rgb.shape!=(1,512,512,3):
                        raise ValueError('Shared parent mutated or final geometry differs')
                    print('TWO_CLOCK_IMAGE',key,counts[c],flush=True)
    finally:
        torch.nn.functional.linear=old_linear
        for hook in hooks:hook.remove()
    if census()!=before or constants!={n:payload(v) for n,v in transformer.named_buffers()} or processors()!=s['attentionProcessors']:
        raise ValueError('Image-consumer mutable closure changed')
    if any(x!={'denoiser':1,'context':1,'joint':5,'single':20,'VAE':2} for x in counts.values()) or len(clock_calls)!=10:
        raise ValueError('Physical work instrument differs')
    assets_unchanged(assets)
    save(root/'images.json',{'schema':'tce-two-clock-images-v1','requestSHA256':r['requestSHA256'],
        'common':common,'fields':fields,'images':images,'actual':actual,'counts':counts,'timings':timings,
        'linears':linears,'VAEOperations':vae_ops,'clockCalls':clock_calls,'loadedParameters':reads,
        'removedParameters':removed,'assets':assets,'numericalProfile':numerical,'attentionProcessors':processors(),
        'remainingParameterBytesLoaded':sum(x['bytes'] for x in reads),'constantBufferPayloads':constants,
        'setupSecondsAfterImports':setup,'workerSecondsAfterImports':time.perf_counter()-tick,
        'cpuSecondsAfterImports':time.process_time()-cpu,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),
        'scope':'Two original clock3 denoisers on actual omitted-clock2 states; four original VAE decodes. Earlier learned prefixes supplied and paid.'})
