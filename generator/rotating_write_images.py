"""Complete-state denoiser with rotating original residual writes."""
from pathlib import Path
import time
from encoder_atoms import np,read,save,sha
from paired_fields import load,payload,retain
from flux_clock import header,partition,selected_load,install_tables,snapshot,check_mode,CLOCK_PREFIXES


def execute(root):
    from flux_clock_study import verify,source_checks,assets_unchanged
    from two_clock import install_guard
    from rotating_writes import image_rows,joint,single
    import types
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
    h,w=np.indices((32,32));expected_ids=np.stack([h*0,h,w,h*0],-1).reshape(1,1024,4)
    if not np.array_equal(load(s['ids']),expected_ids):raise ValueError('Original image row/position binding differs')
    common={};fields={};images={};actual=[];timings=[];linears=[];vae_ops=[];counts={};hooks=[];expected={};attention_calls=[];routes=[];old_forwards=[]
    names={p.untyped_storage().data_ptr():prefix+'.'+n for prefix,m in [('transformer',transformer),('vae',vae)] for n,p in m.named_parameters()}
    old_linear=torch.nn.functional.linear;old_dispatch=native_module.dispatch_attention_fn
    def linear(x,weight,bias=None):
        n=names.get(weight.untyped_storage().data_ptr())
        if n is None:raise ValueError('Unidentified affine operand')
        binding=dict(phase)
        if '.transformer_blocks.' not in n and '.single_transformer_blocks.' not in n:
            binding.pop('blockIndex',None);binding.pop('blockKind',None)
        linears.append({**binding,'parameter':n,'inputShape':list(x.shape),'weightShape':list(weight.shape),'weightStride':list(weight.stride()),'weightStorageOffsetElements':weight.storage_offset(),'biasPresent':bias is not None,'nominalMACs':x.numel()//x.shape[-1]*weight.numel()})
        return old_linear(x,weight,bias)
    def dispatch(q,k,v,*args,**kwargs):
        m=kwargs.get('attn_mask')
        if list(q.shape)!=[1,768,24,128] or list(k.shape)!=[1,1536,24,128] or v.shape!=k.shape or m is not None:
            raise ValueError('Actual rectangular attention geometry differs')
        attention_calls.append({**phase,'queryShape':list(q.shape),'keyShape':list(k.shape),'valueShape':list(v.shape),
            'queryDtype':str(q.dtype),'keyDtype':str(k.dtype),'valueDtype':str(v.dtype),
            'queryStride':list(q.stride()),'keyStride':list(k.stride()),'valueStride':list(v.stride()),
            'requestedBackend':'auto' if kwargs.get('backend') is None else str(kwargs['backend']),'mask':None})
        return old_dispatch(q,k,v,*args,**kwargs)
    def native_pre(module,args,kwargs):
        if phase['policy']!='routed' or phase['step']!=3:raise ValueError('Unexpected routed denoiser')
        check_mode(kwargs,snapshot(torch),table['contract'])
        for n,h in expected.items():
            if payload(kwargs[n])!=h:raise ValueError('Actual complete denoiser input differs: '+n)
        actual.append({**phase,'kind':'routed-input',**{n+'PayloadSHA256':payload(kwargs[n]) for n in ['hidden_states','encoder_hidden_states','timestep','img_ids','txt_ids']}})
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
    def install_route(module,index,kind):
        old_forwards.append((module,module.forward))
        selected_image=torch.tensor(image_rows(index),device='cuda')
        selected=torch.cat([torch.arange(512,device='cuda'),512+selected_image])
        untouched_image=torch.tensor(np.setdiff1d(np.arange(1024),image_rows(index)),device='cuda')
        untouched=512+untouched_image
        def witness(before,after):
            a=before.index_select(1,untouched);b=after.index_select(1,untouched)
            if not torch.equal(a,b):raise ValueError('Unselected carried state changed')
            routes.append({**phase,'selectedImageRows':image_rows(index).tolist(),'selectedGlobalRowsPayloadSHA256':payload(selected),
                'fullInputStatePayloadSHA256':payload(before),'fullOutputStatePayloadSHA256':payload(after),
                'unchangedInputPayloadSHA256':payload(a),'unchangedOutputPayloadSHA256':payload(b),
                'unselectedCarriedRows':768,'fullImageRows':1024,'fullTextRows':512,'selectedQueryRows':768,'readKeyValueRows':1536})
        if kind=='joint':
            def forward(self,hidden_states,encoder_hidden_states,temb_mod_img,temb_mod_txt,image_rotary_emb=None,joint_attention_kwargs=None):
                if joint_attention_kwargs not in (None,{}):raise ValueError('Unexpected attention override')
                phase.update(blockIndex=index,blockKind=kind)
                text,image=joint(self,hidden_states,encoder_hidden_states,temb_mod_img,temb_mod_txt,image_rotary_emb,selected_image,native_module)
                witness(torch.cat([encoder_hidden_states,hidden_states],1),torch.cat([text,image],1))
                return text,image
        else:
            def forward(self,hidden_states,encoder_hidden_states,temb_mod,image_rotary_emb=None,joint_attention_kwargs=None,split_hidden_states=False,text_seq_len=None):
                if encoder_hidden_states is not None or split_hidden_states or text_seq_len is not None or joint_attention_kwargs not in (None,{}):
                    raise ValueError('Unsupported single-block mode')
                phase.update(blockIndex=index,blockKind=kind)
                result=single(self,hidden_states,temb_mod,image_rotary_emb,selected,native_module)
                witness(hidden_states,result)
                return result
        module.forward=types.MethodType(forward,module)
    for i,m in enumerate(transformer.transformer_blocks):install_route(m,i,'joint')
    for i,m in enumerate(transformer.single_transformer_blocks):install_route(m,i+5,'single')
    setup=time.perf_counter()-tick;torch.nn.functional.linear=linear;native_module.dispatch_attention_fn=dispatch
    try:
        with torch.inference_mode():
            for c,fs in s['prefix'].items():
                counts[c]={'denoiser':0,'context':0,'joint':0,'single':0,'VAE':0};phase.update(case=c,policy='routed',step=3)
                z3=load(fs['latent3'],'cuda');e=load(fs['E'],'cuda')
                common[c]={'latent3':fs['latent3'],'txtIDs':retain(root/'arrays/common'/c,'txtIDs',pipe._prepare_text_ids(e).to('cuda'))}
                fields[c]={'latent3':fs['latent3']};f=fields[c];text_ids=load(common[c]['txtIDs'],'cuda')
                t=load(table['rows'][3]['clockOperands']['transformer_argument'],'cuda')
                expected.update({n:payload(a) for n,a in [('hidden_states',z3),('encoder_hidden_states',e),('img_ids',ids),('txt_ids',text_ids),('timestep',t)]})
                torch.cuda.synchronize();start=time.perf_counter()
                with transformer.cache_context('cond'):
                    velocity=transformer(hidden_states=z3,timestep=t,guidance=None,encoder_hidden_states=e,txt_ids=text_ids,img_ids=ids,
                        joint_attention_kwargs=None,return_dict=False)[0]
                torch.cuda.synchronize();timings.append({**phase,'kind':'routedDenoiserWithWitnesses','wallSeconds':time.perf_counter()-start})
                phase.pop('blockIndex',None);phase.pop('blockKind',None)
                f['velocity3']=retain(root/'arrays/images'/c,'velocity3',velocity)
                final=scheduler(s,torch,FlowMatchEulerDiscreteScheduler);z4=final.step(velocity,final.timesteps[3],z3,return_dict=False)[0];f['latent4']=retain(root/'arrays/images'/c,'latent4',z4)
                actual.append({**phase,'kind':'final-commit','incomingPayloadSHA256':payload(z3),'velocityPayloadSHA256':payload(velocity),'outgoingPayloadSHA256':payload(z4),'finalStepIndex':final.step_index})
                z=pipe._unpack_latents_with_ids(z4,ids,32,32);mean=vae.bn.running_mean.view(1,-1,1,1).to(z.device,z.dtype)
                std=torch.sqrt(vae.bn.running_var.view(1,-1,1,1)+vae.config.batch_norm_eps).to(z.device,z.dtype)
                z=pipe._unpatchify_latents(z*std+mean);f['vaeInput']=retain(root/'arrays/images'/c,'vaeInput',z)
                torch.cuda.synchronize();start=time.perf_counter();decoded=vae.decode(z,return_dict=False)[0];torch.cuda.synchronize()
                counts[c]['VAE']+=1;timings.append({**phase,'kind':'VAE','wallSeconds':time.perf_counter()-start});f['decodedRaw']=retain(root/'arrays/images'/c,'decodedRaw',decoded)
                rgb=pipe.image_processor.postprocess(decoded,output_type='np');png=pipe.image_processor.numpy_to_pil(rgb)[0];path=root/('routed-'+c+'.png');png.save(path)
                f['rgbFloat']=retain(root/'arrays/images'/c,'rgbFloat',rgb);f['pngRGB']=retain(root/'arrays/images'/c,'pngRGB',np.asarray(png));images[c]={'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
                if payload(z3)!=fs['latent3']['payloadSHA256'] or final.step_index!=4 or rgb.shape!=(1,512,512,3):raise ValueError('Actual final state differs')
                print('ROTATING_WRITE_IMAGE',c,counts[c],flush=True)
    finally:
        torch.nn.functional.linear=old_linear;native_module.dispatch_attention_fn=old_dispatch
        for hook in hooks:hook.remove()
        for module,forward in old_forwards:module.forward=forward
    if census()!=before or constants!={n:payload(v) for n,v in transformer.named_buffers()} or processors()!=s['attentionProcessors']:raise ValueError('Consumer mutable closure changed')
    if any(x!={'denoiser':1,'context':1,'joint':5,'single':20,'VAE':1} for x in counts.values()) or len(clock_calls)!=10 or len(attention_calls)!=50 or len(routes)!=50:raise ValueError('Actual work instrument differs')
    if len([x for x in linears if x['parameter'].startswith('transformer.')])!=286 or sum(x['nominalMACs'] for x in linears if x['parameter'].startswith('transformer.'))!=5461587787776:
        raise ValueError('Physical sliced affine census differs')
    assets_unchanged(assets)
    save(root/'images.json',{'schema':'tce-rotating-write-images-v1','requestSHA256':r['requestSHA256'],'common':common,'fields':fields,'images':images,'actual':actual,
        'counts':counts,'timings':timings,'linears':linears,'VAEOperations':vae_ops,'attentionCalls':attention_calls,'routes':routes,'clockCalls':clock_calls,'loadedParameters':reads,
        'removedParameters':removed,'assets':assets,'numericalProfile':numerical,'attentionProcessors':processors(),'remainingParameterBytesLoaded':sum(x['bytes'] for x in reads),
        'constantBufferPayloads':constants,'setupSecondsAfterImports':setup,'workerSecondsAfterImports':time.perf_counter()-tick,'cpuSecondsAfterImports':time.process_time()-cpu,
        'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'scope':'Full1024image512textownstate;globalcurrentKV;rotating256image+alltextwrites;2learneddenoisers/2VAE. Originalmaps/prefixes/encodingremainpaid. Timingincludesstatehashwitnesses.'})
