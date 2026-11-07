"""Four own sampler clocks, image-only terminal rewrite, one original VAE."""
from pathlib import Path
import sys
import time
import types
from encoder_atoms import read,save,sha,identity,np


def execute(root):
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from diffusers import Flux2Transformer2DModel,AutoencoderKLFlux2,Flux2KleinPipeline,FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2 import pipeline_flux2_klein
    from paired_fields import load,retain,payload,array
    from terminal_trajectory import denoise,terminal_maps,mechanics
    from terminal_trajectory_ledger import Ledger
    from terminal_capture_guard_v2 import quarantine
    from flux_clock_study import verify,source_checks,assets_unchanged
    from flux_clock import header,selected_load,snapshot
    from complete_masked_study import profile
    root=Path(root).resolve();r,s=read(root/'request.json'),read(root/'inputs.json');verify(root,r)
    started=time.perf_counter();cpu=time.process_time();out=root/'response-result';out.mkdir();save(out/'toy-mechanics.json',mechanics())
    allowed=[s['E']['path']]+[s['noise'][n]['path'] for n in ['raw','packed','ids']]+[str(Path(s['modelRoot'])/n) for n in s['imageAssets']]
    access=quarantine(root,allowed,s['deniedNumericalInputs'],root/'unused-output-store');numerical=profile(torch,4)
    if numerical!=s['numericalProfile']:raise ValueError('Original image numeric profile differs')
    torch.cuda.reset_peak_memory_stats();model,assets=source_checks(s,diffusers);config=read(model/'transformer/config.json')
    if {n:config[n] for n in ['num_layers','num_single_layers','num_attention_heads','attention_head_dim','mlp_ratio','in_channels','patch_size']}!={'num_layers':5,'num_single_layers':20,'num_attention_heads':24,'attention_head_dim':128,'mlp_ratio':3.,'in_channels':128,'patch_size':1} or config['guidance_embeds'] is not False:
        raise ValueError('Ordinary native transformer geometry differs')
    entries,hr=header(model/'transformer/diffusion_pytorch_model.safetensors')
    with torch.device('meta'):transformer=Flux2Transformer2DModel.from_config(config)
    loaded=selected_load(transformer,model/'transformer/diffusion_pytorch_model.safetensors',sorted(entries),entries,'cuda')
    vae=AutoencoderKLFlux2.from_pretrained(str(model/'vae'),torch_dtype=torch.bfloat16,local_files_only=True,low_cpu_mem_usage=True,device_map={'':'cuda'})
    transformer.eval().requires_grad_(False);vae.eval().requires_grad_(False)
    processors=lambda:sorted({type(m.processor).__name__ for m in transformer.modules() if hasattr(m,'processor')})
    if processors()!=s['attentionProcessors'] or transformer.is_cache_enabled or getattr(transformer,'_hf_hook',None) is not None:raise ValueError('Original ordinary attention execution differs')
    scheduler_config=read(model/'scheduler/scheduler_config.json')
    if scheduler_config.get('stochastic_sampling',False):raise ValueError('Only deterministic native schedule supported')
    pipe=Flux2KleinPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler_config),vae=vae,text_encoder=None,tokenizer=None,
        transformer=transformer,is_distilled=read(model/'model_index.json')['is_distilled'])
    if not pipe.config.is_distilled:raise ValueError('Original distilled pipeline required')
    pipe.set_progress_bar_config(disable=True);maps=terminal_maps(transformer)
    physical=[{'name':n,'payloadSHA256':payload(v),'shape':list(v.shape),'bytes':v.numel()*v.element_size()} for n,v in maps.items()]
    parameter_manifest=identity({'transformer':loaded,'vaeCheckpointSHA256':s['imageAssets']['vae/diffusion_pytorch_model.safetensors'],'physicalTerminalMaps':physical})
    ledger=Ledger(r['requestSHA256'],parameter_manifest);phase={'step':0,'kind':'setup'};fields={};linears=[];convolutions=[];sdpa=[];dispatch=[];schedule_records=[];actual=[]
    counts={'denoiser':0,'joint':0,'singlePrefix':0,'reducedTerminal':0,'scheduler':0,'VAE':0};last={};handles=[]
    def census():
        return {prefix+'.'+n:[v.data_ptr(),v._version,list(v.shape),str(v.dtype)] for prefix,module in [('transformer',transformer),('vae',vae)] for n,v in [*module.named_parameters(),*module.named_buffers()]}
    before=census();buffers={prefix+'.'+n:payload(v) for prefix,module in [('transformer',transformer),('vae',vae)] for n,v in module.named_buffers()}
    weight_names={v.data_ptr():prefix+'.'+n for prefix,module in [('transformer',transformer),('vae',vae)] for n,v in module.named_parameters()}
    weight_names.update({maps[n].data_ptr():'terminalPhysical.'+n for n in ['KV','QGU']})
    raw_noise=load(s['noise']['raw'],'cuda');embedding=load(s['E'],'cuda');expected=s['noise']['packed']['payloadSHA256']
    if list(raw_noise.shape)!=[1,128,32,32] or list(embedding.shape)!=[1,512,7680]:raise ValueError('Authenticated source geometry differs')
    rng,rng_cuda=torch.get_rng_state(),torch.cuda.get_rng_state_all()
    old_linear=torch.nn.functional.linear;old_conv=torch.nn.functional.conv2d;old_transpose=torch.nn.functional.conv_transpose2d;old_sdpa=torch.nn.functional.scaled_dot_product_attention
    old_dispatch=native.dispatch_attention_fn;old_randn=pipeline_flux2_klein.randn_tensor;old_prepare=pipe.prepare_latents;old_step=pipe.scheduler.step;old_decode=vae.decode
    def linear(x,w,bias=None):
        name=weight_names.get(w.data_ptr())
        if name is None:raise ValueError('Undeclared learned trajectory affine')
        linears.append({**phase,'parameter':name,'inputShape':list(x.shape),'weightShape':list(w.shape),'nominalMACs':x.numel()//x.shape[-1]*w.numel(),'biasPresent':bias is not None})
        return old_linear(x,w,bias)
    def conv(x,w,bias=None,stride=1,padding=0,dilation=1,groups=1):
        name=weight_names.get(w.data_ptr())
        if name is None or phase['kind']!='decode':raise ValueError('Undeclared learned decoder convolution')
        value=old_conv(x,w,bias,stride,padding,dilation,groups)
        convolutions.append({**phase,'operator':'conv2d','parameter':name,'inputShape':list(x.shape),'weightShape':list(w.shape),'outputShape':list(value.shape),
            'nominalMACs':value.numel()*w.shape[1]*w.shape[2]*w.shape[3],'groups':groups,'stride':stride,'padding':padding,'dilation':dilation,'basis':'Dense output dot products, including nominal padded products.'})
        return value
    def transpose(x,w,bias=None,stride=1,padding=0,output_padding=0,groups=1,dilation=1):
        name=weight_names.get(w.data_ptr())
        if name is None or phase['kind']!='decode':raise ValueError('Undeclared learned decoder transposed convolution')
        value=old_transpose(x,w,bias,stride,padding,output_padding,groups,dilation)
        convolutions.append({**phase,'operator':'conv_transpose2d','parameter':name,'inputShape':list(x.shape),'weightShape':list(w.shape),'outputShape':list(value.shape),
            'nominalMACs':x.numel()*w.shape[1]*w.shape[2]*w.shape[3],'groups':groups,'basis':'Input scatter products before boundary cropping.'})
        return value
    def attention(q,k,v,*args,**kwargs):
        sdpa.append({**phase,'QShape':list(q.shape),'KShape':list(k.shape),'VShape':list(v.shape),
            'nominalQKAndAVMACs':q.numel()//q.shape[-1]*k.shape[-2]*(q.shape[-1]+v.shape[-1])})
        return old_sdpa(q,k,v,*args,**kwargs)
    def dispatch_attention(q,k,v,*args,**kwargs):
        dispatch.append({**phase,'QShape':list(q.shape),'KShape':list(k.shape),'VShape':list(v.shape),'nominalQKAndAVMACs':2*q.numel()*k.shape[1]})
        return old_dispatch(q,k,v,*args,**kwargs)
    def capture(name,value):fields[name]=retain(root/'arrays/candidate',name,value);return fields[name]
    def persist_progress():
        import os
        row={'step':phase['step'],'kind':phase['kind'],'typedActs':ledger.records[-2:],'fields':dict(fields),
            'cumulativeAffineMACs':sum(x['nominalMACs'] for x in linears),'cumulativeAttentionMACs':sum(x['nominalQKAndAVMACs'] for x in sdpa),'cumulativeConvolutionMACs':sum(x['nominalMACs'] for x in convolutions)}
        with (out/'progress.jsonl').open('a') as f:f.write(__import__('json').dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    def no_draw(*args,**kwargs):raise ValueError('Noise redraw prohibited')
    def prepare(*args,**kwargs):
        packed,ids=old_prepare(*args,**kwargs)
        if payload(packed)!=s['noise']['packed']['payloadSHA256'] or payload(ids)!=s['noise']['ids']['payloadSHA256']:raise ValueError('Native supplied initial state/IDs differ')
        actual.append({'kind':'initial-state','latentPayloadSHA256':payload(packed),'IDsPayloadSHA256':payload(ids)})
        return packed,ids
    def adapter(this,**kwargs):
        nonlocal expected
        phase['kind']='denoise';k=phase['step'];x=kwargs['hidden_states'];condition=kwargs['encoder_hidden_states'];time_arg=kwargs['timestep']
        if k not in range(4) or payload(x)!=expected or payload(condition)!=s['E']['payloadSHA256']:raise ValueError('Denoiser did not read its own preceding state/E')
        schedule={'timesteps':array(pipe.scheduler.timesteps).tolist(),'sigmas':array(pipe.scheduler.sigmas).tolist(),'timestepsPayloadSHA256':payload(pipe.scheduler.timesteps),'sigmasPayloadSHA256':payload(pipe.scheduler.sigmas)}
        if schedule!=s['schedule']:raise ValueError('Own trajectory native schedule differs')
        schedule_records.append(schedule);counts['denoiser']+=1
        inputs={n:kwargs[n] for n in ['hidden_states','encoder_hidden_states','timestep','img_ids','txt_ids']}
        abis={'velocity':{'shape':[1,1024,128],'dtype':'torch.bfloat16'},'context':{'shape':[1,512,3072],'dtype':'torch.bfloat16'},
            'terminalRows':{'shape':[1,1536,3072],'dtype':'torch.bfloat16'},'terminalImage':{'shape':[1,1024,3072],'dtype':'torch.bfloat16'},
            'singleMod':{'shape':[1,9216],'dtype':'torch.bfloat16'},'temb':{'shape':[1,3072],'dtype':'torch.bfloat16'},
            'cosine':{'shape':[1536,128],'dtype':'torch.float32'},'sine':{'shape':[1536,128],'dtype':'torch.float32'}}
        def operation(values):
            captured={};result=denoise(this,{**kwargs,**values},maps,native,native.dispatch_attention_fn,lambda n,v:captured.setdefault(n,v))
            return {'velocity':result[0],**captured}
        got=ledger.run('denoise'+str(k),inputs,operation,abis,{'precedingCommitFrameID':last.get('commitFrame'),'expectedOwnLatentPayloadSHA256':expected})
        for name,value in got.items():capture(('C' if name=='context' else name)+str(k),value)
        counts['reducedTerminal']+=1;last['denoiseFrame']=ledger.records[-1]['outputFrame']['frame_id'];last['velocity']=fields['velocity'+str(k)]
        actual.append({'kind':'own-denoiser-input','step':k,'latentPayloadSHA256':payload(x),'EPayloadSHA256':payload(condition),'clockArgumentPayloadSHA256':payload(time_arg)})
        return (got['velocity'],)
    def commit(model_output,timestep,sample,*args,**kwargs):
        phase['kind']='commit';k=phase['step']
        if payload(model_output)!=last['velocity']['payloadSHA256'] or payload(sample)!=expected or timestep.item()!=s['schedule']['timesteps'][k]:raise ValueError('Scheduler did not consume own velocity/state/clock')
        got=ledger.run('commit'+str(k),{'velocity':model_output,'sample':sample,'timestep':timestep,'sigmas':pipe.scheduler.sigmas},
            lambda v:{'nextLatent':old_step(v['velocity'],v['timestep'],v['sample'],*args,**kwargs)[0]},
            {'nextLatent':{'shape':[1,1024,128],'dtype':'torch.bfloat16'}},{'velocityProducerFrameID':last['denoiseFrame'],'nativeSchedulerStepIndexBefore':pipe.scheduler.step_index})
        if pipe.scheduler.step_index!=k+1:raise ValueError('Native scheduler cursor did not advance')
        capture('latent'+str(k+1),got['nextLatent']);last['commitFrame']=ledger.records[-1]['outputFrame']['frame_id'];counts['scheduler']+=1
        return (got['nextLatent'],)
    def callback(pipeline,step,timestep,data):
        nonlocal expected
        if step!=phase['step'] or payload(data['latents'])!=fields['latent'+str(step+1)]['payloadSHA256']:raise ValueError('Pipeline changed own committed state')
        expected=fields['latent'+str(step+1)]['payloadSHA256'];persist_progress();phase['step']=step+1;return {}
    def decode(z,*args,**kwargs):
        phase['kind']='decode'
        if phase['step']!=4:raise ValueError('Decode before complete own sampler trajectory')
        capture('vaeInput',z);got=ledger.run('decode',{'vaeInput':z},lambda v:{'decodedRaw':old_decode(v['vaeInput'],*args,**kwargs)[0]},
            {'decodedRaw':{'shape':[1,3,512,512],'dtype':'torch.bfloat16'}},{'finalCommitFrameID':last['commitFrame'],'finalOwnLatentPayloadSHA256':expected})
        capture('decodedRaw',got['decodedRaw']);counts['VAE']+=1;return (got['decodedRaw'],)
    def block_count(name):
        def hook(module,args):counts[name]+=1
        return hook
    def forbidden_terminal(*args,**kwargs):raise ValueError('Full terminal block/text output execution prohibited')
    for block in transformer.transformer_blocks:handles.append(block.register_forward_pre_hook(block_count('joint')))
    for block in transformer.single_transformer_blocks[:-1]:handles.append(block.register_forward_pre_hook(block_count('singlePrefix')))
    transformer.forward=types.MethodType(adapter,transformer);transformer.single_transformer_blocks[-1].forward=forbidden_terminal
    pipe.prepare_latents=prepare;pipe.scheduler.step=commit;vae.decode=decode;setup=time.perf_counter()-started
    try:
        torch.nn.functional.linear=linear;torch.nn.functional.conv2d=conv;torch.nn.functional.conv_transpose2d=transpose
        torch.nn.functional.scaled_dot_product_attention=attention;native.dispatch_attention_fn=dispatch_attention;pipeline_flux2_klein.randn_tensor=no_draw
        with torch.inference_mode():
            result=pipe(prompt=None,prompt_embeds=embedding,height=512,width=512,num_inference_steps=4,guidance_scale=1.,latents=raw_noise.clone(),output_type='np',
                callback_on_step_end=callback,callback_on_step_end_tensor_inputs=['latents']).images
            if result.shape!=(1,512,512,3) or phase['step']!=4:raise ValueError('Complete candidate image/clock count differs')
            capture('rgbFloat',result);png=pipe.image_processor.numpy_to_pil(result)[0];png_path=root/'candidate-A.png';png.save(png_path);capture('pngRGB',np.asarray(png));persist_progress()
    finally:
        torch.nn.functional.linear=old_linear;torch.nn.functional.conv2d=old_conv;torch.nn.functional.conv_transpose2d=old_transpose
        torch.nn.functional.scaled_dot_product_attention=old_sdpa;native.dispatch_attention_fn=old_dispatch;pipeline_flux2_klein.randn_tensor=old_randn
        pipe.prepare_latents=old_prepare;pipe.scheduler.step=old_step;vae.decode=old_decode
        for handle in handles:handle.remove()
    if counts!={'denoiser':4,'joint':20,'singlePrefix':76,'reducedTerminal':4,'scheduler':4,'VAE':1}:raise ValueError('Declared candidate computation census differs')
    if census()!=before or buffers!={prefix+'.'+n:payload(v) for prefix,module in [('transformer',transformer),('vae',vae)] for n,v in module.named_buffers()}:raise ValueError('Original parameter/buffer closure differs')
    if payload(raw_noise)!=s['noise']['raw']['payloadSHA256'] or payload(embedding)!=s['E']['payloadSHA256'] or snapshot(torch)!=numerical:raise ValueError('Source/numeric closure differs')
    if not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise ValueError('Unexpected stochastic generation')
    dlinears=[x for x in linears if x['kind']=='denoise'];datt=[x for x in sdpa if x['kind']=='denoise']
    if len(dlinears)!=440 or len(dispatch)!=100 or len(datt)!=100 or sum(x['nominalQKAndAVMACs'] for x in datt)!=sum(x['nominalQKAndAVMACs'] for x in dispatch):raise ValueError('Complete affine/attention instrument differs')
    assets_unchanged(assets)
    image={'path':str(png_path),'sha256':sha(png_path),'bytes':png_path.stat().st_size}
    save(out/'candidate.json',{'fields':fields,'image':image,'actual':actual,'counts':counts,'schedules':schedule_records,'sourceEPayloadSHA256':s['E']['payloadSHA256'],
        'scope':'One complete own four-clock trajectory and own final VAE/PNG. Historical E/noise supplied; all earlier learned components remain paid.'})
    save(out/'execution.json',{'linears':linears,'convolutions':convolutions,'SDPA':sdpa,'nativeDispatch':dispatch,'typedActs':ledger.records,'timings':ledger.timings,
        'loadedTransformerParameters':loaded,'physicalTerminalMaps':physical,'parameterManifestSHA256':parameter_manifest,'header':hr,'assets':assets,
        'affineMACs':sum(x['nominalMACs'] for x in linears),'attentionMACs':sum(x['nominalQKAndAVMACs'] for x in sdpa),'convolutionMACs':sum(x['nominalMACs'] for x in convolutions),
        'avoidedTerminalMatrixMACs':231928233984,'sourceParameterBufferRNGClosure':True,'numericalProfile':numerical,'setupSecondsAfterImports':setup,
        'wallSecondsAfterImports':time.perf_counter()-started,'CPUSecondsAfterImports':time.process_time()-cpu,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),
        'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':s['deniedNumericalInputs'],**access}})
    save(out/'candidate-seal.json',{'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json'),'imageSHA256':image['sha256']})
    print('OWN_TERMINAL_TRAJECTORY_SEALED',sha(out/'candidate-seal.json'),flush=True)


if __name__=='__main__':execute(sys.argv[1])
