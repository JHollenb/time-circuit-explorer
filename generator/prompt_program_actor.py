"""Fresh own image actor consuming sealed text programs and one original VAE."""
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
    from prompt_program import consume_clock
    from prompt_program_ledger import Ledger
    from terminal_attention_ledger import wrap_sdpa
    from terminal_capture_guard_v2 import quarantine
    from flux_clock_study import verify,source_checks,assets_unchanged
    from flux_clock import header,selected_load,snapshot
    from complete_masked_study import profile
    root=Path(root).resolve();r,s=read(root/'request.json'),read(root/'inputs.json');verify(root,r)
    started=time.perf_counter();cpu=time.process_time();out=root/'response-result';compiled=read(out/'compiler.json');compiler_seal=read(out/'compiler-seal.json')
    if compiler_seal!={'requestSHA256':r['requestSHA256'],'compilerSHA256':sha(out/'compiler.json'),'executionSHA256':sha(out/'compiler-execution.json')}:raise ValueError('Compiler seal differs before actor')
    if len(compiled['rows'])!=4 or [x['step'] for x in compiled['rows']]!=list(range(4)):raise ValueError('Compiler clock domain differs')
    compiler_sha=sha(out/'compiler-seal.json')
    allowed=[s['E']['path']]+[s['noise'][n]['path'] for n in ['raw','packed','ids']]+[str(Path(s['modelRoot'])/n) for n in s['imageAssets']]
    allowed += [item['path'] for row in compiled['rows'] for group in ['common','cache'] for item in row[group].values()]
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
    pipe.set_progress_bar_config(disable=True);physical=[];block_records=[];map_state={'activeSets':0,'maxActiveSets':0,'maxLogicalMapBytes':0}
    parameter_manifest=identity({'transformer':loaded,'vaeCheckpointSHA256':s['imageAssets']['vae/diffusion_pytorch_model.safetensors'],'promptProgramLawSourceSHA256':r['sourceHashes']['prompt_program.py'],'compilerSealSHA256':compiler_sha})
    ledger=Ledger(r['requestSHA256'],parameter_manifest);phase={'step':0,'kind':'setup'};fields={};linears=[];convolutions=[];sdpa=[];dispatch=[];schedule_records=[];actual=[]
    counts={'denoiser':0,'reducedJoint':0,'reducedSingle':0,'scheduler':0,'VAE':0};last={};handles=[]
    def census():
        return {prefix+'.'+n:[v.data_ptr(),v._version,list(v.shape),str(v.dtype)] for prefix,module in [('transformer',transformer),('vae',vae)] for n,v in [*module.named_parameters(),*module.named_buffers()]}
    before=census();buffers={prefix+'.'+n:payload(v) for prefix,module in [('transformer',transformer),('vae',vae)] for n,v in module.named_buffers()}
    weight_names={v.data_ptr():prefix+'.'+n for prefix,module in [('transformer',transformer),('vae',vae)] for n,v in module.named_parameters()}
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
    attention=wrap_sdpa(old_sdpa,phase,sdpa)
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
        abis={'imageStart':{'shape':[1,1024,3072],'dtype':'torch.bfloat16'},'velocity':{'shape':[1,1024,128],'dtype':'torch.bfloat16'},'context':{'shape':[1,512,3072],'dtype':'torch.bfloat16'},
            'terminalImage':{'shape':[1,1024,3072],'dtype':'torch.bfloat16'},
            'singleMod':{'shape':[1,9216],'dtype':'torch.bfloat16'},'temb':{'shape':[1,3072],'dtype':'torch.bfloat16'},
            'cosine':{'shape':[1536,128],'dtype':'torch.float32'},'sine':{'shape':[1536,128],'dtype':'torch.float32'}}
        row=compiled['rows'][k]
        common={n:load(item,'cuda') for n,item in row['common'].items()}
        banks={n:load(item,'cuda') for n,item in row['cache'].items()}
        if payload(time_arg)!=row['common']['clockArgument']['payloadSHA256'] or payload(kwargs['txt_ids'])!=row['common']['textIDs']['payloadSHA256']:
            raise ValueError('Actor clock/text ID differs from its compiler')
        expected_keys={kind+str(index)+bank for kind,count in [('joint',5),('single',20)] for index in range(count) for bank in ['K','V']}
        if set(banks)!=expected_keys:raise ValueError('Incomplete compiler cache')
        inputs.update({'compiled_'+n:v for n,v in common.items()});inputs.update({'cache_'+n:v for n,v in banks.items()})
        def operation(values):
            captured={};frozen={};previous=[None]
            live_common={n:values['compiled_'+n] for n in common}
            def capture_initial(name,value):
                if name in captured:raise ValueError('Duplicate actor output capture')
                captured[name]=value;frozen[name]=payload(value)
            def cache(kind,index):return values['cache_'+kind+str(index)+'K'],values['cache_'+kind+str(index)+'V']
            def block(kind,index,key,value,before,after):
                import os
                label=kind+str(index);before_hash=payload(before);after_hash=payload(after)
                if payload(key)!=row['cache'][label+'K']['payloadSHA256'] or payload(value)!=row['cache'][label+'V']['payloadSHA256']:
                    raise ValueError('Actor did not consume its own compiled bank')
                if before_hash!=(previous[0] or frozen['imageStart']):raise ValueError('Own image block join differs')
                counts['reducedJoint' if kind=='joint' else 'reducedSingle']+=1
                item={'step':k,'kind':kind,'index':index,'compilerSealSHA256':compiler_sha,'compilerFrameID':row['compilerFrameID'],
                    'textKeyPayloadSHA256':payload(key),'textValuePayloadSHA256':payload(value),'compilerIncomingTextPayloadSHA256':row['states'][label]['payloadSHA256'],
                    'imageInputPayloadSHA256':before_hash,'imageOutputPayloadSHA256':after_hash,'imageShape':list(after.shape),
                    'cumulativeAffineMACs':sum(x['nominalMACs'] for x in linears),'cumulativeAttentionMACs':sum(x['nominalQKAndAVMACs'] for x in sdpa)}
                block_records.append(item);previous[0]=after_hash
                with (out/'block-progress.jsonl').open('a') as f:f.write(__import__('json').dumps(item,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
            velocity=consume_clock(this,values['hidden_states'],values['img_ids'],live_common,cache,native,native.dispatch_attention_fn,block,capture_initial)
            ir=this.pos_embed(values['img_ids'][0])
            captured.update(context=live_common['context'],singleMod=live_common['singleMod'],temb=live_common['temb'],
                cosine=torch.cat([live_common['textCosine'],ir[0]],0),sine=torch.cat([live_common['textSine'],ir[1]],0))
            return {'velocity':velocity,**captured}
        got=ledger.run('denoise'+str(k),inputs,operation,abis,{'precedingCommitFrameID':last.get('commitFrame'),'expectedOwnLatentPayloadSHA256':expected,
            'compilerFrameID':row['compilerFrameID'],'compilerSealSHA256':compiler_sha})
        for name,value in got.items():capture(('C' if name=='context' else name)+str(k),value)
        last['denoiseFrame']=ledger.records[-1]['outputFrame']['frame_id'];last['velocity']=fields['velocity'+str(k)]
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
    def forbidden_block(*args,**kwargs):raise ValueError('Full text-output block execution prohibited')
    for block in [*transformer.transformer_blocks,*transformer.single_transformer_blocks]:block.forward=forbidden_block
    transformer.forward=types.MethodType(adapter,transformer)
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
    if counts!={'denoiser':4,'reducedJoint':20,'reducedSingle':80,'scheduler':4,'VAE':1}:raise ValueError('Declared candidate computation census differs')
    if census()!=before or buffers!={prefix+'.'+n:payload(v) for prefix,module in [('transformer',transformer),('vae',vae)] for n,v in module.named_buffers()}:raise ValueError('Original parameter/buffer closure differs')
    if payload(raw_noise)!=s['noise']['raw']['payloadSHA256'] or payload(embedding)!=s['E']['payloadSHA256'] or snapshot(torch)!=numerical:raise ValueError('Source/numeric closure differs')
    if not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise ValueError('Unexpected stochastic generation')
    dlinears=[x for x in linears if x['kind']=='denoise'];datt=[x for x in sdpa if x['kind']=='denoise']
    if len(dlinears)!=292 or len(dispatch)!=100 or len(datt)!=100 or sum(x['nominalQKAndAVMACs'] for x in datt)!=sum(x['nominalQKAndAVMACs'] for x in dispatch):raise ValueError('Complete affine/attention instrument differs')
    if len(block_records)!=100 or sha(out/'compiler-seal.json')!=compiler_sha:raise ValueError('Actor block/compiler closure differs')
    assets_unchanged(assets)
    image={'path':str(png_path),'sha256':sha(png_path),'bytes':png_path.stat().st_size}
    save(out/'candidate.json',{'fields':fields,'image':image,'actual':actual,'counts':counts,'schedules':schedule_records,'sourceEPayloadSHA256':s['E']['payloadSHA256'],
        'scope':'One own four-clock actor consuming the sealed depth-evolving text program, actual VAE/PNG. Historical E/noise and all compiler/image/head/decoder learned work paid.'})
    save(out/'execution.json',{'linears':linears,'convolutions':convolutions,'SDPA':sdpa,'nativeDispatch':dispatch,'typedActs':ledger.records,'timings':ledger.timings,
        'loadedTransformerParameters':loaded,'compilerSealSHA256':compiler_sha,'blockRecords':block_records,'parameterManifestSHA256':parameter_manifest,'header':hr,'assets':assets,
        'affineMACs':sum(x['nominalMACs'] for x in linears),'attentionMACs':sum(x['nominalQKAndAVMACs'] for x in sdpa),'convolutionMACs':sum(x['nominalMACs'] for x in convolutions),
        'compilerSealSHA256':compiler_sha,'sourceParameterBufferRNGClosure':True,'numericalProfile':numerical,'setupSecondsAfterImports':setup,
        'wallSecondsAfterImports':time.perf_counter()-started,'CPUSecondsAfterImports':time.process_time()-cpu,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),
        'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':s['deniedNumericalInputs'],**access}})
    save(out/'candidate-seal.json',{'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json'),'imageSHA256':image['sha256'],'compilerSealSHA256':compiler_sha})
    print('PROMPT_PROGRAM_ACTOR_SEALED',sha(out/'candidate-seal.json'),flush=True)


if __name__=='__main__':execute(sys.argv[1])
