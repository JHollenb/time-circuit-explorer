"""Four own sampler clocks with live full image-key moments and one original VAE."""
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
    from mean_up_trajectory import denoise,completion_abis
    from gate_first_trajectory import source_norms
    from test_mean_up_trajectory import mechanics
    from mean_trajectory_ledger import Ledger
    from terminal_attention_ledger import wrap_sdpa
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
    pipe.set_progress_bar_config(disable=True);physical=[];block_records=[];map_state={'activeSets':0,'maxActiveSets':0,'maxLogicalMapBytes':0}
    parameter_manifest=identity({'transformer':loaded,'vaeCheckpointSHA256':s['imageAssets']['vae/diffusion_pytorch_model.safetensors'],'meanTrajectorySourceSHA256':r['sourceHashes']['mean_up_trajectory.py']})
    ledger=Ledger(r['requestSHA256'],parameter_manifest);completion_ledger=Ledger(r['requestSHA256'],parameter_manifest);completion_records=[];phase={'step':0,'kind':'setup'};fields={};linears=[];convolutions=[];sdpa=[];dispatch=[];products=[];schedule_records=[];actual=[];selection_records=[]
    counts={'denoiser':0,'joint':0,'single':0,'terminal':0,'scheduler':0,'VAE':0};last={};handles=[]
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
    def with_maps(index,maps,operation):
        names=[n for n in maps if n not in ['qNorm','kNorm']]
        if map_state['activeSets']:raise ValueError('Physical map callback overlap')
        map_state['activeSets']+=1;map_state['maxActiveSets']=max(map_state['maxActiveSets'],map_state['activeSets'])
        total=sum(maps[n].numel()*maps[n].element_size() for n in names);map_state['maxLogicalMapBytes']=max(map_state['maxLogicalMapBytes'],total)
        pointers=[];descriptors={n:[maps[n].data_ptr(),list(maps[n].shape),str(maps[n].dtype)] for n in names}
        try:
            for name in names:
                value=maps[name];pointer=value.data_ptr()
                if pointer in weight_names:raise ValueError('Physical slice aliases a live original parameter')
                label='singlePhysical.'+str(index)+'.'+name;weight_names[pointer]=label;pointers.append(pointer)
                support=next((x['fields']['support'] for x in reversed(selection_records) if x['step']==phase['step'] and x['index']==index),None)
                if name in ['USelected','outSelected'] and support is None:raise ValueError('Selected map lacks its own current support')
                physical.append({'step':phase['step'],'blockIndex':index,'name':name,'parameter':label,
                    'sourceParameter':'single_transformer_blocks.'+str(index)+('.attn.to_out.weight' if name=='outSelected' else '.attn.to_qkv_mlp_proj.weight'),
                    'support':support if name in ['USelected','outSelected'] else None,
                    'payloadSHA256':payload(value),'shape':list(value.shape),'bytes':value.numel()*value.element_size()})
            result=operation()
            if descriptors!={n:[maps[n].data_ptr(),list(maps[n].shape),str(maps[n].dtype)] for n in names}:raise ValueError('Physical map descriptor mutation')
            return result
        finally:
            for pointer in pointers:del weight_names[pointer]
            map_state['activeSets']-=1
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
        dispatch.append({**phase,'QShape':list(q.shape),'KShape':list(k.shape),'VShape':list(v.shape),'QStride':list(q.stride()),'KStride':list(k.stride()),'VStride':list(v.stride()),'nominalQKAndAVMACs':2*q.numel()*k.shape[1]})
        return old_dispatch(q,k,v,*args,**kwargs)
    def product(kind,a,b):
        if phase['kind']!='denoise' or a.dtype!=torch.float32 or b.dtype!=torch.float32 or torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32:
            raise ValueError('Explicit partition products must be FP32 without autocast')
        value=torch.matmul(a,b)
        if value.dtype!=torch.float32:raise ValueError('Explicit product result dtype differs')
        index=len([x for x in block_records if x['step']==phase['step']])
        products.append({**phase,'operator':kind,'blockKind':'joint' if index<5 else 'single','blockIndex':index if index<5 else index-5,
            'leftShape':list(a.shape),'rightShape':list(b.shape),'outputShape':list(value.shape),'leftDtype':str(a.dtype),'rightDtype':str(b.dtype),'outputDtype':str(value.dtype),
            'nominalMACs':value.numel()*a.shape[-1],'autocastEnabled':False})
        return value
    def capture(name,value):
        if name in fields:
            if fields[name]['payloadSHA256']!=payload(value):raise ValueError('Retained own output alias differs')
            return fields[name]
        fields[name]=retain(root/'arrays/candidate',name,value);return fields[name]
    def persist_progress():
        import os
        row={'step':phase['step'],'kind':phase['kind'],'typedActs':ledger.records[-2:],'fields':dict(fields),
            'cumulativeAffineMACs':sum(x['nominalMACs'] for x in linears),'cumulativeAttentionMACs':sum(x['nominalQKAndAVMACs'] for x in sdpa)+sum(x['nominalMACs'] for x in products),'cumulativeConvolutionMACs':sum(x['nominalMACs'] for x in convolutions)}
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
        inputs={n:kwargs[n] for n in ['hidden_states','encoder_hidden_states','timestep','img_ids','txt_ids']};inputs.update(norms)
        abis={'imageStart':{'shape':[1,1024,3072],'dtype':'torch.bfloat16'},'velocity':{'shape':[1,1024,128],'dtype':'torch.bfloat16'},'context':{'shape':[1,512,3072],'dtype':'torch.bfloat16'},
            'terminalRows':{'shape':[1,1536,3072],'dtype':'torch.bfloat16'},'terminalImage':{'shape':[1,1024,3072],'dtype':'torch.bfloat16'},
            'singleMod':{'shape':[1,9216],'dtype':'torch.bfloat16'},'temb':{'shape':[1,3072],'dtype':'torch.bfloat16'},
            'cosine':{'shape':[1536,128],'dtype':'torch.float32'},'sine':{'shape':[1536,128],'dtype':'torch.float32'}}
        def operation(values):
            captured={};frozen={};previous=[None,None]
            def capture_initial(name,value):
                if name in captured:raise ValueError('Duplicate fixed-memory output capture')
                captured[name]=value
                if name in ['context','imageStart']:frozen[name]=payload(value)
            pending_moments={}
            def selection_capture(index,values):
                prefix='s_single'+str(index)+'_';bindings={}
                for name,value in values.items():
                    label=prefix+name
                    if label in captured:raise ValueError('Duplicate current selection')
                    captured[label]=value;bindings[name]={'field':label+str(k),'payloadSHA256':payload(value)}
                selection_records.append({'step':k,'index':index,'liveRows':1024 if index==19 else 1536,'fields':bindings})
            def moment_capture(values):
                index=len([x for x in block_records if x['step']==k]);kind='joint' if index<5 else 'single';index=index if index<5 else index-5
                if pending_moments:raise ValueError('Unconsumed moment capture')
                for name,value in values.items():
                    label='b_'+kind+str(index)+'_moment_'+name
                    if label in captured:raise ValueError('Duplicate moment output')
                    captured[label]=value;pending_moments[name]={'field':label+str(k),'payloadSHA256':payload(value)}
            def block(kind,index,bt,bi,after_t,after_i,alpha,read):
                import os
                prefix='b_'+kind+str(index)+'_';ih,th=payload(bi),payload(bt)
                if ih!=(previous[0] or frozen['imageStart']) or th!=(previous[1] or frozen['context']):raise ValueError('Own image/text block join differs')
                captured[prefix+'imageOut']=after_i
                if after_t is not None:
                    captured[prefix+'textOut']=after_t;captured[prefix+'alpha']=alpha;captured[prefix+'textRead']=read
                counts['joint' if kind=='joint' else 'terminal' if after_t is None else 'single']+=1
                row={'step':k,'kind':kind,'index':index,'imageInputPayloadSHA256':ih,'textInputPayloadSHA256':th,
                    'imageOutputField':prefix+'imageOut'+str(k),'imageOutputPayloadSHA256':payload(after_i),
                    'textOutputField':None if after_t is None else prefix+'textOut'+str(k),'textOutputPayloadSHA256':None if after_t is None else payload(after_t),
                    'alphaField':None if alpha is None else prefix+'alpha'+str(k),'alphaPayloadSHA256':None if alpha is None else payload(alpha),
                    'textReadField':None if read is None else prefix+'textRead'+str(k),'textReadPayloadSHA256':None if read is None else payload(read),
                    'cumulativeAffineMACs':sum(x['nominalMACs'] for x in linears),'cumulativeAttentionMACs':sum(x['nominalQKAndAVMACs'] for x in sdpa)+sum(x['nominalMACs'] for x in products)}
                if set(pending_moments)!=(set(['mean','covariance','logZI','logZT','quadratic']) if after_t is not None else set()):raise ValueError('Moment block binding differs')
                row['moments']=dict(pending_moments);pending_moments.clear()
                block_records.append(row);previous[:]=[row['imageOutputPayloadSHA256'],row['textOutputPayloadSHA256']]
                with (out/'block-progress.jsonl').open('a') as f:f.write(__import__('json').dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
            norm_inputs={n:values[n] for n in ['sourceUpNorm','sourceDownNorm']}
            def run_completion(index,component,text,image,mod,rope,operation):
                import os
                prefix='single_transformer_blocks.'+str(index)+'.attn.'
                parameters={'parameter_'+name:getattr(component.attn,name.split('.')[0]).weight for name in ['to_qkv_mlp_proj.weight','to_out.weight','norm_q.weight','norm_k.weight']}
                ins={'text':text,'image':image,'mod':mod,'cosine':rope[0],'sine':rope[1],**norm_inputs,**parameters}
                prior_kind,prior_index=('joint',4) if index==0 else ('single',index-1)
                metadata={'step':k,'blockIndex':index,'terminal':index==19,'sourceNormProducerFrameID':norm_frame,
                    'ownTextInputField':'b_'+prior_kind+str(prior_index)+'_textOut'+str(k),'ownImageInputField':'b_'+prior_kind+str(prior_index)+'_imageOut'+str(k),
                    'sourceParameterNames':{'parameter_'+name:prefix+name for name in ['to_qkv_mlp_proj.weight','to_out.weight','norm_q.weight','norm_k.weight']},
                    'modulationField':'singleMod'+str(k),'cosineField':'cosine'+str(k),'sineField':'sine'+str(k)}
                def invoke(bound):
                    for name,value in ins.items():
                        if value.data_ptr()!=bound[name].data_ptr():raise ValueError('Completion declared source binding differs')
                    return operation()
                got=completion_ledger.run('completion'+str(k)+'_'+str(index),ins,invoke,completion_abis(index),metadata)
                retained={}
                for name,value in got.items():
                    if name.startswith('selection_'):label='s_single'+str(index)+'_'+name[len('selection_'):]+str(k)
                    elif name in ['textOut','imageOut','alpha','textRead']:label='b_single'+str(index)+'_'+name+str(k)
                    else:label='u_single'+str(index)+'_'+name+str(k)
                    retained[name]=capture(label,value)
                event={'step':k,'index':index,'liveRows':1024 if index==19 else 1536,'anchorRows':1 if index==19 else 2,
                    'fields':retained,'typedOutputFrameID':completion_ledger.records[-1]['outputFrame']['frame_id'],
                    'outputStrides':{name:list(value.stride()) for name,value in got.items()},
                    'retention':'Completeactivation/product/completedUp/currentselectedUp/normalized/update/means/anchors/gate retained nearcompute perblock; onlycompact state/selection/moment GPUrefs advance.'}
                completion_records.append(event)
                with (out/'completion-progress.jsonl').open('a') as stream:stream.write(__import__('json').dumps(event,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
                return {name:got[name] for name in ['textOut','imageOut','alpha','textRead'] if name in got}

            native_inputs={n:v for n,v in {**kwargs,**values}.items() if n not in norm_inputs}
            result=denoise(this,native_inputs,native,native.dispatch_attention_fn,capture_initial,block,with_maps,product,moment_capture,norm_inputs,selection_capture,run_completion)
            return {'velocity':result[0],**captured}
        for kind,count in [('joint',5),('single',20)]:
            for index in range(count):
                prefix='b_'+kind+str(index)+'_';abis[prefix+'imageOut']={'shape':[1,1024,3072],'dtype':'torch.bfloat16'}
                if kind=='single':
                    for name,shape,dtype in [('rowNormSquared',[1024 if index==19 else 1536],'torch.float32'),('weightedGateEnergy',[9216],'torch.float32'),('score',[9216],'torch.float32'),('rankedIndices',[9216],'torch.int64'),('support',[2304],'torch.int64'),('omittedScoreSum',[],'torch.float32'),('selectedScoreSum',[],'torch.float32')]:
                        abis['s_single'+str(index)+'_'+name]={'shape':shape,'dtype':dtype}
                if kind=='single' and index==19:continue
                for name in ['textOut','textRead']:abis[prefix+name]={'shape':[1,512,3072],'dtype':'torch.bfloat16'}
                abis[prefix+'alpha']={'shape':[1,24,512],'dtype':'torch.float32'}
                for name,shape in {'mean':[1,24,128],'covariance':[1,24,128,128],'logZI':[1,24,512],'logZT':[1,24,512],'quadratic':[1,24,512]}.items():
                    abis[prefix+'moment_'+name]={'shape':shape,'dtype':'torch.float32'}
        got=ledger.run('denoise'+str(k),inputs,operation,abis,{'precedingCommitFrameID':last.get('commitFrame'),'expectedOwnLatentPayloadSHA256':expected,'sourceNormProducerFrameID':norm_frame})
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
    norms=ledger.run('sourceNorms',{},lambda values:source_norms(transformer.single_transformer_blocks),
        {name:{'shape':[20,9216],'dtype':'torch.float32'} for name in ['sourceUpNorm','sourceDownNorm']},
        {'preparation':'Once per20blocks,FP32 source U-row/down-column norms; original maps paid.'})
    for name,value in norms.items():capture(name,value)
    norm_hashes={name:payload(value) for name,value in norms.items()}
    norm_frame=ledger.records[-1]['outputFrame']['frame_id']
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
    if counts!={'denoiser':4,'joint':20,'single':76,'terminal':4,'scheduler':4,'VAE':1}:raise ValueError('Declared candidate computation census differs')
    if census()!=before or buffers!={prefix+'.'+n:payload(v) for prefix,module in [('transformer',transformer),('vae',vae)] for n,v in module.named_buffers()}:raise ValueError('Original parameter/buffer closure differs')
    if payload(raw_noise)!=s['noise']['raw']['payloadSHA256'] or payload(embedding)!=s['E']['payloadSHA256'] or snapshot(torch)!=numerical:raise ValueError('Source/numeric closure differs')
    if not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise ValueError('Unexpected stochastic generation')
    dlinears=[x for x in linears if x['kind']=='denoise'];datt=[x for x in sdpa if x['kind']=='denoise']
    if len(dlinears)!=600 or len(dispatch)!=100 or len(datt)!=100 or sum(x['nominalQKAndAVMACs'] for x in datt)!=sum(x['nominalQKAndAVMACs'] for x in dispatch):raise ValueError('Complete affine/attention instrument differs')
    if map_state['activeSets']!=0 or map_state['maxActiveSets']!=1 or map_state['maxLogicalMapBytes']!=113246208 or len(physical)!=244 or len(block_records)!=100:raise ValueError('Physical maps/block census differs')
    if len(products)!=480 or [x['operator'] for x in products]!=['imageCovariance','queryCovariance','queryMean','textScores','textAV']*96:raise ValueError('Explicit text product census differs')
    if len(completion_records)!=80 or len(completion_ledger.records)!=80:raise ValueError('Completion transaction count differs')
    if len(selection_records)!=80 or norm_hashes!={name:payload(value) for name,value in norms.items()}:raise ValueError('Current selection/source norm closure differs')
    assets_unchanged(assets)
    image={'path':str(png_path),'sha256':sha(png_path),'bytes':png_path.stat().st_size}
    save(out/'candidate.json',{'fields':fields,'image':image,'actual':actual,'counts':counts,'schedules':schedule_records,'sourceEPayloadSHA256':s['E']['payloadSHA256'],
        'scope':'One frozen currentstreammean-Up completion through20singleMLPs/4ownclocks; ownmeans/quarter/allgates/fulloutput, livecoupledstreams,70moments/joints/head,4commits1VAE. Completeperblockbanksstreamednearcompute. Originalmaps/E/noise/decoderpaid; notzero-neural.'})
    save(out/'execution.json',{'explicitTextProducts':products,'linears':linears,'convolutions':convolutions,'SDPA':sdpa,'nativeDispatch':dispatch,'typedActs':ledger.records,'timings':ledger.timings,'completionActs':completion_ledger.records,'completionTimings':completion_ledger.timings,'completionRecords':completion_records,
        'selectionRecords':selection_records,'sourceNorms':{n:fields[n] for n in norms},'sourceNormPreparation':{'coefficientSquares':1132462080,'roots':368640,'FP32NormBytes':1474560},'gateActivations':1113587712,'loadedTransformerParameters':loaded,'physicalSingleMaps':physical,'mapMaterializationScope':map_state,'blockRecords':block_records,'parameterManifestSHA256':parameter_manifest,'header':hr,'assets':assets,
        'affineMACs':sum(x['nominalMACs'] for x in linears),'attentionMACs':sum(x['nominalQKAndAVMACs'] for x in sdpa)+sum(x['nominalMACs'] for x in products),'nativeSDPAMACs':sum(x['nominalQKAndAVMACs'] for x in sdpa),'explicitTextMACs':sum(x['nominalMACs'] for x in products),'convolutionMACs':sum(x['nominalMACs'] for x in convolutions),
        'avoidedMatrixMACs':3044322312192,'sourceParameterBufferRNGClosure':True,'numericalProfile':numerical,'setupSecondsAfterImports':setup,
        'wallSecondsAfterImports':time.perf_counter()-started,'CPUSecondsAfterImports':time.process_time()-cpu,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),
        'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':s['deniedNumericalInputs'],**access}})
    save(out/'candidate-seal.json',{'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json'),'imageSHA256':image['sha256']})
    print('MEAN_TRAJECTORY_SEALED',sha(out/'candidate-seal.json'),flush=True)


if __name__=='__main__':execute(sys.argv[1])
