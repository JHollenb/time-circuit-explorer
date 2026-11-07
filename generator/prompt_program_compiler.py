"""Fresh image-blind compiler; seals text programs before the actor starts."""
from pathlib import Path
import json
import os
import sys
import time
from encoder_atoms import read,save,sha,identity


def execute(root):
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from diffusers import Flux2Transformer2DModel,Flux2KleinPipeline
    from prompt_program import compile_clock,mechanics,matrix_account
    from prompt_program_ledger import Ledger
    from terminal_attention_ledger import wrap_sdpa
    from terminal_capture_guard_v2 import quarantine
    from flux_clock_study import verify,source_checks,assets_unchanged
    from flux_clock import header,selected_load,snapshot,clock_path
    from complete_masked_study import profile
    from paired_fields import load,retain,payload
    root=Path(root).resolve();r=read(root/'request.json');s=read(root/'inputs.json');verify(root,r)
    started=time.perf_counter();cpu=time.process_time();out=root/'response-result';out.mkdir()
    save(out/'toy-mechanics.json',mechanics())
    allowed=[s['E']['path']]+[str(Path(s['modelRoot'])/n) for n in s['imageAssets']]
    denied=s['deniedNumericalInputs']+[s['noise'][n]['path'] for n in ['raw','packed','ids']]
    access=quarantine(root,allowed,denied,root/'compiler-unused-output-store')
    numerical=profile(torch,4)
    if numerical!=s['numericalProfile']:raise ValueError('Original numerical policy differs')
    torch.cuda.reset_peak_memory_stats();model,assets=source_checks(s,diffusers)
    config=read(model/'transformer/config.json');entries,hr=header(model/'transformer/diffusion_pytorch_model.safetensors')
    with torch.device('meta'):transformer=Flux2Transformer2DModel.from_config(config)
    loaded=selected_load(transformer,model/'transformer/diffusion_pytorch_model.safetensors',sorted(entries),entries,'cuda')
    transformer.eval().requires_grad_(False)
    if {n:config[n] for n in ['num_layers','num_single_layers','num_attention_heads','attention_head_dim','mlp_ratio','in_channels','patch_size']}!={'num_layers':5,'num_single_layers':20,'num_attention_heads':24,'attention_head_dim':128,'mlp_ratio':3.,'in_channels':128,'patch_size':1} or config['guidance_embeds'] is not False:
        raise ValueError('Compiler geometry differs')
    if any(m.bias is not None for m in transformer.modules() if isinstance(m,torch.nn.Linear)):raise ValueError('Bias-free original maps required')
    if sorted({type(m.processor).__name__ for m in transformer.modules() if hasattr(m,'processor')})!=s['attentionProcessors']:raise ValueError('Original compiler attention processors differ')
    manifest=identity({'transformer':loaded,'lawSourceSHA256':r['sourceHashes']['prompt_program.py']})
    ledger=Ledger(r['requestSHA256'],manifest);embedding=load(s['E'],'cuda')
    text_ids=Flux2KleinPipeline._prepare_text_ids(embedding).to('cuda')
    if list(embedding.shape)!=[1,512,7680] or list(text_ids.shape)!=[1,512,4]:raise ValueError('Compiler source shape differs')
    def census():return {n:[v.data_ptr(),v._version,list(v.shape),str(v.dtype)] for n,v in [*transformer.named_parameters(),*transformer.named_buffers()]}
    before=census();buffers={n:payload(v) for n,v in transformer.named_buffers()}
    rng,rng_cuda=torch.get_rng_state(),torch.cuda.get_rng_state_all()
    names={v.data_ptr():n for n,v in transformer.named_parameters()};phase={'step':0,'kind':'compile'}
    linears=[];sdpa=[];dispatch=[];physical=[];blocks=[];rows=[];map_state={'activeSets':0,'maxActiveSets':0,'maxLogicalMapBytes':0}
    old_linear=torch.nn.functional.linear;old_sdpa=torch.nn.functional.scaled_dot_product_attention;old_dispatch=native.dispatch_attention_fn
    def linear(x,w,bias=None):
        name=names.get(w.data_ptr())
        if name is None:raise ValueError('Unbound compiler affine')
        linears.append({**phase,'parameter':name,'inputShape':list(x.shape),'weightShape':list(w.shape),'nominalMACs':x.numel()//x.shape[-1]*w.numel(),'biasPresent':bias is not None})
        return old_linear(x,w,bias)
    def attention(q,k,v,*args,**kwargs):
        if q.shape[1]!=512 or k.shape[1]!=512 or v.shape[1]!=512:raise ValueError('Compiler attention read non-text rows')
        dispatch.append({**phase,'QShape':list(q.shape),'KShape':list(k.shape),'VShape':list(v.shape),'nominalQKAndAVMACs':2*q.numel()*k.shape[1]})
        return old_dispatch(q,k,v,*args,**kwargs)
    def with_map(weight,operation):
        if map_state['activeSets'] or weight.data_ptr() in names:raise ValueError('Overlapping/aliased terminal compiler map')
        map_state['activeSets']=1;map_state['maxActiveSets']=1
        map_state['maxLogicalMapBytes']=max(map_state['maxLogicalMapBytes'],weight.numel()*weight.element_size())
        names[weight.data_ptr()]='singlePhysical.19.KV'
        physical.append({'step':phase['step'],'blockIndex':19,'name':'KV','parameter':'singlePhysical.19.KV',
            'sourceParameter':'single_transformer_blocks.19.attn.to_qkv_mlp_proj.weight','payloadSHA256':payload(weight),'shape':list(weight.shape),'bytes':weight.numel()*weight.element_size()})
        try:return operation()
        finally:del names[weight.data_ptr()];map_state['activeSets']=0
    def forbidden(*args,**kwargs):raise ValueError('Whole transformer/block execution prohibited in compiler')
    transformer.forward=forbidden
    for b in [*transformer.transformer_blocks,*transformer.single_transformer_blocks]:b.forward=forbidden
    setup=time.perf_counter()-started
    try:
        torch.nn.functional.linear=linear;torch.nn.functional.scaled_dot_product_attention=wrap_sdpa(old_sdpa,phase,sdpa);native.dispatch_attention_fn=attention
        with torch.inference_mode():
            for step,value in enumerate(s['schedule']['timesteps']):
                phase['step']=step;clock=clock_path(torch.tensor(value,dtype=torch.float32,device='cuda'),torch)
                captured={};states={};previous=[None]
                def on_layer(kind,index,incoming,k,v,after):
                    key=kind+str(index);state=retain(root/'arrays/compiler',str(step)+'-'+key+'-text',incoming)
                    if previous[0] is not None and state['payloadSHA256']!=previous[0]:raise ValueError('Compiler text continuation join differs')
                    # V is a view into a much larger fused projection in single blocks.
                    # Own physical banks release that backing storage after this layer.
                    states[key]=state;captured[key+'K']=k.clone();captured[key+'V']=v.clone()
                    row={'step':step,'kind':kind,'index':index,'incomingText':state,'keyPayloadSHA256':payload(k),'valuePayloadSHA256':payload(v),
                        'outgoingTextPayloadSHA256':None if after is None else payload(after),'updated':after is not None,
                        'cacheTiming':'incoming-state','keyContract':'head-normalized and text-RoPE applied once','valueContract':'projected, unnormalized and unrotated',
                        'cumulativeAffineMACs':sum(x['nominalMACs'] for x in linears),'cumulativeAttentionMACs':sum(x['nominalQKAndAVMACs'] for x in sdpa)}
                    previous[0]=row['outgoingTextPayloadSHA256'];blocks.append(row)
                    with (out/'compiler-block-progress.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
                common_abis={'context':([1,512,3072],'torch.bfloat16'),'temb':([1,3072],'torch.bfloat16'),
                    'imageMod':([1,18432],'torch.bfloat16'),'textMod':([1,18432],'torch.bfloat16'),'singleMod':([1,9216],'torch.bfloat16'),
                    'clockArgument':([1],'torch.bfloat16'),'textIDs':([1,512,4],'torch.int64'),'textCosine':([512,128],'torch.float32'),'textSine':([512,128],'torch.float32')}
                abis={n:{'shape':shape,'dtype':dtype} for n,(shape,dtype) in common_abis.items()}
                for kind,count in [('joint',5),('single',20)]:
                    for index in range(count):
                        for bank in ['K','V']:abis[kind+str(index)+bank]={'shape':[1,512,24,128],'dtype':'torch.bfloat16'}
                def operation(values):
                    common=compile_clock(transformer,values['embedding'],values['clockArgument'],values['textIDs'],native,native.dispatch_attention_fn,on_layer,with_map)
                    return {**common,**captured}
                got=ledger.run('compile'+str(step),{'embedding':embedding,'clockArgument':clock['transformer_argument'],'textIDs':text_ids},operation,abis,
                    {'sourceEPayloadSHA256':s['E']['payloadSHA256'],'imageInputsPermitted':False,'clockSchedulerValue':value})
                fields={n:retain(root/'arrays/compiler',str(step)+'-'+n,v) for n,v in got.items()}
                if states['joint0']['payloadSHA256']!=fields['context']['payloadSHA256']:raise ValueError('Clock did not reset to its own C(E)')
                rows.append({'step':step,'common':{n:fields[n] for n in common_abis},'cache':{n:fields[n] for n in fields if n not in common_abis},
                    'states':states,'compilerFrameID':ledger.records[-1]['outputFrame']['frame_id']})
                save(out/('compiler-clock-'+str(step)+'.json'),rows[-1]);del got,captured,states
    finally:
        torch.nn.functional.linear=old_linear;torch.nn.functional.scaled_dot_product_attention=old_sdpa;native.dispatch_attention_fn=old_dispatch
    if census()!=before or buffers!={n:payload(v) for n,v in transformer.named_buffers()} or payload(embedding)!=s['E']['payloadSHA256'] or snapshot(torch)!=numerical:raise ValueError('Compiler source closure differs')
    if not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise ValueError('Compiler unexpected RNG advance')
    if len(blocks)!=100 or len(sdpa)!=96 or len(dispatch)!=96 or len(physical)!=4 or sum(x['nominalQKAndAVMACs'] for x in sdpa)!=sum(x['nominalQKAndAVMACs'] for x in dispatch):raise ValueError('Compiler operation census differs')
    actual_text_macs=sum(x['nominalMACs'] for x in linears if x['parameter'].startswith(('transformer_blocks.','single_transformer_blocks.','singlePhysical.')))+sum(x['nominalQKAndAVMACs'] for x in sdpa)
    # NPY headers are verified separately; logical bytes use declared shape and BF16.
    logical_cache_bytes=sum(__import__('math').prod(x['shape'])*2 for row in rows for x in row['cache'].values())
    if actual_text_macs!=matrix_account()['compilerMatrixMACs'] or logical_cache_bytes!=matrix_account()['cacheBytesBF16']:raise ValueError('Compiler independent work/cache account differs')
    assets_unchanged(assets)
    save(out/'compiler.json',{'rows':rows,'requestSHA256':r['requestSHA256'],'parameterManifestSHA256':manifest,'logicalCacheBytes':logical_cache_bytes,
        'scope':'Four independently reset depth-evolving text-only programs, incoming K/V; no image/noise/hidden target operands.'})
    save(out/'compiler-execution.json',{'linears':linears,'SDPA':sdpa,'nativeDispatch':dispatch,'typedActs':ledger.records,'timings':ledger.timings,
        'loadedTransformerParameters':loaded,'physicalSingleMaps':physical,'mapMaterializationScope':map_state,'blockRecords':blocks,'parameterManifestSHA256':manifest,'header':hr,'assets':assets,
        'affineMACs':sum(x['nominalMACs'] for x in linears),'attentionMACs':sum(x['nominalQKAndAVMACs'] for x in sdpa),'textProgramMatrixMACs':actual_text_macs,
        'commonPreprocessingAffineMACs':sum(x['nominalMACs'] for x in linears if not x['parameter'].startswith(('transformer_blocks.','single_transformer_blocks.','singlePhysical.'))),
        'logicalCacheBytes':logical_cache_bytes,'sourceParameterBufferRNGClosure':True,'numericalProfile':numerical,'setupSecondsAfterImports':setup,
        'wallSecondsAfterImports':time.perf_counter()-started,'CPUSecondsAfterImports':time.process_time()-cpu,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),
        'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':denied,**access}})
    save(out/'compiler-seal.json',{'requestSHA256':r['requestSHA256'],'compilerSHA256':sha(out/'compiler.json'),'executionSHA256':sha(out/'compiler-execution.json')})
    print('PROMPT_PROGRAM_COMPILER_SEALED',sha(out/'compiler-seal.json'),flush=True)


if __name__=='__main__':execute(sys.argv[1])
