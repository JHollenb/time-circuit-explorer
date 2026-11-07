"""Six typed stages: source maps, current gates, select, gather, image, own head."""
from pathlib import Path
import sys
import time
from encoder_atoms import read,save,sha,identity


def execute(root):
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from paired_fields import load,payload,retain
    from gate_first_mlp import source_maps,current_gates,select,selected_maps,image,mechanics
    from gate_first_ledger import Ledger
    from terminal_liveness import PARAMETERS
    from terminal_attention_ledger import wrap_sdpa
    from terminal_capture_guard_v2 import quarantine
    from complete_masked_study import profile
    from flux_clock_study import verify,source_checks,assets_unchanged
    from flux_clock import header,snapshot
    from projected_complement import load_selected
    root=Path(root).resolve();r=read(root/'request.json');s=read(root/'inputs.json');verify(root,r)
    started=time.perf_counter();cpu=time.process_time();out=root/'response-result';out.mkdir();save(out/'toy-mechanics.json',mechanics())
    allowed=[x['path'] for x in s['sourceFields'].values()]+[str(Path(s['modelRoot'])/n) for n in s['imageAssets']]
    access=quarantine(root,allowed,s['deniedNumericalInputs'],root/'unused-output-store');numerical=profile(torch,4)
    if numerical!=s['numericalProfile']:raise ValueError('Original numeric profile differs')
    torch.cuda.reset_peak_memory_stats();model,assets=source_checks(s,diffusers);config=read(model/'transformer/config.json')
    if {n:config[n] for n in ['num_layers','num_single_layers','num_attention_heads','attention_head_dim','mlp_ratio','in_channels','patch_size']}!={'num_layers':5,'num_single_layers':20,'num_attention_heads':24,'attention_head_dim':128,'mlp_ratio':3.,'in_channels':128,'patch_size':1} or r['selectedChannels']!=2304:raise ValueError('Fixed real specimen/budget differs')
    entries,hr=header(model/'transformer/diffusion_pytorch_model.safetensors');original,loaded=load_selected(model/'transformer/diffusion_pytorch_model.safetensors',PARAMETERS,entries)
    if loaded!=s['originalParameters']:raise ValueError('Original terminal parameter recipes differ')
    source={n:load(item,'cuda') for n,item in s['sourceFields'].items()}
    expected={'rows':[1,1536,3072],'mod':[1,9216],'temb':[1,3072],'cosine':[1536,128],'sine':[1536,128]}
    if {n:list(v.shape) for n,v in source.items()}!=expected or any(v.dtype!=(torch.float32 if n in ['cosine','sine'] else torch.bfloat16) for n,v in source.items()):raise ValueError('Source input geometry/dtype differs')
    with torch.device('meta'):
        norm=native.AdaLayerNormContinuous(3072,3072,elementwise_affine=False,eps=1e-6,bias=False);projection=torch.nn.Linear(3072,128,bias=False)
    norm.load_state_dict({'linear.weight':original['norm_out.linear.weight']},strict=True,assign=True);projection.load_state_dict({'weight':original['proj_out.weight']},strict=True,assign=True)
    norm.eval().requires_grad_(False);projection.eval().requires_grad_(False)
    pre='single_transformer_blocks.19.attn.';fused=original[pre+'to_qkv_mlp_proj.weight'];output=original[pre+'to_out.weight']
    manifest=identity({'originalParameters':loaded,'sourceParentCandidateSealSHA256':s['parentCandidateSealSHA256'],'gateFirstSourceSHA256':r['sourceHashes']['gate_first_mlp.py']})
    ledger=Ledger(r['requestSHA256'],manifest);fields={};strides={};physical=[];operations=[];attention_records=[];sdpa=[];phase={'stage':'setup'}
    active={v.data_ptr():n for n,v in original.items()};original_payloads={n:payload(v) for n,v in original.items()};source_payloads={n:payload(v) for n,v in source.items()}
    descriptors={n:[v.data_ptr(),list(v.shape),str(v.dtype)] for n,v in original.items()};rng=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all()
    old_linear=torch.nn.functional.linear;old_sdpa=torch.nn.functional.scaled_dot_product_attention;old_dispatch=native.dispatch_attention_fn
    def linear(x,w,bias=None):
        name=active.get(w.data_ptr())
        if name is None or bias is not None or x.dtype!=torch.bfloat16 or w.dtype!=torch.bfloat16 or torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32:raise ValueError('Unaccounted/invalid BF16 gate-first affine')
        value=old_linear(x,w,bias);operations.append({**phase,'parameter':name,'inputShape':list(x.shape),'inputStride':list(x.stride()),'weightShape':list(w.shape),'weightStride':list(w.stride()),
            'outputShape':list(value.shape),'nominalMACs':x.numel()//x.shape[-1]*w.numel(),'dtype':str(x.dtype),'device':x.device.type});return value
    def attention(q,k,v,**kwargs):
        attention_records.append({**phase,'QShape':list(q.shape),'KShape':list(k.shape),'VShape':list(v.shape),'QStride':list(q.stride()),'KStride':list(k.stride()),'VStride':list(v.stride()),'nominalQKAndAVMACs':2*q.numel()*k.shape[1]})
        return old_dispatch(q,k,v,**kwargs)
    def abi(shape,dtype='torch.bfloat16'):return {'shape':shape,'dtype':dtype}
    def run(name,inputs,operation,abis,metadata=None):
        import os,json
        phase['stage']=name;got=ledger.run(name,inputs,operation,abis,metadata)
        for key,value in got.items():
            label=name+'_'+key;fields[label]=retain(root/'arrays/candidate',label,value);strides[label]=list(value.stride())
        row={'stage':name,'typedAct':ledger.records[-1],'fields':dict(fields),'cumulativeAffineMACs':sum(x['nominalMACs'] for x in operations),'cumulativeAttentionMACs':sum(x['nominalQKAndAVMACs'] for x in attention_records)}
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        return got
    def map_records(stage,maps):
        for name,value in maps.items():
            if name not in ['KV','QG','USelected','outSelected']:continue
            active[value.data_ptr()]=name;physical.append({'name':name,'stage':stage,'field':stage+'_'+name,'shape':list(value.shape),'stride':list(value.stride()),'payloadSHA256':payload(value),'bytes':value.numel()*value.element_size()})
    setup=time.perf_counter()-started
    try:
        torch.nn.functional.linear=linear;torch.nn.functional.scaled_dot_product_attention=wrap_sdpa(old_sdpa,phase,sdpa)
        with torch.inference_mode():
            maps=run('sourceMaps',{'fused':fused,'output':output},lambda x:source_maps(x['fused'],x['output'],3072,9216),
                {'KV':abi([6144,3072]),'QG':abi([12288,3072]),'upNorm':abi([9216],'torch.float32'),'downNorm':abi([9216],'torch.float32')})
            map_records('sourceMaps',maps)
            prepared=run('gates',{'rows':source['rows'],'mod':source['mod'],'KV':maps['KV'],'QG':maps['QG']},lambda x:current_gates(x['rows'],x['mod'],x,torch.nn.functional.linear,native),
                {n:abi(shape) for n,shape in {'normalizedImage':[1,1024,3072],'queryRaw':[1,1024,3072],'keyRaw':[1,1536,3072],'valueRaw':[1,1536,3072],'activation':[1,1024,9216]}.items()},
                {'mapProducerFrameID':ledger.records[-1]['outputFrame']['frame_id']})
            selection=run('selection',{'normalized':prepared['normalizedImage'],'activation':prepared['activation'],'upNorm':maps['upNorm'],'downNorm':maps['downNorm']},
                lambda x:select(x['normalized'],x['activation'],x['upNorm'],x['downNorm'],r['selectedChannels']),
                {'rowNormSquared':abi([1024],'torch.float32'),'weightedGateEnergy':abi([9216],'torch.float32'),'score':abi([9216],'torch.float32'),
                 'rankedIndices':abi([9216],'torch.int64'),'support':abi([2304],'torch.int64'),'omittedScoreSum':abi([],'torch.float32'),'selectedScoreSum':abi([],'torch.float32')},
                {'gateProducerFrameID':ledger.records[-1]['outputFrame']['frame_id']})
            selected=run('gather',{'fused':fused,'output':output,'support':selection['support']},lambda x:selected_maps(x['fused'],x['output'],x['support'],3072,9216),
                {'USelected':abi([2304,3072]),'outSelected':abi([3072,5376])},{'selectionFrameID':ledger.records[-1]['outputFrame']['frame_id']})
            map_records('gather',selected)
            image_inputs={'rows':source['rows'],'mod':source['mod'],'cosine':source['cosine'],'sine':source['sine'],**prepared,'support':selection['support'],**selected,
                'qNorm':original[pre+'norm_q.weight'],'kNorm':original[pre+'norm_k.weight']}
            own=run('image',image_inputs,lambda x:image(x['rows'],x['mod'],x['cosine'],x['sine'],{n:x[n] for n in prepared},x['support'],
                {n:x[n] for n in ['USelected','outSelected','qNorm','kNorm']},torch.nn.functional.linear,attention,native),
                {n:abi(shape) for n,shape in {'ownImage':[1,1024,3072],'selectedUp':[1,1024,2304],'selectedProduct':[1,1024,2304],'attentionRead':[1,1024,3072],'preResidualUpdate':[1,1024,3072]}.items()},
                {'gatherFrameID':ledger.records[-1]['outputFrame']['frame_id'],'gateFrameID':ledger.records[1]['outputFrame']['frame_id'],'selectionFrameID':ledger.records[2]['outputFrame']['frame_id']})
            if list(own['ownImage'].stride())!=[4718592,3072,1]:raise ValueError('Own image-view stride differs')
            run('head',{'image':own['ownImage'],'temb':source['temb'],'norm_out.linear.weight':norm.linear.weight,'proj_out.weight':projection.weight},
                lambda x:{'ownVelocity':projection(norm(x['image'],x['temb']))},{'ownVelocity':abi([1,1024,128])},
                {'imageProducerFrameID':ledger.records[-1]['outputFrame']['frame_id']})
    finally:torch.nn.functional.linear=old_linear;torch.nn.functional.scaled_dot_product_attention=old_sdpa
    if original_payloads!={n:payload(v) for n,v in original.items()} or source_payloads!={n:payload(v) for n,v in source.items()} or descriptors!={n:[v.data_ptr(),list(v.shape),str(v.dtype)] for n,v in original.items()} or snapshot(torch)!=numerical:raise ValueError('Original source/map/numeric closure differs')
    if not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise ValueError('RNG closure differs')
    affine=sum(x['nominalMACs'] for x in operations);attention_mac=sum(x['nominalQKAndAVMACs'] for x in attention_records)
    if len(operations)!=6 or len(attention_records)!=1 or len(sdpa)!=1 or affine!=92226453504 or attention_mac!=9663676416 or affine+attention_mac!=r['prospectiveMatrixMACs'] or len(fields)!=24:raise ValueError('Physical work/output census differs')
    assets_unchanged(assets)
    save(out/'candidate.json',{'fields':fields,'outputStrides':strides,'sourceParentCandidateSealSHA256':s['parentCandidateSealSHA256'],
        'scope':'One paid own terminalsource, full current gates and K/V, fixedquarter selectedup/down; actual ownimage consumed by originalhead. No new fullimage or zero-neural claim.'})
    save(out/'execution.json',{'loadedParameters':loaded,'physicalMaps':physical,'operations':operations,'attention':attention_records,'SDPA':sdpa,'typedActs':ledger.records,'timings':ledger.timings,
        'parameterManifestSHA256':manifest,'originalParameterBytes':sum(x['bytes'] for x in loaded),'affineMACs':affine,'attentionMACs':attention_mac,'totalMatrixMACs':affine+attention_mac,
        'sourceInputParameterRNGClosure':True,'numericalProfile':numerical,'header':hr,'assets':assets,'setupSecondsAfterImports':setup,'wallSecondsAfterImports':time.perf_counter()-started,'CPUSecondsAfterImports':time.process_time()-cpu,
        'nonMatrixWork':{'sourceNormCoefficientSquares':56623104,'sourceNormRoots':18432,'fullCurrentSiLUValues':9437184,'inputNormSquares':3145728,'weightedGateEnergySquaresAndMultiplicationsEach':9437184,
            'rankingChannels':9216,'selectedChannels':2304,'rankingComparisons':'not counted','scope':'Squares/reductions/roots/ranking/gathers/copies/transfer/hash/I/O paid outside nominal matrix MACs; actual stage and wall times retained.'},
        'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':s['deniedNumericalInputs'],**access}})
    save(out/'candidate-seal.json',{'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')})
    print('GATE_FIRST_SEALED',sha(out/'candidate-seal.json'),flush=True)


if __name__=='__main__':execute(sys.argv[1])
