"""One current-stream mean-Up candidate; no comparator arrays before exit."""
from pathlib import Path
import sys
import time
from encoder_atoms import read,save,sha,identity

PREFIX='single_transformer_blocks.0.'
PARAMETERS=[PREFIX+'attn.'+name for name in ['to_qkv_mlp_proj.weight','to_out.weight','norm_q.weight','norm_k.weight']]


def execute(root):
    mode='completion'
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from mean_up_completion import replay,abis
    from mean_up_ledger import Ledger
    from paired_fields import load,retain,payload
    from terminal_attention_ledger import wrap_sdpa
    from terminal_capture_guard_v2 import quarantine
    from complete_masked_study import profile
    from flux_clock_study import verify,source_checks,assets_unchanged
    from flux_clock import header,snapshot
    from projected_complement import load_selected
    root=Path(root).resolve();r=read(root/'request.json');s=read(root/'inputs.json');verify(root,r);out=root/'response-result'
    started=time.perf_counter();cpu=time.process_time();preceding=s['parentSeals']['matched73']
    allowed=[x['path'] for x in s['sourceFields'].values()]+[str(Path(s['modelRoot'])/n) for n in s['imageAssets']]
    denied=list(s['deniedNumericalInputs'])
    access=quarantine(root,allowed,denied,root/('unused-store-'+mode));numeric=profile(torch,4)
    if numeric!=s['numericalProfile']:raise ValueError('Pinned numerical policy differs')
    torch.cuda.reset_peak_memory_stats();model,assets=source_checks(s,diffusers);config=read(model/'transformer/config.json')
    if [config[n] for n in ['num_layers','num_single_layers','num_attention_heads','attention_head_dim','mlp_ratio']]!=[5,20,24,128,3.]:raise ValueError('Original block geometry differs')
    entries,hr=header(model/'transformer/diffusion_pytorch_model.safetensors');weights,loaded=load_selected(model/'transformer/diffusion_pytorch_model.safetensors',PARAMETERS,entries)
    if loaded!=s['originalParameters']:raise ValueError('Original single0 parameters differ')
    with torch.device('meta'):block=native.Flux2SingleTransformerBlock(3072,24,128,mlp_ratio=3.,eps=1e-6,bias=False)
    block.load_state_dict({name[len(PREFIX):]:value for name,value in weights.items()},strict=True,assign=True);block.eval().requires_grad_(False)
    source={name:load(item,'cuda') for name,item in s['sourceFields'].items()}
    expected={'text':[1,512,3072],'image':[1,1024,3072],'mod':[1,9216],'cosine':[1536,128],'sine':[1536,128],'sourceUpNorm':[20,9216],'sourceDownNorm':[20,9216]}
    if {n:list(v.shape) for n,v in source.items()}!=expected or any(v.dtype!=(torch.float32 if n in ['cosine','sine','sourceUpNorm','sourceDownNorm'] else torch.bfloat16) for n,v in source.items()):raise ValueError('Common source geometry/dtype differs')
    manifest=identity({'originalParameters':loaded,'parentSeals':s['parentSeals'],'helpers':{n:r['sourceHashes'][n] for n in ['mean_up_completion.py','gate_first_mlp.py','image_partition_moments.py']}})
    ledger=Ledger(r['requestSHA256'],manifest);old_sdpa=torch.nn.functional.scaled_dot_product_attention;sdpa=[];phase={'stage':mode};trace={};strides={}
    before={n:payload(v) for n,v in weights.items()};source_before={n:payload(v) for n,v in source.items()};census={n:[v.data_ptr(),v._version,list(v.shape),str(v.dtype)] for n,v in block.named_parameters()}
    rng=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all();setup=time.perf_counter()-started
    def affine(row):
        if row['dtype']!='torch.bfloat16' or row['device']!='cuda' or torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32:raise ValueError('Local affine numerical policy differs')
    def product(row):
        if row['dtype']!='torch.float32' or row['device']!='cuda' or torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32:raise ValueError('Moment product numerical policy differs')
    def operation(values):
        for name,value in weights.items():
            if value.data_ptr()!=values['parameter_'+name].data_ptr():raise ValueError('Declared parameter/source binding differs')
        norms={name:values[name] for name in ['sourceUpNorm','sourceDownNorm']}
        got,observed=replay(block,{n:values[n] for n in source},native,norms)
        for row in observed['operations']:affine(row)
        for row in observed['products']:product(row)
        trace.update(observed);strides.update({n:list(v.stride()) for n,v in got.items()});return got
    print('MEAN_UP_START',mode,flush=True)
    try:
        torch.nn.functional.scaled_dot_product_attention=wrap_sdpa(old_sdpa,phase,sdpa)
        with torch.inference_mode():
            got=ledger.run(mode,{**source,**{'parameter_'+n:v for n,v in weights.items()}},operation,abis(),
                {'sourceComparatorSealSHA256':preceding,'commonInputFields':{n:x['payloadSHA256'] for n,x in s['sourceFields'].items()},'frozenHelper':'mean_up_completion.py'})
            fields={n:retain(root/'arrays'/mode,n,v) for n,v in got.items()}
    finally:torch.nn.functional.scaled_dot_product_attention=old_sdpa
    if before!={n:payload(v) for n,v in weights.items()} or source_before!={n:payload(v) for n,v in source.items()} or census!={n:[v.data_ptr(),v._version,list(v.shape),str(v.dtype)] for n,v in block.named_parameters()} or snapshot(torch)!=numeric:raise ValueError('Source/parameter/numeric closure differs')
    if not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise ValueError('Unexpected random computation')
    affine_macs=sum(x['nominalMACs'] for x in trace['operations']);attention_macs=sum(x['nominalMACs'] for x in trace['attention']);moment_macs=sum(x['nominalMACs'] for x in trace['products'])
    if affine_macs!=r['prospectiveAffineMACs'] or attention_macs!=9663676416 or moment_macs!=2216165376 or len(sdpa)!=1 or sum(x['nominalQKAndAVMACs'] for x in sdpa)!=attention_macs:raise ValueError('Local physical work instrument differs')
    assets_unchanged(assets)
    save(out/'candidate.json',{'fields':fields,'outputStrides':strides,'mode':mode,'sourceParentSeals':s['parentSeals'],'requestSHA256':r['requestSHA256']})
    save(out/'execution.json',{'trace':trace,'SDPA':sdpa,'loadedParameters':loaded,'parameterManifestSHA256':manifest,'typedActs':ledger.records,'timings':ledger.timings,
        'affineMACs':affine_macs,'attentionMACs':attention_macs,'momentMACs':moment_macs,'totalMatrixMACs':affine_macs+attention_macs+moment_macs,
        'sourceParameterRNGClosure':True,'numericalProfile':numeric,'sourceComparatorSealSHA256':preceding,'header':hr,'assets':assets,
        'setupSecondsAfterImports':setup,'wallSecondsAfterImports':time.perf_counter()-started,'CPUSecondsAfterImports':time.process_time()-cpu,
        'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),
        'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':sorted(set(denied)),**access}})
    save(out/'candidate-seal.json',{'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')})
    print('MEAN_UP_SEALED',mode,sha(out/'candidate-seal.json'),flush=True)


if __name__=='__main__':execute(sys.argv[1])
