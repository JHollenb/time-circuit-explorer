"""Same frozen87 compute on the matched parent input; seal/exit precedes comparisons."""
from pathlib import Path
import sys
import time
from encoder_atoms import read,save,sha,identity


def execute(root):
    import hashlib
    import numpy as np
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from matched_parent import compute,abis,work,SOURCE_NAMES,PARAMETERS,PREFIX
    from matched_parent_ledger import Ledger
    from paired_fields import load,retain,payload
    from terminal_attention_ledger import wrap_sdpa
    from terminal_capture_guard_v2 import quarantine
    from complete_masked_study import profile
    from flux_clock_study import verify,source_checks,assets_unchanged
    from flux_clock import header,snapshot
    from projected_complement import load_selected
    root=Path(root).resolve();r=read(root/'request.json');s=read(root/'inputs.json');verify(root,r)
    out=root/'response-result';tick=time.perf_counter();cpu=time.process_time()
    allowed=[x['path'] for x in s['sourceFields'].values()]+[str(Path(s['modelRoot'])/n) for n in s['imageAssets']]
    access=quarantine(root,allowed,s['deniedNumericalInputs'],root/'unused-store');numeric=profile(torch,4)
    if numeric!=s['numericalProfile']:raise ValueError('Original numerical policy differs')
    torch.backends.cuda.preferred_linalg_library('cusolver');linalg=str(torch.backends.cuda.preferred_linalg_library())
    qr_profile={'requestedPreference':'cusolver','returnedPreference':linalg,'torch':torch.__version__,
                'CUDA':torch.version.cuda,'torchGitVersion':torch.version.git_version,'buildConfig':torch.__config__.show(),
                'scope':'Observed requested configuration and calls, not vendor-kernel attestation; no retries.'}
    torch.cuda.reset_peak_memory_stats();model,assets=source_checks(s,diffusers)
    config=read(model/'transformer/config.json')
    if [config[n] for n in ['num_layers','num_single_layers','num_attention_heads','attention_head_dim','mlp_ratio']]!=[5,20,24,128,3.]:raise ValueError('Original geometry differs')
    checkpoint=model/'transformer/diffusion_pytorch_model.safetensors';entries,hr=header(checkpoint)
    weights,loaded=load_selected(checkpoint,PARAMETERS,entries)
    if loaded!=s['originalParameters']:raise ValueError('Original four single1 parameters differ')
    with torch.device('meta'):block=native.Flux2SingleTransformerBlock(3072,24,128,mlp_ratio=3.,eps=1e-6,bias=False)
    block.load_state_dict({n[len(PREFIX):]:v for n,v in weights.items()},strict=True,assign=True)
    block.eval().requires_grad_(False)
    source={n:load(s['sourceFields'][n],'cuda') for n in SOURCE_NAMES}
    shapes={'text':[1,512,3072],'image':[1,1024,3072],'mod':[1,9216],'cosine':[1536,128],'sine':[1536,128]}
    if {n:list(v.shape) for n,v in source.items()}!=shapes or any(v.dtype!=(torch.float32 if n in ['cosine','sine'] else torch.bfloat16) for n,v in source.items()):raise ValueError('Own-state source ABIs differ')
    manifest=identity({'originalParameters':loaded,'intakeSHA256':s['intakeSHA256'],'nativeSources':s['nativeSources']})
    ledger=Ledger(r['requestSHA256'],manifest);trace={};join={};sketch={};physical={}
    inputs={**source,**{'parameter_'+n:v for n,v in weights.items()}}
    before={n:payload(v) for n,v in inputs.items()};census={n:[v.data_ptr(),v._version] for n,v in inputs.items()}
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all();np_rng=np.random.get_state()
    setup=time.perf_counter()-tick;sdpa=[];original_sdpa=torch.nn.functional.scaled_dot_product_attention
    def observe_join(value):join.update(payloadSHA256=payload(value),shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size())
    def observe_sketch(raw,omega,gomega,pulled):
        sketch.update(seedSequence=[20261107,86],bitGenerator='PCG64',numpy=np.__version__,rawWordDtype=str(raw.dtype),
                      rawWordsSHA256=hashlib.sha256(raw.tobytes(order='C')).hexdigest(),signsPayloadSHA256=payload(omega),
                      gatedSignsPayloadSHA256=payload(gomega),pulledBackPayloadSHA256=payload(pulled),shape=list(omega.shape))
    def observe_physical(value):
        physical.update(sourceParameter=PARAMETERS[0],rowInterval=[0,9216],shape=list(value.shape),
                        bytes=value.numel()*value.element_size(),payloadSHA256=payload(value))
    def operation(values):
        for n,v in weights.items():
            if v.data_ptr()!=values['parameter_'+n].data_ptr():raise ValueError('Typed parameter binding differs')
        got,observed=compute(block,{n:values[n] for n in SOURCE_NAMES},native,native.dispatch_attention_fn,
                             ledger.linear,ledger.moment,ledger.matmul,ledger.qr,
                             observe_join,observe_sketch,observe_physical)
        trace.update(observed);return got
    print('MATCHED_PARENT_START',flush=True)
    try:
        torch.nn.functional.scaled_dot_product_attention=wrap_sdpa(original_sdpa,{'stage':'ownSuccessor'},sdpa)
        with torch.inference_mode():
            got=ledger.run(inputs,operation,abis());fields={n:retain(root/'arrays/candidate',n,v) for n,v in got.items()}
    finally:torch.nn.functional.scaled_dot_product_attention=original_sdpa
    if before!={n:payload(v) for n,v in inputs.items()} or census!={n:[v.data_ptr(),v._version] for n,v in inputs.items()} or snapshot(torch)!=numeric or str(torch.backends.cuda.preferred_linalg_library())!=linalg:raise ValueError('Source/parameter/profile closure differs')
    after=np.random.get_state()
    if not torch.equal(rng,torch.get_rng_state()) or len(cuda_rng)!=len(torch.cuda.get_rng_state_all()) or not all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())) or np_rng[0]!=after[0] or not np.array_equal(np_rng[1],after[1]) or np_rng[2:]!=after[2:]:raise ValueError('Global RNG changed')
    costs=work();macs=sum(x['nominalMACs'] for x in ledger.products+ledger.linears+ledger.moments)+sum(x['nominalQKAndAVMACs'] for x in sdpa)
    if len(ledger.products)!=17 or len(ledger.linears)!=1 or len(ledger.moments)!=5 or len(ledger.qrs)!=4 or len(sdpa)!=1 or macs!=costs['candidateMACs'] or macs!=r['prospectiveCandidateMACs']:raise ValueError('Actual candidate operator census differs')
    assets_unchanged(assets)
    save(out/'candidate.json',{'fields':fields,'requestSHA256':r['requestSHA256'],'intakeSHA256':s['intakeSHA256']})
    save(out/'execution.json',{'trace':trace,'actualBF16Join':join,'outputSketch':sketch,'physicalQKV':physical,
        'products':ledger.products,'linears':ledger.linears,'momentProducts':ledger.moments,'SDPA':sdpa,'QR':ledger.qrs,'QRProfile':qr_profile,
        'work':costs,'candidateMatrixMACs':macs,'loadedParameters':loaded,'parameterManifestSHA256':manifest,
        'typedActs':ledger.records,'timings':ledger.timings,'sourceParameterRNGClosure':True,'numericalProfile':numeric,'header':hr,'assets':assets,
        'setupSecondsAfterImports':setup,'wallSecondsAfterImports':time.perf_counter()-tick,'CPUSecondsAfterImports':time.process_time()-cpu,
        'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),
        'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':sorted(set(s['deniedNumericalInputs'])),**access}})
    save(out/'candidate-seal.json',{'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')})
    print('MATCHED_PARENT_SEALED',sha(out/'candidate-seal.json'),flush=True)


if __name__=='__main__':execute(sys.argv[1])
