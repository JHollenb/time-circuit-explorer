"""One shared template, two actual inputs, both sealed before comparison access."""
from pathlib import Path
import sys
import time
from encoder_atoms import read,save,sha,identity


def execute(root):
    import numpy as np
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from template_transport import compute,abis,work,SOURCE_NAMES,TEMPLATE_NAMES,PARAMETERS,PREFIX
    from template_transport_ledger import Ledger
    from paired_fields import load,retain,payload
    from terminal_attention_ledger import wrap_sdpa
    from terminal_capture_guard_v2 import quarantine
    from complete_masked_study import profile
    from flux_clock_study import verify,source_checks,assets_unchanged
    from flux_clock import header,snapshot
    from projected_complement import load_selected
    root=Path(root).resolve();r=read(root/'request.json');s=read(root/'inputs.json');verify(root,r)
    out=root/'response-result';tick=time.perf_counter();cpu=time.process_time()
    descriptors=[*s['sourceFields'].values(),*s['comparisonFields']['parentIncoming'].values(),*s['templateFields'].values()]
    allowed=[x['path'] for x in descriptors]+[str(Path(s['modelRoot'])/n) for n in s['imageAssets']]
    access=quarantine(root,allowed,s['deniedNumericalInputs'],root/'unused-store');numeric=profile(torch,4)
    if numeric!=s['numericalProfile']:raise ValueError('Original numerical policy differs')
    torch.cuda.reset_peak_memory_stats();model,assets=source_checks(s,diffusers);config=read(model/'transformer/config.json')
    if [config[n] for n in ['num_layers','num_single_layers','num_attention_heads','attention_head_dim','mlp_ratio']]!=[5,20,24,128,3.]:
        raise ValueError('Original geometry differs')
    checkpoint=model/'transformer/diffusion_pytorch_model.safetensors';entries,hr=header(checkpoint)
    weights,loaded=load_selected(checkpoint,PARAMETERS,entries)
    if loaded!=s['originalParameters']:raise ValueError('Original parameter identity differs')
    with torch.device('meta'):block=native.Flux2SingleTransformerBlock(3072,24,128,mlp_ratio=3.,eps=1e-6,bias=False)
    block.load_state_dict({n[len(PREFIX):]:v for n,v in weights.items()},strict=True,assign=True);block.eval().requires_grad_(False)
    source={n:load(s['sourceFields'][n],'cuda') for n in SOURCE_NAMES}
    parent={n:load(s['comparisonFields']['parentIncoming'][n],'cuda') for n in ['text','image']}
    templates={n:load(s['templateFields'][n],'cuda').contiguous() for n in TEMPLATE_NAMES}
    shapes={'text':[1,512,3072],'image':[1,1024,3072],'mod':[1,9216],'cosine':[1536,128],'sine':[1536,128]}
    if {n:list(v.shape) for n,v in source.items()}!=shapes or any(v.dtype!=(torch.float32 if n in ['cosine','sine'] else torch.bfloat16) for n,v in source.items()):
        raise ValueError('Incoming source ABIs differ')
    if {n:list(v.shape) for n,v in parent.items()}!={n:shapes[n] for n in parent} or any(v.dtype!=torch.bfloat16 for v in parent.values()):
        raise ValueError('Parent source ABI differs')
    expected={'gu_basis':[2,3072,128],'gu_projectedGU':[2,128,18432],'rowBasis':[1,1536,128]}
    if {n:list(v.shape) for n,v in templates.items()}!=expected or any(v.dtype!=torch.float64 for v in templates.values()):
        raise ValueError('Template ABI differs')
    manifest=identity({'originalParameters':loaded,'intakeSHA256':s['intakeSHA256'],'nativeSources':s['nativeSources'],'templateFields':s['templateFields']})
    ledger=Ledger(r['requestSHA256'],manifest)
    all_inputs={**source,**{'parent_'+n:v for n,v in parent.items()},**templates,**{'parameter_'+n:v for n,v in weights.items()}}
    before={n:payload(v) for n,v in all_inputs.items()};census={n:[v.data_ptr(),v._version] for n,v in all_inputs.items()}
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all();np_rng=np.random.get_state()
    original_sdpa=torch.nn.functional.scaled_dot_product_attention;original_silu=torch.nn.functional.silu
    setup=time.perf_counter()-tick;sdpa=[];fields={};traces={};physical={};retained_fingerprints={}
    try:
        for arm in ['parent','own']:
            current={**source,**(parent if arm=='parent' else {})}
            inputs={**current,**templates,**{'parameter_'+n:v for n,v in weights.items()}}
            def observe(weight):
                physical[arm]={'sourceParameter':PARAMETERS[0],'rowInterval':[0,9216],'shape':list(weight.shape),
                               'payloadSHA256':payload(weight),'bytes':weight.numel()*weight.element_size()}
            def operation(values):
                for n,v in weights.items():
                    if v.data_ptr()!=values['parameter_'+n].data_ptr():raise ValueError('Typed original parameter binding differs')
                got,trace=compute(block,{n:values[n] for n in SOURCE_NAMES},{n:values[n] for n in TEMPLATE_NAMES},
                    native,native.dispatch_attention_fn,ledger.linear,ledger.moment,ledger.matmul,observe)
                traces[arm]=trace;return got
            torch.nn.functional.scaled_dot_product_attention=wrap_sdpa(original_sdpa,{'stage':arm},sdpa)
            print('TEMPLATE_TRANSPORT_START',arm,flush=True)
            with torch.inference_mode():
                got=ledger.run(inputs,operation,abis(),arm)
                captured={n:payload(v) for n,v in got.items()}
                fields[arm]={n:retain(root/'arrays/candidate'/arm,n,v) for n,v in got.items()}
                if captured!={n:payload(v) for n,v in got.items()} or captured!={n:v['payloadSHA256'] for n,v in fields[arm].items()}:
                    raise ValueError('Field capture/retention mutated')
                retained_fingerprints[arm]=captured
                del got
            print('TEMPLATE_TRANSPORT_RETAINED',arm,flush=True)
    finally:
        torch.nn.functional.scaled_dot_product_attention=original_sdpa
    if torch.nn.functional.silu is not original_silu or before!={n:payload(v) for n,v in all_inputs.items()} or census!={n:[v.data_ptr(),v._version] for n,v in all_inputs.items()} or snapshot(torch)!=numeric:
        raise ValueError('Source/template/parameter/profile closure differs')
    after=np.random.get_state()
    if not torch.equal(rng,torch.get_rng_state()) or len(cuda_rng)!=len(torch.cuda.get_rng_state_all()) or not all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all())) or np_rng[0]!=after[0] or not np.array_equal(np_rng[1],after[1]) or np_rng[2:]!=after[2:]:
        raise ValueError('Global RNG changed')
    costs=work();macs=sum(x['nominalMACs'] for x in ledger.products+ledger.linears+ledger.moments)+sum(x['nominalQKAndAVMACs'] for x in sdpa)
    if [len(ledger.products),len(ledger.linears),len(ledger.moments),len(ledger.qrs),len(sdpa)]!=[20,2,10,0,2] or macs!=costs['candidateMACs'] or macs!=r['prospectiveCandidateMACs']:
        raise ValueError('Actual two-input operator census differs')
    assets_unchanged(assets)
    save(out/'candidate.json',{'fields':fields,'requestSHA256':r['requestSHA256'],'intakeSHA256':s['intakeSHA256']})
    save(out/'execution.json',{'traces':traces,'physicalQKV':physical,'retainedFingerprints':retained_fingerprints,
        'products':ledger.products,'linears':ledger.linears,'momentProducts':ledger.moments,'SDPA':sdpa,'QR':ledger.qrs,
        'work':costs,'candidateMatrixMACs':macs,'loadedParameters':loaded,'parameterManifestSHA256':manifest,
        'typedActs':ledger.records,'timings':ledger.timings,'sourceParameterTemplateRNGClosure':True,'numericalProfile':numeric,
        'header':hr,'assets':assets,'setupSecondsAfterImports':setup,'wallSecondsAfterImports':time.perf_counter()-tick,
        'CPUSecondsAfterImports':time.process_time()-cpu,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),
        'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),
        'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':sorted(set(s['deniedNumericalInputs'])),**access}})
    save(out/'candidate-seal.json',{'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')})
    print('TEMPLATE_TRANSPORT_SEALED',sha(out/'candidate-seal.json'),flush=True)


if __name__=='__main__':execute(sys.argv[1])
