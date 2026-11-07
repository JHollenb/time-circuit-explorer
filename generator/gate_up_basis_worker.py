"""Five admitted current-state fields; targets unavailable until this process exits."""
from pathlib import Path
import sys
import time
from encoder_atoms import read,save,sha,identity


def execute(root):
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from gate_up_basis import compute,abis,work,SOURCE_NAMES,PARAMETERS
    from gate_up_basis_ledger import Ledger
    from paired_fields import load,retain,payload
    from terminal_capture_guard_v2 import quarantine
    from complete_masked_study import profile
    from flux_clock_study import verify,source_checks,assets_unchanged
    from flux_clock import header,snapshot
    from projected_complement import load_selected
    root=Path(root).resolve();r=read(root/'request.json');s=read(root/'inputs.json');verify(root,r);out=root/'response-result'
    started=time.perf_counter();cpu=time.process_time()
    allowed=[x['path'] for x in s['sourceFields'].values()]+[str(Path(s['modelRoot'])/n) for n in s['imageAssets']]
    access=quarantine(root,allowed,s['deniedNumericalInputs'],root/'unused-store');numeric=profile(torch,4)
    if numeric!=s['numericalProfile']:raise ValueError('Pinned native numerical policy differs')
    requested='cusolver';torch.backends.cuda.preferred_linalg_library(requested);linalg=str(torch.backends.cuda.preferred_linalg_library())
    qr_profile={'requestedPreference':requested,'returnedPreference':linalg,'torch':torch.__version__,'torchGitVersion':torch.version.git_version,
        'CUDA':torch.version.cuda,'buildConfig':torch.__config__.show(),'operator':'torch.linalg.qr(mode=reduced)','dtype':'torch.float64','device':'cuda',
        'scope':'Requested preference may fall back for an unsupported operation; configuration and observed calls, not vendor-kernel attestation. No backend retries.'}
    torch.cuda.reset_peak_memory_stats();model,assets=source_checks(s,diffusers)
    config=read(model/'transformer/config.json')
    if [config[n] for n in ['num_layers','num_single_layers','num_attention_heads','attention_head_dim','mlp_ratio']]!=[5,20,24,128,3.]:raise ValueError('Original block geometry differs')
    checkpoint=model/'transformer/diffusion_pytorch_model.safetensors';entries,hr=header(checkpoint);weights,loaded=load_selected(checkpoint,PARAMETERS,entries)
    if loaded!=s['originalParameters']:raise ValueError('Original two parameters differ')
    source={n:load(s['sourceFields'][n],'cuda') for n in SOURCE_NAMES}
    expected={'normalized':[1,1536,3072],'attentionRead':[1,1536,3072],'gate':[1,1,3072],'text':[1,512,3072],'image':[1,1024,3072]}
    if {n:list(v.shape) for n,v in source.items()}!=expected or any(v.dtype!=torch.bfloat16 for v in source.values()):raise ValueError('Five admitted source ABIs differ')
    activation=native.Flux2SwiGLU().eval();manifest=identity({'originalParameters':loaded,'parentDenseSealSHA256':s['parentDenseSealSHA256'],'nativeSources':s['nativeSources']})
    ledger=Ledger(r['requestSHA256'],manifest);trace={};view=weights[PARAMETERS[0]][9216:]
    physical={'name':'originalGU','shape':list(view.shape),'bytes':view.numel()*view.element_size(),'payloadSHA256':payload(view),'sourceParameter':PARAMETERS[0],'rowInterval':[9216,27648]}
    before={n:payload(v) for n,v in {**source,**weights}.items()};census={n:[v.data_ptr(),v._version] for n,v in {**source,**weights}.items()}
    rng=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all();setup=time.perf_counter()-started
    def operation(values):
        got,observed=compute({n:values[n] for n in SOURCE_NAMES},values['parameter_'+PARAMETERS[0]],values['parameter_'+PARAMETERS[1]],activation,ledger.matmul,ledger.qr)
        trace.update(observed);return got
    print('GATE_UP_BASIS_START',flush=True)
    with torch.inference_mode():
        got=ledger.run({**source,**{'parameter_'+n:v for n,v in weights.items()}},operation,abis())
        fields={n:retain(root/'arrays/candidate',n,v) for n,v in got.items()}
    if before!={n:payload(v) for n,v in {**source,**weights}.items()} or census!={n:[v.data_ptr(),v._version] for n,v in {**source,**weights}.items()} or snapshot(torch)!=numeric or str(torch.backends.cuda.preferred_linalg_library())!=linalg:raise ValueError('Source/parameter/profile closure differs')
    if not torch.equal(rng,torch.get_rng_state()) or len(rng_cuda)!=len(torch.cuda.get_rng_state_all()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise ValueError('Unexpected global RNG change')
    costs=work();macs=sum(x['nominalMACs'] for x in ledger.products)
    if macs!=costs['candidateMACs'] or macs!=r['prospectiveCandidateMACs'] or len(ledger.products)!=9 or len(ledger.qrs)!=2:raise ValueError('Actual producer work differs')
    assets_unchanged(assets)
    save(out/'candidate.json',{'fields':fields,'requestSHA256':r['requestSHA256'],'sourceParentDenseSealSHA256':s['parentDenseSealSHA256']})
    save(out/'execution.json',{'trace':trace,'products':ledger.products,'QR':ledger.qrs,'QRProfile':qr_profile,'work':costs,'candidateMatrixMACs':macs,
        'physicalGUView':physical,'loadedParameters':loaded,'parameterManifestSHA256':manifest,'typedActs':ledger.records,'timings':ledger.timings,
        'sourceParameterRNGClosure':True,'numericalProfile':numeric,'header':hr,'assets':assets,'setupSecondsAfterImports':setup,
        'wallSecondsAfterImports':time.perf_counter()-started,'CPUSecondsAfterImports':time.process_time()-cpu,
        'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),
        'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':sorted(set(s['deniedNumericalInputs'])),**access}})
    save(out/'candidate-seal.json',{'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')})
    print('GATE_UP_BASIS_SEALED',sha(out/'candidate-seal.json'),flush=True)


if __name__=='__main__':execute(sys.argv[1])
