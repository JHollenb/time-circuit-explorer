"""Receiver-weighted source-only output producer; seal/exit before82/85/73 target access."""
from pathlib import Path
import sys
import time
from encoder_atoms import read,save,sha,identity


def execute(root):
    import torch
    import numpy as np
    import hashlib
    import diffusers
    from receiver_row_output import compute,abis,work,SOURCE_NAMES,PARAMETER
    from receiver_row_output_ledger import Ledger
    from paired_fields import load,retain,payload
    from terminal_capture_guard_v2 import quarantine
    from complete_masked_study import profile
    from flux_clock_study import verify,source_checks,assets_unchanged
    from flux_clock import header,snapshot
    from projected_complement import load_selected
    root=Path(root).resolve();r=read(root/'request.json');s=read(root/'inputs.json');verify(root,r);out=root/'response-result';started=time.perf_counter();cpu=time.process_time()
    allowed=[item['path'] for item in s['sourceFields'].values()]+[str(Path(s['modelRoot'])/name) for name in s['imageAssets']]
    access=quarantine(root,allowed,s['deniedNumericalInputs'],root/'unused-store');numeric=profile(torch,4)
    if numeric!=s['numericalProfile']:raise ValueError('Pinned numerical policy differs')
    torch.backends.cuda.preferred_linalg_library('cusolver');linalg=str(torch.backends.cuda.preferred_linalg_library())
    qr_profile={'requestedPreference':'cusolver','returnedPreference':linalg,'torch':torch.__version__,'torchGitVersion':torch.version.git_version,'CUDA':torch.version.cuda,'buildConfig':torch.__config__.show(),
        'operator':'torch.linalg.qr(mode=reduced)','dtype':'torch.float64','device':'cuda','scope':'Requested preference may fall back if unsupported; observed configuration/calls, not vendor-kernel attestation. No retries.'}
    torch.cuda.reset_peak_memory_stats();model,assets=source_checks(s,diffusers);checkpoint=model/'transformer/diffusion_pytorch_model.safetensors';entries,hr=header(checkpoint)
    weights,loaded=load_selected(checkpoint,[PARAMETER],entries)
    if loaded!=s['originalParameters']:raise ValueError('Original complete output parameter differs')
    source={name:load(s['sourceFields'][name],'cuda') for name in SOURCE_NAMES};weight=weights[PARAMETER]
    expected={'product':([1,1536,9216],torch.bfloat16),'attentionRead':([1,1536,3072],torch.bfloat16),
              'gate':([1,1,3072],torch.bfloat16),'text':([1,512,3072],torch.bfloat16),'image':([1,1024,3072],torch.bfloat16)}
    if any(list(value.shape)!=expected[name][0] or value.dtype!=expected[name][1] for name,value in source.items()):raise ValueError('Five admitted source ABIs differ')
    manifest=identity({'originalParameters':loaded,'parentCandidateSealSHA256':s['parentCandidateSealSHA256'],'nativeSources':s['nativeSources']});ledger=Ledger(r['requestSHA256'],manifest);trace={};join={};sketch={}
    all_inputs={**source,'originalOut':weight};before={name:payload(value) for name,value in all_inputs.items()};census={name:[value.data_ptr(),value._version] for name,value in all_inputs.items()}
    rng=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all();np_state=np.random.get_state();setup=time.perf_counter()-started
    def observe_join(value):join.update(payloadSHA256=payload(value),shape=list(value.shape),dtype=str(value.dtype),bytes=value.numel()*value.element_size(),sourceFields=['attentionRead','product'])
    def observe_sketch(raw,omega,gated_omega,pulled):sketch.update(seedSequence=[20261107,86],bitGenerator='PCG64',numpy=np.__version__,rawWordDtype=str(raw.dtype),rawWordsSHA256=hashlib.sha256(raw.tobytes(order='C')).hexdigest(),signsPayloadSHA256=payload(omega),gatedSignsPayloadSHA256=payload(gated_omega),pulledBackPayloadSHA256=payload(pulled),shape=list(omega.shape),scope='One shared unscaled output-coordinate low-bit sketch, current gate scaling and paid original-map pullback; local generator, no global RNG change.')
    def operation(values):
        got,observed=compute({name:values[name] for name in SOURCE_NAMES},values['originalOut'],ledger.matmul,ledger.qr,observe_join,observe_sketch);trace.update(observed);return got
    print('RECEIVER_ROW_OUTPUT_START',flush=True)
    with torch.inference_mode():
        got=ledger.run(all_inputs,operation,abis());fields={name:retain(root/'arrays/candidate',name,value) for name,value in got.items()}
    if before!={name:payload(value) for name,value in all_inputs.items()} or census!={name:[value.data_ptr(),value._version] for name,value in all_inputs.items()} or snapshot(torch)!=numeric or str(torch.backends.cuda.preferred_linalg_library())!=linalg:raise ValueError('Source/parameter/numerical closure differs')
    if not torch.equal(rng,torch.get_rng_state()) or len(rng_cuda)!=len(torch.cuda.get_rng_state_all()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise ValueError('Unexpected random computation')
    np_after=np.random.get_state()
    if np_state[0]!=np_after[0] or not np.array_equal(np_state[1],np_after[1]) or np_state[2:]!=np_after[2:]:raise ValueError('Global NumPy RNG changed')
    costs=work()
    if sum(item['nominalMACs'] for item in ledger.products)!=costs['candidateMACs'] or costs['candidateMACs']!=r['prospectiveCandidateMACs'] or len(ledger.products)!=9 or len(ledger.qrs)!=2:raise ValueError('Actual output work instrument differs')
    assets_unchanged(assets)
    save(out/'candidate.json',{'fields':fields,'requestSHA256':r['requestSHA256'],'sourceParentCandidateSealSHA256':s['parentCandidateSealSHA256']})
    save(out/'execution.json',{'trace':trace,'actualBF16Join':join,'outputSketch':sketch,'products':ledger.products,'QR':ledger.qrs,'QRProfile':qr_profile,'work':costs,'candidateMatrixMACs':costs['candidateMACs'],
        'loadedParameters':loaded,'parameterManifestSHA256':manifest,'typedActs':ledger.records,'timings':ledger.timings,'sourceParameterRNGClosure':True,'numericalProfile':numeric,'header':hr,'assets':assets,
        'setupSecondsAfterImports':setup,'wallSecondsAfterImports':time.perf_counter()-started,'CPUSecondsAfterImports':time.process_time()-cpu,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),
        'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':sorted(set(s['deniedNumericalInputs'])),**access}})
    save(out/'candidate-seal.json',{'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')})
    print('RECEIVER_ROW_OUTPUT_SEALED',sha(out/'candidate-seal.json'),flush=True)


if __name__=='__main__':execute(sys.argv[1])
