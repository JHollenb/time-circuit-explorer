"""One source-bound own-state FP64 diagnostic, no dense model object or image."""
from pathlib import Path
import sys
import time
from encoder_atoms import read,save,sha,identity

PARAMETERS=['single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight','single_transformer_blocks.0.attn.to_out.weight']


def execute(root):
    import torch
    import diffusers
    from paired_fields import load,retain,payload
    from gated_covariance import compute,abis,SOURCE_NAMES
    from gated_covariance_ledger import Ledger
    from test_gated_covariance import mechanics
    from terminal_capture_guard_v2 import quarantine
    from flux_clock_study import verify,source_checks,assets_unchanged
    from flux_clock import header,snapshot
    from projected_complement import load_selected
    from complete_masked_study import profile
    root=Path(root).resolve();r=read(root/'request.json');s=read(root/'inputs.json');verify(root,r);out=root/'response-result'
    started=time.perf_counter();cpu=time.process_time();save(out/'toy-mechanics.json',mechanics())
    allowed=[x['path'] for x in s['sourceFields'].values()]+[str(Path(s['modelRoot'])/name) for name in s['imageAssets']]
    access=quarantine(root,allowed,s['deniedNumericalInputs'],root/'unused-store');numeric=profile(torch,4)
    if numeric!=s['numericalProfile']:raise ValueError('Source numerical policy differs')
    torch.cuda.reset_peak_memory_stats();model,assets=source_checks(s,diffusers);entries,hr=header(model/'transformer/diffusion_pytorch_model.safetensors')
    weights,loaded=load_selected(model/'transformer/diffusion_pytorch_model.safetensors',PARAMETERS,entries)
    if loaded!=s['originalParameters']:raise ValueError('Two original parameters differ')
    source={name:load(s['sourceFields'][name],'cuda') for name in SOURCE_NAMES}
    expected={'normalized':[1,1536,3072],'activation':[1,1536,9216],'upRaw':[1,1536,2304],'completedUp':[1,1536,9216],'product':[1,1536,9216],
        'streamMeanFP32':[1,2,3072],'streamMean':[1,2,3072],'anchorUp':[1,2,9216],'gate':[1,1,3072],'selection_support':[2304]}
    if {n:list(v.shape) for n,v in source.items()}!=expected or any(v.dtype!=({'streamMeanFP32':torch.float32,'selection_support':torch.int64}.get(n,torch.bfloat16)) for n,v in source.items()):raise ValueError('Own receiver geometry/dtype differs')
    support=source['selection_support']
    if not torch.equal(support,support.unique(sorted=True)) or int(support.min())<0 or int(support.max())>=9216:raise ValueError('Source support integrity differs')
    up=weights[PARAMETERS[0]][18432:27648];down=weights[PARAMETERS[1]][:,3072:12288]
    views={'U':{'sourceParameter':PARAMETERS[0],'rows':[18432,27648],'shape':list(up.shape),'payloadSHA256':payload(up),'bytes':up.numel()*2},
           'D':{'sourceParameter':PARAMETERS[1],'columns':[3072,12288],'shape':list(down.shape),'payloadSHA256':payload(down),'bytes':down.numel()*2}}
    manifest=identity({'originalParameters':loaded,'ownParentSealSHA256':s['parentSealSHA256'],'completionOutputFrameID':s['completionOutputFrameID'],'helperSHA256':r['sourceHashes']['gated_covariance.py']})
    ledger=Ledger(r['requestSHA256'],manifest);operations=[];before={n:payload(v) for n,v in weights.items()};census={n:[v.data_ptr(),v._version,list(v.shape),str(v.dtype)] for n,v in weights.items()}
    source_before={n:payload(v) for n,v in source.items()};rng=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all();setup=time.perf_counter()-started
    def affine(name,x,w):
        if x.dtype!=torch.float64 or w.dtype!=torch.float64 or x.device.type!='cuda' or w.device.type!='cuda' or torch.is_autocast_enabled('cuda'):raise ValueError('Explicit FP64 CUDA projection required')
        value=torch.nn.functional.linear(x,w)
        operations.append({'name':name,'inputShape':list(x.shape),'weightShape':list(w.shape),'outputShape':list(value.shape),'dtype':str(value.dtype),'device':value.device.type,
            'inputPayloadSHA256':payload(x),'weightPayloadSHA256':payload(w),'outputPayloadSHA256':payload(value),'nominalMACs':x.numel()//x.shape[-1]*w.numel()})
        return value
    ins={**source,'U':up,'D':down,**{'parameter_'+n:v for n,v in weights.items()}}
    def operation(values):return compute({n:values[n] for n in SOURCE_NAMES},values['U'],values['D'],affine)
    print('GATED_COVARIANCE_START',flush=True)
    with torch.inference_mode():
        got=ledger.run('ownStateCovariance',ins,operation,abis(),{'parentSealSHA256':s['parentSealSHA256'],'completionOutputFrameID':s['completionOutputFrameID'],'fixedAddress':{'clock':3,'singleBlock':0},'sourceViews':views})
        fields={name:retain(root/'arrays/diagnostic',name,v) for name,v in got.items()}
    if before!={n:payload(v) for n,v in weights.items()} or source_before!={n:payload(v) for n,v in source.items()} or census!={n:[v.data_ptr(),v._version,list(v.shape),str(v.dtype)] for n,v in weights.items()} or snapshot(torch)!=numeric:raise ValueError('Source/parameter/numeric closure differs')
    if not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise ValueError('Unexpected stochastic computation')
    if [x['name'] for x in operations]!=['fullUp','totalErrorDown','centeredDown'] or sum(x['nominalMACs'] for x in operations)!=130459631616:raise ValueError('Declared three-product work differs')
    assets_unchanged(assets)
    save(out/'candidate.json',{'fields':fields,'requestSHA256':r['requestSHA256'],'sourceParentSealSHA256':s['parentSealSHA256'],'scope':'Reconstructed FP64 MLP-write discrepancy onone supplied ownstate; no nativefiniteconcatblockerror or new serving producer.'})
    save(out/'execution.json',{'operations':operations,'totalMatrixMACs':sum(x['nominalMACs'] for x in operations),'loadedParameters':loaded,'sourceViews':views,'parameterManifestSHA256':manifest,'typedActs':ledger.records,'timings':ledger.timings,
        'sourceParameterRNGClosure':True,'numericalProfile':numeric,'header':hr,'assets':assets,'setupSecondsAfterImports':setup,'wallSecondsAfterImports':time.perf_counter()-started,'CPUSecondsAfterImports':time.process_time()-cpu,
        'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),'newNativeWholeModelReferences':0,'newFullUpReferenceProjections':1,'newImages':0,'newAttentionCalls':0,'newHeadCalls':0,'newVAECalls':0,
        'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':s['deniedNumericalInputs'],**access}})
    save(out/'candidate-seal.json',{'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')})
    print('GATED_COVARIANCE_SEALED',sha(out/'candidate-seal.json'),flush=True)


if __name__=='__main__':execute(sys.argv[1])
