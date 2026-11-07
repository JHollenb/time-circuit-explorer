"""Target-quarantined saved-operand producer; exits before comparisons open."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha, identity


def execute(root):
    import torch
    from paired_fields import load, retain, payload
    from mean_gate import compute, abis, SOURCE_NAMES
    from mean_gate_ledger import Ledger
    from test_mean_gate import mechanics
    from terminal_capture_guard_v2 import quarantine
    from flux_clock_study import verify
    from flux_clock import snapshot
    from complete_masked_study import profile
    root=Path(root).resolve(); r=read(root/'request.json'); s=read(root/'inputs.json'); verify(root,r); out=root/'response-result'
    started=time.perf_counter(); cpu=time.process_time(); save(out/'toy-mechanics.json',mechanics())
    allowed=[x['path'] for x in s['sourceFields'].values()]
    access=quarantine(root,allowed,s['deniedNumericalInputs'],root/'unused-store'); numeric=profile(torch,4)
    if numeric!=s['numericalProfile'] or torch.__version__!=s['torchVersion']: raise ValueError('Inherited numeric profile differs')
    torch.cuda.reset_peak_memory_stats()
    source={n:load(s['sourceFields'][n],'cuda') for n in SOURCE_NAMES}
    expected={'activation':([1,1536,9216],torch.bfloat16),'upReference':([1,1536,9216],torch.float64),
        'upMean':([1,2,9216],torch.float64),'selection_support':([2304],torch.int64),
        'weightD':([3072,9216],torch.float64),'gate':([1,1,3072],torch.bfloat16)}
    if any((list(source[n].shape),source[n].dtype)!=v for n,v in expected.items()): raise ValueError('Paid source geometry/dtype differs')
    support=source['selection_support']
    if not torch.equal(support,support.unique(sorted=True)) or int(support.min())<0 or int(support.max())>=9216: raise ValueError('Support integrity differs')
    manifest=identity({'parentSealSHA256':s['parentSealSHA256'],'sourceFields':s['sourceFields'],'helperSHA256':r['sourceHashes']['mean_gate.py']})
    ledger=Ledger(r['requestSHA256'],manifest); operations=[]; before={n:payload(v) for n,v in source.items()}
    census={n:[v.data_ptr(),v._version,list(v.shape),str(v.dtype)] for n,v in source.items()}
    rng=torch.get_rng_state(); rng_cuda=torch.cuda.get_rng_state_all(); setup=time.perf_counter()-started
    def affine(name,x,w):
        if x.dtype!=torch.float64 or w.dtype!=torch.float64 or x.device.type!='cuda' or w.device.type!='cuda' or torch.is_autocast_enabled('cuda'): raise ValueError('FP64 CUDA Down required')
        value=torch.nn.functional.linear(x,w)
        operations.append({'name':name,'inputShape':list(x.shape),'weightShape':list(w.shape),'outputShape':list(value.shape),
            'dtype':str(value.dtype),'device':value.device.type,'inputPayloadSHA256':payload(x),
            'weightPayloadSHA256':payload(w),'outputPayloadSHA256':payload(value),'nominalMACs':x.numel()//x.shape[-1]*w.numel()})
        return value
    print('MEAN_GATE_START',flush=True)
    with torch.inference_mode():
        got=ledger.run(source,lambda v:compute(v,affine),abis(),{'parentSealSHA256':s['parentSealSHA256'],'parentOutputFrameID':s['parentOutputFrameID'],'fixedAddress':{'clock':3,'singleBlock':0}})
        fields={n:retain(root/'arrays/producer',n,v) for n,v in got.items()}
    if before!={n:payload(v) for n,v in source.items()} or census!={n:[v.data_ptr(),v._version,list(v.shape),str(v.dtype)] for n,v in source.items()} or snapshot(torch)!=numeric: raise ValueError('Source/numeric closure differs')
    if not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())): raise ValueError('Unexpected stochastic computation')
    if [x['name'] for x in operations]!=['meanGateDown'] or sum(x['nominalMACs'] for x in operations)!=43486543872: raise ValueError('One Down work differs')
    save(out/'candidate.json',{'fields':fields,'requestSHA256':r['requestSHA256'],'sourceParentSealSHA256':s['parentSealSHA256'],
        'scope':'Paid receiver-conditioned linear write only. No covariance/oracle mean correction or comparison target supplied.'})
    save(out/'execution.json',{'operations':operations,'totalMatrixMACs':43486543872,'sourceManifestSHA256':manifest,
        'typedActs':ledger.records,'timings':ledger.timings,'sourceRNGClosure':True,'numericalProfile':numeric,
        'setupSecondsAfterImports':setup,'wallSecondsAfterImports':time.perf_counter()-started,'CPUSecondsAfterImports':time.process_time()-cpu,
        'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),
        'newUpProjections':0,'newOriginalCheckpointTensorLoads':0,'savedOriginalDownMapLoads':1,'savedOriginalDownMapBytes':source['weightD'].numel()*8,
        'newNativeReferences':0,'newImages':0,'newHeadCalls':0,'newVAECalls':0,
        'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':s['deniedNumericalInputs'],**access}})
    save(out/'candidate-seal.json',{'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')})
    print('MEAN_GATE_SEALED',sha(out/'candidate-seal.json'),flush=True)


if __name__=='__main__': execute(sys.argv[1])
