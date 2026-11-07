"""Typed component boundaries for the own-state trajectory, no model loading."""
import time
from paired_fields import payload


class Ledger:
    def __init__(self,request_id,parameter_manifest):
        self.request_id=request_id;self.parameter_manifest=parameter_manifest;self.records=[];self.timings=[]

    def run(self,name,inputs,operation,outputs,metadata=None):
        import torch
        from mdb.titan import Frame,ActSpec,ActRegistry,ProvidedFingerprint
        before={n:payload(v) for n,v in inputs.items()};registry=ActRegistry();got={}
        frame=Frame.from_values(inputs,kinds={n:'tensor' for n in inputs},abis={n:{'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype)} for n,v in inputs.items()},
            slot_fingerprints={n:ProvidedFingerprint.content_sha256(h) for n,h in before.items()},program_fingerprint=self.request_id,
            metadata={'phase':name,'originalParameterManifestSHA256':self.parameter_manifest,**(metadata or {})})
        spec=ActSpec('tce.gate_trajectory_'+name,'v1',reads=tuple(inputs),writes=tuple(outputs),write_kinds={n:'tensor' for n in outputs},write_abis=outputs,
            numerical_contract='FullcurrentG and fixedquarter selectedU/down, cachedpaid source norms; own coupled states, full image attention; FP32 post-RoPE population mean/full covariance image log-partition, TT logsumexp/conditional AV times own alpha before BF16 output, terminal text write omitted; architectural and finite discrepancy measured after seal')
        def act(values):
            torch.cuda.synchronize();tick=time.perf_counter();result=operation(values);torch.cuda.synchronize()
            self.timings.append({'stage':name,'instrumentedSynchronizedSecondsBeforeFinalOutputFingerprint':time.perf_counter()-tick,
                'scope':'Includes map and block fingerprints and progress fsync within the operation; excludes only the final Act-output fingerprint.'});got.update(result)
            if {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in result.items()}!=outputs:raise ValueError('Trajectory component output ABI differs')
            if any(v.is_floating_point() and not bool(torch.isfinite(v).all()) for v in result.values()):raise ValueError('Nonfinite trajectory output')
            return result,{'outputPayloads':{n:payload(v) for n,v in result.items()}}
        registry.register(spec,name,act);after,evidence=registry.run(spec.identity,name,frame)
        if before!={n:payload(v) for n,v in inputs.items()}:raise ValueError('Read-only trajectory operand mutation')
        self.records.append({'stage':name,'sourceFrame':frame.to_dict(),'outputFrame':after.to_dict(),'evidence':evidence.to_dict()})
        return got
