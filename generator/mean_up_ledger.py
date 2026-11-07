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
        spec=ActSpec('tce.mean_up_'+name,'v1',reads=tuple(inputs),writes=tuple(outputs),write_kinds={n:'tensor' for n in outputs},write_abis=outputs,
            numerical_contract='One source-input mean-Up completion: actual normalized stream FP32 means admittedBF16, originaltwo-rowUp andownselectedUp, fullcurrentgates/product/fullconcatoutput; originalmaps/states paid')
        def act(values):
            torch.cuda.synchronize();tick=time.perf_counter();result=operation(values);torch.cuda.synchronize()
            self.timings.append({'stage':name,'instrumentedSynchronizedSecondsBeforeFinalOutputFingerprint':time.perf_counter()-tick,
                'scope':'Operation time including any internal diagnostics/copies; excludes subsequent finite checks and output fingerprints.'});got.update(result)
            if {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in result.items()}!=outputs:raise ValueError('Trajectory component output ABI differs')
            for value in result.values():
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite typed gate-first output')
            return result,{'outputPayloads':{n:payload(v) for n,v in result.items()}}
        registry.register(spec,name,act);after,evidence=registry.run(spec.identity,name,frame)
        if before!={n:payload(v) for n,v in inputs.items()}:raise ValueError('Read-only trajectory operand mutation')
        self.records.append({'stage':name,'sourceFrame':frame.to_dict(),'outputFrame':after.to_dict(),'evidence':evidence.to_dict()})
        return got
