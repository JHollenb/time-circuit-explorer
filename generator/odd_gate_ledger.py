"""Typed isolated quadratic producer transaction."""
import time
from paired_fields import payload


class Ledger:
    def __init__(self,request_id,manifest):self.request_id=request_id;self.manifest=manifest;self.records=[];self.timings=[]

    def run(self,inputs,operation,outputs,metadata):
        import torch
        from mdb.titan import Frame,ActSpec,ActRegistry,ProvidedFingerprint
        before={n:payload(v) for n,v in inputs.items()};registry=ActRegistry();got={}
        frame=Frame.from_values(inputs,kinds={n:'tensor' for n in inputs},abis={n:{'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype)} for n,v in inputs.items()},
            slot_fingerprints={n:ProvidedFingerprint.content_sha256(h) for n,h in before.items()},program_fingerprint=self.request_id,
            metadata={'phase':'oddGate','sourceManifestSHA256':self.manifest,**metadata})
        spec=ActSpec('tce.odd_gate','v1',reads=tuple(inputs),writes=tuple(outputs),write_kinds={n:'tensor' for n in outputs},write_abis=outputs,
            numerical_contract='Paid FP64 gate projection plus Down for half centeredG*centeredU; inheritedL gives full correction. ActualA and old error comparisons unavailable until producer exits. No nativeG/image/cheapcompiler claim.')
        def act(values):
            torch.cuda.synchronize();tick=time.perf_counter();result=operation(values);torch.cuda.synchronize()
            self.timings.append({'stage':'oddGate','instrumentedSynchronizedSecondsBeforeFinalOutputFingerprint':time.perf_counter()-tick,
                'scope':'Two projections/means/product/gating/forecast plus internal affine fingerprints; excludes outer input hashes, subsequent ABI/finite/final output hashes and array writes.'})
            if {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in result.items()}!=outputs or any(not bool(torch.isfinite(v).all()) for v in result.values()):raise ValueError('Producer finite ABI differs')
            got.update(result);return result,{'outputPayloads':{n:payload(v) for n,v in result.items()}}
        registry.register(spec,'oddGate',act);after,evidence=registry.run(spec.identity,'oddGate',frame)
        if before!={n:payload(v) for n,v in inputs.items()}:raise ValueError('Read-only producer input mutation')
        self.records.append({'stage':'oddGate','sourceFrame':frame.to_dict(),'outputFrame':after.to_dict(),'evidence':evidence.to_dict()});return got
