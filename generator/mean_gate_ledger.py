"""One typed producer transaction; no comparison or native model access."""
import time
from paired_fields import payload


class Ledger:
    def __init__(self, request_id, manifest):
        self.request_id=request_id; self.manifest=manifest; self.records=[]; self.timings=[]

    def run(self, inputs, operation, outputs, metadata):
        import torch
        from mdb.titan import Frame, ActSpec, ActRegistry, ProvidedFingerprint
        before={n:payload(v) for n,v in inputs.items()}; registry=ActRegistry(); got={}
        frame=Frame.from_values(inputs, kinds={n:'tensor' for n in inputs},
            abis={n:{'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype)} for n,v in inputs.items()},
            slot_fingerprints={n:ProvidedFingerprint.content_sha256(h) for n,h in before.items()},
            program_fingerprint=self.request_id, metadata={'phase':'meanGate','sourceManifestSHA256':self.manifest,**metadata})
        spec=ActSpec('tce.mean_gate','v1',reads=tuple(inputs),writes=tuple(outputs),
            write_kinds={n:'tensor' for n in outputs},write_abis=outputs,
            numerical_contract='Authenticated paid A/Ustar/upMean/support/D/g; derive meanA and one FP64 Down. No comparison targets, oracle means, new Up, native block or continuation.')
        def act(values):
            torch.cuda.synchronize(); tick=time.perf_counter(); result=operation(values); torch.cuda.synchronize()
            self.timings.append({'stage':'meanGate','instrumentedSynchronizedSecondsBeforeFinalOutputFingerprint':time.perf_counter()-tick,
                'scope':'Mean/product/one Down/gating/centering plus internal affine fingerprints; excludes outer input hashes, final ABI/finite/output hashes and retained-array writes.'})
            if {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in result.items()}!=outputs: raise ValueError('Producer output ABI differs')
            if any(not bool(torch.isfinite(v).all()) for v in result.values()): raise ValueError('Nonfinite producer output')
            got.update(result); return result, {'outputPayloads':{n:payload(v) for n,v in result.items()}}
        registry.register(spec,'meanGate',act); after,evidence=registry.run(spec.identity,'meanGate',frame)
        if before!={n:payload(v) for n,v in inputs.items()}: raise ValueError('Read-only producer operand mutation')
        self.records.append({'stage':'meanGate','sourceFrame':frame.to_dict(),'outputFrame':after.to_dict(),'evidence':evidence.to_dict()})
        return got
