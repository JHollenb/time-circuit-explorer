"""Same numerical operators; typed contract identifies the matched parent input."""
import time
from own_successor_ledger import Ledger as PreviousLedger
from paired_fields import payload


class Ledger(PreviousLedger):
    def run(self,inputs,operation,outputs):
        import torch
        from mdb.titan import Frame,ActSpec,ActRegistry,ProvidedFingerprint
        before={n:payload(v) for n,v in inputs.items()};got={};registry=ActRegistry()
        frame=Frame.from_values(inputs,kinds={n:'tensor' for n in inputs},
            abis={n:{'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype)} for n,v in inputs.items()},
            slot_fingerprints={n:ProvidedFingerprint.content_sha256(h) for n,h in before.items()},
            program_fingerprint=self.request_id,metadata={'phase':'matchedParentSuccessor','originalParameterManifestSHA256':self.manifest})
        spec=ActSpec('tce.matched_parent_successor','v1',reads=tuple(inputs),writes=tuple(outputs),
            write_kinds={n:'tensor' for n in outputs},write_abis=outputs,
            numerical_contract='Authenticated70/73 parent input plus paid70clock/RoPE and four original single1 tensors. '
            'Identical frozen87 compute: own norm/mod, physical sliced BF16QKV,nativeRMS/RoPE/liveattention/moments, '
            'frozen82operand-onlyGU,frozen86receiveroutput,BF16gate/add. All bases reformed. '
            'No saved own-state outputs/bases/targets before seal and observed exit. Learned work paid.')
        def act(values):
            torch.cuda.synchronize();tick=time.perf_counter();result=operation(values);torch.cuda.synchronize()
            self.timings.append({'stage':'matchedParentSuccessor','instrumentedSynchronizedSecondsBeforeFinalFingerprints':time.perf_counter()-tick})
            if {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in result.items()}!=outputs or any(not bool(torch.isfinite(v).all()) for v in result.values()):
                raise ValueError('Matched-parent finite output ABI differs')
            got.update(result);return result,{'outputPayloads':{n:payload(v) for n,v in result.items()}}
        registry.register(spec,'matchedParentSuccessor',act)
        after,evidence=registry.run(spec.identity,'matchedParentSuccessor',frame)
        if before!={n:payload(v) for n,v in inputs.items()}:raise ValueError('Read-only source mutation')
        self.records.append({'stage':'matchedParentSuccessor','sourceFrame':frame.to_dict(),'outputFrame':after.to_dict(),'evidence':evidence.to_dict()})
        return got
