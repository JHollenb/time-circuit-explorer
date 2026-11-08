"""Typed receiver-weighted row-output contract and immutable FP64 instruments."""
import time
from gate_up_basis_ledger import Ledger as Operations
from paired_fields import payload


class Ledger(Operations):
    def run(self,inputs,operation,outputs):
        import torch
        from mdb.titan import Frame,ActSpec,ActRegistry,ProvidedFingerprint
        before={name:payload(value) for name,value in inputs.items()};registry=ActRegistry();got={}
        frame=Frame.from_values(inputs,kinds={name:'tensor' for name in inputs},abis={name:{'shape':list(value.shape),'stride':list(value.stride()),'dtype':str(value.dtype)} for name,value in inputs.items()},
            slot_fingerprints={name:ProvidedFingerprint.content_sha256(digest) for name,digest in before.items()},program_fingerprint=self.request_id,
            metadata={'phase':'receiverRowOutput','originalParameterManifestSHA256':self.manifest})
        spec=ActSpec('tce.receiver_row_output','v1',reads=tuple(inputs),writes=tuple(outputs),write_kinds={name:'tensor' for name in outputs},write_abis=outputs,
            numerical_contract='Own82 product, paid read/gate/residual and originalout. Fixed shared output PCG64 signs, V=Wout.T diag(g)Omega once, J V->canonical FP64 QR; U.TJ->originalWout->U reconstruction, one final FP64->FP32->BF16 admission, separate BF16 gate/add. Oldbases/coordinates and82/85/73 targets denied until producer seal/exit; source/maps/history paid.')
        def act(values):
            torch.cuda.synchronize();tick=time.perf_counter();result=operation(values);torch.cuda.synchronize()
            self.timings.append({'stage':'receiverRowOutput','instrumentedSynchronizedSecondsBeforeFinalFingerprints':time.perf_counter()-tick,'scope':'PCG64/shared learned pullback/sketch/QR/output/gate/add plus internal operation/join fingerprints; excludes setup, outer/final hashes and retention.'})
            if {name:{'shape':list(value.shape),'dtype':str(value.dtype)} for name,value in result.items()}!=outputs or any(not bool(torch.isfinite(value).all()) for value in result.values()):raise ValueError('Finite output ABI differs')
            got.update(result);return result,{'outputPayloads':{name:payload(value) for name,value in result.items()}}
        registry.register(spec,'receiverRowOutput',act);after,evidence=registry.run(spec.identity,'receiverRowOutput',frame)
        if before!={name:payload(value) for name,value in inputs.items()}:raise ValueError('Source mutation')
        self.records.append({'stage':'receiverRowOutput','sourceFrame':frame.to_dict(),'outputFrame':after.to_dict(),'evidence':evidence.to_dict()});return got
