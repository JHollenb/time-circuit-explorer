"""Typed dense-comparator contract and two native BF16 affine observations."""
import time
from matched_parent_ledger import Ledger as PreviousLedger
from paired_fields import payload


class Ledger(PreviousLedger):
    def linear(self,name,value,weight,bias,operation):
        import torch
        if name not in ['fullFused','fullOut'] or value.dtype!=torch.bfloat16 or weight.dtype!=torch.bfloat16 or value.device.type!='cuda' or bias is not None or torch.is_autocast_enabled('cuda'):
            raise ValueError('Original dense affine profile differs')
        torch.cuda.synchronize();tick=time.perf_counter();result=operation(value,weight,bias);torch.cuda.synchronize()
        self.linears.append({'name':name,'inputShape':list(value.shape),'weightShape':list(weight.shape),
            'outputShape':list(result.shape),'inputStride':list(value.stride()),'weightStride':list(weight.stride()),
            'outputStride':list(result.stride()),'inputPayloadSHA256':payload(value),'weightPayloadSHA256':payload(weight),
            'outputPayloadSHA256':payload(result),'dtype':str(result.dtype),'device':'cuda',
            'nominalMACs':result.numel()*value.shape[-1],
            'synchronizedSecondsIncludingFingerprints':time.perf_counter()-tick})
        return result

    def run(self,inputs,operation,outputs):
        import torch
        from mdb.titan import Frame,ActSpec,ActRegistry,ProvidedFingerprint
        before={n:payload(v) for n,v in inputs.items()};got={};registry=ActRegistry()
        frame=Frame.from_values(inputs,kinds={n:'tensor' for n in inputs},
            abis={n:{'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype)} for n,v in inputs.items()},
            slot_fingerprints={n:ProvidedFingerprint.content_sha256(h) for n,h in before.items()},
            program_fingerprint=self.request_id,metadata={'phase':'denseOwnStateCorner','originalParameterManifestSHA256':self.manifest})
        spec=ActSpec('tce.dense_own_state_corner','v1',reads=tuple(inputs),writes=tuple(outputs),
            write_kinds={n:'tensor' for n in outputs},write_abis=outputs,
            numerical_contract='Admitted actual86text/image plus paid70clock/RoPE and four original single1 tensors. '
            'Immutable70single consumer,full original fused/output geometry,nativeRMS/RoPE/SwiGLU/BF16gate/add,imageSDPA/textmoments. '
            'Observer hooks do not replace returns. Comparison-only corners denied until seal/observedexit;all learned work paid.')
        def act(values):
            torch.cuda.synchronize();tick=time.perf_counter();result=operation(values);torch.cuda.synchronize()
            self.timings.append({'stage':'denseOwnStateCorner','instrumentedSynchronizedSecondsBeforeFinalFingerprints':time.perf_counter()-tick})
            if {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in result.items()}!=outputs or any(not bool(torch.isfinite(v).all()) for v in result.values()):
                raise ValueError('Dense finite output ABI differs')
            got.update(result);return result,{'outputPayloads':{n:payload(v) for n,v in result.items()}}
        registry.register(spec,'denseOwnStateCorner',act)
        after,evidence=registry.run(spec.identity,'denseOwnStateCorner',frame)
        if before!={n:payload(v) for n,v in inputs.items()}:raise ValueError('Read-only source mutation')
        self.records.append({'stage':'denseOwnStateCorner','sourceFrame':frame.to_dict(),'outputFrame':after.to_dict(),'evidence':evidence.to_dict()})
        return got
