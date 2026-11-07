"""One typed source-only producer, with actual FP64 and BF16 operation receipts."""
import time
from paired_fields import payload


class Ledger:
    def __init__(self,request_id,manifest):
        self.request_id=request_id;self.manifest=manifest;self.records=[];self.products=[];self.qrs=[];self.timings=[]

    def matmul(self,name,left,right):
        import torch
        torch.cuda.synchronize();tick=time.perf_counter()
        result=torch.nn.functional.linear(left,right.T) if name=='outFull' else left@right
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        if left.dtype!=right.dtype or left.device.type!='cuda' or torch.is_autocast_enabled('cuda') or left.dtype!=(torch.bfloat16 if name=='outFull' else torch.float64):raise ValueError('Physical contraction profile differs')
        self.products.append({'name':name,'operator':'torch.nn.functional.linear' if name=='outFull' else 'torch.matmul','leftShape':list(left.shape),'rightShape':list(right.shape),'outputShape':list(result.shape),
            'leftPayloadSHA256':payload(left),'rightPayloadSHA256':payload(right),'outputPayloadSHA256':payload(result),
            'dtype':str(left.dtype),'device':left.device.type,'nominalMACs':result.numel()*left.shape[-1],'synchronizedSecondsBeforeFingerprints':seconds})
        return result

    def qr(self,name,value):
        import torch
        if value.dtype!=torch.float64 or value.device.type!='cuda':raise ValueError('FP64 CUDA Householder QR required')
        torch.cuda.synchronize();tick=time.perf_counter();q,r=torch.linalg.qr(value,mode='reduced');torch.cuda.synchronize()
        self.qrs.append({'name':name,'operator':'torch.linalg.qr','mode':'reduced','algorithm':'Householder factorization and explicit reduced Q',
            'inputShape':list(value.shape),'QShape':list(q.shape),'RShape':list(r.shape),'dtype':str(value.dtype),'device':'cuda',
            'synchronizedSecondsBeforeFingerprints':time.perf_counter()-tick,'inputPayloadSHA256':payload(value),'rawQPayloadSHA256':payload(q),'rawRPayloadSHA256':payload(r)})
        return q,r

    def run(self,inputs,operation,outputs):
        import torch
        from mdb.titan import Frame,ActSpec,ActRegistry,ProvidedFingerprint
        before={n:payload(v) for n,v in inputs.items()};registry=ActRegistry();got={}
        frame=Frame.from_values(inputs,kinds={n:'tensor' for n in inputs},abis={n:{'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype)} for n,v in inputs.items()},
            slot_fingerprints={n:ProvidedFingerprint.content_sha256(h) for n,h in before.items()},program_fingerprint=self.request_id,
            metadata={'phase':'gateUpBasis','originalParameterManifestSHA256':self.manifest})
        spec=ActSpec('tce.gate_up_basis','v1',reads=tuple(inputs),writes=tuple(outputs),write_kinds={n:'tensor' for n in outputs},write_abis=outputs,
            numerical_contract='Actual BF16Z and originalGU maps promoted FP64; two fixed current-input rank128 Householder bases, complete reconstructedGU via FP64->FP32->BF16, native Flux2SwiGLU and originalfullout/gate/residual. Targets denied until process exit. All learned work paid.')
        def act(values):
            torch.cuda.synchronize();tick=time.perf_counter();result=operation(values);torch.cuda.synchronize()
            self.timings.append({'stage':'gateUpBasis','instrumentedSynchronizedSecondsBeforeFinalFingerprints':time.perf_counter()-tick,
                'scope':'Factorization/native consumer plus internal operation fingerprints; excludes setup/outer source hashes/final output fingerprints and retention.'})
            if {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in result.items()}!=outputs or any(not bool(torch.isfinite(v).all()) for v in result.values()):raise ValueError('Producer finite ABI differs')
            got.update(result);return result,{'outputPayloads':{n:payload(v) for n,v in result.items()}}
        registry.register(spec,'gateUpBasis',act);after,evidence=registry.run(spec.identity,'gateUpBasis',frame)
        if before!={n:payload(v) for n,v in inputs.items()}:raise ValueError('Read-only inputs changed')
        self.records.append({'stage':'gateUpBasis','sourceFrame':frame.to_dict(),'outputFrame':after.to_dict(),'evidence':evidence.to_dict()})
        return got
