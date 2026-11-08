"""Typed own-state transition and explicit BF16/FP32/FP64 operator census."""
import time
from gate_up_basis_ledger import Ledger as Operations
from paired_fields import payload


class Ledger(Operations):
    def __init__(self,request_id,manifest):
        super().__init__(request_id,manifest)
        self.linears=[];self.moments=[]

    def matmul(self,name,left,right):
        if name=='outFull':raise ValueError('Full output affine is outside this transition')
        return super().matmul(name,left,right)

    def linear(self,name,value,weight):
        import torch
        if name!='ownQKV' or value.dtype!=torch.bfloat16 or weight.dtype!=torch.bfloat16 or value.device.type!='cuda' or torch.is_autocast_enabled('cuda'):
            raise ValueError('Physical QKV projection profile differs')
        torch.cuda.synchronize();tick=time.perf_counter()
        result=torch.nn.functional.linear(value,weight)
        torch.cuda.synchronize()
        self.linears.append({'name':name,'inputShape':list(value.shape),'weightShape':list(weight.shape),
                             'outputShape':list(result.shape),'inputStride':list(value.stride()),
                             'weightStride':list(weight.stride()),'outputStride':list(result.stride()),
                             'inputPayloadSHA256':payload(value),'weightPayloadSHA256':payload(weight),
                             'outputPayloadSHA256':payload(result),'dtype':str(value.dtype),'device':'cuda',
                             'nominalMACs':result.numel()*value.shape[-1],
                             'synchronizedSecondsIncludingFingerprints':time.perf_counter()-tick})
        return result

    def moment(self,name,left,right):
        import torch
        if name not in ['imageCovariance','queryCovariance','queryMean','textScores','textAV'] or left.dtype!=torch.float32 or right.dtype!=torch.float32 or left.device.type!='cuda' or torch.is_autocast_enabled('cuda'):
            raise ValueError('Live text-moment product profile differs')
        torch.cuda.synchronize();tick=time.perf_counter();result=left@right;torch.cuda.synchronize()
        self.moments.append({'name':name,'leftShape':list(left.shape),'rightShape':list(right.shape),
                             'outputShape':list(result.shape),'leftPayloadSHA256':payload(left),
                             'rightPayloadSHA256':payload(right),'outputPayloadSHA256':payload(result),
                             'dtype':str(result.dtype),'device':'cuda','nominalMACs':result.numel()*left.shape[-1],
                             'synchronizedSecondsIncludingFingerprints':time.perf_counter()-tick})
        return result

    def run(self,inputs,operation,outputs):
        import torch
        from mdb.titan import Frame,ActSpec,ActRegistry,ProvidedFingerprint
        before={n:payload(v) for n,v in inputs.items()};got={};registry=ActRegistry()
        frame=Frame.from_values(inputs,kinds={n:'tensor' for n in inputs},
            abis={n:{'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype)} for n,v in inputs.items()},
            slot_fingerprints={n:ProvidedFingerprint.content_sha256(h) for n,h in before.items()},
            program_fingerprint=self.request_id,metadata={'phase':'ownSuccessor','originalParameterManifestSHA256':self.manifest})
        spec=ActSpec('tce.own_successor','v1',reads=tuple(inputs),writes=tuple(outputs),
            write_kinds={n:'tensor' for n in outputs},write_abis=outputs,
            numerical_contract='Own86 text/image plus paid clock/RoPE and four original single1 tensors. '
            'Own norm/mod, physical BF16 QKV (finite slicing not parity), native RMS/RoPE/live attention/moments, '
            'operand-only frozen82 GU, frozen86 receiver output and separate BF16 gate/add. '
            'No archived normalized/read/product/bases/targets before seal and observed process exit. All learned work paid.')
        def act(values):
            torch.cuda.synchronize();tick=time.perf_counter();result=operation(values);torch.cuda.synchronize()
            self.timings.append({'stage':'ownSuccessor','instrumentedSynchronizedSecondsBeforeFinalFingerprints':time.perf_counter()-tick})
            if {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in result.items()}!=outputs or any(not bool(torch.isfinite(v).all()) for v in result.values()):raise ValueError('Own-state finite output ABI differs')
            got.update(result);return result,{'outputPayloads':{n:payload(v) for n,v in result.items()}}
        registry.register(spec,'ownSuccessor',act)
        after,evidence=registry.run(spec.identity,'ownSuccessor',frame)
        if before!={n:payload(v) for n,v in inputs.items()}:raise ValueError('Read-only source mutation')
        self.records.append({'stage':'ownSuccessor','sourceFrame':frame.to_dict(),'outputFrame':after.to_dict(),'evidence':evidence.to_dict()})
        return got
