"""Typed template/source binding and explicit geometry on every new contraction."""
import time
from own_successor_ledger import Ledger as Operations
from paired_fields import payload
from gate_up_basis import geometry


class Ledger(Operations):
    def matmul(self,name,left,right):
        result=super().matmul(name,left,right)
        self.products[-1].update(arm=self.arm,leftGeometry=geometry(left),rightGeometry=geometry(right),outputGeometry=geometry(result))
        return result

    def linear(self,name,value,weight):
        result=super().linear(name,value,weight);self.linears[-1]['arm']=self.arm;return result

    def moment(self,name,left,right):
        result=super().moment(name,left,right);self.moments[-1].update(arm=self.arm,leftGeometry=geometry(left),rightGeometry=geometry(right),outputGeometry=geometry(result));return result

    def qr(self,*args):raise ValueError('Template formation/QR forbidden')

    def run(self,inputs,operation,outputs,arm):
        import torch
        from mdb.titan import Frame,ActSpec,ActRegistry,ProvidedFingerprint
        if arm not in ['parent','own']:raise ValueError('Unknown input arm')
        self.arm=arm;before={n:payload(v) for n,v in inputs.items()};got={};registry=ActRegistry()
        frame=Frame.from_values(inputs,kinds={n:'tensor' for n in inputs},
            abis={n:geometry(v) for n,v in inputs.items()},
            slot_fingerprints={n:ProvidedFingerprint.content_sha256(h) for n,h in before.items()},
            program_fingerprint=self.request_id,metadata={'phase':arm,'originalParameterManifestSHA256':self.manifest,
                'sourceParameterTemplateManifestSHA256':self.manifest})
        spec=ActSpec('tce.fixed_template_'+arm,'v1',reads=tuple(inputs),writes=tuple(outputs),
            write_kinds={n:'tensor' for n in outputs},write_abis=outputs,
            numerical_contract='One explicitly row-contiguous reloaded parent88GU/output template. '
            'Actual parent or own incoming state plus paid clocks/maps. Fresh normalization/slicedQKV/nativeRMS/RoPE/attention/moments/GU/product/output/commit. '
            'No QR/compilation/old dynamic operands. Parent output association measured,not assumed. '
            'Both candidates seal and producer exits before comparison-only access;all learned arithmetic paid.')
        def act(values):
            torch.cuda.synchronize();tick=time.perf_counter();result=operation(values);torch.cuda.synchronize()
            self.timings.append({'arm':arm,'instrumentedSynchronizedSecondsBeforeFinalFingerprints':time.perf_counter()-tick})
            if {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in result.items()}!=outputs or any(not bool(torch.isfinite(v).all()) for v in result.values()):
                raise ValueError('Fixed-template finite output ABI differs')
            got.update(result);return result,{'outputPayloads':{n:payload(v) for n,v in result.items()}}
        registry.register(spec,arm,act);after,evidence=registry.run(spec.identity,arm,frame)
        if before!={n:payload(v) for n,v in inputs.items()}:raise ValueError('Read-only template/source mutation')
        self.records.append({'stage':arm,'sourceFrame':frame.to_dict(),'outputFrame':after.to_dict(),'evidence':evidence.to_dict()})
        return got
