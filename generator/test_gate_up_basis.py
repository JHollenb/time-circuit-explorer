"""Tiny independent contractions, projector, native finite consumer and state scores."""
import unittest


def mechanics():
    import numpy as np
    import torch
    from diffusers.models.transformers.transformer_flux2 import Flux2SwiGLU
    from gate_up_basis import compute,signs,abis,work
    from gate_up_basis_analysis import statistics,panel
    def scalar_mm(a,b):
        return np.array([[sum(float(a[i,k])*float(b[k,j]) for k in range(a.shape[1])) for j in range(b.shape[1])] for i in range(a.shape[0])],dtype=np.float64)
    with torch.random.fork_rng(devices=[]),torch.inference_mode():
        torch.manual_seed(82);n,d,m,nt,r=7,4,8,3,2
        source={'normalized':torch.randn(1,n,d).bfloat16(),'attentionRead':torch.randn(1,n,d).bfloat16(),'gate':torch.randn(1,1,d).bfloat16(),
                'text':torch.randn(1,nt,d).bfloat16(),'image':torch.randn(1,n-nt,d).bfloat16()}
        fused=torch.randn(3*d+2*m,d).bfloat16();out=torch.randn(d,d+m).bfloat16();calls=[];qrs=[]
        def mm(name,a,b):calls.append((name,a.shape,b.shape));return torch.nn.functional.linear(a,b.T) if name=='outFull' else a@b
        def qr(name,a):qrs.append(name);return torch.linalg.qr(a,mode='reduced')
        numpy_before=np.random.get_state();torch_before=torch.get_rng_state();got,trace=compute(source,fused,out,Flux2SwiGLU(),mm,qr,nt,r)
        assert torch.equal(torch_before,torch.get_rng_state());numpy_after=np.random.get_state()
        assert all(np.array_equal(a,b) for a,b in zip(numpy_before,numpy_after))
        assert signs(3,4,0).tolist()==[[1,-1,1,1],[-1,-1,1,-1],[-1,-1,-1,-1]]
        assert not np.array_equal(signs(nt,r,0),signs(nt,r,1))
        maximum=0.
        for stream,(lo,hi) in enumerate([(0,nt),(nt,n)]):
            x=source['normalized'][0,lo:hi].double().numpy();omega=signs(hi-lo,r,stream).astype(np.float64);y=scalar_mm(x.T,omega)
            q=got['basis'][stream].numpy();rr=got['qrR'][stream].numpy();np.testing.assert_allclose(q@rr,y,rtol=2e-14,atol=2e-14)
            np.testing.assert_allclose(q.T@q,np.eye(r),rtol=2e-14,atol=2e-14);assert np.all(np.diag(rr)>=0)
            # Independent modified Gram-Schmidt span; column sign is irrelevant to its projector.
            orth=[]
            for j in range(r):
                column=y[:,j].copy()
                for previous in orth:column-=sum(column*previous)*previous
                orth.append(column/np.sqrt(sum(column*column)))
            independent=np.stack(orth,1);np.testing.assert_allclose(q@q.T,independent@independent.T,rtol=3e-14,atol=3e-14)
            expected=scalar_mm(scalar_mm(x,q),scalar_mm(q.T,fused[3*d:].double().numpy().T))
            actual=torch.cat([got['gateFP64'],got['upFP64']],-1)[0,lo:hi].numpy();maximum=max(maximum,float(np.max(np.abs(expected-actual))))
            np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)
        raw=torch.cat([got['gateFP64'],got['upFP64']],-1).float().bfloat16();g,u=raw.split(m,-1)
        a=torch.nn.functional.silu(g);p=a*u;joined=torch.cat([source['attentionRead'],p],-1);update=torch.nn.functional.linear(joined,out)
        expected_exit=torch.cat([source['text'],source['image']],1)+source['gate']*update
        for name,expected in [('gateRaw',g),('upRaw',u),('activation',a),('product',p),('preResidualUpdate',update),('exit',expected_exit)]:assert torch.equal(got[name],expected),name
        assert trace['nativeGUView']['stride']==[n*(3*d+2*m),3*d+2*m,1] and trace['nativeGUView']['storageOffset']==3*d
        assert trace['nativeUpView']['storageOffset']==3*d+m and trace['recordedProduct']['storageOffset']==d
        assert trace['recordedProduct']['stride']==[n*(d+m),d+m,1] and trace['activation']['stride']==[n*m,m,1]
        assert {k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in got.items()}==abis(d,m,nt,n-nt,r)
        special=torch.tensor([1+2**-8+2**-26],dtype=torch.float64)
        # Torch's direct double-to-BF16 path may itself pass through FP32. Compare
        # the declared two-step result to independent ideal nearest-BF16 rounding.
        ideal=min([1.,1+2**-7],key=lambda point:abs(float(special[0])-point))
        assert float(special.float().bfloat16()[0])==1 and ideal==1+2**-7
        assert [x[0] for x in calls]==['sketch0','coordinates0','compileGU0','reconstructGU0','sketch1','coordinates1','compileGU1','reconstructGU1','outFull'] and qrs==['qr0','qr1']
        costs=work();assert costs['candidateMACs']==77309411328 and costs['diagnosticMACs']==704643072 and costs['assayMACs']==78014054400
        # A large residual must not disguise a missed state increment.
        base=np.full((1,n,d),1000.);target=base+np.arange(1,n+1)[None,:,None]*np.ones((1,n,d));candidate=base.copy()
        state=panel(statistics(candidate,target,nt));increment=panel(statistics(candidate-base,target-base,nt))
        assert state['errorOverTargetRMS']<.01 and increment['errorOverTargetRMS']==1 and increment['rowsVsZero']['equal']==n
        return {'schema':'tce-current-state-gate-up-toy-v1','candidateProducts':9,'QRCalls':2,'producerABIs':len(got),'maxIndependentOperandAbsError':maximum,
                'nativeConsumerExact':True,'doubleRoundingDistinguished':True,'globalRNGUnchanged':True,'nativePretrainedModelEvaluations':0,
                'scope':'Tiny scalar contraction and independent Gram-Schmidt span, complete native parameter-free consumer with tiny random maps, physical views/double cast and state-increment denominator. No real model arrays.'}


class GateUpBasisTests(unittest.TestCase):
    def test_independent_factorization_and_native_consumer(self):self.assertTrue(mechanics()['nativeConsumerExact'])

    def test_actual_typed_transaction_and_publication_join(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import torch
        from diffusers.models.transformers.transformer_flux2 import Flux2SwiGLU
        from gate_up_basis import compute,abis
        from gate_up_basis_ledger import Ledger
        from paired_fields import retain
        from publish_terminal_trajectory import check_act
        with torch.inference_mode(),tempfile.TemporaryDirectory() as directory,patch('torch.cuda.synchronize',lambda:None):
            source={'normalized':torch.ones(1,8,4,dtype=torch.bfloat16),'attentionRead':torch.ones(1,8,4,dtype=torch.bfloat16),'gate':torch.ones(1,1,4,dtype=torch.bfloat16),
                    'text':torch.ones(1,4,4,dtype=torch.bfloat16),'image':torch.ones(1,4,4,dtype=torch.bfloat16)}
            fused=torch.ones(28,4,dtype=torch.bfloat16);out=torch.ones(4,12,dtype=torch.bfloat16);ledger=Ledger('toy-request','toy-manifest')
            def operation(values):
                return compute({n:values[n] for n in source},values['originalFused'],values['originalOut'],Flux2SwiGLU(),lambda name,a,b:torch.nn.functional.linear(a,b.T) if name=='outFull' else a@b,
                    lambda name,y:torch.linalg.qr(y,mode='reduced'),4,2)[0]
            values=ledger.run({**source,'originalFused':fused,'originalOut':out},operation,abis(4,8,4,4,2))
            descriptors={n:retain(Path(directory),n,value) for n,value in values.items()}
            check_act(ledger.records[0],descriptors,'toy-request','toy-manifest')
            self.assertEqual(len(ledger.records[0]['sourceFrame']['slots']),7)

    def test_rank_deficiency_keeps_all_columns(self):
        import torch
        from diffusers.models.transformers.transformer_flux2 import Flux2SwiGLU
        from gate_up_basis import compute
        with torch.inference_mode():
            s={'normalized':torch.zeros(1,8,4,dtype=torch.bfloat16),'attentionRead':torch.zeros(1,8,4,dtype=torch.bfloat16),'gate':torch.ones(1,1,4,dtype=torch.bfloat16),
               'text':torch.zeros(1,4,4,dtype=torch.bfloat16),'image':torch.zeros(1,4,4,dtype=torch.bfloat16)}
            got,_=compute(s,torch.ones(28,4,dtype=torch.bfloat16),torch.ones(4,12,dtype=torch.bfloat16),Flux2SwiGLU(),lambda name,a,b:a@b,lambda name,y:torch.linalg.qr(y,mode='reduced'),4,2)
            self.assertEqual(list(got['basis'].shape),[2,4,2]);self.assertTrue(bool((got['columnSigns']==1).all()));self.assertTrue(bool((got['gateFP64']==0).all()))


if __name__=='__main__':unittest.main()
