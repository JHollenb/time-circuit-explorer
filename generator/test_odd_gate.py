"""Tiny independent scalar quadratic producer and finite odd/even account."""
import unittest


def mechanics():
    import torch
    import numpy as np
    from odd_gate import compute,abis
    from odd_gate_analysis import decompose,analysis_abis
    with torch.random.fork_rng(devices=[]),torch.inference_mode():
        torch.manual_seed(78);n,d,m,nt=7,4,8,3
        z=torch.randn(1,n,d).bfloat16();wg=torch.randn(m,d).bfloat16();u=torch.randn(1,n,m,dtype=torch.float64);down=torch.randn(d,m,dtype=torch.float64);gate=torch.randn(1,1,d).bfloat16();support=torch.tensor([1,6])
        def mean(x):return torch.cat([x[:,:nt].mean(1,keepdim=True),x[:,nt:].mean(1,keepdim=True)],1)
        def expand(x):return torch.cat([x[:,:1].expand(-1,nt,-1),x[:,1:].expand(-1,n-nt,-1)],1)
        mu=mean(u);g=z.double()@wg.double().T;a=torch.nn.functional.silu(g).bfloat16().double();ma=mean(a);du=u-expand(mu)
        lp=expand(ma)*du;lp.index_fill_(-1,support,0);yl=lp@down.T
        src={'normalized':z,'upReference':u,'upMean':mu,'selection_support':support,'weightD':down,'gate':gate,'linearOutput':yl};calls=[]
        def linear(name,x,w):calls.append(name);return x@w.T
        got=compute(src,wg,linear,nt);refg=torch.empty_like(g);reft=torch.zeros_like(u);refy=torch.zeros(1,n,d,dtype=torch.float64)
        for i in range(n):
            for j in range(m):refg[0,i,j]=sum(z.double()[0,i,k]*wg.double()[j,k] for k in range(d))
        for i in range(n):
            lo,hi=(0,nt) if i<nt else (nt,n);s=0 if i<nt else 1
            for j in range(m):
                if j not in support.tolist():reft[0,i,j]=.5*(refg[0,i,j]-sum(refg[0,t,j] for t in range(lo,hi))/(hi-lo))*(u[0,i,j]-mu[0,s,j])
            for c in range(d):refy[0,i,c]=sum(reft[0,i,j]*down[c,j] for j in range(m))
        for field,ref in [('gateReference',refg),('quadraticProduct',reft),('quadraticOutput',refy),('forecast',gate.double()*(yl+refy))]:torch.testing.assert_close(got[field],ref,rtol=2e-12,atol=2e-12)
        assert calls==['fullGate','quadraticDown'];assert {k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in got.items()}==abis(d,m,nt,n-nt)
        cv=lambda v:v.double().numpy().copy();bp=(a-expand(ma))*du;bp.index_fill_(-1,support,0);rp=lp+bp;yr=rp@down.T;q=torch.randn_like(yr)*.001
        comparison={'activation':cv(a),'activationMean':cv(ma),'changingProduct':cv(bp),'centeredOutput':cv(yr),'policyOutput':cv(q),'totalOutput':cv(yr+q)}
        values,diag=decompose({k:cv(v) for k,v in got.items()},{k:cv(v) for k,v in src.items()},comparison,nt)
        assert {k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in values.items()}==analysis_abis(d,m,nt,n-nt)
        np.testing.assert_allclose(values['productClosure'],0,rtol=0,atol=2e-12)
        for lo,hi in [(0,nt),(nt,n)]:
            r=values['fullResidual'][:,lo:hi];c=r-r.mean(1,keepdims=True)
            np.testing.assert_allclose(values['centeredResidual'][:,lo:hi],c,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(np.sum(r*r),np.sum(c*c)+(hi-lo)*np.sum(r.mean(1)**2),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(values['fullGram'][0].sum(),values['rowStatistics'][:,:,0].sum(),rtol=1e-12,atol=1e-12)
        return {'schema':'tce-odd-gate-toy-v1','affines':calls,'producerABIs':len(got),'analysisABIs':len(values),'nativeModelEvaluations':0,
            'maxDownAbsError':float((got['quadraticOutput']-refy).abs().max()),'scope':'Tiny scalar originalGate/centeredproduct/Down/forecast; admittedBF16activation and explicitly evaluated even/finite product closure, fullGram and per-stream residual Pythagoras. No pretrained arrays.'}


class OddGateTests(unittest.TestCase):
    def test_independent_reference(self):self.assertEqual(mechanics()['nativeModelEvaluations'],0)


if __name__=='__main__':unittest.main()
