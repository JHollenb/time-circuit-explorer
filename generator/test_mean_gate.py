"""Independent tiny scalar producer/reference and signed decomposition checks."""
import unittest


def mechanics():
    import torch
    import numpy as np
    from mean_gate import compute, abis
    from mean_gate_analysis import decompose, analysis_abis
    with torch.random.fork_rng(devices=[]),torch.inference_mode():
        torch.manual_seed(77);n,d,m,nt=7,4,8,3
        a=torch.randn(1,n,m,dtype=torch.float64);u=torch.randn_like(a);w=torch.randn(d,m,dtype=torch.float64);gate=torch.randn(1,1,d,dtype=torch.float64);support=torch.tensor([1,6])
        mu=torch.cat([u[:,:nt].mean(1,keepdim=True),u[:,nt:].mean(1,keepdim=True)],1)
        src={'activation':a,'upReference':u,'upMean':mu,'selection_support':support,'weightD':w,'gate':gate};calls=[]
        def linear(name,x,w): calls.append(name);return x@w.T
        got=compute(src,linear,nt);lp=torch.zeros_like(a);ref=torch.zeros(1,n,d,dtype=torch.float64);rp=torch.zeros_like(a)
        for i in range(n):
            lo,hi=(0,nt) if i<nt else (nt,n);k=0 if i<nt else 1
            for j in range(m):
                if j not in support.tolist():
                    ma=sum(a[0,t,j] for t in range(lo,hi))/(hi-lo);lp[0,i,j]=ma*(u[0,i,j]-mu[0,k,j]);rp[0,i,j]=a[0,i,j]*(u[0,i,j]-mu[0,k,j])
            for c in range(d):ref[0,i,c]=sum(lp[0,i,j]*w[c,j] for j in range(m))
        torch.testing.assert_close(got['linearProduct'],lp,rtol=1e-12,atol=1e-12);torch.testing.assert_close(got['linearOutput'],ref,rtol=1e-12,atol=1e-12)
        assert calls==['meanGateDown'];assert {n:{'shape':list(x.shape),'dtype':str(x.dtype)} for n,x in got.items()}==abis(d,m,nt,n-nt)
        convert=lambda v:v.numpy().copy();comparison={'centeredProduct':convert(rp),'centeredOutput':convert(rp@w.T),'policyOutput':np.full((1,n,d),.07)};comparison['totalOutput']=comparison['centeredOutput']+comparison['policyOutput']
        values,diagnostics=decompose({k:convert(v) for k,v in got.items()},{k:convert(v) for k,v in src.items()},comparison,nt)
        assert {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in values.items()}==analysis_abis(d,m,nt,n-nt)
        np.testing.assert_allclose(values['changingProduct'],convert(rp-lp),rtol=1e-12,atol=1e-12)
        for k,(lo,hi) in enumerate([(0,nt),(nt,n)]):
            residual=values['fullResidual'][:,lo:hi];centered=residual-residual.mean(1,keepdims=True)
            np.testing.assert_allclose(values['centeredResidual'][:,lo:hi],centered,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(np.sum(residual**2),np.sum(centered**2)+(hi-lo)*np.sum(residual.mean(1)**2),rtol=1e-12,atol=1e-12)
        rows=values['rowStatistics'];np.testing.assert_allclose(rows[:,:,0]-rows[:,:,8],2*rows[:,:,7]-rows[:,:,1],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(rows[:,:,9]-rows[:,:,17],2*rows[:,:,16]-rows[:,:,10],rtol=1e-12,atol=1e-12)
        return {'schema':'tce-mean-gate-toy-v1','affines':calls,'producerABIs':len(got),'analysisABIs':len(values),'nativeModelEvaluations':0,
            'maxDownAbsError':float((got['linearOutput']-ref).abs().max()),'scope':'Tiny synthetic scalar mean/product/Down, subtraction-defined changing term, full/centered Pythagorean and signed row improvement identities. No pretrained data.'}


class MeanGateTests(unittest.TestCase):
    def test_independent_reference(self): self.assertEqual(mechanics()['nativeModelEvaluations'],0)


if __name__=='__main__': unittest.main()
