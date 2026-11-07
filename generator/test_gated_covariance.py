"""Tiny independent scalar reference and constant-family optimality mechanics."""
import unittest


def mechanics():
    import torch
    from gated_covariance import compute,abis
    with torch.random.fork_rng(devices=[]),torch.inference_mode():
        torch.manual_seed(76);n,d,m,nt,k=7,4,8,3,2
        z=torch.randn(1,n,d,dtype=torch.float64);a=torch.randn(1,n,m,dtype=torch.float64);up=torch.randn(m,d,dtype=torch.float64);down=torch.randn(d,m,dtype=torch.float64)
        support=torch.tensor([1,6]);anchor=torch.randn(1,2,m,dtype=torch.float64);completed=torch.cat([anchor[:,:1].expand(-1,nt,-1),anchor[:,1:].expand(-1,n-nt,-1)],1).clone()
        selected=torch.randn(1,n,k,dtype=torch.float64);completed.index_copy_(-1,support,selected);paid=a*completed+torch.randn_like(completed)*.001;gate=torch.randn(1,1,d,dtype=torch.float64)
        src={'normalized':z,'activation':a,'selection_support':support,'completedUp':completed,'product':paid,'gate':gate}
        calls=[]
        def affine(name,x,w):calls.append(name);return x@w.T
        got=compute(src,up,down,affine,nt);ref_u=torch.empty_like(completed)
        for i in range(n):
            for j in range(m):ref_u[0,i,j]=sum(z[0,i,c]*up[j,c] for c in range(d))
        torch.testing.assert_close(got['upReference'],ref_u,rtol=1e-12,atol=1e-12)
        ref_e=torch.zeros(1,n,d,dtype=torch.float64);ref_r=torch.zeros_like(ref_e)
        for i in range(n):
            lo,hi=(0,nt) if i<nt else (nt,n)
            for c in range(d):
                ref_e[0,i,c]=sum((a[0,i,j]*ref_u[0,i,j]-paid[0,i,j])*down[c,j] for j in range(m))
                ref_r[0,i,c]=sum(a[0,i,j]*(ref_u[0,i,j]-sum(ref_u[0,t,j] for t in range(lo,hi))/(hi-lo))*down[c,j] for j in range(m) if j not in support.tolist())
        for field,value in [('totalOutput',ref_e),('centeredOutput',ref_r),('policyOutput',ref_e-ref_r)]:torch.testing.assert_close(got[field],value,rtol=2e-12,atol=2e-12)
        torch.testing.assert_close(got['covarianceDirect'],got['centeredProductMean'],rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(got['policyExpansionResidual'],torch.zeros_like(completed),rtol=0,atol=1e-12)
        for s,(lo,hi) in enumerate([(0,nt),(nt,n)]):
            total=gate*ref_e[:,lo:hi];mu=total.mean(1,keepdim=True)
            torch.testing.assert_close(got['oracleMean'][:,s:s+1],mu)
            centered=total-mu;delta=torch.arange(d,dtype=torch.float64).reshape(1,1,d)*.13
            lhs=((total-(mu+delta))**2).sum();rhs=(centered**2).sum()+(hi-lo)*(delta**2).sum()
            torch.testing.assert_close(lhs,rhs,rtol=1e-12,atol=1e-12)
        expected=abis(d,m,nt,n-nt,k)
        assert {name:{'shape':list(x.shape),'dtype':str(x.dtype)} for name,x in got.items()}==expected
        assert calls==['fullUp','totalErrorDown','centeredDown']
        return {'schema':'tce-gated-covariance-toy-v1','affines':calls,'outputABIs':len(got),'nativeModelEvaluations':0,'maxOutputAbsError':float((got['totalOutput']-ref_e).abs().max()),'scope':'Independent scalar Up/Down, omitted covariance, policy product identity and oracleconstantPythagoreanmechanics. Tiny FP64 synthetic values; no pretrained arrays or model.'}


class GatedCovarianceTests(unittest.TestCase):
    def test_independent_reference(self):self.assertEqual(mechanics()['nativeModelEvaluations'],0)


if __name__=='__main__':unittest.main()
