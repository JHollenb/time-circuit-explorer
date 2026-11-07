"""Independent scalar positive branch/product/Down and finite residual reference."""
import unittest


def mechanics():
    import torch
    import numpy as np
    from rectifier_correction import compute,abis
    from rectifier_correction_analysis import decompose,analysis_abis,panel,ROW_COLUMNS
    with torch.random.fork_rng(devices=[]),torch.inference_mode():
        torch.manual_seed(80);n,d,m,nt=7,4,8,3
        g=torch.randn(1,n,m,dtype=torch.float64);u=torch.randn_like(g);down=torch.randn(d,m,dtype=torch.float64);gate=torch.randn(1,1,d).bfloat16();support=torch.tensor([1,6])
        def mean(x):return torch.cat([x[:,:nt].mean(1,keepdim=True),x[:,nt:].mean(1,keepdim=True)],1)
        def expand(x):return torch.cat([x[:,:1].expand(-1,nt,-1),x[:,1:].expand(-1,n-nt,-1)],1)
        mu=mean(u);a=torch.nn.functional.silu(g).bfloat16().double();ma=mean(a);du=u-expand(mu);lp=expand(ma)*du;lp.index_fill_(-1,support,0);yl=lp@down.T
        source={'gateReference':g,'upReference':u,'upMean':mu,'selection_support':support,'weightD':down,'gate':gate,'linearOutput':yl};calls=[]
        def linear(name,x,w):calls.append(name);return x@w.T
        got=compute(source,linear,nt);rr=torch.zeros_like(g);rp=torch.zeros_like(g);ry=torch.zeros(1,n,d,dtype=torch.float64)
        for i in range(n):
            lo,hi=(0,nt) if i<nt else (nt,n);s=0 if i<nt else 1
            for j in range(m):
                rr[0,i,j]=max(0.,float(g[0,i,j]))
                if j not in support.tolist():rp[0,i,j]=(rr[0,i,j]-sum(max(0.,float(g[0,t,j])) for t in range(lo,hi))/(hi-lo))*(u[0,i,j]-mu[0,s,j])
            for c in range(d):ry[0,i,c]=sum(rp[0,i,j]*down[c,j] for j in range(m))
        for name,value in [('rectifier',rr),('rectifierProduct',rp),('rectifierOutput',ry),('forecast',gate.double()*(yl+ry))]:torch.testing.assert_close(got[name],value,rtol=2e-12,atol=2e-12)
        assert calls==['rectifierDown'];assert {k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in got.items()}==abis(d,m,nt,n-nt)
        bp=(a-expand(ma))*du;bp.index_fill_(-1,support,0);yr=(lp+bp)@down.T;q=torch.randn_like(yr)*.001;quad=.5*(g-expand(mean(g)))*du;quad.index_fill_(-1,support,0)
        cv=lambda x:x.double().numpy().copy();comp={'activation':cv(a),'activationMean':cv(ma),'changingProduct':cv(bp),'centeredOutput':cv(yr),'policyOutput':cv(q),'totalOutput':cv(yr+q),'quadraticForecast':cv(gate.double()*(yl+quad@down.T))}
        fields,diagnostics=decompose({k:cv(v) for k,v in got.items()},{k:cv(v) for k,v in source.items()},comp,nt)
        assert {k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in fields.items()}==analysis_abis(d,m,nt,n-nt)
        np.testing.assert_allclose(fields['productClosure'],0,rtol=0,atol=2e-12)
        for lo,hi in [(0,nt),(nt,n)]:
            full=fields['fullResidual'][:,lo:hi];center=full-full.mean(1,keepdims=True)
            np.testing.assert_allclose(fields['centeredResidual'][:,lo:hi],center,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(np.sum(full*full),np.sum(center*center)+(hi-lo)*np.sum(full.mean(1)**2),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(fields['fullGram'][0].sum(),fields['rowStatistics'][:,:,0].sum(),rtol=1e-12,atol=1e-12)
        scores=panel(fields['rowStatistics'][0],{kind:fields[kind+'Gram'][0] for kind in ['full','mean','centered']})
        expected=np.sqrt(np.sum(fields['fullResidual']**2)/np.sum((gate.double().numpy()*cv(yr+q))**2))
        np.testing.assert_allclose(scores['fullResidualOverTotalRMS']['rectifier'],expected,rtol=1e-12)
        return {'schema':'tce-rectifier-correction-toy-v1','affines':calls,'producerABIs':len(got),'analysisABIs':len(fields),'rowColumns':len(ROW_COLUMNS),'nativeModelEvaluations':0,'maxDownAbsError':float((got['rectifierOutput']-ry).abs().max()),'scope':'Tiny independent scalar rectifier/streammean/product/Down/fullforecast; admittedBF16A-minusnewrectifier regrouping, fullGram and perstream centered Pythagoras. No pretrained arrays.'}


class RectifierCorrectionTests(unittest.TestCase):
    def test_independent_scalar_reference(self):self.assertEqual(mechanics()['nativeModelEvaluations'],0)


if __name__=='__main__':unittest.main()
