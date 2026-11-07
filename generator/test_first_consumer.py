"""Independent finite identities and own-input BF16 local path mechanics."""
import numpy as np
import pytest
import torch
from first_consumer import ln_terms,product_terms,energy_terms,path


def test_finite_layernorm_law_includes_receiver_rescaling():
    a=np.array([[1.,2.,-3.],[4.,-2.,1.]]);b=a+np.array([[.2,-.4,.6],[-1.,.5,.1]]);terms=ln_terms(a,b)
    normalize=lambda x:(x-x.mean(-1,keepdims=True))/np.sqrt(np.mean((x-x.mean(-1,keepdims=True))**2,axis=-1,keepdims=True)+1e-6)
    np.testing.assert_allclose(terms['direct']+terms['rescale'],normalize(b)-normalize(a),rtol=1e-13,atol=1e-15)
    assert np.linalg.norm(terms['rescale'])>0 and np.max(np.abs(terms['closure']))<1e-15


def test_symmetric_product_identity_preserves_negative_cross_energy():
    terms=product_terms(np.array([1.]),np.array([3.]),np.array([4.]),np.array([1.]));e=energy_terms(terms['gate'],terms['up'],terms['reference'])
    assert terms['gate']==5 and terms['up']==-6 and terms['reference']==-1 and terms['closure']==0 and e['twiceCross']==-60 and e['energyClosure']==0


def test_native_toy_path_consumes_its_own_committed_residual():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__();self.norm2=torch.nn.LayerNorm(3,elementwise_affine=False);self.ff=torch.nn.Sequential(torch.nn.Linear(3,4,bias=False,dtype=torch.bfloat16),torch.nn.SiLU(),torch.nn.Linear(4,3,bias=False,dtype=torch.bfloat16))
        def forward(self,e,w,shift,scale,gate):
            u=e+w;z=self.norm2(u)*(1+scale)+shift;return u+gate*self.ff(z)
    torch.manual_seed(5);block=Block().eval();e=torch.tensor([[1.,2.,3.]],dtype=torch.bfloat16);w=torch.tensor([[2.,-.5,1.]],dtype=torch.bfloat16);shift=torch.tensor([[.25,.5,-.5]],dtype=torch.bfloat16);scale=torch.tensor([[.5,-.25,.125]],dtype=torch.bfloat16);gate=torch.tensor([[1.,.5,-1.]],dtype=torch.bfloat16);seen={};native=[]
    hook=block.ff.register_forward_pre_hook(lambda m,args:native.append(args[0].clone()))
    try:
        with torch.inference_mode():expected=block(e,w,shift,scale,gate);actual=path(block,e,w,[(None,None,None),(shift,scale,gate)],lambda k,v:seen.setdefault(k,v.clone()))
    finally:hook.remove()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0);torch.testing.assert_close(seen['mlpInput'],native[0],rtol=0,atol=0)
    assert list(seen)==['residualCommit','normalization','normScaled','mlpInput','mlpWrite','gatedMLP','imageExit'] and not torch.equal(seen['mlpInput'],block.norm2(e)*(1+scale)+shift)


def test_identity_input_contract_rejects_nonfinite_or_mismatched_fields():
    with pytest.raises(ValueError):ln_terms(np.zeros((2,3)),np.zeros((3,2)))
    with pytest.raises(ValueError):product_terms(np.zeros(2),np.ones(2),np.ones(2),np.array([np.nan,0.]))
