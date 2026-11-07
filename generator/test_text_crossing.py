"""Independent response decomposition and real two-boundary BF16 hooks."""
import numpy as np
import pytest
import torch
from carrier_crossing import replace_geometry
from text_crossing import decompose,account


def test_text_write_can_explain_the_response_interaction():
    f=lambda r,t,h:r+2*h+t*h
    old={k:np.array(f(r,t,h)) for k,r,t,h in [('R0A',1,2,3),('R0B',1,2,4),('R1A',5,7,3),('R1B',5,7,4)]};new={k:np.array(f(5,2,h)) for k,h in [('A',3),('B',4)]}
    fs=decompose(old,new);a=account(fs)
    assert fs['originalInteraction']==5 and fs['remainingReceiverDifference']==0 and fs['firstTextWriteEffect']==5 and fs['closure']==0
    assert a['firstTextWriteEnergy']==25 and a['energyClosure']==0


def test_remaining_and_text_terms_can_cancel_with_negative_cross_energy():
    # Original response changes1->2, while fixing the text write gives response4.
    fs=decompose(dict(R0A=np.array(0.),R0B=np.array(1.),R1A=np.array(0.),R1B=np.array(2.)),dict(A=np.array(0.),B=np.array(4.)));a=account(fs)
    assert fs['remainingReceiverDifference']==3 and fs['firstTextWriteEffect']==-2 and a['twiceRemainingText']==-12 and a['originalInteractionEnergy']==1 and a['energyClosure']==0


def test_real_output_and_input_hooks_replace_whole_tensors_in_order():
    class StridedLinear(torch.nn.Linear):
        def forward(self,x):
            y=super().forward(x);backing=torch.empty(32,dtype=y.dtype);out=backing.as_strided((2,3),(7,2),5);out.copy_(y);return out
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__();self.text=StridedLinear(3,3,bias=False,dtype=torch.bfloat16);self.image=torch.nn.Linear(3,3,bias=False,dtype=torch.bfloat16)
            with torch.no_grad():self.text.weight.copy_(torch.eye(3));self.image.weight.copy_(torch.eye(3))
        def forward(self,x):return self.text(x)+self.image(x)
    block=Block();storage=torch.arange(32,dtype=torch.bfloat16);before=storage.clone();x=storage.as_strided((2,3),(7,2),3)
    h=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.bfloat16);t=10*h;donors=[h.clone(),t.clone()];events=[]
    def replace(native,donor,name,offset):
        out=replace_geometry(native,donor);assert out.stride()==native.stride() and out.storage_offset()==offset
        assert out.untyped_storage().data_ptr() not in {native.untyped_storage().data_ptr(),donor.untyped_storage().data_ptr()};torch.testing.assert_close(out,donor,rtol=0,atol=0);events.append(name);return out
    a=block.text.register_forward_hook(lambda module,args,y:replace(y,t,'text',5));b=block.image.register_forward_pre_hook(lambda module,args:(replace(args[0],h,'image',3),))
    try:
        with torch.inference_mode():out=block(x)
    finally:a.remove();b.remove()
    torch.testing.assert_close(out,torch.tensor([[11,22,33],[44,55,66]],dtype=torch.bfloat16),rtol=0,atol=0)
    assert events==['text','image'];torch.testing.assert_close(storage,before,rtol=0,atol=0);torch.testing.assert_close(h,donors[0],rtol=0,atol=0);torch.testing.assert_close(t,donors[1],rtol=0,atol=0)


def test_decomposition_rejects_nonfinite_or_changed_geometry():
    old={k:np.zeros(2) for k in ['R0A','R0B','R1A','R1B']}
    with pytest.raises(ValueError):decompose(old,dict(A=np.zeros(2),B=np.array([np.nan,0.])))
    with pytest.raises(ValueError):decompose(old,dict(A=np.zeros(2),B=np.zeros(3)))
