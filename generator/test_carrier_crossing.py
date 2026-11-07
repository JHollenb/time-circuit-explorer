"""Independent factorial and actual toy prehook/geometry mechanics."""
import numpy as np
import pytest
import torch
from carrier_crossing import factorial,replace_geometry


def test_receiver_dependent_nonlinear_interaction():
    f=lambda r,h:r+h+r*h
    fields,e=factorial({'R0A':f(1.,2.),'R0B':f(1.,3.),'R1A':f(4.,2.),'R1B':f(4.,3.)})
    assert fields['responseR0']==2 and fields['responseR1']==5 and fields['interaction']==3 and fields['closure']==0 and e['energyClosure']==0


def test_additive_receiver_independent_response_has_zero_interaction():
    f=lambda r,h:r+2*h
    fields,e=factorial({'R0A':f(1.,2.),'R0B':f(1.,3.),'R1A':f(4.,2.),'R1B':f(4.,3.)})
    assert fields['interaction']==0 and fields['closure']==0 and e['interactionEnergy']==0


def test_actual_prehook_replaces_whole_argument_preserving_layout_and_alias_closure():
    backing=torch.arange(32,dtype=torch.bfloat16);x=backing.as_strided((2,3),(7,2),3);before=backing.clone();donor=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.bfloat16);original=donor.clone();calls=[]
    linear=torch.nn.Linear(3,1,bias=False,dtype=torch.bfloat16)
    with torch.no_grad():linear.weight.copy_(torch.tensor([[1,2,3]],dtype=torch.bfloat16))
    def hook(module,args):
        out=replace_geometry(args[0],donor);calls.append(out)
        assert out.stride()==x.stride() and out.storage_offset()==3
        return (out,)
    handle=linear.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():y=linear(x)
    finally:handle.remove()
    torch.testing.assert_close(y,torch.tensor([[14],[32]],dtype=torch.bfloat16),rtol=0,atol=0)
    assert len(calls)==1 and calls[0].untyped_storage().data_ptr() not in {x.untyped_storage().data_ptr(),donor.untyped_storage().data_ptr()}
    torch.testing.assert_close(backing,before,rtol=0,atol=0);torch.testing.assert_close(donor,original,rtol=0,atol=0)


def test_replacement_rejects_geometry_or_dtype_changes():
    with pytest.raises(ValueError):replace_geometry(torch.zeros(2,3),torch.zeros(3,2))
    with pytest.raises(ValueError):replace_geometry(torch.zeros(2,3),torch.zeros(2,3,dtype=torch.float64))
