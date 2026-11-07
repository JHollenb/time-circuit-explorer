"""Known row-space, constructive extension freedom and signed modal powers."""
import numpy as np
import pytest
from extension_geometry import analyse


def test_projection_and_constructive_interpolant_freedom():
    vt=np.array([[1.,0.,0.],[0.,1.,0.]]);s=np.array([1.,2.]);c=np.array([[2.,-1.],[3.,4.]]);x=np.array([[1.,2.,5.],[-2.,1.,-3.]])
    fields,q,m,info=analyse(x,vt,s,c,np.ones(2),[1,7]);b0=np.vstack([c[0],c[1]/2,np.zeros(2)])
    np.testing.assert_allclose(fields['reconstructedPrediction'],x@b0,atol=2e-14)
    assert [a['imageQueryIndex'] for a in q]==[1,7] and info['residualEnergy']==34 and info['energyClosure']==0
    source=np.array([[1.,0.,0.],[0.,2.,0.]]);g=np.array([[0.,0.],[0.,0.],[7.,-2.]])
    np.testing.assert_array_equal(source@(b0+g),source@b0)
    np.testing.assert_allclose(x@(b0+g)-x@b0,x[:,2,None]*g[2],atol=2e-14)
    np.testing.assert_allclose(sum(a['signedPredictionPower'] for a in m),np.sum((x@b0)**2),atol=2e-13)


def test_negative_modal_power_and_cancellation_remain_visible():
    fields,q,m,info=analyse(np.array([[1.,1.,0.]]),np.array([[1.,0.,0.],[0.,1.,0.]]),np.ones(2),np.array([[2.],[-1.]]),np.ones(2),[12])
    assert [a['signedPredictionPower'] for a in m]==[2.,-1.]
    assert info['predictionEnergy']==1 and info['diagonalPredictionEnergy']==5 and info['crossPredictionPower']==-4


def test_zero_singular_mode_is_not_silently_discarded():
    with pytest.raises(ValueError):analyse(np.ones((1,2)),np.eye(2),np.array([1.,0.]),np.ones((2,1)),np.ones(2),[0])
