"""Independent tiny minimum-norm oracle and no-truncation contract."""
import numpy as np
import pytest
from minimum_response import minimum_norm


def test_known_minimum_norm_interpolates_and_has_no_null_component():
    x=np.array([[1.,0.,1.],[0.,1.,0.]]);y=np.array([[2.,4.],[3.,6.]])
    u,s,vt=np.linalg.svd(x,full_matrices=False);b,info=minimum_norm(vt,s,u.T@y)
    expected=np.array([[1.,2.],[3.,6.],[1.,2.]])
    np.testing.assert_allclose(b,expected,atol=2e-14)
    np.testing.assert_allclose(x@b,y,atol=2e-14)
    np.testing.assert_allclose(np.array([1.,0.,-1.])@b,0,atol=2e-14)
    assert info['allModesUsed'] and info['coefficientNominalMACs']==3*2*2


def test_random_full_row_rank_matches_independent_lstsq():
    rng=np.random.default_rng(104);x=rng.normal(size=(5,8));y=rng.normal(size=(5,3));u,s,vt=np.linalg.svd(x,full_matrices=False)
    b,info=minimum_norm(vt,s,u.T@y)
    np.testing.assert_allclose(b,np.linalg.lstsq(x,y,rcond=None)[0],atol=2e-14)
    np.testing.assert_allclose(info['coefficientEnergy'],info['modeCoefficientEnergy'],atol=2e-13)


def test_zero_or_nonfinite_modes_fail_without_selecting_another_rank():
    for s in [np.array([1.,0.]),np.array([1.,np.nan]),np.array([1.,-1.])]:
        with pytest.raises(ValueError):minimum_norm(np.eye(2),s,np.ones((2,1)))
