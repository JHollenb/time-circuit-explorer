"""Independent known-spectrum and rectangular ridge oracles."""
import numpy as np
from source_spectrum import analyse


def test_known_spectrum_retains_null_direction_without_clipping():
    x=np.zeros((4,6));x[:4,:4]=np.diag([5.,2.,.1,0.]);y=np.arange(1,9,dtype=float).reshape(4,2)
    filt=np.array([25.,4.,.01,0.])/(np.array([25.,4.,.01,0.])+.04);expected=filt[:,None]*y
    arrays,modes,info=analyse(x,y,expected)
    np.testing.assert_allclose(arrays['reconstructedPrediction'],expected,atol=2e-14)
    np.testing.assert_allclose(info['actualResidualEnergy'],np.sum((y-expected)**2),atol=2e-13)
    assert modes[-1]['pseudoinverseCoefficientEnergy'] is None and modes[-1]['pseudoinverseStatus']=='undefined-zero-singular-value'
    assert info['thinUIsSquare'] and info['outsideThinUEnergy']==0 and info['actualResidualEnergy']>0


def test_rectangular_spectral_residual_matches_independent_augmented_solve():
    rng=np.random.default_rng(93);x=rng.normal(size=(8,5));y=rng.normal(size=(8,3));b=np.linalg.lstsq(np.vstack([x,np.sqrt(.08)*np.eye(5)]),np.vstack([y,np.zeros((5,3))]),rcond=None)[0]
    arrays,modes,info=analyse(x,y,x@b)
    np.testing.assert_allclose(arrays['reconstructedPrediction'],x@b,atol=2e-13)
    np.testing.assert_allclose(info['spectralResidualEnergy']+info['outsideThinUEnergy'],info['actualResidualEnergy'],atol=2e-13)
    assert not info['thinUIsSquare'] and info['outsideThinUEnergy']>0 and len(modes)==5


def test_zero_input_reports_every_undefined_amplification():
    y=np.ones((3,2));arrays,modes,info=analyse(np.zeros((3,5)),y,np.zeros_like(y))
    np.testing.assert_array_equal(arrays['reconstructedPrediction'],np.zeros_like(y))
    assert info['undefinedPseudoinverseDirections']==3 and all(x['relativeSigma'] is None for x in modes)
