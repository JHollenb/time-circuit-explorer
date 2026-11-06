import unittest
import numpy as np
from padding_mlp_remainder import decomposition


class PaddingRemainderMechanics(unittest.TestCase):
    def test_signed_energy_and_closure_preserve_cancellation(self):
        terms={'a':np.array([3.,-2.]),'b':np.array([-2.,1.]),'c':np.array([.5,.25])};error=sum(terms.values());r=decomposition(error,terms)
        self.assertEqual(r['closureMaxAbs'],0.);self.assertAlmostEqual(r['actualErrorEnergy'],sum(r['termEnergy'].values())+sum(x['value'] for x in r['twiceCrossTerms']))
        self.assertTrue(any(x['value']<0 for x in r['twiceCrossTerms']))

    def test_native_smooth_anchor_and_runtime_terms_close(self):
        native=np.array([1.,2.]);smooth=np.array([1.1,1.9]);anchor=np.array([.3,.4]);base=np.array([.31,.39]);linear=np.array([.7,1.4]);served=np.array([1.02,1.8])
        terms={'finite':native-smooth,'curvature':smooth-anchor-linear,'anchor':anchor-base,'runtime':base+linear-served};r=decomposition(native-served,terms)
        self.assertLess(r['closureMaxAbs'],1e-15)
        with self.assertRaises(ValueError):decomposition(native,{'wrong':np.ones(3)})


if __name__=='__main__':unittest.main()
