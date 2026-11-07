from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from two_clock_carry import assay


class Carry(unittest.TestCase):
    def test_fixed_radial_law_cancels_the_incoming_state(self):
        z=np.array([1.,2.]);d=np.array([.25,-.5]);v=np.array([3.,4.]);sigma=.5;e=np.zeros(2)
        f,a=assay(z,v,z-sigma*v,e,z+d,v+d/sigma,z-sigma*v,e,sigma)
        np.testing.assert_array_equal(f['radialVelocityPrediction'],f['deltaVelocity'])
        self.assertEqual(a['outgoingIncomingGain'],0)
        self.assertEqual(a['projection']['radialCancellationStrength'],1)
        self.assertEqual(a['carryWriteFiniteAccount']['closureMaxAbs'],0)

    def test_orthogonal_write_can_amplify_despite_radial_cancellation(self):
        zero=np.zeros(2);d=np.array([1.,0.]);w=np.array([-1.,2.]);sigma=.5
        f,a=assay(zero,zero,zero,zero,d,-w/sigma,d+w,zero,sigma)
        self.assertEqual(a['projection']['radialCancellationStrength'],1)
        self.assertEqual(a['outgoingIncomingGain'],2)
        np.testing.assert_array_equal(f['writeOrthogonal'],[0,2])

    def test_independent_finite_terms_are_preserved(self):
        z=np.array([1.,2.]);d=np.array([.5,0.]);v=np.array([2.,3.]);dv=np.array([1.,2.]);en=np.array([.125,0.]);eh=np.array([0.,.25]);sigma=.5
        f,a=assay(z,v,z-sigma*v+en,en,z+d,v+dv,z+d-sigma*(v+dv)+eh,eh,sigma)
        np.testing.assert_array_equal(f['finiteRemainder'],eh-en)
        self.assertEqual(a['carryWriteFiniteAccount']['closureMaxAbs'],0)
        self.assertEqual(a['projection']['factoredAccount']['closureMaxAbs'],0)
        self.assertEqual(len(a['carryWriteFiniteAccount']['twiceCrossTerms']),3)

    def test_zero_incoming_does_not_erase_nonzero_outgoing(self):
        z=np.zeros(2);v=np.ones(2);sigma=.5
        f,a=assay(z,z,z,z,z,v,-sigma*v,z,sigma)
        self.assertIsNone(a['projection']);self.assertIsNone(a['outgoingIncomingGain'])
        self.assertTrue(a['zeroIncomingNorm']);self.assertGreater(a['outgoingRMS'],0)
        self.assertNotIn('writeParallel',f)
        for bad in [0,-1,np.inf]:
            with self.assertRaises(ValueError):assay(z,z,z,z,z,z,z,z,bad)
        with self.assertRaises(ValueError):assay(z,z,z,z,z,z,[np.nan,0],z,sigma)

    def test_lazy_import_and_execution_guard(self):
        root=str(Path(__file__).parent)
        code='import sys;sys.path.insert(0,'+repr(root)+');import two_clock_carry;assert not {"torch","diffusers","transformers"}.intersection(sys.modules);two_clock_carry.guard()\n'
        code+='try:import torch\nexcept ImportError:pass\nelse:raise AssertionError("neural runtime leaked")\n'
        code+='try:open("/mnt/big/llm-models/test/weights.safetensors","rb")\nexcept PermissionError:pass\nelse:raise AssertionError("weights leaked")\n'
        subprocess.run([sys.executable,'-I','-c',code],check=True,capture_output=True,timeout=20)
