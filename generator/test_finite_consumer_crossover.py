"""Eight tiny supplied mathematical/finite/scalar witnesses, no real model."""
import copy
import math
import struct
import unittest
import numpy as np
from finite_consumer_crossover import admit,construct,decompose,STATE_FIELDS,VECTOR_FIELDS
from publish_finite_consumer_crossover import validate_scalar_analysis


def reference_admit(x):
    word=struct.unpack('<I',struct.pack('<f',float(x)))[0]
    rounded=((word+0x7fff+((word>>16)&1))>>16)&65535
    return struct.unpack('<f',struct.pack('<I',rounded<<16))[0]


class FiniteConsumerCrossoverTests(unittest.TestCase):
    def test_crossed_construction_changes_only_selected_row(self):
        a=np.array([[1.,2.],[3.,4.]]);b=a+np.array([[0.,0.],[.5,-.25]])
        p=a+1;q=p+np.array([[0.,0.],[.25,.5]])
        r=construct(a,b,p,q,1)
        self.assertEqual(set(r),{'Xno','Xon',*STATE_FIELDS})
        np.testing.assert_array_equal(r['Xno'][0],a[0]);np.testing.assert_array_equal(r['Xon'][0],p[0])
        for name,s in [('Xno',a[1]+(q[1]-p[1])),('Xon',p[1]+(b[1]-a[1]))]:np.testing.assert_array_equal(r[name][1],[reference_admit(x) for x in s])
        np.testing.assert_array_equal(r['rNO'],r['Xno'][1]-r['sumNO']);np.testing.assert_array_equal(r['rON'],r['Xon'][1]-r['sumON'])

    def test_admission_matches_independent_struct_ties(self):
        values=[1+2**-8,1+3*2**-8,-1-2**-8,-1-3*2**-8,2**-133,3*2**-134]
        np.testing.assert_array_equal(admit(values),[reference_admit(x) for x in values])

    def test_endpoint_promotion_precedes_difference(self):
        a=np.array([[2**32]],np.float32);b=np.array([[1.]],np.float32)
        r=construct(a,b,a,b,0);self.assertEqual(r['deltaN'][0],1-2**32)
        self.assertNotEqual(float((b-a)[0,0]),r['deltaN'][0])

    def test_diagonal_reconstruction_discrepancy_is_retained(self):
        a=np.array([[2**64]],np.float64);b=np.array([[1.]]);q=np.array([[2.]])
        r=construct(a,b,a,q,0);self.assertEqual(r['diagN'][0],-1.);self.assertEqual(r['diagO'][0],-2.)

    def test_signed_paths_preserve_cancellation_and_gram(self):
        v,r=decompose([0.,0.],[.5,-.5],[2.,0.],[3.,-1.],[8.,-8.],[2.25,-.25])
        self.assertEqual(set(v),set(VECTOR_FIELDS));self.assertLess(r['innerProducts']['Uno:Vno'],0)
        for suffix in ('no','on'):
            expected=math.fsum(float(x)**2 for x in v['E']);parts=r['paths'][suffix]['terms']
            self.assertEqual(math.fsum(parts),expected);np.testing.assert_array_equal(v['U'+suffix]+v['V'+suffix]+v['g'+suffix],v['E'])
        self.assertEqual(validate_scalar_analysis(r,2),47)

    def test_zero_native_response_keeps_absolute_error(self):
        _,r=decompose([0.],[0.],[1.],[2.],[.5],[1.5])
        self.assertEqual(r['rms']['E'],1.);self.assertTrue(all(x is None for x in r['relativeToNative'].values()))
        self.assertTrue(all(x is None for x in r['responseCosines'].values()));validate_scalar_analysis(r,1)

    def test_scalar_auditor_rejects_tampering_and_wrong_abi(self):
        _,r=decompose([0.],[1.],[0.],[.5],[.75],[1.25])
        for path,value in [(('rms','E'),99.),(('paths','no','gramSum'),99.)]:
            bad=copy.deepcopy(r);node=bad
            for k in path[:-1]:node=node[k]
            node[path[-1]]=value
            with self.assertRaises(ValueError):validate_scalar_analysis(bad,1)
        with self.assertRaises(ValueError):validate_scalar_analysis(r,2)

    def test_invalid_geometry_and_nonfinite_values_are_mechanics_errors(self):
        with self.assertRaises(ValueError):construct([[0.]],[[0.,0.]],[[0.]],[[0.]],0)
        with self.assertRaises(ValueError):construct([[0.]],[[0.]],[[0.]],[[0.]],True)
        with self.assertRaises(ValueError):decompose([0.],[float('nan')],[0.],[0.],[0.],[0.])
        with np.errstate(over='ignore'):
            with self.assertRaises(ValueError):admit([1e39])


if __name__=='__main__':unittest.main()
