"""Tiny supplied checks for signs, component orientation and finite-state accounts."""
import json
import math
from pathlib import Path
import time
import unittest
import numpy as np
from writer_direction_math import analyze,columns,components,pullback,TERMS


class Mechanics(unittest.TestCase):
    def parts(self,shape):return {k:np.zeros(shape) for k in TERMS}
    def test_projection_can_reverse_sign(self):
        g=np.array([[2.],[-1.]]);delta=np.array([[-.02],[1.]])
        rows,_=analyze(g,self.parts(g.shape),delta,np.full((2,1),.49),np.full((2,1),.49))
        self.assertAlmostEqual(rows[:,columns().index('saved.proposal')].sum(),-1.04)
        self.assertAlmostEqual(rows[:,columns().index('saved.writer')].sum(),.49)
    def test_component_pair_orientation(self):
        d={'h':np.array([[1.,0.],[3.,0.]]),'z':np.zeros((2,2)),'target':np.zeros((2,2)),'zt':np.zeros((2,2)),
           'weights':np.array([.25,.75]),'edges':np.array([[0,1]]),'pairWeights':np.array([.5]),'gamma':np.ones(2),'scaleVector':np.array([1.,2.,0.,0.])}
        p=components(d,1.);np.testing.assert_array_equal(p['Hendpoint'],[[.25,0.],[2.25,0.]])
        np.testing.assert_array_equal(p['Hresponse'],[[-.5,0.],[.5,0.]])
        np.testing.assert_array_equal(p['Zendpoint'],np.zeros((2,2)));np.testing.assert_array_equal(p['Zresponse'],np.zeros((2,2)))
    def test_zero_reference_absolute_policy(self):
        d={'h':np.zeros((2,2)),'z':np.ones((2,2)),'target':np.zeros((2,2)),'zt':np.zeros((2,2)),
           'weights':np.ones(2),'edges':np.array([[0,1]]),'pairWeights':np.ones(1),'gamma':np.array([2.,3.]),'scaleVector':np.zeros(4)}
        np.testing.assert_array_equal(components(d,1.)['Zendpoint'],[[2.,3.],[2.,3.]])
        s=math.sqrt(1.5)
        actual=pullback(np.array([[1.,0.]]),np.array([[1.,1.]]),np.array([2.,3.]),1.)
        np.testing.assert_allclose(actual,[[2/s-1/s**3,3/s]],rtol=1e-15,atol=0.)
    def test_original_pair_error_order(self):
        d={'h':np.array([[1e16],[1e16+2.]]),'z':np.zeros((2,1)),'target':np.array([[1.],[2.]]),'zt':np.zeros((2,1)),
           'weights':np.ones(2),'edges':np.array([[0,1]]),'pairWeights':np.ones(1),'gamma':np.ones(1),'scaleVector':np.ones(4)}
        np.testing.assert_array_equal(components(d,1.)['Hresponse'],[[-1.],[1.]])
    def test_mapping_and_admission_are_distinct(self):
        g=np.array([[1.,2.]]);p=np.array([[3.,4.]]);u=np.array([[2.,6.]]);v=np.array([[5.,8.]])
        a,_=analyze(g,self.parts(g.shape),p,u,v);r=dict(zip(columns(),a[0]))
        self.assertEqual([r['saved.'+k] for k in ('proposal','writer','admitted','mapping','admission')],[11.,14.,21.,3.,7.])
        self.assertEqual(r['saved.totalRegrouping'],0.)
    def test_supplied_finite_secant_not_zero(self):
        from contextual_writer_program import finite_add_array
        parent=finite_add_array(finite_add_array(np.array([[256.]]),np.array([[1.]])),np.array([[0.]]))
        half=finite_add_array(finite_add_array(np.array([[256.]]),np.array([[1.5]])),np.array([[0.]]))
        full=finite_add_array(finite_add_array(np.array([[256.]]),np.array([[2.]])),np.array([[0.]]))
        self.assertEqual(float((2*(half-parent)-(full-parent))[0,0]),2.)
    def test_duplicate_pairs_rejected(self):
        d={'h':np.zeros((2,2)),'z':np.zeros((2,2)),'target':np.zeros((2,2)),'zt':np.zeros((2,2)),
           'weights':np.ones(2),'edges':np.array([[0,1],[0,1]]),'pairWeights':np.ones(2),'gamma':np.ones(2),'scaleVector':np.ones(4)}
        with self.assertRaisesRegex(ValueError,'Disjoint'):components(d,1.)
    def test_nonfinite_rejected(self):
        a=np.array([[float('nan')]])
        with self.assertRaisesRegex(ValueError,'nonfinite'):analyze(a,self.parts(a.shape),a,a,a)
    def test_product_work_account(self):
        a=np.ones((2,3));_,d=analyze(a,self.parts(a.shape),a,a,a)
        self.assertEqual(d['coordinateProducts'],31*2*3);self.assertEqual(len(d['columns']),52)


if __name__=='__main__':
    import hashlib,sys
    from writer_direction_study import OWNED
    started=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Mechanics);result=unittest.TextTestRunner().run(suite)
    data={'testsRun':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'secondsAfterImports':time.perf_counter()-started,
          'sourceHashes':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in OWNED}}
    with Path(sys.argv[1]).open('x') as output:json.dump(data,output,indent=2,allow_nan=False)
    if not result.wasSuccessful():raise SystemExit(1)
