"""Twelve supplied mathematical/control witnesses; no real arrays or models."""
import math
import struct
import unittest
import numpy as np
from joint_writer_math import objective,pullback,pair_solve,propose,dose_atoms,choose,measure,candidate_add,CandidateAdmissionError
from contextual_writer_program import finite_add_array


def ideal(h,gamma,eps):
    return np.array([[g*x/math.sqrt(math.fsum(float(y)**2 for y in row)/len(row)+eps) for g,x in zip(gamma,row)] for row in h])


class JointWriterTests(unittest.TestCase):
    def test_ideal_pullback_matches_independent_coordinate_differences(self):
        h=np.array([[1.,2.]]);gamma=np.array([2.,-.5]);u=np.array([[.3,-.7]]);eps=.3;expected=[]
        for col in range(2):
            delta=np.zeros_like(h);delta[0,col]=1e-5;expected.append(float(np.sum((ideal(h+delta,gamma,eps)-ideal(h-delta,gamma,eps))*u))/(2e-5))
        np.testing.assert_allclose(pullback(h,u,gamma,eps)[0],expected,rtol=2e-9,atol=1e-10)

    def test_four_loss_normalizers_and_zero_reference_are_explicit(self):
        target=np.array([[0.],[0.]]);h=np.array([[1.],[3.]]);w=np.array([.5,.5]);edges=np.array([[0,1]]);v=np.array([1.]);r=objective(h,2*h,target,target,w,edges,v)
        self.assertEqual(r['errorEnergies'],{'Hendpoint':5.,'Hresponse':4.,'Zendpoint':20.,'Zresponse':16.});self.assertEqual(r['score'],11.25)
        self.assertTrue(all(r['zeroReferenceUsesAbsolute'].values()))

    def test_graph_sign_and_combined_smooth_gradient(self):
        h=np.array([[1.,2.],[2.,1.]]);target=np.array([[.5,1.],[1.,1.5]]);gamma=np.array([2.,.5]);eps=.3;w=np.array([.4,.6]);edges=np.array([[0,1]]);v=np.array([.7]);zt=ideal(target,gamma,eps);z=ideal(h,gamma,eps)
        scales=objective(h,z,target,zt,w,edges,v)['nativeEnergies'];g,_,_=propose(h,z,target,zt,w,edges,v,gamma,eps,scales);expected=np.zeros_like(h)
        for i in range(2):
            for j in range(2):
                delta=np.zeros_like(h);delta[i,j]=1e-5
                a=objective(h+delta,ideal(h+delta,gamma,eps),target,zt,w,edges,v,scales)['score'];b=objective(h-delta,ideal(h-delta,gamma,eps),target,zt,w,edges,v,scales)['score'];expected[i,j]=(a-b)/1e-5
        np.testing.assert_allclose(g,expected,rtol=2e-8,atol=1e-9)

    def test_large_pair_penalty_preserves_exact_common_mode(self):
        g=np.ones((2,1));step,diagnostic=pair_solve(g,np.ones(2),np.ones(2),np.array([[0,1]]),np.array([1e12]),np.zeros(1))
        np.testing.assert_array_equal(step,[[-1.],[-1.]]);self.assertEqual(diagnostic['pairedRows'],2)
        self.assertGreater(abs(step[0,0]),1e9/(1+1e12))

    def test_unequal_z_scales_and_singleton_match_small_dense_reference(self):
        gradient=np.array([[1.,2.],[-3.,4.],[5.,6.]]);e=np.array([2.,3.,4.]);j=np.array([1.,2.,.5]);edges=np.array([[0,1]]);ph=np.array([7.]);pz=np.array([11.])
        matrix=np.diag(e);matrix[:2,:2]+=7*np.array([[1.,-1.],[-1.,1.]])+11*np.array([[1.,-2.],[-2.,4.]])
        step,d=pair_solve(gradient,e,j,edges,ph,pz);np.testing.assert_allclose(step,np.linalg.solve(matrix,-gradient),rtol=2e-15,atol=1e-15);self.assertEqual(d['singletonRows'],1)

    def test_half_dose_refits_old_function_instead_of_halving_dictionary(self):
        labels=np.array([0,0]);old=np.array([[2.],[4.]]);step=np.array([[4.],[-2.]]);scales={'Hendpoint':1.,'Hresponse':1.,'Zendpoint':1.,'Zresponse':1.}
        atoms,_=dose_atoms(labels,1,old,step,np.array([.5,.5]),np.array([[0,1]]),np.array([1.]),scales)
        self.assertEqual(float(atoms[0][0,0]),4.);self.assertEqual(float(atoms[1][0,0]),3.5);self.assertNotEqual(float(atoms[1][0,0]),float(atoms[0][0,0])*.5)
        self.assertFalse(np.array_equal(np.full((2,1),3.),old))

    def test_changing_slot_zero_rebuilds_both_finite_admissions(self):
        def scalar(value):
            bits=struct.unpack('<I',struct.pack('<f',value))[0];word=((bits+0x7fff+((bits>>16)&1))>>16)&65535
            return struct.unpack('<f',struct.pack('<I',word<<16))[0]
        parent=scalar(scalar(256.+1.)-256.);candidate=scalar(scalar(256.+2.)-256.)
        actual=finite_add_array(finite_add_array([[256.]],[[2.]]),[[-256.]])
        self.assertEqual(parent,0.);self.assertEqual(candidate,2.);self.assertEqual(float(actual[0,0]),candidate);self.assertNotEqual(scalar(parent+1.),candidate)

    def test_exact_parent_is_available_and_wins_ties(self):
        parent={'id':'parent','score':1.};tie={'id':'newRoute','score':1.};good={'id':'better','score':.9}
        self.assertIs(choose(parent,[tie]),parent);self.assertIs(choose(parent,[tie,good]),good)

    def test_invalid_nonfinite_or_overlapping_measures_are_mechanics_errors(self):
        with self.assertRaises(ValueError):measure([1,1,1],[[0,1],[1,2]],[1,1],3)
        with self.assertRaises(ValueError):measure([1,1],[[0.,1.]],[1],2)
        with self.assertRaises(ValueError):pair_solve([[float('nan')],[0.]],[1,1],[1,1],[[0,1]],[1],[1])
        with self.assertRaises(ValueError):pullback([[1.,2.]],[[1.,2.]],[1.,1.],0.)
        with self.assertRaises(ValueError):choose({'score':1.},[{'score':float('inf')}])

    def test_proposal_preserves_objective_pair_residual_order(self):
        h=np.array([[1e16],[1e16]]);target=np.array([[1.],[2.]]);w=np.array([.5,.5]);edges=np.array([[0,1]]);scales={'Hendpoint':1e16,'Hresponse':1.,'Zendpoint':1.,'Zresponse':1.}
        g,_,_=propose(h,np.zeros_like(h),target,np.zeros_like(h),w,edges,np.ones(1),np.zeros(1),1.,scales)
        pair=(h[1]-h[0])-(target[1]-target[0]);expected=w[:,None]*(h-target)/1e16;expected[0]-=pair;expected[1]+=pair
        np.testing.assert_array_equal(g,expected);self.assertNotEqual(float(pair[0]),float(((h-target)[1]-(h-target)[0])[0]))

    def test_bad_full_dictionary_dose_preserves_valid_half(self):
        old=np.full((2,1),2e38);scales={k:1. for k in ('Hendpoint','Hresponse','Zendpoint','Zresponse')}
        atoms,record=dose_atoms([0,0],1,old,old,[.5,.5],np.array([[0,1]]),[1.],scales)
        self.assertIsNone(atoms[0]);self.assertTrue(np.isfinite(atoms[1]).all());self.assertEqual(record['admissions'][0]['status'],'numerically-inadmissible');self.assertEqual(record['admissions'][1]['status'],'admitted')
        with np.errstate(over='ignore'):
            with self.assertRaises(CandidateAdmissionError):candidate_add([[3e38]],[[3e38]])

    def test_publisher_rejects_partial_score_and_scalar_tampering(self):
        from publish_joint_writer import eligible_futures,scalar_objective,objective_record,FUTURES
        rows=[{'id':name,'slot':i//2,'dose':1. if i%2==0 else .5,'status':'construction-incomplete','completedHistories':[],'score':None,'programSHA256':None,'reason':'Nonfinite FP32 atom admission','stage':'tree-fit'} for i,name in enumerate(FUTURES)]
        self.assertEqual(eligible_futures(rows),[])
        rows[0].update(status='complete',completedHistories=['A','B','00'],score=0.,programSHA256='a'*64)
        with self.assertRaises(ValueError):eligible_futures(rows)
        scales={k:0. for k in ('Hendpoint','Hresponse','Zendpoint','Zresponse')};record=scalar_objective({k:float(i+1) for i,k in enumerate(scales)},scales)
        self.assertEqual(record['score'],2.5);self.assertGreater(objective_record(record,scales),0);record['score']=0.
        with self.assertRaises(ValueError):objective_record(record,scales)


if __name__=='__main__':unittest.main()
