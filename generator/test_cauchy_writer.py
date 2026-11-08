"""Tiny supplied mechanics only; no model, saved scientific state or image."""
import unittest
import numpy as np
from cauchy_writer_math import propose,pushforward,admit
from joint_writer_math import pullback,objective,propose as old_propose


def fixture():
    h=np.array([[1.,2.,-1.],[2.,-1.,3.],[.5,2.,1.],[3.,2.,-.5]])
    gamma=np.array([.5,2.,-1.]);epsilon=.1
    norm=lambda x:gamma*x/np.sqrt(np.mean(x*x,axis=1)+epsilon)[:,None]
    target=h+np.array([[.1,-.3,.2],[-.2,.4,-.1],[.3,.1,-.2],[-.1,.2,.1]])
    z=norm(h);zt=norm(target);w=np.array([.1,.2,.3,.4]);edges=np.array([[0,1],[2,3]],np.int64);v=np.array([.25,.75])
    scales=objective(target,zt,target,zt,w,edges,v)['nativeEnergies']
    g,_,_=old_propose(h,z,target,zt,w,edges,v,gamma,epsilon,scales)
    return dict(h=h,z=z,target=target,zt=zt,gradient=g,weights=w,edges=edges,pairWeights=v,gamma=gamma,scaleVector=np.array(list(scales.values()))),[np.array([0,0,1,1]),np.array([0,1,0,1])],epsilon


class Mechanics(unittest.TestCase):
    def test_forward_adjoint_nonuniform_gamma(self):
        d,_,eps=fixture();q=np.arange(12).reshape(4,3)/7;u=np.arange(12,24).reshape(4,3)/11
        j=pushforward(d['h'],q,d['gamma'],eps);jt=pullback(d['h'],u,d['gamma'],eps)
        self.assertAlmostEqual(float(np.sum(j*u)),float(np.sum(q*jt)),places=12)
        self.assertGreater(float(np.max(np.abs(j-pullback(d['h'],q,d['gamma'],eps)))),.1)

    def test_dense_jacobian_witness(self):
        d,_,eps=fixture();h=d['h'];q=np.arange(12).reshape(4,3)/7;rows=[]
        for x,dx in zip(h,q):
            s=np.sqrt(np.mean(x*x)+eps);matrix=np.diag(d['gamma'])@(np.eye(3)/s-np.outer(x,x)/(3*s**3));rows.append(matrix@dx)
        np.testing.assert_allclose(pushforward(h,q,d['gamma'],eps),rows,rtol=2e-15,atol=2e-15)

    def test_joint_projection_and_residual_square(self):
        d,labels,eps=fixture();r,q,jq,a=propose(d,labels,[2,2],eps)
        self.assertLess(a['gDotQ'],0);self.assertAlmostEqual(a['gDotQ'],a['negativePreconditionedGradientEnergy'],places=13)
        self.assertGreaterEqual(a['linearResidualSquareObjective']['score'],0);self.assertAlmostEqual(a['modelPolynomialScore'],a['linearResidualSquareObjective']['score'],places=13)
        self.assertLessEqual(a['alpha']*a['rowRelativeTau'],1/16+1e-15)
        self.assertGreater(a['predictedReduction'],0)

    def test_forward_response_uses_both_bases(self):
        d,labels,eps=fixture();_,q,jq,a=propose(d,labels,[2,2],eps);i,j=d['edges'].T
        wrong=pushforward(d['h'][i],q[j]-q[i],d['gamma'],eps)
        self.assertGreater(float(np.max(np.abs((jq[j]-jq[i])-wrong))),1e-4)

    def test_zero_step_preserves_signed_zero(self):
        old=np.array([[0.,-0.,1.]],'<f4');new=admit([old],[np.zeros_like(old,dtype=np.float64)],0)[0]
        self.assertEqual(old.tobytes(),new.tobytes());self.assertIsNot(old,new)

    def test_zero_gradient_aliases_parent(self):
        d,labels,eps=fixture();d['gradient']=np.zeros_like(d['h']);_,q,_,a=propose(d,labels,[2,2],eps)
        self.assertEqual(a['alpha'],0);self.assertFalse(q.any());self.assertEqual(a['predictedReduction'],0)

    def test_empty_route_rejected(self):
        d,labels,eps=fixture()
        with self.assertRaises(ValueError):propose(d,labels,[3,2],eps)

    def test_atom_admission_is_fp32(self):
        old=np.array([[1.,-2.,0.]],'<f4');r=np.array([[.3,.7,-.1]],np.float64)
        new=admit([old],[r],.125)[0];self.assertEqual(new.dtype,np.dtype('<f4'));np.testing.assert_array_equal(new,(old.astype(np.float64)+r*.125).astype('<f4'))


if __name__=='__main__':unittest.main()
