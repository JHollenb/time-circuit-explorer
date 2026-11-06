"""Independent factor identities, signed cancellation and empty-tail mechanics."""
import unittest
import numpy as np
from rank_one_omission import compare


class RankOneOmission(unittest.TestCase):
    def test_independent_positive_distribution_and_factorization(self):
        rng=np.random.default_rng(2001);n=13;d=5;s=rng.normal(size=n);U=rng.normal(size=(n,d));V=rng.normal(size=(n,d));b=rng.normal(size=d);z=U@b;chosen=np.array([3,0,11,6]);delta=rng.normal(size=d);beta=.4
        hf,hg,parts,fields=compare(s,U,V,z,chosen,delta,beta);tail=np.ones(n,dtype=bool);tail[chosen]=False;p=np.exp(s[tail]-s[tail].max());p/=p.sum();mu=p@U[tail];t=(U[tail]-mu)@delta;tr=beta*(z[tail]-p@z[tail])
        selected=np.exp(s[chosen]+U[chosen]@delta);zs=selected.sum();Hs=selected@V[chosen]/zs
        def independent(x):
            w=np.exp(s[tail])*np.exp(mu@delta)*(1+x+.5*x*x);return (zs*Hs+w@V[tail])/(zs+w.sum())
        self.assertTrue(np.allclose(hf,independent(t),rtol=3e-13,atol=3e-13));self.assertTrue(np.allclose(hg,independent(tr),rtol=3e-13,atol=3e-13));self.assertTrue(np.allclose(sum(parts.values()),hf-hg,atol=1e-15));self.assertLess(abs(fields['partitionDifferenceClosure']),2e-13);self.assertLess(fields['tailValueClosure'],3e-13);self.assertGreaterEqual(fields['quadraticMass'],0)

    def test_rank_is_full_identity_and_selected_only(self):
        s=np.array([.1,-.3,1.]);U=np.array([[1.,2.],[2.,3.],[-1.,0.]]);V=U[:,::-1];delta=np.array([.2,-.1]);z=U@delta
        a,b,t,f=compare(s,U,V,z,np.array([0]),delta,1.)
        self.assertTrue(np.allclose(a,b,atol=1e-15));self.assertLess(f['omittedScoreEnergy'],1e-30)
        a,b,t,f=compare(s,U,V,z,np.arange(3),delta,1.)
        self.assertTrue(np.array_equal(a,b));self.assertTrue(all(np.array_equal(v,np.zeros(2)) for v in t.values()));self.assertEqual(f['hasTail'],0);self.assertTrue(np.isnan(f['kappaFull']))

    def test_oblique_projection_ratio_and_signed_linear_term(self):
        U=np.array([[-1.,0.],[1.,0.],[0.,1.]]);V=U.copy();s=np.zeros(3);chosen=np.array([2]);delta=np.array([1.,0.]);z=3*U[:,0]
        _,_,_,f=compare(s,U,V,z,chosen,delta,1.)
        self.assertEqual(f['omittedOverFullEnergy'],4.);self.assertEqual(f['linearMass'],-6.);self.assertEqual(f['quadraticMass'],2.);self.assertAlmostEqual(f['kappaFull']-f['kappaRank'],-4.)
        with self.assertRaises(ValueError):compare(s,U,V,z,np.array([0,0]),delta,1.)

    def test_signed_error_change_and_zero_denominators(self):
        from rank_one_omission import TERMS
        from rank_one_omission_study import score
        rng=np.random.default_rng(2002);parts={k:rng.normal(size=(7,8))*.01 for k in TERMS};correction=sum(parts.values());baseline=rng.normal(size=(7,8));rows=score({'kind':'paired'},correction,baseline,parts,[('all',np.ones(7,dtype=bool))],2,4)
        for r in rows:self.assertLess(abs(r['errorChangeIdentityDefect']),1e-15);self.assertLess(r['vectorClosureMaxAbs'],1e-15)
        z=np.zeros_like(correction);row=score({},z,z,{k:z for k in TERMS},[('all',np.ones(7,dtype=bool))],2,4)[0]
        self.assertIsNone(row['correctionEnergyOverInitialError']);self.assertTrue(all(v is None for v in row['signedAlignmentOverInitialError'].values()))


if __name__=='__main__':unittest.main()
