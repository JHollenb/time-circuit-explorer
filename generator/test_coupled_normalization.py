"""Seven independent tiny supplied witnesses; no real state/model loading."""
import copy
import math
import unittest
import numpy as np
from coupled_normalization import analyze,observe_sum,ANALYSIS_NAMES,EDGES,CONTRASTS
from publish_coupled_normalization import validate_analysis,validate_sum


def fixture():
    states={k:np.array(v,dtype=np.float64) for k,v in {'a':[3,4],'b':[6,8],'p':[4,3],'q':[7,6],'Xno':[6,7],'Xon':[7,7]}.items()}
    gamma=np.array([2.,3.]);eps=.5
    ideal={k:np.array([float(g)*float(x)/math.sqrt(math.fsum(float(y)**2 for y in h)/2+eps) for g,x in zip(gamma,h)]) for k,h in states.items()}
    return states,ideal,gamma,eps


class CoupledNormalizationTests(unittest.TestCase):
    def test_independent_direction_and_denominator_identity(self):
        states,z,gamma,eps=fixture();vectors,report=analyze(states,z,gamma,eps)
        h,k=states['a'],states['b'];sh=math.sqrt(13);sk=math.sqrt(50.5)
        expected_d=[2*3/sk,3*4/sk];expected_b=[2*3*(1/sk-1/sh),3*4*(1/sk-1/sh)]
        np.testing.assert_allclose(vectors['edge-native-D'],expected_d,rtol=2e-15,atol=0)
        np.testing.assert_allclose(vectors['edge-native-B'],expected_b,rtol=2e-15,atol=0)
        np.testing.assert_allclose(np.array(expected_d)+expected_b,z['b']-z['a'],rtol=2e-15,atol=2e-15)
        self.assertEqual(set(vectors),set(ANALYSIS_NAMES));self.assertEqual(len(vectors),79);self.assertEqual(validate_analysis(report,2,eps),402)

    def test_supplied_finite_discrepancy_is_not_lost(self):
        states,z,gamma,eps=fixture();offsets={k:np.array([i/16,-i/32]) for i,k in enumerate(states)}
        supplied={k:z[k]+offsets[k] for k in states};v,r=analyze(states,supplied,gamma,eps)
        for k in states:np.testing.assert_allclose(v['state-'+k+'-rho'],offsets[k],atol=2e-15,rtol=0)
        for name,(a,b) in EDGES.items():np.testing.assert_allclose(v['edge-'+name+'-R'],offsets[b]-offsets[a],atol=3e-15,rtol=0)
        self.assertGreater(r['states']['q']['rhoNormSquared'],0);validate_analysis(r,2,eps)

    def test_negative_gram_and_exact_contrast_signs(self):
        closure,row=observe_sum(np.array([1.,0.]),{'D':np.array([10.,0.]),'B':np.array([-9.,0.]),'R':np.zeros(2),'g':np.zeros(2)},1.)
        self.assertEqual(row['innerProducts']['D:B'],-90.);self.assertEqual(row['gramSum'],1.);self.assertEqual(validate_sum(row,('D','B','R','g'),1.,2),30)
        states,z,gamma,eps=fixture();v,_=analyze(states,z,gamma,eps)
        expected={'E':('own','native'),'Uno':('own','no'),'Vno':('no','native'),'Uon':('own','on'),'Von':('on','native')}
        self.assertEqual(CONTRASTS,expected)
        for name,(left,right) in expected.items():
            for term in ('D','B','R'):np.testing.assert_array_equal(v['factor-'+name+'-'+term],v['edge-'+left+'-'+term]-v['edge-'+right+'-'+term])
        np.testing.assert_array_equal(closure,[0.,0.])

    def test_zero_response_preserves_absolute_error(self):
        states,z,gamma,eps=fixture();z['b']=z['a'].copy();v,r=analyze(states,z,gamma,eps)
        self.assertEqual(r['response']['normSquared']['N'],0.);self.assertGreater(r['response']['rms']['E'],0)
        self.assertTrue(all(x is None for x in r['factors']['E']['relativeToNative'].values()));self.assertEqual(validate_analysis(r,2,eps),402)

    def test_endpoint_metric_has_independent_known_coordinates(self):
        states,z,gamma,eps=fixture();z['a']=np.array([3.,4.]);z['p']=np.array([6.,8.]);v,r=analyze(states,z,gamma,eps);e=r['endpoints']['p']
        self.assertEqual(e['predictionSquared'],100.);self.assertEqual(e['targetSquared'],25.);self.assertEqual(e['errorSquared'],25.);self.assertEqual(e['innerProduct'],50.)
        self.assertEqual(e['relativeRMS'],1.);self.assertEqual(e['cosine'],1.);np.testing.assert_array_equal(v['endpoint-p-error'],[3.,4.])

    def test_publisher_rejects_scalar_and_domain_tampering(self):
        states,z,gamma,eps=fixture();_,r=analyze(states,z,gamma,eps)
        for path,value in [(('states','a','radius'),999.),(('factors','E','gramSum'),999.),(('endpoints','p','cosine'),999.)]:
            bad=copy.deepcopy(r);node=bad
            for key in path[:-1]:node=node[key]
            node[path[-1]]=value
            with self.assertRaises(ValueError):validate_analysis(bad,2,eps)
        bad=copy.deepcopy(r);del bad['edges']['native']['innerProducts']['D:B']
        with self.assertRaises(ValueError):validate_analysis(bad,2,eps)

    def test_invalid_inputs_and_nonzero_final_closure(self):
        states,z,gamma,eps=fixture()
        for invalid_eps in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):analyze(states,z,gamma,invalid_eps)
        with self.assertRaises(ValueError):analyze(states,z,[1.],eps)
        bad=copy.deepcopy(states);bad['q'][0]=float('inf')
        with self.assertRaises(ValueError):analyze(bad,z,gamma,eps)
        huge=float(2**54);closure,row=observe_sum(np.array([1.]),{'D':np.array([huge]),'B':np.zeros(1),'R':np.zeros(1),'g':np.array([1.-huge])},1.)
        self.assertEqual(closure[0],1.);self.assertEqual(row['normSquared']['closure'],1.)


if __name__=='__main__':unittest.main()
