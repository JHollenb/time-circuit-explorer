"""Tiny independent finite/enumeration witnesses; no scientific corpus/model."""
import copy
import itertools
import math
import unittest
import numpy as np
from contextual_writer_program import add_words,decode_words,finite_add_array
from finite_writer_capacity import enumerate_capacity,pair_scores,scale,choose
from publish_finite_writer_capacity import validate_component


def std_add(s,a): return np.asarray(decode_words(add_words(s,a)),np.float64)


def assay(y,atoms,labels,edges,s=None,current_override=None):
    y=np.asarray(y,np.float64);atoms=np.asarray(atoms,np.float32);n,width=y.shape;s=np.zeros_like(y) if s is None else np.asarray(s,np.float64)
    current=np.stack([std_add(x,atoms[a]) for x,a in zip(s,labels)]) if current_override is None else current_override
    rows=[]
    result,selected,report=enumerate_capacity(np.zeros_like(y),s,y,atoms,current,np.asarray(labels,np.int64),np.full(n,1/n),np.asarray(edges,np.int64),np.full(len(edges),1/len(edges)),None,rows.append)
    return result,selected,report,rows


class FiniteCapacityTests(unittest.TestCase):
    def test_pair_scores_against_scalar_coordinate_expansion(self):
        left=np.array([[0.,1.,-.5],[1.,0.,.5]]);right=np.array([[.5,0.,1.],[0.,1.,-1.]])
        yi=np.array([.25,.5,0.]);yj=np.array([.5,.25,.5]);alpha,beta,gamma=.25,.5,2.
        *_,scores=pair_scores(left,right,yi,yj,alpha,beta,gamma)
        for a,b in itertools.product(range(2),repeat=2):
            ni=math.fsum((float(x)-float(y))**2 for x,y in zip(left[a],yi));nj=math.fsum((float(x)-float(y))**2 for x,y in zip(right[b],yj))
            nr=math.fsum(((float(v)-float(u))-(float(y)-float(x)))**2 for u,v,x,y in zip(left[a],right[b],yi,yj))
            self.assertEqual(scores[a,b],(alpha*ni+beta*nj)+gamma*nr)

    def test_disjoint_components_equal_global_bruteforce(self):
        y=np.array([[.25,-.25],[.75,-.75],[.625,-.5],[.375,-.5],[.75,-.75]])
        atoms=np.array([[0.,0.],[1.,-1.]],np.float32);edges=[(0,1),(2,3)]
        result,selected,report,_=assay(y,atoms,[0]*5,edges);e,t=report['referenceEnergies'];w=.2/e;v=.5/t
        def objective(labels):
            z=[std_add([0.,0.],atoms[a]) for a in labels];parts=[]
            for i,j in edges:
                ni=math.fsum(float(x-y0)**2 for x,y0 in zip(z[i],y[i]));nj=math.fsum(float(x-y0)**2 for x,y0 in zip(z[j],y[j]))
                nr=math.fsum(float((b-a)-(d-c))**2 for a,b,c,d in zip(z[i],z[j],y[i],y[j]));parts.append((w*ni+w*nj)+v*nr)
            parts.append(w*math.fsum(float(x-y0)**2 for x,y0 in zip(z[4],y[4])))
            return math.fsum(parts)
        candidates=list(itertools.product(range(2),repeat=5));best=min(candidates,key=objective)
        self.assertEqual(tuple(selected),best);self.assertEqual(report['scores']['capacity']['joint'],objective(best))
        np.testing.assert_array_equal(result,np.stack([std_add([0.,0.],atoms[a]) for a in best]))

    def test_joint_optimum_can_worsen_endpoint_component(self):
        _,selected,report,_=assay([[.375],[.625]],[[0.],[1.]],[0,1],[(0,1)])
        self.assertEqual(list(selected),[0,0]);a=report['scores']['capacity'];b=report['scores']['current']
        self.assertLess(a['joint'],b['joint']);self.assertGreater(a['endpoint'],b['endpoint']);self.assertLess(a['response'],b['response'])

    def test_zero_reference_uses_absolute_components(self):
        _,selected,report,_=assay([[0.],[0.]],[[0.],[1.]],[1,1],[(0,1)])
        self.assertEqual(report['referenceEnergies'],[0.,0.]);self.assertEqual(list(selected),[0,0]);r=report['scores']['capacity']['historicalReduction']
        self.assertIsNone(r['endpointRatio']);self.assertIsNone(r['responseRatio']);self.assertEqual(r['endpointScore'],0.)

    def test_overlapping_graph_and_nonfinite_energy_are_integrity_errors(self):
        with self.assertRaises(ValueError):assay([[0.],[.5],[1.]],[[0.],[1.]],[0,0,0],[(0,1),(1,2)])
        for value in [float('nan'),float('inf'),-1.]:
            with self.assertRaises(ValueError):scale([1.],value)

    def test_admission_and_exact_tie_policy(self):
        s=np.array([1.,-1.]);atoms=np.array([[2**-8,-2**-8],[3*2**-8,-3*2**-8]],np.float32)
        for a in atoms:np.testing.assert_array_equal(finite_add_array(s,a),std_add(s,a))
        index,detail=choose(np.ones((2,2)));self.assertEqual(index,0);self.assertEqual(detail,{'ties':4,'nextDistinctGap':None})

    def test_scalar_publisher_rejects_changed_score_or_noncanonical_choice(self):
        *_,rows=assay([[.375],[.625]],[[0.],[1.]],[0,1],[(0,1)])
        validate_component(rows[0],2)
        bad=copy.deepcopy(rows[0]);bad['scores'][0][0]+=1
        with self.assertRaises(ValueError):validate_component(bad,2)
        bad=copy.deepcopy(rows[0]);bad['selectedIndex']=3
        with self.assertRaises(ValueError):validate_component(bad,2)

    def test_current_state_must_be_the_frozen_feasible_route(self):
        with self.assertRaises(ValueError):assay([[.25],[.75]],[[0.],[1.]],[0,0],[(0,1)],current_override=np.ones((2,1)))


if __name__=='__main__':unittest.main()
