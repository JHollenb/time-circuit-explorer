"""Tiny independent instruments for the proposed positive reader laws."""
import subprocess
import sys
from pathlib import Path
import unittest


class LandmarkMechanics(unittest.TestCase):
    def code(self,source):
        p=subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_laws_mask_occurrences_and_resume(self):
        self.code('''import numpy as np
from context_landmarks import stream,METHODS
from encoder_atoms import words,floats
q=np.array([[.4,0.],[.4,0.],[.4,0.],[.4,0.]])
k=np.array([[9.,0.],[1.,0.],[1.,0.],[-1.,0.]])
v=np.array([[90.,-90.],[2.,1.],[2.,1.],[-3.,4.]])
p={'q':words(q),'k':words(k),'v':words(v),'mask':np.array([0,1,1,1]),'positions':np.arange(4)}
a={'landmarks':np.array([[0.,0.],[2.,0.]]),'metricRoot':np.eye(2),'bandwidthSquared':np.asarray(1.)}
o,s=stream(p,a);q,k,v=[floats(p[n]).astype(float)/ (2**.25 if n!='v' else 1.) for n in ['q','k','v']]
for i in range(1,4):
 ids=np.flatnonzero(p['mask'][:i+1]);score=a['landmarks']@k[ids].T;K=np.exp(score);Z=K.sum(1);N=K@v[ids]
 logits=-np.sum((q[i]-a['landmarks'])**2,axis=1)/2;w=np.exp(logits-logits.max());w/=w.sum()
 direct=[w@(N/Z[:,None]),(N/Z[:,None])[np.argmin(-logits)],(w@N)/(w@Z),v[ids].mean(0)]
 for name,h in zip(METHODS,direct):assert np.allclose(o[name+'_raw'][i],h,rtol=2e-14,atol=2e-14)
assert np.linalg.norm(o['landmark64_raw'][-1]-o['kernel64_raw'][-1])>.1
from context_landmark_study import direct
check=direct(p,{**a,'covariance':np.eye(2)},[1,3])
for name in METHODS:assert np.allclose(check[name],o[name+'_raw'][[1,3]],rtol=2e-14,atol=2e-14)
assert np.array_equal(o['landmark64_raw'][0],np.zeros(2)) and s['count']==3
assert all(o[n+'_hullViolation'].max()<1e-13 for n in METHODS)
_,cut=stream({key:value[:2] for key,value in p.items()},a)
tail,end=stream({key:value[2:] for key,value in p.items()},a,state=cut)
assert np.array_equal(end['M'],s['M']) and np.array_equal(end['z'],s['z'])
assert np.allclose(tail['landmark64_raw'],o['landmark64_raw'][2:],rtol=2e-14,atol=2e-14)
bad={**a,'bandwidthSquared':np.asarray(2.)}
try:stream({key:value[2:] for key,value in p.items()},bad,state=cut)
except ValueError as e:assert 'asset' in str(e)
else:raise AssertionError('Changed reader asset accepted')
try:stream(p,a,state=cut)
except ValueError as e:assert 'clock' in str(e)
else:raise AssertionError('Wrong clock accepted')
''')

    def test_large_scores_stay_finite_and_convex(self):
        self.code('''import numpy as np
from context_landmarks import stream,METHODS
from encoder_atoms import words
p={'q':words(np.array([[1.,0.],[1.,0.],[1.,0.]])),'k':words(np.array([[5.,1.],[-5.,1.],[6.,1.]])),
 'v':words(np.array([[1.,2.],[3.,-4.],[-2.,1.]])),'mask':np.ones(3,dtype=int),'positions':np.arange(3)}
a={'landmarks':np.array([[500.,-500.],[-200.,300.]]),'metricRoot':np.eye(2),'bandwidthSquared':np.asarray(1e5)}
o,s=stream(p,a)
assert s['maximum'].max()>1000 and np.isfinite(s['z']).all()
for name in METHODS:
 assert np.isfinite(o[name+'_raw']).all() and o[name+'_hullViolation'].max()<1e-13
assert o['partitionLogRange'].max()>1000
''')

    def test_source_covariance_scale_and_deterministic_landmarks(self):
        self.code('''import numpy as np
from context_landmarks import covariance_block,covariance_merge,fit,routes
from encoder_atoms import words
from context_source_geometry import rotate
rng=np.random.default_rng(91);x=1e5+rng.normal(size=(9,4));n,mu,M=covariance_merge(covariance_block(x[:4]),covariance_block(x[4:]))
assert n==9 and np.allclose(M,np.cov(x,rowvar=False,bias=True)*9,rtol=2e-10,atol=2e-10)
class Store:
 rowsRead=0;rowBytesRead=0
 def __init__(self):self.table=words(rng.normal(size=(7,16)))
 def gather(self,ids):self.rowsRead+=len(ids);self.rowBytesRead+=len(ids)*32;return self.table[ids]
class Prefix:
 m={'headDim':4,'hiddenSize':4,'qWidth':4,'tokenPolicy':{'padID':6}}
 rope=words(np.stack([np.ones((5,4)),np.zeros((5,4))]))
 def __init__(self):self.store=Store()
p=Prefix();asset,receipt=fit(p,np.arange(6),rank=4,candidates=12,seed=3,batch=2)
again,_=fit(p,np.arange(6),rank=4,candidates=12,seed=3,batch=3)
assert np.array_equal(asset['landmarkIdentities'],again['landmarkIdentities'])
keys=rotate(p.store.table[:6,8:12],p.rope)
from encoder_atoms import floats
y=floats(keys).reshape(-1,4).astype(float)/4**.25
assert np.allclose(asset['covariance'],np.cov(y,rowvar=False,bias=True),rtol=2e-14,atol=2e-14)
delta=np.array([.1,.2,-.3,.4]);assert np.allclose(delta@asset['covariance']@delta,np.var(y@delta),rtol=2e-14,atol=2e-14)
w,logw,dist=routes(np.array([delta]),asset);assert np.allclose(w.sum(1),1.) and np.all(w>=0)
assert np.isclose(asset['candidateWeights'].sum(),1.) and receipt['contextualAttentionCalls']==0
''')

    def test_target_assisted_capacity_bounds_keep_simplex_constraint(self):
        self.code('''import numpy as np
from landmark_routing_diagnostics import simplex,capacity
D=np.array([[0.,0.],[1.,0.],[0.,1.]])
T=np.array([[.2,.3],[2.,2.]])
fit,record=capacity(D,T,np.full((2,3),1/3))
assert np.allclose(fit['affine'],T,atol=2e-14)
assert np.allclose(fit['convex'][0],T[0],atol=2e-14)
assert np.allclose(fit['convex'][1],[.5,.5],atol=2e-14)
assert np.allclose(fit['convexWeights'].sum(1),1.) and fit['convexWeights'].min()>=0
assert np.all(fit['objectiveLowerBound']<=fit['objective']+1e-14)
assert fit['objectiveLowerBound'][1]>2.24 and record['affineMaximumNegativeWeightMass']>1
assert np.allclose(simplex(np.array([[10.,0.,-10.]])),[[1.,0.,0.]])
''')


if __name__=='__main__':unittest.main()
