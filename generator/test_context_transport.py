"""Small independent checks for centered weighted moments and query transport."""
import subprocess
import sys
from pathlib import Path
import unittest


class TransportMechanics(unittest.TestCase):
    def code(self,source):
        p=subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_weighted_centered_cross_moment_with_large_offsets(self):
        self.code('''import numpy as np
from context_transport import new_state,update
U=1e6+np.array([[1.,-2.],[3.,1.],[-1.,5.],[2.,0.]])
V=-2e6+np.array([[3.,1.],[-2.,4.],[2.,-1.],[0.,3.]])
scores=np.array([[1000.,-1000.],[1001.,-999.],[998.,-1003.],[1002.,-1001.]])
s=new_state(2,2)
for u,v,score in zip(U,V,scores):update(s,u,v,score)
for a in range(2):
 w=np.exp(scores[:,a]-scores[:,a].max());w/=w.sum();mu=w@U;h=w@V
 C=np.einsum('j,ji,jk->ik',w,V-h,U-mu)
 assert np.allclose(s['C'][a]/s['z'][a],C,rtol=1e-9,atol=1e-9)
 assert np.allclose(s['M'][a]/s['z'][a],h,rtol=2e-15,atol=2e-9)
assert np.isfinite(s['C']).all()
''')

    def test_actual_prefix_jacobian_mask_resume_and_identity(self):
        self.code('''import numpy as np
from context_transport import stream
from context_landmarks import stream as zero
from encoder_atoms import words,floats
rng=np.random.default_rng(101);q=rng.normal(size=(5,4));k=rng.normal(size=(5,4));v=rng.normal(size=(5,4))
p={'q':words(q),'k':words(k),'v':words(v),'mask':np.array([0,1,1,0,1]),'positions':np.arange(5)}
a={'landmarks':rng.normal(size=(3,4)),'metricRoot':np.eye(4),'keyCenter':np.array([1.,-2.,3.,4.])}
o,s=stream(p,a);old,_=zero(p,{**a,'bandwidthSquared':np.asarray(1.)})
assert np.allclose(o['nearest64_raw'],old['nearest64_raw'],rtol=2e-14,atol=2e-14)
for i in range(1,5):
 j=np.flatnonzero(p['mask'][:i+1]);ell=a['landmarks'][o['nearestIndex'][i]];Y=floats(p['k'][j]).astype(float)/4**.25;V=floats(p['v'][j]).astype(float)
 score=Y@ell;w=np.exp(score-score.max());w/=w.sum();h=w@V;mu=w@Y;J=((V-h)*w[:,None]).T@(Y-mu)
 delta=floats(p['q'][i]).astype(float)/4**.25-ell
 assert np.allclose(o['transport64_raw'][i],h+J@delta,rtol=2e-14,atol=2e-14)
_,cut=stream({key:value[:3] for key,value in p.items()},a)
tail,end=stream({key:value[3:] for key,value in p.items()},a,state=cut)
assert np.array_equal(end['C'],s['C']) and np.array_equal(end['M'],s['M'])
assert np.allclose(tail['transport64_raw'],o['transport64_raw'][3:],rtol=2e-14,atol=2e-14)
try:stream(p,a,state=cut)
except ValueError as e:assert 'clock' in str(e)
else:raise AssertionError('Wrong clock accepted')
try:stream({key:value[3:] for key,value in p.items()},{**a,'keyCenter':np.zeros(4)},state=cut)
except ValueError as e:assert 'asset' in str(e)
else:raise AssertionError('Changed state asset accepted')
''')

    def test_first_order_error_scales_and_signed_weights_remain_visible(self):
        self.code('''import numpy as np
Y=np.array([-1.,0.,1.]);V=np.array([0.,1.,4.]);w=np.ones(3)/3;h=w@V;mu=w@Y;J=np.sum(w*(V-h)*(Y-mu))
errors=[]
for delta in [.01,.005,.0025]:
 weights=np.exp(delta*Y);weights/=weights.sum();errors.append(abs((weights@V)-(h+J*delta)))
assert 3.9<errors[0]/errors[1]<4.1 and 3.9<errors[1]/errors[2]<4.1
delta=3.;effective=w*(1+delta*(Y-mu))
assert np.isclose(effective.sum(),1.) and effective.min()<0 and h+J*delta>V.max()
from context_transport import rounded
raw=np.array([[1.,2.],[1e100,0.],[np.inf,1.]])
bits,valid=rounded(raw,np.array([1,1,0],dtype=bool));assert np.array_equal(valid,[1,0,0])
''')

    def test_receiving_graph_and_effect_normalizer_expose_small_head(self):
        self.code('''import numpy as np
from transport_consumer import consume,effect_scores
from encoder_atoms import words,floats
rng=np.random.default_rng(104);e=words(rng.normal(size=(3,4)));rest=rng.normal(size=(3,4));head=rng.normal(size=(3,2));matrix=rng.normal(size=(4,2))
gamma=words(np.ones(4));weights=[rng.normal(size=(5,4)),rng.normal(size=(5,4)),rng.normal(size=(4,5))];positions=np.array([0,2])
zero=consume(e,rest,np.zeros_like(head),matrix,gamma,1e-6,positions,weights)
exact=consume(e,rest,head,matrix,gamma,1e-6,positions,weights)
assert np.any(exact['rms']!=zero['rms']) and np.any(exact['state']!=zero['state'])
blocked=consume(e,rest,head,np.zeros_like(matrix),gamma,1e-6,positions,weights)
blocked0=consume(e,rest,np.zeros_like(head),np.zeros_like(matrix),gamma,1e-6,positions,weights)
assert all(np.array_equal(blocked[k],blocked0[k]) for k in blocked)
native=np.full((3,4),1e4)+np.arange(12).reshape(3,4);deleted=native-.001
score=effect_scores(deleted,native,deleted,np.array([1,0,1],dtype=bool))
assert score['head0Effect']['relativeRms']==1 and score['endpoint']['relativeRms']<1e-6 and score['invalidQueries']==1
''')

    def test_rms_is_row_local_and_power_two_scale_invariant(self):
        self.code('''import numpy as np
from transport_consumer import rms_norm,mlp
from encoder_atoms import words,floats
x=np.array([[1.,2.,3.,4.],[7.,-1.,2.,0.]])
gamma=words(np.ones(4));a=rms_norm(words(x),gamma,0.)
assert np.array_equal(a,rms_norm(words(x*2),gamma,0.))
y=x.copy();y[1]*=10
assert np.array_equal(a[0],rms_norm(words(y),gamma,0.)[0])
assert np.array_equal(rms_norm(words(np.zeros((2,4))),gamma,1e-6),words(np.zeros((2,4))))
assert np.array_equal(mlp(words(x),np.zeros((3,4)),np.ones((3,4)),np.ones((4,3))),words(np.zeros((2,4))))
''')


if __name__=='__main__':unittest.main()
