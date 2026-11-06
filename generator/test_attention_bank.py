"""Tiny independent checks for GQA, source compilation and shared consumers."""
import subprocess
import sys
from pathlib import Path
import unittest


class CompleteBankMechanics(unittest.TestCase):
    def code(self,source):
        p=subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_grouped_heads_use_distinct_queries_and_correct_shared_operands(self):
        self.code('''import numpy as np
from attention_bank import packet,geometry
from encoder_atoms import words,floats
m={'headDim':4,'qWidth':32,'kvWidth':8}
full={'qpos':words(np.tile(np.repeat(np.arange(8),4),(3,1))),
 'kpos':words(np.tile(np.repeat([10,20],4),(3,1))),
 'v':words(np.tile(np.repeat([30,40],4),(3,1))),
 'mask':np.array([1,1,0]),'positions':np.arange(3),'ids':np.arange(3)}
for head in range(8):
 p=packet(full,m,head)
 assert np.all(floats(p['q'])==head)
 assert np.all(floats(p['k'])==[10,20][head//4])
 assert np.all(floats(p['v'])==[30,40][head//4])
 assert np.array_equal(p['mask'],full['mask']) and np.array_equal(p['positions'],full['positions'])
for head in [-1,8]:
 try:packet(full,m,head)
 except ValueError:pass
 else:raise AssertionError('Invalid query head accepted')
try:geometry({'headDim':4,'qWidth':12,'kvWidth':8})
except ValueError:pass
else:raise AssertionError('Invalid GQA accepted')
''')

    def test_source_recipe_matches_original_head_and_independent_group_covariance(self):
        self.code('''import numpy as np
from attention_bank import fit,head_asset
from context_landmarks import fit as old_fit
from context_source_geometry import rotate
from encoder_atoms import words,floats
rng=np.random.default_rng(111)
class Store:
 def __init__(self):
  self.values=words(rng.normal(size=(10,40)));self.rowsRead=0;self.rowBytesRead=0
 def gather(self,ids):
  self.rowsRead+=len(ids);self.rowBytesRead+=len(ids)*80;return self.values[ids]
class Prefix:pass
p=Prefix();p.store=Store();p.m={'headDim':4,'hiddenSize':8,'qWidth':16,'kvWidth':8,'tokenPolicy':{'padID':9}}
angle=np.arange(6)[:,None]*np.array([.1,.2,.1,.2])[None,:];p.rope=words(np.stack([np.cos(angle),np.sin(angle)]))
ids=np.arange(8);a,r=fit(p,ids,3,12,1909,3);old,receipt=old_fit(p,ids,3,12,1909,3)
assert np.array_equal(a['landmarkIdentities'][0],old['landmarkIdentities'])
assert np.allclose(a['landmarks'][0],old['landmarks'],rtol=0,atol=1e-14)
assert np.allclose(a['metricRoots'][0],old['metricRoot'],rtol=0,atol=1e-14)
for group in range(2):
 Y=floats(rotate(p.store.values[ids,24+group*4:28+group*4],p.rope)).astype(float).reshape(-1,4)/4**.25
 C=np.cov(Y,rowvar=False,bias=True)
 assert np.allclose(C,a['covariances'][group],rtol=2e-14,atol=2e-14)
 assert np.allclose(Y.mean(0),a['keyCenters'][group],rtol=2e-14,atol=2e-14)
for head in range(4):
 assert np.array_equal(head_asset(a,p.m,head)['metricRoot'],a['metricRoots'][head//2])
assert not np.array_equal(a['landmarks'][0],a['landmarks'][1])
assert r['sourceCovarianceMACs']==2*8*6*4*4
''')

    def test_shared_consumer_and_signed_cancellation_do_not_hide_head_errors(self):
        self.code('''import numpy as np
from attention_bank_study import linear_head_effects,effect_scores,regions
from transport_consumer import consume
from encoder_atoms import words,floats
# Two heads have the same writer direction but oppose one another.
pred=np.tile([2.,0.,-1.,0.],(3,1));exact=np.zeros_like(pred)
W=np.array([[1.,0.,1.,0.],[0.,1.,0.,1.]])
r=linear_head_effects(pred,exact,W,regions([1,1,0]),2,np.ones(3,dtype=bool))
c=r['cancellation'][0];assert c['sumHeadEnergyOverTotal']==5 and c['crossHeadSignedEnergy']<0
assert r['linearSumMaxAbsClosure']==0
heads=[x for x in r['heads'] if x['region']=='all'];assert heads[1]['signedErrorDotTotal']<0
e=words(np.ones((3,2)));gamma=words(np.ones(2));weights=[np.eye(2),np.eye(2),np.eye(2)]
out=consume(e,np.zeros((3,2)),pred,W,gamma,1e-6,np.arange(3),weights)
out0=consume(e,np.zeros((3,2)),np.zeros_like(pred),W,gamma,1e-6,np.arange(3),weights)
assert np.any(out['state']!=out0['state'])
target=np.full((3,2),10000.)+.001;zero=target-.001
s=effect_scores(zero,target,zero,np.ones(3,dtype=bool),np.array([1,0,1],dtype=bool))
assert s['attentionEffect']['relativeRms']==1 and s['endpoint']['relativeRms']<1e-6 and s['invalidQueries']==1
''')


if __name__=='__main__':unittest.main()
