"""Independent exhaustive retrieval, numerical bounds and causal mechanics."""
import subprocess
import sys
from pathlib import Path
import unittest


class CertifiedRetrievalMechanics(unittest.TestCase):
    def code(self,source):
        p=subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_random_prefixes_match_complete_scores_and_order(self):
        self.code('''import numpy as np
from certified_retrieval import CausalIndex
rng=np.random.default_rng(1601)
for d,block in [(8,2),(16,4),(128,16)]:
 for repeat in range(5):
  order=rng.permutation(d);index=CausalIndex(24,d,order,block);keys=[];positions=[]
  for i in range(24):
   if rng.random()>.3:
    u=rng.normal(size=d);index.append(u,rng.normal(size=d),i);keys.append(u);positions.append(i)
   for _ in range(3):
    x=rng.normal(size=d);chosen,c=index.query(x,4)
    exact=np.lexsort((np.array(positions),-(np.stack(keys)@x)))[:4] if keys else np.array([],dtype=int)
    assert np.array_equal(chosen,exact),(d,i,c,chosen,exact)
    assert c['initialCandidateFullDotProducts']+c['progressiveScoreProducts']+c['wholePrefixScoreProducts']>=0
''')

    def test_real_pruning_and_ambiguous_ties_fall_back(self):
        self.code('''import numpy as np
from certified_retrieval import CausalIndex
d=16;index=CausalIndex(12,d,np.arange(d),4)
for i in range(12):
 u=np.zeros(d);u[0]=10-i;index.append(u,np.zeros(d),i)
x=np.zeros(d);x[0]=1;chosen,c=index.query(x)
assert np.array_equal(chosen,[0,1,2,3]) and c['certifiedQueries']==1 and c['prunedKeys']==8
assert c['initialCandidateFullDotProducts']+c['progressiveScoreProducts']<c['fullScanScoreProducts'] and c['wholePrefixCalls']==0
tie=CausalIndex(8,d,np.arange(d),4)
for i in range(8):tie.append(np.ones(d),np.zeros(d),2*i)
chosen,c=tie.query(np.ones(d));assert np.array_equal(chosen,[0,1,2,3]) and c['fallbackReason']==3 and c['wholePrefixScoreProducts']==8*d
near=CausalIndex(8,d,np.arange(d),4)
for i in range(8):
 u=np.ones(d);u[0]+=i*1e-14;near.append(u,np.zeros(d),i)
chosen,c=near.query(np.ones(d));assert c['fallbackReason']==3
assert np.array_equal(chosen,np.lexsort((np.arange(8),-(near.keys[:8]@np.ones(d))))[:4])
''')

    def test_reverse_norm_bound_retains_tiny_tail_and_safe_fallback(self):
        self.code('''import numpy as np,decimal
from collections import Counter
from certified_retrieval import CausalIndex,tail_norms
d=16;a=np.zeros(d);a[0]=1e100;a[-1]=1e-100;c=Counter();norms=tail_norms(a,np.arange(d),4,c,'test')
assert norms[1]>=1e-100 and norms[1]>0 and norms[-1]==0
decimal.getcontext().prec=100
for start in [0,4,8,12]:
 true=sum(decimal.Decimal(float(z))**2 for z in a[start:]).sqrt()
 assert decimal.Decimal(float(norms[start//4]))>=true
index=CausalIndex(7,d,np.arange(d),4)
for i in range(7):index.append(np.full(d,2.**-500 if i==5 else i),np.zeros(d),i)
chosen,c=index.query(np.ones(d));assert c['fallbackReason']==2
assert np.array_equal(chosen,np.lexsort((np.arange(7),-(index.keys[:7]@np.ones(d))))[:4])
try:index.append(np.zeros(d),np.zeros(d),6)
except ValueError:pass
else:raise AssertionError('Repeated causal address admitted')
''')

    def test_grouped_causal_banks_seal_before_reference(self):
        self.code('''import numpy as np,tempfile
from pathlib import Path
from certified_retrieval import serve
from encoder_atoms import words,read,save,sha
with tempfile.TemporaryDirectory() as folder:
 root=Path(folder);rng=np.random.default_rng(1602);d=4
 asset=root/'asset.npz';np.savez(asset,landmarks=rng.normal(size=(2,3,d)),metricRoots=np.eye(d)[None],keyCenters=np.zeros((1,d)),covariances=np.diag([1.,4.,3.,2.])[None])
 port=root/'port.npz';np.savez(port,ids=np.arange(9),mask=np.array([0,1,1,0,1,1,1,0,1]),positions=np.arange(9),qpos=words(rng.normal(size=(9,8))),kpos=words(rng.normal(size=(9,d))),v=words(rng.normal(size=(9,d))))
 spec=root/'spec.json';save(spec,{'asset':{'path':str(asset),'sha256':sha(asset)},'ports':[{'id':'toy','path':str(port),'sha256':sha(port)}],'geometry':{'headDim':d,'qWidth':8,'kvWidth':d},'budget':4,'block':2})
 for mode in ['certified','full']:serve(spec,root/mode,mode)
 a,b=[read(root/mode/'prediction-seal.json') for mode in ['certified','full']]
 assert a['coordinateOrders']==[[1,2,3,0]]
 with np.load(a['rows'][0]['path']) as x,np.load(b['rows'][0]['path']) as y:
  assert np.array_equal(x['supportPositions'],y['supportPositions']) and np.array_equal(x['rounded'],y['rounded']) and np.array_equal(x['raw'],y['raw'])
  assert np.all(x['supportPositions'][0]==-1) and not np.any(x['raw'][0])
 state=next(s for s in a['states'] if s['kind']=='final-index')
 with np.load(state['path']) as x:assert np.array_equal(x['positions'],[1,2,4,5,6,8]) and x['keys'].shape==(6,d)
''')


    def test_partial_upper_and_completed_intervals_enclose_old_scores(self):
        self.code('''import numpy as np
from collections import Counter
from certified_retrieval import tail_norms,up,down,product_up
rng=np.random.default_rng(1603);d=128;block=16;eps=2.**-53;eta=up(4*d*eps/(1-4*d*eps))
for _ in range(30):
 x=np.ldexp(rng.uniform(-1,1,d),rng.integers(-300,301,d));U=np.ldexp(rng.uniform(-1,1,(9,d)),rng.integers(-300,301,(9,d)));order=rng.permutation(d);scores=U@x
 qnorm=tail_norms(x,order,block,Counter(),'query')
 for j,u in enumerate(U):
  knorm=tail_norms(u,order,block,Counter(),'key');radius=product_up(2*eta,product_up(qnorm[0],knorm[0]));partial=0.
  for start in range(0,d,block):
   c=order[start:start+block];partial+=float(np.dot(x[c],u[c]));end=(start+block)//block
   upper=up(up(partial+product_up(qnorm[end],knorm[end]))+radius)
   assert upper>=scores[j]
  assert down(partial-radius)<=scores[j]<=up(partial+radius)
''')


if __name__=='__main__':unittest.main()
