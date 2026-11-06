"""Independent restricted-tilt and append-only-tail mechanics checks."""
import subprocess
import sys
from pathlib import Path
import unittest


class SparseTiltMechanics(unittest.TestCase):
    def code(self,source):
        p=subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_restricted_tilt_matches_direct_empty_partial_and_complete_support(self):
        self.code('''import numpy as np
from sparse_tilt import stream
from context_statistics_study import exact_head
from encoder_atoms import words,floats
rng=np.random.default_rng(121);p={k:words(rng.normal(size=(8,4))) for k in ['q','k','v']};p.update(mask=np.array([0,1,1,1,0,1,1,0]),positions=np.arange(8))
a={'landmarks':rng.normal(size=(3,4)),'metricRoot':np.eye(4),'keyCenter':np.array([2.,-1.,3.,4.])}
for budget in [0,1,2,8]:
 o,s=stream(p,a,budget)
 for i in range(8):
  ids=np.flatnonzero(p['mask'][:i+1])
  if not len(ids):assert np.array_equal(o['raw'][i],np.zeros(4));continue
  U=floats(p['k'][ids]).astype(float)/4**.25-a['keyCenter'];V=floats(p['v'][ids]).astype(float);ell=a['landmarks'][o['nearestIndex'][i]]
  score=U@ell;w=np.exp(score-score.max());w/=w.sum();mu=w@U;x=floats(p['q'][i]).astype(float)/4**.25
  chosen=np.lexsort((ids,-score))[:budget];actual=o['supportPositions'][i];assert set(actual[actual>=0])==set(ids[chosen])
  corrected=score.copy();corrected[chosen]+=(U[chosen]-mu)@(x-ell);w=np.exp(corrected-corrected.max());w/=w.sum()
  assert np.allclose(o['raw'][i],w@V,rtol=3e-14,atol=3e-14)
  assert np.all(o['raw'][i]>=V.min(0)-1e-13) and np.all(o['raw'][i]<=V.max(0)+1e-13)
 if budget==8:assert np.allclose(o['raw'],exact_head(p)['raw'],rtol=3e-14,atol=3e-14)
''')

    def test_ties_occurrences_and_tail_are_exact_without_subtraction(self):
        self.code('''import numpy as np
from sparse_tilt import stream,core_bytes
from encoder_atoms import words,floats
p={'q':words(np.ones((6,2))),'k':words(np.array([[1.,0.],[1.,0.],[0.,0.],[3.,0.],[2.,0.],[3.,0.]])),
 'v':words(np.arange(12).reshape(6,2)/4),'mask':np.ones(6),'positions':np.arange(6)}
a={'landmarks':np.array([[1.,0.],[2.,0.]]),'metricRoot':np.eye(2),'keyCenter':np.zeros(2)}
o,s=stream(p,a,2)
assert set(s['positions'][0])=={3,5} and set(o['supportPositions'][1])=={0,1}
U=floats(p['k']).astype(float)/2**.25;V=floats(p['v']).astype(float)
for anchor in range(2):
 tail=np.array([i for i in range(6) if i not in s['positions'][anchor]]);score=U[tail]@a['landmarks'][anchor];w=np.exp(score-score.max());w/=w.sum()
 assert np.allclose(s['M'][anchor]/s['z'][anchor],w@V[tail],atol=1e-14)
 assert np.allclose(s['Y'][anchor]/s['z'][anchor],w@U[tail],atol=1e-14)
assert s['tailUpdates']==2*(6-2) and s['count']==6
actual=sum(s[k].nbytes for k in ['maximum','z','M','Y','scores','positions','keys','values'])+16
assert actual==core_bytes(2,2,2) and 'C' not in s
''')

    def test_resume_clock_identity_and_extreme_positive_mixture(self):
        self.code('''import numpy as np
from sparse_tilt import stream
from encoder_atoms import words
rng=np.random.default_rng(122);p={k:words(rng.normal(size=(7,4))) for k in ['q','k','v']};p.update(mask=np.array([1,1,0,1,1,0,1]),positions=np.arange(7))
a={'landmarks':rng.normal(size=(3,4)),'metricRoot':np.eye(4),'keyCenter':np.zeros(4)}
o,s=stream(p,a,2);_,cut=stream({k:v[:3] for k,v in p.items()},a,2);tail,end=stream({k:v[3:] for k,v in p.items()},a,2,state=cut)
assert np.array_equal(tail['rounded'],o['rounded'][3:]) and np.array_equal(end['positions'],s['positions']) and np.array_equal(end['M'],s['M'])
import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as tmp:
 path=Path(tmp)/'state.npz';np.savez(path,**{k:np.asarray(v) for k,v in cut.items()})
 with np.load(path,allow_pickle=False) as stored:restored={k:stored[k] for k in stored.files}
 replay,_=stream({k:v[3:] for k,v in p.items()},a,2,state=restored)
 assert np.array_equal(replay['rounded'],o['rounded'][3:])
for args in [(p,a,2),({k:v[3:] for k,v in p.items()},{**a,'keyCenter':np.ones(4)},2),({k:v[3:] for k,v in p.items()},a,1)]:
 try:stream(*args,state=cut)
 except ValueError:pass
 else:raise AssertionError('Invalid resumed state accepted')
p={'q':words(np.array([[100.,0.],[-100.,0.],[100.,0.]])),'k':words(np.array([[-100.,0.],[100.,0.],[0.,0.]])),
 'v':words(np.array([[-1.,2.],[3.,-4.],[1.,0.]])),'mask':np.ones(3),'positions':np.arange(3)}
a={'landmarks':np.array([[100.,0.],[-100.,0.]]),'metricRoot':np.eye(2),'keyCenter':np.zeros(2)}
o,s=stream(p,a,1);assert np.isfinite(o['raw']).all() and o['rounded_valid'].all() and o['hullViolation'].max()<1e-13
''')

    def test_full_query_oracle_uses_scores_without_value_target_selection(self):
        self.code('''import numpy as np
from sparse_tilt import stream
from sparse_tilt_study import diagnose,log_abs_expm1
from encoder_atoms import words
t=np.array([-1000.,-1e-8,0.,1e-8,1000.]);logs=log_abs_expm1(t)
assert np.isfinite(logs[[0,1,3,4]]).all() and np.isneginf(logs[2]) and logs[-1]==1000. and logs[0]==0.
p={'q':words(np.tile([-3.,0.],(3,1))),'k':words(np.array([[1.,0.],[-3.,0.],[.1,0.]])),
 'v':words(np.array([[1.,0.],[0.,1.],[.5,.5]])),'mask':np.ones(3),'positions':np.arange(3)}
a={'landmarks':np.array([[1.,0.]]),'metricRoot':np.eye(2),'keyCenter':np.zeros(2)}
primary,_=stream(p,a,1);oracle=diagnose(p,a,primary,1)
assert primary['supportPositions'][-1,0]==0 and oracle['supportPositions'][-1,0]==1
assert oracle['restrictedLawMaxAbsClosure'].max()<1e-14
assert oracle['oracleOmittedAbsoluteTiltMass'][-1]<oracle['eligibleOmittedAbsoluteTiltMass'][-1]
p['v']=words(np.array([[100.,-50.],[-70.,30.],[20.,40.]]));again=diagnose(p,a,stream(p,a,1)[0],1)
assert np.array_equal(again['supportPositions'],oracle['supportPositions'])
''')


if __name__=='__main__':unittest.main()
