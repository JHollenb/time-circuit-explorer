"""Independent tiny source-cell, priority, tail-weight and continuation checks."""
import subprocess
import sys
from pathlib import Path
import unittest


class SourceWitnessMechanics(unittest.TestCase):
    def code(self,source):
        p=subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_source_cells_ties_degeneracy_and_envelope(self):
        self.code('''import numpy as np
from source_witness import select_witnesses
from context_transport import nearest
x=np.array([[0.,0.],[1.,0.],[-1.,0.],[0.,1.],[0.,-1.],[9.,0.]])
ids=np.stack([np.arange(6),np.zeros(6,dtype=int)],axis=1);a={'landmarks':x[[0,5]],'metricRoot':np.eye(2),'keyCenter':np.zeros(2)}
extra,chosen,r,rows=select_witnesses(x,a,ids,ids[[0,5]])
assert np.array_equal(r,nearest(x,a)[0]);assert np.array_equal(chosen[0,:,0],[0,1,2,3]);assert np.array_equal(chosen[1,:,0],[5,5,5,5])
assert rows[1]['repeatedAnchorSlots']==3
rng=np.random.default_rng(141)
for j in range(2):
 witnesses=np.vstack([a['landmarks'][j],extra[j]])
 for _ in range(10):
  alpha=rng.dirichlet(np.ones(4));u=rng.normal(size=2);query=alpha@witnesses
  assert query@u<=(witnesses@u).max()+1e-14
''')

    def test_priority_retains_occurrences_but_tail_keeps_anchor_measure(self):
        self.code('''import numpy as np
from source_witness import stream
from encoder_atoms import words,floats
p={'q':words(np.tile([-2.,0.],(7,1))),'k':words(np.array([[1.,0.],[1.,0.],[-3.,0.],[2.,0.],[-2.,0.],[0.,0.],[2.,0.]])),
'v':words(np.arange(14).reshape(7,2)/3),'mask':np.array([1,1,1,1,0,1,1]),'positions':np.arange(7)}
a={'landmarks':np.array([[1.,0.],[2.,0.]]),'metricRoot':np.eye(2),'keyCenter':np.array([.5,0.]),'extraWitnesses':np.array([[[-2.,0.],[1.,0.],[1.,0.]],[[-1.,0.],[2.,0.],[2.,0.]]])}
o,s=stream(p,a,2);U=floats(p['k']).astype(float)/2**.25-a['keyCenter'];V=floats(p['v']).astype(float)
for i in range(7):
 allowed=np.flatnonzero(p['mask'][:i+1]);j=o['nearestIndex'][i];w=np.vstack([a['landmarks'][j],a['extraWitnesses'][j]]);priority=(U[allowed]@w.T).max(1)
 selected=allowed[np.lexsort((allowed,-priority))[:2]];assert set(o['supportPositions'][i][o['supportPositions'][i]>=0])==set(selected)
assert set(o['supportPositions'][1])=={0,1} and 2 in s['positions'][0]
ids=np.flatnonzero(p['mask'])
for j in range(2):
 tail=ids[~np.isin(ids,s['positions'][j])];score=U[tail]@a['landmarks'][j];w=np.exp(score-score.max());w/=w.sum()
 assert np.allclose(s['M'][j]/s['z'][j],w@V[tail],atol=1e-14)
 assert np.allclose(s['Y'][j]/s['z'][j],w@U[tail],atol=1e-14)
assert s['winningWitnessCounts'].sum()==2*len(ids) and s['tailUpdates']==2*(len(ids)-2)
''')

    def test_unchanged_centroid_read_and_value_independent_selection(self):
        self.code('''import numpy as np
from source_witness import stream
from centroid_tilt_study import direct_centroid
from encoder_atoms import words,floats
rng=np.random.default_rng(142);p={k:words(rng.normal(size=(8,4))) for k in ['q','k','v']};p.update(mask=np.array([0,1,1,0,1,1,1,1]),positions=np.arange(8))
a={'landmarks':rng.normal(size=(3,4)),'metricRoot':np.eye(4),'keyCenter':rng.normal(size=4),'extraWitnesses':rng.normal(size=(3,3,4))}
for budget in [0,2,8]:
 o,s=stream(p,a,budget)
 for i in range(8):
  ids=np.flatnonzero(p['mask'][:i+1])
  if not len(ids):assert np.array_equal(o['raw'][i],np.zeros(4));continue
  U=floats(p['k'][ids]).astype(float)/4**.25-a['keyCenter'];V=floats(p['v'][ids]).astype(float);ell=a['landmarks'][o['nearestIndex'][i]];delta=floats(p['q'][i]).astype(float)/4**.25-ell
  selected=np.flatnonzero(np.isin(ids,o['supportPositions'][i]));direct=direct_centroid(U@ell,U,V,selected,delta)
  assert np.allclose(o['raw'][i],direct,atol=3e-14,rtol=3e-14)
 changed={**p,'v':words(np.ones((8,4))*100)};again,_=stream(changed,a,budget)
 assert np.array_equal(again['supportPositions'],o['supportPositions'])
''')

    def test_resume_identity_monotone_priority_and_actual_core_bytes(self):
        self.code('''import numpy as np,tempfile
from pathlib import Path
from source_witness import stream
from sparse_tilt import core_bytes,stream as anchor_stream
from centroid_tilt import read_state
from encoder_atoms import words
rng=np.random.default_rng(143);p={k:words(rng.normal(size=(7,4))) for k in ['q','k','v']};p.update(mask=np.ones(7),positions=np.arange(7))
a={'landmarks':rng.normal(size=(3,4)),'metricRoot':np.eye(4),'keyCenter':np.zeros(4),'extraWitnesses':rng.normal(size=(3,3,4))}
o,s=stream(p,a,2);_,cut=stream({k:v[:3] for k,v in p.items()},a,2);tail={k:v[3:] for k,v in p.items()}
with tempfile.TemporaryDirectory() as tmp:
 path=Path(tmp)/'cut.npz';np.savez(path,**{k:np.asarray(v) for k,v in cut.items()})
 with np.load(path,allow_pickle=False) as f:restored={k:f[k] for k in f.files}
 again,end=stream(tail,a,2,state=restored)
 assert np.array_equal(again['rounded'],o['rounded'][3:])
 for key in s:assert np.array_equal(s[key],end[key]),key
for action in [lambda:stream(tail,{**a,'extraWitnesses':a['extraWitnesses']+1},2,state=cut),lambda:stream(p,a,2,state=cut),lambda:anchor_stream(tail,a,2,state=cut,readout=read_state)]:
 try:action()
 except ValueError:pass
 else:raise AssertionError('Wrong transition or source/clock resumed')
state=None;minimum=[]
for i in range(7):
 _,state=stream({k:v[i:i+1] for k,v in p.items()},a,2,state=state)
 if i>=1:minimum.append(state['priorities'].min(1))
assert np.all(np.diff(minimum,axis=0)>=0)
actual=sum(s[k].nbytes for k in ['maximum','z','M','Y','scores','positions','keys','values','priorities'])+16
assert actual==core_bytes(4,3,2)+3*2*8 and 'C' not in s
''')


if __name__=='__main__':unittest.main()
