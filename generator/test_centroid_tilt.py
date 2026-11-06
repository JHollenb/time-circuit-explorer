"""Tiny independent checks for a matched tail-law comparison."""
import subprocess
import sys
from pathlib import Path
import unittest


class CentroidTiltMechanics(unittest.TestCase):
    def code(self,source):
        result=subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_direct_law_empty_full_tail_and_resume(self):
        self.code('''import numpy as np
from sparse_tilt import stream,core_bytes
from centroid_tilt import read_state,same_support
from encoder_atoms import words,floats
from context_statistics_study import exact_head
rng=np.random.default_rng(131);p={k:words(rng.normal(size=(9,4))) for k in ['q','k','v']};p.update(mask=np.array([0,1,1,0,1,1,0,1,1]),positions=np.arange(9))
a={'landmarks':rng.normal(size=(3,4)),'metricRoot':np.eye(4),'keyCenter':rng.normal(size=4)}
for budget in [0,1,4,9]:
 o,s=stream(p,a,budget,readout=read_state);unit,us=stream(p,a,budget)
 assert same_support(o['supportPositions'],unit['supportPositions']).all()
 for key in s:assert np.array_equal(s[key],us[key]),key
 for i in range(9):
  ids=np.flatnonzero(p['mask'][:i+1])
  if not len(ids):assert np.array_equal(o['raw'][i],np.zeros(4));continue
  U=floats(p['k'][ids]).astype(float)/4**.25-a['keyCenter'];V=floats(p['v'][ids]).astype(float);ell=a['landmarks'][o['nearestIndex'][i]];delta=floats(p['q'][i]).astype(float)/4**.25-ell
  score=U@ell;base=np.exp(score-score.max());base/=base.sum();selected=np.isin(ids,o['supportPositions'][i]);tail=~selected
  mass=list(base[selected]);key=list(U[selected]);value=list(V[selected])
  if tail.any():
   mass+=[base[tail].sum()];key+=[base[tail]@U[tail]/base[tail].sum()];value+=[base[tail]@V[tail]/base[tail].sum()]
  mass=np.array(mass)*np.exp(np.array(key)@delta);mass/=mass.sum();direct=mass@np.array(value)
  assert np.allclose(o['raw'][i],direct,rtol=3e-14,atol=3e-14)
 if budget==0:assert np.allclose(o['raw'],unit['raw'],atol=1e-14)
 if budget==9:assert np.allclose(o['raw'],exact_head(p)['raw'],atol=1e-14)
 _,cut=stream({k:v[:4] for k,v in p.items()},a,budget,readout=read_state)
 tail,end=stream({k:v[4:] for k,v in p.items()},a,budget,state=cut,readout=read_state)
 assert np.array_equal(tail['rounded'],o['rounded'][4:]) and np.array_equal(end['M'],s['M'])
 actual=sum(s[k].nbytes for k in ['maximum','z','M','Y','scores','positions','keys','values'])+16
 assert actual==core_bytes(4,3,budget)
''')

    def test_gauge_invariance_extremes_and_membership(self):
        self.code('''import numpy as np
from sparse_tilt import stream
from centroid_tilt import read_state,same_support
from encoder_atoms import words
rng=np.random.default_rng(132);p={k:words(rng.normal(size=(6,4))) for k in ['q','k','v']};p.update(mask=np.ones(6),positions=np.arange(6))
a={'landmarks':rng.normal(size=(3,4)),'metricRoot':np.eye(4),'keyCenter':np.zeros(4)}
o,s=stream(p,a,2,readout=read_state);shifted,_=stream(p,{**a,'keyCenter':np.array([4.,-2.,1.,3.])},2,readout=read_state)
assert np.allclose(o['raw'],shifted['raw'],atol=1e-13) and same_support(o['supportPositions'],shifted['supportPositions']).all()
assert np.array_equal(same_support(np.array([[3,-1,2],[1,3,2]]),np.array([[2,3,-1],[1,4,2]])),[True,False])
p={'q':words(np.array([[100.,0.],[-100.,0.],[100.,0.]])),'k':words(np.array([[-100.,0.],[100.,0.],[0.,0.]])),'v':words(np.array([[-1.,2.],[3.,-4.],[1.,0.]])),'mask':np.ones(3),'positions':np.arange(3)}
a={'landmarks':np.array([[100.,0.],[-100.,0.]]),'metricRoot':np.eye(2),'keyCenter':np.zeros(2)}
o,s=stream(p,a,1,readout=read_state);assert np.isfinite(o['raw']).all() and o['rounded_valid'].all() and o['hullViolation'].max()<1e-13
''')

    def test_frozen_oracle_preserves_indices_and_refuses_future_keys(self):
        self.code('''import numpy as np
from sparse_tilt import stream
from centroid_tilt import read_state
from centroid_tilt_study import frozen_reads,direct_centroid
from encoder_atoms import words,floats
rng=np.random.default_rng(134);p={k:words(rng.normal(size=(6,3))) for k in ['q','k','v']};p.update(mask=np.array([0,1,1,0,1,1]),positions=np.arange(6))
a={'landmarks':rng.normal(size=(2,3)),'metricRoot':np.eye(3),'keyCenter':np.zeros(3)}
unit,_=stream(p,a,2);centroid,_=stream(p,a,2,readout=read_state)
primary={k:unit[k] for k in ['supportPositions','nearestIndex']};primary.update(unit=unit['raw'],centroid=centroid['raw'])
frozen=np.full((6,2),-1,dtype=int)
for i in range(6):
 allowed=np.flatnonzero(p['mask'][:i+1]);chosen=allowed[-2:];frozen[i,:len(chosen)]=chosen
before=frozen.copy();out,closure=frozen_reads(p,a,primary,frozen)
assert np.array_equal(frozen,before) and max(v.max() for v in closure.values())<1e-14
ids=np.flatnonzero(p['mask']);U=floats(p['k'][ids]).astype(float)/3**.25;V=floats(p['v'][ids]).astype(float);ell=a['landmarks'][primary['nearestIndex'][-1]];delta=floats(p['q'][-1]).astype(float)/3**.25-ell
selected=np.flatnonzero(np.isin(ids,frozen[-1]));score=U@ell
assert np.allclose(out['centroid'][-1],direct_centroid(score,U,V,selected,delta),atol=1e-14)
bad=frozen.copy();bad[1,0]=5
try:frozen_reads(p,a,primary,bad)
except ValueError:pass
else:raise AssertionError('Future oracle key admitted')
''')

    def test_historical_collection_keeps_staged_source_integrity(self):
        self.code('''import tempfile
from pathlib import Path
from encoder_atoms import save,sha,identity
from publish_context_statistics import checked_collection
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);(root/'payload').mkdir();(root/'result').mkdir();worker=root/'payload/sparse_tilt.py';worker.write_text('historical source')
 request={'sourceHashes':{'sparse_tilt.py':sha(worker)}};request['requestSHA256']=identity(request);save(root/'request.json',request)
 report=root/'result/report.json';save(report,{'requestSHA256':request['requestSHA256']})
 custody={'required':True,**{k:{'sha256':'abc','size_bytes':3} for k in ['declared','sealed','executed']}}
 save(root/'scheduler-job.json',{'state':'succeeded','job_id':'old','payload_custody':custody,'config':{'requestSHA256':request['requestSHA256']},'admission':{'bound':True}})
 save(root/'collection.json',{'jobID':'old','custody':custody,'inventory':{'files':{'report.json':{'bytes':report.stat().st_size,'sha256':sha(report)}}}})
 assert checked_collection(root,current_workers=False)[0]['jobID']=='old'
 try:checked_collection(root)
 except ValueError:pass
 else:raise AssertionError('Fresh current-source check bypassed')
 worker.write_text('changed staged source')
 try:checked_collection(root,current_workers=False)
 except ValueError:pass
 else:raise AssertionError('Historical staged identity bypassed')
''')

    def test_anchor_derivative_missing_within_tail_covariance(self):
        self.code('''import numpy as np
from sparse_tilt import new_state,advance
from centroid_tilt import read_state
from encoder_atoms import words,floats
rng=np.random.default_rng(133);K=words(rng.normal(size=(7,3)));V=words(rng.normal(size=(7,3)));U=floats(K).astype(float)/3**.25;v=floats(V).astype(float);ell=np.array([.3,-.4,.2]);a={'landmarks':ell[None,:],'metricRoot':np.eye(3),'keyCenter':np.zeros(3)}
s=new_state(3,1,2)
for i in range(7):advance(s,K[i],V[i],i,a)
score=U@ell;p=np.exp(score-score.max());p/=p.sum();H=p@v;mu=p@U;exact=(v-H).T@(p[:,None]*(U-mu))
tail=~np.isin(np.arange(7),s['positions'][0]);pt=p[tail].sum();w=p[tail]/pt;mt=w@U[tail];ht=w@v[tail];missing=pt*(v[tail]-ht).T@(w[:,None]*(U[tail]-mt))
eps=1e-5;columns=[]
for dim in range(3):
 e=np.eye(3)[dim]*eps;columns.append((read_state(s,ell+e,0,a)[0]-read_state(s,ell-e,0,a)[0])/(2*eps))
assert np.allclose(exact-np.stack(columns,axis=1),missing,atol=1e-10,rtol=1e-9)
''')


if __name__=='__main__':unittest.main()
