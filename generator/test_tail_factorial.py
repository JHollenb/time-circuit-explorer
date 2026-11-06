"""Independent four-arm mixtures, signed identities and frozen causal records."""
import subprocess
import sys
from pathlib import Path
import unittest


class TailFactorialMechanics(unittest.TestCase):
    def code(self,source):
        p=subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_four_positive_mixtures_and_signed_decomposition(self):
        self.code('''import numpy as np
from tail_factorial import factorial
rng=np.random.default_rng(1701);U=rng.normal(size=(9,4));V=rng.normal(size=(9,3));ell=rng.normal(size=4);delta=rng.normal(size=4);s=U@ell;selected=np.array([6,1,4]);tail=np.ones(9,dtype=bool);tail[selected]=False
out,d=factorial(s,U,V,selected,delta)
p=np.exp(s-s.max());p/=p.sum();current=p*np.exp(U@delta);zs=current[selected].sum();hs=current[selected]@V[selected]/zs
alpha=p[tail].sum();mu=p[tail]@U[tail]/alpha;v0=p[tail]@V[tail]/alpha;z0=alpha*np.exp(mu@delta);ze=current[tail].sum();ve=current[tail]@V[tail]/ze
for arm,z,v in [('00',z0,v0),('10',ze,v0),('01',z0,ve),('11',ze,ve)]:
 expected=(zs*hs+z*v)/(zs+z);assert np.allclose(out[arm],expected,rtol=2e-14,atol=2e-14)
 assert np.all(out[arm]>=V.min(0)-1e-14) and np.all(out[arm]<=V.max(0)+1e-14)
assert np.allclose(out['00']-out['11'],d['mass']+d['value'],atol=1e-14)
assert np.allclose(out['11']-out['10']-out['01']+out['00'],d['interaction'],atol=1e-14)
assert d['logKappa']>=-1e-14
''')

    def test_empty_tail_and_extreme_logs(self):
        self.code('''import numpy as np
from tail_factorial import factorial
U=np.array([[0.],[1.],[-1.]]);V=np.array([[1.,2.],[-1.,0.],[0.,1.]])
out,d=factorial(np.array([0.,-1.,1.]),U,V,np.array([2,0,1]),np.array([.5]))
assert not d['hasTail'] and np.isnan(d['logKappa']) and d['tailWeight']==d['exactTailWeight']==0
assert all(np.array_equal(out['00'],a) for a in out.values()) and not np.any(d['mass']) and not np.any(d['value'])
out,d=factorial(np.array([0.,0.,0.]),U,V,np.array([0]),np.array([1000.]))
assert all(np.isfinite(a).all() for a in out.values()) and d['logKappa']>900 and d['exactTailWeight']==1
assert d['telescopeClosure']<1e-12 and d['interactionClosure']<1e-12
''')

    def test_ordered_support_refuses_future_masked_and_duplicates(self):
        self.code('''import numpy as np
from tail_factorial import ordered_support
allowed=np.array([1,3,4,6]);assert np.array_equal(ordered_support(np.array([6,1,4]),allowed,3),[3,0,2])
for bad in [np.array([1,1,3]),np.array([1,2,3]),np.array([1,3,7])]:
 try:ordered_support(bad,allowed,3)
 except ValueError:pass
 else:raise AssertionError('Invalid frozen support accepted')
assert len(ordered_support(np.array([-1,-1]),np.array([],dtype=int),2))==0
''')

    def test_grouped_seal_ignores_control_output_members(self):
        self.code('''import numpy as np,tempfile
from pathlib import Path
from tail_factorial import serve
from encoder_atoms import words,floats,read,save,sha
from context_transport import nearest
with tempfile.TemporaryDirectory() as folder:
 root=Path(folder);rng=np.random.default_rng(1702);n=8;d=4;mask=np.array([0,1,1,0,1,1,1,1]);q=words(rng.normal(size=(n,8)))
 landmarks=rng.normal(size=(2,3,d));asset=root/'asset.npz';np.savez(asset,landmarks=landmarks,metricRoots=np.eye(d)[None],keyCenters=np.zeros((1,d)))
 port=root/'port.npz';np.savez(port,ids=np.arange(n),mask=mask,positions=np.arange(n),qpos=q,kpos=words(rng.normal(size=(n,d))),v=words(rng.normal(size=(n,d))))
 support=np.full((n,2,4),-1,dtype=int)
 for i in range(n):
  kept=np.flatnonzero(mask[:i+1])[-4:][::-1];support[i,:,:len(kept)]=kept
 routes=np.stack([nearest(floats(q[:,h*d:(h+1)*d]).astype(float)/d**.25,{'landmarks':landmarks[h],'metricRoot':np.eye(d),'keyCenter':np.zeros(d)})[0] for h in range(2)],axis=1)
 frozen=root/'frozen.npz';np.savez(frozen,supportPositions=support,nearestIndex=routes,raw=np.full((n,8),np.nan),rounded=np.full((n,8),65535,dtype=np.uint16))
 spec=root/'spec.json';save(spec,{'asset':{'path':str(asset),'sha256':sha(asset)},'ports':[{'id':'toy','path':str(port),'sha256':sha(port)}],'frozenSupports':[{'id':'toy','path':str(frozen),'sha256':sha(frozen)}],'geometry':{'headDim':d,'qWidth':8,'kvWidth':d},'budget':4})
 serve(spec,root/'out');seal=read(root/'out/prediction-seal.json');assert len(seal['rows'])==4
 for row in seal['rows']:
  with np.load(row['path']) as f:
   assert np.isfinite(f['raw']).all() and np.array_equal(f['rounded'],words(f['raw']))
   assert np.array_equal(f['supportPositions'],support) and np.array_equal(f['nearestIndex'],routes)
 with np.load(seal['diagnostics'][0]['path']) as f:
  assert f['mass'].shape==(n,8) and f['hasTail'].shape==(n,2) and not f['hasTail'][0].any()
''')


    def test_decoded_paired_consumer_interaction_keeps_signed_values(self):
        self.code('''import numpy as np,io
from tail_factorial_study import consumer_interactions
from attention_bank_study import STAGES
rng=np.random.default_rng(1703);results={arm:{stage:rng.normal(scale=.01,size=(3,4)) for stage in STAGES} for arm in ['00','10','01','11','zero']};stream=io.StringIO()
rows,count=consumer_interactions({'kind':'paired'},results,[('all',np.ones(3,dtype=bool))],stream,decoded=True)
assert count==15 and len(rows)==5
for row in rows:
 a={arm:results[arm][row['stage']] for arm in results};cross=a['11']-a['10']-a['01']+a['00'];assert np.isclose(row['interactionRms'],np.sqrt(np.mean(cross**2)))
 assert row['orderedConsumerError']['vectorClosureMaxAbs']<1e-16
''')


if __name__=='__main__':unittest.main()
