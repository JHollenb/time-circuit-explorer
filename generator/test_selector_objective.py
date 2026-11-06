"""Independent ranking, centroid and signed-mass accounting mechanics."""
import subprocess
import sys
from pathlib import Path
import unittest


class SelectorObjectiveMechanics(unittest.TestCase):
    def code(self,source):
        p=subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_causal_rank_ties_and_direct_centroid(self):
        self.code('''import numpy as np
from selector_objective import predict,select
from encoder_atoms import words,floats
assert np.array_equal(select(np.array([2.,2.,1.]),np.array([5,1,3]),2),[1,0])
rng=np.random.default_rng(151);p={k:words(rng.normal(size=(7,4))) for k in ['q','k','v']};p.update(mask=np.array([0,1,1,0,1,1,1]),positions=np.arange(7));a={'landmarks':rng.normal(size=(3,4)),'metricRoot':np.eye(4),'keyCenter':rng.normal(size=4)}
result,work=predict(p,a,2)
for method,out in result.items():
 for i in range(7):
  ids=np.flatnonzero(p['mask'][:i+1])
  if not len(ids):assert np.array_equal(out['raw'][i],np.zeros(4));continue
  U=floats(p['k'][ids]).astype(float)/4**.25-a['keyCenter'];V=floats(p['v'][ids]).astype(float);x=floats(p['q'][i]).astype(float)/4**.25;ell=a['landmarks'][out['nearestIndex'][i]];s=U@ell;cur=U@x
  selected=np.lexsort((ids,-(cur if method=='current' else np.maximum(s,cur))))[:2]
  assert set(out['supportPositions'][i][out['supportPositions'][i]>=0])==set(ids[selected])
  w=np.exp(s-s.max());w/=w.sum();tail=~np.isin(np.arange(len(ids)),selected);m=list(w[selected]);u=list(U[selected]);v=list(V[selected])
  if tail.any():m+=[w[tail].sum()];u+=[w[tail]@U[tail]/w[tail].sum()];v+=[w[tail]@V[tail]/w[tail].sum()]
  m=np.array(m)*np.exp(np.array(u)@(x-ell));m/=m.sum()
  assert np.allclose(out['raw'][i],m@np.array(v),atol=2e-14)
assert work['scannedKeyOccurrences']==sum(p['mask'][:i+1].sum() for i in range(7))
''')

    def test_signed_mass_identity_and_zero_ratios(self):
        self.code('''import numpy as np
from selector_objective import tilt_measure,capture
rng=np.random.default_rng(152);U=rng.normal(size=(6,3));V=rng.normal(size=(6,2));ell=rng.normal(size=3);delta=rng.normal(size=3);s=U@ell
m=tilt_measure(s,U,V,delta);p=np.exp(s-s.max());p/=p.sum();H=p@V;mu=p@U;t=(U-mu)@delta;w=p*np.expm1(t);D=np.dot(p,np.exp(t));chosen=np.array([0,3])
c=capture(m,chosen,V)
for label,positive in [('positive',w>0),('negative',w<0)]:
 mass=np.abs(w[positive]).sum();kept=np.intersect1d(chosen,np.flatnonzero(positive));assert np.allclose(c[label+'CaptureFraction'],np.abs(w[kept]).sum()/mass,atol=1e-14)
 assert np.allclose(m[label+'MassOverD'],mass/D,atol=1e-14)
assert np.allclose(c['signedPartitionOverD'],w[chosen].sum()/D) and np.allclose(c['signedNumeratorOverD'],(w[chosen]/D)@V[chosen])
assert m['partitionClosure']<1e-14 and m['readClosure']<1e-14
zero=tilt_measure(s,U,V,np.zeros(3));z=capture(zero,chosen,V)
assert np.isnan(z['positiveCaptureFraction']) and np.isnan(z['negativeCaptureFraction']) and np.isnan(z['absoluteCaptureFraction'])
assert z['positiveCapturedMassOverD']==0 and z['negativeCapturedMassOverD']==0 and np.array_equal(z['signedNumeratorOverD'],np.zeros(2))
''')

    def test_log_capture_survives_extreme_partition_and_refuses_future_keys(self):
        self.code('''import numpy as np
from selector_objective import tilt_measure,capture,validate_support
U=np.array([[-1.],[0.],[1.]]);V=np.array([[1.,2.],[2.,-1.],[-1.,0.]]);s=np.zeros(3);m=tilt_measure(s,U,V,np.array([1000.]));c=capture(m,np.array([0,2]),V)
assert np.isfinite(m['coefficients']).all() and m['readClosure']<1e-12 and m['partitionClosure']<1e-12
assert c['positiveCaptureFraction']==1 and np.isclose(c['negativeCaptureFraction'],1) and m['negativeMassOverD']==0
assert np.isfinite(m['negativeLogMass'])
for support in [np.array([1,5]),np.array([1,1])]:
 try:validate_support(support,np.array([0,1,2]),2)
 except ValueError:pass
 else:raise AssertionError('Invalid fixed support accepted')
assert np.array_equal(validate_support(np.array([2,0]),np.array([0,1,2]),2),[0,2])
''')

    def test_grouped_head_seal_contains_complete_raw_and_rounded_banks(self):
        self.code('''import numpy as np,tempfile
from pathlib import Path
from encoder_atoms import words,read,save,sha
from selector_objective import serve
with tempfile.TemporaryDirectory() as folder:
 root=Path(folder);rng=np.random.default_rng(153)
 asset=root/'asset.npz';np.savez(asset,landmarks=rng.normal(size=(2,3,2)),metricRoots=np.eye(2)[None],keyCenters=np.zeros((1,2)))
 port=root/'port.npz';np.savez(port,ids=np.arange(6),mask=np.array([0,1,1,0,1,1]),positions=np.arange(6),qpos=words(rng.normal(size=(6,4))),kpos=words(rng.normal(size=(6,2))),v=words(rng.normal(size=(6,2))))
 spec=root/'spec.json';save(spec,{'asset':{'path':str(asset),'sha256':sha(asset)},'ports':[{'id':'toy','path':str(port),'sha256':sha(port)}],'geometry':{'headDim':2,'qWidth':4,'kvWidth':2},'budget':4})
 serve(spec,root/'out');seal=read(root/'out/prediction-seal.json')
 assert len(seal['rows'])==2 and seal['counts'][0]['scannedKeyOccurrencesSharedAcrossSelectors']==24
 for row in seal['rows']:
  assert sha(row['path'])==row['sha256']
  with np.load(row['path']) as f:
   assert f['raw'].shape==(6,4) and np.array_equal(words(f['raw']),f['rounded'])
   assert f['supportPositions'].shape==(6,2,4) and np.all(f['supportPositions'][0]==-1)
   assert f['valid'].all() and not np.any(f['raw'][0])
''')


if __name__=='__main__':unittest.main()
