"""Independent probability/moment checks and candidate-only seal mechanics."""
import subprocess
import sys
from pathlib import Path
import unittest


class PositiveTailMechanics(unittest.TestCase):
    def code(self,source):
        p=subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_independent_positive_probability_and_moment_form(self):
        self.code('''import numpy as np
from positive_tail import positive_read
rng=np.random.default_rng(1801);U=rng.normal(size=(11,4));V=rng.normal(size=(11,3));ell=rng.normal(size=4);delta=rng.normal(size=4);s=U@ell;selected=np.array([7,2,0,8]);tail=np.ones(11,dtype=bool);tail[selected]=False
raw,z=positive_read(s,U,V,selected,delta)
p=np.exp(s-s.max());p/=p.sum();anchor=p[tail]/p[tail].sum();mu=anchor@U[tail];t=(U[tail]-mu)@delta;r2=1+t+.5*t*t;k2=anchor@r2;h2=(anchor*r2)@V[tail]/k2
zs=(p[selected]*np.exp(U[selected]@delta)).sum();hs=(p[selected]*np.exp(U[selected]@delta))@V[selected]/zs;zt=p[tail].sum()*np.exp(mu@delta)*k2
assert np.allclose(raw,(zs*hs+zt*h2)/(zs+zt),rtol=2e-14,atol=2e-14)
h0=anchor@V[tail];mt=anchor@t;second=anchor@(t*t);c1=(anchor*(t-mt))@V[tail];c2=(anchor*(t*t-second))@V[tail]
assert np.allclose(h2,h0+(c1+.5*c2)/k2,rtol=2e-14,atol=2e-14)
assert np.all(raw>=V.min(0)-1e-14) and np.all(raw<=V.max(0)+1e-14)
assert np.isclose(z['logKappa2'],np.log(k2)) and np.isclose(z['tailProbability'].sum(),1)
''')

    def test_empty_tail_extreme_log_weights_and_polynomial_limits(self):
        self.code('''import numpy as np
from positive_tail import positive_read
U=np.array([[0.],[1.],[-1.]]);V=np.array([[1.,2.],[-1.,0.],[0.,1.]])
raw,z=positive_read(np.array([1000.,999.,-1000.]),U,V,np.array([2,0,1]),np.array([.5]));assert not z['hasTail'] and z['tailWeight']==0 and np.isfinite(raw).all()
raw,z=positive_read(np.array([0.,0.,0.]),U,V,np.array([0]),np.array([1000.]));assert np.isfinite(raw).all() and np.isfinite(z['logKappa2']) and np.all(z['tailProbability']>0)
r=lambda t:1+t+.5*t*t
assert r(-3)>r(-2) and np.exp(-3)<np.exp(-2) and r(2)!=r(1)*r(1)
# The canonical logarithmic polynomial remains finite without squaring 1e200.
raw,z=positive_read(np.zeros(3),U,V,np.array([0]),np.array([1e200]));assert np.isfinite(raw).all() and z['logKappa2']>900
''')

    def test_postseal_diagnostics_telescope_and_centering(self):
        self.code('''import numpy as np
from positive_tail import head
from positive_tail_diagnostics import diagnose_head
from encoder_atoms import words,floats
from context_transport import nearest
rng=np.random.default_rng(1802);n=9;d=4;p={key:words(rng.normal(size=(n,d))) for key in ['q','k','v']};p.update(mask=np.array([0,1,1,1,0,1,1,1,1]),positions=np.arange(n));asset={'landmarks':rng.normal(size=(3,d)),'metricRoot':np.eye(d),'keyCenter':rng.normal(size=d)}
support=np.full((n,3),-1,dtype=int)
for i in range(n):
 kept=np.flatnonzero(p['mask'][:i+1])[-3:][::-1];support[i,:len(kept)]=kept
frozen={'supportPositions':support,'nearestIndex':nearest(floats(p['q']).astype(float)/d**.25,asset)[0]}
raw,_=head(p,asset,frozen,3);banks,z=diagnose_head(p,asset,frozen,3);assert np.array_equal(raw,banks['quadratic'])
for m in ['quadratic','00']:assert np.max(np.abs(banks[m]-banks['11']-z[m+'__mass']-z[m+'__value']-z[m+'__controlArithmetic']))<1e-15
valid=z['hasTail'].astype(bool);assert np.isnan(z['meanT'][~valid]).all();assert np.max(np.abs(z['tailValueMomentClosure'][valid]))<1e-14
assert np.allclose(z['gaussianLogRemainder'][valid],z['centeredLogKappa'][valid]-z['meanT'][valid]-.5*z['varianceT'][valid])
for k in ['anchorMassBelowMinus1','exactMassBelowMinus1','quadraticMassBelowMinus1']:assert np.all((z[k][valid]>=0)&(z[k][valid]<=1+1e-14))
''')

    def test_seal_excludes_exact_diagnostics_and_control_outputs(self):
        self.code('''import numpy as np,tempfile
from pathlib import Path
from positive_tail import serve
from encoder_atoms import words,floats,read,save,sha
from context_transport import nearest
import context_statistics_study
def forbidden(*a,**k):raise AssertionError('Exact reference before seal')
context_statistics_study.exact_head=forbidden
with tempfile.TemporaryDirectory() as folder:
 root=Path(folder);rng=np.random.default_rng(1803);n=8;d=4;mask=np.array([0,1,1,0,1,1,1,1]);q=words(rng.normal(size=(n,8)));landmarks=rng.normal(size=(2,3,d))
 asset=root/'asset.npz';np.savez(asset,landmarks=landmarks,metricRoots=np.eye(d)[None],keyCenters=np.zeros((1,d)))
 port=root/'port.npz';np.savez(port,ids=np.arange(n),mask=mask,positions=np.arange(n),qpos=q,kpos=words(rng.normal(size=(n,d))),v=words(rng.normal(size=(n,d))))
 support=np.full((n,2,4),-1,dtype=int)
 for i in range(n):
  kept=np.flatnonzero(mask[:i+1])[-4:][::-1];support[i,:,:len(kept)]=kept
 routes=np.stack([nearest(floats(q[:,h*d:(h+1)*d]).astype(float)/d**.25,{'landmarks':landmarks[h],'metricRoot':np.eye(d),'keyCenter':np.zeros(d)})[0] for h in range(2)],axis=1)
 frozen=root/'frozen.npz';np.savez(frozen,supportPositions=support,nearestIndex=routes,raw=np.full((n,8),np.nan),rounded=np.full((n,8),65535,dtype=np.uint16))
 spec=root/'spec.json';save(spec,{'asset':{'path':str(asset),'sha256':sha(asset)},'ports':[{'id':'toy','path':str(port),'sha256':sha(port)}],'frozenSupports':[{'id':'toy','path':str(frozen),'sha256':sha(frozen)}],'geometry':{'headDim':d,'qWidth':8,'kvWidth':d},'budget':4})
 serve(spec,root/'out');seal=read(root/'out/prediction-seal.json');assert len(seal['rows'])==1 and seal['exactHeadReferenceEvaluations']==0
 with np.load(seal['rows'][0]['path']) as f:assert np.isfinite(f['raw']).all() and np.array_equal(f['rounded'],words(f['raw'])) and np.array_equal(f['supportPositions'],support)
''')

    def test_paired_decomposition_keeps_decoded_signed_vectors(self):
        self.code('''import numpy as np
from positive_tail_diagnostics import METHODS,TERMS,decompositions
rng=np.random.default_rng(1804);banks={m:{k:rng.normal(size=(3,8)) for k in ['raw','rounded']} for m in METHODS};fields={}
for m in ['quadratic','00']:
 a=rng.normal(size=(3,8));b=rng.normal(size=(3,8));fields.update({m+'__mass':a,m+'__value':b,m+'__controlArithmetic':banks[m]['raw']-banks['11']['raw']-a-b})
rows=decompositions({'kind':'paired'},banks,fields,[('all',np.ones(3,dtype=bool))],4,2)
assert len(rows)==6
for row in rows:assert row['vectorClosureMaxAbs']<2e-15 and abs(row['energyClosure'])<1e-13
''')


if __name__=='__main__':unittest.main()
