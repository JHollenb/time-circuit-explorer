"""Independent exact identities for retrospective exponential attribution."""
import subprocess
import sys
from pathlib import Path
import unittest


class RemainderMechanics(unittest.TestCase):
    def code(self,source):
        p=subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_exact_remainder_and_rounding_telescope_with_masks(self):
        self.code('''import numpy as np
from attention_remainder import decompose,attribution
from context_statistics_study import exact_head
from encoder_atoms import words,floats
rng=np.random.default_rng(112);p={k:words(rng.normal(size=(6,4))) for k in ['q','k','v']};p['mask']=np.array([0,1,1,0,1,0])
a={'landmarks':rng.normal(size=(2,4)),'keyCenter':np.array([2.,-3.,4.,1.])};idx=np.arange(6)%2
r=decompose(p,a,idx);exact=exact_head(p)['raw'];forecast=r['H']+r['A']
assert np.allclose(r['exact'],exact,rtol=2e-14,atol=2e-14)
assert np.allclose(forecast-exact,r['P']+r['V'],rtol=2e-14,atol=2e-14)
Q=(floats(words(forecast))-forecast)-(floats(words(exact))-exact);E=floats(words(forecast)).astype(float)-floats(words(exact)).astype(float)
s=attribution(E,r['P'],r['V'],Q,floats(words(exact)).astype(float),np.ones(6,dtype=bool))
assert s['telescopeMaxAbsClosure']<1e-13
assert r['normalizedRemainderCoefficients'].shape==(6,3)
assert r['minimumRemainderCoefficient'].min()>-1e-14
# Stable tilt must retain a key suppressed in the anchor distribution.
p={'q':words(np.array([[-100.,0.],[-100.,0.]])),'k':words(np.array([[-10.,0.],[10.,0.]])),
 'v':words(np.array([[1.,0.],[0.,1.]])),'mask':np.ones(2)}
r=decompose(p,{'landmarks':np.array([[100.,0.]]),'keyCenter':np.zeros(2)},np.zeros(2,dtype=int))
assert np.allclose(r['exact'],exact_head(p)['raw'],atol=1e-13) and np.isfinite(r['V']).all()
''')

    def test_one_address_key_payload_identity_excludes_changed_queries(self):
        self.code('''import numpy as np
from attention_remainder import one_key_change
from context_statistics_study import exact_head
from encoder_atoms import words
rng=np.random.default_rng(113);a={k:words(rng.normal(size=(7,4))) for k in ['q','k','v']};a['mask']=np.array([1,1,1,1,0,0,0])
b={k:v.copy() for k,v in a.items()};b['k'][2]=words([3.,-1.,2.,4.]);b['v'][2]=words([1.,4.,-2.,3.]);b['q'][2]=words([2.,1.,3.,0.])
r=one_key_change(a,b,2);delta=exact_head(b)['raw']-exact_head(a)['raw'];use=r['valid']
assert np.array_equal(use,[0,0,0,1,1,1,1]) and r['applicable']
assert np.allclose((r['key']+r['value'])[use],delta[use],rtol=2e-14,atol=2e-14)
b['k'][1]=words([7.,0.,0.,0.]);assert not one_key_change(a,b,2)['applicable']
''')


if __name__=='__main__':unittest.main()
