"""Small common-parent instruments for identity, masking, multiplicity and signs."""
import subprocess
import sys
from pathlib import Path
import unittest


class CommonBankMechanics(unittest.TestCase):
    def code(self,source):
        p=subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_common_identity_and_exact_causal_restriction(self):
        self.code('''import numpy as np
from context_common_bank import bank,probability,restrict,stream
from context_statistics_study import exact_head
from encoder_atoms import words
rng=np.random.default_rng(801);table={};packets={}
for name,ids in [('a',[2,3,0,0]),('b',[2,4,0,0])]:
 p={'ids':np.asarray(ids),'positions':np.arange(4),'mask':np.array([1,1,0,0])}
 for key in ['q','k','v']:
  rows=[]
  for i,t in enumerate(ids):
   ident=(key,t,i)
   if ident not in table:table[ident]=words(rng.normal(size=4))
   rows.append(table[ident])
  p[key]=np.stack(rows)
 packets[name]=p
atoms,maps,dup=bank(packets)
assert len(atoms['q'])==5 and len(atoms['k'])==3 and dup=={'query':3,'key':1}
P,_=probability(atoms['q'],atoms['k']);U,S,Vt=np.linalg.svd(P,full_matrices=False)
A=U*np.sqrt(S);B=Vt.T*np.sqrt(S)
for name,p in maps.items():
 h,_=restrict(P,p);candidate,_=stream(A,B,p)
 assert np.allclose(h,exact_head(packets[name])['raw'],rtol=2e-14,atol=2e-14)
 assert np.allclose(candidate['raw'],h,rtol=2e-14,atol=2e-14)
packets['b']['v'][0,0]^=np.uint16(1)
try:bank(packets)
except ValueError as e:assert 'Duplicate key/value' in str(e)
else:raise AssertionError('Conflicting identity accepted')
''')

    def test_signed_duplicate_occurrences_zero_read_and_resume(self):
        self.code('''import numpy as np
from context_common_bank import stream,restrict
from encoder_atoms import words
A=np.array([[1.,.2],[.5,-1.],[0.,0.]])
B=np.array([[1.,-.2],[-3.,.5]])
p={'qmap':np.array([0,1,2,0]),'kmap':np.array([0,0,1,-1]),'mask':np.array([1,1,1,0]),'positions':np.arange(4),
   'v':words(np.array([[1.,2.],[1.,2.],[-1.,3.],[0.,0.]]))}
o,s=stream(A,B,p);h,rows=restrict(A@B.T,p)
assert np.array_equal(o['valid'],np.array([1,1,0,1],dtype=bool))
assert np.isnan(o['raw'][2]).all() and not o['rounded_valid'][2]
assert np.allclose(o['raw'][o['valid']],h[o['valid']]) and rows[-1]['nonpositiveDenominator']
assert rows[-1]['negativeEntries'] and rows[-1]['cancellationRatio']<1
assert s['count']==3 and np.allclose(s['z'],2*B[0]+B[1])
first,cut=stream(A,B,{k:v[:2] for k,v in p.items()})
second,end=stream(A,B,{k:v[2:] for k,v in p.items()},state=cut)
assert np.array_equal(end['z'],s['z']) and np.array_equal(end['M'],s['M'])
assert np.allclose(np.concatenate([first['raw'],second['raw']]),o['raw'],equal_nan=True)
try:stream(A,B,p,state=cut)
except ValueError as e:assert 'clock' in str(e)
else:raise AssertionError('Wrong prefix accepted')
''')


if __name__=='__main__':unittest.main()
