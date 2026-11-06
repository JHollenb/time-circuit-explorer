"""Tiny synthetic organisms: streaming closure, clock, mask and finite rescaling."""
import subprocess
import sys
from pathlib import Path
import unittest


class ContextStatisticsMechanics(unittest.TestCase):
    def run_code(self,code):
        r=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_feature_kernel_stream_and_real_state_resume(self):
        self.run_code('''import numpy as np
from context_statistics import stream,words,floats,state_bytes
rng=np.random.default_rng(601);d=4;rank=7;n=11
p={k:words(rng.normal(size=(n,d))) for k in ['q','k','v']};p.update(mask=np.array([0,1,1,0,1,1,0,1,0,0,0]),positions=np.arange(n))
omega=rng.normal(size=(rank,d));out,state=stream(p,omega)
q,k,v=[floats(p[n]).astype(float) for n in ['q','k','v']];q/=d**.25;k/=d**.25
phi=np.exp(q@omega.T-.5*(q*q).sum(1)[:,None])
psi=np.exp(k@omega.T-.5*(k*k).sum(1)[:,None])
kernel=phi@psi.T;kernel*=np.tril(np.ones((n,n)))*p['mask'][None,:]
total=kernel.sum(1);expected=np.divide(kernel@v,total[:,None],out=np.zeros((n,d)),where=total[:,None]>0)
assert np.allclose(out['raw'],expected,rtol=2e-13,atol=2e-13)
assert out['empty'][0] and not out['empty'][1] and out['hullViolation'].max()<1e-14
a={k:v[:5] for k,v in p.items()};b={k:v[5:] for k,v in p.items()};first,cut=stream(a,omega);second,end=stream(b,omega,state=cut)
assert np.array_equal(np.concatenate([first['raw'],second['raw']]),out['raw'])
for key in ['maximum','z','M']:assert np.array_equal(end[key],state[key])
assert state_bytes(d,rank)==rank*(d+2)*8+16
try:stream(a,omega,state=cut)
except ValueError as e:assert 'clock differs' in str(e)
else:raise AssertionError('Wrong state clock admitted')
''')

    def test_large_features_and_padding_do_not_break_amplitude(self):
        self.run_code('''import numpy as np
from context_statistics import stream,words
rng=np.random.default_rng(602);n=13;d=8
p={k:words(rng.normal(size=(n,d))*100) for k in ['q','k','v']};p.update(mask=np.array([0,0,1,1,1]+[0]*8),positions=np.arange(n))
o,s=stream(p,rng.normal(size=(16,d))*50)
assert np.isfinite(o['raw']).all() and o['hullViolation'].max()<1e-10
assert o['empty'][:2].all() and not o['empty'][2:].any()
uniform,_=stream(p,None)
assert np.array_equal(uniform['raw'][4],uniform['raw'][-1])
assert s['count']==3 and s['steps']==n
''')

    def test_selected_head_matches_full_lexical_position_program(self):
        self.run_code('''import numpy as np
from types import SimpleNamespace
from context_statistics import lexical_head,words,floats,bf
rng=np.random.default_rng(603);n=5;d=4;heads=4;kv=2;hidden=7
atoms=words(rng.normal(size=(n,hidden+(heads+kv*2)*d)));rope=words(rng.uniform(-1,1,size=(2,n,d)))
m={'headDim':d,'qWidth':heads*d,'kvWidth':kv*d,'hiddenSize':hidden}
prefix=SimpleNamespace(m=m,rope=rope,tokenize=lambda prompt:('text',np.arange(n),np.array([1,1,1,0,0])),store=SimpleNamespace(gather=lambda ids:atoms))
for head in range(heads):
 p,_=lexical_head(prefix,'raw',head);g=head//2
 def rot(x):
  x=floats(x);turn=np.concatenate([-x[:,d//2:],x[:,:d//2]],axis=1)
  return words(bf(x*floats(rope[0]))+bf(turn*floats(rope[1])))
 assert np.array_equal(p['q'],rot(atoms[:,hidden+head*d:hidden+(head+1)*d]))
 assert np.array_equal(p['k'],rot(atoms[:,hidden+heads*d+g*d:hidden+heads*d+(g+1)*d]))
 assert np.array_equal(p['v'],atoms[:,hidden+(heads+kv)*d+g*d:hidden+(heads+kv)*d+(g+1)*d])
''')

    def test_gauge_key_bias_and_diagnostic_match_stream(self):
        self.run_code('''import numpy as np
from context_statistics import stream,words,floats
from context_key_diagnostics import softmax,feature_probability
rng=np.random.default_rng(604);n=9;d=4
p={k:words(rng.normal(size=(n,d))) for k in ['q','k','v']};p.update(mask=np.ones(n,dtype=int),positions=np.arange(n))
omega=rng.normal(size=(7,d));out,_=stream(p,omega)
x,y,v=[floats(p[k]).astype(float) for k in ['q','k','v']];x/=d**.25;y/=d**.25
a=rng.normal(size=d);b=rng.normal(size=d);c=2.7
for i in range(n):
 keys=y[:i+1];bias=keys@a
 assert np.allclose(softmax(keys@x[i]),softmax(((keys-b)/c)@(c*(x[i]-a))+bias),atol=1e-14)
 weight=feature_probability(x[i],keys,omega,np.zeros(i+1))
 assert np.allclose(weight@v[:i+1],out['raw'][i],atol=1e-14)
''')


if __name__=='__main__':unittest.main()
