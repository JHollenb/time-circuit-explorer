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

    def test_diagonal_gauge_moment_optimum_and_zero_coordinates(self):
        self.run_code('''import numpy as np
from context_diagonal_diagnostics import diagonal
from context_key_diagnostics import softmax
rng=np.random.default_rng(605);x=rng.normal(size=(9,5));y=rng.normal(size=(11,5))*np.array([.01,10.,3.,.4,2.])
a=x.mean(0);b=y.mean(0);u=x-a;w=y-b
D,g=diagonal(u,w,1e-12)
assert not g['flooredQueryCoordinates'] and not g['flooredKeyCoordinates']
A=(u*u).mean(0);B=(w*w).mean(0)
assert np.allclose(D*D*A,B/(D*D))
assert g['operandNormObjectiveAfter']<=g['operandNormObjectiveBefore']
for dose in [.3,.7,1.3,3.]:
 z=D*dose
 assert np.sum(z*z*A+B/(z*z))>=g['operandNormObjectiveAfter']-1e-12
for q in x:
 assert np.allclose(softmax(y@q),softmax((w/D)@((q-a)*D)+y@a),atol=1e-14)
x[:,-1]=0.;D,g=diagonal(x,y,1e-12)
assert g['flooredQueryCoordinates']==[4] and np.isfinite(D).all()
D,g=diagonal(np.zeros_like(x),y,1e-12)
assert g['degenerateEntireSide'] and np.array_equal(D,np.ones(5))
''')

    def test_source_moments_include_between_component_variance(self):
        self.run_code('''import numpy as np
from context_source_geometry import moments,merge,mixture
rng=np.random.default_rng(606);x=rng.normal(size=(21,4))+1e6;y=rng.normal(size=(21,4))-1e6
n,a,M=moments(x);m,b,N=moments(y)
total,mean,M2=merge(moments(x[:7]),moments(x[7:]))
roundoff=np.finfo(float).eps*np.max(np.abs(x))*n*8
assert total==n and np.allclose(mean,a,rtol=0,atol=roundoff) and np.allclose(M2,M,rtol=0,atol=roundoff)
mu,var=mixture(a,M/n,b,N/m,.5);_,expected,expectedM=moments(np.concatenate([x,y]))
assert np.allclose(mu,expected,atol=1e-9) and np.allclose(var,expectedM/(n+m),rtol=1e-14)
assert np.all(var>1e11) and np.all(.5*M/n+.5*N/m<2)
''')

    def test_source_frozen_stream_kernel_gauge_and_state_resume(self):
        self.run_code('''import numpy as np
from context_conditioned import stream
from encoder_atoms import words,floats
rng=np.random.default_rng(607);n=11;d=4;rank=7
p={k:words(rng.normal(size=(n,d))) for k in ['q','k','v']};p.update(mask=np.array([0,1,1,0,1,1,1,0,0,0,0]),positions=np.arange(n))
omega=rng.normal(size=(rank,d));g={'a':rng.normal(size=d),'b':rng.normal(size=d),'D':np.exp(rng.normal(size=d))}
o,s=stream(p,omega,g);q,k,v=[floats(p[key]).astype(float) for key in ['q','k','v']];x=q/d**.25;y=k/d**.25
u=(x-g['a'])*g['D'];w=(y-g['b'])/g['D'];beta=y@g['a']
phi=np.exp(u@omega.T-.5*(u*u).sum(1)[:,None])/np.sqrt(rank)
psi=np.exp(w@omega.T-.5*(w*w).sum(1)[:,None]+beta[:,None])/np.sqrt(rank)
kernel=phi@psi.T;kernel*=np.tril(np.ones((n,n)))*p['mask'][None,:];den=kernel.sum(1)
h=np.divide(kernel@v,den[:,None],out=np.zeros_like(v),where=den[:,None]>0)
assert np.allclose(h,o['raw'],rtol=2e-13,atol=2e-13)
assert np.allclose(np.log(den[1:]),o['logDenominatorGauge'][1:])
assert np.allclose(o['logDenominator'][1:],np.log(den[1:])+x[1:]@g['b']-g['a']@g['b'])
first,cut=stream({k:z[:5] for k,z in p.items()},omega,g)
second,end=stream({k:z[5:] for k,z in p.items()},omega,g,state=cut)
assert np.allclose(np.concatenate([first['raw'],second['raw']]),o['raw'],rtol=2e-13,atol=2e-13)
assert end['count']==s['count'] and end['steps']==n and o['empty'][0]
assert o['hullViolation'].max()<1e-13
''')


if __name__=='__main__':unittest.main()
