"""Source score objective, independent positive law and actual prefix continuation."""
import subprocess
import sys
from pathlib import Path
import unittest


class RankOneTailMechanics(unittest.TestCase):
    def code(self,source):
        p=subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_source_product_objective_and_singular_support(self):
        self.code('''import numpy as np
from rank_one_tail import source_basis
C=np.diag([1.,4.,9.]);D=np.diag([4.,5.,1.]);a,b,r=source_basis(C,D)
assert np.allclose(a,[0,2,0]) and np.allclose(b,[0,.5,0]) and r['resolvedCapturedScoreEnergy']==20 and r['sourceProductScoreEnergy']==33
error=C-np.outer(a,b)@C-C@np.outer(b,a)+np.outer(a,b)@C@np.outer(b,a)
assert np.isclose(np.trace(error@D),13)
a,b,r=source_basis(np.diag([1.,0.,4.]),np.diag([1.,100.,2.]));assert r['resolvedRank']==2 and a[1]==b[1]==0 and np.isfinite(a).all() and np.isfinite(b).all()
''')

    def test_stream_against_independent_positive_probability(self):
        self.code(TOY+'''out,state=stream(packets,asset,a,b)
for i in range(n):
 allowed=np.flatnonzero(mask[:i+1]);U=k[allowed];V=v[allowed]
 if not len(allowed):assert not out['raw'][i].any();continue
 zz=np.array([b@u for u in U]);scores=np.stack([(asset['landmarks'].reshape(-1,d)@u).reshape(heads,rank) for u in U])
 for h in range(heads):
  r=int(out['nearestIndex'][i,h]);ell=asset['landmarks'][h,r];delta=x[h,i]-ell;chosen=select(U@x[h,i],allowed,4);s=scores[:,h,r];tmask=np.ones(len(U),dtype=bool);tmask[chosen]=False;cs=s[chosen]+U[chosen]@delta
  weights=np.exp(cs-cs.max());hs=weights@V[chosen]/weights.sum();logzs=cs.max()+np.log(weights.sum())
  if not tmask.any():expected=hs
  else:
   p=np.exp(s[tmask]-s[tmask].max());p/=p.sum();mu=p@U[tmask];zbar=p@zz[tmask];t=(a@delta)*(zz[tmask]-zbar);r2=1+t+.5*t*t;kappa=p@r2;ht=(p*r2)@V[tmask]/kappa;logzt=s[tmask].max()+np.log(np.exp(s[tmask]-s[tmask].max()).sum())+mu@delta+np.log(kappa);mix=1/(1+np.exp(logzs-logzt));expected=(1-mix)*hs+mix*ht
  assert np.allclose(out['raw'][i,h*d:(h+1)*d],expected,rtol=3e-12,atol=3e-12)
assert state['count']==mask.sum() and state['steps']==n and state['fallbackCalls']==0
assert np.array_equal(state['positions'][:state['count']],np.flatnonzero(mask))
''')

    def test_snapshot_resume_and_identity(self):
        self.code(TOY+'''import tempfile
from pathlib import Path
full,fs=stream(packets,asset,a,b);prefix=lambda p,start,stop:{k:v[start:stop] for k,v in p.items()}
left,ls=stream([prefix(p,0,6) for p in packets],asset,a,b)
with tempfile.TemporaryDirectory() as folder:
 path=Path(folder)/'state.npz';np.savez(path,**{k:np.asarray(v) for k,v in ls.items()})
 with np.load(path,allow_pickle=False) as f:loaded={k:f[k].item() if f[k].ndim==0 else f[k] for k in f.files}
 right,rs=stream([prefix(p,6,n) for p in packets],asset,a,b,state=loaded)
 assert np.array_equal(np.concatenate([left['raw'],right['raw']]),full['raw'])
 for key in fs:assert np.array_equal(np.asarray(rs[key]),np.asarray(fs[key]),equal_nan=True) if np.asarray(fs[key]).dtype.kind not in 'US' else rs[key]==fs[key]
 assert ls['steps']==6 and loaded['steps']==6
 try:stream([prefix(p,6,n) for p in packets],asset,a+1,b,state=loaded)
 except ValueError:pass
 else:raise AssertionError('Wrong basis accepted')
''')

    def test_cancelling_complement_has_counted_fallback(self):
        self.code('''import numpy as np
from rank_one_tail import stream
from encoder_atoms import words
d=2;n=5;p={'q':words(np.zeros((n,d))),'k':words(np.array([[0.,0.]]*4+[[-1000.,0.]])*d**.25),'v':words(np.arange(n*d).reshape(n,d)/10),'positions':np.arange(n),'mask':np.ones(n,dtype=int)}
asset={'landmarks':np.array([[[1.,0.]]]),'metricRoot':np.eye(d),'keyCenter':np.zeros(d)}
out,s=stream([p],asset,np.array([1.,0.]),np.array([1.,0.]));assert np.isfinite(out['raw']).all() and s['fallbackCalls']==1 and s['fallbackKeyVisits']==5 and out['fallbackReason'][-1,0]==2
assert out['selectedOnly'][:4].all() and not out['selectedOnly'][-1,0]
''')

    def test_source_only_compiler_and_candidate_seal(self):
        self.code('''import numpy as np,tempfile,hashlib
from pathlib import Path
from encoder_atoms import words,save,read,sha
from rank_one_tail import compile_asset,serve
from rank_one_tail_study import reference
with tempfile.TemporaryDirectory() as folder:
 root=Path(folder);store=root/'store';store.mkdir();d=4;head=2;width=24;rng=np.random.default_rng(1902);atoms=words(rng.normal(size=(3,width)));np.save(store/'atoms.npy',atoms);np.save(store/'ids.npy',np.arange(3,dtype='<u4'));np.save(store/'hashes.npy',np.stack([np.frombuffer(hashlib.sha256(x.tobytes()).digest(),dtype=np.uint8) for x in atoms]));rope=words(np.stack([np.ones((4,d)),np.zeros((4,d))]));np.save(store/'rope.npy',rope)
 rec=lambda name:{'path':name,'sha256':sha(store/name)}
 manifest={'vocabSize':3,'chunkRows':3,'hiddenSize':8,'qWidth':8,'kvWidth':4,'headDim':d,'coverage':rec('ids.npy'),'rowHashes':rec('hashes.npy'),'rope':rec('rope.npy'),'shards':[dict(rec('atoms.npy'),bytes=(store/'atoms.npy').stat().st_size,shape=[3,width])]};save(store/'manifest.json',manifest)
 asset=root/'asset.npz';np.savez(asset,landmarks=rng.normal(size=(2,3,d)),metricRoots=np.eye(d)[None],keyCenters=np.zeros((1,d)),covariances=np.eye(d)[None],candidateIdentities=np.array([[0,0],[1,1],[2,2],[0,3]]),candidateWeights=np.full(4,.25))
 spec=root/'compile.json';save(spec,{'asset':{'path':str(asset),'sha256':sha(asset)},'store':str(store),'manifestSHA256':sha(store/'manifest.json')});compile_asset(spec,root/'compiled');receipt=read(root/'compiled/compile-receipt.json');assert receipt['learnedAtomRowsRead']==3 and receipt['frameworkCalls']==0
 n=9;port=root/'port.npz';np.savez(port,ids=np.arange(n),mask=np.ones(n,dtype=int),positions=np.arange(n),qpos=words(rng.normal(size=(n,8))),kpos=words(rng.normal(size=(n,d))),v=words(rng.normal(size=(n,d))))
 spec=root/'serve.json';basis=root/'compiled/rank-one-basis.npz';save(spec,{'asset':{'path':str(asset),'sha256':sha(asset)},'basis':{'path':str(basis),'sha256':sha(basis)},'ports':[{'id':'toy','path':str(port),'sha256':sha(port)}],'geometry':{'headDim':d,'qWidth':8,'kvWidth':4}});serve(spec,root/'out');seal=read(root/'out/prediction-seal.json');assert len(seal['rows'])==1 and len(seal['states'])==1 and seal['exactHeadReferences']==0
 with np.load(seal['rows'][0]['path']) as f:pred={k:f[k] for k in f.files}
 with np.load(port) as f:operands={k:f[k] for k in f.files}
 with np.load(asset) as f:geometry={k:f[k] for k in f.files}
 with np.load(basis) as f:coordinates={k:f[k] for k in f.files}
 direct=reference(operands,{'headDim':d,'qWidth':8,'kvWidth':4},geometry,coordinates,pred)
 assert np.isfinite(pred['raw']).all() and np.array_equal(words(pred['raw']),pred['rounded']) and np.max(np.abs(pred['raw']-direct['raw']))<4e-12
 assert np.array_equal(pred['supportPositions'],direct['supportPositions']) and np.array_equal(pred['nearestIndex'],direct['nearestIndex'])
''')


TOY='''import numpy as np
from rank_one_tail import stream
from encoder_atoms import words,floats
from selector_objective import select
rng=np.random.default_rng(1901);n=12;d=4;heads=2;rank=3;mask=np.array([0,1,1,1,1,1,0,1,0,1,1,0]);keys=words(rng.normal(scale=.5,size=(n,d)));values=words(rng.normal(size=(n,d)));packets=[{'q':words(rng.normal(scale=.5,size=(n,d))),'k':keys,'v':values,'positions':np.arange(n),'mask':mask} for h in range(heads)];asset={'landmarks':rng.normal(scale=.2,size=(heads,rank,d)),'metricRoot':np.eye(d),'keyCenter':rng.normal(scale=.1,size=d)};a=rng.normal(scale=.3,size=d);b=rng.normal(scale=.3,size=d);x=np.stack([floats(p['q']).astype(float)/d**.25 for p in packets]);k=floats(keys).astype(float)/d**.25-asset['keyCenter'];v=floats(values).astype(float)
'''


if __name__=='__main__':unittest.main()
