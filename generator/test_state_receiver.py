"""Small synthetic state/basis/custody checks; no model arrays or research runs."""
from pathlib import Path
import subprocess
import sys
import unittest


class StateReceiverMechanics(unittest.TestCase):
    def run_code(self, code):
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_basis_and_projected_residual_identity(self):
        self.run_code('''import numpy as np
from state_receiver import state_basis,receive
rng=np.random.default_rng(177)
coeff=rng.normal(size=(5,1,2,8));parent=rng.normal(size=(1,2,8));gate=rng.normal(size=(1,1,8))
b,info=state_basis(coeff,parent,gate)
assert info['ranks']==[5,5]
assert np.allclose(np.einsum('rkd,rld->rkl',b,b),np.eye(5),atol=1e-13)
state=rng.normal(size=(1,2,8));factor=rng.normal(size=(1,1,8));shift=rng.normal(size=(1,1,8))
weights={k:rng.normal(size=(8,8)) for k in ['q','k','v']}
banks={k:(b*factor[0,0])@w.T for k,w in weights.items()};offsets={k:shift@w.T for k,w in weights.items()}
out=receive(state,b,banks,offsets,1e-6)
v=state-state.mean(axis=-1,keepdims=True)
for k,w in weights.items():
 smooth=(factor*v/out['scale']+shift)@w.T
 missing=(factor*out['residual']/out['scale'])@w.T
 assert np.allclose(smooth-out[k+'_raw'],missing,atol=1e-12)
assert np.max(np.abs(out['residual']))>0
# Degenerate producer directions stay padded rather than inventing directions.
deg=np.zeros_like(coeff);deg[0]=1
_,d=state_basis(deg,np.ones_like(parent),np.ones_like(gate))
assert d['ranks']==[0,0]
''')

    def test_fresh_state_only_serving_and_input_rejection(self):
        self.run_code('''import numpy as np,json,tempfile,subprocess,sys
from pathlib import Path
from state_receiver import state_basis,read_state
from mlp_polynomial import sha,quantize
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);rng=np.random.default_rng(28)
 coeff=rng.normal(size=(5,1,2,8));basis,_=state_basis(coeff,np.zeros((1,2,8)),np.ones((1,1,8)))
 np.save(root/'basis.npy',basis,allow_pickle=False)
 bankrefs={}
 for k in ['q','k','v']:
  path=root/('bank-'+k+'.npz');np.savez(path,bank=basis.copy(),offset=np.zeros((1,1,8)))
  bankrefs[k]={'path':str(path),'sha256':sha(path)}
 state=quantize(rng.normal(size=(1,2,8)));path=root/'opaque-state.npy';np.save(path,state,allow_pickle=False)
 packet={'id':'opaque','path':str(path),'sha256':sha(path),'shape':[1,2,8]}
 assert np.array_equal(read_state(packet),state)
 try:read_state({**packet,'alpha':.5})
 except ValueError:pass
 else:raise AssertionError('Dose admitted')
 (root/'evaluation.json').write_text('PRIVATE')
 spec={'schema':'synthetic','basis':{'path':str(root/'basis.npy'),'sha256':sha(root/'basis.npy')},'banks':bankrefs,
 'packets':[packet],'eps':1e-6,'repeats':2,'scope':'synthetic mechanics','sourceRoots':[str(root)]}
 serve=root/'serve.json';serve.write_text(json.dumps(spec))
 subprocess.run([sys.executable,'state_receiver.py','serve',str(serve),str(root)],check=True)
 seal=json.loads((root/'receiver-seal.json').read_text());assert not seal['doseInput'] and seal['originalDenseProjections']==0
 assert str(root/'evaluation.json') not in seal['opened']
 pred=root/seal['rows'][0]['output'];assert sha(pred)==seal['rows'][0]['sha256']
 assert len(seal['rows'][0]['arithmeticSeconds'])==2
 spec['packets'][0]['alpha']=.5;serve.write_text(json.dumps(spec))
 failed=subprocess.run([sys.executable,'state_receiver.py','serve',str(serve),str(root)],capture_output=True,text=True)
 assert failed.returncode!=0 and 'identity and state bytes' in failed.stderr
 # The source-root guard denies an unlisted private operand even if it exists.
 code="from mlp_polynomial import audit_reads; audit_reads([], ["+repr(str(root))+"]); open("+repr(str(root/'evaluation.json'))+").read()"
 failed=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
 assert failed.returncode!=0 and 'outside declared operands' in failed.stderr
''')

    def test_complete_compiler_with_synthetic_parameter_stream(self):
        self.run_code('''import numpy as np,json,io,hashlib,tempfile,types,sys
from pathlib import Path
from state_receiver import compile_receiver
from mlp_polynomial import sha
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);source=root/'source';source.mkdir();output=root/'out';output.mkdir()
 rng=np.random.default_rng(99);np.save(source/'coefficients.npy',rng.normal(size=(5,1,2,8)),allow_pickle=False)
 np.savez(source/'consumer.npz',parent=np.zeros((1,2,8),np.float32),gate=np.ones((1,1,8),np.float32))
 mod=source/'modulation.npy';np.save(mod,np.zeros((1,48),np.float32),allow_pickle=False)
 entries=[];payloads={}
 for k in ['q','k','v']:
  buf=io.BytesIO();words=(np.eye(8,dtype=np.float32).view(np.uint32)>>16).astype('<u2');np.save(buf,words,allow_pickle=False)
  raw=buf.getvalue();payloads[k]=raw;entries.append({'path':k,'sha256':hashlib.sha256(raw).hexdigest(),'source':{'tensor':'transformer_blocks.2.attn.to_'+k+'.weight'}})
 fake=types.ModuleType('retrieve_weights');fake.validate_manifest=lambda x:x
 fake._source_entries=lambda m,c:((e,None,128) for e in m['artifacts'])
 fake._artifact_chunks=lambda e,s,n:iter([payloads[e['path']]])
 sys.modules['retrieve_weights']=fake
 checkpoint=source/'checkpoint.safetensors';checkpoint.write_bytes(b'SYNTHETIC')
 spec={name:{'path':str(source/file),'sha256':sha(source/file)} for name,file in [('coefficients','coefficients.npy'),('consumer','consumer.npz')]}
 spec.update(modulation={'path':str(mod),'sha256':sha(mod),'kind':'npy','shape':[1,48]},
 parameters={'artifacts':entries},checkpointRoot=str(source),checkpointFile=str(checkpoint),sourceRoots=[str(source)],scope='synthetic')
 path=root/'compile.json';path.write_text(json.dumps(spec));compile_receiver(path,output)
 receipt=json.loads((output/'receiver-compile-receipt.json').read_text())
 assert receipt['compileSpecSHA256']==sha(path) and receipt['basisSHA256']==sha(output/'basis.npy')
 basis=np.load(output/'basis.npy',allow_pickle=False)
 for k,item in receipt['banks'].items():
  assert sha(output/item['path'])==item['sha256']
  with np.load(output/item['path'],allow_pickle=False) as bank:assert np.allclose(bank['bank'],basis)
''')

    def test_signed_error_decomposition(self):
        self.run_code('''import numpy as np
from state_receiver_study import decompose
t=np.array([1.,2.,3.]);c=t+1;p=t-.5
d=decompose(p,c,t,np.ones(3))
assert d['crossPower']<0 and d['decompositionMaxAbs']==0
assert abs(d['totalPower']-d['readerPower']-d['inheritedPower']-d['crossPower'])<1e-12
from state_reader_attribution import telescope
components={'O':np.array([1.,-1.]),'S':np.array([2.,3.]),'N':np.array([-2.,-1.]),'H':np.array([.5,.5])}
result=telescope(components,sum(components.values()),np.ones(2))
assert result['identityMaxAbs']==0 and result['powerClosureAbs']<1e-12
assert result['pairs']['S:N']['crossPower']<0
''')


if __name__ == '__main__':
    unittest.main()
