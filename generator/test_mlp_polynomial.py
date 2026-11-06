"""Tiny mechanics checks, with no real model or native research arrays."""
from pathlib import Path
import subprocess
import sys
import unittest


class CompiledMLPMechanics(unittest.TestCase):
    def run_code(self, code):
        subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parent,
                       check=True, capture_output=True, text=True)

    def test_retained_product_polynomial_and_receiver(self):
        self.run_code('''import numpy as np
from mlp_polynomial import hidden_coefficients,horner,consume
rng=np.random.default_rng(77)
a0,a1,v0,v1=[rng.normal(size=(1,2,3)) for _ in range(4)]
w=rng.normal(size=(4,3));h=hidden_coefficients(a0,a1,v0,v1);coeff=h@w.T
q=1/(1+np.exp(-a0))
for alpha in [0,.25,.5,1]:
 d=alpha*a1
 sig=q+q*(1-q)*d+.5*q*(1-q)*(1-2*q)*d*d
 expected=((a0+d)*sig*(v0+alpha*v1))@w.T
 assert np.allclose(horner(coeff,alpha),expected,atol=1e-13,rtol=1e-13)
assert np.any(coeff[4]!=0)
# Binary-exact operands: the receiving update must consume the predicted write.
rounded,state=consume(np.full((1,2,4),2.,np.float32),np.full((1,1,4),.5,np.float32),np.full((1,2,4),4.))
assert np.array_equal(rounded,np.full((1,2,4),4.))
assert np.array_equal(state,np.full((1,2,4),4.))
''')

    def test_segment_custody_rejects_corruption(self):
        self.run_code('''import numpy as np,hashlib,io,tempfile
from pathlib import Path
from mlp_polynomial import checked_ref
with tempfile.TemporaryDirectory() as directory:
 path=Path(directory)/'native.npyb';buf=io.BytesIO();np.save(buf,np.arange(6,dtype=np.float32).reshape(1,2,3),allow_pickle=False)
 raw=buf.getvalue();path.write_bytes(b'prefix'+raw+b'suffix')
 ref={'path':str(path),'kind':'bundle','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'offset':6,'nbytes':len(raw),'segmentSHA256':hashlib.sha256(raw).hexdigest(),'shape':[1,2,3]}
 assert np.array_equal(checked_ref(ref),np.arange(6).reshape(1,2,3))
 ref['offset']=7
 try:checked_ref(ref)
 except ValueError:pass
 else:raise AssertionError('Incorrect native segment accepted')
''')

    def test_fresh_coefficient_only_serving_and_bundle_identity(self):
        self.run_code('''import numpy as np,json,tempfile,subprocess,sys
from pathlib import Path
from mlp_polynomial import sha
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);coeff=np.zeros((5,1,2,4));coeff[0]=4;coeff[1]=2
 np.save(root/'coefficients.npy',coeff,allow_pickle=False)
 np.savez(root/'consumer.npz',parent=np.full((1,2,4),2.,np.float32),gate=np.full((1,1,4),.5,np.float32))
 np.savez(root/'controls.npz',base=np.full((1,2,4),4.),changed=np.full((1,2,4),6.))
 spec={name:{'path':str(root/file),'sha256':sha(root/file)} for name,file in [('coefficients','coefficients.npy'),('consumer','consumer.npz')]}
 spec.update(mode='primary',sourceRoots=['/nonexistent-native-source'],alphas=[0,.5,1],repeats=2,scope='synthetic mechanics only')
 path=root/'serve.json';path.write_text(json.dumps(spec))
 subprocess.run([sys.executable,'mlp_polynomial.py','serve',str(path),str(root)],check=True,capture_output=True,text=True)
 seal=json.loads((root/'prediction-seal.json').read_text());assert len(seal['rows'])==3
 assert str(root/'controls.npz') not in seal['opened'] and not seal['nativeEndpointOutputsSupplied']
 row=next(r for r in seal['rows'] if r['alpha']==.5 and r['method']=='polynomial')
 with np.load(root/row['path'],allow_pickle=False) as output:
  assert np.array_equal(output['raw'],np.full((1,2,4),5.))
  assert np.array_equal(output['state'],np.full((1,2,4),4.5))
 assert sha(root/row['path'])==row['sha256']
 control={key:spec[key] for key in ['consumer','sourceRoots','alphas','repeats','scope']}
 control.update(mode='controls',controls={'path':str(root/'controls.npz'),'sha256':sha(root/'controls.npz')})
 (root/'control-spec.json').write_text(json.dumps(control))
 subprocess.run([sys.executable,'mlp_polynomial.py','serve',str(root/'control-spec.json'),str(root)],check=True,capture_output=True,text=True)
 assert len(json.loads((root/'control-seal.json').read_text())['rows'])==6
 (root/'coefficients.npy').write_bytes(b'corrupt')
 rejected=subprocess.run([sys.executable,'mlp_polynomial.py','serve',str(path),str(root)],capture_output=True,text=True)
 assert rejected.returncode!=0 and 'Serving bundle changed' in rejected.stderr
''')

    def test_complete_compilation_with_synthetic_parameter_stream(self):
        self.run_code('''import numpy as np,io,json,hashlib,tempfile,types,sys
from pathlib import Path
from mlp_polynomial import compile_line,sha
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);source=root/'source';source.mkdir();out=root/'out';out.mkdir()
 def npy(name,array):
  path=source/name;np.save(path,array,allow_pickle=False)
  return {'path':str(path),'sha256':sha(path),'shape':list(array.shape),'kind':'npy'}
 operands={name:npy(name+'.npy',array) for name,array in {
 'base':np.ones((1,2,4),np.float32),'changed':np.full((1,2,4),2.,np.float32),
 'gateAttention':np.zeros((1,1,4),np.float32),'gateMLP':np.full((1,1,4),.5,np.float32),
 'attention':np.zeros((1,3,4),np.float32)}.items()}
 words=(np.full((1,3,4),2.,np.float32).view(np.uint32)>>16).astype('<u2')
 tensor_bytes=words.tobytes();header=json.dumps({'tensor_0':{'shape':[1,3,4],'dtype':'BF16','data_offsets':[0,len(tensor_bytes)]}}).encode()
 path=source/'parent.safetensors';path.write_bytes(len(header).to_bytes(8,'little')+header+tensor_bytes)
 operands['image']={'path':str(path),'sha256':sha(path),'shape':[1,3,4],'kind':'safetensors','tensor':'tensor_0','tensorSHA256':hashlib.sha256(tensor_bytes).hexdigest()}
 payloads={};entries=[]
 for name,shape in [('linear_in.weight',(12,4)),('linear_out.weight',(4,6))]:
  words=(np.full(shape,.25,np.float32).view(np.uint32)>>16).astype('<u2');buf=io.BytesIO();np.save(buf,words,allow_pickle=False);raw=buf.getvalue()
  payloads[name]=raw;entries.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'source':{'tensor':name}})
 fake=types.ModuleType('retrieve_weights');fake.validate_manifest=lambda value:value
 fake._source_entries=lambda manifest,checkpoint:((entry,None,100) for entry in manifest['artifacts'])
 fake._artifact_chunks=lambda entry,stream,size:iter([payloads[entry['path']]])
 sys.modules['retrieve_weights']=fake
 checkpoint=source/'checkpoint.safetensors';checkpoint.write_bytes(b'synthetic-only')
 spec={'operands':operands,'selectedRows':[0,2],'checkpointFile':str(checkpoint),'checkpointRoot':str(source),
 'sourceRoots':[str(source)],'parameters':{'artifacts':entries},'scope':'synthetic mechanics only'}
 path=root/'compile.json';path.write_text(json.dumps(spec));compile_line(path,out)
 receipt=json.loads((out/'compile-receipt.json').read_text())
 assert receipt['compileSpecSHA256']==sha(path) and receipt['coefficientShape']==[5,1,2,4]
 assert receipt['offlineWork']['denseMatmuls']==2
 assert receipt['coefficientsSHA256']==sha(out/'coefficients.npy')
 assert np.isfinite(np.load(out/'coefficients.npy',allow_pickle=False)).all()
''')


if __name__ == '__main__':
    unittest.main()
