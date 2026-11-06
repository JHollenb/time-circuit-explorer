"""Tiny mechanics organisms for the complete-query incremental attention law."""
from pathlib import Path
import subprocess
import sys
import unittest


class GlobalAttentionMechanics(unittest.TestCase):
    def run_code(self,code):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_partition_and_changed_query_interactions(self):
        self.run_code('''import numpy as np
from global_attention import full_attention,fixed_cache,incremental,merge
rng=np.random.default_rng(919);selected=np.array([1,6,8]);fixed={k:rng.normal(size=(11,4)) for k in ['q','k','v']}
for value in fixed.values():value[selected]=0
cache=fixed_cache(fixed,selected,chunk=3)
fields={k:rng.normal(size=(3,4)) for k in ['q','k','v']}
base=full_attention(merge(fixed,selected,fields),chunk=2)
assert np.allclose(incremental(fields,fixed,cache),base,rtol=1e-13,atol=1e-13)
changed={k:v.copy() for k,v in fields.items()};changed['q']*=4
expected=full_attention(merge(fixed,selected,changed),chunk=4)
actual=incremental(changed,fixed,cache)
assert np.allclose(actual,expected,rtol=1e-13,atol=1e-13)
unchanged=cache['unchanged'];assert np.allclose(actual[unchanged],base[unchanged])
assert np.max(np.abs(actual[selected]-base[selected]))>.1
# Updated keys dwarf the fixed-source maximum: exponent rescaling stays finite.
changed['k']*=1000
actual=incremental(changed,fixed,cache);expected=full_attention(merge(fixed,selected,changed),chunk=4)
assert np.isfinite(actual).all() and np.allclose(actual,expected,atol=1e-12,rtol=1e-12)
# Updated values affect queries outside the changed set too.
changed={k:v.copy() for k,v in fields.items()};changed['v']+=2
actual=incremental(changed,fixed,cache);assert np.max(np.abs(actual[unchanged]-base[unchanged]))>.1
''')

    def test_state_only_fresh_serving_and_controls(self):
        self.run_code('''import numpy as np,json,tempfile,subprocess,sys
from pathlib import Path
from global_attention import fixed_cache,full_attention,merge
from state_receiver import state_basis
from mlp_polynomial import sha,quantize
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);rng=np.random.default_rng(52);changed=np.array([2,5])
 basis,_=state_basis(rng.normal(size=(5,1,2,8)),np.zeros((1,2,8)),np.ones((1,1,8)))
 reader={'basis':basis,'gamma_q':np.ones(4,np.float32),'gamma_k':np.ones(4,np.float32),'cos':np.ones((2,4),np.float32),'sin':np.zeros((2,4),np.float32)}
 for k in ['q','k','v']:reader['bank_'+k]=rng.normal(size=(2,5,4));reader['offset_'+k]=np.zeros((1,1,4))
 fixed={k:quantize(rng.normal(size=(7,4))) for k in ['q','k','v']}
 for value in fixed.values():value[changed]=0
 cache=fixed_cache(fixed,changed);spec={}
 for key,values in [('reader',reader),('fixed',fixed),('cache',cache)]:
  path=root/(key+'.npz');np.savez(path,**values);spec[key]={'path':str(path),'sha256':sha(path)}
 packets=[]
 for i in range(2):
  path=root/('opaque'+str(i)+'.npy');np.save(path,quantize(rng.normal(size=(1,2,8))),allow_pickle=False)
  packets.append({'id':'p'+str(i),'path':str(path),'sha256':sha(path),'shape':[1,2,8]})
 spec.update(packets=packets,eps=1e-6,repeats=2,scope='synthetic',sourceRoots=[str(root)])
 path=root/'serve.json';path.write_text(json.dumps(spec));(root/'evaluation.json').write_text('DENIED')
 subprocess.run([sys.executable,'global_attention.py','serve',str(path),str(root)],check=True)
 seal=json.loads((root/'prediction-seal.json').read_text());assert len(seal['rows'])==8 and not seal['doseInput']
 assert str(root/'evaluation.json') not in seal['opened'] and seal['learnedReducedMapsPerPrimaryEvaluation']==3
 outputs={}
 for row in seal['rows']:
  assert sha(root/row['path'])==row['sha256']
  with np.load(root/row['path'],allow_pickle=False) as data:outputs[row['method'],row['id']]={k:data[k] for k in data.files}
  fields={k:outputs[row['method'],row['id']]['changed_'+k] for k in ['q','k','v']}
  assert np.allclose(outputs[row['method'],row['id']]['raw'],full_attention(merge(fixed,changed,fields)),atol=1e-12)
 assert np.array_equal(outputs['all','p0']['raw'],outputs['unchanged_replay','p1']['raw'])
 # The public wrapper must preserve the state-only transaction on a tiny bundle.
 import shutil
 compiled={'scope':'synthetic','bundles':{}}
 for key,file in [('reader','reader-head.npz'),('fixed','fixed-head.npz'),('cache','attention-cache.npz')]:
  shutil.copyfile(root/(key+'.npz'),root/file);compiled['bundles'][file]={'sha256':sha(root/file)}
 (root/'compile-receipt.json').write_text(json.dumps(compiled))
 sample={**packets[0],'path':Path(packets[0]['path']).name};(root/'sample-states.json').write_text(json.dumps({'packets':[sample]}))
 needed=['reader-head.npz','fixed-head.npz','attention-cache.npz','compile-receipt.json','sample-states.json',sample['path']]
 (root/'files.json').write_text(json.dumps({name:sha(root/name) for name in needed}))
 subprocess.run([sys.executable,'render_global_head.py','--bundle',str(root),'--state',packets[1]['path'],'--output',str(root/'rendered')],check=True)
 with np.load(root/'rendered/all-input.npz',allow_pickle=False) as rendered:assert np.array_equal(rendered['raw'],outputs['all','p1']['raw'])
 spec['packets'][1]['alpha']=.5;path.write_text(json.dumps(spec))
 failed=subprocess.run([sys.executable,'global_attention.py','serve',str(path),str(root)],capture_output=True,text=True)
 assert failed.returncode!=0 and 'identity and state bytes' in failed.stderr
''')

    def test_complete_compiler_and_frozen_selected_row_exclusion(self):
        self.run_code('''import numpy as np,json,hashlib,io,tempfile,types,sys
from pathlib import Path
from global_attention import compile_head
from state_receiver import state_basis
from mlp_polynomial import sha
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);source=root/'source';source.mkdir();output=root/'out';output.mkdir();rng=np.random.default_rng(90)
 def tensor(name,value,bf=True):
  words=(value.astype(np.float32).view(np.uint32)>>16).astype('<u2') if bf else value.astype('<f4')
  raw=words.tobytes();head=json.dumps({'x':{'shape':list(value.shape),'dtype':'BF16' if bf else 'F32','data_offsets':[0,len(raw)]}}).encode()
  path=source/(name+'.safetensors');path.write_bytes(len(head).to_bytes(8,'little')+head+raw)
  return {'path':str(path),'sha256':sha(path),'shape':list(value.shape),'dtype':'BF16' if bf else 'F32','tensor':'x','tensorSHA256':hashlib.sha256(raw).hexdigest()}
 operands={name:tensor(name,value,bf) for name,value,bf in [('image',rng.normal(size=(1,6,8)),True),('text',rng.normal(size=(1,3,8)),True),
 ('imageMod',np.zeros((1,48)),True),('textMod',np.zeros((1,48)),True),('cos',np.ones((9,4)),False),('sin',np.zeros((9,4)),False)]}
 entries=[];payloads={}
 for name in ['to_q','to_k','to_v','add_q_proj','add_k_proj','add_v_proj','norm_q','norm_k','norm_added_q','norm_added_k']:
  array=np.ones(4,np.float32) if name.startswith('norm_') else np.eye(8,dtype=np.float32)
  words=(array.view(np.uint32)>>16).astype('<u2');buf=io.BytesIO();np.save(buf,words,allow_pickle=False);raw=buf.getvalue();payloads[name]=raw
  entries.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'source':{'tensor':'transformer_blocks.2.attn.'+name+'.weight'}})
 fake=types.ModuleType('retrieve_weights');fake.validate_manifest=lambda x:x
 fake._source_entries=lambda m,c:((e,None,128) for e in m['artifacts']);fake._artifact_chunks=lambda e,s,n:iter([payloads[e['path']]])
 sys.modules['retrieve_weights']=fake
 basis,_=state_basis(rng.normal(size=(5,1,2,8)),np.zeros((1,2,8)),np.ones((1,1,8)))
 p=source/'basis.npy';np.save(p,basis,allow_pickle=False);basis_ref={'path':str(p),'sha256':sha(p)};banks={}
 for k in ['q','k','v']:
  path=source/('bank-'+k+'.npz');np.savez(path,bank=basis.copy(),offset=np.zeros((1,1,8)));banks[k]={'path':str(path),'sha256':sha(path)}
 checkpoint=source/'checkpoint.safetensors';checkpoint.write_bytes(b'SYNTHETIC')
 spec={'operands':operands,'basis':basis_ref,'banks':banks,'parameters':{'artifacts':entries},'checkpointFile':str(checkpoint),'checkpointRoot':str(source),
 'sourceRoots':[str(source)],'selectedRows':[1,4],'head':0,'headDim':4,'eps':1e-6,'scope':'synthetic'}
 p=root/'compile.json';p.write_text(json.dumps(spec));compile_head(p,output)
 receipt=json.loads((output/'compile-receipt.json').read_text());assert receipt['originalHeadProjectionEvaluations']==6
 with np.load(output/'fixed-head.npz',allow_pickle=False) as data:
  assert all(np.all(data[k][[4,7]]==0) for k in ['q','k','v'])
 with np.load(output/'attention-cache.npz',allow_pickle=False) as data:assert np.array_equal(data['changed'],[4,7])
 with np.load(output/'reader-head.npz',allow_pickle=False) as data:assert data['bank_q'].shape==(2,5,4)
''')

    def test_matched_value_delta_control_transaction(self):
        self.run_code('''import numpy as np,tempfile,json,subprocess,sys,shutil
from pathlib import Path
from global_attention import full_attention,fixed_cache,merge
from mlp_polynomial import sha,quantize
from response_study import identity
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);source=root/'source';source.mkdir();rng=np.random.default_rng(12);selected=np.array([2,5])
 for name in ['global_value_address.py','global_attention.py','state_receiver.py','mlp_polynomial.py','response_chart.py','prompt_closure.py','response_study.py']:shutil.copyfile(Path(name),root/name)
 fixed={k:quantize(rng.normal(size=(7,4))) for k in ['q','k','v']}
 for value in fixed.values():value[selected]=0
 base={k:quantize(rng.normal(size=(2,4))) for k in ['q','k','v']}
 changed={k:quantize(base[k]+rng.normal(size=(2,4))*.1) for k in ['q','k','v']}
 refs={}
 def keep(name,**values):
  path=source/(name+'.npz');np.savez(path,**values);return {'path':str(path),'sha256':sha(path)}
 for label,fields in [('baseline',base),('changed',changed)]:
  head=full_attention(merge(fixed,selected,fields));refs[label]=keep(label,raw=head,rounded=quantize(head),**{'changed_'+k:v for k,v in fields.items()})
 refs['fixed']=keep('fixed',**fixed);refs['cache']=keep('cache',**fixed_cache(fixed,selected))
 predict={**refs,'sourceRoot':str(source),'scope':'synthetic'};(root/'predict.json').write_text(json.dumps(predict))
 value_fields={**base,'v':changed['v']};head=full_attention(merge(fixed,selected,value_fields))
 evaluation={'nativeBase':refs['baseline'],'nativeChanged':refs['changed'],'correctValuesOnly':keep('correct',raw=head,rounded=quantize(head))}
 (root/'evaluation.json').write_text(json.dumps(evaluation))
 request={'schema':'synthetic','textRows':2,'parent':{},'sourceHashes':{}};request['requestSHA256']=identity(request);(root/'request.json').write_text(json.dumps(request))
 subprocess.run([sys.executable,'global_value_address.py','execute',str(root)],check=True)
 report=json.loads((root/'response-result/report.json').read_text());assert report['correctValueReadMaxAbs']<1e-12
 seal=json.loads((root/'response-result/candidate-seal.json').read_text());assert str(root/'evaluation.json') not in seal['opened']
 for row in seal['rows']:
  if row['method']!='zero':assert abs(row['proposedDeltaPower']-report['sourceDeltaPower'])<1e-12
 assert len(report['scores'])==80 and report['newNeuralFrameworkCalls']==0
''')


if __name__=='__main__':unittest.main()
