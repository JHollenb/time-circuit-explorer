"""Tiny local integrity/numeric checks. Real prefix comparisons run only in mrun."""
from pathlib import Path
import subprocess
import sys
import unittest


class EncoderAtomMechanics(unittest.TestCase):
    def run_code(self,code):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_bf16_ties_and_nonfinite(self):
        self.run_code('''import numpy as np
from encoder_atoms import words,floats
x=np.array([1+2**-8,1+3*2**-8,-1-2**-8,-1-3*2**-8],dtype=np.float32)
assert np.array_equal(floats(words(x)),[1,1+2**-6,-1,-1-2**-6])
try:words([float('nan')])
except ValueError:pass
else:raise AssertionError('Nonfinite admitted')
''')

    def test_missing_rows_and_authenticated_lazy_reads(self):
        self.run_code('''import tempfile,hashlib,numpy as np
from pathlib import Path
from encoder_atoms import AtomStore,save,sha
with tempfile.TemporaryDirectory() as d:
 root=Path(d);ids=np.array([1,4,8],dtype='<u4');rows=np.arange(18,dtype='<u2').reshape(3,6)
 np.save(root/'coverage.npy',ids);np.save(root/'atoms-0000.npy',rows)
 hashes=np.stack([np.frombuffer(hashlib.sha256(x.tobytes()).digest(),dtype=np.uint8) for x in rows]);np.save(root/'row-hashes.npy',hashes)
 def item(n):return {'path':n,'sha256':sha(root/n),'bytes':(root/n).stat().st_size}
 m={'coverage':item('coverage.npy'),'rowHashes':item('row-hashes.npy'),'vocabSize':10,'chunkRows':512,
    'shards':[{**item('atoms-0000.npy'),'shape':[3,6]}]};save(root/'manifest.json',m);digest=sha(root/'manifest.json')
 store=AtomStore(root,digest);assert np.array_equal(store.gather([8,1,8]),rows[[2,0,2]])
 assert store.rowsRead==2 and store.rowBytesRead==24
 try:store.gather([2])
 except ValueError as e:assert 'Uncovered token IDs: 2' in str(e)
 else:raise AssertionError('Missing token silently filled')
 # Integrity is checked for each newly consumed row, without reading every large shard.
 broken=rows.copy();broken[1,0]+=1;np.save(root/'atoms-0000.npy',broken)
 try:AtomStore(root,digest).gather([4])
 except ValueError as e:assert 'checksum differs' in str(e)
 else:raise AssertionError('Corrupt learned atom admitted')
''')

    def test_serving_isolation(self):
        self.run_code('''from encoder_atoms import guard
guard()
try:import torch
except ImportError as e:assert 'prohibited' in str(e)
else:raise AssertionError('Neural runtime admitted')
try:open('/mnt/big/llm-models/never-read.safetensors','rb')
except PermissionError:pass
else:raise AssertionError('Original weights admitted')
''')

    def test_full_domain_contract_and_frozen_prompts(self):
        self.run_code('''import tempfile
from pathlib import Path
from encoder_atoms_study import stage,HELD
from encoder_atoms import read,sha,identity
with tempfile.TemporaryDirectory() as d:
 root=Path(d)/'run';stage(root,'full');r=read(root/'request.json');p=root/'payload'
 assert r['compilerTokenRows']==151936 and r['physicalBatchGeometry']==[1,512,2560]
 assert r['requestSHA256']==identity({k:v for k,v in r.items() if k!='requestSHA256'})
 for n,h in r['sourceHashes'].items():assert sha(p/n)==h
 assert read(p/'raw-prompts.json')['prompts']==HELD
 assert 'prompts' not in read(p/'compile.json')
''')


if __name__=='__main__':unittest.main()
