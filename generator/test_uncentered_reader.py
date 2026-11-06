import copy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from encoder_atoms import words, floats, sha, identity, save
from paired_fields import retain
from uncentered_reader import stream
from uncentered_reader_study import OWNED, qualify_inputs, execute
import test_direct_reader


class UncenteredReaderMechanics(unittest.TestCase):
    def fixture(self):
        return test_direct_reader.DirectReaderMechanics.fixture(self)[0]

    def test_independent_uncentered_causal_expression(self):
        p=self.fixture();r,state=stream(p);expected=np.zeros((12,8))
        for i in range(12):
            allowed=np.flatnonzero(p[0]['mask'][:i+1])
            if not len(allowed):continue
            K=floats(p[0]['k'][allowed]).astype(float)/4**.25;V=floats(p[0]['v'][allowed]).astype(float)
            for h in range(2):
                q=floats(p[h]['q'][i]).astype(float)/4**.25;s=K@q;chosen=np.lexsort((allowed,-s))[:4]
                w=np.exp(s[chosen]-s[chosen].max());w/=w.sum();expected[i,h*4:(h+1)*4]=w@V[chosen]
                np.testing.assert_array_equal(r['supportPositions'][i,h,:len(chosen)],allowed[chosen])
        np.testing.assert_allclose(r['raw'],expected,rtol=2e-15,atol=2e-15)
        self.assertFalse({'centers','scores','moment','U','landmarks','maximum','z'}&state.keys())

    def test_common_shift_preserves_toy_support_and_rounded_heads(self):
        from direct_reader import stream as centered
        p=self.fixture();a,_=stream(p);b,_=centered(p,np.array([.25,-.125,.5,1.]))
        np.testing.assert_array_equal(a['supportPositions'],b['supportPositions'])
        np.testing.assert_array_equal(a['rounded'],b['rounded'])

    def test_finite_common_shift_can_change_support(self):
        from direct_reader import stream as centered
        p=[{'q':words(np.tile([1.,0.],(5,1))),'k':words(np.column_stack([np.arange(5.),np.zeros(5)])),
            'v':words(np.column_stack([np.arange(5.),np.ones(5)])),'mask':np.ones(5,dtype=int),'positions':np.arange(5)}]
        a,_=stream(p);b,_=centered(p,np.array([1e16,0.]))
        np.testing.assert_array_equal(a['supportPositions'][-1,0],[4,3,2,1])
        self.assertFalse(np.array_equal(a['supportPositions'][-1],b['supportPositions'][-1]))

    def test_split_npz_continuation_and_parent_immutability(self):
        p=self.fixture();whole,_=stream(p);a,parent=stream([{k:v[:6] for k,v in x.items()} for x in p]);before=copy.deepcopy(parent)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.npz';np.savez(path,**parent)
            with np.load(path,allow_pickle=False) as f:restored={k:f[k] for k in f.files}
            b,final=stream([{k:v[6:] for k,v in x.items()} for x in p],restored)
        for k in ['raw','rounded','supportPositions']:np.testing.assert_array_equal(np.concatenate([a[k],b[k]]),whole[k])
        np.testing.assert_array_equal(parent['keys'],before['keys']);self.assertEqual(final['steps'],12)

    def test_causality_finite_clock_and_group_bound_resume(self):
        p=self.fixture();a,_=stream(p);changed=copy.deepcopy(p)
        for x in changed:x['v'][0]=words([100.]*4);x['q'][-1]=words([-50.]*4)
        b,_=stream(changed);np.testing.assert_array_equal(a['rounded'][:-1],b['rounded'][:-1]);self.assertFalse(a['rounded'][0].any())
        _,state=stream([{k:v[:6] for k,v in x.items()} for x in p]);suffix=[{k:v[6:] for k,v in x.items()} for x in p]
        for kwargs in [{'source_identity':'wrong'},{'kv_group':1},{'query_heads':[4,5]}]:
            with self.assertRaises(ValueError):stream(suffix,state,**kwargs)
        with self.assertRaises(ValueError):stream(p,state)
        p[0]['q'][0,0]=0x7f80
        with self.assertRaises(ValueError):stream(p)

    def test_reuse_binds_actual_consumer_instead_of_comparator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,own,target=test_direct_reader.DirectReaderMechanics.proof_fixture(self,tmp)
            changed=np.zeros((1,512,4096),dtype=np.uint16);changed[0,511,4095]=1
            field=retain(root,'different-comparator',changed);field['logicalDtype']='torch.bfloat16'
            target['comparisonReader']=copy.deepcopy(target['reader']);target['comparisonReader']['fields']['B']=field
            self.assertTrue(qualify_inputs(own,target,{}, {})['completeDemandedInputsIdentical'])
            candidate=copy.deepcopy(own);candidate['fields']['B']=field
            proof=qualify_inputs(candidate,target,{},{});self.assertFalse(proof['completeDemandedInputsIdentical'])
            self.assertEqual(proof['rows'][1]['headBF16WordMismatches'],1)

    def test_isolated_payload_excludes_all_source_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in OWNED:shutil.copyfile(Path(__file__).resolve().parent/name,root/name)
            excluded=[]
            for name in ['geometry.npz','centers.npy','rank-one-basis.npz']:
                path=root/name;path.write_bytes(b'excluded');excluded.append({'path':str(path),'sha256':sha(path)})
            save(root/'inputs.json',{'excludedSources':excluded,'forbiddenTargets':[]})
            code='import sys;sys.path.insert(0,".");from pathlib import Path;from uncentered_reader import install_serve_guard;from encoder_atoms import read;assert len(install_serve_guard(Path("."),read("inputs.json")))==3;assert not {"direct_reader","selected_reader","rank_one_tail","attention_bank","torch","diffusers","transformers"}.intersection(sys.modules)'
            subprocess.run([sys.executable,'-I','-c',code],cwd=root,check=True,capture_output=True,timeout=20)

    def test_failed_child_cannot_seal_or_open_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r={'sourceHashes':{}};r['requestSHA256']=identity(r);save(root/'request.json',r)
            with patch('uncentered_reader_study.subprocess.run',side_effect=RuntimeError('partial')),patch('uncentered_reader_study.score') as score:
                with self.assertRaises(RuntimeError):execute(root)
                score.assert_not_called();self.assertFalse((root/'head-seal.json').exists())


if __name__=='__main__':unittest.main()
