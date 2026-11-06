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
from direct_reader import stream
from direct_reader_study import OWNED, qualify_inputs, execute


class DirectReaderMechanics(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(32);n,d,heads=12,4,2
        common={'k':words(rng.normal(size=(n,d))),'v':words(rng.normal(size=(n,d))),
            'mask':np.array([0,1,1,0,1,1,1,1,1,0,1,1]),'positions':np.arange(n)}
        return [{**common,'q':words(rng.normal(size=(n,d)))} for _ in range(heads)],rng.normal(size=d)

    def test_direct_selected_scores_against_independent_causal_expression(self):
        p,c=self.fixture();result,state=stream(p,c);expected=np.zeros((12,8))
        for i in range(12):
            allowed=np.flatnonzero(p[0]['mask'][:i+1])
            if not len(allowed):continue
            U=floats(p[0]['k'][allowed]).astype(float)/4**.25-c;V=floats(p[0]['v'][allowed]).astype(float)
            for h in range(2):
                q=floats(p[h]['q'][i]).astype(float)/4**.25;s=U@q;chosen=np.lexsort((allowed,-s))[:4];a=s[chosen];w=np.exp(a-a.max());w/=w.sum()
                expected[i,h*4:(h+1)*4]=w@V[chosen]
                np.testing.assert_array_equal(result['supportPositions'][i,h,:len(chosen)],allowed[chosen])
        np.testing.assert_allclose(result['raw'],expected,rtol=2e-15,atol=2e-15)
        self.assertFalse({'scores','moment','U','landmarks','maximum','z'}&state.keys());self.assertEqual(state['queries'],24)

    def test_parent_selector_and_kv_are_unchanged(self):
        from selected_reader import stream as parent
        p,c=self.fixture();r,s=stream(p,c);a={'landmarks':np.ones((2,3,4)),'metricRoot':np.eye(4),'keyCenter':c};old,t=parent(p,a)
        np.testing.assert_array_equal(r['supportPositions'],old['supportPositions'])
        for k in ['keys','values','positions']:np.testing.assert_array_equal(s[k],t[k])
        np.testing.assert_array_equal(r['rounded'],old['rounded'])

    def test_split_npz_continuation_closes_and_parent_is_immutable(self):
        p,c=self.fixture();whole,_=stream(p,c);cut=6;a,parent=stream([{k:v[:cut] for k,v in x.items()} for x in p],c);before=copy.deepcopy(parent)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.npz';np.savez(path,**parent)
            with np.load(path,allow_pickle=False) as f:restored={k:f[k] for k in f.files}
            b,final=stream([{k:v[cut:] for k,v in x.items()} for x in p],c,restored)
        for k in ['raw','rounded','supportPositions']:np.testing.assert_array_equal(np.concatenate([a[k],b[k]]),whole[k])
        np.testing.assert_array_equal(parent['keys'],before['keys']);self.assertEqual(parent['steps'],6);self.assertEqual(final['steps'],12)

    def test_causal_mask_empty_prefix_finite_clock_and_source(self):
        p,c=self.fixture();a,_=stream(p,c);changed=copy.deepcopy(p)
        for x in changed:x['v'][0]=words([100.]*4);x['q'][-1]=words([-50.]*4)
        b,_=stream(changed,c);np.testing.assert_array_equal(a['rounded'][:-1],b['rounded'][:-1]);self.assertFalse(a['rounded'][0].any())
        _,state=stream([{k:v[:6] for k,v in x.items()} for x in p],c)
        with self.assertRaises(ValueError):stream(p,c,state)
        with self.assertRaises(ValueError):stream([{k:v[6:] for k,v in x.items()} for x in p],c+1,state)
        p[0]['q'][0,0]=0x7f80
        with self.assertRaises(ValueError):stream(p,c)

    def proof_fixture(self,tmp):
        root=Path(tmp)
        def field(name,shape):
            x=retain(root,name,np.zeros(shape,dtype=np.uint16));x['logicalDtype']='torch.bfloat16';return x
        head=field('head',[1,512,4096]);h0=field('h0',[1,512,2560]);lex=root/'lex.npz'
        np.savez(lex,ids=np.arange(512,dtype=np.int64),mask=np.ones(512,dtype=np.int64),positions=np.arange(512,dtype=np.int64),e=np.zeros((512,2560),dtype=np.uint16))
        lexical=[{'id':c,'path':str(lex),'sha256':sha(lex)} for c in ['A','B']]
        own={'fields':{c:head for c in ['A','B']},'h0Fields':{c:h0 for c in ['A','B']},'lexicalRows':lexical}
        target={'reader':{'fields':copy.deepcopy(own['fields']),'lexicalRows':copy.deepcopy(lexical)},'encoder':{'fields':{c:{'H0':h0} for c in ['A','B']},
            'actualHeadReads':[{'case':c,'actualHeadPayloadSHA256':head['payloadSHA256'],'readerFieldSHA256':head['sha256']} for c in ['A','B']],
            'handoffs':[{'case':c,'layer':0,'actualInputPayloadSHA256':h0['payloadSHA256'],'inputFieldSHA256':h0['sha256']} for c in ['A','B']]}}
        return root,own,target

    def test_reuse_requires_every_input_bit_and_lexical_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,own,target=self.proof_fixture(tmp);self.assertTrue(qualify_inputs(own,target,{}, {})['completeDemandedInputsIdentical'])
            changed=np.zeros((1,512,4096),dtype=np.uint16);changed[0,511,4095]=1;d=retain(root,'changed-head',changed);d['logicalDtype']='torch.bfloat16';new=copy.deepcopy(own);new['fields']['B']=d
            proof=qualify_inputs(new,target,{},{});self.assertFalse(proof['completeDemandedInputsIdentical']);self.assertEqual(proof['rows'][1]['headBF16WordMismatches'],1)
            with np.load(own['lexicalRows'][0]['path']) as f:lex={k:f[k] for k in f.files}
            lex['mask'][511]=0;path=root/'changed-lex.npz';np.savez(path,**lex);new=copy.deepcopy(own);new['lexicalRows'][1]={'id':'B','path':str(path),'sha256':sha(path)}
            proof=qualify_inputs(new,target,{},{});self.assertFalse(proof['completeDemandedInputsIdentical']);self.assertEqual(proof['rows'][1]['lexicalControlMismatches']['mask'],1)

    def test_reuse_rejects_foreign_contract_and_broken_actual_consumer_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,own,target=self.proof_fixture(tmp)
            with self.assertRaises(ValueError):qualify_inputs(own,target,{'clock':1},{'clock':2})
            target['encoder']['actualHeadReads'][0]['actualHeadPayloadSHA256']='wrong'
            with self.assertRaises(ValueError):qualify_inputs(own,target,{}, {})

    def test_centers_export_and_original_asset_exclusion_in_isolated_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in OWNED:shutil.copyfile(Path(__file__).resolve().parent/name,root/name)
            geometry=root/'original.npz';np.savez(geometry,keyCenters=np.arange(1024,dtype=np.float64).reshape(8,128),landmarks=np.ones((2,3,4)))
            source={'originalGeometry':{'path':str(geometry),'sha256':sha(geometry)},'forbiddenTargets':[]};save(root/'inputs.json',source)
            r={'sourceHashes':{}};r['requestSHA256']=identity(r);save(root/'request.json',r)
            code='import sys;sys.path.insert(0,".");from pathlib import Path;from direct_reader import compile_centers,install_serve_guard;from encoder_atoms import read,np;compile_centers(Path("."));assert install_serve_guard(Path("."),read("inputs.json"));x=np.load(read("compile-receipt.json")["centers"]["path"]);assert x.nbytes==8192 and x.shape==(8,128);assert not {"selected_reader","rank_one_tail","attention_bank","torch","diffusers","transformers"}.intersection(sys.modules)'
            subprocess.run([sys.executable,'-I','-c',code],cwd=root,check=True,capture_output=True,timeout=20)

    def test_failed_child_cannot_seal_or_open_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r={'sourceHashes':{}};r['requestSHA256']=identity(r);save(root/'request.json',r)
            with patch('direct_reader_study.subprocess.run',side_effect=RuntimeError('partial')),patch('direct_reader_study.score') as score:
                with self.assertRaises(RuntimeError):execute(root)
                score.assert_not_called();self.assertFalse((root/'head-seal.json').exists())


if __name__=='__main__':unittest.main()
