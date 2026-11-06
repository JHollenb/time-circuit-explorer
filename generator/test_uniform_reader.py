import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from encoder_atoms import words, floats, save, identity
from uniform_reader import stream, direct_means
from uniform_reader_study import OWNED, execute, contribution


class UniformReaderMechanics(unittest.TestCase):
    def fixture(self):
        v=words([[1,2,10,20],[3,4,30,40],[100,200,300,400],[-2,-4,2,4]])
        return v,np.array([0,1,0,1]),np.arange(4)

    def test_grouped_causal_means_empty_prefix_and_direct_calculation(self):
        v,m,p=self.fixture();packet,state,count=stream(v,m,p,groups=2,heads=4,dim=2)
        expected=np.array([[0]*8,[3,4,3,4,30,40,30,40],[3,4,3,4,30,40,30,40],[.5,0,.5,0,16,22,16,22]])
        np.testing.assert_array_equal(packet['raw'],expected);np.testing.assert_array_equal(floats(packet['rounded']),expected)
        np.testing.assert_array_equal(direct_means(v,m,2,4,2),expected);np.testing.assert_array_equal(packet['prefixCounts'],[0,1,1,2])
        self.assertEqual(int(state['count']),2);self.assertEqual(int(state['clock']),4)
        self.assertEqual(count['coreStatePayloadBytes'],2*2*8+16)
        empty,_,_=stream(v,np.zeros(4),p,groups=2,heads=4,dim=2)
        self.assertFalse(empty['rounded'].any())

    def test_split_state_closes_and_does_not_mutate_parent(self):
        v,m,p=self.fixture();kw={'groups':2,'heads':4,'dim':2,'source_identity':'fixed'}
        full,_,_=stream(v,m,p,**kw);a,parent,_=stream(v[:2],m[:2],p[:2],**kw);before=parent['sums'].copy()
        b,child,_=stream(v[2:],m[2:],p[2:],parent,**kw)
        np.testing.assert_array_equal(np.concatenate([a['rounded'],b['rounded']]),full['rounded']);np.testing.assert_array_equal(parent['sums'],before)
        self.assertEqual(int(parent['count']),1);self.assertEqual(int(child['count']),2)
        with self.assertRaises(ValueError):stream(v[2:],m[2:],p[2:],parent,**{**kw,'source_identity':'wrong'})

    def test_masked_or_future_values_cannot_change_earlier_outputs(self):
        v,m,p=self.fixture();kw={'groups':2,'heads':4,'dim':2};a,_,_=stream(v,m,p,**kw)
        changed=v.copy();changed[0]=words([1000]*4);changed[2]=words([-1000]*4);b,_,_=stream(changed,m,p,**kw)
        np.testing.assert_array_equal(a['rounded'],b['rounded'])
        changed[3]=words([1000]*4);c,_,_=stream(changed,m,p,**kw);np.testing.assert_array_equal(c['rounded'][:3],a['rounded'][:3])
        self.assertFalse(np.array_equal(c['rounded'][3],a['rounded'][3]))
        with self.assertRaises(ValueError):stream(v,m,p+1,**kw)
        bad=v.copy();bad[1,0]=0x7f80
        with self.assertRaises(ValueError):stream(bad,m,p,**kw)

    def test_npz_round_trip_can_resume_the_same_causal_state(self):
        v,m,p=self.fixture();kw={'groups':2,'heads':4,'dim':2,'source_identity':'fixed'}
        full,_,_=stream(v,m,p,**kw);a,parent,_=stream(v[:2],m[:2],p[:2],**kw)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.npz';np.savez(path,**parent)
            with np.load(path,allow_pickle=False) as f:restored={k:f[k].copy() for k in f.files}
            b,final,_=stream(v[2:],m[2:],p[2:],restored,**kw)
            np.testing.assert_array_equal(np.concatenate([a['rounded'],b['rounded']]),full['rounded'])
            self.assertEqual(int(final['count']),2);self.assertEqual(int(final['clock']),4)
            restored['geometry']=np.array([2,8,2])
            with self.assertRaises(ValueError):stream(v[2:],m[2:],p[2:],restored,**kw)

    def test_declared_conversion_preserves_inherited_double_rounding(self):
        just_above_midpoint=np.array([1+2**-8+2**-26],dtype=np.float64)
        self.assertGreater(just_above_midpoint[0],1+2**-8)
        self.assertEqual(words(just_above_midpoint)[0],0x3f80)

    def test_isolated_payload_imports_scoring_without_old_readers_or_neural_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in OWNED:shutil.copyfile(Path(__file__).resolve().parent/name,Path(tmp)/name)
            code='import sys;sys.path.insert(0,".");from encoder_atoms import guard;guard();import uniform_reader;import uniform_reader_study;from paired_fields import metrics;assert metrics([1.],[1.])["rms"]==0;assert not {"rank_one_tail","attention_bank","torch","diffusers","transformers"}.intersection(sys.modules)'
            subprocess.run([sys.executable,'-I','-c',code],cwd=tmp,check=True,capture_output=True,timeout=20)

    def test_failed_child_prevents_seal_and_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r={'sourceHashes':{}};r['requestSHA256']=identity(r);save(root/'request.json',r)
            with patch('uniform_reader_study.subprocess.run',side_effect=RuntimeError('partial')),patch('uniform_reader_study.seal') as seal,patch('uniform_reader_study.score') as score:
                with self.assertRaises(RuntimeError):execute(root)
                seal.assert_not_called();score.assert_not_called()

    def test_contribution_cross_term_direction(self):
        a=contribution([0],[.5],[1]);b=contribution([0],[-.5],[1])
        self.assertEqual(a['twiceBaseErrorDotCandidateMinusBase'],-1);self.assertEqual(b['twiceBaseErrorDotCandidateMinusBase'],1)
        self.assertEqual(a['energyIdentityRemainder'],0);self.assertEqual(b['energyIdentityRemainder'],0)


if __name__=='__main__':unittest.main()
