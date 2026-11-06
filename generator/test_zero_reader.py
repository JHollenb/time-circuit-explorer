import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from encoder_atoms import save, identity
from paired_fields import load
from zero_reader import zero_bank
from zero_reader_study import OWNED, execute, rank_contribution


class ZeroReaderMechanics(unittest.TestCase):
    def test_signed_reader_contribution_uses_rank_minus_zero(self):
        useful = rank_contribution([0.], [.5], [1.])
        self.assertEqual(useful['twiceZeroErrorDotRankMinusZero'], -1.)
        self.assertEqual(useful['rankErrorEnergy'], .25); self.assertEqual(useful['energyIdentityRemainder'], 0.)
        harmful = rank_contribution([0.], [-.5], [1.])
        self.assertEqual(harmful['twiceZeroErrorDotRankMinusZero'], 1.)
        self.assertEqual(harmful['rankErrorEnergy'], 2.25); self.assertEqual(harmful['energyIdentityRemainder'], 0.)
        with self.assertRaises(ValueError): rank_contribution([0.], [0.,1.], [1.])

    def test_bank_is_complete_positive_zero_bf16_and_authenticated(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = zero_bank(tmp, 'head')
            self.assertEqual(item['shape'], [1,512,4096]); self.assertEqual(item['logicalDtype'], 'torch.bfloat16')
            bits = np.load(item['path'], allow_pickle=False)
            self.assertEqual(bits.dtype, np.uint16); self.assertFalse(bits.any())
            value = load(item); self.assertFalse(np.signbit(value).any())
            bits[0,0,0] = 0x8000; np.save(item['path'],bits,allow_pickle=False)
            with self.assertRaises(ValueError): load(item)

    def test_isolated_payload_has_scoring_closure_without_deleted_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in OWNED: shutil.copyfile(Path(__file__).resolve().parent/name,Path(tmp)/name)
            self.assertFalse(any('rank_one' in name or 'attention_bank' in name for name in OWNED))
            code='import sys;sys.path.insert(0,".");from encoder_atoms import guard;guard();import zero_reader;import zero_reader_study;from paired_fields import metrics;assert metrics([1.],[1.])["rms"]==0;assert not {"rank_one_tail","attention_bank","torch","diffusers","transformers"}.intersection(sys.modules)'
            subprocess.run([sys.executable,'-I','-c',code],cwd=tmp,check=True,capture_output=True,timeout=20)

    def test_no_seal_or_scoring_after_failed_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r={'sourceHashes':{}};r['requestSHA256']=identity(r);save(root/'request.json',r);steps=[]
            def child(cmd,**kwargs):
                steps.append(cmd[-2])
                if cmd[-2]=='encoder':raise RuntimeError('partial candidate')
            with patch('zero_reader_study.subprocess.run',side_effect=child),patch('zero_reader_study.seal') as seal,patch('zero_reader_study.score') as score:
                with self.assertRaises(RuntimeError):execute(root)
                seal.assert_not_called();score.assert_not_called()
            self.assertEqual(steps,['reader','encoder'])


if __name__=='__main__':unittest.main()
