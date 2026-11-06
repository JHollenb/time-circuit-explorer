import tempfile
from pathlib import Path
import subprocess
import shutil
import sys
import unittest
from unittest.mock import patch
import numpy as np
from encoder_atoms import read, save, sha, identity, words
from paired_fields import load
from raw_rank_one_reader import bank_field
from raw_rank_one_study import inherited_clock, execute, EXPERIMENT


class RawReaderIntegrationMechanics(unittest.TestCase):
    def test_complete_bank_preserves_bf16_words_and_rejects_unqualified_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = np.zeros((512, 4096), dtype=np.float64); raw[0, :4] = [1, -2, .5, -0.0]; bits = words(raw)
            item = bank_field(tmp, 'head', raw, bits)
            self.assertEqual(item['logicalDtype'], 'torch.bfloat16'); self.assertEqual(item['shape'], [1, 512, 4096])
            self.assertTrue(np.signbit(load(item)[0, 0, 3])); self.assertEqual(load(item)[0, 0, 1], -2)
            for bad_raw, bad_bits in [(raw[:511], bits[:511]), (raw, bits.astype(np.int32))]:
                with self.assertRaises(ValueError): bank_field(tmp, 'invalid', bad_raw, bad_bits)
            bad = bits.copy(); bad[0, 0] ^= 1
            with self.assertRaises(ValueError): bank_field(tmp, 'wrong-round', raw, bad)
            raw[0, 0] = np.nan
            with self.assertRaises(ValueError): bank_field(tmp, 'nonfinite', raw, bits)
            Path(item['path']).write_bytes(b'corrupt head bank')
            with self.assertRaises(ValueError): load(item)

    def test_inherited_clock_keeps_its_own_request_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'clock-table.json'
            save(p, {'requestSHA256': 'original-clock-request', 'contract': {'checkpointSHA256': 'original-model'}})
            source = {'clockTableSHA256': sha(p), 'clockParentRequestSHA256': 'original-clock-request',
                'imageAssets': {'transformer/diffusion_pytorch_model.safetensors': 'original-model'}}
            self.assertEqual(inherited_clock(tmp, source)['requestSHA256'], 'original-clock-request')
            for change in [{'clockTableSHA256': 'wrong-bytes'}, {'clockParentRequestSHA256': 'new-experiment-request'},
                {'imageAssets': {'transformer/diffusion_pytorch_model.safetensors': 'other-model'}}]:
                with self.assertRaises(ValueError): inherited_clock(tmp, {**source, **change})

    def test_reader_import_closure_needs_no_neural_runtime(self):
        code = 'import sys; sys.path.insert(0, '+repr(str(Path(__file__).resolve().parent))+'); from encoder_atoms import guard; guard(); import raw_rank_one_reader; from rank_one_tail import serve; assert not {"torch","transformers","diffusers"}.intersection(sys.modules)'
        subprocess.run([sys.executable, '-c', code], check=True, timeout=20, capture_output=True)

    def test_scoring_helper_is_staged_and_imports_in_isolation(self):
        self.assertIn('masked_kv_study.py', EXPERIMENT)
        code = 'import sys; sys.path.insert(0,"."); from paired_fields import metrics; assert metrics([1.],[1.])["rms"] == 0'
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['encoder_atoms.py', 'paired_fields.py']:
                shutil.copyfile(Path(__file__).resolve().parent / name, Path(tmp) / name)
            missing = subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(missing.returncode, 0); self.assertIn('masked_kv_study', missing.stderr)
            shutil.copyfile(Path(__file__).resolve().parent / 'masked_kv_study.py', Path(tmp) / 'masked_kv_study.py')
            subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp, check=True, capture_output=True, timeout=20)

    def test_encoder_failure_cannot_release_targets_to_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); r = {'sourceHashes': {}}; r['requestSHA256'] = identity(r); save(root / 'request.json', r)
            launches = []
            def child(cmd, **kwargs):
                launches.append(cmd[-2])
                if cmd[-2] == 'encoder': raise RuntimeError('incomplete candidate')
            with patch('raw_rank_one_study.subprocess.run', side_effect=child), patch('raw_rank_one_study.score') as score:
                with self.assertRaises(RuntimeError): execute(root)
                score.assert_not_called()
            self.assertEqual(launches, ['reader', 'encoder']); self.assertFalse((root / 'candidate-seal.json').exists())


if __name__ == '__main__': unittest.main()
