import unittest
import torch
import subprocess
import sys
from pathlib import Path
from sparse_text_kv import complement, update, rows


class SparseTextKV(unittest.TestCase):
    def test_fresh_actor_denies_arrays_even_inside_its_root(self):
        code = '''
import pathlib,tempfile
from sparse_text_kv import quarantine
with tempfile.TemporaryDirectory() as name:
 r=pathlib.Path(name).resolve()
 for n in ['allowed.npy','old-full-kv.npy','target.json']:(r/n).write_bytes(b'fixture')
 seen=quarantine(r,[r/'allowed.npy'],[r/'target.json'])
 assert (r/'allowed.npy').read_bytes()==b'fixture'
 for n in ['old-full-kv.npy','target.json']:
  try:(r/n).read_bytes()
  except PermissionError:pass
  else:raise AssertionError('Undeclared old array or target admitted')
 (r/'new.npy').write_bytes(b'own output')
 assert (r/'new.npy').read_bytes()==b'own output'
 assert list(seen)==[str(r/'allowed.npy')]
'''
        subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parent, check=True, capture_output=True)

    def test_partition_update_matches_full_changed_bank(self):
        q = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 9
        k = torch.sin(torch.arange(40, dtype=torch.float64)).reshape(2, 5, 4)
        v = torch.cos(torch.arange(40, dtype=torch.float64)).reshape(2, 5, 4)
        support = [1, 3]
        m, z, n = complement(q, k, v, support)
        kb, vb = k.clone(), v.clone()
        kb[:, support] *= -3
        vb[:, support] += 7
        result = update(q, kb[:, support], vb[:, support], m, z, n)
        exact = torch.softmax(q @ kb.transpose(-1, -2) / 2, -1) @ vb
        torch.testing.assert_close(result['H_FP64'], exact, rtol=1e-13, atol=1e-13)
        self.assertTrue(bool((result['denominator'] > 0).all()))

    def test_extreme_new_peak_rescales_numerator_and_partition(self):
        q = torch.ones((1, 1, 1), dtype=torch.float64)
        k = torch.tensor([[[0.], [1.]]], dtype=torch.float64)
        v = torch.tensor([[[2.], [3.]]], dtype=torch.float64)
        m, z, n = complement(q, k, v, [1])
        result = update(q, torch.tensor([[[1000.]]], dtype=torch.float64), torch.tensor([[[9.]]], dtype=torch.float64), m, z, n)
        self.assertEqual(result['H_FP64'].item(), 9)
        self.assertEqual(result['cacheScale'].item(), 0)

    def test_bf16_writer_preserves_operation_order(self):
        r = torch.tensor([[[1., 2., 3.]]], dtype=torch.bfloat16)
        a, m = r * 3, r / 3
        ga, gm = torch.ones_like(r) / 7, torch.ones_like(r) / 5
        u = r + ga * a
        expected = u + gm * m
        self.assertTrue(torch.equal(rows(r, a, m, ga, gm).view(torch.int16), expected.view(torch.int16)))


if __name__ == '__main__':
    unittest.main()
