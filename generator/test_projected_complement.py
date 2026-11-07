import unittest
import torch
from projected_complement import project_cache, projected_read
from sparse_text_kv import complement


class ProjectedComplement(unittest.TestCase):
    def test_composition_matches_full_changed_attention_then_full_Wo(self):
        q = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2) / 13
        k = torch.sin(torch.arange(16, dtype=torch.float64)).reshape(2, 4, 2)
        v = torch.cos(torch.arange(16, dtype=torch.float64)).reshape(2, 4, 2)
        wo = torch.arange(20, dtype=torch.float64).reshape(5, 4) / 11
        support = [0, 2]
        m, z, n = complement(q, k, v, support)
        calls = []
        def linear(x, w):
            calls.append((tuple(x.shape), tuple(w.shape), w.storage_offset()))
            return torch.nn.functional.linear(x, w)
        nt = project_cache(n, wo, linear)
        kb, vb = k.clone(), v.clone()
        kb[:, support] *= -3
        vb[:, support] += 4
        result = projected_read(q, kb[:, support], vb[:, support], m, z, nt, wo, linear)
        # Independent full-bank reference, followed by one unsliced full Wo.
        full_heads = torch.softmax(q @ kb.transpose(-1, -2) / 2 ** .5, -1) @ vb
        full = full_heads.permute(1, 0, 2).reshape(3, 4) @ wo.T
        torch.testing.assert_close(result['projectionFP64'], full, atol=1e-12, rtol=1e-12)
        self.assertEqual(calls, [((3, 2), (5, 2), 0), ((3, 2), (5, 2), 2),
                                 ((2, 2), (5, 2), 0), ((2, 2), (5, 2), 2)])

    def test_extreme_new_peak_scales_projected_cache_and_partition_together(self):
        q = torch.ones((1, 1, 1), dtype=torch.float64)
        k = torch.tensor([[[0.], [1.]]], dtype=torch.float64)
        v = torch.tensor([[[2.], [3.]]], dtype=torch.float64)
        wo = torch.tensor([[2.], [-4.]], dtype=torch.float64)
        m, z, n = complement(q, k, v, [1])
        nt = project_cache(n, wo, torch.nn.functional.linear)
        result = projected_read(q, torch.tensor([[[1000.]]], dtype=torch.float64), torch.tensor([[[9.]]], dtype=torch.float64),
                                m, z, nt, wo, torch.nn.functional.linear)
        torch.testing.assert_close(result['projectionFP64'], torch.tensor([[18., -36.]], dtype=torch.float64), rtol=0, atol=0)
        self.assertEqual(result['cacheScale'].item(), 0)


if __name__ == '__main__':
    unittest.main()
