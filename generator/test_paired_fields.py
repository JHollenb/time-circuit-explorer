import tempfile
from pathlib import Path
import unittest
import numpy as np
from paired_fields import retain, load, payload, paired


class PairedFieldMechanics(unittest.TestCase):
    def test_unsigned_prompt_difference_keeps_its_sign(self):
        ca = np.array([255, 0], dtype=np.uint8); cb = np.array([0, 255], dtype=np.uint8)
        ra = np.array([254, 1], dtype=np.uint8); rb = np.array([1, 254], dtype=np.uint8)
        result = paired(ca, cb, ra, rb)
        self.assertEqual(result['rms'], 2)
        self.assertEqual(result['referenceRms'], 253)
        self.assertEqual(result['mae'], 2)
        self.assertAlmostEqual(result['relativeRms'], 2 / 253)

    def test_zero_response_denominator_stays_undefined(self):
        result = paired([0, 1], [2, 3], [4, 5], [4, 5])
        self.assertIsNone(result['relativeRms'])
        self.assertIsNone(result['cosine'])
        self.assertEqual(result['referenceRms'], 0)
        self.assertEqual(result['rms'], 2)

    def test_bf16_field_round_trip_preserves_words_and_negative_zero(self):
        import torch
        x = torch.tensor([0., -0., 1., -2.], dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as tmp:
            item = retain(tmp, 'state', x)
            y = load(item, 'cpu')
            self.assertTrue(torch.equal(x.view(torch.uint16), y.view(torch.uint16)))
            self.assertEqual(payload(x), payload(y))
            self.assertEqual(item['shape'], [4])
            self.assertTrue(np.signbit(load(item)[1]))
            with self.assertRaises(ValueError): retain(tmp, 'state', x)
            Path(item['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError): load(item)

    def test_two_blocks_keep_padding_queries_and_consume_their_own_updates(self):
        import torch
        from torch.nn import functional as F
        from masked_kv import block
        torch.manual_seed(31); width, heads, kv, dim, inter, length = 6, 2, 1, 2, 7, 4
        shapes = {'input_layernorm': (width,), 'post_attention_layernorm': (width,), 'self_attn.q_proj': (heads*dim,width),
            'self_attn.k_proj': (kv*dim,width), 'self_attn.v_proj': (kv*dim,width), 'self_attn.o_proj': (width,heads*dim),
            'self_attn.q_norm': (dim,), 'self_attn.k_norm': (dim,), 'mlp.gate_proj': (inter,width), 'mlp.up_proj': (inter,width), 'mlp.down_proj': (width,inter)}
        params = {f'model.layers.{i}.'+n+'.weight': torch.randn(shape, dtype=torch.float64) for i in [1, 2] for n, shape in shapes.items()}
        profile = {'num_attention_heads': heads, 'num_key_value_heads': kv, 'head_dim': dim, 'intermediate_size': inter, 'num_key_value_groups': 2, 'rms_norm_eps': 1e-6}
        mask = torch.tensor([[1, 1, 0, 0]])
        support = torch.tril(torch.ones(length,length,dtype=torch.bool))[None,None] & mask[:,None,None,:].bool()
        graph = {'_rms_norm': lambda x,w,e: x*torch.rsqrt(x.square().mean(-1,keepdim=True)+e)*w,
            '_rotate_half': lambda x: torch.cat([-x[...,1:],x[...,:1]],-1),
            '_attention': lambda q,k,v,m,**kw: F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),attn_mask=m).transpose(1,2).contiguous()}
        context = (support,False,None,torch.ones(1,length,dim,dtype=torch.float64),torch.zeros(1,length,dim,dtype=torch.float64),{})
        x = torch.randn(1,length,width,dtype=torch.float64); compact = x.clone(); full = x.clone(); first = None
        for i in [1, 2]:
            compact = block(graph,compact,params,profile,i,context,None,mask,'compact',lambda *x:None)
            full = block(graph,full,params,profile,i,context,None,mask,'zero',lambda *x:None)
            torch.testing.assert_close(compact,full,rtol=1e-12,atol=1e-12)
            self.assertEqual(compact.shape,x.shape)
            self.assertFalse(torch.equal(compact[:,2:],x[:,2:]))
            if i == 1: first = compact.clone()
        refreshed = block(graph,x,params,profile,2,context,None,mask,'compact',lambda *x:None)
        self.assertFalse(torch.equal(compact,refreshed))
        self.assertFalse(torch.equal(compact,first))

    def test_corrupt_candidate_stops_reference_release(self):
        from unittest.mock import patch
        from encoder_atoms import save, sha, identity
        from complete_masked_study import execute
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r = {'sourceHashes': {}}; r['requestSHA256'] = identity(r); save(root / 'request.json', r)
            item = retain(root / 'arrays/candidate/A', 'H1', np.ones((2,3)))
            png = root / 'candidate-A.png'; png.write_bytes(b'synthetic fixture')
            save(root / 'encoder-candidate.json', {'fields': {'A': {'H1': item}}})
            save(root / 'images-candidate.json', {'fields': {}, 'images': {'A': {'path': str(png), 'sha256': sha(png)}}})
            Path(item['path']).write_bytes(b'corrupted before seal')
            launches = []
            with patch('complete_masked_study.subprocess.run', side_effect=lambda cmd,**kw: launches.append(cmd[-2:])):
                with self.assertRaises(ValueError): execute(root)
            self.assertEqual(len(launches), 3)
            self.assertEqual(launches[-1][-1], 'candidate')
            self.assertFalse((root / 'candidate-seal.json').exists())
