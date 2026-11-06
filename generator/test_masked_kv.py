import unittest
import torch
from masked_kv import block


class MaskedKVMechanics(unittest.TestCase):
    def test_zero_signs_remain_distinct_from_numerical_error(self):
        import numpy as np
        from masked_kv_study import field_metrics
        a=np.array([0.,-0.,1.]);b=np.array([-0.,-0.,1.])
        m=field_metrics(a,b,retained=True)
        self.assertEqual(m['changedValues'],0);self.assertEqual(m['signedZeroDefects'],1)
        self.assertEqual(m['BF16WordMismatches'],1)

    def test_masked_keys_only_and_all_queries_survive(self):
        from torch.nn import functional as F
        torch.manual_seed(22); b, n, h, d, kv, heads, inter = 1, 4, 6, 2, 1, 2, 7
        fields = {'input_layernorm': (h,), 'post_attention_layernorm': (h,),
            'self_attn.q_proj': (heads*d,h), 'self_attn.k_proj': (kv*d,h), 'self_attn.v_proj': (kv*d,h),
            'self_attn.o_proj': (h,heads*d), 'self_attn.q_norm': (d,), 'self_attn.k_norm': (d,),
            'mlp.gate_proj': (inter,h), 'mlp.up_proj': (inter,h), 'mlp.down_proj': (h,inter)}
        params={'model.layers.1.'+k+'.weight':torch.randn(v,dtype=torch.float64)for k,v in fields.items()}
        profile={'num_attention_heads':heads,'num_key_value_heads':kv,'head_dim':d,'intermediate_size':inter,
                 'num_key_value_groups':heads//kv,'rms_norm_eps':1e-6}
        mask=torch.tensor([[1,1,0,0]]); permitted=torch.tril(torch.ones(n,n,dtype=torch.bool))[None,None]&mask[:,None,None,:].bool()
        graph={'_rms_norm':lambda x,w,e:x*torch.rsqrt((x*x).mean(-1,keepdim=True)+e)*w,
               '_rotate_half':lambda x:torch.cat([-x[...,d//2:],x[...,:d//2]],-1),
               '_attention':lambda q,k,v,m,**kw:F.scaled_dot_product_attention(q,k.repeat_interleave(heads//kv,1),v.repeat_interleave(heads//kv,1),attn_mask=m).transpose(1,2).contiguous()}
        context=(permitted,False,None,torch.ones((1,n,d),dtype=torch.float64),torch.zeros((1,n,d),dtype=torch.float64),{})
        x=torch.randn((b,n,h),dtype=torch.float64);ports={}; other={}
        a=block(graph,x,params,profile,1,context,None,mask,'compact',lambda k,v:ports.__setitem__(k,v))
        z=block(graph,x,params,profile,1,context,None,mask,'zero',lambda k,v:other.__setitem__(k,v))
        torch.testing.assert_close(a,z,rtol=1e-12,atol=1e-12)
        self.assertEqual(a.shape,x.shape);self.assertEqual(ports['k'].stride(),(8,2,2,1))
        self.assertTrue(torch.equal(ports['k'][:,:,2:],torch.zeros_like(ports['k'][:,:,2:])))
        self.assertGreater(float(a[:,2:].abs().sum()),0)

    def test_empty_support_refuses(self):
        with self.assertRaises(ValueError):
            block({},torch.zeros(1,2,3),{}, {'num_attention_heads':1,'num_key_value_heads':1,'head_dim':3,'intermediate_size':3},1,(None,True,None,None,None,{}),None,torch.zeros(1,2),'compact',lambda *x:None)
