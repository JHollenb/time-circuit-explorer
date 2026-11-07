import unittest


class MeanUpTests(unittest.TestCase):
    def test_independent_stream_completion_and_full_support(self):
        import torch
        import torch.nn.functional as F
        import diffusers.models.transformers.transformer_flux2 as native
        from mean_up_completion import replay
        from gate_first_trajectory import source_norms
        from image_partition_moments import single as dense
        with torch.random.fork_rng(devices=[]),torch.inference_mode():
            torch.manual_seed(74)
            b=native.Flux2SingleTransformerBlock(16,2,8).double().eval().requires_grad_(False)
            source={'text':torch.randn(1,3,16,dtype=torch.float64)+2,'image':torch.randn(1,4,16,dtype=torch.float64)-2,
                'mod':torch.randn(1,48,dtype=torch.float64)*.02,'cosine':torch.ones(7,8,dtype=torch.float64),'sine':torch.zeros(7,8,dtype=torch.float64)}
            norms=source_norms([b]);got,trace=replay(b,source,native,norms)
            rows=torch.cat([source['text'],source['image']],1)
            sh,sc,ga=native.Flux2Modulation.split(source['mod'],1)[0]
            z=(1+sc)*b.norm(rows)+sh
            q,k,v,g,u=F.linear(z,b.attn.to_qkv_mlp_proj.weight).split([16,16,16,48,48],-1)
            means=torch.cat([z[:,:3].float().sum(1,keepdim=True)/3,z[:,3:].float().sum(1,keepdim=True)/4],1)
            torch.testing.assert_close(got['streamMeanFP32'],means,rtol=2e-7,atol=2e-7)
            anchors=F.linear(got['streamMean'],b.attn.to_qkv_mlp_proj.weight[96:])
            wanted=torch.empty_like(u);support=got['selection_support'].tolist()
            for i in range(7):
                for j in range(48):wanted[0,i,j]=u[0,i,j] if j in support else anchors[0,0 if i<3 else 1,j]
            torch.testing.assert_close(got['completedUp'],wanted,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(got['product'],F.silu(g)*wanted,rtol=1e-12,atol=1e-12)
            ref=rows+ga*F.linear(torch.cat([got['attentionRead'],F.silu(g)*wanted],-1),b.attn.to_out.weight)
            torch.testing.assert_close(got['exit'],ref,rtol=1e-12,atol=1e-12)
            full,full_trace=replay(b,source,native,norms,count=48)
            ref=dense(b,source['text'],source['image'],source['mod'],(source['cosine'],source['sine']),native,native.dispatch_attention_fn,lambda kind,a,b:a@b,None)
            torch.testing.assert_close(full['exit'],torch.cat(ref[:2],1),rtol=1e-12,atol=1e-12)
            self.assertEqual([x['name'] for x in trace['operations']],['QKVG','USelected','UAnchor','outFull'])
            self.assertEqual(len(full_trace['products']),5)


if __name__=='__main__':unittest.main()
