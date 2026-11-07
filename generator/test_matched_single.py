import unittest


class MatchedSingleTests(unittest.TestCase):
    def test_capture_and_admitted_product(self):
        import torch
        import torch.nn.functional as F
        import diffusers.models.transformers.transformer_flux2 as native
        from matched_single import replay
        from gate_first_trajectory import source_norms
        with torch.random.fork_rng(devices=[]),torch.inference_mode():
            torch.manual_seed(73);b=native.Flux2SingleTransformerBlock(16,2,8).double().eval().requires_grad_(False)
            source={'text':torch.randn(1,3,16,dtype=torch.float64),'image':torch.randn(1,4,16,dtype=torch.float64),'mod':torch.randn(1,48,dtype=torch.float64)*.02,
                'cosine':torch.ones(7,8,dtype=torch.float64),'sine':torch.zeros(7,8,dtype=torch.float64)}
            norms=source_norms([b]);old=(F.linear,F.silu)
            sparse,sp=replay('sparse',b,source,native,norms);dense,de=replay('dense',b,source,native)
            self.assertEqual(old,(F.linear,F.silu));self.assertEqual(len(sp['operations']),3);self.assertEqual(len(de['operations']),2)
            from gate_first_trajectory import single as raw_sparse
            from image_partition_moments import single as raw_dense
            args=(b,source['text'],source['image'],source['mod'],(source['cosine'],source['sine']),native,native.dispatch_attention_fn,lambda kind,a,b:a@b,None)
            references=[raw_sparse(*args,lambda index,values,operation:operation(),0,norms,lambda index,values:None),raw_dense(*args)]
            for captured,reference in zip([sparse,dense],references):
                for name,value in zip(['textOut','imageOut','alpha','textRead'],reference):torch.testing.assert_close(captured[name],value,rtol=0,atol=0)
            for got in [sparse,dense]:
                support=sparse['selection_support'] if got is sparse else torch.arange(48)
                torch.testing.assert_close(got['product'],got['activation'].index_select(-1,support)*got['upRaw'],rtol=0,atol=0)
                torch.testing.assert_close(got['exit'],torch.cat([source['text'],source['image']],1)+got['gate']*got['preResidualUpdate'],rtol=0,atol=0)
            for name in ['normalized','queryRaw','keyRaw','valueRaw','gateRaw','imageAttention','textRead']:
                torch.testing.assert_close(sparse[name],dense[name],rtol=1e-12,atol=1e-12)

    def test_signed_cancellation_and_remainder(self):
        import torch
        from matched_single import common_write,energy_panel
        p=torch.ones(1,516,4,dtype=torch.float64);d=torch.tensor([[2.,3.,1.,-1.],[0.,0.,1.,-1.]],dtype=torch.float64);s=torch.tensor([0,1]);g=torch.tensor([[[.5,2.]]],dtype=torch.float64)
        got=common_write(p,d,s,lambda name,a,b:a@b);rows,channels=energy_panel(p,d,s,got['fullWrite'],got['selectedWrite'],g)
        self.assertEqual(rows[0]['omittedWriteEnergy'],0);self.assertLess(rows[0]['omittedCrossChannelEnergy'],0)
        self.assertEqual(len(rows),6);self.assertEqual(set(channels),{'all','text','image'})
        delta=-g*got['omittedWrite']+torch.ones_like(got['fullWrite'])*.125;remainder=delta+g*got['omittedWrite']
        torch.testing.assert_close(remainder,torch.full_like(remainder,.125),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
