import unittest
from gate_first_trajectory import mechanics


class GateFirstTrajectoryTests(unittest.TestCase):
    def test_independent_composition_and_last_use(self):
        result=mechanics()
        self.assertEqual(result['pretrainedModelEvaluations'],0)
        self.assertEqual(len(result['maxAbsTwoStreamErrors']),4)
        self.assertLess(max(result['maxAbsTwoStreamErrors']),4e-7)
        self.assertIn(['QKVG'],result['physicalMaps'])
        self.assertIn(['USelected','outSelected'],result['physicalMaps'])
        self.assertIn(['KV','QG'],result['physicalMaps'])

    def test_complete_support_whole_model_own_clocks(self):
        import torch
        import diffusers.models.transformers.transformer_flux2 as native
        from gate_first_trajectory import denoise,source_norms
        from image_partition_moments import denoise as dense
        with torch.random.fork_rng(devices=[]),torch.inference_mode():
            torch.manual_seed(720)
            model=native.Flux2Transformer2DModel(in_channels=4,num_layers=1,num_single_layers=2,num_attention_heads=2,attention_head_dim=8,
                joint_attention_dim=12,timestep_guidance_channels=16,axes_dims_rope=(2,2,2,2),guidance_embeds=False).double().eval().requires_grad_(False)
            candidate=torch.randn(1,4,4,dtype=torch.float64);reference=candidate.clone();norms=source_norms(model.single_transformer_blocks)
            embedding=torch.randn(1,3,12,dtype=torch.float64);selected=[]
            def maps(index,values,operation):return operation()
            for clock in [.88398,.71749]:
                args={'encoder_hidden_states':embedding,'timestep':torch.tensor([clock],dtype=torch.float64),'img_ids':torch.arange(16).reshape(1,4,4),'txt_ids':torch.arange(12).reshape(1,3,4),'return_dict':False}
                wanted=[]
                ref=dense(model,{'hidden_states':reference,**args},native,native.dispatch_attention_fn,lambda n,v:None,
                    lambda kind,index,bt,bi,at,ai,alpha,read:wanted.append((at,ai)),maps,lambda kind,a,b:a@b,lambda v:None)[0]
                seen=[]
                def block(kind,index,bt,bi,at,ai,alpha,read):
                    wt,wi=wanted[len(seen)];torch.testing.assert_close(ai,wi,rtol=4e-6,atol=4e-7)
                    if wt is not None:torch.testing.assert_close(at,wt,rtol=4e-6,atol=4e-7)
                    seen.append(True)
                got=denoise(model,{'hidden_states':candidate,**args},native,native.dispatch_attention_fn,lambda n,v:None,block,maps,
                    lambda kind,a,b:a@b,lambda v:None,norms,lambda index,s:selected.append(s['support']),count=48)[0]
                torch.testing.assert_close(got,ref,rtol=4e-6,atol=4e-7);self.assertEqual(len(seen),3)
                candidate=candidate-.125*got;reference=reference-.125*ref
            self.assertEqual(len(selected),4)


if __name__=='__main__':unittest.main()
