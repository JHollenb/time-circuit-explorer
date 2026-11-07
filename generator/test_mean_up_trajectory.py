"""Independent tiny live-domain and own-state mechanics, no pretrained weights."""
import unittest


def mechanics():
    import torch
    import torch.nn.functional as F
    import diffusers.models.transformers.transformer_flux2 as native
    from mean_up_trajectory import single,denoise
    from gate_first_trajectory import source_norms
    from image_partition_moments import denoise as dense
    with torch.random.fork_rng(devices=[]),torch.inference_mode():
        torch.manual_seed(75);b=native.Flux2SingleTransformerBlock(16,2,8).double().eval().requires_grad_(False)
        t=torch.randn(1,3,16,dtype=torch.float64);i=torch.randn(1,4,16,dtype=torch.float64);mod=torch.randn(1,48,dtype=torch.float64)*.02
        angles=torch.randn(7,8,dtype=torch.float64)*.2;rope=(angles.cos(),angles.sin());norms=source_norms([b]);maps=lambda index,values,operation:operation();errors=[]
        for terminal in [False,True]:
            for count in [12,48]:
                got=single(b,t,i,mod,rope,native,native.dispatch_attention_fn,lambda kind,a,b:a@b,None,maps,0,norms,lambda index,s:None,count,terminal)
                if terminal:
                    assert got['imageOut'].stride()==(7*16,16,1)
                else:
                    from mean_up_completion import replay as frozen74
                    prior,_=frozen74(b,{'text':t,'image':i,'mod':mod,'cosine':rope[0],'sine':rope[1]},native,norms,count=count)
                    for name,value in got.items():torch.testing.assert_close(value,prior[name],rtol=0,atol=0)
                rows=torch.cat([t,i],1);sh,sc,ga=native.Flux2Modulation.split(mod,1)[0];z=(1+sc)*b.norm(rows)+sh
                q,k,v,g,u=F.linear(z,b.attn.to_qkv_mlp_proj.weight).split([16,16,16,48,48],-1)
                if terminal:z,g,u,q=z[:,3:],g[:,3:],u[:,3:],q[:,3:]
                means=z.float().sum(1,keepdim=True)/4 if terminal else torch.cat([z[:,:3].float().sum(1,keepdim=True)/3,z[:,3:].float().sum(1,keepdim=True)/4],1)
                torch.testing.assert_close(got['streamMeanFP32'],means,rtol=2e-7,atol=2e-7)
                anchors=F.linear(got['streamMean'],b.attn.to_qkv_mlp_proj.weight[96:]);support=got['selection_support'].tolist()
                completed=torch.empty_like(u)
                for row in range(u.shape[1]):
                    for channel in range(48):completed[0,row,channel]=u[0,row,channel] if channel in support else anchors[0,0 if terminal or row<3 else 1,channel]
                product=F.silu(g)*completed
                q=b.attn.norm_q(q.unflatten(-1,(2,-1)));k=b.attn.norm_k(k.unflatten(-1,(2,-1)));v=v.unflatten(-1,(2,-1))
                q=native.apply_rotary_emb(q,(rope[0][3:],rope[1][3:]) if terminal else rope,sequence_dim=1)
                k=native.apply_rotary_emb(k,rope,sequence_dim=1)
                ih=native.dispatch_attention_fn(q if terminal else q[:,3:],k,v,attn_mask=None,backend=None,parallel_config=None).flatten(2,3)
                if terminal:h=ih;incoming=rows[:,3:]
                else:
                    qt=q[:,:3].transpose(1,2).float();kt=k.transpose(1,2).float();vt=v[:,:3].transpose(1,2).float();scale=8**-.5
                    scores=qt@kt[:,:,3:].transpose(-1,-2)*scale;mean=scores.mean(-1);li=__import__('math').log(4)+mean+(scores-mean.unsqueeze(-1)).square().mean(-1)*.5
                    tt=qt@kt[:,:,:3].transpose(-1,-2)*scale;alpha=torch.sigmoid(tt.logsumexp(-1)-li);th=((tt.softmax(-1)@vt)*alpha.unsqueeze(-1)).transpose(1,2).flatten(2,3).double()
                    h=torch.cat([th,ih],1);incoming=rows
                ref=incoming+ga*F.linear(torch.cat([h,product],-1),b.attn.to_out.weight)
                actual=got['imageOut'] if terminal else torch.cat([got['textOut'],got['imageOut']],1)
                torch.testing.assert_close(actual,ref,rtol=4e-6,atol=4e-7);errors.append(float((actual-ref).abs().max()))
                torch.testing.assert_close(got['completedUp'],completed,rtol=1e-12,atol=1e-12)
        model=native.Flux2Transformer2DModel(in_channels=4,num_layers=1,num_single_layers=2,num_attention_heads=2,attention_head_dim=8,
            joint_attention_dim=12,timestep_guidance_channels=16,axes_dims_rope=(2,2,2,2),guidance_embeds=False).double().eval().requires_grad_(False)
        candidate=torch.randn(1,4,4,dtype=torch.float64);reference=candidate.clone();norms=source_norms(model.single_transformer_blocks);embedding=torch.randn(1,3,12,dtype=torch.float64)
        for clock in [.88398,.71749]:
            args={'encoder_hidden_states':embedding,'timestep':torch.tensor([clock],dtype=torch.float64),'img_ids':torch.arange(16).reshape(1,4,4),'txt_ids':torch.arange(12).reshape(1,3,4),'return_dict':False}
            ref=dense(model,{'hidden_states':reference,**args},native,native.dispatch_attention_fn,lambda n,v:None,lambda *a:None,maps,lambda kind,a,b:a@b,lambda v:None)[0]
            got=denoise(model,{'hidden_states':candidate,**args},native,native.dispatch_attention_fn,lambda n,v:None,lambda *a:None,maps,lambda kind,a,b:a@b,lambda v:None,norms,lambda index,s:None,
                lambda index,block,t,i,mod,rope,operation:operation(),count=48)[0]
            torch.testing.assert_close(got,ref,rtol=4e-6,atol=4e-7);candidate=candidate-.125*got;reference=reference-.125*ref
        return {'schema':'tce-mean-trajectory-toy-v1','maxAbsLocalErrors':errors,'wholeOwnClocks':2,'pretrainedModelEvaluations':0,
                'scope':'Independent tiny full/quarter Up streammeans, selected overwrites, direct text momentattention, terminal image-only and fullsupportwholemodel twoownclocks. FP64toy/FP32mean policy, not pretrainedBF16parity.'}


class MeanTrajectoryTests(unittest.TestCase):
    def test_live_domains_and_own_clocks(self):
        r=mechanics();self.assertEqual(r['wholeOwnClocks'],2);self.assertEqual(len(r['maxAbsLocalErrors']),4);self.assertEqual(r['pretrainedModelEvaluations'],0)


if __name__=='__main__':unittest.main()
