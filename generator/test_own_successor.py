"""Tiny independent dense/score-moment, scalar operand and typed references."""
import unittest


def specimen():
    import torch
    import diffusers.models.transformers.transformer_flux2 as native
    block=native.Flux2SingleTransformerBlock(4,1,4,mlp_ratio=3.,eps=1e-6,bias=False).bfloat16().eval().requires_grad_(False)
    angles=torch.arange(16,dtype=torch.float32).reshape(8,2)/13
    source={'text':torch.randn(1,4,4).bfloat16(),'image':torch.randn(1,4,4).bfloat16(),
            'mod':(torch.randn(1,12)*.2).bfloat16(),
            'cosine':angles.cos().repeat_interleave(2,-1),'sine':angles.sin().repeat_interleave(2,-1)}
    return native,block,source


def attention(q,k,v,**kwargs):
    import torch
    return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)


def mechanics():
    import numpy as np
    import torch
    from own_successor import compute,abis,work
    from own_successor_analysis import accounts,summarize
    with torch.random.fork_rng(devices=[]),torch.inference_mode():
        torch.manual_seed(87);native,block,s=specimen();a=block.attn;calls=[];moments=[];qrs=[];joins=[];physical=[]
        rng=torch.get_rng_state();nrng=np.random.get_state()
        def mm(name,x,w):calls.append((name,list(x.shape),list(w.shape)));return x@w
        def prod(name,x,w):moments.append((name,list(x.shape),list(w.shape)));return x@w
        def qr(name,x):qrs.append(name);return torch.linalg.qr(x,mode='reduced')
        got,trace=compute(block,s,native,attention,lambda name,x,w:torch.nn.functional.linear(x,w),prod,mm,qr,
                           lambda x:joins.append(x.clone()),lambda raw,o,go,v:None,
                           lambda w:physical.append(w.clone()),4,4)
        assert torch.equal(rng,torch.get_rng_state())
        after=np.random.get_state();assert nrng[0]==after[0] and np.array_equal(nrng[1],after[1]) and nrng[2:]==after[2:]
        rows=torch.cat([s['text'],s['image']],1);shift,scale,gate=native.Flux2Modulation.split(s['mod'],1)[0]
        z=(1+scale)*block.norm(rows)+shift
        assert torch.equal(z,got['normalized']) and torch.equal(physical[0],a.to_qkv_mlp_proj.weight[:12])
        raw=torch.nn.functional.linear(z,a.to_qkv_mlp_proj.weight[:12]);q,k,v=raw.split(4,-1)
        q=native.apply_rotary_emb(a.norm_q(q.unflatten(-1,(1,4))),(s['cosine'],s['sine']),sequence_dim=1)
        k=native.apply_rotary_emb(a.norm_k(k.unflatten(-1,(1,4))),(s['cosine'],s['sine']),sequence_dim=1);v=v.unflatten(-1,(1,4))
        image=attention(q[:,4:],k,v).flatten(2,3)
        # Independent score-grid population mean/variance, rather than key covariance.
        qt=q[:,:4].transpose(1,2).float();kt=k.transpose(1,2).float();vt=v[:,:4].transpose(1,2).float()
        scores=qt@kt.transpose(-1,-2)*.5;si=scores[...,4:];st=scores[...,:4]
        mean=si.mean(-1);variance=(si-mean[...,None]).square().mean(-1)
        alpha=torch.sigmoid(st.logsumexp(-1)-(np.log(4)+mean+variance*.5))
        text=((st.softmax(-1)@vt)*alpha[...,None]).transpose(1,2).flatten(2,3).bfloat16()
        torch.testing.assert_close(got['textRead'].float(),text.float(),rtol=.015625,atol=.00390625)
        gu=(z.double()@a.to_qkv_mlp_proj.weight[12:].double().T).float().bfloat16()
        product=a.mlp_act_fn(gu)
        assert torch.equal(got['gu_gateRaw'],gu[...,:12]) and torch.equal(got['gu_upRaw'],gu[...,12:])
        read=torch.cat([text,image],1);joined=torch.cat([read,product],-1)
        update=(joined.double()@a.to_out.weight.double().T).float().bfloat16()
        expected=rows+gate*update
        torch.testing.assert_close(got['exit'].float(),expected.float(),rtol=.015625,atol=.0078125)
        assert len(calls)==17 and len(qrs)==4 and len(moments)==5 and len(joins)==1
        assert {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in got.items()}==abis(4,12,4,4,4,1)
        assert trace['restoredQKVViews']['key']['storageOffset']==4 and trace['restoredQKVViews']['key']['stride']==[288,36,1]
        # Perturb the incoming state; normalization and the next read must change.
        changed={**s,'text':s['text'].clone()};changed['text'][0,0,0]+=1
        later,_=compute(block,changed,native,attention,lambda name,x,w:torch.nn.functional.linear(x,w),lambda name,x,w:x@w,
                         lambda name,x,w:x@w,lambda name,x:torch.linalg.qr(x,mode='reduced'),lambda x:None,lambda *args:None,lambda w:None,4,4)
        assert not torch.equal(later['normalized'],got['normalized']) and not torch.equal(later['attentionRead'],got['attentionRead'])
        ci=rows.double().numpy();co=got['exit'].double().numpy();pi=ci+.125;po=expected.double().numpy()-.0625
        result=summarize(accounts(ci,co,pi,po,4))
        for k in ['energyAccountRemainder','centeredEnergyAccountRemainder','identityClosureEnergy']:
            np.testing.assert_allclose(result['signedAccount'][k],0,rtol=0,atol=1e-12)
        costs=work();assert costs['candidateMACs']==94625071104 and costs['diagnosticMACs']==754974720 and costs['assayMACs']==95380045824
        return {'schema':'tce-own-successor-toy-v1','producerABIs':len(got),'FP64Products':len(calls),'momentProducts':len(moments),'QRCalls':len(qrs),
                'maxDenseReferenceExitAbsError':float((got['exit'].float()-expected.float()).abs().max()),
                'ownInputChangesNormalizationAndRead':True,'globalRNGUnchanged':True,'nativePretrainedModelEvaluations':0,
                'scope':'Tiny full-row-rank dense FP64 GU/output and independent direct score-moment reference; component native norm/RMS/RoPE/SDPA reused. BF16-sized reference tolerances, not production parity or general quality.'}


class OwnSuccessorTests(unittest.TestCase):
    def test_independent_dense_and_score_moment_reference(self):self.assertTrue(mechanics()['ownInputChangesNormalizationAndRead'])

    def test_scalar_operand_only_law(self):
        import numpy as np
        import torch
        import diffusers.models.transformers.transformer_flux2 as native
        from own_successor import operands
        def scalar(a,b):return np.array([[sum(float(a[i,k])*float(b[k,j]) for k in range(a.shape[1])) for j in range(b.shape[1])] for i in range(a.shape[0])])
        with torch.random.fork_rng(devices=[]),torch.inference_mode():
            torch.manual_seed(88);z=torch.randn(1,9,4).bfloat16();fused=torch.randn(36,4).bfloat16()
            got,_=operands(z,fused,native.Flux2SwiGLU(),lambda name,a,b:a@b,lambda name,x:torch.linalg.qr(x,mode='reduced'),4,2)
            for stream,(lo,hi) in enumerate([(0,4),(4,9)]):
                x=z[0,lo:hi].double().numpy();raw=np.random.PCG64(np.random.SeedSequence([20261107,82,stream])).random_raw((hi-lo)*2)
                omega=np.array([1 if int(v)&1 else -1 for v in raw]).reshape(hi-lo,2)
                np.testing.assert_array_equal(got['sketchSigns'][0,lo:hi].numpy(),omega)
                sketch=scalar(x.T,omega);np.testing.assert_allclose(got['sketch'][stream].numpy(),sketch,rtol=2e-14,atol=2e-14)
                q=got['basis'][stream].numpy();c=scalar(x,q);b=scalar(q.T,fused[12:].double().numpy().T);gu=scalar(c,b)
                actual=torch.cat([got['gateFP64'][0,lo:hi],got['upFP64'][0,lo:hi]],-1).numpy()
                np.testing.assert_allclose(actual,gu,rtol=3e-13,atol=3e-13)
            self.assertNotIn('preResidualUpdate',got)

    def test_actual_typed_transaction_and_zero_receiver_gate(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import torch
        from own_successor import compute,abis,PARAMETERS,PREFIX
        from own_successor_ledger import Ledger
        from paired_fields import retain
        from publish_terminal_trajectory import check_act
        with torch.random.fork_rng(devices=[]),torch.inference_mode(),tempfile.TemporaryDirectory() as directory,patch('torch.cuda.synchronize',lambda:None):
            torch.manual_seed(89);native,block,source=specimen();source['mod'].zero_()
            params={n:block.state_dict()[n[len(PREFIX):]] for n in PARAMETERS}
            ledger=Ledger('toy-request','toy-manifest')
            def operation(values):
                return compute(block,{n:values[n] for n in source},native,attention,lambda name,x,w:torch.nn.functional.linear(x,w),
                               lambda name,x,w:x@w,lambda name,x,w:x@w,lambda name,x:torch.linalg.qr(x,mode='reduced'),
                               lambda x:None,lambda *args:None,lambda w:None,4,2)[0]
            with patch('gate_up_basis.compute',side_effect=AssertionError('Old complete producer forbidden')):
                got=ledger.run({**source,**{'parameter_'+n:v for n,v in params.items()}},operation,abis(4,12,4,4,2,1))
            descriptors={n:retain(Path(directory),n,v) for n,v in got.items()}
            check_act(ledger.records[0],descriptors,'toy-request','toy-manifest')
            self.assertEqual(len(got),45);self.assertEqual(len(ledger.records[0]['sourceFrame']['slots']),9)
            self.assertTrue(torch.equal(got['exit'],torch.cat([source['text'],source['image']],1)))
            self.assertTrue(bool((got['qrR']==0).all()));self.assertTrue(bool((got['columnSigns']==1).all()))


if __name__=='__main__':unittest.main()
