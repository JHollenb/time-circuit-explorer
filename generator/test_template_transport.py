"""Tiny independent nonlinear contractions, typed two-input execution and nonzero-A0 accounts."""
import unittest
import numpy as np


class TemplateTransportTests(unittest.TestCase):
    def test_full_rank_identity_template_nonlinear_consumer(self):
        import torch
        from test_own_successor import specimen,attention
        from template_transport import compute,abis,work
        with torch.random.fork_rng(devices=[]),torch.inference_mode():
            torch.manual_seed(90);native,block,s=specimen();w=block.attn.to_qkv_mlp_proj.weight[12:].double().T
            templates={'gu_basis':torch.stack([torch.eye(4,dtype=torch.float64)]*2),
                'gu_projectedGU':torch.stack([w,w]),'rowBasis':torch.cat([torch.eye(4,dtype=torch.float64)]*2)[None]}
            calls=[]
            def mm(n,a,b):calls.append(n);return a@b
            got,trace=compute(block,s,templates,native,attention,lambda n,a,b:torch.nn.functional.linear(a,b),lambda n,a,b:a@b,mm,lambda w:None,4)
            gu=(got['normalized'].double()@w).float().bfloat16();product=block.attn.mlp_act_fn(gu)
            self.assertTrue(torch.equal(got['gu_product'],product))
            joined=torch.cat([got['attentionRead'],product],-1)
            expected=(joined.double()@block.attn.to_out.weight.double().T).float().bfloat16()
            self.assertTrue(torch.equal(got['preResidualUpdate'],expected))
            self.assertTrue(torch.equal(got['exit'],torch.cat([s['text'],s['image']],1)+got['gate']*expected))
            self.assertEqual(len(calls),10);self.assertEqual(len(got),33)
            self.assertEqual({n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in got.items()},abis(4,12,4,4,4,1))
            self.assertEqual(trace['GU']['templateViews'][0]['basis']['stride'],[4,1])
            self.assertEqual(work()['perInputCandidateMACs'],72277819392);self.assertEqual(work()['candidateMACs'],144555638784)

    def test_scalar_frozen_operands_and_output(self):
        import torch
        from template_transport import operands,output
        import diffusers.models.transformers.transformer_flux2 as native
        def scalar(a,b):return np.array([[sum(float(a[i,k])*float(b[k,j]) for k in range(a.shape[1])) for j in range(b.shape[1])] for i in range(a.shape[0])])
        with torch.random.fork_rng(devices=[]),torch.inference_mode():
            torch.manual_seed(901);z=torch.randn(1,8,4).bfloat16()
            q=torch.randn(2,4,2,dtype=torch.float64);a=torch.randn(2,2,24,dtype=torch.float64)
            rb=torch.randn(1,8,2,dtype=torch.float64);t={'gu_basis':q,'gu_projectedGU':a,'rowBasis':rb}
            gu,_=operands(z,t,native.Flux2SwiGLU(),lambda n,a,b:a@b,12,4)
            for st,(lo,hi) in enumerate([(0,4),(4,8)]):
                c=scalar(z[0,lo:hi].double().numpy(),q[st].numpy());wanted=scalar(c,a[st].numpy())
                actual=torch.cat([gu['gateFP64'][0,lo:hi],gu['upFP64'][0,lo:hi]],-1).numpy()
                np.testing.assert_allclose(actual,wanted,rtol=4e-13,atol=4e-13)
            src={'attentionRead':torch.randn(1,8,4).bfloat16(),'product':gu['product'],'gate':torch.randn(1,1,4).bfloat16(),'text':z[:,:4],'image':z[:,4:]}
            w=torch.randn(4,16).bfloat16();out,_=output(src,t,w,lambda n,a,b:a@b,4)
            for lo,hi in [(0,4),(4,8)]:
                j=out['outputJoin'][0,lo:hi].double().numpy();b=rb[0,lo:hi].numpy()
                wanted=scalar(b,scalar(scalar(b.T,j),w.double().numpy().T))
                np.testing.assert_allclose(out['updateFP64'][0,lo:hi].numpy(),wanted,rtol=5e-13,atol=5e-13)

    def test_two_actual_typed_arms_and_source_closure(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import torch
        from test_own_successor import specimen,attention
        from template_transport import compute,abis,PARAMETERS,PREFIX
        from template_transport_ledger import Ledger
        from paired_fields import retain,payload
        from publish_terminal_trajectory import check_act
        with torch.random.fork_rng(devices=[]),torch.inference_mode(),tempfile.TemporaryDirectory() as directory,patch('torch.cuda.synchronize',lambda:None):
            torch.manual_seed(902);native,block,s=specimen()
            templates={'gu_basis':torch.randn(2,4,2,dtype=torch.float64),'gu_projectedGU':torch.randn(2,2,24,dtype=torch.float64),'rowBasis':torch.randn(1,8,2,dtype=torch.float64)}
            params={'parameter_'+n:block.state_dict()[n[len(PREFIX):]] for n in PARAMETERS}
            ledger=Ledger('template-toy','template-manifest');before={n:payload(v) for n,v in templates.items()};results=[]
            for arm in ['parent','own']:
                source={**s,'text':s['text']+(0 if arm=='parent' else 1)}
                def operation(values):
                    return compute(block,{n:values[n] for n in source},{n:values[n] for n in templates},native,attention,
                        lambda n,a,b:torch.nn.functional.linear(a,b),lambda n,a,b:a@b,lambda n,a,b:a@b,lambda w:None,4)[0]
                got=ledger.run({**source,**templates,**params},operation,abis(4,12,4,4,2,1),arm);results.append(got['exit'])
                desc={n:retain(Path(directory)/arm,n,v) for n,v in got.items()}
                check_act(ledger.records[-1],desc,'template-toy','template-manifest')
                self.assertEqual(len(ledger.records[-1]['sourceFrame']['slots']),12)
            self.assertEqual(len(ledger.records),2);self.assertFalse(torch.equal(*results))
            self.assertEqual(before,{n:payload(v) for n,v in templates.items()})

    def test_eight_corners_nonzero_parent_and_zero_denominators(self):
        from template_transport_analysis import accounts,summarize,ROW_COLUMNS
        x=np.arange(12,dtype=float).reshape(1,4,3);xh=x+1;y=x+2;y1=y+np.arange(12).reshape(1,4,3)%3
        y0=y+3;yh=y1-2;z0=y0+.5;z1=yh+np.arange(12).reshape(1,4,3)%2
        values,rows=accounts(x,xh,y,yh,y0,y1,z0,z1,2);r=summarize(rows)
        expected={'Ef0':z0-y,'Ef1':z1-y1,'A0':y0-z0,'A1':yh-z1,'B':(yh-z1)-(y0-z0),'Tf':z1-z0,'Qf':(z1-y1)-(z0-y)}
        for name,a in expected.items():
            np.testing.assert_array_equal(values[name],a)
            self.assertEqual(r['signedAccount'][name+'_energy'],sum(float(v)**2 for v in a.ravel()))
            c=[]
            for st in [a[0,:2],a[0,2:]]:
                mean=[sum(float(row[j]) for row in st)/2 for j in range(3)]
                c.extend(float(row[j])-mean[j] for row in st for j in range(3))
            self.assertEqual(r['signedAccount'][name+'_centeredEnergy'],sum(v*v for v in c))
        self.assertEqual(r['nonzeroParentAssociationRows'],4);self.assertEqual(rows.shape,(4,92));self.assertEqual(len(ROW_COLUMNS),92)
        for name,value in r['signedAccount'].items():
            if 'ClosureEnergy' in name or 'closureEnergy' in name or 'AccountRemainder' in name:self.assertEqual(value,0)
        _,zero=accounts(x,x,y,yh,y0,y,z0,z1,2);zero=summarize(zero)
        self.assertIsNone(zero['relativeNorms']['QfOverTN']);self.assertEqual(zero['zeroDenominatorRows']['fixedResponse'],4)


if __name__=='__main__':unittest.main()
