"""Tiny observation transparency, typed native fields and independent six-corner accounts."""
import unittest
import numpy as np


class DenseCornerTests(unittest.TestCase):
    def test_frozen_consumer_observer_and_typed_fields(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import torch
        from test_own_successor import specimen,attention
        from image_partition_moments import single
        from dense_corner import compute,abis,PARAMETERS,PREFIX,work
        from dense_corner_ledger import Ledger
        from paired_fields import payload,retain
        from publish_terminal_trajectory import check_act
        with torch.random.fork_rng(devices=[]),torch.inference_mode(),tempfile.TemporaryDirectory() as directory,patch('torch.cuda.synchronize',lambda:None):
            torch.manual_seed(91);native,block,source=specimen()
            original_linear=torch.nn.functional.linear;original_rope=native.apply_rotary_emb;original_split=native.Flux2Modulation.__dict__['split']
            wanted=single(block,source['text'],source['image'],source['mod'],(source['cosine'],source['sine']),native,attention,lambda name,a,b:a@b,None)
            params={n:block.state_dict()[n[len(PREFIX):]] for n in PARAMETERS}
            ledger=Ledger('dense-toy','dense-manifest');observed={};trace={};rng=torch.get_rng_state();before={n:payload(v) for n,v in source.items()}
            def observer(name,value):observed[name]=payload(value)
            def operation(values):
                got,info=compute(block,{n:values[n] for n in source},native,attention,
                    lambda name,x,w,bias,op:op(x,w,bias),lambda name,a,b:a@b,observer)
                trace.update(info);return got
            got=ledger.run({**source,**{'parameter_'+n:v for n,v in params.items()}},operation,abis(4,12,4,4,1))
            self.assertEqual(observed,{n:payload(v) for n,v in got.items()})
            self.assertEqual(before,{n:payload(v) for n,v in source.items()})
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertIs(torch.nn.functional.linear,original_linear);self.assertIs(native.apply_rotary_emb,original_rope)
            self.assertIs(native.Flux2Modulation.__dict__['split'],original_split)
            self.assertTrue(torch.equal(got['textOut'],wanted[0]));self.assertTrue(torch.equal(got['imageOut'],wanted[1]))
            self.assertTrue(torch.equal(got['exit'],torch.cat([wanted[0],wanted[1]],1)))
            self.assertTrue(torch.equal(got['fused'],original_linear(got['normalized'],block.attn.to_qkv_mlp_proj.weight)))
            self.assertEqual(trace['linears'],['fullFused','fullOut'])
            self.assertEqual(trace['actualNativeViews']['upRaw']['storageOffset'],24)
            self.assertEqual(trace['actualNativeViews']['gateRaw']['stride'],[288,36,1])
            descriptors={n:retain(Path(directory),n,v) for n,v in got.items()}
            check_act(ledger.records[0],descriptors,'dense-toy','dense-manifest')
            self.assertEqual(len(got),27);self.assertEqual(len(ledger.records[0]['sourceFrame']['slots']),9)
            self.assertEqual(work()['candidateMACs'],200321531904)

    def test_six_corner_independent_signed_accounts(self):
        from dense_corner_analysis import accounts,summarize
        x=np.arange(12,dtype=float).reshape(1,4,3);xh=x+np.array([1,-2,0,2,1,-1,3,0,2,-1,1,2]).reshape(1,4,3)
        y=x+2;y0=y+np.arange(12).reshape(1,4,3)%3;yh=y0+1;y1=y+np.array([2,-1,3,1,0,2,4,-2,0,1,2,-1]).reshape(1,4,3)
        values,rows=accounts(x,xh,y,yh,y0,y1,2);result=summarize(rows)
        expected={'TN':y1-y,'E1':yh-y1,'Q':(yh-y1)-(y0-y),'V':(y1-xh)-(y-x)}
        for name,value in expected.items():
            np.testing.assert_array_equal(values[name],value)
            self.assertEqual(result['signedAccount'][name+'_energy'],sum(float(v)**2 for v in value.ravel()))
            centered=[]
            for stream in [value[0,:2],value[0,2:]]:
                mean=[sum(float(row[j]) for row in stream)/2 for j in range(3)]
                centered += [float(row[j])-mean[j] for row in stream for j in range(3)]
            self.assertEqual(result['signedAccount'][name+'_centeredEnergy'],sum(v*v for v in centered))
        for key,value in result['signedAccount'].items():
            if 'ClosureEnergy' in key or 'closureEnergy' in key or 'AccountRemainder' in key:self.assertEqual(value,0)
        self.assertEqual(rows.shape,(4,86))

    def test_zero_response_denominators(self):
        from dense_corner_analysis import accounts,summarize
        x=np.arange(12,dtype=float).reshape(1,4,3);y=x+2;y0=y+1
        _,rows=accounts(x,x,y,y0,y0,y,2);r=summarize(rows)
        self.assertIsNone(r['relativeNorms']['TNOverD']);self.assertIsNone(r['relativeNorms']['QOverTN'])
        self.assertEqual(r['zeroDenominatorRows']['denseResponseVsCarry'],4)
        self.assertEqual(r['zeroDenominatorRows']['approxResponseVsDense'],4)
        self.assertEqual(r['signedAccount']['dualQClosureEnergy'],0)


if __name__=='__main__':unittest.main()
