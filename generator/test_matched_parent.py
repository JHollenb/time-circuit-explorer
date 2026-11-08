"""Tiny independent five-corner accounts, zero denominators and typed parent bindings."""
import unittest
import numpy as np


class MatchedParentTests(unittest.TestCase):
    def test_five_corner_signed_and_centered_accounts(self):
        from matched_parent_analysis import accounts,summarize,ROW_COLUMNS
        shape=(1,4,3);x=np.arange(12,dtype=float).reshape(shape)
        xh=x+np.array([1,-2,3,0,1,-1,2,0,-3,1,4,2]).reshape(shape)
        y=x+np.array([4,2,-1,3,-2,1,0,1,2,-1,2,3]).reshape(shape)
        y0=y+np.array([1,0,-1,2,-1,3,-2,1,0,3,-2,1]).reshape(shape)
        yh=y0+np.array([-2,1,0,1,0,2,3,-1,2,-1,3,0]).reshape(shape)
        values,rows=accounts(x,xh,y,yh,y0,2)
        result=summarize(rows)
        direct={'D':xh-x,'E0':y0-y,'T':yh-y0,'U':(yh-xh)-(y0-x),'W':(yh-xh)-(y-x),'Eout':yh-y}
        for name,value in direct.items():
            np.testing.assert_array_equal(values[name],value)
            self.assertEqual(result['signedAccount'][name+'_energy'],sum(float(z)**2 for z in value.ravel()))
            centered=[]
            for stream in [value[0,:2],value[0,2:]]:
                mean=[sum(float(row[j]) for row in stream)/2 for j in range(3)]
                centered += [float(row[j])-mean[j] for row in stream for j in range(3)]
            self.assertEqual(result['signedAccount'][name+'_centeredEnergy'],sum(z*z for z in centered))
        e0t=2*sum(float(a)*float(b) for a,b in zip(direct['E0'].ravel(),direct['T'].ravel()))
        self.assertEqual(result['signedAccount']['twiceE0TInner'],e0t)
        for name,value in result['signedAccount'].items():
            if 'closureEnergy' in name or 'ClosureEnergy' in name or 'AccountRemainder' in name:self.assertEqual(value,0)
        self.assertEqual(rows.shape,(4,64));self.assertEqual(len(ROW_COLUMNS),64)

    def test_zero_response_denominators_are_explicit(self):
        from matched_parent_analysis import accounts,summarize
        x=np.arange(12,dtype=float).reshape(1,4,3);y=x+2;y0=y+1
        _,rows=accounts(x,x,y,y0,y0,2);value=summarize(rows)
        self.assertIsNone(value['responseVsCarry']['errorOverTargetRMS'])
        self.assertEqual(value['zeroDenominatorRows']['responseVsCarry'],4)
        self.assertIsNone(value['relativeNorms']['TOverD'])
        self.assertIsNone(value['relativeNorms']['E0OverT'])
        self.assertEqual(value['signedAccount']['T_energy'],0)
        self.assertEqual(value['signedAccount']['U_energy'],0)

    def test_parent_typed_transaction_and_frozen_compute(self):
        from unittest.mock import patch
        import tempfile
        from pathlib import Path
        import torch
        from test_own_successor import specimen,attention
        import own_successor
        from matched_parent import compute,abis,work,PARAMETERS,PREFIX
        from matched_parent_ledger import Ledger
        from paired_fields import retain
        from publish_terminal_trajectory import check_act
        self.assertIs(compute,own_successor.compute)
        self.assertEqual(work()['candidateMACs'],94625071104)
        self.assertEqual(work()['diagnosticMACs'],0)
        with torch.random.fork_rng(devices=[]),torch.inference_mode(),tempfile.TemporaryDirectory() as directory,patch('torch.cuda.synchronize',lambda:None):
            torch.manual_seed(90);native,block,source=specimen();source['mod'].zero_()
            params={n:block.state_dict()[n[len(PREFIX):]] for n in PARAMETERS};ledger=Ledger('matched-toy','new-intake-manifest')
            def operation(values):
                return compute(block,{n:values[n] for n in source},native,attention,
                    lambda name,x,w:torch.nn.functional.linear(x,w),lambda name,x,w:x@w,lambda name,x,w:x@w,
                    lambda name,x:torch.linalg.qr(x,mode='reduced'),lambda x:None,lambda *args:None,lambda w:None,4,2)[0]
            got=ledger.run({**source,**{'parameter_'+n:v for n,v in params.items()}},operation,abis(4,12,4,4,2,1))
            descriptors={n:retain(Path(directory),n,v) for n,v in got.items()}
            check_act(ledger.records[0],descriptors,'matched-toy','new-intake-manifest')
            self.assertEqual(ledger.records[0]['sourceFrame']['metadata']['phase'],'matchedParentSuccessor')
            self.assertEqual(len(ledger.records[0]['sourceFrame']['slots']),9)
            self.assertTrue(torch.equal(got['exit'],torch.cat([source['text'],source['image']],1)))


if __name__=='__main__':unittest.main()
