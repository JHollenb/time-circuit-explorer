"""Independent tiny projected-write, signed-target and typed-output references."""
import unittest


def mechanics():
    import numpy as np
    import torch
    from row_output import compute,abis,work
    from row_output_analysis import accounts,summarize,ROW_COLUMNS
    def scalar_mm(a,b):return np.array([[sum(float(a[i,k])*float(b[k,j]) for k in range(a.shape[1])) for j in range(b.shape[1])] for i in range(a.shape[0])],dtype=np.float64)
    with torch.random.fork_rng(devices=[]),torch.inference_mode():
        torch.manual_seed(83);n,nt,d,m,r=7,3,4,8,2
        source={'coordinates':torch.randn(1,n,r,dtype=torch.float64),'product':torch.randn(1,n,m).bfloat16(),'attentionRead':torch.randn(1,n,d).bfloat16(),
                'gate':torch.randn(1,1,d).bfloat16(),'text':torch.randn(1,nt,d).bfloat16(),'image':torch.randn(1,n-nt,d).bfloat16()};weight=torch.randn(d,d+m).bfloat16();calls=[];joins=[];qrs=[];rng=torch.get_rng_state()
        def mm(name,a,b):calls.append(name);assert a.dtype==b.dtype==torch.float64;return a@b
        def qr(name,c):qrs.append(name);return torch.linalg.qr(c,mode='reduced')
        got,trace=compute(source,weight,mm,qr,lambda value:joins.append(value.clone()),nt)
        assert torch.equal(rng,torch.get_rng_state()) and len(joins)==1 and joins[0].dtype==torch.bfloat16
        np.testing.assert_array_equal(joins[0].view(torch.uint16).numpy(),torch.cat([source['attentionRead'],source['product']],-1).view(torch.uint16).numpy())
        expected=[];maximum=0.
        for stream,(lo,hi) in enumerate([(0,nt),(nt,n)]):
            u=got['rowBasis'][0,lo:hi].numpy();rr=got['qrR'][stream].numpy();c=source['coordinates'][0,lo:hi].numpy()
            np.testing.assert_allclose(scalar_mm(u,rr),c,rtol=3e-14,atol=3e-14);np.testing.assert_allclose(scalar_mm(u.T,u),np.eye(r),rtol=3e-14,atol=3e-14);assert np.all(np.diag(rr)>=0)
            # Independent Gram-Schmidt checks the selected column span, not QR signs.
            orth=[]
            for j in range(r):
                column=c[:,j].copy()
                for previous in orth:column-=sum(column*previous)*previous
                orth.append(column/np.sqrt(sum(column*column)))
            independent=np.stack(orth,1);np.testing.assert_allclose(u@u.T,independent@independent.T,rtol=4e-14,atol=4e-14)
            j=joins[0][0,lo:hi].double().numpy();k=scalar_mm(u.T,j);b=scalar_mm(k,weight.double().numpy().T);y=scalar_mm(u,b);expected.append(y)
            maximum=max(maximum,float(np.max(np.abs(y-got['updateFP64'][0,lo:hi].numpy()))))
            np.testing.assert_allclose(y,got['updateFP64'][0,lo:hi].numpy(),rtol=3e-13,atol=3e-13)
        update=torch.from_numpy(np.concatenate(expected)[None]).float().bfloat16();gated=source['gate']*update;exited=torch.cat([source['text'],source['image']],1)+gated
        for name,value in [('preResidualUpdate',update),('gatedWrite',gated),('exit',exited)]:assert torch.equal(value,got[name]),name
        assert trace['admittedUpdate']['stride']==[n*d,d,1] and trace['admittedUpdate']['storageOffset']==0
        assert calls==['projectInputs0','compressedOut0','reconstructUpdate0','projectInputs1','compressedOut1','reconstructUpdate1'] and len(qrs)==2
        assert {name:{'shape':list(value.shape),'dtype':str(value.dtype)} for name,value in got.items()}==abis(d,m,nt,n-nt,r)
        old=torch.randn(1,n,d,dtype=torch.float64).numpy();target=torch.randn(1,n,d,dtype=torch.float64).numpy();candidate=got['updateFP64'].numpy();rows=accounts(candidate,old,target,nt)
        assert rows.shape==(n,len(ROW_COLUMNS))
        for i in range(n):
            expected_cross=2*sum((old[0,i,k]-target[0,i,k])*(candidate[0,i,k]-old[0,i,k]) for k in range(d))
            np.testing.assert_allclose(rows[i,24],expected_cross,rtol=2e-14,atol=2e-14)
        summary=summarize(rows);np.testing.assert_allclose(summary['signedError']['totalEnergy'],sum((float(candidate[0,i,k])-float(target[0,i,k]))**2 for i in range(n) for k in range(d)),rtol=2e-14)
        np.testing.assert_allclose(summary['signedError']['energyAccountRemainder'],0,rtol=0,atol=1e-11)
        np.testing.assert_allclose(summary['signedError']['centeredEnergyAccountRemainder'],0,rtol=0,atol=1e-11)
        costs=work();assert costs['candidateMACs']==12683575296 and costs['diagnosticMACs']==50331648 and costs['assayMACs']==12733906944 and costs['prospectiveComposedProducerMACs']==32010928128
        return {'schema':'tce-current-row-output-toy-v1','candidateProducts':6,'QRCalls':2,'producerABIs':len(got),'maxIndependentUpdateAbsError':maximum,'nativeGateResidualExact':True,'globalRNGUnchanged':True,'actualJoinObservedBeforePromotion':True,'nativePretrainedModelEvaluations':0,
            'scope':'Tiny independent scalar contractions/Gram-Schmidt span/final double-admission/native gate-add/layout, signed full/centered primary-secondary energy. No real model arrays.'}


class RowOutputTests(unittest.TestCase):
    def test_independent_scalar_projection_and_signed_accounts(self):self.assertTrue(mechanics()['nativeGateResidualExact'])

    def test_actual_typed_transaction(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import torch
        from row_output import compute,abis
        from row_output_ledger import Ledger
        from paired_fields import retain
        from publish_terminal_trajectory import check_act
        with torch.inference_mode(),tempfile.TemporaryDirectory() as directory,patch('torch.cuda.synchronize',lambda:None):
            source={'coordinates':torch.zeros(1,8,2,dtype=torch.float64),'product':torch.ones(1,8,8,dtype=torch.bfloat16),'attentionRead':torch.ones(1,8,4,dtype=torch.bfloat16),
                    'gate':torch.ones(1,1,4,dtype=torch.bfloat16),'text':torch.ones(1,4,4,dtype=torch.bfloat16),'image':torch.ones(1,4,4,dtype=torch.bfloat16)};weight=torch.ones(4,12,dtype=torch.bfloat16);ledger=Ledger('toy-request','toy-manifest')
            def operation(values):return compute({name:values[name] for name in source},values['originalOut'],lambda name,a,b:a@b,lambda name,c:torch.linalg.qr(c,mode='reduced'),lambda joined:None,4)[0]
            got=ledger.run({**source,'originalOut':weight},operation,abis(4,8,4,4,2));descriptors={name:retain(Path(directory),name,value) for name,value in got.items()}
            check_act(ledger.records[0],descriptors,'toy-request','toy-manifest')
            self.assertEqual(len(ledger.records[0]['sourceFrame']['slots']),7);self.assertEqual(list(got['rowBasis'].shape),[1,8,2]);self.assertTrue(bool((got['columnSigns']==1).all()));self.assertTrue(bool((got['qrR']==0).all()))


if __name__=='__main__':unittest.main()
