import torch
from gate_first_mlp import source_maps,select,selected_maps,mechanics


def test_source_intervals_and_norms():
    fused=torch.arange(48*4,dtype=torch.float32).reshape(48,4)/100;out=torch.arange(4*22,dtype=torch.float32).reshape(4,22)/70
    got=source_maps(fused,out,4,18)
    torch.testing.assert_close(got['KV'],fused[4:12]);torch.testing.assert_close(got['QG'],torch.cat([fused[:4],fused[12:30]]))
    torch.testing.assert_close(got['upNorm'],fused[30:].square().sum(1).sqrt());torch.testing.assert_close(got['downNorm'],out[:,4:].square().sum(0).sqrt())
    assert got['KV'].data_ptr()!=fused[4:].data_ptr() and got['QG'].is_contiguous()


def test_zero_gate_ties_choose_source_indices():
    result=select(torch.ones(1,4,8),torch.zeros(1,4,12),torch.ones(12),torch.ones(12),3)
    assert result['support'].tolist()==[0,1,2] and result['rankedIndices'].tolist()==list(range(12))
    assert result['omittedScoreSum']==0 and result['selectedScoreSum']==0


def test_scores_and_ascending_support_align_physical_maps():
    x=torch.tensor([[[1.,2.],[3.,4.]]]);a=torch.tensor([[[1.,2.,3.,4.],[1.,1.,1.,1.]]]);up=torch.ones(4);down=torch.tensor([1.,3.,1.,2.])
    result=select(x,a,up,down,2);expected=down*((a[0].square()*x[0].square().sum(-1)[:,None]).sum(0)).sqrt()
    torch.testing.assert_close(result['score'],expected)
    ranking=sorted(range(4),key=lambda i:(-float(expected[i]),i));support=sorted(ranking[:2])
    assert result['support'].tolist()==support
    fused=torch.arange(14*2,dtype=torch.float32).reshape(14,2);out=torch.arange(2*6,dtype=torch.float32).reshape(2,6)
    maps=selected_maps(fused,out,result['support'],2,4)
    torch.testing.assert_close(maps['USelected'],fused[10+result['support']]);torch.testing.assert_close(maps['outSelected'],torch.cat([out[:,:2],out[:,2+result['support']]],1))
    assert maps['USelected'].shape==(2,2) and maps['outSelected'].shape==(2,4)


def test_fp32_scores_and_norms_under_autocast():
    fused=torch.ones(48,4,dtype=torch.bfloat16);out=torch.ones(4,22,dtype=torch.bfloat16)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        maps=source_maps(fused,out,4,18);got=select(torch.ones(1,4,4,dtype=torch.bfloat16),torch.ones(1,4,18,dtype=torch.bfloat16),maps['upNorm'],maps['downNorm'],4)
    assert maps['upNorm'].dtype==maps['downNorm'].dtype==got['score'].dtype==got['omittedScoreSum'].dtype==torch.float32
    assert got['support'].dtype==torch.int64


def test_independent_dense_and_selected_reference_and_bound():
    result=mechanics();assert result['pretrainedModelEvaluations']==0
    assert [x['count'] for x in result['results']]==[12,3]
    assert len(result['physicalLinearShapes'])==6
    assert result['physicalLinearShapes'][-2:]==[[[1,4,8],[3,8]],[[1,4,11],[8,11]]]
    assert all(x['outputStride']==[56,8,1] for x in result['results'])


def test_nonfinite_scores_cannot_enter_ranking():
    import pytest
    with pytest.raises(ValueError,match='Nonfinite'):
        select(torch.ones(1,4,8),torch.full((1,4,12),float('nan')),torch.ones(12),torch.ones(12),3)
