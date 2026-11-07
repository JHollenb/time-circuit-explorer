import pytest
from terminal_liveness import slices,account,mechanics


def test_physical_consumed_rows_and_complete_bank():
    result=mechanics()
    assert result['completeImageMatchesFP64Toy'] and result['modelEvaluations']==0


def test_original_geometry_work_account():
    cost=account(512,1024,3072,9216)
    assert cost=={'nativeAffine':188441690112,'candidateAffine':135291469824,'nativeAttention':14495514624,'candidateAttention':9663676416}
    assert cost['nativeAffine']+cost['nativeAttention']-cost['candidateAffine']-cost['candidateAttention']==57982058496


def test_wrong_fused_layout_rejected():
    import torch
    with pytest.raises(ValueError,match='layout'):slices(torch.zeros(9,3),3,2)
