import pytest
from terminal_trajectory import denoise,mechanics


def test_two_clock_tiny_consumer_boundary():
    result=mechanics()
    assert result['ownInputChanges']
    assert result['terminalImageShapes']==[[1,4,16],[1,4,16]]
    assert result['sourceRowShapes']==[[1,7,16],[1,7,16]]
    assert result['pretrainedModelEvaluations']==0


def test_unknown_inputs_fail_before_execution():
    with pytest.raises(ValueError,match='Undeclared'):denoise(None,{'undeclared':True},None,None,None,None)
