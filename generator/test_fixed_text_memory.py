import pytest
from fixed_text_memory import denoise,mechanics


def test_fixed_memory_two_clock_image_continuation():
    result=mechanics()
    assert result['ownInputChanges'] and result['maxConcurrentPhysicalMapSets']==1
    assert len(result['blocks'])==10 and all(x['textFixed'] for x in result['blocks'])
    assert [x['kind'] for x in result['blocks']]==['joint','joint','single','single','single']*2
    assert result['pretrainedModelEvaluations']==0


def test_undeclared_input_stops_before_model_access():
    with pytest.raises(ValueError,match='Undeclared'):denoise(None,{'unknown':True},None,None,None,None,None)
