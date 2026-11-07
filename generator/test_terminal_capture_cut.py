"""Tiny source-bound custody fixtures; no model/checkpoint data."""
from pathlib import Path
import tempfile
import pytest
import torch
from saturn_pub.core import StateCut
from saturn_pub.store import LocalStore
from terminal_capture_cut import persist


def fixture():
    combined=torch.arange(20,dtype=torch.bfloat16).reshape(1,5,4)
    return {'conditioning':torch.ones(1,2,6,dtype=torch.bfloat16),'context':{'rope':(torch.ones(5,2),torch.zeros(5,2)),'single_mod':torch.ones(1,12,dtype=torch.bfloat16)},
        'cursor':19,'image':combined[:,2:],'img_ids':torch.zeros(1,3,4),'latent':torch.zeros(1,3,2,dtype=torch.bfloat16),'noise_prediction':None,
        'phase':'single','sigmas':torch.tensor([1.,.5,0.]),'step':1,'text':combined[:,:2],'timesteps':torch.tensor([1000.,500.]),'txt_ids':torch.zeros(1,2,4)}


def test_complete_cut_roundtrip_keeps_noncontiguous_views_context_and_parent():
    state=fixture();original=StateCut('original-model',{'contract':'source-native'},'original',None,state)
    before=original.manifest()
    with tempfile.TemporaryDirectory() as directory:
        directory=Path(directory).resolve()
        record=persist(state,original,Path(directory),'request','single',19,original.fingerprint)
        cut=LocalStore(directory).load(record['cutID'])
        assert cut.parent==original.fingerprint and cut.execution['requestSHA256']=='request'
        assert cut.execution['nativeAdapterSessionCompatible'] is False
        assert cut.execution_point['next_operation']=='single_block.19'
        assert cut.payload['image'].stride()==state['image'].stride()
        assert cut.manifest()['payload']==original.manifest()['payload']
        assert original.manifest()==before


def test_incomplete_or_mismatched_clock_is_refused_before_publication():
    state=fixture();original=StateCut('model',{},'original',None,state)
    with tempfile.TemporaryDirectory() as directory:
        directory=Path(directory).resolve()
        with pytest.raises(ValueError,match='Complete'):
            persist({k:v for k,v in state.items() if k!='conditioning'},original,directory,'request','single',19,'parent')
        with pytest.raises(ValueError,match='clock'):
            persist(state,original,directory,'request','readout',20,'parent')
        assert not list(Path(directory).iterdir())
