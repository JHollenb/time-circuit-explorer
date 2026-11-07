import pytest
import torch
from terminal_attention_ledger import wrap_sdpa


@pytest.mark.parametrize('mode',['positional','keyword','mixed'])
def test_native_sdpa_calls_forward_unchanged(mode):
    q=torch.ones(1,2,3,4);k=torch.ones(1,2,5,4);v=torch.ones(1,2,5,6);mask=object();records=[];calls=[];result=object()
    def native(query,key,value,*args,**kwargs):
        calls.append((query,key,value,args,kwargs));return result
    wrapped=wrap_sdpa(native,{'kind':'denoise','step':2},records)
    if mode=='positional':got=wrapped(q,k,v,mask,0.0,False,scale=.5)
    elif mode=='keyword':got=wrapped(query=q,key=k,value=v,attn_mask=mask,dropout_p=0.0,is_causal=False,scale=.5)
    else:got=wrapped(q,key=k,value=v,attn_mask=mask,scale=.5)
    assert got is result
    assert calls[0][:3]==(q,k,v)
    assert calls[0][4]['scale']==.5
    assert calls[0][3]==(mask,0.0,False) if mode=='positional' else calls[0][4]['attn_mask'] is mask
    assert records==[{'kind':'denoise','step':2,'QShape':[1,2,3,4],'KShape':[1,2,5,4],'VShape':[1,2,5,6],'nominalQKAndAVMACs':300}]
