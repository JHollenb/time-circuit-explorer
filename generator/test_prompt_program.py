import inspect
import pytest
from prompt_program import compile_clock,matrix_account,mechanics,single_text


def test_two_clock_compiler_and_own_image_reference():
    result=mechanics()
    assert result['incomingKVMatchesNativeTextOnly'] and result['actorMatchesNativeImageAtSameIncomingText']
    assert result['ownInputChanges'] and result['pretrainedModelEvaluations']==0
    assert len(result['blocks'])==10 and sum(x['updated'] for x in result['blocks'])==8
    assert result['terminalKVMapShapes']==[[32,16]]*2


def test_compiler_has_no_image_or_noise_argument():
    names=set(inspect.signature(compile_clock).parameters)
    assert names=={'model','embedding','clock_argument','text_ids','native','attention','on_layer','with_terminal_map'}
    with pytest.raises(TypeError):compile_clock(None,None,None,None,None,None,None,image=None)


def test_independent_prospective_account():
    a=matrix_account()
    assert a['compilerMatrixMACs']==6223407611904
    assert a['cacheBytesBF16']==629145600
    assert a['textKVFormations']==100 and a['liveTextUpdates']==96
    assert a['removedImageToTextAttentionMACs']==309237645312
    assert a['terminalDeadTextMACs']==231928233984


def test_terminal_text_never_computes_query_mlp_or_output():
    import torch
    import diffusers.models.transformers.transformer_flux2 as native
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(168)
        block=native.Flux2SingleTransformerBlock(16,2,8).double().eval().requires_grad_(False)
        text=torch.arange(48,dtype=torch.float64).reshape(1,3,16)/40;original=text.clone()
        mod=torch.arange(48,dtype=torch.float64).reshape(1,48)/200
        rope=(torch.ones(3,8,dtype=torch.float32),torch.zeros(3,8,dtype=torch.float32))
        def forbidden(*args,**kwargs):raise AssertionError('Dead terminal computation ran')
        block.attn.to_qkv_mlp_proj.forward=forbidden;block.attn.norm_q.forward=forbidden
        block.attn.mlp_act_fn.forward=forbidden;block.attn.to_out.forward=forbidden
        maps=[]
        with torch.inference_mode():
            k,v,after=single_text(block,text,mod,rope,native,forbidden,terminal=True,
                with_map=lambda weight,operation:maps.append(weight.clone()) or operation())
        assert after is None and list(k.shape)==[1,3,2,8] and list(v.shape)==[1,3,2,8]
        assert len(maps)==1 and torch.equal(maps[0],block.attn.to_qkv_mlp_proj.weight[16:48])
        assert torch.equal(text,original)
