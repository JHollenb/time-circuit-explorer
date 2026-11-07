"""Ordinary FLUX denoising with an image-only terminal consumer boundary."""
from terminal_liveness import slices,compute


def terminal_maps(model):
    attn=model.single_transformer_blocks[-1].attn;dim=attn.query_dim;mlp=attn.mlp_hidden_dim
    return {**slices(attn.to_qkv_mlp_proj.weight,dim,mlp),'qNorm':attn.norm_q.weight,'kNorm':attn.norm_k.weight,'out':attn.to_out.weight}


def denoise(model,inputs,weights,native,attention,capture):
    import torch
    if set(inputs)-{'hidden_states','encoder_hidden_states','timestep','img_ids','txt_ids','guidance','joint_attention_kwargs','return_dict','kv_cache','kv_cache_mode','num_ref_tokens','ref_fixed_timestep'}:
        raise ValueError('Undeclared ordinary forward input')
    if torch.is_grad_enabled() or model.training or model.is_cache_enabled or getattr(model,'_hf_hook',None) is not None:
        raise ValueError('Only ordinary inference without model cache/offload is supported')
    if any(inputs.get(n) is not None for n in ['guidance','kv_cache','kv_cache_mode']) or inputs.get('num_ref_tokens',0)!=0 or inputs.get('ref_fixed_timestep',0.)!=0.:
        raise ValueError('Guidance/reference/cache mode is outside this rewrite')
    if inputs.get('joint_attention_kwargs') not in (None,{}):raise ValueError('Attention overrides are outside this rewrite')
    image=inputs['hidden_states'];text=inputs['encoder_hidden_states'];time=inputs['timestep'];image_ids=inputs['img_ids'];text_ids=inputs['txt_ids']
    ntext=text.shape[1]
    if image.shape[0]!=1 or text.shape[0]!=1 or image_ids.ndim!=3 or text_ids.ndim!=3:raise ValueError('Single supplied ordinary image batch required')
    # Preserve the native cast/scale boundary; do not cancel the pipeline's /1000.
    time=time.to(image.dtype)*1000;temb=model.time_guidance_embed(time,None)
    img_mod=model.double_stream_modulation_img(temb);txt_mod=model.double_stream_modulation_txt(temb);single_mod=model.single_stream_modulation(temb)
    image=model.x_embedder(image);text=model.context_embedder(text);capture('context',text)
    ir=model.pos_embed(image_ids[0]);tr=model.pos_embed(text_ids[0]);rope=(torch.cat([tr[0],ir[0]],0),torch.cat([tr[1],ir[1]],0))
    for block in model.transformer_blocks:
        text,image=block(hidden_states=image,encoder_hidden_states=text,temb_mod_img=img_mod,temb_mod_txt=txt_mod,image_rotary_emb=rope,joint_attention_kwargs=None)
    rows=torch.cat([text,image],1)
    for block in model.single_transformer_blocks[:-1]:
        rows=block(hidden_states=rows,encoder_hidden_states=None,temb_mod=single_mod,image_rotary_emb=rope,joint_attention_kwargs=None)
    capture('terminalRows',rows);capture('singleMod',single_mod);capture('temb',temb);capture('cosine',rope[0]);capture('sine',rope[1])
    last=model.single_transformer_blocks[-1]
    image=compute(rows,single_mod,rope[0],rope[1],weights,torch.nn.functional.linear,attention,native,text=ntext,heads=last.attn.heads,eps=last.norm.eps)
    capture('terminalImage',image)
    output=model.proj_out(model.norm_out(image,temb))
    return (output,) if not inputs.get('return_dict',True) else native.Flux2Transformer2DModelOutput(sample=output)


def mechanics():
    """Tiny CPU two-clock implementation fixture, never a pretrained specimen."""
    import torch
    import diffusers.models.transformers.transformer_flux2 as native
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(66)
        model=native.Flux2Transformer2DModel(in_channels=4,num_layers=1,num_single_layers=2,num_attention_heads=2,attention_head_dim=8,
            joint_attention_dim=12,timestep_guidance_channels=16,axes_dims_rope=(2,2,2,2),guidance_embeds=False).double().eval().requires_grad_(False)
        rows=torch.arange(16,dtype=torch.float64).reshape(1,4,4)/20;text=torch.arange(36,dtype=torch.float64).reshape(1,3,12)/40
        iid=torch.arange(16).reshape(1,4,4);tid=torch.arange(12).reshape(1,3,4);candidate=rows.clone();reference=rows.clone();errors=[];own_inputs=[];captures=[]
        with torch.inference_mode():
            maps=terminal_maps(model)
            for k,t in enumerate([.88398,.71749]):
                args={'encoder_hidden_states':text,'timestep':torch.tensor([t],dtype=torch.float64),'img_ids':iid,'txt_ids':tid,'return_dict':False}
                ref=model(hidden_states=reference,**args)[0];own_inputs.append(candidate.clone());captured={}
                got=denoise(model,{'hidden_states':candidate,**args},maps,native,native.dispatch_attention_fn,lambda n,v:captured.setdefault(n,v))
                got=got[0];torch.testing.assert_close(got,ref,rtol=2e-6,atol=2e-7)
                captures.append({n:list(v.shape) for n,v in captured.items()});errors.append(float((got-ref).abs().max()))
                reference=reference-.125*ref;candidate=candidate-.125*got
            torch.testing.assert_close(candidate,reference,rtol=2e-6,atol=2e-7)
        return {'schema':'tce-terminal-trajectory-toy-v1','nativeMaxAbsErrors':errors,'ownInputChanges':not torch.equal(own_inputs[0],own_inputs[1]),
            'terminalImageShapes':[x['terminalImage'] for x in captures],'sourceRowShapes':[x['terminalRows'] for x in captures],
            'relativeTolerance':2e-6,'absoluteTolerance':2e-7,'pretrainedModelEvaluations':0,
            'scope':'Tiny supplied-seed CPU model, double weights/state with native internal finite primitives; two own toy commits. Tolerance covers native FP32 rotary rounding, not a real BF16 quality gate.'}
