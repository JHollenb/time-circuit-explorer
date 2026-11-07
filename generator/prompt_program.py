"""Image-independent depth-evolving text program and its own image consumer.

The compiler's public numerical interface contains no image, noise or image IDs.
K is head-normalized and rotated; V is the original projected value. Both are
formed from the incoming text state, before the optional text update.
"""


def attend(q,k,v,attention):
    return attention(q,k,v,attn_mask=None,backend=None,parallel_config=None).flatten(2,3).to(q.dtype)


def joint_text(block,text,mod,rope,native,attention):
    (shift,scale,gate),(ms,mc,mg)=native.Flux2Modulation.split(mod,2)
    z=(1+scale)*block.norm1_context(text)+shift;a=block.attn
    q=a.norm_added_q(a.add_q_proj(z).unflatten(-1,(a.heads,-1)))
    k=a.norm_added_k(a.add_k_proj(z).unflatten(-1,(a.heads,-1)))
    v=a.add_v_proj(z).unflatten(-1,(a.heads,-1))
    q=native.apply_rotary_emb(q,rope,sequence_dim=1)
    k=native.apply_rotary_emb(k,rope,sequence_dim=1)
    after=text+gate*a.to_add_out(attend(q,k,v,attention))
    z=block.norm2_context(after);z=z*(1+mc)+ms
    after=after+mg*block.ff_context(z)
    if after.dtype==__import__('torch').float16:after=after.clip(-65504,65504)
    return k,v,after


def single_text(block,text,mod,rope,native,attention,terminal=False,with_map=None):
    import torch
    import torch.nn.functional as F
    shift,scale,gate=native.Flux2Modulation.split(mod,1)[0]
    z=(1+scale)*block.norm(text)+shift;a=block.attn;dim=a.inner_dim
    if terminal:
        weight=a.to_qkv_mlp_proj.weight[dim:3*dim].clone()
        operation=lambda:F.linear(z,weight)
        kv=operation() if with_map is None else with_map(weight,operation)
        k,v=kv.chunk(2,dim=-1);q=gu=None
    else:
        q,k,v,gu=a.to_qkv_mlp_proj(z).split([dim,dim,dim,2*a.mlp_hidden_dim],dim=-1)
    k=a.norm_k(k.unflatten(-1,(a.heads,-1)));v=v.unflatten(-1,(a.heads,-1))
    k=native.apply_rotary_emb(k,rope,sequence_dim=1)
    if terminal:return k,v,None
    q=a.norm_q(q.unflatten(-1,(a.heads,-1)));q=native.apply_rotary_emb(q,rope,sequence_dim=1)
    after=text+gate*a.to_out(torch.cat([attend(q,k,v,attention),a.mlp_act_fn(gu)],dim=-1))
    if after.dtype==torch.float16:after=after.clip(-65504,65504)
    return k,v,after


def compile_clock(model,embedding,clock_argument,text_ids,native,attention,on_layer,with_terminal_map=None):
    """Compile exactly one clock; no image-dependent operand is accepted."""
    import torch
    if torch.is_grad_enabled() or model.training or model.is_cache_enabled or getattr(model,'_hf_hook',None) is not None:
        raise ValueError('Ordinary inference without cache/offload required')
    if embedding.ndim!=3 or embedding.shape[0]!=1 or text_ids.shape!=(1,embedding.shape[1],4):raise ValueError('Text compiler geometry differs')
    time=clock_argument.to(embedding.dtype)*1000;temb=model.time_guidance_embed(time,None)
    im=model.double_stream_modulation_img(temb);tm=model.double_stream_modulation_txt(temb);sm=model.single_stream_modulation(temb)
    text=model.context_embedder(embedding);initial=text;rope=model.pos_embed(text_ids[0])
    for index,block in enumerate(model.transformer_blocks):
        k,v,after=joint_text(block,text,tm,rope,native,attention)
        on_layer('joint',index,text,k,v,after);text=after
    for index,block in enumerate(model.single_transformer_blocks):
        final=index==len(model.single_transformer_blocks)-1
        k,v,after=single_text(block,text,sm,rope,native,attention,terminal=final,with_map=with_terminal_map if final else None)
        on_layer('single',index,text,k,v,after)
        if after is not None:text=after
    return {'context':initial,'temb':temb,'imageMod':im,'textMod':tm,'singleMod':sm,
            'clockArgument':clock_argument,'textIDs':text_ids,'textCosine':rope[0],'textSine':rope[1]}


def joint_image(block,image,mod,image_rope,text_key,text_value,native,attention):
    import torch
    (shift,scale,gate),(ms,mc,mg)=native.Flux2Modulation.split(mod,2)
    z=(1+scale)*block.norm1(image)+shift;a=block.attn
    q=a.norm_q(a.to_q(z).unflatten(-1,(a.heads,-1)))
    k=a.norm_k(a.to_k(z).unflatten(-1,(a.heads,-1)));v=a.to_v(z).unflatten(-1,(a.heads,-1))
    q=native.apply_rotary_emb(q,image_rope,sequence_dim=1);k=native.apply_rotary_emb(k,image_rope,sequence_dim=1)
    h=attend(q,torch.cat([text_key,k],1),torch.cat([text_value,v],1),attention)
    after=image+gate*a.to_out[1](a.to_out[0](h))
    z=block.norm2(after);z=z*(1+mc)+ms
    return after+mg*block.ff(z)


def single_image(block,image,mod,image_rope,text_key,text_value,native,attention):
    import torch
    shift,scale,gate=native.Flux2Modulation.split(mod,1)[0]
    z=(1+scale)*block.norm(image)+shift;a=block.attn;dim=a.inner_dim
    q,k,v,gu=a.to_qkv_mlp_proj(z).split([dim,dim,dim,2*a.mlp_hidden_dim],dim=-1)
    q=a.norm_q(q.unflatten(-1,(a.heads,-1)));k=a.norm_k(k.unflatten(-1,(a.heads,-1)));v=v.unflatten(-1,(a.heads,-1))
    q=native.apply_rotary_emb(q,image_rope,sequence_dim=1);k=native.apply_rotary_emb(k,image_rope,sequence_dim=1)
    h=attend(q,torch.cat([text_key,k],1),torch.cat([text_value,v],1),attention)
    after=image+gate*a.to_out(torch.cat([h,a.mlp_act_fn(gu)],dim=-1))
    if after.dtype==torch.float16:after=after.clip(-65504,65504)
    return after


def consume_clock(model,image,image_ids,common,cache,native,attention,on_layer,capture):
    """Consume only this compiler's cache and advance the actor's own image."""
    image=model.x_embedder(image);capture('imageStart',image);rope=model.pos_embed(image_ids[0])
    for kind,blocks in [('joint',model.transformer_blocks),('single',model.single_transformer_blocks)]:
        for index,block in enumerate(blocks):
            k,v=cache(kind,index);before=image
            operation=joint_image if kind=='joint' else single_image
            image=operation(block,image,common['imageMod' if kind=='joint' else 'singleMod'],rope,k,v,native,attention)
            on_layer(kind,index,k,v,before,image)
    capture('terminalImage',image)
    return model.proj_out(model.norm_out(image,common['temb']))


def matrix_account(text=512,image=1024,dim=3072,mlp=9216,clocks=4,joint=5,single=20):
    formations=clocks*(joint+single);updates=formations-clocks
    compiler_kv=formations*2*text*dim*dim
    compiler_updates=updates*(text*(2*dim*dim+3*dim*mlp)+2*text*text*dim)
    return {'textKVFormations':formations,'liveTextUpdates':updates,'compilerMatrixMACs':compiler_kv+compiler_updates,
            'cacheBytesBF16':formations*2*text*dim*2,
            'removedImageToTextAttentionMACs':updates*2*text*image*dim,
            'terminalDeadTextMACs':clocks*text*(2*dim*dim+3*dim*mlp)+clocks*2*text*(text+image)*dim}


def mechanics():
    """Tiny checkpoint-free native-block reference for this altered architecture."""
    import torch
    import diffusers.models.transformers.transformer_flux2 as native
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(68)
        model=native.Flux2Transformer2DModel(in_channels=4,num_layers=2,num_single_layers=3,num_attention_heads=2,attention_head_dim=8,
            joint_attention_dim=12,timestep_guidance_channels=16,axes_dims_rope=(2,2,2,2),guidance_embeds=False).double().eval().requires_grad_(False)
        embedding=torch.arange(36,dtype=torch.float64).reshape(1,3,12)/40
        tid=torch.cartesian_prod(torch.arange(1),torch.arange(1),torch.arange(1),torch.arange(3)).unsqueeze(0)
        iid=torch.arange(16).reshape(1,4,4);image=torch.arange(16,dtype=torch.float64).reshape(1,4,4)/20
        reference=image.clone();all_rows=[];errors=[];inputs=[];terminal_calls=[]
        with torch.inference_mode():
            for clock,t in enumerate([.88398,.71749]):
                arg=torch.tensor([t],dtype=torch.float64);cache={};states={}
                def compiled(kind,index,before,k,v,after):
                    cache[kind,index]=(k.clone(),v.clone());states[kind,index]=before.clone()
                    all_rows.append({'clock':clock,'kind':kind,'index':index,'updated':after is not None})
                common=compile_clock(model,embedding,arg,tid,native,native.dispatch_attention_fn,compiled,
                    lambda w,f:terminal_calls.append(list(w.shape)) or f())
                text=model.context_embedder(embedding);text_rope=model.pos_embed(tid[0]);image_rope=model.pos_embed(iid[0])
                joint_rope=tuple(torch.cat([tr,ir],0) for tr,ir in zip(text_rope,image_rope));ri=model.x_embedder(reference);wanted=[]
                for kind,blocks in [('joint',model.transformer_blocks),('single',model.single_transformer_blocks)]:
                    for index,b in enumerate(blocks):
                        torch.testing.assert_close(states[kind,index],text,rtol=2e-6,atol=2e-7)
                        captured=[];original=native.dispatch_attention_fn
                        def record(q,k,v,**kw):captured.append((k.clone(),v.clone()));return original(q,k,v,**kw)
                        try:
                            native.dispatch_attention_fn=record
                            if kind=='joint':
                                next_text,_=b(hidden_states=torch.empty(1,0,16,dtype=torch.float64),encoder_hidden_states=text,
                                    temb_mod_img=common['imageMod'],temb_mod_txt=common['textMod'],image_rotary_emb=text_rope)
                            else:next_text=b(hidden_states=text,encoder_hidden_states=None,temb_mod=common['singleMod'],image_rotary_emb=text_rope)
                        finally:native.dispatch_attention_fn=original
                        for a,bv in zip(cache[kind,index],captured[0]):torch.testing.assert_close(a,bv,rtol=2e-6,atol=2e-7)
                        if kind=='joint':
                            _,ri=b(hidden_states=ri,encoder_hidden_states=text,temb_mod_img=common['imageMod'],temb_mod_txt=common['textMod'],image_rotary_emb=joint_rope)
                        else:ri=b(hidden_states=torch.cat([text,ri],1),encoder_hidden_states=None,temb_mod=common['singleMod'],image_rotary_emb=joint_rope)[:,3:]
                        wanted.append(ri.clone());text=next_text
                count=[0]
                def consumed(kind,index,k,v,before,after):
                    torch.testing.assert_close(after,wanted[count[0]],rtol=2e-6,atol=2e-7);count[0]+=1
                inputs.append(image.clone());got=consume_clock(model,image,iid,common,lambda kind,index:cache[kind,index],native,native.dispatch_attention_fn,consumed,lambda n,v:None)
                ref=model.proj_out(model.norm_out(ri,common['temb']));torch.testing.assert_close(got,ref,rtol=2e-6,atol=2e-7)
                errors.append(float((got-ref).abs().max()));image=image-.125*got;reference=reference-.125*ref
        return {'schema':'tce-prompt-program-toy-v1','blocks':all_rows,'maxAbsVelocityErrors':errors,'ownInputChanges':not torch.equal(inputs[0],inputs[1]),
            'terminalKVMapShapes':terminal_calls,'incomingKVMatchesNativeTextOnly':True,'actorMatchesNativeImageAtSameIncomingText':True,
            'relativeTolerance':2e-6,'absoluteTolerance':2e-7,'pretrainedModelEvaluations':0,
            'scope':'Tiny supplied double model; text-only native references and same-incoming-text native image consumers over two own commits. Native rotary finite policy retained. No real-model quality or BF16 equivalence claim.'}
