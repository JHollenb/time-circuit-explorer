"""Image evolution with source-projected text memory fixed at every block."""
from terminal_liveness import slices,compute


def prepare(model,inputs,native,capture):
    import torch
    if set(inputs)-{'hidden_states','encoder_hidden_states','timestep','img_ids','txt_ids','guidance','joint_attention_kwargs','return_dict','kv_cache','kv_cache_mode','num_ref_tokens','ref_fixed_timestep'}:
        raise ValueError('Undeclared fixed-memory forward input')
    if torch.is_grad_enabled() or model.training or model.is_cache_enabled or getattr(model,'_hf_hook',None) is not None:raise ValueError('Ordinary inference without cache/offload required')
    if any(inputs.get(n) is not None for n in ['guidance','kv_cache','kv_cache_mode']) or inputs.get('num_ref_tokens',0)!=0 or inputs.get('ref_fixed_timestep',0.)!=0. or inputs.get('joint_attention_kwargs') not in (None,{}):
        raise ValueError('Guidance/reference/cache/attention override outside this candidate')
    image,text=inputs['hidden_states'],inputs['encoder_hidden_states'];iid,tid=inputs['img_ids'],inputs['txt_ids']
    if image.shape[0]!=1 or text.shape[0]!=1 or iid.ndim!=3 or tid.ndim!=3:raise ValueError('One ordinary supplied image batch required')
    time=inputs['timestep'].to(image.dtype)*1000;temb=model.time_guidance_embed(time,None)
    im=model.double_stream_modulation_img(temb);tm=model.double_stream_modulation_txt(temb);sm=model.single_stream_modulation(temb)
    image=model.x_embedder(image);text=model.context_embedder(text);capture('context',text);capture('imageStart',image)
    ir,tr=model.pos_embed(iid[0]),model.pos_embed(tid[0]);rope=(torch.cat([tr[0],ir[0]],0),torch.cat([tr[1],ir[1]],0))
    return image,text,temb,im,tm,sm,rope


def joint(block,text,image,img_mod,txt_mod,rope,native,attention):
    import torch
    (shift,scale,gate),(ms,mc,mg)=native.Flux2Modulation.split(img_mod,2)
    (ts,tc,_),_=native.Flux2Modulation.split(txt_mod,2)
    iz=(1+scale)*block.norm1(image)+shift;tz=(1+tc)*block.norm1_context(text)+ts;a=block.attn
    q=a.norm_q(a.to_q(iz).unflatten(-1,(a.heads,-1)))
    ik=a.norm_k(a.to_k(iz).unflatten(-1,(a.heads,-1)));iv=a.to_v(iz).unflatten(-1,(a.heads,-1))
    tk=a.norm_added_k(a.add_k_proj(tz).unflatten(-1,(a.heads,-1)));tv=a.add_v_proj(tz).unflatten(-1,(a.heads,-1))
    k=torch.cat([tk,ik],1);v=torch.cat([tv,iv],1)
    q=native.apply_rotary_emb(q,(rope[0][text.shape[1]:],rope[1][text.shape[1]:]),sequence_dim=1)
    k=native.apply_rotary_emb(k,rope,sequence_dim=1)
    h=attention(q,k,v,attn_mask=None,backend=None,parallel_config=None).flatten(2,3).to(q.dtype)
    image=image+gate*a.to_out[1](a.to_out[0](h))
    z=block.norm2(image);z=z*(1+mc)+ms
    return image+mg*block.ff(z)


def denoise(model,inputs,native,attention,capture,on_block,with_maps):
    import torch
    image,text,temb,im,tm,sm,rope=prepare(model,inputs,native,capture)
    for index,block in enumerate(model.transformer_blocks):
        before=image;image=joint(block,text,image,im,tm,rope,native,attention);on_block('joint',index,text,before,image)
    for index,block in enumerate(model.single_transformer_blocks):
        rows=torch.cat([text,image],1)
        if index==len(model.single_transformer_blocks)-1:capture('terminalRows',rows)
        a=block.attn;maps={**slices(a.to_qkv_mlp_proj.weight,a.query_dim,a.mlp_hidden_dim),
            'qNorm':a.norm_q.weight,'kNorm':a.norm_k.weight,'out':a.to_out.weight}
        before=image
        image=with_maps(index,maps,lambda:compute(rows,sm,rope[0],rope[1],maps,torch.nn.functional.linear,attention,native,
            text=text.shape[1],heads=a.heads,eps=block.norm.eps))
        on_block('single',index,text,before,image)
        del maps,rows,before
    capture('terminalImage',image);capture('singleMod',sm);capture('temb',temb);capture('cosine',rope[0]);capture('sine',rope[1])
    output=model.proj_out(model.norm_out(image,temb))
    return (output,) if not inputs.get('return_dict',True) else native.Flux2Transformer2DModelOutput(sample=output)


def mechanics():
    """Tiny native block images at the same fixed text; no pretrained model."""
    import torch
    import diffusers.models.transformers.transformer_flux2 as native
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(67)
        model=native.Flux2Transformer2DModel(in_channels=4,num_layers=2,num_single_layers=3,num_attention_heads=2,attention_head_dim=8,
            joint_attention_dim=12,timestep_guidance_channels=16,axes_dims_rope=(2,2,2,2),guidance_embeds=False).double().eval().requires_grad_(False)
        x=torch.arange(16,dtype=torch.float64).reshape(1,4,4)/20;embedding=torch.arange(36,dtype=torch.float64).reshape(1,3,12)/40
        candidate=x.clone();reference=x.clone();errors=[];blocks=[];inputs=[];map_live=0;peak_maps=0
        def invoke(index,maps,operation):
            nonlocal map_live,peak_maps
            map_live+=1;peak_maps=max(peak_maps,map_live)
            try:return operation()
            finally:map_live-=1
        with torch.inference_mode():
            for clock,time in enumerate([.88398,.71749]):
                args={'encoder_hidden_states':embedding,'timestep':torch.tensor([time],dtype=torch.float64),'img_ids':torch.arange(16).reshape(1,4,4),'txt_ids':torch.arange(12).reshape(1,3,4),'return_dict':False}
                image,text,temb,im,tm,sm,rope=prepare(model,{'hidden_states':reference,**args},native,lambda n,v:None);fixed=text.clone();wanted=[]
                for index,b in enumerate(model.transformer_blocks):
                    _,image=b(hidden_states=image,encoder_hidden_states=text,temb_mod_img=im,temb_mod_txt=tm,image_rotary_emb=rope,joint_attention_kwargs=None);wanted.append(image.clone())
                for index,b in enumerate(model.single_transformer_blocks):
                    rows=b(hidden_states=torch.cat([text,image],1),encoder_hidden_states=None,temb_mod=sm,image_rotary_emb=rope,joint_attention_kwargs=None);image=rows[:,text.shape[1]:];wanted.append(image.clone())
                ref=model.proj_out(model.norm_out(image,temb));observed=[]
                def block(kind,index,actual_text,before,after):
                    torch.testing.assert_close(actual_text,fixed,rtol=0,atol=0);torch.testing.assert_close(after,wanted[len(observed)],rtol=2e-6,atol=2e-7)
                    observed.append(after.clone());blocks.append({'clock':clock,'kind':kind,'index':index,'imageShape':list(after.shape),'textFixed':True})
                inputs.append(candidate.clone());got=denoise(model,{'hidden_states':candidate,**args},native,native.dispatch_attention_fn,lambda n,v:None,block,invoke)[0]
                torch.testing.assert_close(got,ref,rtol=2e-6,atol=2e-7);errors.append(float((got-ref).abs().max()));candidate=candidate-.125*got;reference=reference-.125*ref
        return {'schema':'tce-fixed-text-memory-toy-v1','maxAbsVelocityErrors':errors,'blocks':blocks,'ownInputChanges':not torch.equal(inputs[0],inputs[1]),
            'maxConcurrentPhysicalMapSets':peak_maps,'textRows':3,'imageRows':4,'pretrainedModelEvaluations':0,'relativeTolerance':2e-6,'absoluteTolerance':2e-7,
            'scope':'Tiny double supplied model, two own toy commits; compares reduced image law to native block image output with C reset before every block. Native FP32 rotary finite policy retained. Not native BF16 parity or scientific quality certification.'}
