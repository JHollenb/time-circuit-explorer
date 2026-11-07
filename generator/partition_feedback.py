"""Live image-key competition, no image-value contribution to text writes."""
from fixed_text_memory import prepare
from terminal_liveness import slices,compute


def text_read(query,key,text_value,product):
    """Scale after FP32 QK, one full softmax, physically text-only AV."""
    import torch
    if query.ndim!=4 or key.ndim!=4 or text_value.shape!=query.shape or query.shape[0]!=1 or key.shape[1]<=query.shape[1]:
        raise ValueError('Partition attention geometry differs')
    with torch.autocast(device_type=query.device.type,enabled=False):
        q=query.transpose(1,2).float();k=key.transpose(1,2).float();v=text_value.transpose(1,2).float()
        scores=product('textScores',q,k.transpose(-1,-2));scores=scores*(query.shape[-1]**-.5)
        probability=scores.softmax(dim=-1);text_probability=probability[...,:query.shape[1]]
        alpha=text_probability.sum(dim=-1)
        output=product('textAV',text_probability,v).transpose(1,2).flatten(2,3).to(query.dtype)
    return output,alpha


def joint(block,text,image,img_mod,txt_mod,rope,native,attention,product):
    import torch
    (shift,scale,gate),(ms,mc,mg)=native.Flux2Modulation.split(img_mod,2)
    (ts,tc,tg),(tms,tmc,tmg)=native.Flux2Modulation.split(txt_mod,2)
    iz=(1+scale)*block.norm1(image)+shift;tz=(1+tc)*block.norm1_context(text)+ts;a=block.attn;nt=text.shape[1]
    iq=a.norm_q(a.to_q(iz).unflatten(-1,(a.heads,-1)));ik=a.norm_k(a.to_k(iz).unflatten(-1,(a.heads,-1)));iv=a.to_v(iz).unflatten(-1,(a.heads,-1))
    tq=a.norm_added_q(a.add_q_proj(tz).unflatten(-1,(a.heads,-1)));tk=a.norm_added_k(a.add_k_proj(tz).unflatten(-1,(a.heads,-1)));tv=a.add_v_proj(tz).unflatten(-1,(a.heads,-1))
    q=native.apply_rotary_emb(torch.cat([tq,iq],1),rope,sequence_dim=1)
    key=native.apply_rotary_emb(torch.cat([tk,ik],1),rope,sequence_dim=1);value=torch.cat([tv,iv],1)
    ih=attention(q[:,nt:],key,value,attn_mask=None,backend=None,parallel_config=None).flatten(2,3).to(q.dtype)
    th,alpha=text_read(q[:,:nt],key,tv,product)
    image=image+gate*a.to_out[1](a.to_out[0](ih));z=block.norm2(image);z=z*(1+mc)+ms;image=image+mg*block.ff(z)
    text=text+tg*a.to_add_out(th);z=block.norm2_context(text);z=z*(1+tmc)+tms;text=text+tmg*block.ff_context(z)
    if text.dtype==torch.float16:text=text.clip(-65504,65504)
    return text,image,alpha,th


def single(block,text,image,mod,rope,native,attention,product):
    import torch
    rows=torch.cat([text,image],1);nt=text.shape[1];a=block.attn;dim=a.inner_dim
    shift,scale,gate=native.Flux2Modulation.split(mod,1)[0];z=(1+scale)*block.norm(rows)+shift
    q,k,v,gu=a.to_qkv_mlp_proj(z).split([dim,dim,dim,2*a.mlp_hidden_dim],dim=-1)
    q=a.norm_q(q.unflatten(-1,(a.heads,-1)));k=a.norm_k(k.unflatten(-1,(a.heads,-1)));v=v.unflatten(-1,(a.heads,-1))
    q=native.apply_rotary_emb(q,rope,sequence_dim=1);k=native.apply_rotary_emb(k,rope,sequence_dim=1)
    ih=attention(q[:,nt:],k,v,attn_mask=None,backend=None,parallel_config=None).flatten(2,3).to(q.dtype)
    th,alpha=text_read(q[:,:nt],k,v[:,:nt],product)
    rows=rows+gate*a.to_out(torch.cat([torch.cat([th,ih],1),a.mlp_act_fn(gu)],dim=-1))
    if rows.dtype==torch.float16:rows=rows.clip(-65504,65504)
    return rows[:,:nt],rows[:,nt:],alpha,th


def denoise(model,inputs,native,attention,capture,on_block,with_maps,product):
    import torch
    image,text,temb,im,tm,sm,rope=prepare(model,inputs,native,capture)
    for index,block in enumerate(model.transformer_blocks):
        before_t,before_i=text,image;text,image,alpha,read=joint(block,text,image,im,tm,rope,native,attention,product)
        on_block('joint',index,before_t,before_i,text,image,alpha,read)
    for index,block in enumerate(model.single_transformer_blocks):
        before_t,before_i=text,image
        if index==len(model.single_transformer_blocks)-1:
            rows=torch.cat([text,image],1);capture('terminalRows',rows);a=block.attn
            maps={**slices(a.to_qkv_mlp_proj.weight,a.query_dim,a.mlp_hidden_dim),'qNorm':a.norm_q.weight,'kNorm':a.norm_k.weight,'out':a.to_out.weight}
            image=with_maps(index,maps,lambda:compute(rows,sm,rope[0],rope[1],maps,torch.nn.functional.linear,attention,native,text=text.shape[1],heads=a.heads,eps=block.norm.eps))
            on_block('single',index,before_t,before_i,None,image,None,None);del maps,rows
        else:
            text,image,alpha,read=single(block,text,image,sm,rope,native,attention,product)
            on_block('single',index,before_t,before_i,text,image,alpha,read)
    capture('terminalImage',image);capture('singleMod',sm);capture('temb',temb);capture('cosine',rope[0]);capture('sine',rope[1])
    velocity=model.proj_out(model.norm_out(image,temb))
    return (velocity,) if not inputs.get('return_dict',True) else native.Flux2Transformer2DModelOutput(sample=velocity)


def mechanics():
    """Native blocks with zero-image-value FP32 text SDPA as separate reference."""
    import torch
    import torch.nn.functional as F
    import diffusers.models.transformers.transformer_flux2 as native
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(69)
        model=native.Flux2Transformer2DModel(in_channels=4,num_layers=2,num_single_layers=3,num_attention_heads=2,attention_head_dim=8,
            joint_attention_dim=12,timestep_guidance_channels=16,axes_dims_rope=(2,2,2,2),guidance_embeds=False).double().eval().requires_grad_(False)
        x=torch.arange(16,dtype=torch.float64).reshape(1,4,4)/20;embedding=torch.arange(36,dtype=torch.float64).reshape(1,3,12)/40
        candidate=x.clone();reference=x.clone();blocks=[];errors=[];inputs=[];products=[];map_calls=[];nt=3
        with torch.inference_mode():
            for clock,time in enumerate([.88398,.71749]):
                args={'encoder_hidden_states':embedding,'timestep':torch.tensor([time],dtype=torch.float64),'img_ids':torch.arange(16).reshape(1,4,4),'txt_ids':torch.arange(12).reshape(1,3,4),'return_dict':False}
                image,text,temb,im,tm,sm,rope=prepare(model,{'hidden_states':reference,**args},native,lambda n,v:None)
                wanted=[];old=native.dispatch_attention_fn
                def reference_attention(q,k,v,**kwargs):
                    image_h=old(q[:,nt:],k,v,**kwargs)
                    value=torch.cat([v[:,:nt],torch.zeros_like(v[:,nt:])],1)
                    text_h=F.scaled_dot_product_attention(q[:,:nt].transpose(1,2).float(),k.transpose(1,2).float(),value.transpose(1,2).float()).transpose(1,2).to(q.dtype)
                    return torch.cat([text_h,image_h],1)
                try:
                    native.dispatch_attention_fn=reference_attention
                    for b in model.transformer_blocks:
                        text,image=b(hidden_states=image,encoder_hidden_states=text,temb_mod_img=im,temb_mod_txt=tm,image_rotary_emb=rope);wanted.append((text.clone(),image.clone()))
                    for b in model.single_transformer_blocks:
                        rows=b(hidden_states=torch.cat([text,image],1),encoder_hidden_states=None,temb_mod=sm,image_rotary_emb=rope)
                        text,image=rows[:,:nt],rows[:,nt:];wanted.append((text.clone(),image.clone()))
                finally:native.dispatch_attention_fn=old
                ref=model.proj_out(model.norm_out(image,temb));observed=[]
                def block(kind,index,bt,bi,after_t,after_i,alpha,read):
                    wt,wi=wanted[len(observed)];torch.testing.assert_close(after_i,wi,rtol=3e-6,atol=3e-7)
                    if after_t is not None:torch.testing.assert_close(after_t,wt,rtol=3e-6,atol=3e-7)
                    observed.append(True);blocks.append({'clock':clock,'kind':kind,'index':index,'textUpdated':after_t is not None})
                def product(kind,a,b):products.append((kind,list(a.shape),list(b.shape),str(a.dtype),str(b.dtype)));return torch.matmul(a,b)
                def maps(index,weights,operation):map_calls.append(index);return operation()
                inputs.append(candidate.clone());got=denoise(model,{'hidden_states':candidate,**args},native,old,lambda n,v:None,block,maps,product)[0]
                torch.testing.assert_close(got,ref,rtol=3e-6,atol=3e-7);errors.append(float((got-ref).abs().max()));candidate=candidate-.125*got;reference=reference-.125*ref
        return {'schema':'tce-partition-feedback-toy-v1','blocks':blocks,'maxAbsVelocityErrors':errors,'ownInputChanges':not torch.equal(inputs[0],inputs[1]),
            'explicitProducts':products,'terminalMapCalls':map_calls,'pretrainedModelEvaluations':0,'relativeTolerance':3e-6,'absoluteTolerance':3e-7,
            'scope':'Tiny supplied double model with explicit FP32 text law; separate native block reference uses full FP32 text SDPA with zero image values and ordinary image SDPA, two own commits. No native BF16 parity or quality claim.'}
