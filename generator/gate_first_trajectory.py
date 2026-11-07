"""Current-input quarter MLPs, composed through both streams and own clocks."""
from fixed_text_memory import prepare
from image_partition_moments import joint,text_read
from gate_first_mlp import select,selected_maps,current_gates,image


def source_norms(blocks):
    import torch
    up=[];down=[]
    with torch.autocast(device_type=blocks[0].attn.to_out.weight.device.type,enabled=False):
        for block in blocks:
            a=block.attn;d=a.inner_dim;m=a.mlp_hidden_dim
            up.append(a.to_qkv_mlp_proj.weight[3*d+m:].float().square().sum(1).sqrt())
            down.append(a.to_out.weight[:,d:].float().square().sum(0).sqrt())
    return {'sourceUpNorm':torch.stack(up),'sourceDownNorm':torch.stack(down)}


def single(block,text,image_state,mod,rope,native,attention,product,moment_capture,with_maps,index,norms,on_selection,count=None,terminal=False):
    import torch
    import torch.nn.functional as F
    a=block.attn;dim=a.inner_dim;mlp=a.mlp_hidden_dim;nt=text.shape[1];rows=torch.cat([text,image_state],1)
    count=mlp//4 if count is None else count
    shift,scale,gate=native.Flux2Modulation.split(mod,1)[0]
    fused,out=a.to_qkv_mlp_proj.weight,a.to_out.weight
    if terminal:
        base={'KV':fused[dim:3*dim].clone(),'QG':torch.cat([fused[:dim],fused[3*dim:3*dim+mlp]],0)}
        prepared=with_maps(index,base,lambda:current_gates(rows,mod,base,F.linear,native,text=nt,eps=block.norm.eps))
        del base
        selected=select(prepared['normalizedImage'],prepared['activation'],norms['sourceUpNorm'][index],norms['sourceDownNorm'][index],count)
        on_selection(index,selected)
        maps=selected_maps(fused,out,selected['support'],dim,mlp)
        maps.update(qNorm=a.norm_q.weight,kNorm=a.norm_k.weight)
        got=with_maps(index,maps,lambda:image(rows,mod,rope[0],rope[1],prepared,selected['support'],maps,F.linear,attention,native,text=nt,heads=a.heads,eps=block.norm.eps))
        return None,got['ownImage'],None,None
    z=(1+scale)*block.norm(rows)+shift
    base={'QKVG':fused[:3*dim+mlp].clone()}
    q,k,v,g=with_maps(index,base,lambda:F.linear(z,base['QKVG'])).split([dim,dim,dim,mlp],-1)
    del base
    activation=F.silu(g)
    selected=select(z,activation,norms['sourceUpNorm'][index],norms['sourceDownNorm'][index],count)
    on_selection(index,selected)
    q=a.norm_q(q.unflatten(-1,(a.heads,-1)));k=a.norm_k(k.unflatten(-1,(a.heads,-1)));v=v.unflatten(-1,(a.heads,-1))
    q=native.apply_rotary_emb(q,rope,sequence_dim=1);k=native.apply_rotary_emb(k,rope,sequence_dim=1)
    ih=attention(q[:,nt:],k,v,attn_mask=None,backend=None,parallel_config=None).flatten(2,3).to(q.dtype)
    th,alpha=text_read(q[:,:nt],k,v[:,:nt],product,moment_capture)
    maps=selected_maps(fused,out,selected['support'],dim,mlp)
    def emit():
        up=F.linear(z,maps['USelected']);p=activation.index_select(-1,selected['support'])*up
        return rows+gate*F.linear(torch.cat([torch.cat([th,ih],1),p],-1),maps['outSelected'])
    result=with_maps(index,maps,emit)
    if result.dtype==torch.float16:result=result.clip(-65504,65504)
    return result[:,:nt],result[:,nt:],alpha,th


def denoise(model,inputs,native,attention,capture,on_block,with_maps,product,moment_capture,norms,on_selection,count=None):
    import torch
    image_state,text,temb,im,tm,sm,rope=prepare(model,inputs,native,capture)
    for index,block in enumerate(model.transformer_blocks):
        bt,bi=text,image_state;text,image_state,alpha,read=joint(block,text,image_state,im,tm,rope,native,attention,product,moment_capture)
        on_block('joint',index,bt,bi,text,image_state,alpha,read)
    for index,block in enumerate(model.single_transformer_blocks):
        bt,bi=text,image_state;terminal=index==len(model.single_transformer_blocks)-1
        if terminal:capture('terminalRows',torch.cat([text,image_state],1))
        text,image_state,alpha,read=single(block,text,image_state,sm,rope,native,attention,product,moment_capture,with_maps,index,norms,on_selection,count,terminal)
        on_block('single',index,bt,bi,text,image_state,alpha,read)
    for name,value in [('terminalImage',image_state),('singleMod',sm),('temb',temb),('cosine',rope[0]),('sine',rope[1])]:capture(name,value)
    velocity=model.proj_out(model.norm_out(image_state,temb))
    return (velocity,) if not inputs.get('return_dict',True) else native.Flux2Transformer2DModelOutput(sample=velocity)


def mechanics():
    """Independent full/selected two-stream references and two own commits."""
    import torch
    import torch.nn.functional as F
    import diffusers.models.transformers.transformer_flux2 as native
    from image_partition_moments import mechanics as moment_mechanics
    from gate_first_mlp import mechanics as terminal_mechanics
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(72)
        model=native.Flux2Transformer2DModel(in_channels=4,num_layers=1,num_single_layers=2,num_attention_heads=2,attention_head_dim=8,
            joint_attention_dim=12,timestep_guidance_channels=16,axes_dims_rope=(2,2,2,2),guidance_embeds=False).double().eval().requires_grad_(False)
        norms=source_norms(model.single_transformer_blocks);text=torch.randn(1,3,16,dtype=torch.float64);image_state=torch.randn(1,4,16,dtype=torch.float64)
        mod=torch.randn(1,48,dtype=torch.float64)*.02;angles=torch.randn(7,8,dtype=torch.float64)*.1;rope=(angles.cos(),angles.sin())
        b=model.single_transformer_blocks[0];errors=[];supports=[];map_names=[]
        def maps(index,values,op):map_names.append(sorted(n for n in values if n not in ['qNorm','kNorm']));return op()
        def reference(t,i,selected,terminal=False):
            a=b.attn;d=a.inner_dim;m=a.mlp_hidden_dim;rows=torch.cat([t,i],1);nt=t.shape[1]
            sh,sc,ga=native.Flux2Modulation.split(mod,1)[0];z=(1+sc)*b.norm(rows)+sh
            q,k,v,g,u=F.linear(z,a.to_qkv_mlp_proj.weight).split([d,d,d,m,m],-1)
            q=native.apply_rotary_emb(a.norm_q(q.unflatten(-1,(2,-1))),rope,sequence_dim=1)
            k=native.apply_rotary_emb(a.norm_k(k.unflatten(-1,(2,-1))),rope,sequence_dim=1);v=v.unflatten(-1,(2,-1))
            ih=native.dispatch_attention_fn(q[:,nt:],k,v,attn_mask=None,backend=None,parallel_config=None).flatten(2,3)
            qt=q[:,:nt].transpose(1,2).float();kt=k.transpose(1,2).float();vt=v[:,:nt].transpose(1,2).float();scale=q.shape[-1]**-.5
            scores=qt@kt[:,:,nt:].transpose(-1,-2)*scale;mean=scores.mean(-1);log_i=__import__('math').log(i.shape[1])+mean+(scores-mean.unsqueeze(-1)).square().mean(-1)*.5
            tt=qt@kt[:,:,:nt].transpose(-1,-2)*scale;alpha=torch.sigmoid(tt.logsumexp(-1)-log_i)
            th=((tt.softmax(-1)@vt)*alpha.unsqueeze(-1)).transpose(1,2).flatten(2,3).to(q.dtype)
            h=torch.cat([th,ih],1);p=F.silu(g)*u
            if terminal:h=h[:,nt:];p=p[:,nt:];rows=rows[:,nt:]
            # Full dense native operands are evaluated only in this tiny independent reference.
            update=F.linear(h,a.to_out.weight[:,:d])+F.linear(p.index_select(-1,selected),a.to_out.weight[:,d:].index_select(1,selected))
            return rows+ga*update
        for count in [b.attn.mlp_hidden_dim,b.attn.mlp_hidden_dim//4]:
            ct,ci=text.clone(),image_state.clone();rt,ri=text.clone(),image_state.clone()
            for clock in range(2):
                selection=[]
                got=single(b,ct,ci,mod,rope,native,native.dispatch_attention_fn,lambda kind,a,b:a@b,None,maps,0,norms,lambda index,s:selection.append(s),count)
                s=selection[0];want=reference(rt,ri,s['support']);torch.testing.assert_close(torch.cat(got[:2],1),want,rtol=4e-6,atol=4e-7)
                errors.append(float((torch.cat(got[:2],1)-want).abs().max()));supports.append(s['support'].tolist())
                ct,ci=ct-.125*got[0],ci-.125*got[1];rt,ri=rt-.125*want[:,:3],ri-.125*want[:,3:]
            selection=[];terminal=single(b,ct,ci,mod,rope,native,native.dispatch_attention_fn,lambda kind,a,b:a@b,None,maps,0,norms,lambda index,s:selection.append(s),count,True)
            want=reference(ct,ci,selection[0]['support'],True);torch.testing.assert_close(terminal[1],want,rtol=1e-12,atol=1e-12)
            if terminal[0] is not None or terminal[2:]!=(None,None):raise ValueError('Terminal text output should be absent')
        return {'schema':'tce-gate-first-trajectory-toy-v1','maxAbsTwoStreamErrors':errors,'supports':supports,'physicalMaps':map_names,
            'momentMechanics':moment_mechanics(),'terminalMechanics':terminal_mechanics(),'pretrainedModelEvaluations':0,
            'scope':'Independent tiny complete/selected dense operands, direct score-moment text reference, two own commits and terminal last-use; not pretrained BF16 parity or image quality.'}
