"""Frozen mean-Up law on live stream domains, composed through own model states."""


def completion_abis(index,dim=3072,mlp=9216,text=512,image=1024,heads=24):
    n=image if index==19 else text+image;streams=1 if index==19 else 2
    shapes={'normalized':[1,n,dim],'activation':[1,n,mlp],'upRaw':[1,n,mlp//4],
        'completedUp':[1,n,mlp],'product':[1,n,mlp],'streamMeanFP32':[1,streams,dim],
        'streamMean':[1,streams,dim],'anchorUp':[1,streams,mlp],'preResidualUpdate':[1,n,dim],
        'gate':[1,1,dim],'imageOut':[1,image,dim]}
    if index!=19:shapes.update(textOut=[1,text,dim],alpha=[1,heads,text],textRead=[1,text,dim])
    result={name:{'shape':shape,'dtype':'torch.float32' if name in ['streamMeanFP32','alpha'] else 'torch.bfloat16'} for name,shape in shapes.items()}
    for name,shape,dtype in [('rowNormSquared',[n],'torch.float32'),('weightedGateEnergy',[mlp],'torch.float32'),('score',[mlp],'torch.float32'),
        ('rankedIndices',[mlp],'torch.int64'),('support',[mlp//4],'torch.int64'),('omittedScoreSum',[],'torch.float32'),('selectedScoreSum',[],'torch.float32')]:
        result['selection_'+name]={'shape':shape,'dtype':dtype}
    return result


def single(block,text,image,mod,rope,native,attention,product,moment_capture,with_maps,index,norms,on_selection,count=None,terminal=False):
    import torch
    import torch.nn.functional as F
    from gate_first_mlp import select,current_gates
    from image_partition_moments import text_read
    a=block.attn;d=a.inner_dim;m=a.mlp_hidden_dim;nt=text.shape[1];rows=torch.cat([text,image],1)
    count=m//4 if count is None else count
    shift,scale,gate=native.Flux2Modulation.split(mod,1)[0];fused=a.to_qkv_mlp_proj.weight
    if terminal:
        maps={'KV':fused[d:3*d].clone(),'QG':torch.cat([fused[:d],fused[3*d:3*d+m]],0)}
        prepared=with_maps(index,maps,lambda:current_gates(rows,mod,maps,F.linear,native,text=nt,eps=block.norm.eps));del maps
        z=prepared['normalizedImage'];activation=prepared['activation'];q=prepared['queryRaw'];k=prepared['keyRaw'];v=prepared['valueRaw']
        mean32=z.float().mean(1,keepdim=True);live=rows[:,nt:]
    else:
        z=(1+scale)*block.norm(rows)+shift
        maps={'QKVG':fused[:3*d+m].clone()}
        q,k,v,g=with_maps(index,maps,lambda:F.linear(z,maps['QKVG'])).split([d,d,d,m],-1);del maps
        activation=F.silu(g);mean32=torch.stack([z[:,:nt].float().mean(1),z[:,nt:].float().mean(1)],1);live=rows
    selection=select(z,activation,norms['sourceUpNorm'][index],norms['sourceDownNorm'][index],count)
    on_selection(index,selection);support=selection['support'];mean=mean32.to(z.dtype)
    original_up=fused[3*d+m:]
    maps={'USelected':original_up.index_select(0,support).clone()}
    up=with_maps(index,maps,lambda:F.linear(z,maps['USelected']));del maps
    maps={'UAnchor':original_up.clone()}
    anchor=with_maps(index,maps,lambda:F.linear(mean,maps['UAnchor']));del maps
    completed=anchor.expand(-1,image.shape[1],-1).clone() if terminal else torch.cat([anchor[:,:1].expand(-1,nt,-1),anchor[:,1:].expand(-1,image.shape[1],-1)],1)
    completed.index_copy_(-1,support,up);admitted=activation*completed
    q=a.norm_q(q.unflatten(-1,(a.heads,-1)));k=a.norm_k(k.unflatten(-1,(a.heads,-1)));v=v.unflatten(-1,(a.heads,-1))
    q=native.apply_rotary_emb(q,rope,sequence_dim=1) if not terminal else native.apply_rotary_emb(q,(rope[0][nt:],rope[1][nt:]),sequence_dim=1)
    k=native.apply_rotary_emb(k,rope,sequence_dim=1)
    ih=attention(q if terminal else q[:,nt:],k,v,attn_mask=None,backend=None,parallel_config=None).flatten(2,3).to(q.dtype)
    th=alpha=None
    if terminal:h=ih
    else:
        th,alpha=text_read(q[:,:nt],k,v[:,:nt],product,moment_capture);h=torch.cat([th,ih],1)
    update=F.linear(torch.cat([h,admitted],-1),a.to_out.weight);result=live+gate*update
    if result.dtype==torch.float16:result=result.clip(-65504,65504)
    if terminal:
        own=torch.empty_strided(result.shape,(rows.shape[1]*d,d,1),dtype=result.dtype,device=result.device)
        own.copy_(result);result=own
    got={'normalized':z,'activation':activation,'upRaw':up,'completedUp':completed,'product':admitted,'streamMeanFP32':mean32,
        'streamMean':mean,'anchorUp':anchor,'preResidualUpdate':update,'gate':gate,'imageOut':result if terminal else result[:,nt:]}
    if not terminal:got.update(textOut=result[:,:nt],alpha=alpha,textRead=th)
    got.update({'selection_'+name:value for name,value in selection.items()})
    return got


def denoise(model,inputs,native,attention,capture,on_block,with_maps,product,moment_capture,norms,on_selection,run_completion,count=None):
    import torch
    from fixed_text_memory import prepare
    from image_partition_moments import joint
    image,text,temb,im,tm,sm,rope=prepare(model,inputs,native,capture)
    for index,block in enumerate(model.transformer_blocks):
        bt,bi=text,image;text,image,alpha,read=joint(block,text,image,im,tm,rope,native,attention,product,moment_capture)
        on_block('joint',index,bt,bi,text,image,alpha,read)
    for index,block in enumerate(model.single_transformer_blocks):
        bt,bi=text,image;terminal=index==len(model.single_transformer_blocks)-1
        if terminal:capture('terminalRows',torch.cat([text,image],1))
        got=run_completion(index,block,text,image,sm,rope,lambda:single(block,text,image,sm,rope,native,attention,product,moment_capture,with_maps,index,norms,on_selection,count,terminal))
        text=got.get('textOut');image=got['imageOut'];on_block('single',index,bt,bi,text,image,got.get('alpha'),got.get('textRead'))
        del got
    for name,value in [('terminalImage',image),('singleMod',sm),('temb',temb),('cosine',rope[0]),('sine',rope[1])]:capture(name,value)
    velocity=model.proj_out(model.norm_out(image,temb))
    return (velocity,) if not inputs.get('return_dict',True) else native.Flux2Transformer2DModelOutput(sample=velocity)
