"""Current stream means complete omitted Up channels; all gates/down remain paid."""


def abis():
    from matched_single import abis as prior
    result = prior('sparse')
    result['product']['shape'] = [1,1536,9216]
    for name, shape, dtype in [('streamMeanFP32',[1,2,3072],'torch.float32'),
                               ('streamMean',[1,2,3072],'torch.bfloat16'),
                               ('anchorUp',[1,2,9216],'torch.bfloat16'),
                               ('completedUp',[1,1536,9216],'torch.bfloat16')]:
        result[name] = {'shape':shape,'dtype':dtype}
    return result


def replay(block,source,native,norms,count=None):
    import torch
    import torch.nn.functional as F
    from paired_fields import payload
    from gate_first_mlp import select
    from image_partition_moments import text_read
    a=block.attn;d=a.inner_dim;m=a.mlp_hidden_dim;nt=source['text'].shape[1]
    rows=torch.cat([source['text'],source['image']],1);n=rows.shape[1]
    count=m//4 if count is None else count
    operations=[];maps=[];products=[];moments={};captured={};attention=[]
    def linear(name,x,w):
        value=F.linear(x,w)
        operations.append({'name':name,'inputShape':list(x.shape),'inputStride':list(x.stride()),'weightShape':list(w.shape),
            'weightStride':list(w.stride()),'outputShape':list(value.shape),'outputStride':list(value.stride()),
            'dtype':str(value.dtype),'device':value.device.type,'nominalMACs':x.numel()//x.shape[-1]*w.numel()})
        return value
    def physical(name,w):
        value=w.clone()
        maps.append({'name':name,'shape':list(value.shape),'bytes':value.numel()*value.element_size(),'payloadSHA256':payload(value)})
        return value
    def product(name,left,right):
        value=torch.matmul(left,right)
        products.append({'operator':name,'leftShape':list(left.shape),'rightShape':list(right.shape),'outputShape':list(value.shape),
            'dtype':str(value.dtype),'device':value.device.type,'nominalMACs':value.numel()*left.shape[-1]})
        return value
    shift,scale,gate=native.Flux2Modulation.split(source['mod'],1)[0]
    z=(1+scale)*block.norm(rows)+shift
    qkvg=physical('QKVG',a.to_qkv_mlp_proj.weight[:3*d+m])
    q,k,v,g=linear('QKVG',z,qkvg).split([d,d,d,m],-1)
    del qkvg
    captured.update(normalized=z,queryRaw=q,keyRaw=k,valueRaw=v,gateRaw=g)
    activation=F.silu(g)
    selected=select(z,activation,norms['sourceUpNorm'][0],norms['sourceDownNorm'][0],count)
    support=selected['support']
    captured.update({'selection_'+name:value for name,value in selected.items()})
    # Mean arithmetic is part of the finite law even in a float64 synthetic toy.
    mean32=torch.stack([z[:,:nt].float().mean(dim=1),z[:,nt:].float().mean(dim=1)],dim=1)
    mean=mean32.to(z.dtype)
    original_up=a.to_qkv_mlp_proj.weight[3*d+m:]
    up_selected=physical('USelected',original_up.index_select(0,support))
    up=linear('USelected',z,up_selected)
    del up_selected
    up_anchor=physical('UAnchor',original_up)
    anchors=linear('UAnchor',mean,up_anchor)
    del up_anchor
    completed=torch.cat([anchors[:,:1].expand(-1,nt,-1),anchors[:,1:].expand(-1,n-nt,-1)],dim=1)
    completed.index_copy_(-1,support,up)
    admitted_product=activation*completed
    q=a.norm_q(q.unflatten(-1,(a.heads,-1)));k=a.norm_k(k.unflatten(-1,(a.heads,-1)))
    v=v.unflatten(-1,(a.heads,-1));rope=(source['cosine'],source['sine'])
    q=native.apply_rotary_emb(q,rope,sequence_dim=1);k=native.apply_rotary_emb(k,rope,sequence_dim=1)
    iq=q[:,nt:]
    ih=native.dispatch_attention_fn(iq,k,v,attn_mask=None,backend=None,parallel_config=None)
    attention.append({'QShape':list(iq.shape),'KShape':list(k.shape),'VShape':list(v.shape),'nominalMACs':2*iq.numel()*k.shape[1]})
    th,alpha=text_read(q[:,:nt],k,v[:,:nt],product,lambda values:moments.update(values))
    h=torch.cat([th,ih.flatten(2,3).to(q.dtype)],1)
    update=linear('outFull',torch.cat([h,admitted_product],-1),a.to_out.weight)
    result=rows+gate*update
    if result.dtype==torch.float16:result=result.clip(-65504,65504)
    captured.update(upRaw=up,activation=activation,attentionQuery=iq,attentionKey=k,attentionValue=v,imageAttention=ih,
        attentionRead=h,product=admitted_product,preResidualUpdate=update,textOut=result[:,:nt],imageOut=result[:,nt:],exit=result,
        alpha=alpha,textRead=th,gate=gate,streamMeanFP32=mean32,streamMean=mean,anchorUp=anchors,completedUp=completed)
    captured.update({'moment_'+name:value for name,value in moments.items()})
    trace={'operations':operations,'attention':attention,'products':products,'physicalMaps':maps,
        'mapScope':{'activeSets':0,'maxActiveSets':1,'maxLogicalBytes':max(x['bytes'] for x in maps)},
        'streamMeanDomains':[{'name':'text','start':0,'stop':nt},{'name':'image','start':nt,'stop':n}],
        'meanPolicy':'Actual admitted Z -> FP32 mean over each typed stream -> input dtype admission -> one stacked two-row original U affine.',
        'completionPolicy':'Broadcast own anchors; index_copy selected current Up without subtract/add reassociation; current SiLU product; original single full concat output.'}
    return captured,trace
