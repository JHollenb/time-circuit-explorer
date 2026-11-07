"""Compute only consumed image outputs of an ordinary final FLUX single block."""
PREFIX='single_transformer_blocks.19.attn.'
PARAMETERS=[PREFIX+n for n in ['to_qkv_mlp_proj.weight','norm_q.weight','norm_k.weight','to_out.weight']]+['norm_out.linear.weight','proj_out.weight']


def slices(fused,dim,mlp):
    import torch
    if list(fused.shape)!=[3*dim+2*mlp,dim]:raise ValueError('Fused native parameter layout differs')
    # Materialized maps: no full fused projection followed by discarded outputs.
    return {'KV':fused[dim:3*dim].clone(),
            'QGU':torch.cat([fused[:dim],fused[3*dim:]],dim=0)}


def account(text,image,dim,mlp):
    rows=text+image
    return {'nativeAffine':rows*dim*(3*dim+2*mlp)+rows*(dim+mlp)*dim,
            'candidateAffine':rows*dim*2*dim+image*dim*(dim+2*mlp)+image*(dim+mlp)*dim,
            'nativeAttention':2*rows*rows*dim,'candidateAttention':2*image*rows*dim}


def compute(rows,mod,cosine,sine,weights,linear,attention,native,text=512,heads=24,eps=1e-6):
    import torch
    import torch.nn.functional as F
    dim=rows.shape[-1];mlp=weights['out'].shape[1]-dim;image=rows.shape[1]-text
    if rows.ndim!=3 or rows.shape[0]!=1 or image<=0 or dim%heads or cosine.shape!=sine.shape or list(cosine.shape)!=[rows.shape[1],dim//heads]:
        raise ValueError('Terminal source geometry differs')
    shift,scale,gate=native.Flux2Modulation.split(mod,1)[0]
    normalized=F.layer_norm(rows,(dim,),eps=eps)
    normalized=(1+scale)*normalized+shift
    kv=linear(normalized,weights['KV'])
    qgu=linear(normalized[:,text:],weights['QGU'])
    key,value=kv.chunk(2,dim=-1);query,gu=qgu.split([dim,2*mlp],dim=-1)
    query=query.unflatten(-1,(heads,-1));key=key.unflatten(-1,(heads,-1));value=value.unflatten(-1,(heads,-1))
    query=F.rms_norm(query,(dim//heads,),weights['qNorm'],eps=eps)
    key=F.rms_norm(key,(dim//heads,),weights['kNorm'],eps=eps)
    query=native.apply_rotary_emb(query,(cosine[text:],sine[text:]),sequence_dim=1)
    key=native.apply_rotary_emb(key,(cosine,sine),sequence_dim=1)
    h=attention(query,key,value,attn_mask=None,backend=None,parallel_config=None).flatten(2,3).to(query.dtype)
    product=native.Flux2SwiGLU()(gu)
    update=linear(torch.cat([h,product],dim=-1),weights['out'])
    result=rows[:,text:]+gate*update
    if result.dtype==torch.float16:result=result.clip(-65504,65504)
    # Match the declared native image-view ABI, without allocating/writing text outputs.
    image_output=torch.empty_strided(result.shape,(rows.shape[1]*dim,dim,1),device=result.device,dtype=result.dtype)
    image_output.copy_(result)
    return image_output


def mechanics():
    """Small supplied FP64 real-law fixture; no checkpoint or scientific inputs."""
    import torch
    import torch.nn.functional as F
    import diffusers.models.transformers.transformer_flux2 as native
    text,image,dim,mlp,heads=3,4,8,12,2;n=text+image
    values=lambda shape,scale:torch.arange(__import__('math').prod(shape),dtype=torch.float64).reshape(shape).remainder(17).sub(8).mul(scale)
    x=values((1,n,dim),.05);mod=values((1,3*dim),.02);w=values((3*dim+2*mlp,dim),.01)
    weights={**slices(w,dim,mlp),'out':values((dim,dim+mlp),.02),'qNorm':torch.ones(dim//heads,dtype=torch.float64),'kNorm':torch.ones(dim//heads,dtype=torch.float64)}
    angles=torch.arange(n*(dim//heads),dtype=torch.float64).reshape(n,dim//heads)*.03
    cosine=angles.cos();sine=angles.sin();ops=[];att=[]
    def attention(q,k,v,**kwargs):
        return ((q.transpose(1,2)@k.transpose(1,2).transpose(-1,-2)/(dim//heads)**.5).softmax(-1)@v.transpose(1,2)).transpose(1,2)
    def linear(a,b):ops.append((list(a.shape),list(b.shape)));return F.linear(a,b)
    def demand(q,k,v,**kwargs):att.append((list(q.shape),list(k.shape),list(v.shape)));return attention(q,k,v)
    got=compute(x,mod,cosine,sine,weights,linear,demand,native,text=text,heads=heads)
    shift,scale,gate=native.Flux2Modulation.split(mod,1)[0];z=(1+scale)*F.layer_norm(x,(dim,),eps=1e-6)+shift
    fused=F.linear(z,w);q,k,v,gu=fused.split([dim,dim,dim,2*mlp],dim=-1)
    q=F.rms_norm(q.unflatten(-1,(heads,-1)),(dim//heads,),weights['qNorm'],eps=1e-6)
    k=F.rms_norm(k.unflatten(-1,(heads,-1)),(dim//heads,),weights['kNorm'],eps=1e-6);v=v.unflatten(-1,(heads,-1))
    q=native.apply_rotary_emb(q,(cosine,sine),sequence_dim=1);k=native.apply_rotary_emb(k,(cosine,sine),sequence_dim=1)
    full=x+gate*F.linear(torch.cat([attention(q,k,v).flatten(2,3),native.Flux2SwiGLU()(gu)],dim=-1),weights['out'])
    torch.testing.assert_close(got,full[:,text:],rtol=1e-12,atol=1e-12)
    if ops!=[([1,n,dim],[2*dim,dim]),([1,image,dim],[dim+2*mlp,dim]),([1,image,dim+mlp],[dim,dim+mlp])]:raise ValueError('Physical demand projection geometry differs')
    if att!=[([1,image,heads,dim//heads],[1,n,heads,dim//heads],[1,n,heads,dim//heads])]:raise ValueError('Rectangular full-bank attention differs')
    if got.stride()!=(n*dim,dim,1) or not torch.equal(weights['KV'],w[dim:3*dim]) or not torch.equal(weights['QGU'],torch.cat([w[:dim],w[3*dim:]],0)):
        raise ValueError('Weight slicing or consumed image-view ABI differs')
    cost=account(text,image,dim,mlp)
    return {'schema':'tce-terminal-liveness-toy-mechanics-v1','completeImageMatchesFP64Toy':True,'absoluteTolerance':1e-12,'relativeTolerance':1e-12,
            'projectionShapes':ops,'attentionShapes':att,'modelEvaluations':0,'toyNativeAndCandidateMatrixMACs':sum(cost.values()),
            'scope':'Fixed tiny supplied FP64 weights/state with original rotary primitive in both arms, including its internal finite policy. No native BF16 parity or model quality certification.'}
