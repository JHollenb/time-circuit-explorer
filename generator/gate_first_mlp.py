"""Current gates choose physically evaluated up rows/down columns; learned work paid."""


def source_maps(fused,out,dim,mlp):
    import torch
    if list(fused.shape)!=[3*dim+2*mlp,dim] or list(out.shape)!=[dim,dim+mlp]:raise ValueError('Source fused map geometry differs')
    with torch.autocast(device_type=fused.device.type,enabled=False):
        up_norm=fused[3*dim+mlp:].float().square().sum(dim=1).sqrt()
        down_norm=out[:,dim:].float().square().sum(dim=0).sqrt()
    return {'KV':fused[dim:3*dim].clone(),'QG':torch.cat([fused[:dim],fused[3*dim:3*dim+mlp]],dim=0),'upNorm':up_norm,'downNorm':down_norm}


def current_gates(rows,mod,weights,linear,native,text=512,eps=1e-6):
    import torch.nn.functional as F
    dim=rows.shape[-1]
    shift,scale,_=native.Flux2Modulation.split(mod,1)[0]
    normalized=(1+scale)*F.layer_norm(rows,(dim,),eps=eps)+shift
    key,value=linear(normalized,weights['KV']).chunk(2,dim=-1)
    query_gate=linear(normalized[:,text:],weights['QG'])
    return {'normalizedImage':normalized[:,text:],'queryRaw':query_gate[...,:dim],'keyRaw':key,'valueRaw':value,'activation':F.silu(query_gate[...,dim:])}


def select(normalized,activation,up_norm,down_norm,count):
    import torch
    if normalized.ndim!=3 or normalized.shape[0]!=1 or activation.shape[:2]!=normalized.shape[:2] or up_norm.shape!=down_norm.shape or list(up_norm.shape)!=[activation.shape[-1]] or not 0<count<=activation.shape[-1]:raise ValueError('Gate selector geometry differs')
    with torch.autocast(device_type=normalized.device.type,enabled=False):
        row_norm=normalized[0].float().square().sum(dim=-1)
        energy=(activation[0].float().square()*row_norm[:,None]).sum(dim=0)
        scores=up_norm.float()*down_norm.float()*energy.sqrt()
        if not bool(torch.isfinite(scores).all()):raise ValueError('Nonfinite current gate ranking scores')
        ranking=torch.argsort(scores,descending=True,stable=True)
        support=ranking[:count].sort().values
        omitted=torch.ones(scores.shape,dtype=torch.bool,device=scores.device);omitted[support]=False
        return {'rowNormSquared':row_norm,'weightedGateEnergy':energy,'score':scores,'rankedIndices':ranking,'support':support,
            'omittedScoreSum':scores[omitted].sum(),'selectedScoreSum':scores[support].sum()}


def selected_maps(fused,out,support,dim,mlp):
    import torch
    if support.ndim!=1 or support.dtype!=torch.int64 or support.numel()==0 or not bool(torch.all(support[1:]>support[:-1])) or support[0]<0 or support[-1]>=mlp:raise ValueError('Sorted source support differs')
    return {'USelected':fused[3*dim+mlp:].index_select(0,support),
            'outSelected':torch.cat([out[:,:dim],out[:,dim:].index_select(1,support)],dim=1)}


def image(rows,mod,cosine,sine,prepared,support,weights,linear,attention,native,text=512,heads=24,eps=1e-6):
    import torch
    import torch.nn.functional as F
    dim=rows.shape[-1]
    query=F.rms_norm(prepared['queryRaw'].unflatten(-1,(heads,-1)),(dim//heads,),weights['qNorm'],eps=eps)
    key=F.rms_norm(prepared['keyRaw'].unflatten(-1,(heads,-1)),(dim//heads,),weights['kNorm'],eps=eps)
    value=prepared['valueRaw'].unflatten(-1,(heads,-1))
    query=native.apply_rotary_emb(query,(cosine[text:],sine[text:]),sequence_dim=1)
    key=native.apply_rotary_emb(key,(cosine,sine),sequence_dim=1)
    h=attention(query,key,value,attn_mask=None,backend=None,parallel_config=None).flatten(2,3).to(query.dtype)
    up=linear(prepared['normalizedImage'],weights['USelected'])
    product=prepared['activation'].index_select(-1,support)*up
    update=linear(torch.cat([h,product],dim=-1),weights['outSelected'])
    _,_,gate=native.Flux2Modulation.split(mod,1)[0]
    result=rows[:,text:]+gate*update
    if result.dtype==torch.float16:result=result.clip(-65504,65504)
    own=torch.empty_strided(result.shape,(rows.shape[1]*dim,dim,1),device=result.device,dtype=result.dtype);own.copy_(result)
    return {'ownImage':own,'selectedUp':up,'selectedProduct':product,'attentionRead':h,'preResidualUpdate':update}


def mechanics():
    """Independent tiny dense/selected references, no checkpoint or model states."""
    import math
    import torch
    import torch.nn.functional as F
    import diffusers.models.transformers.transformer_flux2 as native
    text,ni,dim,mlp,heads=3,4,8,12,2;n=text+ni
    def values(shape,scale):return torch.arange(math.prod(shape),dtype=torch.float64).reshape(shape).remainder(17).sub(8).mul(scale)
    rows=values((1,n,dim),.05);mod=values((1,3*dim),.02);fused=values((3*dim+2*mlp,dim),.013);out=values((dim,dim+mlp),.021)
    angles=torch.arange(n*(dim//heads),dtype=torch.float64).reshape(n,dim//heads)*.03;cosine,sine=angles.cos(),angles.sin()
    norm=torch.ones(dim//heads,dtype=torch.float64);maps=source_maps(fused,out,dim,mlp);ops=[]
    def linear(a,b):ops.append([list(a.shape),list(b.shape)]);return F.linear(a,b)
    def attention(q,k,v,**kwargs):return ((q.transpose(1,2)@k.transpose(1,2).transpose(-1,-2)/(dim//heads)**.5).softmax(-1)@v.transpose(1,2)).transpose(1,2)
    prepared=current_gates(rows,mod,maps,linear,native,text=text)
    shift,scale,gate=native.Flux2Modulation.split(mod,1)[0];z=(1+scale)*F.layer_norm(rows,(dim,),eps=1e-6)+shift
    q,k,v,g,u=F.linear(z,fused).split([dim,dim,dim,mlp,mlp],dim=-1)
    q=F.rms_norm(q.unflatten(-1,(heads,-1)),(dim//heads,),norm,eps=1e-6);k=F.rms_norm(k.unflatten(-1,(heads,-1)),(dim//heads,),norm,eps=1e-6)
    q=native.apply_rotary_emb(q,(cosine,sine),sequence_dim=1);k=native.apply_rotary_emb(k,(cosine,sine),sequence_dim=1)
    h=attention(q,k,v.unflatten(-1,(heads,-1))).flatten(2,3);full_product=F.silu(g)*u;results=[]
    for count in [mlp,mlp//4]:
        selection=select(prepared['normalizedImage'],prepared['activation'],maps['upNorm'],maps['downNorm'],count);support=selection['support']
        weights={**selected_maps(fused,out,support,dim,mlp),'qNorm':norm,'kNorm':norm}
        got=image(rows,mod,cosine,sine,prepared,support,weights,linear,attention,native,text=text,heads=heads)
        wanted=rows[:,text:]+gate*(F.linear(h[:,text:],out[:,:dim])+F.linear(full_product[:,text:].index_select(-1,support),out[:,dim:].index_select(1,support)))
        torch.testing.assert_close(got['ownImage'],wanted,rtol=1e-12,atol=1e-12)
        if count==mlp:
            dense=rows[:,text:]+gate*F.linear(torch.cat([h[:,text:],full_product[:,text:]],dim=-1),out)
            torch.testing.assert_close(got['ownImage'],dense,rtol=1e-12,atol=1e-12)
        a=F.silu(g[:,text:]);x=z[:,text:];exact_b=out[:,dim:].norm(dim=0)*fused[3*dim+mlp:].norm(dim=1)*((a.square()*x.square().sum(-1,keepdim=True)).sum((0,1))).sqrt()
        omitted=torch.ones(mlp,dtype=torch.bool);omitted[support]=False
        full=F.linear(full_product[:,text:],out[:,dim:]);part=F.linear(full_product[:,text:].index_select(-1,support),out[:,dim:].index_select(1,support))
        error=(full-part).norm();bound=exact_b[omitted].sum()
        if error>bound+1e-12:raise ValueError('Toy real-arithmetic omitted bound differs')
        results.append({'count':count,'support':support.tolist(),'maxAbsImageError':float((got['ownImage']-wanted).abs().max()),'omittedRealBound':float(bound),'omittedMLPError':float(error),'outputStride':list(got['ownImage'].stride())})
    return {'schema':'tce-gate-first-mlp-toy-v1','results':results,'physicalLinearShapes':ops,'pretrainedModelEvaluations':0,
        'scope':'Tiny supplied FP64 weights/state with explicit FP32 selection. Independent complete and selected dense references and real-score triangle bound; no native BF16 parity, tight bound or quality claim.'}
