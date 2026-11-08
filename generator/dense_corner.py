"""Observe the immutable70 single-block consumer at full original geometry."""
from own_successor import SOURCE_NAMES,PARAMETERS,PREFIX


def work(dim=3072,mlp=9216,text=512,image=1024,heads=24):
    n=text+image;dh=dim//heads
    c={'fullFused':n*dim*(3*dim+2*mlp),'fullOut':n*(dim+mlp)*dim,
       'imageQKAndAV':2*image*n*dim,'imageCovariance':heads*dh*image*dh,
       'queryCovariance':heads*text*dh*dh,'queryMean':heads*text*dh,
       'textScores':heads*text*dh*text,'textAV':heads*text*text*dh}
    return {'candidate':c,'candidateMACs':sum(c.values()),'diagnostic':{},'diagnosticMACs':0,
            'assayMACs':sum(c.values()),'QRCalls':0,
            'scope':'Original full fused/output BF16 maps and frozen70moment attention paid. No QR or FP64 candidate product. CPU scalar/data accounting and normalization/nonlinearities/copies/hashes additional.'}


def abis(dim=3072,mlp=9216,text=512,image=1024,heads=24):
    n=text+image;dh=dim//heads
    shapes={'normalized':[1,n,dim],'fused':[1,n,3*dim+2*mlp],
            **{k:[1,n,dim] for k in ['queryRaw','keyRaw','valueRaw','attentionRead','preResidualUpdate','exit']},
            **{k:[1,n,mlp] for k in ['gateRaw','upRaw','activation','product']},
            **{k:[1,n,heads,dh] for k in ['attentionQuery','attentionKey','attentionValue']},
            'imageAttention':[1,image,heads,dh],'textRead':[1,text,dim],'alpha':[1,heads,text],
            'outputJoin':[1,n,dim+mlp],'textOut':[1,text,dim],'imageOut':[1,image,dim],
            'gate':[1,1,dim],'moment_mean':[1,heads,dh],'moment_covariance':[1,heads,dh,dh],
            **{'moment_'+k:[1,heads,text] for k in ['logZI','logZT','quadratic']}}
    return {k:{'shape':v,'dtype':'torch.float32' if k=='alpha' or k.startswith('moment_') else 'torch.bfloat16'} for k,v in shapes.items()}


def compute(block,source,native,attention,linear,product,observer=lambda name,value:None):
    import torch
    from image_partition_moments import single
    from gate_up_basis import geometry
    a=block.attn;dim=a.inner_dim;mlp=a.mlp_hidden_dim
    got={};handles=[];original_linear=torch.nn.functional.linear
    original_silu=torch.nn.functional.silu;original_rope=native.apply_rotary_emb
    split_descriptor=native.Flux2Modulation.__dict__['split'];original_split=native.Flux2Modulation.split
    rotary=[];linears=[]
    def keep(name,value):
        if name in got:raise ValueError('Duplicate native field:'+name)
        got[name]=value;observer(name,value)
    def fused_pre(module,args):keep('normalized',args[0])
    def fused_post(module,args,value):
        keep('fused',value)
        q,k,v,gu=value.split([dim,dim,dim,2*mlp],-1)
        for name,item in [('queryRaw',q),('keyRaw',k),('valueRaw',v),('gateRaw',gu[...,:mlp]),('upRaw',gu[...,mlp:])]:keep(name,item)
    def out_pre(module,args):
        keep('outputJoin',args[0]);keep('attentionRead',args[0][...,:dim])
    def out_post(module,args,value):keep('preResidualUpdate',value)
    def activation_post(module,args,value):keep('product',value)
    def lin(value,weight,bias=None):
        if weight is a.to_qkv_mlp_proj.weight:name='fullFused'
        elif weight is a.to_out.weight:name='fullOut'
        else:raise ValueError('Undeclared native affine')
        linears.append(name);return linear(name,value,weight,bias,original_linear)
    def silu(*args,**kwargs):
        result=original_silu(*args,**kwargs);keep('activation',result);return result
    def rope(*args,**kwargs):
        value=original_rope(*args,**kwargs);rotary.append(value)
        if len(rotary)>2:raise ValueError('Extra native RoPE call')
        keep('attentionQuery' if len(rotary)==1 else 'attentionKey',value);return value
    def split(*args,**kwargs):
        value=original_split(*args,**kwargs);keep('gate',value[0][2]);return value
    def attend(q,k,v,**kwargs):
        keep('attentionValue',v);value=attention(q,k,v,**kwargs);keep('imageAttention',value);return value
    def moments(values):
        for name,value in values.items():keep('moment_'+name,value)
    try:
        handles=[a.to_qkv_mlp_proj.register_forward_pre_hook(fused_pre),
                 a.to_qkv_mlp_proj.register_forward_hook(fused_post),
                 a.to_out.register_forward_pre_hook(out_pre),a.to_out.register_forward_hook(out_post),
                 a.mlp_act_fn.register_forward_hook(activation_post)]
        torch.nn.functional.linear=lin;torch.nn.functional.silu=silu
        native.apply_rotary_emb=rope;native.Flux2Modulation.split=staticmethod(split)
        text,image,alpha,read=single(block,source['text'],source['image'],source['mod'],
            (source['cosine'],source['sine']),native,attend,product,moments)
        for name,value in [('textOut',text),('imageOut',image),('exit',torch.cat([text,image],1)),('alpha',alpha),('textRead',read)]:keep(name,value)
    finally:
        torch.nn.functional.linear=original_linear;torch.nn.functional.silu=original_silu
        native.apply_rotary_emb=original_rope;native.Flux2Modulation.split=split_descriptor
        for handle in handles:handle.remove()
    if linears!=['fullFused','fullOut'] or len(rotary)!=2:raise ValueError('Original native call history differs')
    return got,{'frozenConsumer':'image_partition_moments.single','linears':linears,
                'actualNativeViews':{k:geometry(got[k]) for k in ['fused','queryRaw','keyRaw','valueRaw','gateRaw','upRaw','outputJoin']},
                'scope':'Original full-geometry70function invoked directly. Hooks return None;wrappers invoke original functions and return original result. No changed views/rounding or source-defined replacement of the numerical consumer.'}
