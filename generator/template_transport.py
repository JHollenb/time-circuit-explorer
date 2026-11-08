"""Same retained template on both inputs; fresh operands and explicit reload geometry."""
from own_successor import SOURCE_NAMES,PARAMETERS,PREFIX,geometry,abis as previous_abis

TEMPLATE_NAMES=['gu_basis','gu_projectedGU','rowBasis']
GU_NAMES=['coordinates','gateFP64','upFP64','gateRaw','upRaw','activation','product']
OUT_NAMES=['projectedInputs','compressedWrites','updateFP64','preResidualUpdate','gatedWrite','exit','textOut','imageOut']
NATIVE_NAMES=['normalized','queryRaw','keyRaw','valueRaw','attentionQuery','attentionKey','attentionValue',
              'imageAttention','attentionRead','textRead','alpha','gate',
              'moment_mean','moment_covariance','moment_logZI','moment_logZT','moment_quadratic']


def work(dim=3072,mlp=9216,text=512,image=1024,rank=128,heads=24):
    n=text+image;dh=dim//heads
    per={'QKV':n*dim*3*dim,'GUCoordinates':n*dim*rank,'GUReconstruct':n*rank*2*mlp,
         'outputProjectInputs':n*rank*(dim+mlp),'outputOriginal':2*rank*(dim+mlp)*dim,
         'outputReconstruct':n*rank*dim,'imageQKAndAV':2*image*n*dim,
         'imageCovariance':heads*dh*image*dh,'queryCovariance':heads*text*dh*dh,
         'queryMean':heads*text*dh,'textScores':heads*text*dh*text,'textAV':heads*text*text*dh}
    return {'perInput':per,'perInputCandidateMACs':sum(per.values()),'candidate':{n:2*v for n,v in per.items()},
            'candidateMACs':2*sum(per.values()),'diagnostic':{},'diagnosticMACs':0,
            'assayMACs':2*sum(per.values()),'QRCalls':0,
            'scope':'Two fixed-template evaluations;20FP64/2BF16QKV/10FP32moment/2imageSDPA. '
                    'Original maps/template formation/source/clocks/history and scalar/data work paid.'}


def abis(dim=3072,mlp=9216,text=512,image=1024,rank=128,heads=24):
    old=previous_abis(dim,mlp,text,image,rank,heads)
    keys=OUT_NAMES+['gu_'+n for n in GU_NAMES]+NATIVE_NAMES
    return {n:old[n] for n in keys}|{'outputJoin':{'shape':[1,text+image,dim+mlp],'dtype':'torch.bfloat16'}}


def operands(z,templates,activation,matmul,mlp,text):
    import torch
    n,dim=z.shape[1:];coordinates=[];reconstructed=[];views=[]
    for stream,(lo,hi) in enumerate([(0,text),(text,n)]):
        basis=templates['gu_basis'][stream].contiguous()
        compiled=templates['gu_projectedGU'][stream].contiguous()
        c=matmul('guCoordinates'+str(stream),z[0,lo:hi].double(),basis)
        reconstructed.append(matmul('guReconstruct'+str(stream),c,compiled));coordinates.append(c)
        views.append({'basis':geometry(basis),'compiledGU':geometry(compiled)})
    complete=torch.cat(reconstructed,0)[None]
    backing=torch.zeros((1,n,3*dim+2*mlp),dtype=torch.bfloat16,device=z.device)
    gu=backing[...,3*dim:];gu.copy_(complete.float().bfloat16())
    captured=[];original=torch.nn.functional.silu
    def silu(*args,**kwargs):
        value=original(*args,**kwargs);captured.append(value);return value
    try:
        torch.nn.functional.silu=silu;product=activation(gu)
    finally:torch.nn.functional.silu=original
    if len(captured)!=1:raise ValueError('Native SwiGLU observation differs')
    return {'coordinates':torch.cat(coordinates,0)[None],'gateFP64':complete[...,:mlp],
            'upFP64':complete[...,mlp:],'gateRaw':gu[...,:mlp],'upRaw':gu[...,mlp:],
            'activation':captured[0],'product':product}, {'templateViews':views,'nativeGUView':geometry(gu),
                'scope':'Explicit row-contiguous reloaded views; no historical CUDA stride reconstruction or new template compilation.'}


def output(source,templates,weight,matmul,text):
    import torch
    joined=torch.cat([source['attentionRead'],source['product']],-1);j=joined.double();w=weight.double().T
    kept={n:[] for n in ['projectedInputs','compressedWrites']};updates=[];views=[]
    for stream,(lo,hi) in enumerate([(0,text),(text,joined.shape[1])]):
        basis=templates['rowBasis'][0,lo:hi].contiguous();views.append(geometry(basis))
        k=matmul('projectInputs'+str(stream),basis.T,j[0,lo:hi])
        b=matmul('compressedOut'+str(stream),k,w)
        updates.append(matmul('reconstructUpdate'+str(stream),basis,b))
        kept['projectedInputs'].append(k);kept['compressedWrites'].append(b)
    complete=torch.cat(updates,0)[None];update=complete.float().bfloat16()
    gated=source['gate']*update;exited=torch.cat([source['text'],source['image']],1)+gated
    return {n:torch.stack(v) for n,v in kept.items()}|{'updateFP64':complete,'preResidualUpdate':update,
        'gatedWrite':gated,'exit':exited,'textOut':exited[:,:text],'imageOut':exited[:,text:],
        'outputJoin':joined}, {'rowBasisViews':views,'join':geometry(joined),'admittedUpdate':geometry(update),
            'scope':'Fresh J and receiving coefficients; original FP64 output map paid. No old output values/sketch/QR.'}


def compute(block,source,templates,native,attention,linear,product,matmul,physical_observer,text=512):
    import torch
    from image_partition_moments import text_read
    a=block.attn;dim=a.inner_dim;mlp=a.mlp_hidden_dim;rows=torch.cat([source['text'],source['image']],1)
    shift,scale,gate=native.Flux2Modulation.split(source['mod'],1)[0]
    z=(1+scale)*block.norm(rows)+shift
    weight=a.to_qkv_mlp_proj.weight[:3*dim].clone(memory_format=torch.contiguous_format);physical_observer(weight)
    projected=linear('ownQKV',z,weight)
    backing=torch.zeros((1,rows.shape[1],3*dim+2*mlp),dtype=projected.dtype,device=z.device)
    backing[...,:3*dim].copy_(projected);qraw,kraw,vraw=backing[...,:3*dim].split(dim,-1)
    q=a.norm_q(qraw.unflatten(-1,(a.heads,-1)));k=a.norm_k(kraw.unflatten(-1,(a.heads,-1)))
    v=vraw.unflatten(-1,(a.heads,-1));rope=(source['cosine'],source['sine'])
    q=native.apply_rotary_emb(q,rope,sequence_dim=1);k=native.apply_rotary_emb(k,rope,sequence_dim=1)
    ih=attention(q[:,text:],k,v,attn_mask=None,backend=None,parallel_config=None);moments={}
    th,alpha=text_read(q[:,:text],k,v[:,:text],product,lambda values:moments.update({'moment_'+n:v for n,v in values.items()}))
    read=torch.cat([th,ih.flatten(2,3).to(q.dtype)],1)
    gu,gt=operands(z,templates,a.mlp_act_fn,matmul,mlp,text)
    out,ot=output({'product':gu['product'],'attentionRead':read,'gate':gate,'text':source['text'],'image':source['image']},
                  templates,a.to_out.weight,matmul,text)
    out.update({'gu_'+n:v for n,v in gu.items()})
    out.update(normalized=z,queryRaw=qraw,keyRaw=kraw,valueRaw=vraw,attentionQuery=q,attentionKey=k,
               attentionValue=v,imageAttention=ih,attentionRead=read,textRead=th,alpha=alpha,gate=gate,**moments)
    return out,{'GU':gt,'output':ot,'physicalQKV':geometry(weight),'slicedQKVProjection':geometry(projected),
                'restoredQKVViews':{n:geometry(v) for n,v in [('query',qraw),('key',kraw),('value',vraw)]},
                'scope':'Same explicit fixed template policy for parent/own. All current operands reformed; '
                        'physical QKV slicing/finite policy retained and paid, no original full-attention parity claim.'}
