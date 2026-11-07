"""One source-only current-input subspace; complete G/Up, then native local consumer."""
import numpy as np

SOURCE_NAMES=['normalized','attentionRead','gate','text','image']
COMPARISON_NAMES=['gateRaw','upRaw','activation','product','preResidualUpdate','exit']
PARAMETERS=['single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight','single_transformer_blocks.0.attn.to_out.weight']


def signs(rows,rank,stream):
    generator=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261107,82,stream])))
    return ((generator.bit_generator.random_raw(rows*rank)&np.uint64(1)).astype(np.int8)*2-1).reshape(rows,rank)


def work(dim=3072,mlp=9216,text=512,image=1024,rank=128):
    n=text+image
    candidate={'sketch':n*dim*rank,'coordinates':n*dim*rank,'compileGU':2*dim*rank*2*mlp,
               'reconstructGU':n*rank*2*mlp,'outFull':n*(dim+mlp)*dim}
    diagnostic={'inputReconstruction':n*rank*dim,'basisGram':2*dim*rank*rank}
    # geqrf + orgqr leading operation counts, both d-by-r streams; FLOPs, not MACs.
    qr=2*(4*dim*rank*rank-4*rank**3/3)
    return {'candidate':candidate,'candidateMACs':sum(candidate.values()),'diagnostic':diagnostic,
            'diagnosticMACs':sum(diagnostic.values()),'assayMACs':sum(candidate.values())+sum(diagnostic.values()),
            'matchedDenseGUAndOutMACs':n*dim*2*mlp+n*(dim+mlp)*dim,'QRApproxFLOPs':qr}


def abis(dim=3072,mlp=9216,text=512,image=1024,rank=128):
    n=text+image
    result={'sketchSigns':{'shape':[1,n,rank],'dtype':'torch.int8'}}
    for name,shape in [('sketch',[2,dim,rank]),('basis',[2,dim,rank]),('qrR',[2,rank,rank]),
                       ('columnSigns',[2,rank]),('coordinates',[1,n,rank]),('projectedGU',[2,rank,2*mlp]),
                       ('gateFP64',[1,n,mlp]),('upFP64',[1,n,mlp])]:
        result[name]={'shape':shape,'dtype':'torch.float64'}
    for name,width in [('gateRaw',mlp),('upRaw',mlp),('activation',mlp),('product',mlp),
                       ('preResidualUpdate',dim),('gatedWrite',dim),('exit',dim)]:
        result[name]={'shape':[1,n,width],'dtype':'torch.bfloat16'}
    result.update(textOut={'shape':[1,text,dim],'dtype':'torch.bfloat16'},imageOut={'shape':[1,image,dim],'dtype':'torch.bfloat16'})
    return result


def geometry(value):
    return {'shape':list(value.shape),'stride':list(value.stride()),'storageOffset':value.storage_offset(),
            'dtype':str(value.dtype),'device':value.device.type,'logicalBytes':value.numel()*value.element_size()}


def compute(source,fused,out_weight,native_activation,matmul,qr,text=512,rank=128):
    import torch
    z=source['normalized'];n,dim=z.shape[1:];mlp=(fused.shape[0]-3*dim)//2
    if z.shape[0]!=1 or fused.shape!=(3*dim+2*mlp,dim) or out_weight.shape!=(dim,dim+mlp) or not 0<text<n or not 0<rank<=min(dim,text,n-text):
        raise ValueError('Current-state factorization geometry differs')
    if any(v.dtype!=torch.bfloat16 for v in [*source.values(),fused,out_weight]):raise ValueError('Expected actual native BF16 operands')
    original_gu=fused[3*dim:];w=original_gu.double().T
    kept={name:[] for name in ['sketchSigns','sketch','basis','qrR','columnSigns','coordinates','projectedGU']};operands=[]
    for stream,(lo,hi) in enumerate([(0,text),(text,n)]):
        x=z[0,lo:hi].double();omega=torch.from_numpy(signs(hi-lo,rank,stream)).to(z.device)
        y=matmul('sketch'+str(stream),x.T,omega.double());q,r=qr('qr'+str(stream),y)
        column_sign=torch.where(torch.diagonal(r)<0,-torch.ones(rank,dtype=q.dtype,device=q.device),torch.ones(rank,dtype=q.dtype,device=q.device))
        q=q*column_sign;r=column_sign[:,None]*r
        c=matmul('coordinates'+str(stream),x,q);b=matmul('compileGU'+str(stream),q.T,w)
        gu=matmul('reconstructGU'+str(stream),c,b)
        for name,value in zip(kept,[omega,y,q,r,column_sign,c,b]):kept[name].append(value)
        operands.append(gu)
    complete=torch.cat(operands,0)[None];del w,operands
    # QKV padding is initialized and charged, but never evaluated from learned QKV weights.
    backing=torch.zeros((1,n,3*dim+2*mlp),dtype=torch.bfloat16,device=z.device)
    gu=backing[...,3*dim:];gu.copy_(complete.float().bfloat16())
    activation=[];original_silu=torch.nn.functional.silu
    def capture_silu(*args,**kwargs):
        value=original_silu(*args,**kwargs);activation.append(value);return value
    try:
        torch.nn.functional.silu=capture_silu
        product=native_activation(gu)
    finally:torch.nn.functional.silu=original_silu
    if len(activation)!=1:raise ValueError('Native SiLU observation did not run exactly once')
    immediate=geometry(product);joined=torch.cat([source['attentionRead'],product],-1);product=joined[...,dim:]
    update=matmul('outFull',joined,out_weight.T)
    gated=source['gate']*update;incoming=torch.cat([source['text'],source['image']],1);exited=incoming+gated
    result={name:(torch.cat(values,0)[None] if name in ['sketchSigns','coordinates'] else torch.stack(values)) for name,values in kept.items()}
    result.update(gateFP64=complete[...,:mlp],upFP64=complete[...,mlp:],gateRaw=gu[...,:mlp],upRaw=gu[...,mlp:],activation=activation[0],
                  product=product,preResidualUpdate=update,gatedWrite=gated,exit=exited,textOut=exited[:,:text],imageOut=exited[:,text:])
    trace={'originalGUView':geometry(original_gu),'nativeGUView':geometry(gu),'nativeGateView':geometry(gu[...,:mlp]),'nativeUpView':geometry(gu[...,mlp:]),
           'activation':geometry(activation[0]),'immediateProduct':immediate,'recordedProduct':geometry(product),'outputAffineInput':geometry(joined),
           'offsetAuthority':'Source-derived pinned split/cat source; old capture retains strides, not offsets.',
           'dataWork':{'GUWeightBF16ToFP64Elements':original_gu.numel(),'normalizedBF16ToFP64Elements':z.numel(),
                       'signInt8ToFP64Elements':n*rank,'reconstructedFP64ToFP32Elements':complete.numel(),'reconstructedFP32ToBF16Elements':complete.numel(),
                       'initializedBackingBF16Elements':backing.numel(),'admittedBF16CopyElements':gu.numel(),'outputConcatBF16Elements':joined.numel(),
                       'incomingConcatBF16Elements':incoming.numel(),'SiLUElements':activation[0].numel(),'SwiGLUMultiplyElements':product.numel(),
                       'gateMultiplyElements':gated.numel(),'residualAddElements':exited.numel()},
           'scope':'Two current-input bases, original full GU maps and complete output affine paid. No full original GU/QKV projection, source refresh, continuation or image.'}
    return result,trace
