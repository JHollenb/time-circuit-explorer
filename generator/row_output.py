"""Complete output through the same current-input row coordinates, without output fitting."""
SOURCE_NAMES=['coordinates','product','attentionRead','gate','text','image']
PARAMETER='single_transformer_blocks.0.attn.to_out.weight'


def work(dim=3072,mlp=9216,text=512,image=1024,rank=128):
    n=text+image;width=dim+mlp
    candidate={'projectInputs':n*rank*width,'compressedOriginalOut':2*rank*width*dim,'reconstructUpdate':n*rank*dim}
    diagnostic={'coordinateReconstruction':n*rank*rank,'basisGram':n*rank*rank}
    inherited=2*n*dim*rank+4*dim*rank*mlp+2*n*rank*mlp
    return {'candidate':candidate,'candidateMACs':sum(candidate.values()),'diagnostic':diagnostic,'diagnosticMACs':sum(diagnostic.values()),
        'assayMACs':sum(candidate.values())+sum(diagnostic.values()),'inherited82GateUpFormationMACs':inherited,
        'prospectiveComposedProducerMACs':inherited+sum(candidate.values()),'matched82FullOutMACs':n*width*dim,
        'additionalQRApproxFLOPs':4*n*rank*rank-8*rank**3/3,
        'scope':'Actual new output/diagnostic products separate. Composed count retains historical82 GU formation, is prospective and excludes paid upstream source/attention and QR/scalar/data work.'}


def abis(dim=3072,mlp=9216,text=512,image=1024,rank=128):
    n=text+image
    shapes={'rowBasis':[1,n,rank],'qrR':[2,rank,rank],'columnSigns':[2,rank],'projectedInputs':[2,rank,dim+mlp],
            'compressedWrites':[2,rank,dim],'updateFP64':[1,n,dim]}
    result={name:{'shape':shape,'dtype':'torch.float64'} for name,shape in shapes.items()}
    result.update({name:{'shape':[1,n,dim],'dtype':'torch.bfloat16'} for name in ['preResidualUpdate','gatedWrite','exit']})
    result.update(textOut={'shape':[1,text,dim],'dtype':'torch.bfloat16'},imageOut={'shape':[1,image,dim],'dtype':'torch.bfloat16'})
    return result


def geometry(value):
    return {'shape':list(value.shape),'stride':list(value.stride()),'storageOffset':value.storage_offset(),'dtype':str(value.dtype),'device':value.device.type}


def compute(source,weight,matmul,qr,join_observer,text=512):
    import torch
    c=source['coordinates'];n,rank=c.shape[1:];dim=weight.shape[0];mlp=source['product'].shape[-1]
    expected={'coordinates':([1,n,rank],torch.float64),'product':([1,n,mlp],torch.bfloat16),'attentionRead':([1,n,dim],torch.bfloat16),
              'gate':([1,1,dim],torch.bfloat16),'text':([1,text,dim],torch.bfloat16),'image':([1,n-text,dim],torch.bfloat16)}
    if set(source)!=set(expected) or any(list(v.shape)!=expected[name][0] or v.dtype!=expected[name][1] for name,v in source.items()) or weight.shape!=(dim,dim+mlp) or weight.dtype!=torch.bfloat16 or not 0<rank<=min(text,n-text):raise ValueError('Row-output source ABI differs')
    joined=torch.cat([source['attentionRead'],source['product']],-1);join_observer(joined)
    joined64=joined.double();w=weight.double().T;kept={name:[] for name in ['rowBasis','qrR','columnSigns','projectedInputs','compressedWrites']};updates=[]
    for stream,(lo,hi) in enumerate([(0,text),(text,n)]):
        u,r=qr('rowQR'+str(stream),c[0,lo:hi]);sign=torch.where(torch.diagonal(r)<0,-torch.ones(rank,dtype=r.dtype,device=r.device),torch.ones(rank,dtype=r.dtype,device=r.device))
        u=u*sign;r=sign[:,None]*r
        k=matmul('projectInputs'+str(stream),u.T,joined64[0,lo:hi]);b=matmul('compressedOut'+str(stream),k,w);y=matmul('reconstructUpdate'+str(stream),u,b)
        for name,value in zip(kept,[u,r,sign,k,b]):kept[name].append(value)
        updates.append(y)
    complete=torch.cat(updates,0)[None];update=complete.float().bfloat16().contiguous()
    gated=source['gate']*update;incoming=torch.cat([source['text'],source['image']],1);exited=incoming+gated
    result={name:(torch.cat(values,0)[None] if name=='rowBasis' else torch.stack(values)) for name,values in kept.items()}
    result.update(updateFP64=complete,preResidualUpdate=update,gatedWrite=gated,exit=exited,textOut=exited[:,:text],imageOut=exited[:,text:])
    trace={'BF16JoinedInput':geometry(joined),'admittedUpdate':geometry(update),'gatedWrite':geometry(gated),'exit':geometry(exited),
        'layoutAuthority':'Update contiguous shape/stride/offset source-derived from82 F.linear output; no native numerical equivalence implied.',
        'dataWork':{'joinedBF16Elements':joined.numel(),'joinedBF16ToFP64Elements':joined.numel(),'originalOutBF16ToFP64Elements':weight.numel(),
                    'canonicalBasisMultiplyElements':n*rank,'canonicalRMultiplyElements':2*rank*rank,'completeFP64ToFP32Elements':complete.numel(),
                    'finalFP32ToBF16Elements':complete.numel(),'gateMultiplyElements':gated.numel(),'incomingConcatElements':incoming.numel(),'residualAddElements':exited.numel()},
        'scope':'Six FP64 products; one final double-conversion admission, no intermediate compressed-row cast, original full output map paid. Supplied82 C/product and all source formation paid.'}
    return result,trace
