"""One own-state single block; every normalized/read/product operand is re-formed."""
from gate_up_basis import signs as input_signs, abis as gu_abis, work as gu_work, geometry
from receiver_row_output import compute as output_compute, abis as output_abis, work as output_work

SOURCE_NAMES = ['text', 'image', 'mod', 'cosine', 'sine']
PREFIX = 'single_transformer_blocks.1.'
PARAMETERS = [PREFIX+'attn.'+n for n in ['to_qkv_mlp_proj.weight','to_out.weight','norm_q.weight','norm_k.weight']]
GU_NAMES = ['sketchSigns','sketch','basis','qrR','columnSigns','coordinates','projectedGU',
            'gateFP64','upFP64','gateRaw','upRaw','activation','product']


def work(dim=3072, mlp=9216, text=512, image=1024, rank=128, heads=24):
    n, dh = text+image, dim//heads
    gw, ow = gu_work(dim,mlp,text,image,rank), output_work(dim,mlp,text,image,rank)
    candidate = {'QKV':n*dim*3*dim, 'GU':gw['candidateMACs']-gw['candidate']['outFull'],
                 'output':ow['candidateMACs'], 'imageQKAndAV':2*image*n*dim,
                 'imageCovariance':heads*dh*image*dh, 'queryCovariance':heads*text*dh*dh,
                 'queryMean':heads*text*dh, 'textScores':heads*text*dh*text,
                 'textAV':heads*text*text*dh}
    diagnostic = {'inputReconstructionAndGram':gw['diagnosticMACs'],
                  'outputSketchReconstructionAndGram':ow['diagnosticMACs']}
    return {'candidate':candidate, 'candidateMACs':sum(candidate.values()),
            'diagnostic':diagnostic, 'diagnosticMACs':sum(diagnostic.values()),
            'assayMACs':sum(candidate.values())+sum(diagnostic.values()),
            'additionalQRApproxFLOPs':gw['QRApproxFLOPs']+ow['additionalQRApproxFLOPs'],
            'scope':'Physical BF16 QKV plus own FP64 GU/output and live attention/moments. '
                    'Both post-exit diagnostics retained as observations. Original maps, clocks, '
                    'normalization/RMS/RoPE/SiLU/softmax/scalar/data work remain paid/additional.'}


def abis(dim=3072, mlp=9216, text=512, image=1024, rank=128, heads=24):
    n, dh = text+image, dim//heads
    result = output_abis(dim,mlp,text,image,rank)
    result.update({'gu_'+k:v for k,v in gu_abis(dim,mlp,text,image,rank).items() if k in GU_NAMES})
    shapes = {'normalized':[1,n,dim], 'queryRaw':[1,n,dim], 'keyRaw':[1,n,dim],
              'valueRaw':[1,n,dim], 'attentionQuery':[1,n,heads,dh],
              'attentionKey':[1,n,heads,dh], 'attentionValue':[1,n,heads,dh],
              'imageAttention':[1,image,heads,dh], 'attentionRead':[1,n,dim],
              'textRead':[1,text,dim], 'alpha':[1,heads,text], 'gate':[1,1,dim],
              'moment_mean':[1,heads,dh], 'moment_covariance':[1,heads,dh,dh],
              **{'moment_'+k:[1,heads,text] for k in ['logZI','logZT','quadratic']}}
    result.update({k:{'shape':s,'dtype':'torch.float32' if k=='alpha' or k.startswith('moment_') else 'torch.bfloat16'} for k,s in shapes.items()})
    return result


def operands(z, fused, activation, matmul, qr, text=512, rank=128):
    """Frozen82 GU law only: never evaluate or discard its complete output affine."""
    import torch
    n, dim = z.shape[1:]; mlp = (fused.shape[0]-3*dim)//2
    w = fused[3*dim:].double().T
    kept = {k:[] for k in ['sketchSigns','sketch','basis','qrR','columnSigns','coordinates','projectedGU']}
    reconstructed = []
    for stream,(lo,hi) in enumerate([(0,text),(text,n)]):
        x = z[0,lo:hi].double()
        omega = torch.from_numpy(input_signs(hi-lo,rank,stream)).to(z.device)
        f = matmul('guSketch'+str(stream),x.T,omega.double())
        q,r = qr('guQR'+str(stream),f)
        sign = torch.where(torch.diagonal(r)<0,-torch.ones(rank,dtype=q.dtype,device=q.device),torch.ones(rank,dtype=q.dtype,device=q.device))
        q,r = q*sign,sign[:,None]*r
        c = matmul('guCoordinates'+str(stream),x,q)
        b = matmul('guCompile'+str(stream),q.T,w)
        gu = matmul('guReconstruct'+str(stream),c,b)
        for name,value in zip(kept,[omega,f,q,r,sign,c,b]):kept[name].append(value)
        reconstructed.append(gu)
    complete = torch.cat(reconstructed,0)[None]
    del w,reconstructed
    backing = torch.zeros((1,n,3*dim+2*mlp),dtype=torch.bfloat16,device=z.device)
    gu = backing[...,3*dim:]
    gu.copy_(complete.float().bfloat16())
    captures = []; original_silu = torch.nn.functional.silu
    def silu(*args,**kwargs):
        value = original_silu(*args,**kwargs);captures.append(value);return value
    try:
        torch.nn.functional.silu=silu
        product=activation(gu)
    finally:torch.nn.functional.silu=original_silu
    if len(captures)!=1:raise ValueError('Native SiLU did not execute exactly once')
    got={k:(torch.cat(v,0)[None] if k in ['sketchSigns','coordinates'] else torch.stack(v)) for k,v in kept.items()}
    got.update(gateFP64=complete[...,:mlp],upFP64=complete[...,mlp:],gateRaw=gu[...,:mlp],
               upRaw=gu[...,mlp:],activation=captures[0],product=product)
    trace={'nativeGUView':geometry(gu),'nativeGateView':geometry(gu[...,:mlp]),
           'nativeUpView':geometry(gu[...,mlp:]),'immediateProduct':geometry(product),
           'dataWork':{'originalGUToFP64Elements':fused[3*dim:].numel(),
                       'normalizedToFP64Elements':z.numel(),'signInt8ToFP64Elements':n*rank,
                       'FP64ToFP32ToBF16Elements':complete.numel(),
                       'initializedBackingBF16Elements':backing.numel(),'admittedCopyElements':gu.numel()},
           'scope':'Operand-only frozen82 law with native fused-view admission. No full original GU or output affine.'}
    return got,trace


def compute(block, source, native, attention, linear, product, matmul, qr,
            join_observer, sketch_observer, physical_observer, text=512, rank=128):
    import torch
    from image_partition_moments import text_read
    a=block.attn;dim=a.inner_dim;mlp=a.mlp_hidden_dim
    rows=torch.cat([source['text'],source['image']],1);n=rows.shape[1]
    shift,scale,gate=native.Flux2Modulation.split(source['mod'],1)[0]
    z=(1+scale)*block.norm(rows)+shift
    qkv_weight=a.to_qkv_mlp_proj.weight[:3*dim].clone(memory_format=torch.contiguous_format)
    physical_observer(qkv_weight)
    projected=linear('ownQKV',z,qkv_weight)
    # Restore native fused views; this does not undo the sliced GEMM's finite policy.
    backing=torch.zeros((1,n,3*dim+2*mlp),dtype=projected.dtype,device=projected.device)
    backing[...,:3*dim].copy_(projected)
    q_raw,k_raw,v_raw=backing[...,:3*dim].split(dim,-1)
    q=a.norm_q(q_raw.unflatten(-1,(a.heads,-1)))
    k=a.norm_k(k_raw.unflatten(-1,(a.heads,-1)))
    v=v_raw.unflatten(-1,(a.heads,-1))
    rope=(source['cosine'],source['sine'])
    q=native.apply_rotary_emb(q,rope,sequence_dim=1)
    k=native.apply_rotary_emb(k,rope,sequence_dim=1)
    ih=attention(q[:,text:],k,v,attn_mask=None,backend=None,parallel_config=None)
    moments={}
    def capture(values):moments.update({'moment_'+name:value for name,value in values.items()})
    th,alpha=text_read(q[:,:text],k,v[:,:text],product,capture)
    read=torch.cat([th,ih.flatten(2,3).to(q.dtype)],1)
    gu,gu_trace=operands(z,a.to_qkv_mlp_proj.weight,a.mlp_act_fn,matmul,qr,text,rank)
    output,out_trace=output_compute({'product':gu['product'],'attentionRead':read,'gate':gate,
                                      'text':source['text'],'image':source['image']},
                                     a.to_out.weight,matmul,qr,join_observer,sketch_observer,text,rank)
    output.update({'gu_'+name:value for name,value in gu.items()})
    output.update(normalized=z,queryRaw=q_raw,keyRaw=k_raw,valueRaw=v_raw,
                  attentionQuery=q,attentionKey=k,attentionValue=v,imageAttention=ih,
                  attentionRead=read,textRead=th,alpha=alpha,gate=gate,**moments)
    trace={'GU':gu_trace,'output':out_trace,'physicalQKV':geometry(qkv_weight),
           'slicedQKVProjection':geometry(projected),
           'restoredQKVViews':{name:geometry(value) for name,value in [('query',q_raw),('key',k_raw),('value',v_raw)]},
           'initializedQKVBackingElements':backing.numel(),'copiedQKVElements':projected.numel(),
           'scope':'Own incoming state forms every operand. Physical QKV GEMM changes width/kernel; '
                   'restored fused strides do not establish native projection parity. '
                   'No archived normalized/read/product/bases or full GU/output contraction.'}
    return output,trace
