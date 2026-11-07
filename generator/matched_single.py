"""Passive capture of frozen sparse/dense single operators and signed-write anatomy."""


def abis(mode,dim=3072,mlp=9216,text=512,image=1024,heads=24):
    n=text+image;k=mlp//4 if mode=='sparse' else mlp;dh=dim//heads
    shapes={'normalized':[1,n,dim],'queryRaw':[1,n,dim],'keyRaw':[1,n,dim],'valueRaw':[1,n,dim],
        'gateRaw':[1,n,mlp],'upRaw':[1,n,k],'activation':[1,n,mlp],'attentionQuery':[1,image,heads,dh],
        'attentionKey':[1,n,heads,dh],'attentionValue':[1,n,heads,dh],'imageAttention':[1,image,heads,dh],
        'attentionRead':[1,n,dim],'product':[1,n,k],'preResidualUpdate':[1,n,dim],'textOut':[1,text,dim],
        'imageOut':[1,image,dim],'exit':[1,n,dim],'alpha':[1,heads,text],'textRead':[1,text,dim],'gate':[1,1,dim],
        'moment_mean':[1,heads,dh],'moment_covariance':[1,heads,dh,dh],
        **{'moment_'+name:[1,heads,text] for name in ['logZI','logZT','quadratic']}}
    result={name:{'shape':shape,'dtype':'torch.float32' if name=='alpha' or name.startswith('moment_') else 'torch.bfloat16'} for name,shape in shapes.items()}
    if mode=='sparse':
        for name,shape,dtype in [('rowNormSquared',[n],'torch.float32'),('weightedGateEnergy',[mlp],'torch.float32'),('score',[mlp],'torch.float32'),
            ('rankedIndices',[mlp],'torch.int64'),('support',[k],'torch.int64'),('omittedScoreSum',[],'torch.float32'),('selectedScoreSum',[],'torch.float32')]:
            result['selection_'+name]={'shape':shape,'dtype':dtype}
    return result


def replay(mode,block,source,native,norms=None,physical_observer=None,affine_observer=None,product_observer=None):
    """Capture original tensor views; never replace execution with capture copies."""
    import torch
    import torch.nn.functional as F
    from gate_first_trajectory import single as sparse
    from image_partition_moments import single as dense
    from paired_fields import payload
    a=block.attn;d=a.inner_dim;m=a.mlp_hidden_dim;captured={};operations=[];products=[];maps=[];attention=[]
    old_linear,old_silu,old_dispatch=F.linear,F.silu,native.dispatch_attention_fn
    active={a.to_qkv_mlp_proj.weight.data_ptr():'QKVGU',a.to_out.weight.data_ptr():'outFull'};map_scope={'activeSets':0,'maxActiveSets':0,'maxLogicalBytes':0}
    def put(name,value):
        if name in captured:raise ValueError('Duplicate actual operand capture:'+name)
        captured[name]=value
    def linear(x,w,bias=None):
        name=active.get(w.data_ptr())
        if name is None or bias is not None:raise ValueError('Undeclared local affine or bias')
        value=old_linear(x,w,bias)
        row={'name':name,'inputShape':list(x.shape),'inputStride':list(x.stride()),'weightShape':list(w.shape),'weightStride':list(w.stride()),
            'outputShape':list(value.shape),'outputStride':list(value.stride()),'dtype':str(x.dtype),'device':x.device.type,'nominalMACs':x.numel()//x.shape[-1]*w.numel()}
        operations.append(row)
        if affine_observer is not None:affine_observer(row)
        if name in ['QKVG','QKVGU']:
            put('normalized',x)
            for label,v in zip(['queryRaw','keyRaw','valueRaw','gateRaw']+(['upRaw'] if name=='QKVGU' else []),value.split([d,d,d,m]+([m] if name=='QKVGU' else []),-1)):put(label,v)
        elif name=='USelected':put('upRaw',value)
        elif name in ['outFull','outSelected']:
            put('attentionRead',x[...,:d]);put('product',x[...,d:]);put('preResidualUpdate',value)
        else:raise ValueError('Unexpected affine source')
        return value
    def silu(x,inplace=False):
        value=old_silu(x,inplace=inplace);put('activation',value);return value
    def dispatch(query,key,value,*args,**kwargs):
        for name,v in [('attentionQuery',query),('attentionKey',key),('attentionValue',value)]:put(name,v)
        result=old_dispatch(query,key,value,*args,**kwargs);put('imageAttention',result)
        attention.append({'QShape':list(query.shape),'KShape':list(key.shape),'VShape':list(value.shape),'nominalMACs':2*query.numel()*key.shape[1]})
        return result
    def product(kind,left,right):
        value=torch.matmul(left,right);row={'operator':kind,'leftShape':list(left.shape),'rightShape':list(right.shape),'outputShape':list(value.shape),
            'dtype':str(value.dtype),'device':value.device.type,'nominalMACs':value.numel()*left.shape[-1]};products.append(row)
        if product_observer is not None:product_observer(row)
        return value
    def with_maps(index,values,operation):
        if index!=0 or map_scope['activeSets']:raise ValueError('One common-input block/map callback required')
        map_scope['activeSets']+=1;map_scope['maxActiveSets']=1;map_scope['maxLogicalBytes']=max(map_scope['maxLogicalBytes'],sum(v.numel()*v.element_size() for v in values.values()))
        pointers=[]
        try:
            for name,value in values.items():
                pointer=value.data_ptr()
                if pointer in active:raise ValueError('Physical map aliases original parameter')
                active[pointer]=name;pointers.append(pointer)
                row={'name':name,'shape':list(value.shape),'bytes':value.numel()*value.element_size(),'payloadSHA256':payload(value)};maps.append(row)
                if physical_observer is not None:physical_observer(row)
            return operation()
        finally:
            for pointer in pointers:del active[pointer]
            map_scope['activeSets']-=1
    def moments(values):
        for name,value in values.items():put('moment_'+name,value)
    def selection(index,values):
        if index!=0:raise ValueError('Wrong selector address')
        for name,value in values.items():put('selection_'+name,value)
    try:
        F.linear=linear;F.silu=silu
        args=(block,source['text'],source['image'],source['mod'],(source['cosine'],source['sine']),native,dispatch,product,moments)
        if mode=='sparse':result=sparse(*args,with_maps,0,norms,selection)
        elif mode=='dense':result=dense(*args)
        else:raise ValueError('Unknown local path')
    finally:F.linear=old_linear;F.silu=old_silu
    for name,value in zip(['textOut','imageOut','alpha','textRead'],result):put(name,value)
    put('exit',torch.cat(result[:2],1));put('gate',native.Flux2Modulation.split(source['mod'],1)[0][2])
    if len(operations)!=(3 if mode=='sparse' else 2) or len(attention)!=1 or len(products)!=5:raise ValueError('Local operator capture census differs')
    return captured,{'operations':operations,'attention':attention,'products':products,'physicalMaps':maps,'mapScope':map_scope}


def common_write(product,down,support,matmul):
    import torch
    if product.dtype!=torch.float64 or down.dtype!=torch.float64 or support.dtype!=torch.int64:raise ValueError('FP64 common-write operands required')
    full=matmul('fullDown',product,down.T)
    selected=matmul('selectedDown',product.index_select(-1,support),down.index_select(1,support).T)
    return {'fullWrite':full,'selectedWrite':selected,'omittedWrite':full-selected}


def energy_panel(product,down,support,full,selected,gate):
    """Diagonal and signed aggregate energies, without a channel Gram matrix."""
    import torch
    retained=torch.zeros(product.shape[-1],dtype=torch.bool,device=product.device);retained[support]=True
    down_energy=down.square().sum(0);gated_down_energy=(down*gate.reshape(-1,1)).square().sum(0)
    rows=[];channel={}
    for name,part in [('all',slice(None)),('text',slice(0,512)),('image',slice(512,None))]:
        p=product[:,part];y=full[:,part];ys=selected[:,part];omitted=y-ys;pe=p.square().sum((0,1));channel[name]=(pe*down_energy).sqrt()
        for gated in [False,True]:
            ds=gated_down_energy if gated else down_energy;scale=gate if gated else 1
            diag=pe*ds;o=omitted*scale;s=ys*scale;f=y*scale
            omitted_diagonal=diag[~retained].sum();omitted_total=o.square().sum();selected_diagonal=diag[retained].sum();selected_total=s.square().sum()
            rows.append({'stratum':name,'gated':gated,'omittedDiagonalEnergy':float(omitted_diagonal),'omittedWriteEnergy':float(omitted_total),
                'omittedCrossChannelEnergy':float(omitted_total-omitted_diagonal),'selectedDiagonalEnergy':float(selected_diagonal),'selectedWriteEnergy':float(selected_total),
                'selectedCrossChannelEnergy':float(selected_total-selected_diagonal),'selectedOmittedCrossEnergy':float(2*(s*o).sum()),'fullWriteEnergy':float(f.square().sum())})
    return rows,channel
