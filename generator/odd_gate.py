"""Fixed quadratic SwiGLU component on paid own-state operands."""
SOURCE_NAMES=['normalized','upReference','upMean','selection_support','weightD','gate','linearOutput']
COMPARISON_NAMES=['activation','activationMean','changingProduct','centeredOutput','policyOutput','totalOutput']
PARAMETER='single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight'


def abis(dim=3072,mlp=9216,text=512,image=1024):
    shapes={'normalizedFP64':[1,text+image,dim],'weightG':[mlp,dim],'gateReference':[1,text+image,mlp],'gateMean':[1,2,mlp],'quadraticProduct':[1,text+image,mlp]}
    shapes.update({n:[1,text+image,dim] for n in ['quadraticOutput','gatedQuadratic','forecast','forecastCentered']})
    shapes.update({n:[1,2,dim] for n in ['gatedQuadraticMean','forecastMean']})
    return {n:{'shape':v,'dtype':'torch.float64'} for n,v in shapes.items()}


def compute(source,gate_weight,linear,text=512):
    import torch
    n=source['normalized'].shape[1]
    def mean(x):return torch.cat([x[:,:text].mean(1,keepdim=True),x[:,text:].mean(1,keepdim=True)],1)
    def expand(x):return torch.cat([x[:,:1].expand(-1,text,-1),x[:,1:].expand(-1,n-text,-1)],1)
    z=source['normalized'].double();wg=gate_weight.double();gr=linear('fullGate',z,wg);mg=mean(gr)
    product=.5*(gr-expand(mg))*(source['upReference']-expand(source['upMean']))
    product.index_fill_(-1,source['selection_support'],0)
    y=linear('quadraticDown',product,source['weightD']);gate=source['gate'].double();gt=gate*y
    forecast=gate*(source['linearOutput']+y);mf=mean(forecast)
    return {'normalizedFP64':z,'weightG':wg,'gateReference':gr,'gateMean':mg,'quadraticProduct':product,'quadraticOutput':y,
        'gatedQuadratic':gt,'gatedQuadraticMean':mean(gt),'forecast':forecast,'forecastMean':mf,'forecastCentered':forecast-expand(mf)}
