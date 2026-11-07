"""Combined positive-branch product on paid G*/Up/L and original Down."""
SOURCE_NAMES=['gateReference','upReference','upMean','selection_support','weightD','gate','linearOutput']
COMPARISON_NAMES=['activation','activationMean','changingProduct','centeredOutput','policyOutput','totalOutput','quadraticForecast']


def abis(dim=3072,mlp=9216,text=512,image=1024):
    shapes={n:[1,text+image,mlp] for n in ['rectifier','rectifierProduct']};shapes['rectifierMean']=[1,2,mlp]
    shapes.update({n:[1,text+image,dim] for n in ['rectifierOutput','gatedRectifier','forecast','forecastCentered']})
    shapes.update({n:[1,2,dim] for n in ['gatedRectifierMean','forecastMean']})
    return {n:{'shape':v,'dtype':'torch.float64'} for n,v in shapes.items()}


def compute(source,linear,text=512):
    import torch
    n=source['gateReference'].shape[1]
    def mean(x):return torch.cat([x[:,:text].mean(1,keepdim=True),x[:,text:].mean(1,keepdim=True)],1)
    def expand(x):return torch.cat([x[:,:1].expand(-1,text,-1),x[:,1:].expand(-1,n-text,-1)],1)
    rectifier=torch.clamp_min(source['gateReference'],0.);mr=mean(rectifier)
    product=(rectifier-expand(mr))*(source['upReference']-expand(source['upMean']))
    product.index_fill_(-1,source['selection_support'],0)
    y=linear('rectifierDown',product,source['weightD']);gate=source['gate'].double();gy=gate*y
    forecast=gate*(source['linearOutput']+y);mf=mean(forecast)
    return {'rectifier':rectifier,'rectifierMean':mr,'rectifierProduct':product,'rectifierOutput':y,'gatedRectifier':gy,
        'gatedRectifierMean':mean(gy),'forecast':forecast,'forecastMean':mf,'forecastCentered':forecast-expand(mf)}
