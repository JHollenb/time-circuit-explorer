"""One saved own-state MLP write diagnostic; three paid FP64 affine products."""

SOURCE_NAMES=['normalized','activation','upRaw','completedUp','product','streamMeanFP32','streamMean','anchorUp','gate','selection_support']
ROW_COLUMNS=['meanEnergy','centeredEnergy','policyEnergy','twiceMeanCentered','twiceMeanPolicy','twiceCenteredPolicy','totalEnergy','oracleResidualEnergy','meanRResidualEnergy','outputClosureEnergy']


def abis(dim=3072,mlp=9216,text=512,image=1024,count=2304):
    n=text+image
    banks=['upReference','idealCompleted','productReference','totalProductError','centeredProduct','policyProduct','omittedAnchorAdmissionProduct','productRounding','policyExpansionResidual']
    shapes={name:[1,n,mlp] for name in banks}
    shapes.update(weightU=[mlp,dim],weightD=[dim,mlp],upMean=[1,2,mlp],selectedAdmissionProduct=[1,n,count],covarianceDirect=[1,2,mlp],centeredProductMean=[1,2,mlp],covarianceResidual=[1,2,mlp])
    shapes.update({name:[1,n,dim] for name in ['totalOutput','centeredOutput','policyOutput','gatedTotal','gatedCentered','gatedPolicy','oracleResidual','outputClosure']})
    shapes.update(gatedMean=[1,2,dim],oracleMean=[1,2,dim],stratumGram=[3,3,3],rowStatistics=[1,n,len(ROW_COLUMNS)])
    return {name:{'shape':shape,'dtype':'torch.float64'} for name,shape in shapes.items()}


def compute(source,up,down,linear,text=512):
    import torch
    z=source['normalized'];a=source['activation'].double();support=source['selection_support'];n=z.shape[1]
    def mean(x):return torch.cat([x[:,:text].mean(1,keepdim=True),x[:,text:].mean(1,keepdim=True)],1)
    def expand(x):return torch.cat([x[:,:1].expand(-1,text,-1),x[:,1:].expand(-1,n-text,-1)],1)
    wu=up.double();u=linear('fullUp',z.double(),wu);mu=mean(u)
    ideal=expand(mu);ideal.index_copy_(-1,support,u.index_select(-1,support))
    p=a*u;e=p-source['product'].double()
    r=a*(u-expand(mu));r.index_fill_(-1,support,0)
    q=e-r;admission=a*(ideal-source['completedUp'].double());selected=admission.index_select(-1,support);admission.index_fill_(-1,support,0)
    rounding=a*source['completedUp'].double()-source['product'].double()
    residual=q-admission-rounding
    residual.index_copy_(-1,support,residual.index_select(-1,support)-selected)
    direct=torch.cat([((a[:,lo:hi]-a[:,lo:hi].mean(1,keepdim=True))*(u[:,lo:hi]-mu[:,k:k+1])).mean(1,keepdim=True)
                      for k,(lo,hi) in enumerate([(0,text),(text,n)])],1)
    direct.index_fill_(-1,support,0);rmean=mean(r)
    wd=down.double();ye=linear('totalErrorDown',e,wd);yr=linear('centeredDown',r,wd);yq=ye-yr
    gate=source['gate'].double();gt=gate*ye;gr=gate*yr;gq=gate*yq;gm=mean(gr);gm_rows=expand(gm);gc=gr-gm_rows
    om=mean(gt);ores=gt-expand(om);closure=gt-(gm_rows+gc+gq);mean_residual=gt-gm_rows
    parts=[gm_rows,gc,gq]
    stats=torch.stack([(gm_rows*gm_rows).sum(-1),(gc*gc).sum(-1),(gq*gq).sum(-1),2*(gm_rows*gc).sum(-1),
        2*(gm_rows*gq).sum(-1),2*(gc*gq).sum(-1),(gt*gt).sum(-1),(ores*ores).sum(-1),(mean_residual*mean_residual).sum(-1),(closure*closure).sum(-1)],-1)
    grams=torch.stack([torch.stack([torch.stack([(x[:,lo:hi]*y[:,lo:hi]).sum() for y in parts]) for x in parts]) for lo,hi in [(0,n),(0,text),(text,n)]])
    return {'weightU':wu,'weightD':wd,'upReference':u,'upMean':mu,'idealCompleted':ideal,'productReference':p,'totalProductError':e,'centeredProduct':r,'policyProduct':q,
        'selectedAdmissionProduct':selected,'omittedAnchorAdmissionProduct':admission,'productRounding':rounding,'policyExpansionResidual':residual,
        'covarianceDirect':direct,'centeredProductMean':rmean,'covarianceResidual':rmean-direct,'totalOutput':ye,'centeredOutput':yr,'policyOutput':yq,
        'gatedTotal':gt,'gatedMean':gm,'gatedCentered':gc,'gatedPolicy':gq,'oracleMean':om,'oracleResidual':ores,'outputClosure':closure,'stratumGram':grams,'rowStatistics':stats}
