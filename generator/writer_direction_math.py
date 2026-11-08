"""Saved-field local-model accounts, never a new finite normalization or fit."""
import numpy as np

TERMS=('Hendpoint','Hresponse','Zendpoint','Zresponse')
DIRECTIONS=('proposal','writer','admitted','mapping','admission')
GRADIENTS=('saved',*TERMS)
POLICY='FP64 saved-gradient products on alpha*Delta,actual FP32 routed write change u,and saved finite H change v. Products divided by2 are anchored ideal local-model slopes,not derivatives of the discrete BF16 objective.'


def finite(value):
    a=np.asarray(value,np.float64)
    if not a.size or not np.isfinite(a).all():raise ValueError('Empty or nonfinite explanatory input')
    return a


def pair_part(value,target,edges,weights,scale):
    out=np.zeros_like(value);left,right=edges.T
    change=weights[:,None]*((value[right]-value[left])-(target[right]-target[left]))/(scale or 1.)
    np.add.at(out,left,-change);np.add.at(out,right,change)
    return out


def pullback(h,value,gamma,epsilon):
    radius=np.sqrt(np.mean(h*h,axis=1,dtype=np.float64)+epsilon)
    weighted=gamma*value
    return weighted/radius[:,None]-h*(np.sum(h*weighted,axis=1,dtype=np.float64)/(h.shape[1]*radius**3))[:,None]


def components(data,epsilon):
    h,z,target,zt=[finite(data[k]) for k in ('h','z','target','zt')]
    w=finite(data['weights']);edges=np.asarray(data['edges']);v=finite(data['pairWeights']);gamma=finite(data['gamma']);scales=finite(data['scaleVector'])
    if h.ndim!=2 or any(x.shape!=h.shape for x in (z,target,zt)) or w.shape!=(len(h),) or gamma.shape!=(h.shape[1],) or scales.shape!=(4,) or np.any(scales<0) or np.any(w<=0) or np.any(v<0) or not np.isfinite(epsilon) or epsilon<=0:raise ValueError('Gradient component ABI differs')
    if edges.dtype.kind not in 'iu' or edges.ndim!=2 or edges.shape[1]!=2 or v.shape!=(len(edges),) or edges.min()<0 or edges.max()>=len(h) or len(np.unique(edges))!=2*len(edges):raise ValueError('Disjoint oriented pair ABI differs')
    eh=h-target;ez=z-zt
    return {'Hendpoint':w[:,None]*eh/(scales[0] or 1.),'Hresponse':pair_part(h,target,edges,v,scales[1]),
            'Zendpoint':pullback(h,w[:,None]*ez/(scales[2] or 1.),gamma,epsilon),
            'Zresponse':pullback(h,pair_part(z,zt,edges,v,scales[3]),gamma,epsilon)}


def columns():
    result=[g+'.'+d for g in GRADIENTS for d in DIRECTIONS]
    result += [g+'.'+r for g in GRADIENTS for r in ('mappingRegrouping','admissionRegrouping','totalRegrouping')]
    result += ['componentSumResidual.'+d for d in DIRECTIONS]
    result += ['squaredNorm.'+d for d in DIRECTIONS]
    return result+['vectorClosureSquaredNorm','vectorClosureMaximumAbsolute']


def analyze(gradient,parts,proposal,writer,admitted):
    gradient,proposal,writer,admitted=[finite(x) for x in (gradient,proposal,writer,admitted)]
    if gradient.ndim!=2 or any(x.shape!=gradient.shape for x in (proposal,writer,admitted)) or tuple(parts)!=TERMS or any(finite(x).shape!=gradient.shape for x in parts.values()):raise ValueError('Direction ABI differs')
    vectors={'proposal':proposal,'writer':writer,'admitted':admitted,'mapping':writer-proposal,'admission':admitted-writer}
    products={};coordinate_products=0
    def dot(a,b):
        nonlocal coordinate_products
        coordinate_products+=a.size
        return np.sum(a*b,axis=1,dtype=np.float64)
    for g,value in {'saved':gradient,**parts}.items():
        for d,vector in vectors.items():products[g+'.'+d]=dot(value,vector)
    for g in GRADIENTS:
        products[g+'.mappingRegrouping']=products[g+'.mapping']-(products[g+'.writer']-products[g+'.proposal'])
        products[g+'.admissionRegrouping']=products[g+'.admission']-(products[g+'.admitted']-products[g+'.writer'])
        products[g+'.totalRegrouping']=products[g+'.admitted']-((products[g+'.proposal']+products[g+'.mapping'])+products[g+'.admission'])
    for d in DIRECTIONS:products['componentSumResidual.'+d]=products['saved.'+d]-sum((products[g+'.'+d] for g in TERMS),np.zeros(len(gradient)))
    for d,value in vectors.items():products['squaredNorm.'+d]=dot(value,value)
    closure=admitted-((proposal+vectors['mapping'])+vectors['admission'])
    products['vectorClosureSquaredNorm']=dot(closure,closure);products['vectorClosureMaximumAbsolute']=np.max(np.abs(closure),axis=1)
    matrix=np.column_stack([products[k] for k in columns()])
    if not np.isfinite(matrix).all():raise ValueError('Nonfinite explanatory reduction')
    return matrix,{'coordinateProducts':coordinate_products,'rows':len(gradient),'coordinates':gradient.shape[1],'columns':columns()}
