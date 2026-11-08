"""One fixed-route, parent-preserving ray; no route fitting or dose search."""
import math
import numpy as np
from joint_writer_math import objective, energy, measure

TERMS=('Hendpoint','Hresponse','Zendpoint','Zresponse')
ETA=1/16
POLICY='One joint fixed-route leaf-mass Cauchy direction;full ideal forward RMS/gamma J;alpha=min(ray minimum,row-relative eta=1/16 trust bound);FP32 atom admission;actual two BF16 state adds and source-policy Z decide discovery continuation.'


def pushforward(h,q,gamma,epsilon):
    h,q,gamma=[np.asarray(x,np.float64) for x in (h,q,gamma)]
    if h.ndim!=2 or h.shape!=q.shape or gamma.shape!=(h.shape[1],) or not all(np.isfinite(x).all() for x in (h,q,gamma)) or not math.isfinite(epsilon) or epsilon<=0:raise ValueError('Forward Jacobian ABI differs')
    radius=np.sqrt(np.mean(h*h,axis=1,dtype=np.float64)+epsilon)
    return gamma*(q/radius[:,None]-h*(np.sum(h*q,axis=1,dtype=np.float64)/(h.shape[1]*radius**3))[:,None])


def admit(atoms,directions,alpha):
    if not math.isfinite(alpha) or alpha<0 or len(atoms)!=len(directions):raise ValueError('Atom update ABI differs')
    if alpha==0:return [a.copy() for a in atoms]  # preserves signed-zero bytes exactly
    out=[]
    for old,r in zip(atoms,directions):
        if old.dtype!=np.dtype('<f4') or old.shape!=r.shape:raise ValueError('FP32 parent atom ABI differs')
        with np.errstate(over='ignore',invalid='ignore'):new=(old.astype(np.float64)+alpha*r).astype('<f4')
        if not np.isfinite(new).all():raise ValueError('Nonfinite FP32 candidate atom')
        out.append(new)
    return out


def propose(data,labels,counts,epsilon):
    h,z,target,zt,g=[np.asarray(data[k],np.float64) for k in ('h','z','target','zt','gradient')]
    if h.ndim!=2 or not h.size or any(x.shape!=h.shape or not np.isfinite(x).all() for x in (h,z,target,zt,g)):raise ValueError('Complete proposal geometry differs')
    w,edges,v=measure(data['weights'],data['edges'],data['pairWeights'],len(h));scales=dict(zip(TERMS,data['scaleVector'].tolist()))
    parent=objective(h,z,target,zt,w,edges,v,scales);directions=[];masses=[];gradients=[]
    if len(labels)!=2 or len(counts)!=2:raise ValueError('Two fixed slots required')
    for assignment,count in zip(labels,counts):
        a=np.asarray(assignment)
        if a.shape!=(len(h),) or a.dtype.kind not in 'iu' or set(a.tolist())!=set(range(count)):raise ValueError('All fixed leaves must be occupied')
        mass=np.bincount(a,weights=w,minlength=count);btg=np.zeros((count,h.shape[1]),np.float64);np.add.at(btg,a,g)
        if np.any(mass<=0):raise ValueError('Nonpositive leaf mass')
        masses.append(mass);gradients.append(btg);directions.append(-btg/mass[:,None])
    q=sum((r[a] for r,a in zip(directions,labels)),np.zeros_like(h));jq=pushforward(h,q,data['gamma'],epsilon);i,j=edges.T
    terms={'Hendpoint':energy(q,w)/(scales['Hendpoint'] or 1.),'Hresponse':energy(q[j]-q[i],v)/(scales['Hresponse'] or 1.),
           'Zendpoint':energy(jq,w)/(scales['Zendpoint'] or 1.),'Zresponse':energy(jq[j]-jq[i],v)/(scales['Zresponse'] or 1.)}
    beta=math.fsum(terms.values());gq=float(np.sum(g*q,dtype=np.float64));identity=-math.fsum(float(np.sum(b*b/m[:,None],dtype=np.float64)) for b,m in zip(gradients,masses))
    tau=float(np.max(np.sqrt(np.mean(q*q,axis=1,dtype=np.float64))/np.sqrt(np.mean(h*h,axis=1,dtype=np.float64)+epsilon)))
    if beta==0 and tau==0:alpha=0.;model_alpha=None;trust_alpha=None
    else:
        if not math.isfinite(beta) or beta<=0 or not math.isfinite(gq) or gq>=0 or not math.isfinite(tau) or tau<=0:raise ValueError('Nonzero ray mechanics inconsistent')
        model_alpha=-gq/beta;trust_alpha=ETA/tau;alpha=min(model_alpha,trust_alpha)
    linear=objective(h+alpha*q,z+alpha*jq,target,zt,w,edges,v,scales)
    polynomial=parent['score']+alpha*gq/2+alpha*alpha*beta/4
    report={'parent':parent,'gDotQ':gq,'negativePreconditionedGradientEnergy':identity,'signIdentityRegrouping':gq-identity,'curvature':beta,'curvatureTerms':terms,
            'rowRelativeTau':tau,'eta':ETA,'alphaModel':model_alpha,'alphaTrust':trust_alpha,'alpha':alpha,'boundChosen':'zero-ray' if alpha==0 else 'model' if model_alpha<=trust_alpha else 'trust',
            'modelPolynomialScore':polynomial,'linearResidualSquareObjective':linear,'modelRegroupingDifference':polynomial-linear['score'],'predictedReduction':parent['score']-polynomial,
            'leafMasses':[m.tolist() for m in masses],'leaves':list(counts),'directionRMS':float(np.sqrt(np.mean(q*q))),'scope':'Saved source-qualified g,anchored ideal Jacobian model;not a derivative or minimizer of discrete finite BF16 execution.'}
    return directions,q,jq,report
