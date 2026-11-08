"""Offline proposal algebra; finite consumer feedback decides continuation."""
import math
import numpy as np
from finite_consumer_crossover import finite
from contrast_writer_fit import quadratic,system
from contextual_writer_program import finite_add_array

POLICY='Four equal native-discovery-normalized finite H/Z endpoint and paired losses on learned rows2..511;zero energy uses absolute squared loss. Ideal Jacobian and scalar paired surrogate propose only;complete two-admission H and source-policy Z decide continuation.'
DOSES=(1.,.5)


class CandidateAdmissionError(ValueError):
    """Optional candidate arithmetic only; never custody/schema failures."""


def candidate_add(left,right):
    try:return finite_add_array(left,right)
    except ValueError as error:
        if str(error) not in ('Nonfinite offline writer output','Nonfinite rounded offline writer output'):raise
        raise CandidateAdmissionError(str(error)) from error


def measure(w,edges,v,n):
    w=finite(w);edges=np.asarray(edges);v=finite(v)
    if w.shape!=(n,) or np.any(w<=0) or edges.ndim!=2 or edges.shape[1]!=2 or edges.dtype.kind not in 'iu' or v.shape!=(len(edges),) or np.any(v<0):raise ValueError('Measure ABI differs')
    if len(edges) and (edges.min()<0 or edges.max()>=n or len(np.unique(edges))!=2*len(edges)):raise ValueError('Expected disjoint oriented pair edges')
    return w,edges.astype(np.int64),v


def energy(value,weights):
    result=float(np.sum(weights[:,None]*finite(value)**2,dtype=np.float64))
    if not math.isfinite(result):raise ValueError('Nonfinite objective energy')
    return result


def objective(h,z,target,zt,w,edges,v,scales=None):
    h,z,target,zt=[finite(a) for a in (h,z,target,zt)]
    if h.ndim!=2 or not h.size or any(a.shape!=h.shape for a in (z,target,zt)):raise ValueError('Complete row geometry differs')
    w,edges,v=measure(w,edges,v,len(h));i,j=edges.T
    native={'Hendpoint':energy(target,w),'Hresponse':energy(target[j]-target[i],v),'Zendpoint':energy(zt,w),'Zresponse':energy(zt[j]-zt[i],v)}
    scales=native if scales is None else scales
    if set(scales)!=set(native) or any(not math.isfinite(s) or s<0 for s in scales.values()):raise ValueError('Frozen native denominators differ')
    errors={'Hendpoint':energy(h-target,w),'Hresponse':energy((h[j]-h[i])-(target[j]-target[i]),v),
            'Zendpoint':energy(z-zt,w),'Zresponse':energy((z[j]-z[i])-(zt[j]-zt[i]),v)}
    terms={k:errors[k]/(scales[k] or 1.) for k in errors}
    return {'errorEnergies':errors,'nativeEnergies':dict(scales),'terms':terms,'score':math.fsum(terms.values())/4,
            'zeroReferenceUsesAbsolute':{k:not bool(scales[k]) for k in scales}}


def graph_error(error,pair_error,w,edges,v,endpoint_scale,response_scale):
    result=w[:,None]*error/(endpoint_scale or 1.);i,j=edges.T;change=v[:,None]*pair_error/(response_scale or 1.)
    np.add.at(result,i,-change);np.add.at(result,j,change);return result


def pullback(h,value,gamma,epsilon):
    h,value,gamma=finite(h),finite(value),finite(gamma)
    if h.ndim!=2 or value.shape!=h.shape or gamma.shape!=(h.shape[1],) or not math.isfinite(epsilon) or epsilon<=0:raise ValueError('Jacobian ABI differs')
    radius=np.sqrt(np.mean(h*h,axis=1,dtype=np.float64)+epsilon);weighted=gamma*value
    return weighted/radius[:,None]-h*(np.sum(h*weighted,axis=1,dtype=np.float64)/(h.shape[1]*radius**3))[:,None]


def pair_solve(gradient,e,j,edges,ph,pz):
    """Scaled positive determinant expansion; never a subtractive determinant."""
    gradient,e,j,ph,pz=[finite(a) for a in (gradient,e,j,ph,pz)];edges=np.asarray(edges)
    if gradient.ndim!=2 or not gradient.size or e.shape!=(len(gradient),) or j.shape!=e.shape or ph.shape!=(len(edges),) or pz.shape!=ph.shape or np.any(e<=0) or np.any(j<0) or np.any(ph<0) or np.any(pz<0) or edges.dtype.kind not in 'iu':raise ValueError('Paired surrogate ABI differs')
    if edges.ndim!=2 or edges.shape[1]!=2 or len(np.unique(edges))!=2*len(edges) or len(edges) and (edges.min()<0 or edges.max()>=len(e)):raise ValueError('Pair solve addresses differ')
    step=-gradient/e[:,None];residual=e[:,None]*step+gradient
    for at,(left,right) in enumerate(edges):
        ei,ej,ji,jj,p,q=e[left],e[right],j[left],j[right],ph[at],pz[at]
        scale=max(ei,ej,p,q*ji*ji,q*jj*jj);ei,ej,p,q=ei/scale,ej/scale,p/scale,q/scale
        a,b,c=ei+p+q*ji*ji,ej+p+q*jj*jj,p+q*ji*jj
        determinant=math.fsum([ei*ej,ei*(p+q*jj*jj),ej*(p+q*ji*ji),p*q*(ji-jj)**2])
        if not math.isfinite(determinant) or determinant<=0:raise ValueError('Invalid positive surrogate determinant')
        gl,gr=gradient[left]/scale,gradient[right]/scale
        step[left]=-(b*gl+c*gr)/determinant;step[right]=-(c*gl+a*gr)/determinant
        residual[left]=scale*(a*step[left]-c*step[right]+gl);residual[right]=scale*(b*step[right]-c*step[left]+gr)
    if not np.isfinite(step).all() or not np.isfinite(residual).all():raise ValueError('Nonfinite paired proposal')
    return step,{'solveResidualRMS':float(np.sqrt(np.mean(residual*residual,dtype=np.float64))),
                 'maximumAbsoluteSolveResidual':float(np.max(np.abs(residual))),'pairedRows':2*len(edges),'singletonRows':len(e)-2*len(edges)}


def propose(h,z,target,zt,w,edges,v,gamma,epsilon,scales):
    h,z,target,zt=[finite(a) for a in (h,z,target,zt)];w,edges,v=measure(w,edges,v,len(h));gamma=finite(gamma)
    if h.ndim!=2 or not h.size or any(a.shape!=h.shape for a in (z,target,zt)):raise ValueError('Proposal row ABI differs')
    i,j=edges.T
    gh=graph_error(h-target,(h[j]-h[i])-(target[j]-target[i]),w,edges,v,scales['Hendpoint'],scales['Hresponse'])
    gz=graph_error(z-zt,(z[j]-z[i])-(zt[j]-zt[i]),w,edges,v,scales['Zendpoint'],scales['Zresponse'])
    gradient=gh+pullback(h,gz,gamma,epsilon);radius=np.sqrt(np.mean(h*h,axis=1,dtype=np.float64)+epsilon)
    j=math.sqrt(float(np.mean(gamma*gamma,dtype=np.float64)))/radius
    e=w/(scales['Hendpoint'] or 1.)+w*j*j/(scales['Zendpoint'] or 1.)
    step,diagnostic=pair_solve(gradient,e,j,edges,v/(scales['Hresponse'] or 1.),v/(scales['Zresponse'] or 1.))
    return gradient,step,{**diagnostic,'gradientRMS':float(np.sqrt(np.mean(gradient*gradient,dtype=np.float64))),
                           'stepRMS':float(np.sqrt(np.mean(step*step,dtype=np.float64))),'proposalOnly':'g and A omit the same objective factor. j is heuristic RMS-gamma/radius,not an exact Hessian or descent bound. Finite admissions and rho treated approximately.'}


def dose_atoms(labels,leaves,old_slot,step,w,edges,v,scales,doses=DOSES):
    """Fixed proposed route, H graph metric; reuse one Cholesky for both RHS."""
    labels=np.asarray(labels,np.int64);w,edges,v=measure(w,edges,v,len(labels));old_slot,step=finite(old_slot),finite(step)
    if old_slot.shape!=step.shape or len(old_slot)!=len(labels) or set(labels)!=set(range(leaves)):raise ValueError('Complete route/atom domain differs')
    def q(value):return quadratic(value,w/(scales['Hendpoint'] or 1.),edges,v/(scales['Hresponse'] or 1.))
    base=q(old_slot);matrix,chol,_=system(base,labels,leaves);result=[];admissions=[]
    for dose in doses:
        target=old_slot+float(dose)*step;qr=q(target)['qr'];rhs=np.zeros((leaves,target.shape[1]),np.float64);np.add.at(rhs,labels,qr)
        solved=np.linalg.solve(chol.T,np.linalg.solve(chol,rhs))
        with np.errstate(over='ignore',invalid='ignore'):atoms=solved.astype('<f4')
        valid=bool(np.isfinite(atoms).all());result.append(atoms if valid else None)
        admissions.append({'dose':float(dose),'status':'admitted' if valid else 'numerically-inadmissible','reason':None if valid else 'Nonfinite FP32 dictionary admission'})
    return result,{'normalMatrixCondition':float(np.linalg.cond(matrix)),'doses':list(doses),'admissions':admissions,'metric':'H endpoint/pair graph only;not complete paired H/Z surrogate minimization. Dose refits oldSlot+dose*step;zero dose on this new route need not reproduce parent.'}


def choose(parent,candidates):
    """Parent wins ties; finite score is a continuation policy,not certification."""
    rows=[parent,*candidates]
    if any(not math.isfinite(r['score']) or r['score']<0 for r in rows):raise ValueError('Invalid finite objective')
    return min(enumerate(rows),key=lambda pair:(pair[1]['score'],pair[0]))[1]
