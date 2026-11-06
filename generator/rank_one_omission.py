"""Saved-only exact expansion of the two frozen positive-quadratic laws."""
from encoder_atoms import np
from sparse_tilt_study import logsum

TERMS=['mass','linearValue','quadraticValue','contractionArithmetic','legacyFiniteProfile','rounding']
SCALARS=['hasTail','fullScoreEnergy','rankScoreEnergy','omittedScoreEnergy','scoreCrossMoment','omittedOverFullEnergy','fullScoreMean','rankScoreMean','omittedScoreMean',
    'linearMass','quadraticMass','kappaFull','kappaRank','partitionDifferenceClosure','polynomialDifferenceClosure','fullTailWeight','rankTailWeight','tailValueClosure','headClosure']


def compare(s,U,V,z,selected,delta,beta):
    """All inputs are source-bound saved operands; no reference outputs enter."""
    n,d=V.shape
    if s.shape!=(n,) or U.shape!=(n,d) or z.shape!=(n,) or delta.shape!=(d,) or not len(selected) or len(np.unique(selected))!=len(selected) or np.any(selected<0) or np.any(selected>=n):raise ValueError('Invalid diagnostic geometry/support')
    if not all(np.isfinite(x).all() for x in [s,U,V,z,delta,np.asarray(beta)]):raise ValueError('Nonfinite diagnostic operand')
    score=s[selected]+U[selected]@delta;ls=logsum(score);hs=np.exp(score-ls)@V[selected];tail=np.ones(n,dtype=bool);tail[selected]=False;terms={k:np.zeros(d) for k in TERMS[:4]};scalars={k:np.nan for k in SCALARS};scalars['hasTail']=float(tail.any())
    if not tail.any():return hs,hs,terms,scalars
    st=s[tail];ut=U[tail];vt=V[tail];lz=logsum(st);p=np.exp(st-lz);mu=p@ut;t=(ut-mu)@delta;tr=beta*(z[tail]-float(p@z[tail]));eps=t-tr;L=eps*(1+tr);Q=.5*eps*eps;f=.5*((t+1)**2+1);g=.5*((tr+1)**2+1)
    if not all(np.isfinite(x).all() for x in [t,tr,L,Q,f,g]):raise ValueError('Polynomial diagnostic out of finite range')
    kf=float(p@f);kg=float(p@g);centroid=float(mu@delta)
    def law(x):
        with np.errstate(divide='ignore'):logr=np.logaddexp(2*np.log(np.abs(x+1)),0.)-np.log(2.)
        lp=logsum(st+logr);h=np.exp(st+logr-lp)@vt;logmass=lp+centroid;w=float(np.exp(logmass-logsum(np.array([ls,logmass]))));return (1-w)*hs+w*h,h,w
    hf,Hf,wf=law(t);hg,Hg,wg=law(tr);centered=vt-Hg;lv=(p*L)@centered/kf;qv=(p*Q)@centered/kf
    terms.update(mass=(wf-wg)*(Hg-hs),linearValue=wf*lv,quadraticValue=wf*qv);terms['contractionArithmetic']=hf-hg-terms['mass']-terms['linearValue']-terms['quadraticValue']
    second=float(p@(t*t));lost=float(p@(eps*eps));ml=float(p@L);mq=float(p@Q)
    scalars.update(fullScoreEnergy=second,rankScoreEnergy=float(p@(tr*tr)),omittedScoreEnergy=lost,scoreCrossMoment=float(p@(eps*tr)),omittedOverFullEnergy=lost/second if second else np.nan,
        fullScoreMean=float(p@t),rankScoreMean=float(p@tr),omittedScoreMean=float(p@eps),linearMass=ml,quadraticMass=mq,kappaFull=kf,kappaRank=kg,
        partitionDifferenceClosure=kf-kg-ml-mq,polynomialDifferenceClosure=float(np.max(np.abs(f-g-L-Q))),fullTailWeight=wf,rankTailWeight=wg,
        tailValueClosure=float(np.max(np.abs(Hf-Hg-lv-qv))),headClosure=float(np.max(np.abs(hf-hg-sum(terms.values())))))
    return hf,hg,terms,scalars
