"""Source-atom moments only. No prompt strings or contextual target arrays enter fit."""
import time
from encoder_atoms import np,floats,words,bf


def moments(value):
    value=np.asarray(value,dtype=np.float64).reshape(-1,value.shape[-1]);mean=value.mean(0)
    return len(value),mean,np.sum((value-mean)**2,axis=0)


def merge(left,right):
    if left is None:return right
    n,a,M=left;m,b,N=right;delta=b-a
    return n+m,a+delta*(m/(n+m)),M+N+delta*delta*(n*m/(n+m))


def mixture(mean0,var0,mean1,var1,weight):
    return (1-weight)*mean0+weight*mean1,(1-weight)*var0+weight*var1+weight*(1-weight)*(mean1-mean0)**2


def rotate(value,rope):
    d=value.shape[-1];x=floats(value)[:,None,:];cos,sin=[floats(a)[None,:,:] for a in rope]
    turn=np.concatenate([-x[...,d//2:],x[...,:d//2]],axis=-1)
    return words(bf(x*cos)+bf(turn*sin))


def fit(prefix,ids,pad_weight=.5,batch=64,eps=1e-12):
    start=time.perf_counter();m=prefix.m;h=m['hiddenSize'];qw=m['qWidth'];d=m['headDim'];qstats=None;kstats=None
    if not 0<=pad_weight<=1 or eps<=0:raise ValueError('Invalid source moment policy')
    for offset in range(0,len(ids),batch):
        atoms=prefix.store.gather(ids[offset:offset+batch])
        q=rotate(atoms[:,h:h+d],prefix.rope);k=rotate(atoms[:,h+qw:h+qw+d],prefix.rope)
        qstats=merge(qstats,moments(floats(q).astype(np.float64)/d**.25))
        kstats=merge(kstats,moments(floats(k).astype(np.float64)/d**.25))
    pad=prefix.store.gather(np.asarray([m['tokenPolicy']['padID']],dtype=np.int64))
    pstats=moments(floats(rotate(pad[:,h:h+d],prefix.rope)).astype(np.float64)/d**.25)
    nq,muQ,Mq=qstats;nk,b,Mk=kstats;npad,muP,Mp=pstats
    a,A=mixture(muQ,Mq/nq,muP,Mp/npad,pad_weight);B=Mk/nk
    floorA=eps*float(A.mean());floorB=eps*float(B.mean());aa=np.maximum(A,floorA);bb=np.maximum(B,floorB)
    degenerate=float(A.mean())==0 or float(B.mean())==0;D=np.ones(d) if degenerate else (bb/aa)**.25
    if not all(np.isfinite(z).all() for z in [a,b,A,B,D]) or np.any(D<=0):raise ValueError('Nonfinite source geometry')
    receipt={'scope':'Source-token/position moments only; no raw prompts, contextual outputs or target-fitted parameters.',
        'sampleTokens':len(ids),'positionsPerToken':len(prefix.rope[0]),'queryPoolRows':nq,'keyPoolRows':nk,'padRows':npad,
        'padQueryMixtureWeight':pad_weight,'queryPoolMean':muQ.tolist(),'padQueryMean':muP.tolist(),
        'queryPoolVariance':(Mq/nq).tolist(),'padQueryVariance':(Mp/npad).tolist(),
        'queryCenter':a.tolist(),'keyCenter':b.tolist(),'queryVariance':A.tolist(),'keyVariance':B.tolist(),'scale':D.tolist(),
        'momentPolicy':'Stable FP64 block means/M2 with Chan merge; query mixture includes between-component mean variance.',
        'epsilon':eps,'flooredQueryCoordinates':np.flatnonzero(A<floorA).tolist(),'flooredKeyCoordinates':np.flatnonzero(B<floorB).tolist(),
        'degenerateEntireSide':degenerate,'fitSeconds':time.perf_counter()-start,
        'learnedAtomRowsRead':prefix.store.rowsRead,'learnedAtomPayloadBytesRead':prefix.store.rowBytesRead,
        'positionedQKVectorsFormed':nq+nk+npad,'sourceRoPEMultiplyOperations':2*d*(nq+nk+npad),
        'sourceRoPEAddOperations':d*(nq+nk+npad),'sourceRoPEBF16Boundaries':3*d*(nq+nk+npad),
        'geometryPayloadBytes':3*d*8,'frameworkCalls':0,'originalDenseProjections':0,'contextualAttentionCalls':0,
        'learnedInformationConsumed':True}
    return {'a':a,'b':b,'D':D},receipt
