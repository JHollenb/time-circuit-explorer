"""Saved-number operator envelopes. No matrix products or activation evaluations.

All numerical intervals are conditional on the explicit model in MODEL. The
exact-real identity is independent of that model; cuBLAS is not attested here.
"""
import math
import numpy as np

U=2.0**-53
MODEL={
    'norms':'IEEE binary64 round-to-nearest, finite normal nonzero intermediates; n-term reductions use at most n-1 additions; sqrt correctly rounded. Norm allowance gamma_(2n+4).',
    'historicalContractions':'Conditional only: each original FP64 contraction satisfies elementwise |fl(AD^T)-AD^T| <= gamma_(2m)|A||D|^T, m=9216. Source/operand fingerprints do not certify cuBLAS rounding.',
    'elementwise':'IEEE binary64 round-to-nearest; no overflow/underflow; stored products are the target, not ideal native activations.',
    'intervalArithmetic':'Positive scalar bounds expanded with nextafter after each operation; norm/dot/reduction assumptions remain explicit and are not hardware certification.',
    'scope':'All conclusions involving eta are conditional, even when normality scans and independent toy references agree.'}


def up(x):
    x=float(x)
    return 0.0 if x==0 else math.nextafter(x,math.inf)


def add(*xs):return up(math.fsum(xs))
def mul(x,y):return up(float(x)*float(y))
def div(x,y):return up(float(x)/float(y))
def gamma(n):return div(n*U,math.nextafter(1-n*U,-math.inf))


class Bounds:
    def __init__(self):self.calls=0;self.nonNormalInputs=0;self.nonNormalSquares=0

    def norm(self,x):
        x=np.asarray(x,dtype=np.float64);sq=x*x;self.calls+=1
        tiny=np.finfo(np.float64).tiny
        self.nonNormalInputs+=int(np.count_nonzero((x!=0)&(np.abs(x)<tiny)))
        self.nonNormalSquares+=int(np.count_nonzero((x!=0)&((sq==0)|(sq<tiny))))
        if not np.isfinite(sq).all():raise ValueError('Nonfinite norm intermediate')
        q=float(np.sum(sq));z=gamma(2*x.size+4)
        if not math.isfinite(q):raise ValueError('Nonfinite norm reduction')
        if q==0:return [0.0,0.0]
        lo=max(0.0,math.nextafter(q/math.nextafter(1+z,math.inf),-math.inf));hi=div(q,math.nextafter(1-z,-math.inf))
        return [max(0.0,math.nextafter(math.sqrt(lo),-math.inf)),up(math.sqrt(hi))]

    def dot(self,a,b,an,bn):
        a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);products=a*b
        self.nonNormalSquares+=int(np.count_nonzero((a!=0)&(b!=0)&((products==0)|(np.abs(products)<np.finfo(np.float64).tiny))))
        z=float(np.sum(products));error=mul(gamma(2*a.size+4),mul(an[1],bn[1]))
        return [math.nextafter(z-error,-math.inf),math.nextafter(z+error,math.inf)]

    def status(self):return {'normCalls':self.calls,'nonNormalInputs':self.nonNormalInputs,'nonNormalProducts':self.nonNormalSquares,
        'normalityScanApplicable':self.nonNormalInputs==0 and self.nonNormalSquares==0,'historicalContractionModelAttested':False,
        'scope':'Normality scans cover norm/dot inputs and products, not every historical or current internal reduction instruction. Overall numerical envelope remains conditional.'}


def map_norms(d,g,support,bounds):
    omitted=np.ones(d.shape[1],dtype=bool);omitted[support]=False
    # g and D are retained numbers; map formation gets its own one-op envelope.
    result={}
    for name,index in [('omitted',omitted),('selected',~omitted),('full',slice(None))]:
        blocks=[]
        for lo in range(0,len(d),64):
            formed=g[lo:lo+64,None]*d[lo:lo+64,index]
            blocks.append(bounds.norm(formed)[1])
        result[name]=div(bounds.norm(blocks)[1],math.nextafter(1-U,-math.inf))
    return omitted,result


def consequences(r,t,dot,eta):
    lower=max(0.0,math.nextafter(r[0]-eta,-math.inf));upper=add(r[1],eta)
    uncertainty=mul(t[1],eta)
    alignment=[math.nextafter(dot[0]-uncertainty,-math.inf),math.nextafter(dot[1]+uncertainty,math.inf)]
    cosine=None
    if lower>0 and t[0]>0:
        dlo=max(0.0,math.nextafter(t[0]*lower,-math.inf));dhi=mul(t[1],upper)
        if dlo>0:
            candidates=[v/d for v in alignment for d in [dlo,dhi]]
            cosine=[max(-1.0,math.nextafter(min(candidates),-math.inf)),min(1.0,math.nextafter(max(candidates),math.inf))]
    return {'evenOutputNormInterval': [lower,upper],'quadraticEvenInnerInterval':alignment,'quadraticEvenCosineInterval':cosine,
        'etaOverRemainingNormUpper':div(eta,r[0]) if r[0]>0 else None,'conditionalNegativeAlignment':alignment[1]<0}


def row_record(f,i,omitted,maps,bounds):
    selected=~omitted;g=f['gate'].reshape(-1);m=f['R'].shape[-1]
    a={n:np.asarray(f[n][0,i],dtype=np.float64) for n in ['R','L','B','T','H','F','closure78','YR','YL','YT','remaining','gatedRemaining','gatedQuadratic']}
    norms={n:bounds.norm(a[n][omitted]) for n in ['R','L','B','T','H','F']}
    fullnorms={n:bounds.norm(a[n]) for n in ['R','L','T']}
    c1=(a['R']-a['L'])-a['B'];c2=a['B']-((a['T']+a['H'])+a['F'])
    direct=(((a['R']-a['L'])-a['T'])-a['H'])-a['F']
    c1n=bounds.norm(c1[omitted]);c2n=bounds.norm(c2[omitted]);directn=bounds.norm(direct[omitted])
    formation1=mul(gamma(2),add(*(norms[n][1] for n in ['R','L','B'])))
    formation2=mul(gamma(3),add(*(norms[n][1] for n in ['B','T','H','F'])))
    formationDirect=mul(gamma(4),add(*(norms[n][1] for n in ['R','L','T','H','F'])))
    c1Upper=add(c1n[1],formation1);c2Upper=add(c2n[1],formation2)
    selectedNorms={n:bounds.norm(a[n][selected]) for n in ['R','L','T']}
    selectedCalc=(a['R'][selected]-a['L'][selected])-a['T'][selected]
    selectedUpper=add(bounds.norm(selectedCalc)[1],mul(gamma(2),add(*(v[1] for v in selectedNorms.values()))))
    remainingCalc=(a['YR']-a['YL'])-a['YT'];gatedCalc=g*remainingCalc
    groupingDiff=a['gatedRemaining']-gatedCalc;groupingNorm=bounds.norm(groupingDiff)
    rn=bounds.norm(a['gatedRemaining']);tn=bounds.norm(a['gatedQuadratic']);calcNorm=bounds.norm(gatedCalc)
    # Difference formation plus source two subtractions and one multiplication.
    mismatchAllowance=mul(U,add(rn[1],calcNorm[1])) if np.any(groupingDiff) else 0.0
    sourceOutputNorms={n:bounds.norm(a[n]) for n in ['YR','YL','YT']}
    groupArithmetic=mul(gamma(3),mul(float(np.max(np.abs(g))),add(*(v[1] for v in sourceOutputNorms.values()))))
    components={'finiteProduct':mul(norms['F'][1],maps['omitted']),
        'closure1':mul(c1Upper,maps['omitted']),'closure2':mul(c2Upper,maps['omitted']),
        'selectedResidue':mul(selectedUpper,maps['selected']),
        'historicalContractions':mul(gamma(2*m),mul(add(*(v[1] for v in fullnorms.values())),maps['full'])),
        'groupingMismatch':add(groupingNorm[1],mismatchAllowance),'groupingArithmetic':groupArithmetic}
    eta=add(*components.values());dot=bounds.dot(a['gatedQuadratic'],a['gatedRemaining'],tn,rn)
    directBound=mul(add(directn[1],formationDirect),maps['omitted'])
    directEta=add(components['finiteProduct'],directBound,*(components[n] for n in ['selectedResidue','historicalContractions','groupingMismatch','groupingArithmetic']))
    closureDiff=bounds.norm(a['closure78']-c2)
    record={'row':i,'stratum':'text' if i<512 else 'image','remainingNormInterval':rn,'quadraticNormInterval':tn,
        'retainedQuadraticRemainingInnerInterval':dot,'etaUpper':eta,'components':components,
        'omittedProductNormIntervals':norms,'closure1ComputedNormInterval':c1n,'closure2ComputedNormInterval':c2n,
        'closureFormationAllowances':[formation1,formation2],'directProductResidualComputedNormInterval':directn,
        'directProductFormationAllowance':formationDirect,'directAlternativeEtaUpper':directEta,
        'selectedStoredNonzeros':{n:int(np.count_nonzero(a[n][selected])) for n in ['R','L','T']},
        'storedClosureVsRecomputedNormInterval':closureDiff,
        'remainingReplayMaxAbs':float(np.max(np.abs(a['remaining']-remainingCalc))),
        'gatedReplayMaxAbs':float(np.max(np.abs(groupingDiff))),**consequences(rn,tn,dot,eta)}
    return record


def summarize(rows,bounds):
    def aggregate_interval(name):return [bounds.norm([r[name][0] for r in rows])[0],bounds.norm([r[name][1] for r in rows])[1]]
    def sum_endpoint(index):
        vals=[r['retainedQuadraticRemainingInnerInterval'][index] for r in rows];s=math.fsum(vals)
        allowance=mul(gamma(len(vals)+2),add(*(abs(x) for x in vals)))
        return math.nextafter(s-allowance if index==0 else s+allowance,-math.inf if index==0 else math.inf)
    rn=aggregate_interval('remainingNormInterval');tn=aggregate_interval('quadraticNormInterval');dot=[sum_endpoint(0),sum_endpoint(1)]
    components={n:bounds.norm([r['components'][n] for r in rows])[1] for n in rows[0]['components']}
    # Norm of rowwise total envelopes can improve on summing aggregate components.
    eta=bounds.norm([r['etaUpper'] for r in rows])[1]
    return {'rows':len(rows),'remainingNormInterval':rn,'quadraticNormInterval':tn,'retainedQuadraticRemainingInnerInterval':dot,
        'etaUpper':eta,'componentNormUpper':components,'directAlternativeEtaUpper':bounds.norm([r['directAlternativeEtaUpper'] for r in rows])[1],
        'negativeAlignmentRowsConditional':sum(r['conditionalNegativeAlignment'] for r in rows),
        'selectedStoredNonzeros':sum(sum(r['selectedStoredNonzeros'].values()) for r in rows),
        'maxStoredClosureReplayNormUpper':max(r['storedClosureVsRecomputedNormInterval'][1] for r in rows),
        'maxRemainingReplayAbs':max(r['remainingReplayMaxAbs'] for r in rows),'maxGatedReplayAbs':max(r['gatedReplayMaxAbs'] for r in rows),
        **consequences(rn,tn,dot,eta)}
