"""FP64 explanatory algebra around six actual finite normalization operands."""
import itertools
import math
import numpy as np
from finite_consumer_crossover import finite,norm2,dot,decompose,VECTOR_FIELDS

STATES=('a','b','p','q','Xno','Xon')
EDGES={'native':('a','b'),'own':('p','q'),'no':('a','Xno'),'on':('p','Xon')}
CONTRASTS={'E':('own','native'),'Uno':('own','no'),'Vno':('no','native'),'Uon':('own','on'),'Von':('on','native')}
EDGE_FIELDS=('delta','D','B','R','Z','g')
FACTOR_FIELDS=('D','B','R','G','g','closure')
ANALYSIS_NAMES=tuple([f'state-{s}-{v}' for s in STATES for v in ('ideal','rho')]+[f'edge-{e}-{v}' for e in EDGES for v in EDGE_FIELDS]+
                     ['response-'+v for v in VECTOR_FIELDS]+[f'factor-{c}-{v}' for c in CONTRASTS for v in FACTOR_FIELDS]+['endpoint-p-error','endpoint-q-error'])
SCORING='Contiguous FP64 coordinate products and np.sum(dtype=float64); ordered vector sums; all signed Gram terms via math.fsum. No scientific threshold.'
SOURCE_POLICY='Original CUDA FP32 square/mean/epsilon/rsqrt/rescale, BF16 admission, then original BF16 gamma product; full [1,512,2560].'
EXPLANATORY_POLICY='FP64 sum(h*h)/width+epsilon, sqrt, reciprocal; ideal=gamma*(h*inverse); D=gamma*(delta/radiusRight); B=gamma*(h*(inverseRight-inverseLeft)). rho includes rounding of this explanatory evaluation. Left-associated sums retain edge g, contrast g and final closure separately.'


def ordered_sum(values):
    result=values[0].copy()
    for value in values[1:]:result=result+value
    return result


def sum_summary(width,norms,dots,components,reference):
    relatives={k:math.sqrt(v/reference) if reference else None for k,v in norms.items()}
    terms=[norms[k] for k in components]+[2*dots[a+':'+b] for a,b in itertools.combinations(components,2)]
    cosines={a+':'+b:dots[a+':'+b]/math.sqrt(norms[a]*norms[b]) if norms[a]*norms[b] else None for a,b in itertools.combinations(components,2)}
    return {'rms':{k:math.sqrt(v/width) for k,v in norms.items()},'relativeToNative':relatives,'gramTerms':terms,'gramSum':math.fsum(terms),
            'normSquaredMinusGram':norms['target']-math.fsum(terms),'componentCosines':cosines}


def observe_sum(target,components,reference):
    closure=target-ordered_sum(list(components.values()));values={'target':target,'closure':closure,**components}
    norms={k:norm2(v) for k,v in values.items()};dots={a+':'+b:dot(components[a],components[b]) for a,b in itertools.combinations(components,2)}
    report={'width':len(target),'components':list(components),'referenceSquared':reference,'normSquared':norms,'innerProducts':dots,
            **sum_summary(len(target),norms,dots,tuple(components),reference)}
    return closure,report


def endpoint_metric(pred,target):
    p,t,e,ip=norm2(pred),norm2(target),norm2(pred-target),dot(pred,target);d=len(pred)
    return {'predictionSquared':p,'targetSquared':t,'errorSquared':e,'innerProduct':ip,'errorRMS':math.sqrt(e/d),'normalizerRMS':math.sqrt(t/d),
            'relativeRMS':math.sqrt(e/t) if t else None,'predictionRMS':math.sqrt(p/d),'targetRMS':math.sqrt(t/d),'cosine':ip/math.sqrt(p*t) if p*t else None}


def analyze(states,operands,gamma,epsilon):
    if set(states)!=set(STATES) or set(operands)!=set(STATES) or not math.isfinite(epsilon) or epsilon<=0:raise ValueError('State domain or epsilon differs')
    states={k:finite(v) for k,v in states.items()};operands={k:finite(v) for k,v in operands.items()};gamma=finite(gamma)
    if gamma.ndim!=1 or not gamma.size or any(v.shape!=gamma.shape for v in [*states.values(),*operands.values()]):raise ValueError('Row/gamma ABI differs')
    width=len(gamma);vectors={};radii={};state_rows={}
    for key in STATES:
        h=states[key];energy=norm2(h);squared=energy/width+epsilon;radius=math.sqrt(squared);inverse=1/radius;radii[key]=radius
        ideal=gamma*(h*inverse);rho=operands[key]-ideal;vectors['state-'+key+'-ideal']=ideal;vectors['state-'+key+'-rho']=rho
        state_rows[key]={'inputNormSquared':energy,'squaredRadius':squared,'radius':radius,'inverseRadius':inverse,'idealNormSquared':norm2(ideal),'rhoNormSquared':norm2(rho)}
    response,response_report=decompose(*(operands[k] for k in STATES));reference=response_report['normSquared']['N'];edges={};edge_reports={}
    for name,(left,right) in EDGES.items():
        h,k=states[left],states[right];delta=k-h;direction=gamma*(delta/radii[right]);baseline=gamma*(h*(1/radii[right]-1/radii[left]))
        remainder=vectors['state-'+right+'-rho']-vectors['state-'+left+'-rho'];actual=operands[right]-operands[left]
        regroup=actual-ordered_sum([direction,baseline,remainder]);parts={'D':direction,'B':baseline,'R':remainder,'g':regroup}
        _,edge_reports[name]=observe_sum(actual,parts,reference);edges[name]={'delta':delta,**parts,'Z':actual}
        for field in EDGE_FIELDS:vectors['edge-'+name+'-'+field]=edges[name][field]
    for key,value in response.items():vectors['response-'+key]=value
    factors={}
    for name,(left,right) in CONTRASTS.items():
        parts={k:edges[left][k]-edges[right][k] for k in ('D','B','R')};parts['G']=edges[left]['g']-edges[right]['g']
        parts['g']=response[name]-ordered_sum(list(parts.values()));closure,factors[name]=observe_sum(response[name],parts,reference)
        for key,value in {**parts,'closure':closure}.items():vectors['factor-'+name+'-'+key]=value
    endpoints={}
    for own,native in [('p','a'),('q','b')]:
        vectors['endpoint-'+own+'-error']=operands[own]-operands[native];endpoints[own]=endpoint_metric(operands[own],operands[native])
    if set(vectors)!=set(ANALYSIS_NAMES) or len(vectors)!=79:raise ValueError('Complete analysis field domain differs')
    return vectors,{'width':width,'scoring':SCORING,'explanatoryPolicy':EXPLANATORY_POLICY,'edgeMap':EDGES,'contrastMap':CONTRASTS,'states':state_rows,'response':response_report,'edges':edge_reports,'factors':factors,'endpoints':endpoints}
