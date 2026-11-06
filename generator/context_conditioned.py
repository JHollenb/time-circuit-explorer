"""Raw-text serving with one source-frozen, diagonally conditioned prefix state."""
import argparse
import time
import resource
from pathlib import Path

from encoder_atoms import np,Prefix,read,save,sha,guard,floats,words
from context_statistics import lexical_head,new_state,copy_state,stream as original_stream,state_bytes


def advance_logs(state,lq,lk,v,allowed):
    if allowed:
        state['count']+=1;state['lo']=np.minimum(state['lo'],v);state['hi']=np.maximum(state['hi'],v)
        top=np.maximum(state['maximum'],lk);rescale=np.exp(state['maximum']-top);added=np.exp(lk-top)
        state['z']=state['z']*rescale+added;state['M']=state['M']*rescale[:,None]+added[:,None]*v[None,:];state['maximum']=top
    state['steps']+=1
    if not state['count']:return np.zeros(len(v)),{'empty':True,'featureMaxMass':0.,'featureEntropy':0.,'logDenominator':0.,'hullViolation':0.}
    logs=lq+state['maximum'];top=float(logs.max());weights=np.exp(logs-top);den=float(weights@state['z'])
    h=(weights@state['M'])/den
    if den<=0 or not np.isfinite(den) or not np.isfinite(h).all():raise ValueError('Invalid conditioned read')
    mass=weights*state['z']/den;positive=mass>0
    return h,{'empty':False,'featureMaxMass':float(mass.max()),'featureEntropy':float(-np.sum(mass[positive]*np.log(mass[positive]))),
        'logDenominator':top+float(np.log(den)),
        'hullViolation':max(0.,float(np.max(h-state['hi'])),float(np.max(state['lo']-h)))}


def stream(packet,omega,geometry,state=None,checkpoints=()):
    q,k,v=[floats(packet[n]).astype(np.float64) for n in ['q','k','v']];n,d=q.shape;r=len(omega)
    a,b,D=[geometry[key] for key in ['a','b','D']]
    if q.shape!=k.shape or k.shape!=v.shape or any(z.shape!=(d,) for z in [a,b,D]) or np.any(D<=0) or not all(np.isfinite(z).all() for z in [a,b,D,omega]):raise ValueError('Conditioned geometry differs')
    state=new_state(d,r) if state is None else copy_state(state)
    if int(packet['positions'][0])!=state['steps']:raise ValueError('State clock differs')
    x=q/d**.25;y=k/d**.25;u=(x-a)*D;valid=packet['mask'].astype(bool);w=(y[valid]-b)/D
    lq=u@omega.T-.5*np.sum(u*u,1)[:,None]-.5*np.log(r)
    lk=np.zeros((n,r));lk[valid]=w@omega.T-.5*np.sum(w*w,1)[:,None]+(y[valid]@a)[:,None]-.5*np.log(r)
    output=[];trace=[];snapshots=[];indices=[]
    for i in range(n):
        h,t=advance_logs(state,lq[i],lk[i],v[i],bool(valid[i]));output.append(h);trace.append(t)
        if int(packet['positions'][i]) in checkpoints:snapshots.append(copy_state(state));indices.append(int(packet['positions'][i]))
    raw=np.stack(output);result={'raw':raw,'rounded':words(raw),'empty':np.asarray([t['empty'] for t in trace]),
        **{key:np.asarray([t[key] for t in trace]) for key in ['featureMaxMass','featureEntropy','logDenominator','hullViolation']},
        'state_count':np.asarray(state['count']),'state_steps':np.asarray(state['steps']),'checkpoint_positions':np.asarray(indices,dtype=np.int64)}
    for key in ['maximum','z','M','lo','hi']:
        result['state_'+key]=state[key]
        if snapshots:result['checkpoint_'+key]=np.stack([s[key] for s in snapshots])
    result['logDenominatorGauge']=result['logDenominator'].copy()
    result['queryGaugeLogCorrection']=x@b-float(a@b)
    result['logDenominator']=np.where(result['empty'],0.,result['logDenominatorGauge']+result['queryGaugeLogCorrection'])
    return result,state


def serve(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False)
    for key in ['omega','geometry']:
        if sha(spec[key]['path'])!=spec[key]['sha256']:raise ValueError('Frozen asset changed: '+key)
    omega=np.load(spec['omega']['path'],allow_pickle=False)
    with np.load(spec['geometry']['path'],allow_pickle=False) as p:geometry={k:p[k] for k in ['a','b','D']}
    t=time.perf_counter();cpu=time.process_time();prefix=Prefix(spec['store'],spec['manifestSHA256']);init=time.perf_counter()-t
    rows=[];ports=[]
    for item in spec['prompts']:
        first=time.perf_counter();packet,text_sha=lexical_head(prefix,item['prompt'],0);first=time.perf_counter()-first
        p=out/(item['id']+'-port.npz');np.savez(p,**packet);active=int(packet['mask'].sum());n,d=packet['q'].shape
        ports.append({'id':item['id'],'path':str(p),'sha256':sha(p),'renderedTextSHA256':text_sha,'activeTokens':active,'firstPrefixSeconds':first})
        checkpoints=sorted({0,active-1,n-1})
        for method,rank in [('conditioned16',16),('conditioned32',32),('conditioned64',64),('original64',64),('uniform',0)]:
            def call(p):
                if method.startswith('conditioned'):return stream(p,omega[:rank],geometry,checkpoints=checkpoints)
                return original_stream(p,omega[:rank] if rank else None,checkpoints=checkpoints)
            start=time.perf_counter();prediction,_=call(packet);first_s=time.perf_counter()-start
            p=out/(item['id']+'-'+method+'.npz');np.savez(p,**prediction);samples=[]
            for _ in range(spec['repeats']):
                start=time.perf_counter();again,_=lexical_head(prefix,item['prompt'],0);call(again);samples.append(time.perf_counter()-start)
            conditioned=method.startswith('conditioned')
            rows.append({'id':item['id'],'method':method,'rank':rank,'path':str(p),'sha256':sha(p),'kernelSeconds':first_s,
                'warmRawTextSeconds':samples,'warmRawTextMedianSeconds':float(np.median(samples)),
                'contextStateBytes':state_bytes(d,rank),'diagnosticRangeBytes':2*d*8,'featureConstantBytes':rank*d*8,
                'geometryBytes':3*d*8 if conditioned else 0,'preformedLogFeatureBytes':2*n*rank*8 if conditioned else 0,
                'featureProjectionMACsPerCall':(n+active)*d*rank,'keyBiasMACsPerCall':active*d if conditioned else 0,
                'queryGaugeCorrectionMACsPerCall':n*d+d if conditioned else 0,
                'rawOperandScaleDivisionsPerCall':2*n*d if conditioned else (n+active)*d if rank else 0,
                'inputNormSquareSumMACsPerCall':(n+active)*d if rank else 0,
                'coordinateShiftOperationsPerCall':(n+active)*d if conditioned else 0,'coordinateScaleOperationsPerCall':(n+active)*d if conditioned else 0,
                'readNumeratorMACsPerCall':n*rank*d,'partitionMACsPerCall':n*rank,
                'stateRescaleMultipliesPerCall':active*rank*(d+1),'stateNewValueMultipliesPerCall':active*rank*d,
                'stateAdditionsPerCall':active*rank*(d+1),'keyFeatureAndRescaleExpPerCall':2*active*rank,'queryFeatureExpPerCall':n*rank,
                'featureEntropyLogsPerCall':n*rank,'otherScalarLogsPerCall':active+2*n if rank else n,
                'outputAndTraceBytes':sum(z.nbytes for z in prediction.values()),'maximumHullViolation':float(prediction['hullViolation'].max())})
    save(out/'prediction-seal.json',{'schema':'tce-source-conditioned-context-seal-v1','specSHA256':sha(spec_path),'manifestSHA256':spec['manifestSHA256'],
        'omega':spec['omega'],'geometry':spec['geometry'],'rows':rows,'ports':ports,'initializationSecondsAfterImports':init,
        'wallSecondsAfterImports':time.perf_counter()-t,'cpuSecondsAfterImports':time.process_time()-cpu,
        'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'learnedAtomRowsRead':prefix.store.rowsRead,'learnedAtomPayloadBytesRead':prefix.store.rowBytesRead,
        'frameworkCalls':0,'originalDenseProjections':0,'exactAttentionEvaluations':0,'learnedInformationConsumed':True,
        'denominatorPolicy':'Conditioned logDenominatorGauge is transformed-kernel partition. logDenominator adds x dot b minus a dot b for original-gauge comparison, not a claim of exact kernel accuracy.',
        'scope':'Source-frozen first-layer approximate context read from new raw text. No prompt-fitted geometry, original model or target fallback; not an image generator or strict zero-neural result.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('output');a=p.parse_args();serve(a.spec,a.output)
