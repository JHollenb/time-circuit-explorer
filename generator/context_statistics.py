"""A fixed positive-kernel context state, not an exact-attention implementation.

Serving receives raw text, compiled lexical atoms and fixed Gaussian features.
It never receives original matrices, attention targets or a native fallback.
"""
import argparse
import hashlib
from pathlib import Path
import re
import resource
import time

from encoder_atoms import np,Prefix,guard,read,save,sha,words,floats,bf


def lexical_head(prefix,prompt,head):
    text,ids,mask=prefix.tokenize(prompt);m=prefix.m;d=m['headDim']
    heads=m['qWidth']//d;kv_heads=m['kvWidth']//d
    if not 0<=head<heads or heads%kv_heads:raise ValueError('Invalid GQA mapping')
    kv=head//(heads//kv_heads);atoms=prefix.store.gather(ids);h=m['hiddenSize'];qw=m['qWidth'];kw=m['kvWidth']
    q=atoms[:,h+head*d:h+(head+1)*d];k=atoms[:,h+qw+kv*d:h+qw+(kv+1)*d]
    v=atoms[:,h+qw+kw+kv*d:h+qw+kw+(kv+1)*d]
    cos,sin=[floats(a) for a in prefix.rope]
    def rotate(value):
        x=floats(value);half=d//2;turn=np.concatenate([-x[:,half:],x[:,:half]],axis=1)
        return words(bf(x*cos)+bf(turn*sin))
    return {'ids':ids,'mask':mask,'positions':np.arange(len(ids),dtype=np.int64),
            'q':rotate(q),'k':rotate(k),'v':v.copy()},hashlib.sha256(text.encode()).hexdigest()


def new_state(d,rank):
    state={'count':0,'steps':0,'lo':np.full(d,np.inf),'hi':np.full(d,-np.inf)}
    if rank:
        state.update(maximum=np.full(rank,-np.inf),z=np.zeros(rank),M=np.zeros((rank,d)))
    else:state['sum']=np.zeros(d)
    return state


def copy_state(state):return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in state.items()}


def advance(state,q,k,v,allowed,omega):
    d=len(v);rank=0 if omega is None else len(omega)
    if allowed:
        state['count']+=1;state['lo']=np.minimum(state['lo'],v);state['hi']=np.maximum(state['hi'],v)
        if rank:
            y=k/(d**.25);logkey=omega@y-.5*np.dot(y,y)-.5*np.log(rank)
            maximum=np.maximum(state['maximum'],logkey)
            rescale=np.exp(state['maximum']-maximum);added=np.exp(logkey-maximum)
            state['z']=state['z']*rescale+added
            state['M']=state['M']*rescale[:,None]+added[:,None]*v[None,:]
            state['maximum']=maximum
        else:state['sum']+=v
    state['steps']+=1
    if state['count']==0:return np.zeros(d),{'empty':True,'logDenominator':0.,'featureMaxMass':0.,'featureEntropy':0.,'hullViolation':0.}
    if rank:
        x=q/(d**.25);logquery=omega@x-.5*np.dot(x,x)-.5*np.log(rank)
        logs=logquery+state['maximum'];maximum=float(np.max(logs));weights=np.exp(logs-maximum)
        denominator=float(weights@state['z']);h=(weights@state['M'])/denominator
        if denominator<=0 or not np.isfinite(denominator) or not np.isfinite(h).all():raise ValueError('Invalid positive-kernel read')
        mass=weights*state['z']/denominator
        entropy=float(-np.sum(mass*np.log(mass,where=mass>0,out=np.zeros_like(mass))))
        logden=maximum+float(np.log(denominator));maxmass=float(np.max(mass))
    else:
        h=state['sum']/state['count'];logden=float(np.log(state['count']));maxmass=1.;entropy=0.
    violation=max(0.,float(np.max(h-state['hi'])),float(np.max(state['lo']-h)))
    return h,{'empty':False,'logDenominator':logden,'featureMaxMass':maxmass,'featureEntropy':entropy,'hullViolation':violation}


def stream(packet,omega,state=None,checkpoints=()):
    q,k,v=[floats(packet[n]).astype(np.float64) for n in ['q','k','v']]
    if q.shape!=k.shape or q.shape!=v.shape or q.ndim!=2:raise ValueError('Invalid head operands')
    rank=0 if omega is None else len(omega);state=new_state(q.shape[1],rank) if state is None else copy_state(state)
    if int(packet['positions'][0])!=state['steps']:raise ValueError('State clock differs from consumed positions')
    heads=[];trace=[];snapshots=[];indices=[]
    for i in range(len(q)):
        h,record=advance(state,q[i],k[i],v[i],bool(packet['mask'][i]),omega);heads.append(h);trace.append(record)
        if int(packet['positions'][i]) in checkpoints:snapshots.append(copy_state(state));indices.append(int(packet['positions'][i]))
    raw=np.stack(heads)
    output={'raw':raw,'rounded':words(raw),'empty':np.asarray([r['empty'] for r in trace]),
            **{k:np.asarray([r[k] for r in trace]) for k in ['logDenominator','featureMaxMass','featureEntropy','hullViolation']},
            'state_count':np.asarray(state['count']),'state_steps':np.asarray(state['steps']),
            'state_lo':state['lo'],'state_hi':state['hi'],'checkpoint_positions':np.asarray(indices,dtype=np.int64)}
    for key in (['maximum','z','M'] if rank else ['sum']):
        output['state_'+key]=state[key]
        if snapshots:output['checkpoint_'+key]=np.stack([s[key] for s in snapshots])
    return output,state


def state_bytes(d,rank):return (rank*(d+2)*8 if rank else d*8)+16


def serve(spec_path,output):
    guard();spec=read(spec_path);out=Path(output);out.mkdir(exist_ok=False)
    ids=[p['id'] for p in spec['prompts']]
    if len(set(ids))!=len(ids) or any(not re.fullmatch('[a-z0-9_-]{1,64}',i) for i in ids):raise ValueError('Invalid prompt IDs')
    if sha(spec['omega']['path'])!=spec['omega']['sha256']:raise ValueError('Feature constants changed')
    omega=np.load(spec['omega']['path'],allow_pickle=False)
    if omega.shape!=(max(spec['ranks']),128) or not np.isfinite(omega).all():raise ValueError('Invalid frozen features')
    start=time.perf_counter();cpu=time.process_time();t=time.perf_counter();prefix=Prefix(spec['store'],spec['manifestSHA256']);init=time.perf_counter()-t
    rows=[];ports=[]
    for item in spec['prompts']:
        t=time.perf_counter();packet,text_sha=lexical_head(prefix,item['prompt'],spec['head']);first=time.perf_counter()-t
        path=out/(item['id']+'-port.npz');np.savez(path,**packet)
        ports.append({'id':item['id'],'path':str(path),'sha256':sha(path),'renderedTextSHA256':text_sha,'firstPrefixSeconds':first,
                      'activeTokens':int(packet['mask'].sum()),'uniqueTokenIDs':int(len(np.unique(packet['ids'])))})
        checkpoints=sorted({0,int(packet['mask'].sum())-1,len(packet['ids'])-1})
        for rank in [*spec['ranks'],0]:
            features=omega[:rank] if rank else None;method='rank'+str(rank) if rank else 'uniform'
            t=time.perf_counter();prediction,state=stream(packet,features,checkpoints=checkpoints);kernel_s=time.perf_counter()-t
            path=out/(item['id']+'-'+method+'.npz');np.savez(path,**prediction)
            samples=[]
            for _ in range(spec['repeats']):
                t=time.perf_counter();again,_=lexical_head(prefix,item['prompt'],spec['head']);stream(again,features,checkpoints=checkpoints);samples.append(time.perf_counter()-t)
            n=len(packet['ids']);active=int(packet['mask'].sum());d=packet['q'].shape[1]
            rows.append({'id':item['id'],'method':method,'rank':rank,'path':str(path),'sha256':sha(path),
                         'kernelWithDiagnosticsSeconds':kernel_s,'warmRawTextSeconds':samples,'warmRawTextMedianSeconds':float(np.median(samples)),
                         'contextStateBytes':state_bytes(d,rank),'diagnosticRangeBytes':2*d*8,'featureConstantBytes':rank*d*8,
                         'featureProjectionMACsPerCall':(n+active)*rank*d,'readNumeratorMACsPerCall':n*rank*d,
                         'partitionMACsPerCall':n*rank,'inputNormSquareSumMACsPerCall':(n+active)*d if rank else 0,
                         'stateRescaleMultipliesPerCall':active*rank*(d+1),'stateNewValueMultipliesPerCall':active*rank*d,
                         'stateAdditionsPerCall':active*rank*(d+1),'keyFeatureAndRescaleExpPerCall':2*active*rank,
                         'queryFeatureExpPerCall':n*rank,'featureEntropyLogsPerCall':n*rank,
                         'otherScalarLogsPerCall':active+2*n if rank else n,
                         'outputAndTraceBytes':sum(v.nbytes for v in prediction.values()),'emptyQueryRows':int(prediction['empty'].sum()),
                         'maximumHullViolation':float(prediction['hullViolation'].max())})
    save(out/'prediction-seal.json',{'schema':'tce-context-statistics-seal-v1','specSHA256':sha(spec_path),'manifestSHA256':spec['manifestSHA256'],
         'omega':spec['omega'],'rows':rows,'ports':ports,'initializationSecondsAfterImports':init,
         'wallSecondsAfterImports':time.perf_counter()-start,'cpuSecondsAfterImports':time.process_time()-cpu,
         'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'uniqueLearnedAtomRowsRead':prefix.store.rowsRead,'learnedAtomPayloadBytesRead':prefix.store.rowBytesRead,
         'frameworkCalls':0,'originalDenseProjections':0,'exactAttentionEvaluations':0,'learnedInformationConsumed':True,
         'fullLearnedAtomStorePayloadBytes':prefix.m['coverageCount']*prefix.m['rowWidth']*2,
         'scope':'One first-layer Qwen3 head. Fixed positive-kernel approximation is model-derived computation, not strict zero-neural image generation. Full lexical rows remain cached; no target/native attention fallback.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('output');a=p.parse_args();serve(a.spec,a.output)
