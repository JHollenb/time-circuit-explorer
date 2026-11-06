"""Live nearest-landmark query transport via stable prefix cross moments.

The output is model-derived approximate attention, not a zero-neural image.
Original projections/frameworks and target/native refresh are absent at serving.
"""
import argparse
import hashlib
from pathlib import Path
import resource
import time

from encoder_atoms import np,Prefix,read,save,sha,guard,floats,words
from context_statistics import lexical_head,copy_state

METHODS=['nearest64','transport64']


def nearest(x,asset):
    points=x@asset['metricRoot'];landmarks=asset['landmarks']@asset['metricRoot']
    distances=np.sum((points[:,None,:]-landmarks[None,:,:])**2,axis=2);index=np.argmin(distances,axis=1)
    return index,distances[np.arange(len(x)),index]


def new_state(d,rank):
    return {'maximum':np.full(rank,-np.inf),'z':np.zeros(rank),'M':np.zeros((rank,d)),
        'Y':np.zeros((rank,d)),'C':np.zeros((rank,d,d)),'lo':np.full(d,np.inf),'hi':np.full(d,-np.inf),'count':0,'steps':0}


def update(state,u,v,score):
    maximum=np.maximum(state['maximum'],score);rescale=np.exp(state['maximum']-maximum);added=np.exp(score-maximum)
    old_mass=state['z']*rescale;mass=old_mass+added
    h=np.divide(state['M'],state['z'][:,None],out=np.zeros_like(state['M']),where=state['z'][:,None]>0)
    mu=np.divide(state['Y'],state['z'][:,None],out=np.zeros_like(state['Y']),where=state['z'][:,None]>0)
    factor=old_mass*added/mass
    state['C']*=rescale[:,None,None]
    state['C']+=((v-h)*factor[:,None])[:,:,None]*(u-mu)[:,None,:]
    state['M']=state['M']*rescale[:,None]+added[:,None]*v
    state['Y']=state['Y']*rescale[:,None]+added[:,None]*u
    state['z']=mass;state['maximum']=maximum;state['count']+=1
    state['lo']=np.minimum(state['lo'],v);state['hi']=np.maximum(state['hi'],v)
    if not all(np.isfinite(state[key]).all() for key in ['C','M','Y','z','maximum']) or np.any(state['z']<=0):raise ValueError('Nonfinite positive moment state')


def rounded(raw,valid):
    result=np.zeros(raw.shape,dtype=np.uint16);ok=valid&np.all(np.abs(raw)<=np.finfo(np.float32).max,axis=1)
    if ok.any():result[ok]=words(raw[ok]);ok&=np.isfinite(floats(result)).all(1)
    return result,ok


def stream(packet,asset,state=None,checkpoints=()):
    q,k,v=[floats(packet[n]).astype(np.float64) for n in ['q','k','v']];n,d=q.shape;ell=asset['landmarks'];rank=len(ell)
    if n==0 or q.shape!=k.shape or q.shape!=v.shape or ell.shape!=(rank,d) or asset['metricRoot'].shape!=(d,d) or asset['keyCenter'].shape!=(d,):raise ValueError('Transport shapes differ')
    if not all(np.isfinite(z).all() for z in [q,k,v,ell,asset['metricRoot'],asset['keyCenter']]):raise ValueError('Nonfinite transport operands')
    state=new_state(d,rank) if state is None else copy_state(state)
    tag=hashlib.sha256(b''.join(np.asarray(asset[key],dtype='<f8').tobytes() for key in ['landmarks','metricRoot','keyCenter'])).hexdigest()
    if state.get('assetIdentity',tag)!=tag:raise ValueError('State transport asset differs')
    state['assetIdentity']=tag
    if not np.array_equal(packet['positions'],np.arange(state['steps'],state['steps']+n)):raise ValueError('State clock differs')
    if packet['mask'].shape!=(n,) or np.any((packet['mask']!=0)&(packet['mask']!=1)):raise ValueError('Invalid binary mask')
    if state['C'].shape!=(rank,d,d) or state['Y'].shape!=(rank,d) or state['M'].shape!=(rank,d) or state['z'].shape!=(rank,):raise ValueError('Moment state shape differs')
    x=q/d**.25;u=k/d**.25-asset['keyCenter'];indices,distances=nearest(x,asset);allowed=packet['mask'].astype(bool)
    scores=np.zeros((n,rank));scores[allowed]=u[allowed]@ell.T
    heads={m:[] for m in METHODS};trace=[];snapshots=[];clocks=[]
    for i in range(n):
        if allowed[i]:update(state,u[i],v[i],scores[i])
        state['steps']+=1;a=int(indices[i]);delta=x[i]-ell[a];empty=not state['count']
        if empty:h=np.zeros(d);correction=np.zeros(d)
        else:h=state['M'][a]/state['z'][a];correction=(state['C'][a]@delta)/state['z'][a]
        values=[h,h+correction];record={'empty':empty,'nearestIndex':a,'sourceDistanceSquared':float(distances[i]),
            'queryDisplacementSquared':float(delta@delta),'correctionRms':float(np.sqrt(np.mean(correction*correction)))}
        for method,value in zip(METHODS,values):
            heads[method].append(value);finite=bool(np.isfinite(value).all());record[method+'_valid']=finite
            record[method+'_hullViolation']=0. if empty else max(0.,float(np.max(value-state['hi'])),float(np.max(state['lo']-value))) if finite else np.nan
        trace.append(record)
        if int(packet['positions'][i]) in checkpoints:snapshots.append(copy_state(state));clocks.append(int(packet['positions'][i]))
    output={key:np.asarray([r[key] for r in trace]) for key in trace[0]}
    for method in METHODS:
        output[method+'_raw']=np.stack(heads[method]);bits,valid=rounded(output[method+'_raw'],output[method+'_valid'])
        output[method+'_rounded']=bits;output[method+'_rounded_valid']=valid
    output['checkpoint_positions']=np.asarray(clocks,dtype=np.int64)
    for key,value in state.items():
        output['state_'+key]=np.asarray(value)
        if snapshots:output['checkpoint_'+key]=np.asarray([s[key] for s in snapshots])
    return output,state


def serve(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False)
    for name in ['transportAsset','interfaces']:
        if sha(spec[name]['path'])!=spec[name]['sha256']:raise ValueError('Frozen transport input differs')
    with np.load(spec['transportAsset']['path'],allow_pickle=False) as f:asset={key:f[key] for key in ['landmarks','metricRoot','keyCenter']}
    interfaces=read(spec['interfaces']['path']);start=time.perf_counter();cpu=time.process_time();prefix=Prefix(spec['store'],spec['manifestSHA256']);init=time.perf_counter()-start;rows=[];ports=[]
    for item in spec['prompts']:
        t=time.perf_counter();packet,text_sha=lexical_head(prefix,item['prompt'],0);first=time.perf_counter()-t;frozen=interfaces['prompts'][item['id']]
        if text_sha!=frozen['renderedTextSHA256'] or packet['ids'].tolist()!=frozen['ids'] or packet['mask'].tolist()!=frozen['mask']:raise ValueError('Prompt interface differs')
        path=out/(item['id']+'-port.npz');np.savez(path,**packet);n,d=packet['q'].shape;rank=len(asset['landmarks']);active=int(packet['mask'].sum())
        ports.append({'id':item['id'],'path':str(path),'sha256':sha(path),'activeTokens':active,'firstPrefixSeconds':first,'renderedTextSHA256':text_sha})
        checkpoints=sorted({0,active-1,n-1});t=time.perf_counter();output,_=stream(packet,asset,checkpoints=checkpoints);first_s=time.perf_counter()-t
        path=out/(item['id']+'-transport.npz');np.savez(path,**output);samples=[]
        for _ in range(spec['repeats']):
            t=time.perf_counter();again,_=lexical_head(prefix,item['prompt'],0);stream(again,asset,checkpoints=checkpoints);samples.append(time.perf_counter()-t)
        rows.append({'id':item['id'],'path':str(path),'sha256':sha(path),'methods':METHODS,'firstStatisticSeconds':first_s,
            'warmRawTextSeconds':samples,'warmMedianSeconds':float(np.median(samples)),
            'timingScope':'Nearest and firstorder together, raw prefix, all64crossmoments, diagnostics and fixed state checkpoint copies. Not standalone read latency.',
            'coreStateBytes':rank*(d*d+2*d+2)*8+16,'additionalCrossMomentBytes':rank*d*d*8,'additionalKeyMeanSumBytes':rank*d*8,
            'rangeDiagnosticBytes':2*d*8,'residentTransportAssetBytes':sum(a.nbytes for a in asset.values()),
            'landmarkScoreMACs':active*rank*d,'routingTransformMACs':(n+rank)*d*d,'routingDistanceSquaredProducts':n*rank*d,
            'mixedMomentScalarProducts':2*active*rank*d*d+active*rank*d,'mixedMomentAdditions':active*rank*d*d,
            'mixedStateReadWriteBytesMinimum':4*active*rank*d*d*8,'jacobianReadMACs':n*d*d,
            'finiteStateScanBytesMinimum':active*rank*(d*d+2*d+2)*8,
            'stateExponentials':2*active*rank,'outputAndTraceBytes':sum(a.nbytes for a in output.values()),
            'invalidQueries':{m:int(np.count_nonzero(~output[m+'_valid'])) for m in METHODS},
            'maximumHullViolations':{m:float(np.nanmax(output[m+'_hullViolation'])) for m in METHODS}})
    save(out/'prediction-seal.json',{'schema':'tce-query-transport-seal-v1','specSHA256':sha(spec_path),'transportAsset':spec['transportAsset'],
        'interfaces':spec['interfaces'],'manifestSHA256':spec['manifestSHA256'],'rows':rows,'ports':ports,'initializationSecondsAfterImports':init,
        'wallSecondsAfterImports':time.perf_counter()-start,'cpuSecondsAfterImports':time.process_time()-cpu,'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'learnedAtomRowsRead':prefix.store.rowsRead,'learnedAtomPayloadBytesRead':prefix.store.rowBytesRead,
        'frameworkCalls':0,'originalDenseProjections':0,'exactCurrentQueryAttentionEvaluations':0,'learnedInformationConsumed':True,
        'scope':'Source-defined nearest landmark plus actual-prefix value/key covariance. Fresh raw strings, no targetfit/fallback. Model-derived approximate attention, not strictzero-neural images.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('output');a=p.parse_args();serve(a.spec,a.output)
