"""Source-frozen landmark readers with live positive prefix state.

This is reduced model-derived attention, not zero-neural image generation.
No new prompt's contextual output enters compilation or serving.
"""
import argparse
import hashlib
import resource
import time
from pathlib import Path

from encoder_atoms import np,Prefix,read,save,sha,guard,floats,words
from context_source_geometry import rotate
from context_statistics import lexical_head,copy_state

METHODS=['landmark64','nearest64','kernel64','uniform']


def covariance_block(values):
    values=np.asarray(values,dtype=np.float64).reshape(-1,values.shape[-1]);mean=values.mean(0);centered=values-mean
    return len(values),mean,centered.T@centered


def covariance_merge(left,right):
    if left is None:return right
    n,a,M=left;m,b,N=right;delta=b-a
    return n+m,a+delta*(m/(n+m)),M+N+np.outer(delta,delta)*(n*m/(n+m))


def fit(prefix,ids,rank=64,candidates=4096,seed=1909,batch=64):
    start=time.perf_counter();m=prefix.m;d=m['headDim'];h=m['hiddenSize'];qw=m['qWidth'];npos=len(prefix.rope[0]);stats=None
    if ids.ndim!=1 or len(ids)==0 or len(np.unique(ids))!=len(ids) or rank<1 or candidates>len(ids)*npos:raise ValueError('Invalid source domain')
    for offset in range(0,len(ids),batch):
        atoms=prefix.store.gather(ids[offset:offset+batch]);keys=rotate(atoms[:,h+qw:h+qw+d],prefix.rope)
        stats=covariance_merge(stats,covariance_block(floats(keys).astype(np.float64)/d**.25))
    n,mean,M=stats;C=M/n;C=(C+C.T)*.5;eigenvalues,eigenvectors=np.linalg.eigh(C)
    # Covariance is PSD in real arithmetic. Keep the raw spectrum and repair only solver roundoff.
    tolerance=64*d*np.finfo(np.float64).eps*max(float(np.max(np.abs(eigenvalues))),np.finfo(np.float64).tiny)
    if eigenvalues.min() < -tolerance:raise ValueError('Source covariance is numerically inconsistent with PSD')
    root=eigenvectors*np.sqrt(np.maximum(eigenvalues,0.))
    selected=np.random.Generator(np.random.PCG64(seed)).choice(len(ids)*npos,size=candidates,replace=False)
    lexical=np.stack([ids[selected//npos],selected%npos],axis=1)
    padding=np.stack([np.full(npos,m['tokenPolicy']['padID']),np.arange(npos)],axis=1)
    identities,inverse=np.unique(np.concatenate([lexical,padding]),axis=0,return_inverse=True)
    weights=np.zeros(len(identities));np.add.at(weights,inverse[:candidates],.5/candidates);np.add.at(weights,inverse[candidates:],.5/npos)
    if rank>len(identities):raise ValueError('Too few source candidate identities')
    atoms=prefix.store.gather(identities[:,0]);q=floats(atoms[:,h:h+d]);pos=identities[:,1]
    cos,sin=[floats(a[pos]) for a in prefix.rope];turn=np.concatenate([-q[:,d//2:],q[:,:d//2]],axis=1)
    x=floats(words(floats(words(q*cos))+floats(words(turn*sin)))).astype(np.float64)/d**.25
    points=x@root;center=weights@points;first=int(np.argmin(np.sum((points-center)**2,axis=1)))
    chosen=[];closest=np.full(len(points),np.inf)
    for _ in range(rank):
        index=first if not chosen else int(np.argmax(closest));chosen.append(index)
        closest=np.minimum(closest,np.sum((points-points[index])**2,axis=1));closest[chosen]=-np.inf
    landmarks=x[chosen];lp=points[chosen];dist=np.sum((lp[:,None,:]-lp[None,:,:])**2,axis=2);np.fill_diagonal(dist,np.inf)
    nearest=dist.min(1);positive=nearest[np.isfinite(nearest)&(nearest>0)]
    degenerate=len(positive)==0;tau2=1. if degenerate else float(np.median(positive))
    asset={'landmarks':landmarks,'metricRoot':root,'covariance':C,'bandwidthSquared':np.asarray(tau2),
        'landmarkIdentities':identities[chosen],'candidateIdentities':identities,'candidateWeights':weights}
    receipt={'scope':'Source-only token/position covariance and query landmarks. No prompt string, contextual output or target-selected parameter enters fit.',
        'sampleTokens':len(ids),'positionsPerToken':npos,'keyPoolRows':n,'candidateSeed':seed,'lexicalCandidates':candidates,
        'paddingCandidates':npos,'uniqueCandidates':len(identities),'duplicateCandidateIdentities':candidates+npos-len(identities),
        'queryMixtureWeights':[.5,.5],'landmarks':rank,'landmarkIdentities':identities[chosen].tolist(),
        'sourceKeyMean':mean.tolist(),'rawCovarianceEigenvalues':eigenvalues.tolist(),'psdRoundoffTolerance':tolerance,
        'negativeEigenvaluesRepaired':int(np.count_nonzero(eigenvalues<0)),
        'metricReconstructionMaxAbs':float(np.max(np.abs(root@root.T-C))),
        'sourceNearestLandmarkSquaredDistances':nearest.tolist(),'bandwidthSquared':tau2,'degenerateBandwidth':degenerate,
        'selection':'Lexicographic identities; nearest weighted source-query mean, then deterministic farthest-first; selected identities excluded from later selection.',
        'routing':'All-landmark RBF exp(-sourceScoreVariance/(2*tau2)); tau2 median positive nearest-other-landmark squared distance, or1 for degenerate source.',
        'metric':'x=q/d^.25,y=k/d^.25; distance=(x-l)^T Cov_source(y)(x-l). Source score-displacement variance, not fresh-prefix read-error bound.',
        'covariancePolicy':'FP64 block-centered full M2 matrices with Chan merge, population divisor; raw spectrum retained.',
        'learnedAtomRowsRead':prefix.store.rowsRead,'learnedAtomPayloadBytesRead':prefix.store.rowBytesRead,
        'positionedKeyVectorsFormed':n,'candidateQueryVectorsFormed':len(x),'sourceCovarianceMACs':n*d*d,
        'assetPayloadBytes':sum(z.nbytes for z in asset.values()),'fitSeconds':time.perf_counter()-start,
        'frameworkCalls':0,'originalDenseProjections':0,'contextualAttentionCalls':0,'learnedInformationConsumed':True}
    return asset,receipt


def routes(x,asset):
    points=x@asset['metricRoot'];landmarks=asset['landmarks']@asset['metricRoot']
    distances=np.sum((points[:,None,:]-landmarks[None,:,:])**2,axis=2)
    logits=-distances/(2*float(asset['bandwidthSquared']));top=logits.max(1,keepdims=True)
    logw=logits-top-np.log(np.exp(logits-top).sum(1,keepdims=True))
    return np.exp(logw),logw,distances


def new_state(d,rank):
    return {'maximum':np.full(rank,-np.inf),'z':np.zeros(rank),'M':np.zeros((rank,d)),
        'sum':np.zeros(d),'lo':np.full(d,np.inf),'hi':np.full(d,-np.inf),'count':0,'steps':0}


def stream(packet,asset,state=None,checkpoints=()):
    q,k,v=[floats(packet[n]).astype(np.float64) for n in ['q','k','v']];n,d=q.shape;ell=asset['landmarks'];rank=len(ell)
    if n==0 or q.shape!=k.shape or q.shape!=v.shape or ell.shape!=(rank,d) or asset['metricRoot'].shape!=(d,d):raise ValueError('Landmark shapes differ')
    if not all(np.isfinite(z).all() for z in [q,k,v,ell,asset['metricRoot']]) or not np.isfinite(asset['bandwidthSquared']) or asset['bandwidthSquared']<=0:raise ValueError('Invalid finite reader inputs')
    state=new_state(d,rank) if state is None else copy_state(state)
    asset_id=hashlib.sha256(b''.join(np.asarray(asset[key],dtype='<f8').tobytes() for key in ['landmarks','metricRoot','bandwidthSquared'])).hexdigest()
    if state.get('assetIdentity',asset_id)!=asset_id:raise ValueError('State reader asset differs')
    state['assetIdentity']=asset_id
    if not np.array_equal(packet['positions'],np.arange(state['steps'],state['steps']+n)):raise ValueError('State clock differs')
    if packet['mask'].shape!=(n,) or np.any((packet['mask']!=0)&(packet['mask']!=1)):raise ValueError('Invalid binary key mask')
    if state['z'].shape!=(rank,) or state['M'].shape!=(rank,d) or state['maximum'].shape!=(rank,):raise ValueError('State shape differs')
    w,logw,distances=routes(q/d**.25,asset);allowed=packet['mask'].astype(bool);scores=np.zeros((n,rank));scores[allowed]=(k[allowed]/d**.25)@ell.T
    heads={method:[] for method in METHODS};trace=[];snapshots=[];indices=[]
    for i in range(n):
        if allowed[i]:
            top=np.maximum(state['maximum'],scores[i]);rescale=np.exp(state['maximum']-top);added=np.exp(scores[i]-top)
            state['z']=state['z']*rescale+added;state['M']=state['M']*rescale[:,None]+added[:,None]*v[i];state['maximum']=top
            state['count']+=1;state['sum']+=v[i];state['lo']=np.minimum(state['lo'],v[i]);state['hi']=np.maximum(state['hi'],v[i])
        state['steps']+=1;nearest=int(np.argmin(distances[i]));empty=not state['count']
        if empty:reads=[np.zeros(d) for _ in METHODS];kernel_mass=np.zeros(rank);log_partition=np.zeros(rank)
        else:
            if np.any(state['z']<=0) or not all(np.isfinite(state[z]).all() for z in ['z','M','maximum']):raise ValueError('Invalid positive landmark state')
            readers=state['M']/state['z'][:,None];log_partition=state['maximum']+np.log(state['z'])
            logmass=logw[i]+log_partition;kernel_mass=np.exp(logmass-logmass.max());kernel_mass/=kernel_mass.sum()
            reads=[w[i]@readers,readers[nearest],kernel_mass@readers,state['sum']/state['count']]
        record={'empty':empty,'routeEntropy':float(-np.sum(w[i]*logw[i])),'routeMaxMass':float(w[i].max()),
            'nearestDistanceSquared':float(distances[i,nearest]),'nearestIndex':nearest,
            'kernelRouteMaxMass':float(kernel_mass.max()),'partitionLogRange':float(np.ptp(log_partition))}
        for method,h in zip(METHODS,reads):
            if not np.isfinite(h).all():raise ValueError('Nonfinite optional landmark read')
            heads[method].append(h);record[method+'_hullViolation']=0. if empty else max(0.,float(np.max(h-state['hi'])),float(np.max(state['lo']-h)))
        trace.append(record)
        if int(packet['positions'][i]) in checkpoints:snapshots.append(copy_state(state));indices.append(int(packet['positions'][i]))
    output={}
    for method in METHODS:output[method+'_raw']=np.stack(heads[method]);output[method+'_rounded']=words(output[method+'_raw'])
    for key in trace[0]:output[key]=np.asarray([row[key] for row in trace])
    output['checkpoint_positions']=np.asarray(indices,dtype=np.int64)
    for key,value in state.items():
        output['state_'+key]=np.asarray(value)
        if snapshots:output['checkpoint_'+key]=np.asarray([s[key] for s in snapshots])
    return output,state


def serve(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False)
    for key in ['landmarkAsset','interfaces']:
        if sha(spec[key]['path'])!=spec[key]['sha256']:raise ValueError('Frozen reader asset changed')
    with np.load(spec['landmarkAsset']['path'],allow_pickle=False) as f:asset={k:f[k] for k in ['landmarks','metricRoot','bandwidthSquared']}
    interfaces=read(spec['interfaces']['path']);start=time.perf_counter();cpu=time.process_time();prefix=Prefix(spec['store'],spec['manifestSHA256']);init=time.perf_counter()-start
    rows=[];ports=[]
    for item in spec['prompts']:
        t=time.perf_counter();packet,text_sha=lexical_head(prefix,item['prompt'],0);first=time.perf_counter()-t
        interface=interfaces['prompts'][item['id']]
        if text_sha!=interface['renderedTextSHA256'] or packet['ids'].tolist()!=interface['ids'] or packet['mask'].tolist()!=interface['mask']:raise ValueError('Frozen prompt interface differs')
        path=out/(item['id']+'-port.npz');np.savez(path,**packet);n,d=packet['q'].shape;active=int(packet['mask'].sum());rank=len(asset['landmarks'])
        ports.append({'id':item['id'],'path':str(path),'sha256':sha(path),'firstPrefixSeconds':first,'activeTokens':active,'renderedTextSHA256':text_sha})
        points=sorted({0,active-1,n-1});t=time.perf_counter();prediction,_=stream(packet,asset,checkpoints=points);kernel_s=time.perf_counter()-t
        path=out/(item['id']+'-readers.npz');np.savez(path,**prediction);samples=[]
        for _ in range(spec['repeats']):
            t=time.perf_counter();again,_=lexical_head(prefix,item['prompt'],0);stream(again,asset,checkpoints=points);samples.append(time.perf_counter()-t)
        rows.append({'id':item['id'],'path':str(path),'sha256':sha(path),'methods':METHODS,'allFourReadersSeconds':kernel_s,
            'warmRawTextAllFourReadersSeconds':samples,'warmMedianSeconds':float(np.median(samples)),
            'timingScope':'Joint primary plus nearest/kernel/uniform controls, routing, state, diagnostics/checkpoint copies; warm includes raw prefix. Not standalone primary latency.',
            'coreLandmarkStateBytes':rank*(d+2)*8+16,'uniformControlStateBytes':d*8,'rangeDiagnosticBytes':2*d*8,
            'residentReaderAssetBytes':sum(a.nbytes for a in asset.values()),'preformedRouteWorkspaceBytes':3*n*rank*8+n*d*8+rank*d*8+n*rank*d*8,
            'landmarkScoreMACs':active*rank*d,'routingTransformMACs':(n+rank)*d*d,'routingDistanceSquaredProducts':n*rank*d,
            'primaryReaderMACs':n*rank*d,'kernelControlReaderMACs':n*rank*d,'stateValueProducts':active*rank*d,
            'stateRescaleProducts':active*rank*(d+1),'stateAdditions':active*rank*(d+1),
            'stateExponentials':2*active*rank,'routingExponentials':2*n*rank,'kernelMixExponentials':n*rank,
            'outputAndTraceBytes':sum(a.nbytes for a in prediction.values()),
            'maximumHullViolations':{m:float(prediction[m+'_hullViolation'].max()) for m in METHODS}})
    save(out/'prediction-seal.json',{'schema':'tce-source-landmark-seal-v1','specSHA256':sha(spec_path),'landmarkAsset':spec['landmarkAsset'],
        'interfaces':spec['interfaces'],'manifestSHA256':spec['manifestSHA256'],'rows':rows,'ports':ports,
        'initializationSecondsAfterImports':init,'wallSecondsAfterImports':time.perf_counter()-start,'cpuSecondsAfterImports':time.process_time()-cpu,
        'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'learnedAtomRowsRead':prefix.store.rowsRead,
        'learnedAtomPayloadBytesRead':prefix.store.rowBytesRead,'frameworkCalls':0,'originalDenseProjections':0,
        'exactCurrentQueryAttentionEvaluations':0,'learnedInformationConsumed':True,
        'scope':'New raw strings, source-only64landmark exact prefix statistics and smooth routing. Reduced model-derived attention, not strictzero-neural images. No per-prompt fit or target fallback.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('output');a=p.parse_args();serve(a.spec,a.output)
