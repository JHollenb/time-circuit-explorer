"""Source-only compilation and raw-text serving for every first-layer head.

Original matrices, contextual targets and reference heads are absent. This is
reduced model-derived attention, not a zero-neural contextual encoder.
"""
import argparse
import hashlib
from pathlib import Path
import resource
import time

from encoder_atoms import np,Prefix,read,save,sha,guard,floats,words,bf
from context_source_geometry import rotate
from context_landmarks import covariance_block,covariance_merge
from context_transport import stream,METHODS


def geometry(m):
    d=m['headDim'];heads=m['qWidth']//d;groups=m['kvWidth']//d
    if d%2 or m['qWidth']%d or m['kvWidth']%d or heads%groups:raise ValueError('Invalid grouped-head geometry')
    return d,heads,groups


def packet(full,m,head):
    d,heads,groups=geometry(m)
    if not 0<=head<heads:raise ValueError('Head outside query bank')
    group=head//(heads//groups)
    return {**{k:full[k] for k in ['ids','mask','positions']},
        'q':full['qpos'][:,head*d:(head+1)*d],
        'k':full['kpos'][:,group*d:(group+1)*d],
        'v':full['v'][:,group*d:(group+1)*d]}


def candidate_domain(ids,npos,pad,candidates,seed):
    selected=np.random.Generator(np.random.PCG64(seed)).choice(len(ids)*npos,size=candidates,replace=False)
    lexical=np.stack([ids[selected//npos],selected%npos],axis=1)
    padding=np.stack([np.full(npos,pad),np.arange(npos)],axis=1)
    identities,inverse=np.unique(np.concatenate([lexical,padding]),axis=0,return_inverse=True)
    weights=np.zeros(len(identities));np.add.at(weights,inverse[:candidates],.5/candidates);np.add.at(weights,inverse[candidates:],.5/npos)
    return identities,weights


def select_landmarks(x,root,weights,rank):
    points=x@root;center=weights@points;first=int(np.argmin(np.sum((points-center)**2,axis=1)))
    chosen=[];closest=np.full(len(points),np.inf)
    for _ in range(rank):
        index=first if not chosen else int(np.argmax(closest));chosen.append(index)
        closest=np.minimum(closest,np.sum((points-points[index])**2,axis=1));closest[chosen]=-np.inf
    return np.asarray(chosen,dtype=np.int64)


def fit(prefix,ids,rank=64,candidates=4096,seed=1909,batch=64):
    started=time.perf_counter();m=prefix.m;d,heads,groups=geometry(m);npos=len(prefix.rope[0]);h=m['hiddenSize'];qw=m['qWidth']
    if ids.ndim!=1 or not len(ids) or len(np.unique(ids))!=len(ids) or rank<1 or batch<1 or not 0<candidates<=len(ids)*npos:raise ValueError('Invalid source domain')
    # Gather complete source rows once; KV covariance is computed once per group.
    t=time.perf_counter();atoms=prefix.store.gather(ids);gather_s=time.perf_counter()-t
    roots=[];means=[];covariances=[];group_receipts=[]
    for group in range(groups):
        t=time.perf_counter();stats=None
        for offset in range(0,len(ids),batch):
            keys=rotate(atoms[offset:offset+batch,h+qw+group*d:h+qw+(group+1)*d],prefix.rope)
            stats=covariance_merge(stats,covariance_block(floats(keys).astype(np.float64)/d**.25))
        n,mean,M=stats;C=M/n;C=(C+C.T)*.5;eigenvalues,eigenvectors=np.linalg.eigh(C)
        tolerance=64*d*np.finfo(np.float64).eps*max(float(np.max(np.abs(eigenvalues))),np.finfo(np.float64).tiny)
        if eigenvalues.min() < -tolerance:raise ValueError('Source covariance inconsistent with PSD')
        root=eigenvectors*np.sqrt(np.maximum(eigenvalues,0.));roots.append(root);means.append(mean);covariances.append(C)
        group_receipts.append({'kvGroup':group,'keyRows':n,'seconds':time.perf_counter()-t,'sourceKeyMean':mean.tolist(),
            'rawCovarianceEigenvalues':eigenvalues.tolist(),'roundoffTolerance':tolerance,
            'metricReconstructionMaxAbs':float(np.max(np.abs(root@root.T-C)))})
        print('BANK_SOURCE_GROUP_FROZEN',group,flush=True)
    identities,weights=candidate_domain(ids,npos,m['tokenPolicy']['padID'],candidates,seed)
    if rank>len(identities):raise ValueError('Too few candidate identities')
    source=prefix.store.gather(identities[:,0]);pos=identities[:,1];cos,sin=[floats(a[pos]) for a in prefix.rope]
    landmarks=[];head_receipts=[];selected=[]
    for head in range(heads):
        t=time.perf_counter();group=head//(heads//groups);q=floats(source[:,h+head*d:h+(head+1)*d])
        turn=np.concatenate([-q[:,d//2:],q[:,:d//2]],axis=1);x=floats(words(bf(q*cos)+bf(turn*sin))).astype(np.float64)/d**.25
        chosen=select_landmarks(x,roots[group],weights,rank);landmarks.append(x[chosen]);selected.append(identities[chosen])
        head_receipts.append({'queryHead':head,'kvGroup':group,'landmarkIdentities':identities[chosen].tolist(),'seconds':time.perf_counter()-t})
    asset={'landmarks':np.stack(landmarks),'metricRoots':np.stack(roots),'keyCenters':np.stack(means),'covariances':np.stack(covariances),
        'landmarkIdentities':np.stack(selected),'candidateIdentities':identities,'candidateWeights':weights}
    receipt={'scope':'Only frozen token/position source operands; no raw prompt outputs or contextual targets enter compilation.',
        'queryHeads':heads,'kvGroups':groups,'headDim':d,'sampleTokens':len(ids),'positionsPerToken':npos,
        'rankPerQueryHead':rank,'lexicalCandidates':candidates,'uniqueCandidates':len(identities),'candidateSeed':seed,
        'queryMixtureWeights':[.5,.5],'sourceGatherSeconds':gather_s,'groups':group_receipts,'heads':head_receipts,
        'sourceCovarianceMACs':groups*len(ids)*npos*d*d,'positionedKeyVectorsFormed':groups*len(ids)*npos,
        'candidateQueryVectorsFormed':heads*len(identities),'assetPayloadBytes':sum(x.nbytes for x in asset.values()),
        'learnedAtomRowsRead':prefix.store.rowsRead,'learnedAtomPayloadBytesRead':prefix.store.rowBytesRead,
        'fitSeconds':time.perf_counter()-started,'frameworkCalls':0,'originalDenseProjections':0,'contextualAttentionCalls':0}
    return asset,receipt


def compile_asset(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False)
    if sha(spec['sampleIDs']['path'])!=spec['sampleIDs']['sha256']:raise ValueError('Source IDs differ')
    prefix=Prefix(spec['store'],spec['manifestSHA256']);ids=np.load(spec['sampleIDs']['path'],allow_pickle=False)
    asset,receipt=fit(prefix,ids,spec['landmarks'],spec['candidates'],spec['seed'],spec['batch'])
    np.savez(out/'bank-assets.npz',**asset);receipt.update(specSHA256=sha(spec_path),assetSHA256=sha(out/'bank-assets.npz'))
    save(out/'compile-receipt.json',receipt)


def head_asset(asset,m,head):
    _,heads,groups=geometry(m);group=head//(heads//groups)
    return {'landmarks':asset['landmarks'][head],'metricRoot':asset['metricRoots'][group],'keyCenter':asset['keyCenters'][group]}


def serve(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False);started=time.perf_counter();cpu=time.process_time()
    for name in ['asset','interfaces']:
        if sha(spec[name]['path'])!=spec[name]['sha256']:raise ValueError('Frozen serving input differs')
    with np.load(spec['asset']['path'],allow_pickle=False) as f:asset={k:f[k] for k in ['landmarks','metricRoots','keyCenters']}
    prefix=Prefix(spec['store'],spec['manifestSHA256']);m=prefix.m;d,heads,groups=geometry(m);rank=asset['landmarks'].shape[1]
    if asset['landmarks'].shape!=(heads,rank,d) or asset['metricRoots'].shape!=(groups,d,d) or asset['keyCenters'].shape!=(groups,d):raise ValueError('Bank source geometry differs')
    interfaces=read(spec['interfaces']['path']);rows=[];states=[]
    for item in spec['prompts']:
        start=time.perf_counter();full,text=prefix.forward(item['prompt']);frozen=interfaces['prompts'][item['id']]
        text_sha=hashlib.sha256(text.encode()).hexdigest()
        if text_sha!=frozen['renderedTextSHA256'] or full['ids'].tolist()!=frozen['ids'] or full['mask'].tolist()!=frozen['mask']:raise ValueError('Raw prompt interface changed')
        n=len(full['ids']);port=out/(item['id']+'-port.npz');np.savez(port,**{k:full[k] for k in ['ids','mask','positions','e','qpos','kpos','v']})
        outputs={method+'_raw':np.empty((n,heads*d),dtype=np.float64) for method in METHODS}
        outputs.update({method+'_rounded':np.empty((n,heads*d),dtype=np.uint16) for method in METHODS})
        outputs.update({method+'_valid':np.empty((n,heads),dtype=bool) for method in METHODS});head_times=[]
        trace_keys=['nearestIndex','sourceDistanceSquared','queryDisplacementSquared','correctionRms']+[method+'_hullViolation' for method in METHODS]
        traces={key:np.empty((n,heads),dtype=np.int64 if key=='nearestIndex' else np.float64) for key in trace_keys}
        for head in range(heads):
            t=time.perf_counter();prediction,state=stream(packet(full,m,head),head_asset(asset,m,head));head_times.append(time.perf_counter()-t)
            for method in METHODS:
                outputs[method+'_raw'][:,head*d:(head+1)*d]=prediction[method+'_raw']
                outputs[method+'_rounded'][:,head*d:(head+1)*d]=prediction[method+'_rounded']
                outputs[method+'_valid'][:,head]=prediction[method+'_rounded_valid']
            for key in trace_keys:traces[key][:,head]=prediction[key]
            path=out/(item['id']+'-state-'+str(head)+'.npz');np.savez(path,**{k:np.asarray(v) for k,v in state.items()})
            states.append({'id':item['id'],'queryHead':head,'kvGroup':head//(heads//groups),'path':str(path),'sha256':sha(path)})
            del prediction,state
        path=out/(item['id']+'-bank.npz');np.savez(path,**outputs,**traces)
        rows.append({'id':item['id'],'path':str(path),'sha256':sha(path),'port':{'path':str(port),'sha256':sha(port)},
            'activeTokens':int(full['mask'].sum()),'headSeconds':head_times,'rawTextBankSeconds':time.perf_counter()-start,
            'invalidHeadReads':{method:int(np.count_nonzero(~outputs[method+'_valid'])) for method in METHODS},
            'maximumHullViolations':{method:float(np.nanmax(traces[method+'_hullViolation'])) for method in METHODS}})
        print('COMPLETE_BANK_PREDICTED',item['id'],flush=True)
    core=rank*(d*d+2*d+2)*8+16
    save(out/'prediction-seal.json',{'schema':'tce-complete-bank-prediction-seal-v1','specSHA256':sha(spec_path),
        'asset':spec['asset'],'interfaces':spec['interfaces'],'manifestSHA256':spec['manifestSHA256'],'rows':rows,'states':states,
        'queryHeads':heads,'kvGroups':groups,'headDim':d,'rankPerHead':rank,'coreStateBytesPerHead':core,'logicalCoreStateBytes':heads*core,
        'sequentialHeadExecution':True,'retainedFinalStateBytes':sum(Path(z['path']).stat().st_size for z in states),
        'runtimeAssetBytes':sum(x.nbytes for x in asset.values()),'wallSecondsAfterImports':time.perf_counter()-started,
        'cpuSecondsAfterImports':time.process_time()-cpu,'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'learnedAtomRowsRead':prefix.store.rowsRead,'learnedAtomPayloadBytesRead':prefix.store.rowBytesRead,
        'frameworkCalls':0,'originalDenseProjections':0,'exactCurrentQueryAttentionEvaluations':0,'referenceHeadsSupplied':0,
        'scope':'Complete first-layer attention candidate banks from raw text. Reduced model-derived attention, not complete contextual encoder or zero-neural images.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['compile','serve']);p.add_argument('spec');p.add_argument('output');a=p.parse_args()
    (compile_asset if a.action=='compile' else serve)(a.spec,a.output)
