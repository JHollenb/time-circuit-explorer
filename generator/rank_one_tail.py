"""Source-only rank-one compilation and an actual causal mixed-moment state."""
import argparse
import copy
import hashlib
from pathlib import Path
import resource
import time

from encoder_atoms import np,AtomStore,checked,read,save,sha,guard,floats,words,bf
from attention_bank import geometry,packet,head_asset
from context_transport import nearest
from selector_objective import select
from sparse_tilt_study import logsum

POLICY='rank1-fullcentroid-moments-fp64-envelope-v1'
COUNTERS=['admissionAnchorMACs','zProjectionProducts','stateCellsUpdated','envelopeCellsUpdated','selectorScoreProducts','selectedCoordinateProducts','momentContractionMACs','envelopeContractionMACs','downdateSelectedOccurrences','fallbackCalls','fallbackKeyVisits','fallbackCoreMACs','queries']


def source_basis(C,D):
    d=len(C);C=(C+C.T)*.5;lam,Q=np.linalg.eigh(C);tol=64*d*np.finfo(float).eps*max(float(np.max(np.abs(lam))),np.finfo(float).tiny)
    if lam.min() < -tol:raise ValueError('Source covariance inconsistent with PSD')
    resolved=lam>tol;root=(Q*np.sqrt(np.where(resolved,lam,0.)))@Q.T;inverse=(Q*np.where(resolved,1/np.sqrt(np.maximum(lam,np.finfo(float).tiny)),0.))@Q.T
    M=root@D@root;M=(M+M.T)*.5;eig,V=np.linalg.eigh(M);v=V[:,-1];pivot=int(np.argmax(np.abs(v)));v=v if v[pivot]>=0 else -v
    a=root@v;b=inverse@v
    return a,b,{'rawKeyEigenvalues':lam.tolist(),'resolvedRank':int(resolved.sum()),'roundoffTolerance':tol,'discardedPositiveVariance':float(lam[(lam>0)&~resolved].sum()),
        'scoreEigenvalues':eig.tolist(),'leadingScoreEigenvalueGap':float(eig[-1]-eig[-2]) if d>1 else None,'sourceProductScoreEnergy':float(np.trace(C@D)),'resolvedCapturedScoreEnergy':float(eig[-1]),'dualPairDot':float(a@b),'signPivot':pivot,
        'policy':'Symmetric eigensystem; eigenvalues above source FP64 roundoff envelope resolved, Moore-Penrose inverse on that support. Leading vector largest-absolute coordinate positive. Uncentered D, no scored-prompt fit.'}


def compile_asset(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False);start=time.perf_counter()
    if sha(spec['asset']['path'])!=spec['asset']['sha256']:raise ValueError('Source geometry differs')
    with np.load(spec['asset']['path'],allow_pickle=False) as f:parent={k:f[k] for k in f.files}
    store=AtomStore(spec['store'],spec['manifestSHA256']);m=store.manifest;d,heads,groups=geometry(m);ids=parent['candidateIdentities'];weight=parent['candidateWeights'];rope=np.load(checked(spec['store'],m['rope']),allow_pickle=False)
    if ids.shape[1]!=2 or weight.shape!=(len(ids),) or np.any(weight<0) or not np.isfinite(weight).all():raise ValueError('Invalid source candidate measure')
    tick=time.perf_counter();atoms=store.gather(ids[:,0]);gather=time.perf_counter()-tick;cos,sin=[floats(a[ids[:,1]]) for a in rope];D=np.zeros((groups,d,d));means=np.zeros((groups,d));head_rows=[]
    for h in range(heads):
        g=h//(heads//groups);q=floats(atoms[:,m['hiddenSize']+h*d:m['hiddenSize']+(h+1)*d]);turn=np.concatenate([-q[:,d//2:],q[:,:d//2]],axis=1);x=floats(words(bf(q*cos)+bf(turn*sin))).astype(float)/d**.25;a=head_asset(parent,m,h);route=nearest(x,a)[0];delta=x-a['landmarks'][route]
        D[g]+=(delta.T@(weight[:,None]*delta))/(heads//groups);means[g]+=(weight@delta)/(heads//groups);head_rows.append({'queryHead':h,'kvGroup':g,'weightedDisplacementSecondMomentTrace':float(np.sum(weight*np.sum(delta*delta,axis=1)))})
    aa=[];bb=[];rows=[]
    for g in range(groups):
        a,b,r=source_basis(parent['covariances'][g],D[g]);aa.append(a);bb.append(b);rows.append({'kvGroup':g,**r,'weightedDisplacementMean':means[g].tolist()})
    np.savez(out/'rank-one-basis.npz',a=np.stack(aa),b=np.stack(bb));np.savez(out/'basis-diagnostics.npz',sourceKeyCovariances=parent['covariances'],sourceDisplacementSecondMoments=D,sourceDisplacementMeans=means)
    sources=[{'path':str(Path(spec['store'])/'manifest.json'),'sha256':spec['manifestSHA256']}]+[{'path':str(Path(spec['store'])/m[k]['path']),'sha256':m[k]['sha256']} for k in ['coverage','rowHashes','rope']]
    sources += [{'path':str(Path(spec['store'])/m['shards'][j]['path']),'sha256':m['shards'][j]['sha256']} for j in sorted(store.shards)]
    save(out/'compile-receipt.json',{'schema':'tce-rank-one-tail-compile-v1','specSHA256':sha(spec_path),'assetSHA256':sha(out/'rank-one-basis.npz'),'diagnosticsSHA256':sha(out/'basis-diagnostics.npz'),'sourceFiles':sources,
        'groups':rows,'heads':head_rows,'sourceCandidateCount':len(ids),'sourceCandidateWeightSum':float(weight.sum()),'sourceGatherSeconds':gather,'compileSeconds':time.perf_counter()-start,
        'learnedAtomRowsRead':store.rowsRead,'learnedAtomPayloadBytesRead':store.rowBytesRead,'basisPayloadBytes':2*groups*d*8,'frameworkCalls':0,'originalDenseProjections':0,
        'scope':'Only immutable source query candidates, frozen source routes and key covariance. D is uncentered and shared across4queryheads. No scored prompt, original weights or contextual targets.'})


def direct(s,U,V,z,selected,delta,beta):
    a=s[selected]+U[selected]@delta;logzs=logsum(a);hs=np.exp(a-logzs)@V[selected];tail=np.ones(len(s),dtype=bool);tail[selected]=False
    if not tail.any():return hs
    st=s[tail];logz0=logsum(st);anchor=np.exp(st-logz0);mu=anchor@U[tail];mean=float(anchor@z[tail]);t=beta*(z[tail]-mean)
    with np.errstate(divide='ignore'):logr=np.logaddexp(2*np.log(np.abs(t+1)),0.)-np.log(2.)
    lp=logsum(st+logr);ht=np.exp(st+logr-lp)@V[tail];lzt=lp+float(mu@delta);w=float(np.exp(lzt-logsum(np.array([logzs,lzt]))));return (1-w)*hs+w*ht


def new_state(heads,rank,d,identity):
    state={'moment':np.zeros((heads,rank,3,d+1)),'absoluteMoment':np.zeros((heads,rank,3,d+1)),'U':np.zeros((heads,rank,d)),'absoluteU':np.zeros((heads,rank,d)),
        'maximum':np.full((heads,rank),-np.inf),'maxAbsScore':np.zeros((heads,rank)),'keys':np.zeros((8,d)),'values':np.zeros((8,d)),'z':np.zeros(8),'scores':np.zeros((heads,rank,8)),
        'positions':np.full(8,-1,dtype=np.int64),'count':0,'steps':0,'assetIdentity':identity}
    state.update({key:0 for key in COUNTERS});return state


def advance(state,u,v,position,ell,b):
    count=state['count'];heads,rank,d=ell.shape
    if count==len(state['keys']):
        for key,axis in [('keys',0),('values',0),('z',0),('positions',0),('scores',2)]:
            shape=list(state[key].shape);shape[axis]*=2;array=np.zeros(shape,dtype=state[key].dtype);slots=[slice(None)]*array.ndim;slots[axis]=slice(0,count);array[tuple(slots)]=state[key];state[key]=array
    s=(ell.reshape(-1,d)@u).reshape(heads,rank);top=np.maximum(state['maximum'],s);old=np.exp(state['maximum']-top);weight=np.exp(s-top);z=float(b@u);phi=np.array([1.,z,z*z]);value=np.r_[1.,v];feature=phi[:,None]*value[None,:]
    state['moment']=state['moment']*old[:,:,None,None]+weight[:,:,None,None]*feature
    state['absoluteMoment']=state['absoluteMoment']*old[:,:,None,None]+weight[:,:,None,None]*np.abs(feature)
    state['U']=state['U']*old[:,:,None]+weight[:,:,None]*u;state['absoluteU']=state['absoluteU']*old[:,:,None]+weight[:,:,None]*np.abs(u)
    state['maximum']=top;state['maxAbsScore']=np.maximum(state['maxAbsScore'],np.abs(s));state['keys'][count]=u;state['values'][count]=v;state['z'][count]=z;state['scores'][:,:,count]=s;state['positions'][count]=position;state['count']+=1
    state['admissionAnchorMACs']+=heads*rank*d;state['zProjectionProducts']+=d;state['stateCellsUpdated']+=heads*rank*(3*(d+1)+d);state['envelopeCellsUpdated']+=heads*rank*(3*(d+1)+d)


def gamma(n):
    z=n*np.finfo(float).eps;return z/(1-z)


def moment_read(state,h,r,selected,delta,beta):
    count=state['count'];U=state['keys'][:count];V=state['values'][:count];z=state['z'][:count];s=state['scores'][h,r,:count];d=U.shape[1]
    a=s[selected]+U[selected]@delta;logzs=logsum(a);hs=np.exp(a-logzs)@V[selected];state['selectedCoordinateProducts']+=2*len(selected)*d
    record={'selectedOnly':count==len(selected),'fallbackReason':0,'tailMass':np.nan,'tailRadius':np.nan,'polyMass':np.nan,'polyRadius':np.nan,'complementCondition':np.nan,'centroidRadiusRms':np.nan,'centroidLogRadius':np.nan}
    if record['selectedOnly']:return hs,record
    w=np.exp(s[selected]-state['maximum'][h,r]);phi=np.stack([np.ones(len(selected)),z[selected],z[selected]**2],axis=1);feature=phi[:,:,None]*np.c_[np.ones(len(selected)),V[selected]][:,None,:]
    sub=np.sum(w[:,None,None]*feature,axis=0);abs_sub=np.sum(w[:,None,None]*np.abs(feature),axis=0);tail=state['moment'][h,r]-sub;tu=state['U'][h,r]-w@U[selected]
    # Declared engineering envelope: FP64 sum/product model and <=2ulp exp,
    # including score-range sensitivity. Diagnostic safety policy, not a BLAS proof.
    factor=gamma(12*count+16)*(1+state['maxAbsScore'][h,r]);radius=factor*(state['absoluteMoment'][h,r]+abs_sub)+gamma(2)*(np.abs(state['moment'][h,r])+np.abs(sub));ru=factor*(state['absoluteU'][h,r]+w@np.abs(U[selected]))
    mass=float(tail[0,0]);er=float(radius[0,0]);record.update(tailMass=mass,tailRadius=er,complementCondition=float((state['moment'][h,r,0,0]+sub[0,0])/mass) if mass>0 else np.inf)
    state['downdateSelectedOccurrences']+=len(selected)
    reason=2 if not np.isfinite(mass) or mass<=er else 0
    if not reason:
        mean=float(tail[1,0]/mass);mr=(radius[1,0]+abs(mean)*er)/(mass-er)+gamma(2)*abs(mean);b0=-beta*mean;coeff=np.array([1+b0+.5*b0*b0,(1+b0)*beta,.5*beta*beta]);br=abs(beta)*mr
        cr=np.array([br*(1+abs(b0)+.5*br)+gamma(6)*(1+abs(b0)+.5*b0*b0),abs(beta)*br+gamma(3)*abs(coeff[1]),gamma(2)*abs(coeff[2])]);poly=coeff@tail;pr=np.abs(coeff)@radius+cr@(np.abs(tail)+radius)+gamma(3)*(np.abs(coeff)@np.abs(tail));den=float(poly[0]);de=float(pr[0]);mu=tu/mass;mur=(ru+np.abs(mu)*er)/(mass-er)+gamma(2)*np.abs(mu)
        record.update(polyMass=den,polyRadius=de,centroidRadiusRms=float(np.sqrt(np.mean(mur*mur))),centroidLogRadius=float(np.abs(delta)@mur+gamma(d)*(np.abs(delta)@np.abs(mu))))
        state['momentContractionMACs']+=3*(d+1);state['envelopeContractionMACs']+=9*(d+1)
        if not np.isfinite(poly).all() or not np.isfinite(pr).all() or den<=de:reason=3
        else:
            lzt=float(state['maximum'][h,r]+delta@mu+np.log(den));mix=float(np.exp(lzt-logsum(np.array([logzs,lzt]))));raw=(1-mix)*hs+mix*(poly[1:]/den)
            if np.isfinite(raw).all():return raw,record
            reason=4
    record['fallbackReason']=reason;state['fallbackCalls']+=1;state['fallbackKeyVisits']+=count;state['fallbackCoreMACs']+=2*count*d+d
    return direct(s,U,V,z,selected,delta,beta),record


def stream(packets,asset,a,b,state=None):
    heads,rank,d=asset['landmarks'].shape;n=len(packets[0]['positions']);tag=hashlib.sha256((POLICY+repr((heads,rank,d))).encode()+b''.join(np.asarray(x,dtype='<f8').tobytes() for x in [asset['landmarks'],asset['metricRoot'],asset['keyCenter'],a,b])).hexdigest()
    state=new_state(heads,rank,d,tag) if state is None else copy.deepcopy(state)
    if state['assetIdentity']!=tag:raise ValueError('State geometry/basis identity differs')
    if state['moment'].shape!=(heads,rank,3,d+1) or state['U'].shape!=(heads,rank,d) or state['scores'].shape!=(heads,rank,len(state['keys'])) or not 0<=state['count']<=len(state['keys']) or state['steps']<state['count']:raise ValueError('Typed state dimensions/count differ')
    active=state['count']
    if active and (np.any(np.diff(state['positions'][:active])<=0) or state['positions'][active-1]>=state['steps']):raise ValueError('State causal history differs')
    for p in packets:
        if not np.array_equal(p['positions'],np.arange(state['steps'],state['steps']+n)) or not np.array_equal(p['mask'],packets[0]['mask']):raise ValueError('State clock/mask differs')
        if not np.array_equal(p['k'],packets[0]['k']) or not np.array_equal(p['v'],packets[0]['v']):raise ValueError('Shared KV group differs')
    mask=packets[0]['mask'];q=np.stack([floats(p['q']).astype(float)/d**.25 for p in packets]);k=floats(packets[0]['k']).astype(float)/d**.25-asset['keyCenter'];v=floats(packets[0]['v']).astype(float)
    if len(packets)!=heads or a.shape!=(d,) or b.shape!=(d,) or np.any((mask!=0)&(mask!=1)) or not all(np.isfinite(x).all() for x in [q,k,v,a,b,asset['landmarks'],asset['metricRoot'],asset['keyCenter']]):raise ValueError('Invalid finite packet')
    route_landmarks=asset['landmarks']@asset['metricRoot']
    raw=np.zeros((n,heads*d));support=np.full((n,heads,4),-1,dtype=np.int64);routes=np.empty((n,heads),dtype=np.int64);fields={'selectedOnly':np.ones((n,heads)),'fallbackReason':np.zeros((n,heads))}
    for i in range(n):
        if mask[i]:advance(state,k[i],v[i],int(packets[0]['positions'][i]),asset['landmarks'],b)
        for h in range(heads):
            route=int(np.argmin(np.sum((q[h,i]@asset['metricRoot']-route_landmarks[h])**2,axis=1)));routes[i,h]=route;state['queries']+=1;count=state['count']
            if not count:continue
            chosen=select(state['keys'][:count]@q[h,i],state['positions'][:count],4);state['selectorScoreProducts']+=count*d;support[i,h,:len(chosen)]=state['positions'][chosen];delta=q[h,i]-asset['landmarks'][h,route];beta=float(a@delta)
            raw[i,h*d:(h+1)*d],record=moment_read(state,h,route,chosen,delta,beta)
            for key,value in record.items():
                if key not in fields:fields[key]=np.full((n,heads),np.nan)
                fields[key][i,h]=value
        state['steps']+=1
    return {'raw':raw,'rounded':words(raw),'supportPositions':support,'nearestIndex':routes,**fields},state


def serve(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False);start=time.perf_counter();cpu=time.process_time()
    for key in ['asset','basis']:
        if sha(spec[key]['path'])!=spec[key]['sha256']:raise ValueError('Source serving asset differs')
    with np.load(spec['asset']['path'],allow_pickle=False) as f:asset={k:f[k] for k in ['landmarks','metricRoots','keyCenters']}
    with np.load(spec['basis']['path'],allow_pickle=False) as f:basis={k:f[k] for k in f.files}
    m=spec['geometry'];d,heads,groups=geometry(m);hpg=heads//groups;rows=[];states=[];counts=[]
    for item in spec['ports']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Saved operands differ')
        with np.load(item['path'],allow_pickle=False) as f:full={k:f[k] for k in f.files}
        n=len(full['ids']);raw=np.empty((n,heads*d));support=np.empty((n,heads,4),dtype=np.int64);routes=np.empty((n,heads),dtype=np.int64);fields={};group_rows=[];tick=time.perf_counter()
        for g in range(groups):
            first=g*hpg;sub={'landmarks':asset['landmarks'][first:first+hpg],'metricRoot':asset['metricRoots'][g],'keyCenter':asset['keyCenters'][g]};t=time.perf_counter();result,state=stream([packet(full,m,h) for h in range(first,first+hpg)],sub,basis['a'][g],basis['b'][g]);seconds=time.perf_counter()-t
            raw[:,first*d:(first+hpg)*d]=result['raw'];support[:,first:first+hpg]=result['supportPositions'];routes[:,first:first+hpg]=result['nearestIndex']
            for key,value in result.items():
                if key in ['raw','rounded','supportPositions','nearestIndex']:continue
                if key not in fields:fields[key]=np.full((n,heads),np.nan)
                fields[key][:,first:first+hpg]=value
            path=out/(item['id']+'-group'+str(g)+'-state.npz');np.savez(path,**{k:np.asarray(v) for k,v in state.items()});states.append({'id':item['id'],'kvGroup':g,'path':str(path),'sha256':sha(path)})
            group_rows.append({'kvGroup':g,'seconds':seconds,'admittedKeys':state['count'],'steps':state['steps'],**{key:state[key] for key in COUNTERS},
                'momentPayloadBytes':state['moment'].nbytes+state['U'].nbytes,'envelopePayloadBytes':state['absoluteMoment'].nbytes+state['absoluteU'].nbytes,
                'allocatedIndexPayloadBytes':sum(state[k].nbytes for k in ['keys','values','z','scores','positions']),
                'usedIndexPayloadBytes':state['count']*((2*d+2)*8+hpg*len(sub['landmarks'][0])*8)})
        path=out/(item['id']+'-stream-rank1.npz');np.savez(path,raw=raw,rounded=words(raw),supportPositions=support,nearestIndex=routes,**fields);rows.append({'id':item['id'],'method':'stream','path':str(path),'sha256':sha(path)});counts.append({'id':item['id'],'groups':group_rows,'withSerializationSeconds':time.perf_counter()-tick});print('RANK_ONE_STREAM_SEALED',item['id'],flush=True)
    save(out/'prediction-seal.json',{'schema':'tce-rank-one-tail-seal-v1','specSHA256':sha(spec_path),'asset':spec['asset'],'basis':spec['basis'],'ports':spec['ports'],'rows':rows,'states':states,'counts':counts,
        'runtimeGeometryBytes':sum(a.nbytes for a in asset.values()),'basisPayloadBytes':sum(a.nbytes for a in basis.values()),'wallSecondsAfterImports':time.perf_counter()-start,'cpuSecondsAfterImports':time.process_time()-cpu,'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'frameworkCalls':0,'originalDenseProjections':0,'exactHeadReferences':0,'scope':'Own causal mixed-moment state and full current-query KV selection. Explicit rank1 direct-tail fallbacks only on normalization safety envelope; no reference output access. Full centroid retained. Not compact encoder/zero-neural image.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['compile','serve']);p.add_argument('spec');p.add_argument('output');a=p.parse_args();(compile_asset if a.action=='compile' else serve)(a.spec,a.output)
