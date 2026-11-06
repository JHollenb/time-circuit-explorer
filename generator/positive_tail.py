"""One positive directional quadratic tail read; no exact-tail diagnostic in serving."""
import argparse
from pathlib import Path
import resource
import time

from encoder_atoms import np,read,save,sha,guard,floats,words
from attention_bank import geometry,packet,head_asset
from context_transport import nearest
from sparse_tilt_study import logsum
from tail_factorial import ordered_support


def positive_read(s,U,V,selected,delta):
    if not len(selected):raise ValueError('Nonempty selected support required')
    if not all(np.isfinite(a).all() for a in [s,U,V,delta]):raise ValueError('Nonfinite operand')
    a=s+U@delta;logzs=logsum(a[selected]);hS=np.exp(a[selected]-logzs)@V[selected]
    tail=np.ones(len(s),dtype=bool);tail[selected]=False
    if not tail.any():return hS,{'hasTail':False,'tailWeight':0.,'selectedMean':hS,'tail':tail}
    st=s[tail];ut=U[tail];vt=V[tail];logzt0=logsum(st);anchor=np.exp(st-logzt0);mu=anchor@ut;t=(ut-mu)@delta
    if not np.isfinite(t).all():raise ValueError('Nonfinite directional tilt')
    # r2(t)=((t+1)^2+1)/2. Log arithmetic avoids squaring large t.
    with np.errstate(divide='ignore'):logr=np.logaddexp(2*np.log(np.abs(t+1)),0.)-np.log(2.)
    logpoly=logsum(st+logr);p2=np.exp(st+logr-logpoly);h2=p2@vt;logztc=logzt0+float(mu@delta);logzt2=logpoly+float(mu@delta)
    w2=float(np.exp(logzt2-logsum(np.array([logzs,logzt2]))));raw=(1-w2)*hS+w2*h2
    if not np.isfinite(raw).all():raise ValueError('Nonfinite quadratic read')
    return raw,{'hasTail':True,'tailWeight':w2,'selectedMean':hS,'tail':tail,'anchor':anchor,'tailMean':h2,'tailProbability':p2,
        't':t,'values':vt,'currentScores':a,'logSelectedMass':logzs,'logAnchorTailMass':logzt0,'logCentroidTailMass':logztc,
        'logQuadraticTailMass':logzt2,'logKappa2':logpoly-logzt0}


def head(p,asset,frozen,budget):
    q,k,v=[floats(p[key]).astype(float) for key in ['q','k','v']];n,d=q.shape;x=q/d**.25;u=k/d**.25-asset['keyCenter'];routes=nearest(x,asset)[0]
    if not np.array_equal(routes,frozen['nearestIndex']):raise ValueError('Frozen routes changed')
    if not np.array_equal(p['positions'],np.arange(n)) or np.any((p['mask']!=0)&(p['mask']!=1)):raise ValueError('Invalid prefix clock/mask')
    raw=np.zeros((n,d))
    for i in range(n):
        allowed=np.flatnonzero(p['mask'][:i+1]);chosen=ordered_support(frozen['supportPositions'][i],allowed,budget)
        if not len(allowed):continue
        ell=asset['landmarks'][routes[i]];U=u[allowed];raw[i]=positive_read(U@ell,U,v[allowed],chosen,x[i]-ell)[0]
    return raw,routes


def serve(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False);start=time.perf_counter();cpu=time.process_time()
    if sha(spec['asset']['path'])!=spec['asset']['sha256']:raise ValueError('Source geometry differs')
    with np.load(spec['asset']['path'],allow_pickle=False) as f:asset={k:f[k] for k in ['landmarks','metricRoots','keyCenters']}
    m=spec['geometry'];d,heads,_=geometry(m);rows=[];timings=[]
    for item in spec['ports']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Saved port differs')
        with np.load(item['path'],allow_pickle=False) as f:full={k:f[k] for k in f.files}
        support=next(x for x in spec['frozenSupports'] if x['id']==item['id'])
        if sha(support['path'])!=support['sha256']:raise ValueError('Frozen support differs')
        with np.load(support['path'],allow_pickle=False) as f:frozen={k:f[k] for k in ['supportPositions','nearestIndex']}
        raw=np.zeros((len(full['ids']),heads*d));head_times=[];tick=time.perf_counter()
        for h in range(heads):
            t=time.perf_counter();raw[:,h*d:(h+1)*d],_=head(packet(full,m,h),head_asset(asset,m,h),{k:a[:,h] for k,a in frozen.items()},spec['budget']);head_times.append(time.perf_counter()-t)
        path=out/(item['id']+'-quadratic.npz');np.savez(path,raw=raw,rounded=words(raw),supportPositions=frozen['supportPositions'],nearestIndex=frozen['nearestIndex'])
        rows.append({'id':item['id'],'method':'quadratic','path':str(path),'sha256':sha(path)});timings.append({'id':item['id'],'headSeconds':head_times,'withSerializationSeconds':time.perf_counter()-tick})
        print('POSITIVE_TAIL_BANK_SEALED',item['id'],flush=True)
    save(out/'prediction-seal.json',{'schema':'tce-positive-tail-seal-v1','specSHA256':sha(spec_path),'asset':spec['asset'],'ports':spec['ports'],'frozenSupports':spec['frozenSupports'],
        'rows':rows,'timings':timings,'wallSecondsAfterImports':time.perf_counter()-start,'cpuSecondsAfterImports':time.process_time()-cpu,
        'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'runtimeSourceGeometryBytes':sum(a.nbytes for a in asset.values()),
        'frameworkCalls':0,'originalDenseProjections':0,'exactHeadReferenceEvaluations':0,
        'scope':'One positive directional quadratic measure with fixed current support/routes. Fullprefix supplied operands; no exact-tail/control-output/original-parameter access. Not compact serving or zero-neural image.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('output');a=p.parse_args();serve(a.spec,a.output)
