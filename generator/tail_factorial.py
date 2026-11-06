"""Fixed-support positive tail mass/value factorial; full-prefix diagnostic only."""
import argparse
from pathlib import Path
import resource
import time

from encoder_atoms import np,read,save,sha,guard,floats,words
from attention_bank import geometry,packet,head_asset
from context_transport import nearest
from context_statistics_study import exact_head
from sparse_tilt_study import logsum
from centroid_tilt_study import direct_centroid

ARMS=['00','10','01','11']
VECTORS=['mass','value','interaction','controlArithmetic']


def ordered_support(positions,allowed,budget):
    kept=np.asarray(positions);kept=kept[kept>=0]
    if len(kept)!=min(len(allowed),budget) or len(set(kept))!=len(kept) or not set(kept).issubset(set(allowed)):raise ValueError('Frozen support is not causal/complete')
    # Preserve the frozen score order; chronological sorting changes reduction order.
    return np.searchsorted(allowed,kept)


def factorial(s,U,V,selected,delta):
    if not len(selected):raise ValueError('Nonempty selected support required')
    a=s+U@delta;logzs=logsum(a[selected]);hS=np.exp(a[selected]-logzs)@V[selected]
    tail=np.ones(len(s),dtype=bool);tail[selected]=False
    if not tail.any():
        return {arm:hS.copy() for arm in ARMS},{**{key:np.zeros(V.shape[1]) for key in ['mass','value','interaction']},
            'tailWeight':0.,'exactTailWeight':0.,'logKappa':np.nan,'hasTail':False,'telescopeClosure':0.,'interactionClosure':0.}
    logzt0=logsum(s[tail]);anchor=np.exp(s[tail]-logzt0);mu=anchor@U[tail];hT0=anchor@V[tail]
    logztc=logzt0+float(mu@delta);logzte=logsum(a[tail]);hTe=np.exp(a[tail]-logzte)@V[tail]
    w=float(np.exp(logztc-logsum(np.array([logzs,logztc]))));we=float(np.exp(logzte-logsum(np.array([logzs,logzte]))))
    out={'00':(1-w)*hS+w*hT0,'10':(1-we)*hS+we*hT0,'01':(1-w)*hS+w*hTe,'11':(1-we)*hS+we*hTe}
    mass=(w-we)*(hT0-hS);value=we*(hT0-hTe);interaction=(w-we)*(hT0-hTe)
    return out,{'mass':mass,'value':value,'interaction':interaction,'tailWeight':w,'exactTailWeight':we,'logKappa':logzte-logztc,'hasTail':True,
        'telescopeClosure':float(np.max(np.abs(out['00']-out['11']-mass-value))),
        'interactionClosure':float(np.max(np.abs(out['11']-out['10']-out['01']+out['00']-interaction)))}


def head(p,asset,frozen,budget):
    q,k,v=[floats(p[key]).astype(float) for key in ['q','k','v']];n,d=q.shape;x=q/d**.25;u=k/d**.25-asset['keyCenter'];routes=nearest(x,asset)[0]
    if not np.array_equal(routes,frozen['nearestIndex']):raise ValueError('Frozen routes changed')
    if not np.array_equal(p['positions'],np.arange(n)) or np.any((p['mask']!=0)&(p['mask']!=1)):raise ValueError('Invalid full prefix clock/mask')
    if not all(np.isfinite(a).all() for a in [q,k,v,asset['landmarks'],asset['keyCenter']]):raise ValueError('Nonfinite operand')
    banks={arm:np.zeros((n,d)) for arm in ARMS};diagnostics={key:np.zeros((n,d)) for key in VECTORS}
    diagnostics.update({key:np.zeros(n) for key in ['tailWeight','exactTailWeight','hasTail','telescopeClosure','interactionClosure','centroidControlClosure','exactControlClosure']});diagnostics['logKappa']=np.full(n,np.nan)
    exact=exact_head(p)['raw']
    for i in range(n):
        allowed=np.flatnonzero(p['mask'][:i+1]);chosen=ordered_support(frozen['supportPositions'][i],allowed,budget)
        if not len(allowed):continue
        U=u[allowed];V=v[allowed];ell=asset['landmarks'][routes[i]];s=U@ell;delta=x[i]-ell
        smooth,diag=factorial(s,U,V,chosen,delta);canonical=direct_centroid(s,U,V,chosen,delta)
        # Controls preserve their established arithmetic, not just an equivalent grouping.
        for arm in ARMS:banks[arm][i]=canonical if arm=='00' else exact[i] if arm=='11' else smooth[arm]
        for key,value in diag.items():diagnostics[key][i]=value
        diagnostics['controlArithmetic'][i]=(canonical-exact[i])-(diag['mass']+diag['value'])
        diagnostics['centroidControlClosure'][i]=float(np.max(np.abs(canonical-smooth['00'])))
        diagnostics['exactControlClosure'][i]=float(np.max(np.abs(exact[i]-smooth['11'])))
    return banks,diagnostics,routes


def serve(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False);start=time.perf_counter();cpu=time.process_time()
    if sha(spec['asset']['path'])!=spec['asset']['sha256']:raise ValueError('Source geometry differs')
    with np.load(spec['asset']['path'],allow_pickle=False) as f:asset={k:f[k] for k in ['landmarks','metricRoots','keyCenters']}
    m=spec['geometry'];d,heads,_=geometry(m);rows=[];diagnostic_rows=[];timings=[]
    for item in spec['ports']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Saved port differs')
        with np.load(item['path'],allow_pickle=False) as f:full={k:f[k] for k in f.files}
        support=next(x for x in spec['frozenSupports'] if x['id']==item['id'])
        if sha(support['path'])!=support['sha256']:raise ValueError('Frozen support bank differs')
        # Only these two NPZ members become operands; raw/control output members are not loaded.
        with np.load(support['path'],allow_pickle=False) as f:frozen={k:f[k] for k in ['supportPositions','nearestIndex']}
        n=len(full['ids']);banks={arm:np.zeros((n,heads*d)) for arm in ARMS};fields={};head_times=[];tick=time.perf_counter()
        for h in range(heads):
            t=time.perf_counter();values,diag,route=head(packet(full,m,h),head_asset(asset,m,h),{k:a[:,h] for k,a in frozen.items()},spec['budget']);head_times.append(time.perf_counter()-t);span=slice(h*d,(h+1)*d)
            for arm in ARMS:banks[arm][:,span]=values[arm]
            for key,value in diag.items():
                if key not in fields:fields[key]=np.empty((n,heads*d) if key in VECTORS else (n,heads))
                fields[key][:,span if key in VECTORS else h]=value
        for arm,raw in banks.items():
            if not np.isfinite(raw).all():raise ValueError('Nonfinite factorial bank')
            path=out/(item['id']+'-'+arm+'.npz');np.savez(path,raw=raw,rounded=words(raw),supportPositions=frozen['supportPositions'],nearestIndex=frozen['nearestIndex'])
            rows.append({'id':item['id'],'arm':arm,'path':str(path),'sha256':sha(path)})
        path=out/(item['id']+'-factorial-diagnostics.npz');np.savez(path,**fields);diagnostic_rows.append({'id':item['id'],'path':str(path),'sha256':sha(path)})
        timings.append({'id':item['id'],'headSeconds':head_times,'withSerializationSeconds':time.perf_counter()-tick})
        print('TAIL_FACTORIAL_BANKS_SEALED',item['id'],flush=True)
    save(out/'prediction-seal.json',{'schema':'tce-tail-factorial-seal-v1','specSHA256':sha(spec_path),'asset':spec['asset'],'ports':spec['ports'],'frozenSupports':spec['frozenSupports'],
        'rows':rows,'diagnostics':diagnostic_rows,'timings':timings,'wallSecondsAfterImports':time.perf_counter()-start,'cpuSecondsAfterImports':time.process_time()-cpu,
        'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'runtimeSourceGeometryBytes':sum(a.nbytes for a in asset.values()),
        'exactHeadReferenceEvaluations':heads*len(spec['ports']),'frameworkCalls':0,'originalDenseProjections':0,
        'scope':'Full current-prefix mass/value oracle repairs with frozen support/order/routes. Support input reads only supportPositions/nearestIndex NPZ members; output members absent from operands. Canonical00/11 arithmetic separate from smooth positive-mixture identities. Not serving or strictzero-neural image.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('output');a=p.parse_args();serve(a.spec,a.output)
