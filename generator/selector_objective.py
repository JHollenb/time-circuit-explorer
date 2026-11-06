"""Full-prefix selector diagnostics. These are not compact serving algorithms."""
import argparse
from pathlib import Path
import resource
import time

from encoder_atoms import np,read,save,sha,guard,floats
from attention_bank import geometry,packet,head_asset
from context_transport import nearest,rounded
from sparse_tilt_study import logsum,log_abs_expm1
from centroid_tilt_study import direct_centroid

METHODS=['current','envelope']


def select(score,positions,budget=4):
    score=np.asarray(score);positions=np.asarray(positions)
    if score.shape!=positions.shape or score.ndim!=1 or not np.isfinite(score).all() or budget<0:raise ValueError('Invalid selector scores')
    return np.lexsort((positions,-score))[:budget]


def predict(p,asset,budget=4):
    q,k,v=[floats(p[key]).astype(float) for key in ['q','k','v']];n,d=q.shape
    if q.shape!=k.shape or q.shape!=v.shape or p['mask'].shape!=(n,) or np.any((p['mask']!=0)&(p['mask']!=1)):raise ValueError('Selector packet differs')
    if not np.array_equal(p['positions'],np.arange(n)):raise ValueError('Diagnostic requires complete prefix clocks')
    if not all(np.isfinite(a).all() for a in [q,k,v,asset['landmarks'],asset['metricRoot'],asset['keyCenter']]):raise ValueError('Nonfinite selector operand')
    x=q/d**.25;u=k/d**.25-asset['keyCenter'];route,distance=nearest(x,asset)
    result={method:{'raw':np.zeros((n,d)),'supportPositions':np.full((n,budget),-1,dtype=np.int64),'hullViolation':np.zeros(n)} for method in METHODS}
    scanned=0
    for i in range(n):
        allowed=np.flatnonzero(p['mask'][:i+1]);scanned+=len(allowed)
        if not len(allowed):continue
        U=u[allowed];V=v[allowed];ell=asset['landmarks'][route[i]];s=U@ell;a=U@x[i]
        for method,priority in [('current',a),('envelope',np.maximum(s,a))]:
            chosen=select(priority,allowed,budget);h=direct_centroid(s,U,V,chosen,x[i]-ell)
            result[method]['raw'][i]=h;result[method]['supportPositions'][i,:len(chosen)]=allowed[chosen]
            result[method]['hullViolation'][i]=max(0.,float(np.max(h-V.max(0))),float(np.max(V.min(0)-h)))
    for method in METHODS:
        raw=result[method]['raw'];bits,valid=rounded(raw,np.isfinite(raw).all(1));result[method].update(rounded=bits,valid=valid,nearestIndex=route)
    return result,{'scannedKeyOccurrences':scanned,'sharedAnchorAndCurrentScoreMACs':2*scanned*d,'sourceDistanceSquared':distance}


def validate_support(positions,allowed,budget):
    positions=np.asarray(positions);kept=positions[positions>=0]
    if len(set(kept))!=len(kept) or not set(kept).issubset(set(allowed)) or len(kept)!=min(len(allowed),budget):raise ValueError('Support is not a complete causal occurrence set')
    return np.flatnonzero(np.isin(allowed,kept))


def tilt_measure(s,U,V,delta):
    """Signed tilt mass scaled by its exact partition; log masses retain tails."""
    logp=s-logsum(s);p=np.exp(logp);H=p@V;mu=p@U;t=(U-mu)@delta
    logmass=logp+log_abs_expm1(t);logD=logsum(logp+t);beta=float(np.exp(-logD))
    c=np.sign(t)*np.exp(logmass-logD);positive=t>0;negative=t<0
    lp=logsum(logmass[positive]);ln=logsum(logmass[negative])
    full_num=c@V;full_partition=float(c.sum());exact=np.exp(logp+t-logD)@V
    return {'logmass':logmass,'positive':positive,'negative':negative,'coefficients':c,'logD':logD,'beta':beta,
        'positiveLogMass':lp,'negativeLogMass':ln,'positiveMassOverD':float(np.exp(lp-logD)),
        'negativeMassOverD':float(np.exp(ln-logD)),'anchorRead':H,'fullSignedNumeratorOverD':full_num,'fullSignedPartitionOverD':full_partition,
        'partitionClosure':abs(beta+full_partition-1.),'readClosure':float(np.max(np.abs(beta*H+full_num-exact)))}


def capture(measure,selected,V):
    keep=np.zeros(len(V),dtype=bool);keep[selected]=True;out={}
    for label,sign,total_log in [('positive',measure['positive'],measure['positiveLogMass']),('negative',measure['negative'],measure['negativeLogMass'])]:
        captured_log=logsum(measure['logmass'][keep&sign])
        out[label+'CaptureFraction']=float(np.exp(captured_log-total_log)) if np.isfinite(total_log) else np.nan
        out[label+'CapturedMassOverD']=float(np.exp(captured_log-measure['logD']))
    total=logsum(measure['logmass']);chosen=logsum(measure['logmass'][selected])
    out['absoluteCaptureFraction']=float(np.exp(chosen-total)) if np.isfinite(total) else np.nan
    c=measure['coefficients'][selected];out['signedPartitionOverD']=float(c.sum());out['signedNumeratorOverD']=c@V[selected]
    return out


def serve(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False);start=time.perf_counter();cpu=time.process_time()
    if sha(spec['asset']['path'])!=spec['asset']['sha256']:raise ValueError('Source geometry differs')
    with np.load(spec['asset']['path'],allow_pickle=False) as f:asset={k:f[k] for k in ['landmarks','metricRoots','keyCenters']}
    m=spec['geometry'];d,heads,_=geometry(m);rows=[];counts=[]
    for item in spec['ports']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Saved packet differs')
        with np.load(item['path'],allow_pickle=False) as f:full={k:f[k] for k in f.files}
        n=len(full['ids']);buffers={method:{'raw':np.empty((n,heads*d)),'rounded':np.empty((n,heads*d),dtype=np.uint16),
            'valid':np.empty((n,heads),dtype=bool),'supportPositions':np.empty((n,heads,spec['budget']),dtype=np.int64),
            'nearestIndex':np.empty((n,heads),dtype=np.int64),'hullViolation':np.empty((n,heads))} for method in METHODS}
        head_times=[];scan=0;macs=0;tick=time.perf_counter()
        for head in range(heads):
            t=time.perf_counter();values,account=predict(packet(full,m,head),head_asset(asset,m,head),spec['budget']);head_times.append(time.perf_counter()-t)
            scan+=account['scannedKeyOccurrences'];macs+=account['sharedAnchorAndCurrentScoreMACs'];span=slice(head*d,(head+1)*d)
            for method in METHODS:
                for key,value in values[method].items():buffers[method][key][:,span if key in ['raw','rounded'] else head]=value
        for method in METHODS:
            values=buffers[method];path=out/(item['id']+'-'+method+'.npz');np.savez(path,**values)
            rows.append({'id':item['id'],'method':method,'path':str(path),'sha256':sha(path),'invalidHeadReads':int(np.count_nonzero(~values['valid'])),
                'maximumHullViolation':float(np.nanmax(values['hullViolation']))})
        counts.append({'id':item['id'],'headSecondsBothSelectors':head_times,'bothSelectorsWithSerializationSeconds':time.perf_counter()-tick,
            'scannedKeyOccurrencesSharedAcrossSelectors':scan,'anchorAndCurrentScoreMACsShared':macs,
            'scope':'One full-prefix anchor/current score pair per query/head, shared by both selectors; tail read/sort/routing/diagnostics work additional.'})
        print('FULL_PREFIX_SELECTORS_SEALED',item['id'],flush=True)
    save(out/'prediction-seal.json',{'schema':'tce-selector-objective-seal-v1','specSHA256':sha(spec_path),'asset':spec['asset'],'ports':spec['ports'],'rows':rows,'counts':counts,
        'runtimeAssetBytes':sum(a.nbytes for a in asset.values()),'wallSecondsAfterImports':time.perf_counter()-start,'cpuSecondsAfterImports':time.process_time()-cpu,
        'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'frameworkCalls':0,'originalDenseProjections':0,
        'scope':'Full current causal prefix scans; diagnostic selectors, not compact ownstate serving. Same source geometry/centroid law; new supports and complete banks sealed before targets/original consumers.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('output');a=p.parse_args();serve(a.spec,a.output)
