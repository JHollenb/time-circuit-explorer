"""Positive sparse query tilt with fixed-score heaps and append-only tails.

No covariance tensor, reference head or original learned projection is used.
This remains model-derived attention, not a zero-neural image generator.
"""
import argparse
import hashlib
from pathlib import Path
import resource
import time

from encoder_atoms import np,read,save,sha,guard,floats
from attention_bank import geometry,packet,head_asset
from context_transport import nearest,rounded
from context_statistics import copy_state

BUDGETS=[1,2,4]


def new_state(d,rank,budget):
    if min(d,rank)<1 or budget<0:raise ValueError('Invalid sparse state geometry')
    return {'maximum':np.full(rank,-np.inf),'z':np.zeros(rank),'M':np.zeros((rank,d)),'Y':np.zeros((rank,d)),
        'scores':np.full((rank,budget),-np.inf),'positions':np.full((rank,budget),-1,dtype=np.int64),
        'keys':np.zeros((rank,budget,d),dtype=np.uint16),'values':np.zeros((rank,budget,d),dtype=np.uint16),
        'lo':np.full(d,np.inf),'hi':np.full(d,-np.inf),'count':0,'steps':0,'evictions':0,'tailUpdates':0}


def append_tail(state,u,v,score):
    maximum=np.maximum(state['maximum'],score);rescale=np.exp(state['maximum']-maximum);added=np.exp(score-maximum)
    state['M']=state['M']*rescale[:,None]+added[:,None]*v
    state['Y']=state['Y']*rescale[:,None]+added[:,None]*u
    state['z']=state['z']*rescale+added;state['maximum']=maximum;state['tailUpdates']+=len(score)
    if not all(np.isfinite(state[k]).all() for k in ['M','Y','z','maximum']) or np.any(state['z']<=0):raise ValueError('Invalid positive tail state')


def advance(state,key,value,position,asset):
    rank,budget=state['scores'].shape;d=len(key);u=floats(key).astype(float)/d**.25-asset['keyCenter'];v=floats(value).astype(float)
    score=asset['landmarks']@u;filled=min(state['count'],budget);rows=np.arange(rank)
    if filled<budget:
        state['scores'][:,filled]=score;state['positions'][:,filled]=position
        state['keys'][:,filled]=key;state['values'][:,filled]=value
    elif budget:
        minimum=state['scores'].min(1)
        # Equal scores retain earlier occurrences; the latest admitted minimum loses.
        worst=np.argmax(np.where(state['scores']==minimum[:,None],state['positions'],-1),axis=1)
        replace=score>minimum
        old_u=floats(state['keys'][rows,worst]).astype(float)/d**.25-asset['keyCenter']
        old_v=floats(state['values'][rows,worst]).astype(float);old_score=state['scores'][rows,worst]
        append_tail(state,np.where(replace[:,None],old_u,u),np.where(replace[:,None],old_v,v),np.where(replace,old_score,score))
        selected=rows[replace];slots=worst[replace]
        state['scores'][selected,slots]=score[replace];state['positions'][selected,slots]=position
        state['keys'][selected,slots]=key;state['values'][selected,slots]=value;state['evictions']+=int(replace.sum())
    else:append_tail(state,np.broadcast_to(u,(rank,d)),np.broadcast_to(v,(rank,d)),score)
    state['count']+=1;state['lo']=np.minimum(state['lo'],v);state['hi']=np.maximum(state['hi'],v)


def mixture(logs,values):
    maximum=logs.max();weights=np.exp(logs-maximum);partition=weights.sum();weights/=partition
    return weights@values,float(maximum+np.log(partition)),weights


def read_state(state,x,index,asset):
    d=len(x);budget=state['scores'].shape[1];filled=min(state['count'],budget)
    if not state['count']:return np.zeros(d),np.zeros(d),{'tailCount':0,'supportCount':0,'selectedAnchorMass':0.,'selectedTiltedMass':0.,'partialLogPartitionShift':0.}
    has_tail=state['z'][index]>0;logs=[];values=[];keys=[]
    if has_tail:
        logs.append(state['maximum'][index]+np.log(state['z'][index]));values.append(state['M'][index]/state['z'][index]);keys.append(state['Y'][index]/state['z'][index])
    support_u=floats(state['keys'][index,:filled]).astype(float)/d**.25-asset['keyCenter']
    support_v=floats(state['values'][index,:filled]).astype(float)
    logs=np.r_[logs,state['scores'][index,:filled]];values=np.vstack([np.asarray(values).reshape(-1,d),support_v]);keys=np.vstack([np.asarray(keys).reshape(-1,d),support_u])
    H,logZ,base=mixture(logs,values);mu=base@keys;tilted=logs.copy();offset=int(has_tail)
    tilted[offset:]+=(support_u-mu)@(x-asset['landmarks'][index])
    h,logT,weights=mixture(tilted,values)
    return h,H,{'tailCount':state['count']-filled,'supportCount':filled,
        'selectedAnchorMass':float(base[offset:].sum()),'selectedTiltedMass':float(weights[offset:].sum()),'partialLogPartitionShift':logT-logZ}


def core_bytes(d,rank,budget):return rank*(16+2*d*8+budget*(16+4*d))+16


def stream(p,asset,budget,state=None,readout=read_state):
    q,k,v=[floats(p[key]).astype(float) for key in ['q','k','v']];n,d=q.shape;rank=len(asset['landmarks'])
    if q.shape!=k.shape or q.shape!=v.shape or asset['landmarks'].shape!=(rank,d) or asset['metricRoot'].shape!=(d,d) or asset['keyCenter'].shape!=(d,):raise ValueError('Sparse operand shapes differ')
    if not all(np.isfinite(a).all() for a in [q,k,v,asset['landmarks'],asset['metricRoot'],asset['keyCenter']]):raise ValueError('Nonfinite sparse operand')
    state=new_state(d,rank,budget) if state is None else copy_state(state)
    tag=hashlib.sha256(str(budget).encode()+b''.join(np.asarray(asset[key],dtype='<f8').tobytes() for key in ['landmarks','metricRoot','keyCenter'])).hexdigest()
    if state.get('assetIdentity',tag)!=tag:raise ValueError('Sparse state source differs')
    state['assetIdentity']=tag
    if state['scores'].shape!=(rank,budget) or state['keys'].shape!=(rank,budget,d) or state['values'].shape!=(rank,budget,d) or state['M'].shape!=(rank,d) or state['Y'].shape!=(rank,d):raise ValueError('Sparse state shape differs')
    if not np.array_equal(p['positions'],np.arange(state['steps'],state['steps']+n)):raise ValueError('Sparse state clock differs')
    if p['mask'].shape!=(n,) or np.any((p['mask']!=0)&(p['mask']!=1)):raise ValueError('Sparse binary mask differs')
    x=q/d**.25;indices,distances=nearest(x,asset);out=[];baseline=[];trace=[];supports=[]
    for i in range(n):
        if p['mask'][i]:advance(state,p['k'][i],p['v'][i],int(p['positions'][i]),asset)
        state['steps']+=1;a=int(indices[i]);h,H,row=readout(state,x[i],a,asset);out.append(h);baseline.append(H)
        finite=bool(np.isfinite(h).all());violation=0. if not state['count'] else max(0.,float(np.max(h-state['hi'])),float(np.max(state['lo']-h))) if finite else np.nan
        trace.append({**row,'nearestIndex':a,'sourceDistanceSquared':float(distances[i]),'hullViolation':violation,
            'evictions':state['evictions'],'tailUpdates':state['tailUpdates'],'valid':finite});supports.append(state['positions'][a].copy())
    raw=np.stack(out);valid=np.asarray([r['valid'] for r in trace]);bits,ok=rounded(raw,valid)
    return {'raw':raw,'rounded':bits,'rounded_valid':ok,'anchor_raw':np.stack(baseline),'supportPositions':np.stack(supports),
        **{key:np.asarray([r[key] for r in trace]) for key in trace[0]}},state


def serve(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False);start=time.perf_counter();cpu=time.process_time()
    if sha(spec['asset']['path'])!=spec['asset']['sha256']:raise ValueError('Sparse source asset differs')
    with np.load(spec['asset']['path'],allow_pickle=False) as f:asset={k:f[k] for k in ['landmarks','metricRoots','keyCenters']}
    m=spec['geometry'];d,heads,groups=geometry(m);rows=[];states=[]
    for item in spec['ports']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Supplied port differs')
        with np.load(item['path'],allow_pickle=False) as f:full={k:f[k] for k in f.files}
        n=len(full['ids']);rank=asset['landmarks'].shape[1]
        for budget in spec['budgets']:
            t=time.perf_counter();raw=np.empty((n,heads*d));bits=np.empty((n,heads*d),dtype=np.uint16);valid=np.empty((n,heads),dtype=bool);support=np.empty((n,heads,budget),dtype=np.int64);traces={};head_times=[]
            for head in range(heads):
                tick=time.perf_counter();output,state=stream(packet(full,m,head),head_asset(asset,m,head),budget);head_times.append(time.perf_counter()-tick)
                span=slice(head*d,(head+1)*d);raw[:,span]=output['raw'];bits[:,span]=output['rounded'];valid[:,head]=output['rounded_valid'];support[:,head]=output['supportPositions']
                for key in ['nearestIndex','hullViolation','selectedAnchorMass','selectedTiltedMass','partialLogPartitionShift','evictions','tailUpdates','sourceDistanceSquared']:
                    if key not in traces:traces[key]=np.empty((n,heads),dtype=output[key].dtype)
                    traces[key][:,head]=output[key]
                path=out/(item['id']+'-m'+str(budget)+'-state'+str(head)+'.npz');np.savez(path,**{k:np.asarray(v) for k,v in state.items()})
                states.append({'id':item['id'],'budget':budget,'queryHead':head,'path':str(path),'sha256':sha(path)})
            path=out/(item['id']+'-m'+str(budget)+'.npz');np.savez(path,raw=raw,rounded=bits,valid=valid,supportPositions=support,**traces)
            rows.append({'id':item['id'],'budget':budget,'path':str(path),'sha256':sha(path),'headSeconds':head_times,
                'withSerializationSeconds':time.perf_counter()-t,'activeTokens':int(full['mask'].sum()),'coreBytesPerHead':core_bytes(d,rank,budget),
                'logicalCoreBytes':heads*core_bytes(d,rank,budget),'rangeDiagnosticBytes':heads*2*d*8,'counterDiagnosticBytes':heads*16,
                'sourceIdentityUTF8Bytes':heads*64,'invalidHeadReads':int(np.count_nonzero(~valid)),
                'maximumHullViolation':float(np.nanmax(traces['hullViolation'])),'totalTailUpdates':int(traces['tailUpdates'][-1].sum()),'totalEvictions':int(traces['evictions'][-1].sum())})
            print('SPARSE_BANK_SEALED',item['id'],budget,flush=True)
    save(out/'prediction-seal.json',{'schema':'tce-sparse-joint-tilt-seal-v1','specSHA256':sha(spec_path),'asset':spec['asset'],'ports':spec['ports'],'rows':rows,'states':states,
        'runtimeAssetBytes':sum(x.nbytes for x in asset.values()),'wallSecondsAfterImports':time.perf_counter()-start,'cpuSecondsAfterImports':time.process_time()-cpu,
        'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'frameworkCalls':0,'originalDenseProjections':0,'referenceHeadsSupplied':0,
        'scope':'Serving-eligible sparse state law on supplied saved ports, not fresh raw-text validation. All methods seal before full-query-score oracle/references/original consumers. NoCvy.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('output');a=p.parse_args();serve(a.spec,a.output)
