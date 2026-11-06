"""Read the existing sparse state with one positive tail-centroid atom.

No selector, state transition, persistent byte count or learned parameter changes.
This is reduced model-derived attention, not a zero-neural image generator.
"""
import argparse
from pathlib import Path
import resource
import time

from encoder_atoms import np,read,save,sha,guard,floats
from attention_bank import geometry,packet,head_asset
from sparse_tilt import stream,read_state as unit_read,mixture,core_bytes


def read_state(state,x,index,asset):
    d=len(x);filled=min(state['count'],state['scores'].shape[1])
    if not state['count']:return unit_read(state,x,index,asset)
    has_tail=state['z'][index]>0;logs=[];values=[];keys=[]
    if has_tail:
        logs.append(state['maximum'][index]+np.log(state['z'][index]))
        values.append(state['M'][index]/state['z'][index]);keys.append(state['Y'][index]/state['z'][index])
    logs=np.r_[logs,state['scores'][index,:filled]]
    values=np.vstack([np.asarray(values).reshape(-1,d),floats(state['values'][index,:filled]).astype(float)])
    keys=np.vstack([np.asarray(keys).reshape(-1,d),floats(state['keys'][index,:filled]).astype(float)/d**.25-asset['keyCenter']])
    H,logZ,base=mixture(logs,values);mu=base@keys;tilt=(keys-mu)@(x-asset['landmarks'][index])
    h,logT,weights=mixture(logs+tilt,values);offset=int(has_tail)
    return h,H,{'tailCount':state['count']-filled,'supportCount':filled,
        'selectedAnchorMass':float(base[offset:].sum()),'selectedTiltedMass':float(weights[offset:].sum()),'partialLogPartitionShift':logT-logZ}


def position_sets(support):
    """Canonical occurrence sets, including the number of unused slots."""
    return np.sort(np.asarray(support),axis=-1)


def same_support(a,b):return np.all(position_sets(a)==position_sets(b),axis=-1)


def serve(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False);start=time.perf_counter();cpu=time.process_time()
    if sha(spec['asset']['path'])!=spec['asset']['sha256']:raise ValueError('Centroid asset differs')
    with np.load(spec['asset']['path'],allow_pickle=False) as f:asset={k:f[k] for k in ['landmarks','metricRoots','keyCenters']}
    m=spec['geometry'];d,heads,_=geometry(m);budget=spec['budget'];rows=[];states=[]
    for item in spec['ports']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Centroid port differs')
        with np.load(item['path'],allow_pickle=False) as f:full={k:f[k] for k in f.files}
        old=next(r for r in spec['primarySupport'] if r['id']==item['id'])
        if sha(old['path'])!=old['sha256']:raise ValueError('Frozen eligible support differs')
        with np.load(old['path'],allow_pickle=False) as f:frozen=f['supportPositions'];routes=f['nearestIndex']
        n=len(full['ids']);rank=asset['landmarks'].shape[1]
        for law,reader in [('unit',unit_read),('centroid',read_state)]:
            tick=time.perf_counter();raw=np.empty((n,heads*d));bits=np.empty((n,heads*d),dtype=np.uint16)
            valid=np.empty((n,heads),dtype=bool);support=np.empty((n,heads,budget),dtype=np.int64);indices=np.empty((n,heads),dtype=np.int64)
            hull=np.empty((n,heads));head_times=[]
            for head in range(heads):
                t=time.perf_counter();output,state=stream(packet(full,m,head),head_asset(asset,m,head),budget,readout=reader);head_times.append(time.perf_counter()-t)
                if not same_support(output['supportPositions'],frozen[:,head]).all() or not np.array_equal(output['nearestIndex'],routes[:,head]):raise ValueError('Eligible support or route changed')
                span=slice(head*d,(head+1)*d);raw[:,span]=output['raw'];bits[:,span]=output['rounded'];valid[:,head]=output['rounded_valid']
                support[:,head]=output['supportPositions'];indices[:,head]=output['nearestIndex'];hull[:,head]=output['hullViolation']
                path=out/(item['id']+'-'+law+'-state'+str(head)+'.npz');np.savez(path,**{k:np.asarray(v) for k,v in state.items()})
                states.append({'id':item['id'],'law':law,'queryHead':head,'path':str(path),'sha256':sha(path)})
            path=out/(item['id']+'-'+law+'.npz');np.savez(path,raw=raw,rounded=bits,valid=valid,supportPositions=support,nearestIndex=indices,hullViolation=hull)
            rows.append({'id':item['id'],'method':'eligible-'+law,'path':str(path),'sha256':sha(path),'headSeconds':head_times,
                'withSerializationSeconds':time.perf_counter()-tick,'logicalCoreBytesPerLaw':heads*core_bytes(d,rank,budget),
                'invalidHeadReads':int(np.count_nonzero(~valid)),'maximumHullViolation':float(np.nanmax(hull)),'frozenSupportMismatches':0,'routeMismatches':0})
            print('FROZEN_SUPPORT_BANK_SEALED',item['id'],law,flush=True)
    save(out/'prediction-seal.json',{'schema':'tce-tail-centroid-seal-v1','specSHA256':sha(spec_path),'asset':spec['asset'],'ports':spec['ports'],
        'rows':rows,'states':states,'runtimeAssetBytes':sum(x.nbytes for x in asset.values()),'wallSecondsAfterImports':time.perf_counter()-start,
        'cpuSecondsAfterImports':time.process_time()-cpu,'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'frameworkCalls':0,'originalDenseProjections':0,'scope':'Retrospective supplied ports. Same eligible supports/routes and state law; unit/centroid reads both seal before oracle/reference/learned consumer.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('output');a=p.parse_args();serve(a.spec,a.output)
