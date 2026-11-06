"""Source-cell score envelopes select keys; original anchor weights read them.

Four frozen witnesses per anchor, fixed source-key-mean gauge, zero offsets.
This is reduced model-derived attention, not zero-neural image generation.
"""
import argparse
import hashlib
from pathlib import Path
import resource
import time

from encoder_atoms import np,AtomStore,checked,read,save,sha,guard,floats,words,bf
from attention_bank import geometry,packet,head_asset
from context_transport import rounded
from sparse_tilt import new_state,append_tail,stream as base_stream,core_bytes
from centroid_tilt import read_state

POLICY='source-cell-max4-source-key-mean-zero-offset-v1'


def select_witnesses(x,asset,identities,anchor_ids):
    """Existing source metric, canonical ties, farthest-first within each cell."""
    n,d=x.shape;rank=len(asset['landmarks']);points=x@asset['metricRoot'];anchors=asset['landmarks']@asset['metricRoot']
    if identities.shape!=(n,2) or anchor_ids.shape!=(rank,2) or not np.array_equal(np.lexsort((identities[:,1],identities[:,0])),np.arange(n)):raise ValueError('Source identities are not canonical')
    lookup={tuple(z):i for i,z in enumerate(identities)};first=np.array([lookup[tuple(z)] for z in anchor_ids])
    if not np.array_equal(x[first],asset['landmarks']):raise ValueError('Reconstructed anchor differs')
    distance=np.stack([np.sum((points-p)**2,axis=1) for p in anchors],axis=1);route=np.argmin(distance,axis=1)
    chosen=np.empty((rank,4),dtype=np.int64);rows=[]
    for j in range(rank):
        cell=np.flatnonzero(route==j);selected=[int(first[j])];closest=np.sum((points[cell]-anchors[j])**2,axis=1);steps=[];repeated=0
        for _ in range(3):
            remaining=~np.isin(cell,selected)
            if not remaining.any() or np.max(closest[remaining])<=0:
                index=int(first[j]);steps.append(0.);repeated+=1
            else:
                eligible=np.where(remaining,closest,-np.inf);slot=int(np.argmax(eligible));index=int(cell[slot]);steps.append(float(closest[slot]))
                closest=np.minimum(closest,np.sum((points[cell]-points[index])**2,axis=1))
            selected.append(index)
        chosen[j]=selected
        rows.append({'anchor':j,'cellCandidates':len(cell),'witnessIdentities':identities[selected].tolist(),
            'selectionDistanceSquared':steps,'repeatedAnchorSlots':repeated,'witnessNorms':np.linalg.norm(x[selected],axis=1).tolist()})
    return x[chosen[:,1:]],identities[chosen],route,rows


def compile_asset(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False);start=time.perf_counter()
    if sha(spec['asset']['path'])!=spec['asset']['sha256']:raise ValueError('Parent source asset differs')
    with np.load(spec['asset']['path'],allow_pickle=False) as f:parent={k:f[k] for k in f.files}
    store=AtomStore(spec['store'],spec['manifestSHA256']);m=store.manifest;d,heads,_=geometry(m);ids=parent['candidateIdentities']
    rope=np.load(checked(spec['store'],m['rope']),allow_pickle=False)
    if rope.shape!=(2,m['sequenceLength'],d) or rope.dtype!=np.dtype('<u2'):raise ValueError('Source rotary table differs')
    tick=time.perf_counter();atoms=store.gather(ids[:,0]);gather_s=time.perf_counter()-tick;cos,sin=[floats(a[ids[:,1]]) for a in rope]
    extra=[];identities=[];receipts=[]
    for head in range(heads):
        q=floats(atoms[:,m['hiddenSize']+head*d:m['hiddenSize']+(head+1)*d]);turn=np.concatenate([-q[:,d//2:],q[:,:d//2]],axis=1)
        x=floats(words(bf(q*cos)+bf(turn*sin))).astype(float)/d**.25;asset=head_asset(parent,m,head)
        witness,identity,route,rows=select_witnesses(x,asset,ids,parent['landmarkIdentities'][head]);extra.append(witness);identities.append(identity)
        receipts.append({'queryHead':head,'rows':rows,'sourceCandidateCount':len(x)})
    asset={k:parent[k] for k in ['landmarks','metricRoots','keyCenters']};asset['extraWitnesses']=np.stack(extra)
    np.savez(out/'witness-assets.npz',**asset);np.savez(out/'witness-identities.npz',witnessIdentities=np.stack(identities),candidateIdentities=ids)
    sources=[{'path':str(Path(spec['store'])/'manifest.json'),'sha256':spec['manifestSHA256']}]
    sources += [{'path':str(Path(spec['store'])/m[k]['path']),'sha256':m[k]['sha256']} for k in ['coverage','rowHashes','rope']]
    sources += [{'path':str(Path(spec['store'])/m['shards'][j]['path']),'sha256':m['shards'][j]['sha256']} for j in sorted(store.shards)]
    save(out/'compile-receipt.json',{'schema':'tce-source-witness-compile-v1','specSHA256':sha(spec_path),'assetSHA256':sha(out/'witness-assets.npz'),
        'identityPacketSHA256':sha(out/'witness-identities.npz'),'policy':POLICY,'heads':receipts,'sourceFiles':sources,
        'runtimeAssetBytes':sum(a.nbytes for a in asset.values()),'additionalWitnessBytes':asset['extraWitnesses'].nbytes,
        'identityMetadataArrayBytes':np.stack(identities).nbytes+ids.nbytes,'sourceGatherSeconds':gather_s,'compileSeconds':time.perf_counter()-start,
        'learnedAtomRowsRead':store.rowsRead,'learnedAtomPayloadBytesRead':store.rowBytesRead,'frameworkCalls':0,'originalDenseProjections':0,
        'scope':'Reconstruct existing source query pool only. Existing anchors and metric fixed; no current-prompt queries/targets or original weights.'})


def advance(state,key,value,position,asset):
    rank,budget=state['scores'].shape;d=len(key);u=floats(key).astype(float)/d**.25-asset['keyCenter'];v=floats(value).astype(float)
    score=asset['landmarks']@u;all_scores=np.column_stack([score,*[asset['extraWitnesses'][:,b]@u for b in range(3)]])
    winner=np.argmax(all_scores,axis=1);priority=all_scores[np.arange(rank),winner];rows=np.arange(rank)
    np.add.at(state['winningWitnessCounts'],(rows,winner),1);filled=min(state['count'],budget)
    if filled<budget:
        state['scores'][:,filled]=score;state['priorities'][:,filled]=priority;state['positions'][:,filled]=position
        state['keys'][:,filled]=key;state['values'][:,filled]=value
    elif budget:
        minimum=state['priorities'].min(1);worst=np.argmax(np.where(state['priorities']==minimum[:,None],state['positions'],-1),axis=1);replace=priority>minimum
        old_u=floats(state['keys'][rows,worst]).astype(float)/d**.25-asset['keyCenter'];old_v=floats(state['values'][rows,worst]).astype(float)
        # Selection uses rho; the tail always uses the original anchor score s.
        append_tail(state,np.where(replace[:,None],old_u,u),np.where(replace[:,None],old_v,v),np.where(replace,state['scores'][rows,worst],score))
        selected=rows[replace];slots=worst[replace];state['scores'][selected,slots]=score[replace];state['priorities'][selected,slots]=priority[replace]
        state['positions'][selected,slots]=position;state['keys'][selected,slots]=key;state['values'][selected,slots]=value;state['evictions']+=int(replace.sum())
    else:append_tail(state,np.broadcast_to(u,(rank,d)),np.broadcast_to(v,(rank,d)),score)
    state['count']+=1;state['lo']=np.minimum(state['lo'],v);state['hi']=np.maximum(state['hi'],v)


def stream(p,asset,budget,state=None):
    n,d=p['q'].shape;rank=len(asset['landmarks'])
    if asset['extraWitnesses'].shape!=(rank,3,d) or not np.isfinite(asset['extraWitnesses']).all():raise ValueError('Witness geometry differs')
    if state is None:
        state=new_state(d,rank,budget);state['priorities']=np.full((rank,budget),-np.inf);state['winningWitnessCounts']=np.zeros((rank,4),dtype=np.int64)
    if state.get('priorities',np.empty(0)).shape!=(rank,budget) or state.get('winningWitnessCounts',np.empty(0)).shape!=(rank,4):raise ValueError('Witness state differs')
    tag=hashlib.sha256(POLICY.encode()+np.asarray(asset['extraWitnesses'],dtype='<f8').tobytes()).hexdigest()
    return base_stream(p,asset,budget,state=state,readout=read_state,advance_fn=advance,transition_tag=tag)


def serve(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False);start=time.perf_counter();cpu=time.process_time()
    if sha(spec['asset']['path'])!=spec['asset']['sha256']:raise ValueError('Compiled witnesses differ')
    with np.load(spec['asset']['path'],allow_pickle=False) as f:asset={k:f[k] for k in f.files}
    m=spec['geometry'];d,heads,_=geometry(m);rows=[];states=[];winner_rows=[]
    for item in spec['ports']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Saved port differs')
        with np.load(item['path'],allow_pickle=False) as f:full={k:f[k] for k in f.files}
        n=len(full['ids']);rank=asset['landmarks'].shape[1]
        for method in ['anchor','witness']:
            tick=time.perf_counter();raw=np.empty((n,heads*d));bits=np.empty((n,heads*d),dtype=np.uint16);valid=np.empty((n,heads),dtype=bool)
            support=np.empty((n,heads,4),dtype=np.int64);indices=np.empty((n,heads),dtype=np.int64);hull=np.empty((n,heads));head_times=[];metadata=0
            for head in range(heads):
                a=head_asset(asset,m,head);a['extraWitnesses']=asset['extraWitnesses'][head];t=time.perf_counter()
                output,state=stream(packet(full,m,head),a,4) if method=='witness' else base_stream(packet(full,m,head),a,4,readout=read_state)
                head_times.append(time.perf_counter()-t);span=slice(head*d,(head+1)*d);raw[:,span]=output['raw'];bits[:,span]=output['rounded'];valid[:,head]=output['rounded_valid']
                support[:,head]=output['supportPositions'];indices[:,head]=output['nearestIndex'];hull[:,head]=output['hullViolation']
                metadata+=sum(len(str(state[k]).encode()) for k in ['assetIdentity','transitionIdentity'] if k in state)
                if method=='witness':winner_rows.append({'id':item['id'],'queryHead':head,'admittedKeys':int(state['count']),'counts':state['winningWitnessCounts'].tolist()})
                path=out/(item['id']+'-'+method+'-state'+str(head)+'.npz');np.savez(path,**{k:np.asarray(v) for k,v in state.items()})
                states.append({'id':item['id'],'method':method,'queryHead':head,'path':str(path),'sha256':sha(path)})
            path=out/(item['id']+'-'+method+'.npz');np.savez(path,raw=raw,rounded=bits,valid=valid,supportPositions=support,nearestIndex=indices,hullViolation=hull)
            rows.append({'id':item['id'],'method':method,'path':str(path),'sha256':sha(path),'headSeconds':head_times,'withSerializationSeconds':time.perf_counter()-tick,
                'logicalCoreBytes':heads*(core_bytes(d,rank,4)+(rank*4*8 if method=='witness' else 0)),'winnerCountDiagnosticBytes':heads*rank*4*8 if method=='witness' else 0,
                'rangeAndCounterDiagnosticBytes':heads*(2*d*8+16),'identityUTF8Bytes':metadata,'invalidHeadReads':int(np.count_nonzero(~valid)),
                'maximumHullViolation':float(np.nanmax(hull)),'admissionScoreMACs':int(full['mask'].sum())*heads*rank*d*(4 if method=='witness' else 1)})
            print('SOURCE_SUPPORT_BANK_SEALED',item['id'],method,flush=True)
    save(out/'prediction-seal.json',{'schema':'tce-source-witness-seal-v1','policy':POLICY,'specSHA256':sha(spec_path),'asset':spec['asset'],'ports':spec['ports'],
        'rows':rows,'states':states,'winningWitnessCounts':winner_rows,'runtimeAssetBytes':sum(a.nbytes for a in asset.values()),
        'wallSecondsAfterImports':time.perf_counter()-start,'cpuSecondsAfterImports':time.process_time()-cpu,'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'frameworkCalls':0,'originalDenseProjections':0,'scope':'Retrospective supplied operands; source-only fixed witnesses/ownstate prefixes. Both selectors seal before reference/consumer reads. Original anchor weights and centroid read unchanged.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['compile','serve']);p.add_argument('spec');p.add_argument('output');a=p.parse_args()
    (compile_asset if a.action=='compile' else serve)(a.spec,a.output)
