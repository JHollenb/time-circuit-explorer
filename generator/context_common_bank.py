"""Transductive signed factors over one finite lexical query/key bank."""
import argparse
from pathlib import Path
import resource
import time

from encoder_atoms import np,read,save,sha,guard,floats,words


def bank(packets):
    qi={};ki={};qs=[];ks=[];vs=[];qids=[];kids=[];maps={};duplicates={'query':0,'key':0}
    for name,p in packets.items():
        qm=[];km=[]
        for i,(token,position) in enumerate(zip(p['ids'],p['positions'])):
            key=(int(token),int(position))
            if key not in qi:qi[key]=len(qs);qs.append(p['q'][i].copy());qids.append(key)
            else:
                duplicates['query']+=1
                if not np.array_equal(qs[qi[key]],p['q'][i]):raise ValueError('Duplicate query identity has different BF16 bytes')
            qm.append(qi[key])
            if not p['mask'][i]:km.append(-1);continue
            if key not in ki:ki[key]=len(ks);ks.append(p['k'][i].copy());vs.append(p['v'][i].copy());kids.append(key)
            else:
                duplicates['key']+=1
                if not np.array_equal(ks[ki[key]],p['k'][i]) or not np.array_equal(vs[ki[key]],p['v'][i]):raise ValueError('Duplicate key/value identity has different BF16 bytes')
            km.append(ki[key])
        maps[name]={'qmap':np.asarray(qm,dtype=np.int64),'kmap':np.asarray(km,dtype=np.int64),
            'v':p['v'].copy(),'mask':p['mask'].copy(),'positions':p['positions'].copy()}
    return {'q':np.stack(qs),'k':np.stack(ks),'v':np.stack(vs),'queryIdentities':np.asarray(qids,dtype=np.int64),
        'keyIdentities':np.asarray(kids,dtype=np.int64)},maps,duplicates


def probability(q,k):
    q,k=[floats(a).astype(np.float64) for a in [q,k]]
    if not np.isfinite(q).all() or not np.isfinite(k).all():raise ValueError('Nonfinite bank operand')
    scores=q@k.T/np.sqrt(q.shape[-1]);top=scores.max(1,keepdims=True);weights=np.exp(scores-top);P=weights/weights.sum(1,keepdims=True)
    return P,{'logitMin':float(scores.min()),'logitMax':float(scores.max()),'kernelUnderflowEntries':int(np.count_nonzero(weights==0)),
        'rowMassMaxError':float(np.max(np.abs(P.sum(1)-1))),'globalRowLogPartitions':(top[:,0]+np.log(weights.sum(1))).tolist()}


def stream(A,B,p,state=None,checkpoints=()):
    n=len(p['qmap']);rank=A.shape[1];v=floats(p['v']).astype(np.float64);d=v.shape[1]
    if A.ndim!=2 or B.ndim!=2 or B.shape[1]!=rank or not np.isfinite(A).all() or not np.isfinite(B).all() or not np.isfinite(v).all():raise ValueError('Invalid finite factor bank')
    if np.any(p['qmap']<0) or np.any(p['qmap']>=len(A)) or np.any(p['kmap'][p['mask'].astype(bool)]<0) or np.any(p['kmap'][p['mask'].astype(bool)]>=len(B)):raise ValueError('Bank address outside compiled domain')
    s={'z':np.zeros(rank),'M':np.zeros((rank,d)),'count':0,'steps':0} if state is None else {k:z.copy() if isinstance(z,np.ndarray) else z for k,z in state.items()}
    if int(p['positions'][0])!=s['steps']:raise ValueError('State clock differs')
    heads=[];denoms=[];valid=[];empty=[];snapshots=[];indices=[]
    for i in range(n):
        if p['mask'][i]:
            key=int(p['kmap'][i])
            if key<0:raise ValueError('Allowed key has no bank address')
            b=B[key];s['z']+=b;s['M']+=b[:,None]*v[i];s['count']+=1
        s['steps']+=1;a=A[int(p['qmap'][i])];den=float(a@s['z'])
        if not s['count']:h=np.zeros(d);ok=True
        elif den==0:h=np.full(d,np.nan);ok=False
        else:
            h=(a@s['M'])/den;ok=bool(np.isfinite(h).all())
        heads.append(h);denoms.append(den);valid.append(ok);empty.append(not s['count'])
        if int(p['positions'][i]) in checkpoints:snapshots.append({k:z.copy() if isinstance(z,np.ndarray) else z for k,z in s.items()});indices.append(int(p['positions'][i]))
    raw=np.stack(heads);valid=np.asarray(valid);rounded=np.zeros(raw.shape,dtype=np.uint16)
    rounded_valid=valid&np.all(np.abs(raw)<=np.finfo(np.float32).max,axis=1)
    if rounded_valid.any():
        rounded[rounded_valid]=words(raw[rounded_valid]);rounded_valid&=np.isfinite(floats(rounded)).all(1)
    output={'raw':raw,'valid':valid,'rounded':rounded,'rounded_valid':rounded_valid,'empty':np.asarray(empty),
        'denominator':np.asarray(denoms),'state_z':s['z'],'state_M':s['M'],'state_count':np.asarray(s['count']),
        'state_steps':np.asarray(s['steps']),'checkpoint_positions':np.asarray(indices,dtype=np.int64)}
    for key in ['z','M']:
        if snapshots:output['checkpoint_'+key]=np.stack([z[key] for z in snapshots])
    return output,s


def restrict(P,p):
    v=floats(p['v']).astype(np.float64);heads=[];rows=[]
    for i,q in enumerate(p['qmap']):
        selected=np.flatnonzero(p['mask'][:i+1]);w=P[q,p['kmap'][selected]];den=float(w.sum());absolute=float(np.abs(w).sum())
        if not len(selected):h=np.zeros(v.shape[1]);ok=True;negative=0.
        elif den==0:h=np.full(v.shape[1],np.nan);ok=False;negative=None
        else:
            normalized=w/den;h=normalized@v[selected];ok=bool(np.isfinite(h).all());negative=float(-normalized[normalized<0].sum())
        heads.append(h);rows.append({'denominator':den,'absoluteMass':absolute,'cancellationRatio':abs(den)/absolute if absolute else 0.,
            'negativeUnnormalizedMass':float(-w[w<0].sum()),'negativeProbabilityMass':negative,'negativeEntries':int(np.count_nonzero(w<0)),
            'nonpositiveDenominator':bool(den<=0),'valid':ok,'allowedKeys':len(selected)})
    return np.stack(heads),rows


def serve(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False)
    for key in ['factors','packets']:
        if sha(spec[key]['path'])!=spec[key]['sha256']:raise ValueError('Compiled bank asset changed')
    with np.load(spec['factors']['path'],allow_pickle=False) as f:A=f['A'];B=f['B']
    with np.load(spec['packets']['path'],allow_pickle=False) as f:packets={name:{key:f[name+'__'+key] for key in ['qmap','kmap','v','mask','positions']} for name in spec['ids']}
    start=time.perf_counter();cpu=time.process_time();rows=[]
    for name,p in packets.items():
        checkpoints=sorted({0,int(p['mask'].sum())-1,len(p['mask'])-1})
        for rank in spec['ranks']:
            t=time.perf_counter();output,state=stream(A[:,:rank],B[:,:rank],p,checkpoints=checkpoints);first=time.perf_counter()-t
            path=out/(name+'-rank'+str(rank)+'.npz');np.savez(path,**output);samples=[]
            for _ in range(spec['repeats']):
                t=time.perf_counter();stream(A[:,:rank],B[:,:rank],p,checkpoints=checkpoints);samples.append(time.perf_counter()-t)
            n=len(p['mask']);active=int(p['mask'].sum());d=p['v'].shape[1]
            rows.append({'id':name,'rank':rank,'method':'rank'+str(rank),'path':str(path),'sha256':sha(path),
                'firstSeconds':first,'warmSeconds':samples,'warmMedianSeconds':float(np.median(samples)),
                'coreStateBytes':rank*(d+1)*8+16,'logicalRankFactorBytes':(len(A)+len(B))*rank*8,
                'residentMaxRankFactorBytes':A.nbytes+B.nbytes,'residentPacketBytes':sum(z.nbytes for q in packets.values() for z in q.values()),
                'readNumeratorMACsPerCall':n*rank*d,'partitionMACsPerCall':n*rank,'stateValueMultipliesPerCall':active*rank*d,
                'stateAdditionsPerCall':active*rank*(d+1),'invalidReadRows':int(np.count_nonzero(~output['valid'])),
                'nonpositiveDenominators':int(np.count_nonzero(output['denominator']<=0)),'outputAndTraceBytes':sum(z.nbytes for z in output.values())})
    save(out/'prediction-seal.json',{'schema':'tce-common-bank-context-seal-v1','specSHA256':sha(spec_path),'rows':rows,'factors':spec['factors'],'packets':spec['packets'],
        'wallSecondsAfterInputs':time.perf_counter()-start,'cpuSecondsAfterInputs':time.process_time()-cpu,
        'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'frameworkCalls':0,'originalDenseProjections':0,'exactAttentionEvaluations':0,
        'scope':'Transductive signed finite-bank factors. Query coordinates are lookups fitted to all eight prompts, with no unseen-query feature rule. Actual prefix state; no clipping, exact/native fallback or image claim.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('output');a=p.parse_args();serve(a.spec,a.output)
