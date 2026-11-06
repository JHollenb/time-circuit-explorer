"""Causal top-four retrieval with outward score bounds and explicit fallback.

The certificate assumes ordinary IEEE binary64 round-to-nearest dot arithmetic
with at most 4*d rounded operations. Nonzero input magnitudes must lie in
[2**-400, 2**400]; this keeps products, cancellation quanta and norm arithmetic
normal/finite for d=128. The interval outward operations may bracket zero with
subnormal endpoints. Unsupported input ranges use the original full GEMV.

A partial or completed block reduction differs from the real dot by <=eta*A,
eta=gamma_(4*d), A>=||x||||u||. The old GEMV has the same error allowance.
Thus partial+tail+2eta*A bounds its returned score, and f+-2eta*A encloses
that score when f completes all coordinates in the new reduction order.
"""
import argparse
from collections import Counter
import ctypes
from functools import cmp_to_key
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time

from encoder_atoms import np,read,save,sha,guard,floats
from attention_bank import geometry,head_asset
from context_transport import nearest,rounded
from centroid_tilt_study import direct_centroid
from selector_objective import select

LOW=2.**-400;HIGH=2.**400
FALLBACK={0:'none',1:'short-prefix',2:'unsupported-range-or-bound',3:'overlapping-membership-or-order'}


def execution_profile():
    mode=None
    try:
        f=ctypes.CDLL(None).fegetround;f.restype=ctypes.c_int;f.argtypes=[];mode=f()
    except (AttributeError,OSError):pass
    supported=sys.platform in ['linux','darwin'] and platform.machine().lower() in ['x86_64','amd64','arm64','aarch64'] and mode==0
    return {'platform':sys.platform,'machine':platform.machine(),'fegetround':mode,'roundToNearestQualified':supported,
        'python':sys.version,'numpy':np.__version__,'threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
        'assumption':'Ordinary IEEEbinary64 dot/sqrt arithmetic with stated operation/error bounds; profile qualification is not independent BLAS/kernel formal verification.'}


def up(x):return math.nextafter(float(x),math.inf)
def down(x):return math.nextafter(float(x),-math.inf)
def product_up(a,b):return up(a*b) if a and b else 0.


def eligible(a):
    a=np.asarray(a);z=np.abs(a)
    return bool(np.isfinite(a).all() and np.all((z==0)|((z>=LOW)&(z<=HIGH))))


def tail_norms(a,order,block,counts,label):
    """Positive reverse accumulation; each rounded square/sum/root goes up."""
    d=len(a);out=np.zeros(d//block+1);total=0.
    for j in range(d-1,-1,-1):
        z=float(a[order[j]]);square=z*z;counts[label+'NormSquareProducts']+=1
        if square:square=up(square);counts[label+'NormNextUpCalls']+=1
        total=total+square;counts[label+'NormAdds']+=1
        if total:total=up(total);counts[label+'NormNextUpCalls']+=1
        if j%block==0:
            out[j//block]=up(math.sqrt(total)) if total else 0.;counts[label+'NormRoots']+=1
            if total:counts[label+'NormNextUpCalls']+=1
    if not np.isfinite(out).all():raise ArithmeticError('Nonfinite norm bound')
    return out


def ranking(items,key,counts):
    counts['sortCalls']+=1;counts['sortInputItems']+=len(items)
    def compare(a,b):
        counts['sortComparatorCalls']+=1;x,y=key(a),key(b)
        return int(x>y)-int(x<y)
    return sorted(items,key=cmp_to_key(compare))


class CausalIndex:
    def __init__(self,capacity,d,order,block=16,certified=True):
        order=np.asarray(order,dtype=np.int64)
        if d%block or not np.array_equal(np.sort(order),np.arange(d)) or d>128:raise ValueError('Invalid certificate geometry')
        if np.finfo(np.float64).eps!=2.**-52:raise ValueError('IEEE binary64 required')
        self.d=d;self.block=block;self.order=order;self.certified=certified;self.count=0;self.profile_supported=execution_profile()['roundToNearestQualified']
        self.keys=np.zeros((capacity,d));self.values=np.zeros((capacity,d));self.positions=np.full(capacity,-1,dtype=np.int64)
        self.norms=np.zeros((capacity,d//block+1)) if certified else None;self.safe=np.ones(capacity,dtype=bool)
        self.counts=Counter();n=4*d;eps=2.**-53;self.eta=up(n*eps/(1-n*eps))

    def append(self,key,value,position):
        if self.count and position<=self.positions[self.count-1]:raise ValueError('Noncausal admission')
        i=self.count
        if i>=len(self.keys) or np.shape(key)!=(self.d,) or np.shape(value)!=(self.d,):raise ValueError('Index admission shape')
        if not np.isfinite(key).all() or not np.isfinite(value).all():raise ValueError('Nonfinite index payload')
        self.keys[i]=key;self.values[i]=value;self.positions[i]=position
        if self.certified:
            self.counts['keyRangeCoordinatesChecked']+=self.d;self.safe[i]=eligible(key)
            if self.safe[i]:
                try:self.norms[i]=tail_norms(key,self.order,self.block,self.counts,'key')
                except ArithmeticError:self.safe[i]=False
        self.count+=1;self.counts['keyAdmissions']+=1

    def full(self,x,m,counts,reason):
        scores=self.keys[:self.count]@x
        if not np.isfinite(scores).all():raise ValueError('Nonfinite original scores')
        counts['wholePrefixScoreProducts']+=self.count*self.d;counts['wholePrefixCalls']+=1
        if reason:counts['fallbackQueries']+=1
        counts['fallbackReason']=reason
        # Same whole-prefix GEMV, same coordinate/layout order and tie rule.
        return select(scores,self.positions[:self.count],m)

    def query(self,x,m=4):
        if np.shape(x)!=(self.d,) or m<1:raise ValueError('Invalid query')
        counts=Counter(queries=1,eligibleKeyOccurrences=self.count,fullScanScoreProducts=self.count*self.d,fallbackReason=0)
        if not self.count:return np.empty(0,dtype=np.int64),counts
        if not self.certified:return self.full(x,m,counts,0),counts
        if self.count<=m:return self.full(x,m,counts,1),counts
        counts['queryRangeCoordinatesChecked']+=self.d
        if not self.profile_supported or not eligible(x) or not self.safe[:self.count].all():return self.full(x,m,counts,2),counts
        try:
            qnorm=tail_norms(x,self.order,self.block,counts,'query');entries={}
            def radius(j):
                counts['normProductBounds']+=1;counts['radiusProducts']+=1
                return product_up(2*self.eta,product_up(qnorm[0],self.norms[j,0]))
            def record(j,f,r):
                counts['intervalEndpoints']+=2;entries[j]=(f,down(f-r),up(f+r))
                if not all(math.isfinite(z) for z in entries[j]):raise ArithmeticError('Invalid interval')
            for j in range(m):
                f=float(np.dot(self.keys[j],x));counts['initialCandidateFullDotProducts']+=self.d;counts['initialCandidateFullDots']+=1
                record(j,f,radius(j))
            for j in range(m,self.count):
                ranked=ranking(list(entries),lambda k:(-entries[k][1],int(self.positions[k])),counts);cut=entries[ranked[m-1]][1]
                r=radius(j);partial=0.;pruned=False
                for start in range(0,self.d,self.block):
                    coords=self.order[start:start+self.block];partial+=float(np.dot(x[coords],self.keys[j,coords]))
                    counts['progressiveScoreProducts']+=self.block;counts['progressiveBlockCalls']+=1;counts['partialAccumulatorAdds']+=1
                    end=start+self.block;tail=product_up(qnorm[end//self.block],self.norms[j,end//self.block]);counts['tailNormProductBounds']+=1
                    upper=up(up(partial+tail)+r);counts['upperBoundAdds']+=2;counts['upperBoundNextUpCalls']+=2;counts['pruneComparisons']+=1
                    if not math.isfinite(upper):raise ArithmeticError('Invalid score bound')
                    if upper<cut:
                        counts['prunedKeys']+=1;counts['scoreCoordinatesSkippedByBounds']+=self.d-end;pruned=True;break
                if not pruned:
                    # f is the completed progressive dot, not an uncharged extra dot.
                    record(j,partial,r);counts['completedProgressiveCandidates']+=1
            ranked=ranking(list(entries),lambda k:(-entries[k][0],int(self.positions[k])),counts);chosen=ranked[:m]
            for a,b in zip(chosen,chosen[1:]):
                counts['finalIntervalComparisons']+=1
                if entries[a][1]<=entries[b][2]:return self.full(x,m,counts,3),counts
            for j in ranked[m:]:
                counts['finalIntervalComparisons']+=1
                if entries[chosen[-1]][1]<=entries[j][2]:return self.full(x,m,counts,3),counts
            counts['certifiedQueries']+=1
            return np.asarray(chosen,dtype=np.int64),counts
        except (ArithmeticError,FloatingPointError,OverflowError):return self.full(x,m,counts,2),counts

    def payload_bytes(self,used=True):
        n=self.count if used else len(self.keys)
        arrays=[self.keys,self.values,self.positions,self.safe]+([self.norms] if self.norms is not None else [])
        return sum(a[:n].nbytes for a in arrays)


def serve(spec_path,out,mode='certified'):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False);start=time.perf_counter();cpu=time.process_time()
    if mode not in ['certified','full']:raise ValueError('Unknown selector')
    if sha(spec['asset']['path'])!=spec['asset']['sha256']:raise ValueError('Source geometry differs')
    with np.load(spec['asset']['path'],allow_pickle=False) as f:asset={k:f[k] for k in ['landmarks','metricRoots','keyCenters','covariances']}
    m=spec['geometry'];d,heads,groups=geometry(m);orders=np.stack([np.lexsort((np.arange(d),-np.diag(c))) for c in asset['covariances']])
    rows=[];states=[];timings=[]
    for item in spec['ports']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Saved port differs')
        with np.load(item['path'],allow_pickle=False) as f:port={k:f[k] for k in f.files}
        n=len(port['ids']);mask=port['mask'];tick=time.perf_counter()
        if not np.array_equal(port['positions'],np.arange(n)) or np.any((mask!=0)&(mask!=1)):raise ValueError('Invalid prefix clock/mask')
        q=floats(port['qpos']).astype(float)/d**.25;k=floats(port['kpos']).astype(float)/d**.25;v=floats(port['v']).astype(float)
        routes=np.stack([nearest(q[:,h*d:(h+1)*d],head_asset(asset,m,h))[0] for h in range(heads)],axis=1);route_s=time.perf_counter()-tick
        indexes=[CausalIndex(n,d,orders[g],spec['block'],mode=='certified') for g in range(groups)]
        raw=np.zeros((n,heads*d));support=np.full((n,heads,spec['budget']),-1,dtype=np.int64);counters={};admission_s=0.;selection_s=0.;reader_s=0.
        for i in range(n):
            t=time.perf_counter()
            if mask[i]:
                for g,index in enumerate(indexes):index.append(k[i,g*d:(g+1)*d]-asset['keyCenters'][g],v[i,g*d:(g+1)*d],i)
            admission_s+=time.perf_counter()-t
            for h in range(heads):
                index=indexes[h//(heads//groups)];span=slice(h*d,(h+1)*d);x=q[i,span];t=time.perf_counter();chosen,count=index.query(x,spec['budget']);selection_s+=time.perf_counter()-t
                count['anchorScoreProducts']=index.count*d
                for name,value in count.items():
                    if name not in counters:counters[name]=np.zeros((n,heads),dtype=np.int64)
                    counters[name][i,h]=value
                if not index.count:continue
                support[i,h,:len(chosen)]=index.positions[chosen];t=time.perf_counter();U=index.keys[:index.count];V=index.values[:index.count];ell=asset['landmarks'][h,routes[i,h]]
                raw[i,span]=direct_centroid(U@ell,U,V,chosen,x-ell);reader_s+=time.perf_counter()-t
            if i%128==127:print('RETRIEVAL_CLOCK',mode,item['id'],i,flush=True)
        bits,valid=rounded(raw,np.isfinite(raw).all(1));path=out/(item['id']+'-'+mode+'.npz')
        np.savez(path,raw=raw,rounded=bits,valid=valid,supportPositions=support,nearestIndex=routes)
        rows.append({'id':item['id'],'method':mode,'path':str(path),'sha256':sha(path),'invalidRows':int(np.count_nonzero(~valid))})
        count_path=out/(item['id']+'-counts.npz');np.savez(count_path,**counters);states.append({'id':item['id'],'kind':'query-counters','path':str(count_path),'sha256':sha(count_path)})
        admission_counts=Counter()
        for g,index in enumerate(indexes):
            admission_counts.update(index.counts);path=out/(item['id']+'-index'+str(g)+'.npz');num=index.count
            arrays={'keys':index.keys[:num],'values':index.values[:num],'positions':index.positions[:num],'safe':index.safe[:num],'order':index.order}
            if index.norms is not None:arrays['tailNormUpper']=index.norms[:num]
            np.savez(path,**arrays);states.append({'id':item['id'],'kind':'final-index','kvGroup':g,'path':str(path),'sha256':sha(path),'admittedKeys':num})
        total={name:int(a.sum()) for name,a in counters.items() if name!='fallbackReason'}
        score_products=sum((counters.get(key,np.zeros((n,heads),dtype=np.int64)) for key in ['initialCandidateFullDotProducts','progressiveScoreProducts','wholePrefixScoreProducts']))
        total['actualScoreProducts']=int(score_products.sum());total['scoreProductRatio']=total['actualScoreProducts']/total['fullScanScoreProducts'] if total['fullScanScoreProducts'] else None
        total['certifiedButNoScoreProductSavings']=int(np.count_nonzero((counters.get('certifiedQueries',np.zeros((n,heads)))>0)&(score_products>=counters['fullScanScoreProducts'])))
        total['fallbackReasons']={FALLBACK[code]:int(np.count_nonzero(counters['fallbackReason']==code)) for code in FALLBACK}
        timings.append({'id':item['id'],'routeAndOperandSeconds':route_s,'admissionSeconds':admission_s,'selectionSeconds':selection_s,'anchorAndCentroidReadSeconds':reader_s,
            'queryCounters':total,'admissionCounters':dict(admission_counts),'usedIndexPayloadBytes':sum(x.payload_bytes() for x in indexes),
            'allocatedIndexPayloadBytes':sum(x.payload_bytes(False) for x in indexes),'sourceOrderPayloadBytes':orders.nbytes})
    save(out/'prediction-seal.json',{'schema':'tce-certified-retrieval-seal-v1','mode':mode,'specSHA256':sha(spec_path),'asset':spec['asset'],'ports':spec['ports'],
        'rows':rows,'states':states,'timings':timings,'coordinateOrders':orders.tolist(),'runtimeSourceGeometryBytes':sum(a.nbytes for a in asset.values()),
        'wallSecondsAfterImports':time.perf_counter()-start,'cpuSecondsAfterImports':time.process_time()-cpu,'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'executionProfile':execution_profile(),
        'frameworkCalls':0,'originalDenseProjections':0,'scope':'Selection-cost assay with full causal key/value index and unchanged fullprefix centroid read. Not compact ownstate encoder or strictzero-neural image.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['certified','full']);p.add_argument('spec');p.add_argument('output');a=p.parse_args();serve(a.spec,a.output,a.mode)
