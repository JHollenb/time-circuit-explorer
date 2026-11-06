"""Saved-only diagonal softmax-gauge diagnosis, never a serving candidate."""
import argparse
import json
from pathlib import Path
import shutil
import time

from encoder_atoms import np,read,save,sha,identity,floats,words,guard
from context_kernel_diagnostics import load,distribution
from context_key_diagnostics import feature_probability,softmax,stats

HERE=Path(__file__).resolve().parent


def stage(root,parent):
    from mrun.client.api import Api
    from publish_context_statistics import checked_collection
    root=Path(root).resolve();parent=Path(parent).resolve();c,j,prior=checked_collection(parent)
    job=Api().json('GET','/api/jobs/'+c['jobID'])
    if job['state']!='succeeded' or job['payload_custody']!=c['custody'] or job['config']['requestSHA256']!=prior['requestSHA256']:
        raise ValueError('Authenticated parent differs')
    seal=read(parent/'result/prediction-seal.json');ref=read(parent/'result/reference-receipt.json')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    save(p/'source.json',{'ports':seal['ports'],'forecasts':seal['rows'],'reference':ref['rows'],'omega':seal['omega'],
         'parentJobID':job['job_id'],'parentSealSHA256':sha(parent/'result/prediction-seal.json')})
    for n in ['context_diagonal_diagnostics.py','context_key_diagnostics.py','context_kernel_diagnostics.py','encoder_atoms.py']:
        shutil.copyfile(HERE/n,p/n)
    request={'schema':'tce-context-diagonal-diagnostic-request-v1','parentJobID':job['job_id'],'parentRequestSHA256':prior['requestSHA256'],
        'scope':'Saved-only retrospective coordinate-moment diagnosis on the same14ports/40mask-selectedqueries. Whole held arrays fit centers/scales with lookahead. No serving, fresh-prompt or image claim.',
        'querySelection':'Sorted unique activeTokens//2, activeTokens-1, sequenceLength-1; unchanged mask-only rule.',
        'ranks':[16,32,64],'variants':['original','centered_scalar','diagonal_only_eps12','centered_diagonal_eps12','centered_diagonal_eps10'],
        'momentPolicy':'Query moments use all512positions; key moments use activekeys. Centered means use complete arrays. Scalar scale matches prior assay. Diagonal D_l=(max(B_l,eps*meanB)/max(A_l,eps*meanA))**.25; entire zero side uses unit D and a degenerate label.',
        'momentFloors':[1e-12,1e-10],'sourceHashes':{f.name:sha(f) for f in p.iterdir()}}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from response_study import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/'payload'/n)!=h:raise ValueError('Staged source changed')
    cmd=[v.replace('response_study.py','context_diagonal_diagnostics.py') for v in command()]
    job=launch(cmd,experiment='tce-context-diagonal-geometry-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu'},timeout_s=240,
        detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 saved-only geometry:14ports,40same mask-selected queries,3same ranks,5scalar/diagonal variants. Full-array fitted moments/lookahead, exact gauges, key/head diagnostics. No checkpoint/framework/GPU/new serving. Finite240s,no retries; parent arrays host-local.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def diagonal(x,y,eps):
    A=np.mean(x*x,axis=0);B=np.mean(y*y,axis=0)
    if not np.isfinite(A).all() or not np.isfinite(B).all() or eps<=0:raise ValueError('Invalid moment geometry')
    am=float(A.mean());bm=float(B.mean());floorA=eps*am;floorB=eps*bm
    aa=np.maximum(A,floorA);bb=np.maximum(B,floorB);degenerate=am==0 or bm==0
    D=np.ones_like(A) if degenerate else (bb/aa)**.25
    if not np.isfinite(D).all() or np.any(D<=0):raise ValueError('Noninvertible diagnostic transform')
    return D,{'epsilon':eps,'rawQueryMoments':A.tolist(),'rawKeyMoments':B.tolist(),
         'flooredQueryMoments':aa.tolist(),'flooredKeyMoments':bb.tolist(),
         'flooredQueryCoordinates':np.flatnonzero(A<floorA).tolist(),'flooredKeyCoordinates':np.flatnonzero(B<floorB).tolist(),
         'degenerateEntireSide':degenerate,'scale':D.tolist(),
         'operandNormObjectiveBefore':float(np.sum(A+B)),'operandNormObjectiveAfter':float(np.sum(D*D*A+B/(D*D)))}


def norm_panel(x,y,valid):
    norm=np.sum(x*x,1)[:,None]+np.sum(y*y,1)[None,:]+2*(x@y.T)
    raw=norm[valid];s=np.maximum(raw,0.);logvar=np.empty_like(s);large=s>40
    logvar[large]=s[large]+np.log1p(-np.exp(-s[large]))
    with np.errstate(divide='ignore'):logvar[~large]=np.log(np.expm1(s[~large]))
    return {'pairSumNormSquared':distribution(s),'negativeSquaredNormRoundoffCount':int(np.count_nonzero(raw<0)),
         'logRelativeKernelVarianceByRank':{str(r):distribution(logvar-np.log(r)) for r in [16,32,64]}}


def execute(root):
    guard();root=Path(root).resolve();r=read(root/'request.json');source=read(root/'source.json')
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/n)!=h:raise ValueError('Staged source changed')
    if sha(source['omega']['path'])!=source['omega']['sha256']:raise ValueError('Frozen omega changed')
    omega=np.load(source['omega']['path'],allow_pickle=False)
    out=root/'response-result';out.mkdir(exist_ok=False);start=time.perf_counter();rows=[];arrays={};geometry=[];exact_calls=0
    refs={f['id']:f for f in source['reference']};forecasts={(f['id'],f['rank']):f for f in source['forecasts']}
    for port in source['ports']:
        packet=load(port);q,k,v=[floats(packet[n]).astype(np.float64) for n in ['q','k','v']]
        if q.shape!=k.shape or k.shape!=v.shape or q.shape!=(512,128):raise ValueError('Saved port geometry differs')
        d=q.shape[1];x=q/d**.25;y=k/d**.25;mask=packet['mask'].astype(bool);active=int(mask.sum())
        if not active:raise ValueError('Expected nonempty original prompt')
        a=x.mean(0);b=y[mask].mean(0);xc=x-a;yc=y-b
        c=(np.mean(np.sum(yc[mask]**2,1))/np.mean(np.sum(xc**2,1)))**.25
        D0,g0=diagonal(x,y[mask],1e-12);D1,g1=diagonal(xc,yc[mask],1e-12);D2,g2=diagonal(xc,yc[mask],1e-10)
        variants={'original':(x,y,np.zeros(len(y))), 'centered_scalar':(xc*c,yc/c,y@a),
            'diagonal_only_eps12':(x*D0,y/D0,np.zeros(len(y))),
            'centered_diagonal_eps12':(xc*D1,yc/D1,y@a),'centered_diagonal_eps10':(xc*D2,yc/D2,y@a)}
        valid=np.tril(np.ones((len(x),len(y)),dtype=bool))&mask[None,:]
        geometry.append({'id':port['id'],'queryCenter':a.tolist(),'keyCenter':b.tolist(),'centeredScalar':float(c),
            'diagonalMoments':{'uncentered_eps12':g0,'centered_eps12':g1,'centered_eps10':g2},
            'conditioning':{name:norm_panel(u,w,valid) for name,(u,w,_) in variants.items()}})
        reference=load(refs[port['id']]);pred={rank:load(forecasts[port['id'],rank]) for rank in r['ranks']}
        for i in sorted({active//2,active-1,len(x)-1}):
            selected=np.flatnonzero(mask[:i+1]);ids=packet['ids'][selected];exact=softmax(y[selected]@x[i]);h=exact@v[selected];exact_calls+=1
            base=port['id']+'__q'+str(i);arrays[base+'__positions']=selected;arrays[base+'__exact']=exact
            for name,(u,w,bias) in variants.items():
                gauge=softmax(w[selected]@u[i]+bias[selected])
                for rank in r['ranks']:
                    p=feature_probability(u[i],w[selected],omega[:rank],bias[selected]);candidate=p@v[selected]
                    rounded=floats(words(candidate)).astype(float);rh=floats(reference['rounded'][i]).astype(float)
                    row={'id':port['id'],'query':i,'queryActive':bool(mask[i]),'allowedKeys':len(selected),'variant':name,'rank':rank,
                        'exact':stats(exact,selected,ids),'approximate':stats(p,selected,ids),
                        'totalVariation':float(.5*np.sum(np.abs(p-exact))),'argmaxMatches':bool(np.argmax(p)==np.argmax(exact)),
                        'gaugeProbabilityMaxAbs':float(np.max(np.abs(gauge-exact))),
                        'referenceHeadMaxAbs':float(np.max(np.abs(h-reference['raw'][i]))),
                        'headErrorRms':float(np.sqrt(np.mean((candidate-h)**2))),'headReferenceRms':float(np.sqrt(np.mean(h*h))),
                        'roundedHeadErrorRms':float(np.sqrt(np.mean((rounded-rh)**2))),'roundedHeadReferenceRms':float(np.sqrt(np.mean(rh*rh))),
                        'roundedReferenceWordMismatches':int(np.count_nonzero(words(h)!=reference['rounded'][i])),
                        'sealedFeatureHeadMaxAbs':float(np.max(np.abs(candidate-pred[rank]['raw'][i]))) if name=='original' else None}
                    rows.append(row);arrays[base+'__'+name+'__r'+str(rank)]=p
    np.savez_compressed(out/'key-probabilities.npz',**arrays)
    save(out/'report.json',{'schema':'tce-context-diagonal-diagnostics-v1','requestSHA256':r['requestSHA256'],
        'parentJobID':source['parentJobID'],'parentSealSHA256':source['parentSealSHA256'],'scope':r['scope'],'rows':rows,
        'retrospectiveGeometry':geometry,'probabilities':{'file':'key-probabilities.npz','sha256':sha(out/'key-probabilities.npz')},
        'workerSecondsAfterImports':time.perf_counter()-start,'selectedExactProbabilityCalls':exact_calls,'featureProbabilityCalls':len(rows),
        'neuralFrameworkCalls':0,'originalWeightBytesRead':0,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limitations':'Whole held-array fitted geometry with lookahead. Norm objective is not an attention-error oracle. Finite Gaussian approximation and selected-query diagnosis only; no source-frozen serving or images.'})
    with (out/'observations.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    shutil.copyfile(root/'request.json',out/'request.json');print('CONTEXT_DIAGONAL_DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
