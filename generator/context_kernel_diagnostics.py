"""Saved-only Gaussian-kernel conditioning and softmax-gauge diagnostics.

Centers/scales use complete held operand arrays retrospectively. They are capacity
diagnostics with lookahead, not frozen serving candidates or image predictions.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

from encoder_atoms import np,read,save,sha,identity,floats

HERE=Path(__file__).resolve().parent


def stage(root,parent):
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c=read(parent/'collection.json');r=read(parent/'request.json')
    job=Api().json('GET','/api/jobs/'+c['jobID'])
    if job['state']!='succeeded' or job['payload_custody']!=c['custody'] or job['config']['requestSHA256']!=r['requestSHA256']:
        raise ValueError('Context-state parent custody differs')
    for n in ['prediction-seal.json','reference-receipt.json']:
        if sha(parent/'result'/n)!=c['inventory']['files'][n]['sha256']:raise ValueError('Parent changed')
    seal=read(parent/'result/prediction-seal.json');reference=read(parent/'result/reference-receipt.json')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    save(p/'source.json',{'ports':seal['ports'],'forecasts':seal['rows'],'reference':reference['rows'],
                        'parentSealSHA256':sha(parent/'result/prediction-seal.json'),'parentJobID':job['job_id']})
    for n in ['context_kernel_diagnostics.py','encoder_atoms.py']:shutil.copyfile(HERE/n,p/n)
    request={'schema':'tce-context-kernel-diagnostic-request-v1','parentJobID':job['job_id'],'parentRequestSHA256':r['requestSHA256'],
             'scope':'Saved-only Gaussian feature variance and retrospective operand-fitted reciprocal scale/center capacity diagnostics. No new serving candidate, learned weights or model/framework evaluation.',
             'sourceHashes':{f.name:sha(f) for f in p.iterdir()}}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from response_study import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    cmd=[v.replace('response_study.py','context_kernel_diagnostics.py') for v in command()]
    job=launch(cmd,experiment='tce-context-kernel-conditioning-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
         config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu'},timeout_s=240,
         detach=True,retry_on_kill=False,preflight=True,guarded=True,
         note='Saved-only CPU1 kernel diagnosis:14sealed head ports, positiveRF variance risk, feature concentration, retrospective reciprocal/center conditioning and exact softmax-gauge checks. No original weights/framework/GPU/new candidate. Finite240s,no retries.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def load(ref):
    if sha(ref['path'])!=ref['sha256']:raise ValueError('Saved array changed')
    with np.load(ref['path'],allow_pickle=False) as d:return {k:d[k] for k in d.files}
def distribution(x):
    x=np.asarray(x,dtype=float).ravel();finite=x[np.isfinite(x)]
    if not len(finite):return {'n':len(x),'finite':0,'negativeInfinite':int(np.isneginf(x).sum())}
    return {'n':len(x),'finite':len(finite),'negativeInfinite':int(np.isneginf(x).sum()),
            **{n:float(v) for n,v in zip(['min','p05','median','p95','max'],np.quantile(finite,[0,.05,.5,.95,1]))}}


def head(scores,mask,v):
    n=len(mask);valid=np.tril(np.ones((n,n),dtype=bool))&mask.astype(bool)[None,:]
    score=scores.copy();score[~valid]=-np.inf;live=valid.any(1);p=np.zeros_like(score)
    a=score[live];p[live]=np.exp(a-a.max(1,keepdims=True));den=p.sum(1)
    p=np.divide(p,den[:,None],out=np.zeros_like(p),where=den[:,None]>0)
    return p,p@v


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');source=read(root/'source.json');out=root/'response-result';out.mkdir(exist_ok=False)
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/n)!=h:raise ValueError('Staged source changed')
    started=time.perf_counter();rows=[];concentration=[];reference={x['id']:x for x in source['reference']}
    for port in source['ports']:
        packet=load(port);q,k,v=[floats(packet[n]).astype(np.float64) for n in ['q','k','v']];d=q.shape[1];x=q/(d**.25);y=k/(d**.25)
        mask=packet['mask'].astype(bool);valid=np.tril(np.ones((len(x),len(x)),dtype=bool))&mask[None,:]
        xx=np.sum(x*x,1);yy=np.sum(y*y,1);dot=x@y.T;scale=(np.mean(yy[mask])/np.mean(xx))**.25
        a=x.mean(0);b=y[mask].mean(0)
        variants={'original':(x,y),'reciprocal_balanced':(x*scale,y/scale),'key_centered':(x,y-b),'query_key_centered':(x-a,y-b)}
        panels={}
        for name,(u,w) in variants.items():
            norm=np.sum(u*u,1)[:,None]+np.sum(w*w,1)[None,:]+2*(u@w.T)
            negative=int(np.count_nonzero(norm[valid]<0));s=np.maximum(norm[valid],0.)
            # log(Var(Khat)/E(Khat)^2) = log(expm1(||u+w||²)) - log(r).
            logvar=np.empty_like(s);large=s>40;logvar[large]=s[large]+np.log1p(-np.exp(-s[large]))
            with np.errstate(divide='ignore'):logvar[~large]=np.log(np.expm1(s[~large]))
            panels[name]={'pairSumNormSquared':distribution(s),'negativeSquaredNormRoundoffCount':negative,
                          'logRelativeKernelVarianceByRank':{str(rank):distribution(logvar-np.log(rank)) for rank in [16,32,64]}}
        prob,h0=head(dot,mask,v);gauge_score=(x-a)@(y-b).T+(a@y.T)[None,:];pg,hg=head(gauge_score,mask,v)
        prior=load(reference[port['id']]);rows.append({'id':port['id'],'activeTokens':int(mask.sum()),
             'queryNormSquared':distribution(xx),'activeKeyNormSquared':distribution(yy[mask]),'reciprocalScale':float(scale),
             'queryCenterNormSquared':float(a@a),'keyCenterNormSquared':float(b@b),'conditioning':panels,
             'originalReferenceMaxAbs':float(np.max(np.abs(h0-prior['raw']))),'gaugeProbabilityMaxAbs':float(np.max(np.abs(pg-prob))),
             'gaugeHeadMaxAbs':float(np.max(np.abs(hg-h0))),
             'scope':'Centers and scale use complete held operand arrays, with sequence lookahead. Retrospective conditioning capacity only, not a frozen predictor.'})
    for record in source['forecasts']:
        if record['rank']==0:continue
        prediction=load(record);concentration.append({'id':record['id'],'method':record['method'],
             'featureMaxMass':distribution(prediction['featureMaxMass']),'effectiveFeaturesExpEntropy':distribution(np.exp(prediction['featureEntropy'])),
             'logDenominator':distribution(prediction['logDenominator'])})
    save(out/'report.json',{'schema':'tce-context-kernel-conditioning-v1','requestSHA256':r['requestSHA256'],'parentJobID':source['parentJobID'],
         'parentSealSHA256':source['parentSealSHA256'],'scope':r['scope'],'rows':rows,'featureConcentration':concentration,
         'exactSavedOperandAttentionCalls':2*len(rows),'originalWeightBytesRead':0,'neuralFrameworkCalls':0,
         'workerSecondsAfterImports':time.perf_counter()-started,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed'})
    shutil.copyfile(root/'request.json',out/'request.json');print('CONTEXT_KERNEL_DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(__import__('json').dumps(value))
