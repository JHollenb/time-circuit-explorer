"""Retrospective saved-operand key distributions; no serving/generalization claim."""
import argparse
from pathlib import Path
import shutil
import time

from encoder_atoms import np,read,save,sha,identity,floats
from context_kernel_diagnostics import load,distribution

HERE=Path(__file__).resolve().parent


def stage(root,parent):
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c=read(parent/'collection.json');r=read(parent/'request.json')
    job=Api().json('GET','/api/jobs/'+c['jobID'])
    if job['state']!='succeeded' or job['payload_custody']!=c['custody'] or job['config']['requestSHA256']!=r['requestSHA256']:
        raise ValueError('Parent custody differs')
    for n in ['prediction-seal.json','reference-receipt.json']:
        if sha(parent/'result'/n)!=c['inventory']['files'][n]['sha256']:raise ValueError('Parent changed')
    seal=read(parent/'result/prediction-seal.json');ref=read(parent/'result/reference-receipt.json')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    save(p/'source.json',{'ports':seal['ports'],'forecasts':seal['rows'],'reference':ref['rows'],'omega':seal['omega'],
                         'parentJobID':job['job_id'],'parentSealSHA256':sha(parent/'result/prediction-seal.json')})
    for n in ['context_key_diagnostics.py','context_kernel_diagnostics.py','encoder_atoms.py']:shutil.copyfile(HERE/n,p/n)
    request={'schema':'tce-context-key-diagnostic-request-v1','parentJobID':job['job_id'],'parentRequestSHA256':r['requestSHA256'],
             'scope':'Saved-only selected-query key-probability diagnosis and retrospective operand-fitted softmax-gauge capacity. No new serving/generalization/image claim.',
             'querySelection':'Sorted unique activeTokens//2, activeTokens-1, sequenceLength-1; mask-only, no error-based selection.',
             'variants':['original','balanced','key_centered','both_centered','both_centered_balanced'],
             'ranks':[16,32,64],'sourceHashes':{f.name:sha(f) for f in p.iterdir()}}
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
    cmd=[v.replace('response_study.py','context_key_diagnostics.py') for v in command()]
    job=launch(cmd,experiment='tce-context-key-diagnostics-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu'},timeout_s=240,
        detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 saved-only key probabilities:14sealed ports,3mask-selected queries each,3fixed ranks,5conditioning variants. Operand-fitted retrospective centers/lookahead; no serving claim, original weights,framework or GPU. Finite240s,no retries; host-local parent.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def softmax(logs):
    weights=np.exp(logs-np.max(logs));return weights/weights.sum()


def feature_probability(x,y,omega,key_bias):
    lq=omega@x-.5*np.dot(x,x)
    lk=y@omega.T-.5*np.sum(y*y,1)[:,None]+key_bias[:,None]
    terms=lk+lq[None,:];top=terms.max(1)
    return softmax(top+np.log(np.exp(terms-top[:,None]).sum(1)))


def stats(p,positions,ids):
    j=int(np.argmax(p));positive=p>0
    return {'maxKeyMass':float(p[j]),'entropy':float(-np.sum(p[positive]*np.log(p[positive]))),
            'argmaxPosition':int(positions[j]),'argmaxTokenID':int(ids[j])}


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');source=read(root/'source.json');out=root/'response-result';out.mkdir(exist_ok=False)
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/n)!=h:raise ValueError('Staged source changed')
    if sha(source['omega']['path'])!=source['omega']['sha256']:raise ValueError('Frozen omega changed')
    omega=np.load(source['omega']['path'],allow_pickle=False)
    forecasts={(f['id'],f['rank']):f for f in source['forecasts']};references={f['id']:f for f in source['reference']}
    start=time.perf_counter();rows=[];arrays={};geometry=[];exact_calls=0
    for port in source['ports']:
        packet=load(port);q,k,v=[floats(packet[n]).astype(np.float64) for n in ['q','k','v']];d=q.shape[1];x=q/d**.25;y=k/d**.25
        mask=packet['mask'].astype(bool);active=int(mask.sum());a=x.mean(0);b=y[mask].mean(0)
        c=(np.mean(np.sum(y[mask]**2,1))/np.mean(np.sum(x*x,1)))**.25
        cc=(np.mean(np.sum((y[mask]-b)**2,1))/np.mean(np.sum((x-a)**2,1)))**.25
        geometry.append({'id':port['id'],'a':a.tolist(),'b':b.tolist(),'scale':float(c),'centeredScale':float(cc)})
        variants={'original':(x,y,np.zeros(len(y))),'balanced':(x*c,y/c,np.zeros(len(y))),
                  'key_centered':(x,y-b,np.zeros(len(y))),'both_centered':(x-a,y-b,y@a),
                  'both_centered_balanced':((x-a)*cc,(y-b)/cc,y@a)}
        reference=load(references[port['id']]);pred={rank:load(forecasts[port['id'],rank]) for rank in r['ranks']}
        for i in sorted({active//2,active-1,len(x)-1}):
            exact_calls+=1
            selected=np.flatnonzero(mask[:i+1]);ids=packet['ids'][selected];exact=softmax(y[selected]@x[i]);h=exact@v[selected]
            base=port['id']+'__q'+str(i);arrays[base+'__positions']=selected;arrays[base+'__exact']=exact
            for name,(u,w,bias) in variants.items():
                gauge=softmax(w[selected]@u[i]+bias[selected])
                for rank in r['ranks']:
                    p=feature_probability(u[i],w[selected],omega[:rank],bias[selected]);candidate=p@v[selected]
                    row={'id':port['id'],'query':i,'queryActive':bool(mask[i]),'allowedKeys':len(selected),'variant':name,'rank':rank,
                         'exact':stats(exact,selected,ids),'approximate':stats(p,selected,ids),
                         'totalVariation':float(.5*np.sum(np.abs(p-exact))),'argmaxMatches':bool(np.argmax(p)==np.argmax(exact)),
                         'gaugeProbabilityMaxAbs':float(np.max(np.abs(gauge-exact))),
                         'referenceHeadMaxAbs':float(np.max(np.abs(h-reference['raw'][i]))),
                         'headErrorRms':float(np.sqrt(np.mean((candidate-h)**2))),'headReferenceRms':float(np.sqrt(np.mean(h*h))),
                         'sealedFeatureHeadMaxAbs':float(np.max(np.abs(candidate-pred[rank]['raw'][i]))) if name=='original' else None}
                    rows.append(row);arrays[base+'__'+name+'__r'+str(rank)]=p
    np.savez_compressed(out/'key-probabilities.npz',**arrays)
    save(out/'report.json',{'schema':'tce-context-key-diagnostics-v1','requestSHA256':r['requestSHA256'],'parentJobID':source['parentJobID'],
        'parentSealSHA256':source['parentSealSHA256'],'scope':r['scope'],'rows':rows,'retrospectiveGeometry':geometry,
        'probabilities':{'file':'key-probabilities.npz','sha256':sha(out/'key-probabilities.npz')},
        'workerSecondsAfterImports':time.perf_counter()-start,'originalWeightBytesRead':0,'neuralFrameworkCalls':0,
        'selectedExactProbabilityCalls':exact_calls,
        'featureProbabilityCalls':len(rows),'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limitations':'Centers/scales fitted to complete held arrays, including future/padding queries. This is capacity/conditioning diagnosis, not frozen online state or prompt generalization. Variance of unnormalized kernel is not variance of normalized head.'})
    with (out/'observations.jsonl').open('x') as f:
        import json
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    shutil.copyfile(root/'request.json',out/'request.json');print('CONTEXT_KEY_DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(__import__('json').dumps(value))
