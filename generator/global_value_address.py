"""Saved-only matched value-delta address controls for the global-head result."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from response_study import read,save,sha,identity,command,collect

HERE=Path(__file__).resolve().parent
SCOPE=('Retrospective saved-only value-delta address diagnostic, fixed queries/keys in joint2head0. '
       'Permute change vectors while retaining baseline carriers; show both smooth and BF16-admitted deltas. '
       'No new weights or native calls; no semantic-role or arbitrary-prompt claim.')


def stage(root,parent):
    from mrun.client.api import Api
    root,parent=Path(root).resolve(),Path(parent).resolve()
    prior,custody=read(parent/'request.json'),read(parent/'collection.json')
    job=Api().json('GET','/api/jobs/'+custody['jobID'])
    if job['state']!='succeeded' or job['config']['requestSHA256']!=prior['requestSHA256'] or job['payload_custody']!=custody['custody']:
        raise ValueError('Parent not authenticated')
    inventory=custody['inventory']['files']
    for name,item in inventory.items():
        path=parent/'result'/name
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('Parent collection changed')
    remote=job['result']['diagnostics']['cwd']+'/response-result'
    def ref(name):return {'path':remote+'/'+name,'sha256':inventory[name]['sha256']}
    report=read(parent/'result/report.json');p=report['serving']['rows'];primary=[r for r in p if r['method']=='all']
    native=[r for r in report['references'] if r['arm']=='nativeWrite']
    root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    save(payload/'predict.json',{'baseline':ref(primary[0]['path']),'changed':ref(primary[1]['path']),
         'fixed':ref('fixed-head.npz'),'cache':ref('attention-cache.npz'),'sourceRoot':remote,'scope':SCOPE})
    save(payload/'evaluation.json',{'nativeBase':ref(native[0]['path']),'nativeChanged':ref(native[1]['path']),
         'correctValuesOnly':ref(next(r for r in p if r['method']=='values_only')['path'])})
    for name in ['global_value_address.py','global_attention.py','state_receiver.py','mlp_polynomial.py','response_chart.py','prompt_closure.py','response_study.py']:
        shutil.copyfile(HERE/name,payload/name)
    request={'schema':'tce-global-value-address-request-v1','scope':SCOPE,'textRows':report['compile']['textRows'],
             'parent':{'jobID':job['job_id'],'requestSHA256':prior['requestSHA256'],'report':ref('report.json'),
                       'collectionSHA256':sha(parent/'collection.json'),'payloadCustody':job['payload_custody']},
             'sourceHashes':{p.name:sha(p) for p in payload.iterdir()}}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(payload/'request.json',request)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    root=Path(root).resolve();request=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    cmd=[v.replace('response_study.py','global_value_address.py') for v in command()]
    job=launch(cmd,experiment='tce-global-value-address-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
               config={'schema':request['schema'],'requestSHA256':request['requestSHA256'],'backend':'numpy','device':'cpu','newNeuralEvaluations':0,'weightBytesRead':0},
               timeout_s=120,detach=True,retry_on_kill=False,preflight=True,guarded=True,
               note='Saved-only matched value-delta address controls: preserve baseline, correct/cyclic/reverse/seeded-shuffle/zero deltas, smooth+BF16. Fixed1536query/key head0; seal then score native-derived head references. CPU1,no weights/no nativeNN/GPU. Finite120s,no retries.')
    result={'jobID':str(job),'requestSHA256':request['requestSHA256']};save(root/'job.json',result);return result


def predict(spec_path,output):
    from global_attention import np,load_bundle,merge,quantize,audit_reads
    spec,output=read(spec_path),Path(output)
    paths=[spec_path,output/'candidates.npz',output/'probabilities.npz',output/'candidate-seal.json']+[spec[k]['path'] for k in ['baseline','changed','fixed','cache']]
    opened=audit_reads(paths,[spec['sourceRoot'],str(Path(spec_path).parent)],serving=True)
    base,current,fixed,cache=[load_bundle(spec[k]) for k in ['baseline','changed','fixed','cache']]
    selected=cache['changed'];fields={n:base['changed_'+n] for n in ['q','k','v']};complete=merge(fixed,selected,fields)
    q,k,v=[complete[n].astype(np.float64) for n in ['q','k','v']]
    weights=np.empty((len(q),len(selected)));h0=np.empty_like(q)
    for first in range(0,len(q),64):
        score=q[first:first+64]@k.T/np.sqrt(float(q.shape[-1]));score-=score.max(axis=-1,keepdims=True)
        p=np.exp(score);p/=p.sum(axis=-1,keepdims=True)
        weights[first:first+64]=p[:,selected];h0[first:first+64]=p@v
    v0=base['changed_v'].astype(np.float64);delta=current['changed_v'].astype(np.float64)-v0
    m=len(selected);rng=np.random.default_rng(6301)
    orders={'correct':np.arange(m),'cyclic':np.roll(np.arange(m),1),'reverse':np.arange(m)[::-1],'shuffle':rng.permutation(m),'zero':np.arange(m)}
    rows,raws,roundeds,deltas=[],[],[],[]
    for method,order in orders.items():
        proposal=np.zeros_like(delta) if method=='zero' else delta[order]
        for profile in ['smooth','bf16']:
            actual=proposal if profile=='smooth' else quantize(v0+proposal).astype(np.float64)-v0
            head=h0+weights@actual
            rows.append({'method':method,'profile':profile,'order':order.tolist(),'proposedDeltaPower':float(np.mean(proposal*proposal)),
                         'appliedDeltaPower':float(np.mean(actual*actual)), 'admissionDisplacementRms':float(np.sqrt(np.mean((actual-proposal)**2)))})
            raws.append(head);roundeds.append(quantize(head));deltas.append(actual)
    np.savez(output/'candidates.npz',raw=np.stack(raws),rounded=np.stack(roundeds),appliedDelta=np.stack(deltas),baseline=h0)
    np.savez(output/'probabilities.npz',selected=selected,weights=weights,v0=v0,delta=delta)
    save(output/'candidate-seal.json',{'schema':'tce-global-value-address-seal-v1','specSHA256':sha(spec_path),'rows':rows,
         'candidatesSHA256':sha(output/'candidates.npz'),'probabilitiesSHA256':sha(output/'probabilities.npz'),
         'opened':sorted(opened),'scope':SCOPE})
    print('VALUE_ADDRESS_CANDIDATES_SEALED',flush=True)


def execute(root):
    from global_attention import np,load_bundle,quantize
    from mlp_polynomial import rms,ratio,peak_mib
    root=Path(root).resolve();request=read(root/'request.json')
    if identity({k:v for k,v in request.items() if k!='requestSHA256'})!=request['requestSHA256']:raise ValueError('Request changed')
    for name,digest in request['sourceHashes'].items():
        if sha(root/name)!=digest:raise ValueError('Analysis source changed')
    output=root/'response-result';output.mkdir(exist_ok=False);started,cpu=time.perf_counter(),time.process_time()
    subprocess.run([sys.executable,str(root/'global_value_address.py'),'predict',str(root/'predict.json'),'--output',str(output)],check=True)
    seal=read(output/'candidate-seal.json');seal_hash=sha(output/'candidate-seal.json')
    candidates=load_bundle({'path':str(output/'candidates.npz'),'sha256':seal['candidatesSHA256']})
    weights=load_bundle({'path':str(output/'probabilities.npz'),'sha256':seal['probabilitiesSHA256']})
    evaluation=read(root/'evaluation.json');a,b,correct=[load_bundle(evaluation[k]) for k in ['nativeBase','nativeChanged','correctValuesOnly']]
    n=len(candidates['baseline']);regions={'all':np.arange(n),'text':np.arange(request['textRows']),
            'changedImage':weights['selected'],'fixedImage':np.setdiff1d(np.arange(request['textRows'],n),weights['selected'])}
    scores=[]
    for i,item in enumerate(seal['rows']):
        for view in ['raw','rounded']:
            baseline=candidates['baseline'] if view=='raw' else quantize(candidates['baseline']).astype(np.float64)
            delta=candidates[view][i].astype(np.float64)-baseline
            target=b[view].astype(np.float64)-a[view]
            correct_delta=correct[view].astype(np.float64)-baseline
            for name,index in regions.items():
                value,truth,reference=delta[index],target[index],correct_delta[index]
                scores.append({'method':item['method'],'profile':item['profile'],'view':view,'region':name,
                               'effectRms':rms(value),'nativeDerivedEffectRms':rms(truth),
                               'errorOverNativeEffect':ratio(rms(value-truth),rms(truth)),
                               'errorOverCorrectValueEffect':ratio(rms(value-reference),rms(reference)),
                               'signedCosineToNativeEffect':ratio(float(np.mean(value*truth)),rms(value)*rms(truth))})
    source_power=float(np.mean(weights['delta']**2))
    if sha(output/'candidate-seal.json')!=seal_hash or sha(output/'candidates.npz')!=seal['candidatesSHA256']:raise ValueError('Sealed candidates mutated')
    save(output/'report.json',{'schema':'tce-global-value-address-report-v1','requestSHA256':request['requestSHA256'],'scope':SCOPE,
         'parent':request['parent'],'predictionSealSHA256':seal_hash,'serving':seal,'scores':scores,
         'sourceDeltaPower':source_power,'correctValueReadMaxAbs':float(np.max(np.abs(candidates['raw'][0]-correct['raw']))),
         'selectedAttentionMass':{'min':float(weights['weights'].sum(axis=-1).min()),'max':float(weights['weights'].sum(axis=-1).max()),'mean':float(weights['weights'].sum(axis=-1).mean())},
         'workerWallSeconds':time.perf_counter()-started,'workerCPUSeconds':time.process_time()-cpu,'peakRSSMiB':peak_mib(),
         'newNeuralFrameworkCalls':0,'newOriginalDenseProjections':0,'weightBytesRead':0,
         'mechanics_status':'complete','terminal_status':'unassessed'})
    print('VALUE_ADDRESS_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','execute','collect','predict']);parser.add_argument('run',type=Path);parser.add_argument('--parent',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.action=='stage':result=stage(args.run,args.parent)
    elif args.action=='predict':result=predict(args.run,args.output)
    else:result=globals()[args.action](args.run)
    if result is not None:print(json.dumps(result))
