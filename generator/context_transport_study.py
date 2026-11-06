"""Fresh raw-text query transport, sealed before exact/contextual references."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from encoder_atoms import np,Prefix,read,save,sha,identity,guard,floats
from context_transport import METHODS
from context_statistics_study import exact_head,summary,rowrms

HERE=Path(__file__).resolve().parent
PROMPTS=[
 {'id':'compass-inside','prompt':'A copper compass rests inside a clear glass case beside a black feather.'},
 {'id':'compass-outside','prompt':'A copper compass rests outside a clear glass case beside a black feather.'},
 {'id':'balloon-pink','prompt':'A pink balloon floats beside a brown suitcase under a striped awning.'},
 {'id':'balloon-brown','prompt':'A brown balloon floats beside a pink suitcase under a striped awning.'},
 {'id':'tower-above','prompt':'青い鳥が白い塔の上で歌う。緑の庭と赤い門。'},
 {'id':'tower-below','prompt':'青い鳥が白い塔の下で歌う。緑の庭と赤い門。'},
 {'id':'compass-forward','prompt':'copper compass clear case black feather pink balloon striped awning '*90},
 {'id':'compass-reverse','prompt':'striped awning pink balloon black feather clear case copper compass '*90},
]
PAIRS=[['compass-inside','compass-outside'],['balloon-pink','balloon-brown'],['tower-above','tower-below'],['compass-forward','compass-reverse']]


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,old,prior=checked_collection(parent);job=Api().json('GET','/api/jobs/'+c['jobID'])
    if job['state']!='succeeded' or job['payload_custody']!=c['custody'] or job['config']['requestSHA256']!=prior['requestSHA256']:raise ValueError('Parent differs')
    seal=read(parent/'result/prediction-seal.json');spec=read(parent/'result/serve.json');compiled=read(parent/'result/compile-receipt.json');ref=read(parent/'result/reference-receipt.json')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    save(p/'source.json',{'landmarkAsset':seal['landmarkAsset'],'keyCenter':compiled['sourceKeyMean'],
        'compiledSourceReceiptSHA256':sha(parent/'result/compile-receipt.json'),'projection':ref['projection']})
    save(p/'serve-source.json',{'store':spec['store'],'manifestSHA256':spec['manifestSHA256'],'prompts':PROMPTS,'repeats':2})
    for n in ['context_transport.py','context_transport_study.py','context_statistics.py','context_statistics_study.py','encoder_atoms.py']:shutil.copyfile(HERE/n,p/n)
    request={'schema':'tce-query-transport-request-v1','parentJobID':job['job_id'],'parentRequestSHA256':prior['requestSHA256'],
        'scope':'Eight new rawstrings, existing source-only64landmarks, actual causal stable cross moments. Nearest and firstorder reads seal before references. One firstlayerhead, model-derived computation, not strictzero-neural images.',
        'pairs':PAIRS,'methods':METHODS,'momentPolicy':'Weighted centered cross-sum with running-max rescaling; fixed sourcekey center. No targetfit, clip or native fallback; optional invalid read rows retained with masks.',
        'routePolicy':'Nearest in frozen source key-covariance metric. Actual query displacement contracts actual-prefix Cov(v,y).',
        'diagnosticPositions':'Source/mask clocks0,lastactive,firstpad,511 plus16uniformly spaced positions; exact exponential transport and signed firstorder keyweights evaluated postseal only.',
        'interfaceFreeze':'Rawtext/sourcehashes requestbound; worker seals tokenizer/template IDs/masks before serving. Canonical full-prefix refs follow complete forecast seal.',
        'futureConsumerPlan':'Next guarded diagnostic replaces only predicted head0 through actual output/residual/postattentionRMS and fixed mask-clock MLP probes; otherheads explicitly supplied reference context, not closedencoder.',
        'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'sourceHashes':{f.name:sha(f) for f in p.iterdir()}}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/'payload'/n)!=h:raise ValueError('Staged source changed')
    cmd=command()[:-5]+['python','-u','context_transport_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-query-transport-context-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512,'landmarks':64},timeout_s=600,
        detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 fresh8rawstrings,64landmarks,8.52MB actual covariance state. First+2warm calls, fixed3checkpoints; seal before FP64head/W_o/direct-signed refs. No checkpoint/framework/GPU. Host-local lexical store; expected under3min,finite600s,no retry.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def scored(pred,target,mask,valid,matrix):
    use=mask&valid&np.isfinite(pred).all(1)&np.isfinite(target).all(1)
    return {'queries':int(mask.sum()),'validQueries':int(use.sum()),'invalidQueries':int(np.count_nonzero(mask&~use)),
        'coverage':'all' if np.all(use[mask]) else 'finite-subset-only',
        'head':summary(pred[use],target[use]) if use.any() else None,
        'projection':summary(pred[use]@matrix.T,target[use]@matrix.T) if use.any() else None}


def diagnostics(packet,asset,bundle,target):
    q,k,v=[floats(packet[n]).astype(float) for n in ['q','k','v']];n,d=q.shape
    positions=sorted(set([0,int(packet['mask'].sum())-1,min(int(packet['mask'].sum()),n-1),n-1]+np.linspace(0,n-1,16,dtype=int).tolist()));rows=[]
    for i in positions:
        keys=np.flatnonzero(packet['mask'][:i+1]);a=int(bundle['nearestIndex'][i]);x=q[i]/d**.25;u=k[keys]/d**.25-asset['keyCenter'];ell=asset['landmarks'][a];delta=x-ell
        logits=u@ell;p=np.exp(logits-logits.max());p/=p.sum();h=p@v[keys];mu=p@u;t=(u-mu)@delta;effective=p*(1+t)
        first=effective@v[keys];logtransport=logits+u@delta;weight=np.exp(logtransport-logtransport.max());weight/=weight.sum();exact=weight@v[keys]
        rows.append({'position':i,'active':bool(packet['mask'][i]),'allowedKeys':len(keys),'nearestIndex':a,
            'degreeZeroMaxAbsClosure':float(np.max(np.abs(h-bundle['nearest64_raw'][i]))),
            'firstOrderMaxAbsClosure':float(np.max(np.abs(first-bundle['transport64_raw'][i]))),
            'exponentialTransportMaxAbsClosure':float(np.max(np.abs(exact-target[i]))),
            'effectiveWeightSum':float(effective.sum()),'negativeEffectiveMass':float(-effective[effective<0].sum()),
            'minimumEffectiveWeight':float(effective.min()),'maximumEffectiveWeight':float(effective.max()),
            'prefixDisplacementVariance':float(p@(t*t)),'prefixDisplacementMin':float(t.min()),'prefixDisplacementMax':float(t.max())})
    return rows


def execute(root):
    guard();root=Path(root).resolve();r=read(root/'request.json');start=time.perf_counter()
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/n)!=h:raise ValueError('Source changed')
    source=read(root/'source.json');spec=read(root/'serve-source.json');out=root/'response-result';out.mkdir(exist_ok=False)
    if sha(source['landmarkAsset']['path'])!=source['landmarkAsset']['sha256']:raise ValueError('Source landmarks changed')
    prior=load(source['landmarkAsset']['path']);asset={k:prior[k] for k in ['landmarks','metricRoot']};asset['keyCenter']=np.asarray(source['keyCenter'],dtype=float)
    np.savez(root/'transport-asset.npz',**asset);spec['transportAsset']={'path':str(root/'transport-asset.npz'),'sha256':sha(root/'transport-asset.npz')}
    prefix=Prefix(spec['store'],spec['manifestSHA256']);interfaces={}
    for item in spec['prompts']:
        text,ids,mask=prefix.tokenize(item['prompt']);interfaces[item['id']]={'ids':ids.tolist(),'mask':mask.tolist(),'renderedTextSHA256':hashlib.sha256(text.encode()).hexdigest()}
    del prefix
    save(root/'prompt-interfaces.json',{'requestSHA256':r['requestSHA256'],'prompts':interfaces,'scope':'Tokenization only, sealed before candidate heads and reference reads.'})
    spec['interfaces']={'path':str(root/'prompt-interfaces.json'),'sha256':sha(root/'prompt-interfaces.json')};save(root/'serve.json',spec)
    t=time.perf_counter();subprocess.run([sys.executable,str(root/'context_transport.py'),str(root/'serve.json'),str(root/'predictions')],check=True)
    child_s=time.perf_counter()-t;seal_path=root/'predictions/prediction-seal.json';seal_sha=sha(seal_path);seal=read(seal_path);print('TRANSPORT_HEADS_SEALED',seal_sha,flush=True)
    projection=source['projection']
    if sha(projection['path'])!=projection['sha256']:raise ValueError('Projection changed')
    matrix=floats(np.load(projection['path'],allow_pickle=False)).astype(float);canonical=Prefix(spec['store'],spec['manifestSHA256']);work=root/'reference-work';work.mkdir()
    refs={};ports={};predictions={};refrows=[];direct=[];scores=[];pairs=[];observations=0;paired=0
    for item in spec['prompts']:
        name=item['id'];port=next(p for p in seal['ports'] if p['id']==name)
        if sha(port['path'])!=port['sha256']:raise ValueError('Port changed')
        packet=load(port['path']);ports[name]=packet;full,_=canonical.forward(item['prompt'])
        actual={**{k:full[k] for k in ['ids','mask','positions']},'q':full['qpos'][:,:128],'k':full['kpos'][:,:128],'v':full['v'][:,:128]}
        target=exact_head(actual);refs[name]=target;path=work/(name+'-exact.npz');np.savez(path,**target)
        refrows.append({'id':name,'path':str(path),'sha256':sha(path),'canonicalPortMismatches':{k:int(np.count_nonzero(packet[k]!=actual[k])) for k in actual}})
        record=next(z for z in seal['rows'] if z['id']==name)
        if sha(record['path'])!=record['sha256']:raise ValueError('Prediction changed')
        predictions[name]=load(record['path']);direct.append({'id':name,'rows':diagnostics(packet,asset,predictions[name],target['raw'])})
    with (out/'observations.jsonl').open('x') as f:
        for name,bundle in predictions.items():
            mask=ports[name]['mask'].astype(bool);target=refs[name]
            for method in METHODS:
                pred=bundle[method+'_raw'];valid=bundle[method+'_valid'];er,tr=[rowrms(z) for z in [pred-target['raw'],target['raw']]]
                for region,selected in [('all',np.ones(len(mask),dtype=bool)),('active',mask),('padding',~mask)]:
                    if selected.any():scores.append({'id':name,'method':method,'region':region,**scored(pred,target['raw'],selected,valid,matrix),
                        'rounded':scored(floats(bundle[method+'_rounded']),floats(target['rounded']),selected,bundle[method+'_rounded_valid'],matrix)})
                for i in range(len(mask)):
                    def finite(z):return float(z) if np.isfinite(z) else None
                    f.write(json.dumps({'id':name,'method':method,'position':i,'active':bool(mask[i]),'valid':bool(valid[i]),
                        'headErrorRms':finite(er[i]),'headReferenceRms':finite(tr[i]),'hullViolation':finite(bundle[method+'_hullViolation'][i]),
                        **{key:finite(bundle[key][i]) for key in ['sourceDistanceSquared','queryDisplacementSquared','correctionRms']},'nearestIndex':int(bundle['nearestIndex'][i])},allow_nan=False)+'\n');observations+=1
    with (out/'paired-observations.jsonl').open('x') as f:
        for a,b in r['pairs']:
            target=refs[b]['raw']-refs[a]['raw'];ma,mb=[ports[k]['mask'].astype(bool) for k in [a,b]]
            for method in METHODS:
                pred=predictions[b][method+'_raw']-predictions[a][method+'_raw'];valid=predictions[a][method+'_valid']&predictions[b][method+'_valid']
                for region,selected in [('all',np.ones(len(ma),dtype=bool)),('active_both',ma&mb),('padding_both',~ma&~mb),('mask_changed',ma!=mb)]:
                    if selected.any():pairs.append({'a':a,'b':b,'method':method,'region':region,**scored(pred,target,selected,valid,matrix)})
                er,tr=[rowrms(z) for z in [pred-target,target]]
                for i in range(len(ma)):
                    f.write(json.dumps({'a':a,'b':b,'method':method,'position':i,'activeA':bool(ma[i]),'activeB':bool(mb[i]),'valid':bool(valid[i]),
                        'headErrorRms':float(er[i]) if np.isfinite(er[i]) else None,'targetChangeRms':float(tr[i])},allow_nan=False)+'\n');paired+=1
    if sha(seal_path)!=seal_sha:raise ValueError('Seal changed')
    save(out/'reference-receipt.json',{'rows':refrows,'projection':projection,'directMomentDiagnostics':direct,
        'scope':'Post-seal canonical compiled-prefix/FP64head references, exponential identity and signed-weight diagnostics. No nativeFlash/fullencoder/image claim.'})
    for name,path in [('request.json',root/'request.json'),('prediction-seal.json',seal_path),('serve.json',root/'serve.json'),('transport-asset.npz',root/'transport-asset.npz'),('prompt-interfaces.json',root/'prompt-interfaces.json')]:shutil.copyfile(path,out/name)
    save(out/'source-receipt.json',source)
    save(out/'report.json',{'schema':'tce-query-transport-result-v1','requestSHA256':r['requestSHA256'],'scope':r['scope'],'serving':seal,'scores':scores,'pairedScores':pairs,
        'reference':read(out/'reference-receipt.json'),'predictionSealSHA256':seal_sha,'freshServingChildSeconds':child_s,'workerSeconds':time.perf_counter()-start,
        'observations':observations,'pairedObservations':paired,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'goalStatus':'Active. Source-defined first-order head computation, not fullconditioning or strictzero-neural images; actual future-consumer diagnostic next.'})
    print('QUERY_TRANSPORT_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
