"""Compile source-only readers; seal new raw-text predictions before references."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from encoder_atoms import np,Prefix,read,save,sha,identity,guard,floats
from context_landmarks import fit,METHODS
from context_statistics_study import exact_head,summary,rms,rowrms

HERE=Path(__file__).resolve().parent
PROMPTS=[
 {'id':'lantern-above','prompt':'A silver lantern hangs above a violet bench in a snowy courtyard.'},
 {'id':'lantern-below','prompt':'A silver lantern hangs below a violet bench in a snowy courtyard.'},
 {'id':'pear-green','prompt':'A ceramic bowl holds a green pear beside a purple plum.'},
 {'id':'pear-purple','prompt':'A ceramic bowl holds a purple pear beside a green plum.'},
 {'id':'bridge-left','prompt':'白い猫が赤い橋の左で座る。青い舟と黄色い花。'},
 {'id':'bridge-right','prompt':'白い猫が赤い橋の右で座る。青い舟と黄色い花。'},
 {'id':'courtyard-forward','prompt':'silver lantern violet bench snowy courtyard green pear purple plum '*90},
 {'id':'courtyard-reverse','prompt':'purple plum green pear snowy courtyard violet bench silver lantern '*90},
]
PAIRS=[['lantern-above','lantern-below'],['pear-green','pear-purple'],['bridge-left','bridge-right'],['courtyard-forward','courtyard-reverse']]


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,old,prior=checked_collection(parent)
    job=Api().json('GET','/api/jobs/'+c['jobID'])
    if job['state']!='succeeded' or job['payload_custody']!=c['custody'] or job['config']['requestSHA256']!=prior['requestSHA256']:raise ValueError('Parent identity differs')
    original=read(parent/'result/serve.json');ref=read(parent/'result/reference-receipt.json')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir();shutil.copyfile(parent/'result/source-ids.npy',p/'source-ids.npy')
    save(p/'compile-source.json',{'store':original['store'],'manifestSHA256':original['manifestSHA256'],
        'sampleIDs':'source-ids.npy','sampleIDsSHA256':sha(p/'source-ids.npy'),'landmarks':64,'candidates':4096,'seed':1909,'batch':64})
    save(p/'serve-source.json',{'store':original['store'],'manifestSHA256':original['manifestSHA256'],'prompts':PROMPTS,'repeats':5})
    save(p/'reference-source.json',{'projection':{k:ref['projection'][k] for k in ['path','sha256','payloadBytes']}})
    for name in ['context_landmark_study.py','context_landmarks.py','context_source_geometry.py','context_statistics.py','context_statistics_study.py','encoder_atoms.py']:
        shutil.copyfile(HERE/name,p/name)
    request={'schema':'tce-source-landmark-request-v1','parentJobID':job['job_id'],'parentRequestSHA256':prior['requestSHA256'],
        'scope':'Source-only64query landmarks and key covariance; fresh8rawstrings advance positive landmark prefix state. One first-layer head; reduced model-derived attention, not strictzero-neural images.',
        'pairs':PAIRS,'methods':METHODS,'sourceIDsSHA256':sha(p/'source-ids.npy'),
        'sourceRule':'Full2048sourceIDsx512positions key covariance, scaledq/k by128^.25. PCG64(seed1909)4096lexical query candidates plus512padpositions,50/50mixture;64deterministic farthest-first landmarks. All-landmarkRBF with source median-positive-nearest distance bandwidth.',
        'interfaceFreeze':'Raw strings/source manifest hashes fixed in request. Worker seals deterministic tokenizer/template IDs/masks before compiling source readers or forming any target head; fresh serving checks those exact interfaces. No target-based string/parameter selection.',
        'referencePolicy':'Complete serving seal precedes canonical full-prefix assemblies, exact FP64 attention, direct landmark checks and saved originalW_o slice transport. Not nativeSDPA/Flash parity.',
        'sourceHashes':{f.name:sha(f) for f in p.iterdir()},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':600}
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
    cmd=command()[:-5]+['python','-u','context_landmark_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-source-landmark-context-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512,'landmarks':64},
        timeout_s=600,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 source2048IDsx512key covariance/64query landmarks, fresh8rawstrings/4same-state reader controls. Seal before exact-head/W_o refs. No checkpoint/framework/GPU. Host-local lexical store; finite600s,no retry; expected under2min, count all reduced attention work.')
    result={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',result);return result


def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def direct(packet,asset,positions):
    q,k,v=[floats(packet[n]).astype(np.float64) for n in ['q','k','v']];d=q.shape[1];ell=asset['landmarks'];result={m:[] for m in METHODS}
    for i in positions:
        indices=np.flatnonzero(packet['mask'][:i+1]);x=q[i]/d**.25;y=k[indices]/d**.25
        delta=x-ell;dist=np.einsum('ad,df,af->a',delta,asset['covariance'],delta)
        route=-dist/(2*float(asset['bandwidthSquared']));logw=route-route.max()-np.log(np.exp(route-route.max()).sum());w=np.exp(logw)
        score=ell@y.T;top=score.max(1);K=np.exp(score-top[:,None]);Z=K.sum(1);readers=(K@v[indices])/Z[:,None]
        logs=logw+top+np.log(Z);mass=np.exp(logs-logs.max());mass/=mass.sum()
        rows=[w@readers,readers[np.argmin(dist)],mass@readers,v[indices].mean(0)]
        for m,h in zip(METHODS,rows):result[m].append(h)
    return {m:np.stack(v) for m,v in result.items()}


def execute(root):
    guard();root=Path(root).resolve();r=read(root/'request.json');start=time.perf_counter()
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/n)!=h:raise ValueError('Source changed')
    out=root/'response-result';out.mkdir(exist_ok=False);gs=read(root/'compile-source.json');spec=read(root/'serve-source.json')
    if sha(root/gs['sampleIDs'])!=gs['sampleIDsSHA256']:raise ValueError('Source IDs changed')
    prefix=Prefix(gs['store'],gs['manifestSHA256']);interfaces={}
    for item in spec['prompts']:
        text,ids,mask=prefix.tokenize(item['prompt']);interfaces[item['id']]={'ids':ids.tolist(),'mask':mask.tolist(),
            'renderedTextSHA256':hashlib.sha256(text.encode()).hexdigest(),'activeTokens':int(mask.sum())}
    save(root/'prompt-interfaces.json',{'requestSHA256':r['requestSHA256'],'manifestSHA256':gs['manifestSHA256'],'prompts':interfaces,
        'scope':'Tokenization only, frozen before source reader compilation and all target contextual reads.'})
    asset,receipt=fit(prefix,np.load(root/gs['sampleIDs'],allow_pickle=False),gs['landmarks'],gs['candidates'],gs['seed'],gs['batch'])
    np.savez(root/'landmarks.npz',**asset);receipt.update(source=gs,assetSHA256=sha(root/'landmarks.npz'));save(root/'compile-receipt.json',receipt);del prefix
    print('SOURCE_LANDMARKS_FROZEN',sha(root/'landmarks.npz'),flush=True)
    spec['landmarkAsset']={'path':str(root/'landmarks.npz'),'sha256':sha(root/'landmarks.npz')}
    spec['interfaces']={'path':str(root/'prompt-interfaces.json'),'sha256':sha(root/'prompt-interfaces.json')};save(root/'serve.json',spec)
    t=time.perf_counter();subprocess.run([sys.executable,str(root/'context_landmarks.py'),str(root/'serve.json'),str(root/'predictions')],check=True)
    child_s=time.perf_counter()-t;seal_path=root/'predictions/prediction-seal.json';seal_sha=sha(seal_path);seal=read(seal_path)
    print('LANDMARK_FORECASTS_SEALED',seal_sha,flush=True)
    projection=read(root/'reference-source.json')['projection']
    if sha(projection['path'])!=projection['sha256']:raise ValueError('Saved output slice changed')
    matrix=floats(np.load(projection['path'],allow_pickle=False)).astype(np.float64)
    canonical=Prefix(spec['store'],spec['manifestSHA256']);work=root/'reference-work';work.mkdir();refs={};ports={};predictions={};refrows=[];closures=[]
    for item in spec['prompts']:
        port=next(p for p in seal['ports'] if p['id']==item['id'])
        if sha(port['path'])!=port['sha256']:raise ValueError('Sealed port changed')
        packet=load(port['path']);ports[item['id']]=packet;full,_=canonical.forward(item['prompt'])
        actual={**{k:full[k] for k in ['ids','mask','positions']},'q':full['qpos'][:,:128],'k':full['kpos'][:,:128],'v':full['v'][:,:128]}
        mismatch={k:int(np.count_nonzero(packet[k]!=actual[k])) for k in actual};target=exact_head(actual);refs[item['id']]=target
        path=work/(item['id']+'-exact.npz');np.savez(path,**target);refrows.append({'id':item['id'],'path':str(path),'sha256':sha(path),'canonicalPortMismatches':mismatch})
        record=next(z for z in seal['rows'] if z['id']==item['id'])
        if sha(record['path'])!=record['sha256']:raise ValueError('Sealed readers changed')
        predictions[item['id']]=load(record['path'])
        if item['id'] in [PROMPTS[0]['id'],PROMPTS[-2]['id']]:
            positions=sorted({0,int(packet['mask'].sum())-1,min(int(packet['mask'].sum()),511),511});raw=direct(packet,asset,positions)
            for method in METHODS:closures.append({'id':item['id'],'method':method,'positions':positions,
                'maxAbs':float(np.max(np.abs(predictions[item['id']][method+'_raw'][positions]-raw[method])))})
    scores=[];pairs=[];observations=0;pair_count=0
    with (out/'observations.jsonl').open('x') as f:
        for item in spec['prompts']:
            name=item['id'];bundle=predictions[name];packet=ports[name];target=refs[name];mask=packet['mask'].astype(bool)
            for method in METHODS:
                pred=bundle[method+'_raw'];rounded=floats(bundle[method+'_rounded']).astype(float);error=pred-target['raw'];transport=error@matrix.T;refproj=target['raw']@matrix.T
                for region,selected in [('all',np.ones(len(mask),dtype=bool)),('active',mask),('padding',~mask)]:
                    if selected.any():scores.append({'id':name,'method':method,'region':region,'queries':int(selected.sum()),'head':summary(pred[selected],target['raw'][selected]),
                        'roundedHead':summary(rounded[selected],floats(target['rounded'][selected])),'projection':summary(pred[selected]@matrix.T,refproj[selected])})
                er,tr,pe,pr=[rowrms(a) for a in [error,target['raw'],transport,refproj]]
                for i in range(len(mask)):
                    f.write(json.dumps({'id':name,'method':method,'position':i,'active':bool(mask[i]),'headErrorRms':float(er[i]),'headReferenceRms':float(tr[i]),
                        'projectedErrorRms':float(pe[i]),'projectedReferenceRms':float(pr[i]),'hullViolation':float(bundle[method+'_hullViolation'][i]),
                        **{key:float(bundle[key][i]) for key in ['routeEntropy','routeMaxMass','nearestDistanceSquared','kernelRouteMaxMass','partitionLogRange']},
                        'nearestLandmarkIndex':int(bundle['nearestIndex'][i])},allow_nan=False)+'\n');observations+=1
    with (out/'paired-observations.jsonl').open('x') as f:
        for a,b in r['pairs']:
            target=refs[b]['raw']-refs[a]['raw'];ma,mb=[ports[k]['mask'].astype(bool) for k in [a,b]];refproj=target@matrix.T
            for method in METHODS:
                pred=predictions[b][method+'_raw']-predictions[a][method+'_raw'];error=pred-target;transport=error@matrix.T
                for region,selected in [('all',np.ones(len(ma),dtype=bool)),('active_both',ma&mb),('padding_both',~ma&~mb),('mask_changed',ma!=mb)]:
                    if selected.any():pairs.append({'a':a,'b':b,'method':method,'region':region,'queries':int(selected.sum()),'head':summary(pred[selected],target[selected]),
                        'projection':summary(pred[selected]@matrix.T,refproj[selected])})
                er,tr,pe,pr=[rowrms(z) for z in [error,target,transport,refproj]]
                for i in range(len(ma)):
                    f.write(json.dumps({'a':a,'b':b,'method':method,'position':i,'activeA':bool(ma[i]),'activeB':bool(mb[i]),
                        'headErrorRms':float(er[i]),'targetChangeRms':float(tr[i]),'projectedErrorRms':float(pe[i]),'projectedTargetChangeRms':float(pr[i])},allow_nan=False)+'\n');pair_count+=1
    if sha(seal_path)!=seal_sha:raise ValueError('Prediction seal changed')
    save(out/'reference-receipt.json',{'rows':refrows,'projection':projection,'directReaderClosures':closures,'exactHeadCalls':len(refs),
        'scope':'Post-seal full compiled-prefix canonical operands and ideal FP64 attention, not native Flash parity or complete encoder/image. Original checkpoint/framework not opened.'})
    for name,path in [('request.json',root/'request.json'),('prediction-seal.json',seal_path),('serve.json',root/'serve.json'),
        ('compile-receipt.json',root/'compile-receipt.json'),('landmarks.npz',root/'landmarks.npz'),('source-ids.npy',root/'source-ids.npy'),('prompt-interfaces.json',root/'prompt-interfaces.json')]:shutil.copyfile(path,out/name)
    save(out/'report.json',{'schema':'tce-source-landmark-result-v1','requestSHA256':r['requestSHA256'],'scope':r['scope'],
        'sourceCompile':receipt,'serving':seal,'scores':scores,'pairedScores':pairs,'reference':read(out/'reference-receipt.json'),
        'predictionSealSHA256':seal_sha,'freshServingChildSeconds':child_s,'workerSeconds':time.perf_counter()-start,
        'observations':observations,'pairedObservations':pair_count,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'goalStatus':'Active. Reusable source-defined first-head reader tested on new raw strings; reduced attention remains learned/model-derived, no strictzero-neural image result.'})
    print('SOURCE_LANDMARK_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
