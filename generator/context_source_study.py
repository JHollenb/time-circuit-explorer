"""Compile geometry on source atoms, then seal fresh raw-text context forecasts."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from encoder_atoms import np,read,save,sha,identity,Prefix,guard,floats
from context_statistics_study import exact_head,summary,rms,rowrms

HERE=Path(__file__).resolve().parent
PROMPTS=[
 {'id':'kite-above','prompt':'A red paper kite floats above a turquoise sailboat on a calm lake at sunrise.'},
 {'id':'kite-below','prompt':'A red paper kite floats below a turquoise sailboat on a calm lake at sunrise.'},
 {'id':'key-inside','prompt':'A bronze key lies inside a green wooden box, beside three white shells.'},
 {'id':'key-outside','prompt':'A bronze key lies outside a green wooden box, beside three white shells.'},
 {'id':'rain-yellow','prompt':'黄色い傘の下で黒い犬が眠る。雨の通り、青い自転車。'},
 {'id':'rain-blue','prompt':'青い傘の下で黒い犬が眠る。雨の通り、黄色い自転車。'},
 {'id':'marble-forward','prompt':'polished marble arch woven crimson carpet pale willow tree small golden fountain '*90},
 {'id':'marble-reverse','prompt':'small golden fountain pale willow tree woven crimson carpet polished marble arch '*90},
]
PAIRS=[['kite-above','kite-below'],['key-inside','key-outside'],['rain-yellow','rain-blue'],['marble-forward','marble-reverse']]
METHODS=['conditioned16','conditioned32','conditioned64','original64','uniform']


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,old,prior=checked_collection(parent)
    job=Api().json('GET','/api/jobs/'+c['jobID'])
    if job['state']!='succeeded' or job['payload_custody']!=c['custody'] or job['config']['requestSHA256']!=prior['requestSHA256']:raise ValueError('Parent identity differs')
    original=read(parent/'result/serve.json');ref=read(parent/'result/reference-receipt.json');seal=read(parent/'result/prediction-seal.json')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    ids=np.sort(np.random.Generator(np.random.PCG64(1707)).choice(151936,size=2048,replace=False)).astype('<i8')
    np.save(p/'source-ids.npy',ids,allow_pickle=False)
    save(p/'geometry-source.json',{'store':original['store'],'manifestSHA256':original['manifestSHA256'],
        'sampleIDs':'source-ids.npy','sampleIDsSHA256':sha(p/'source-ids.npy'),'queryPadMixtureWeight':.5,'batch':64,'epsilon':1e-12})
    save(p/'serve-source.json',{'store':original['store'],'manifestSHA256':original['manifestSHA256'],'head':0,'ranks':[16,32,64],
         'prompts':PROMPTS,'repeats':3,'omega':seal['omega']})
    save(p/'reference-source.json',{'projection':{k:ref['projection'][k] for k in ['path','sha256','payloadBytes']},
        'scope':'Saved original W_o head0 slice from the authenticated parent, used only for post-seal ideal linear error transport. No original checkpoint.'})
    for n in ['context_source_study.py','context_source_geometry.py','context_conditioned.py','context_statistics.py','context_statistics_study.py','context_key_diagnostics.py','context_kernel_diagnostics.py','encoder_atoms.py']:
        shutil.copyfile(HERE/n,p/n)
    request={'schema':'tce-source-conditioned-context-request-v1','parentJobID':job['job_id'],'parentRequestSHA256':prior['requestSHA256'],
        'scope':'Source-only positioned-atom geometry frozen before fresh raw-text forecasts; actual approximate streaming state, one first-layer head; full attention and saved output-slice references post-seal only. Not an image generator or zero-neural terminal result.',
        'sourceDistribution':'PCG64(seed1707)2048IDs without replacement from all151936IDs; exact ID bytes frozen. All512positionIDs uniform. Key pool uniform; query pool50percent uniformtoken/position plus50percent padID/position. No target operands in fit.',
        'pairs':PAIRS,'methods':METHODS,'sourceHashes':{f.name:sha(f) for f in p.iterdir()},
        'featureDraws':'Same frozen PCG64(seed6306)64x128Gaussian array as parent; nested16/32/64rows.',
        'referencePolicy':'Post-seal FP64 mathematical attention on canonical compiled BF16 operands, not native SDPA/Flash. Saved original W_o slice transported error offline.',
        'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':600}
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
    cmd=command()[:-5]+['python','-u','context_source_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-source-conditioned-context-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512},timeout_s=600,
        detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 compile source-only2048IDsx512positions(+pad) geometry, then fresh guarded raw8strings/5methods/512queries. Seal before canonical exact FP64heads/savedW_o error transport. No model/framework/checkpoint/GPU, finite600s,no retries. Compiled table host-local.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def load(path):
    with np.load(path,allow_pickle=False) as p:return {k:p[k] for k in p.files}


def direct(packet,omega,geometry):
    from context_key_diagnostics import feature_probability
    q,k,v=[floats(packet[n]).astype(np.float64) for n in ['q','k','v']];d=q.shape[1]
    x=q/d**.25;y=k/d**.25;a,b,D=[geometry[k] for k in ['a','b','D']];u=(x-a)*D;w=(y-b)/D;bias=y@a;heads=[]
    for i in range(len(x)):
        keys=np.flatnonzero(packet['mask'][:i+1]);heads.append(feature_probability(u[i],w[keys],omega,bias[keys])@v[keys])
    return np.stack(heads)


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json')
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/n)!=h:raise ValueError('Source changed')
    guard();started=time.perf_counter();out=root/'response-result';out.mkdir(exist_ok=False)
    from context_source_geometry import fit
    gs=read(root/'geometry-source.json')
    if sha(root/gs['sampleIDs'])!=gs['sampleIDsSHA256']:raise ValueError('Source IDs changed')
    prefix=Prefix(gs['store'],gs['manifestSHA256']);ids=np.load(root/gs['sampleIDs'],allow_pickle=False)
    geometry,compile_receipt=fit(prefix,ids,gs['queryPadMixtureWeight'],gs['batch'],gs['epsilon'])
    np.savez(root/'geometry.npz',**geometry);compile_receipt.update(source=gs,geometrySHA256=sha(root/'geometry.npz'))
    save(root/'compile-receipt.json',compile_receipt);del prefix
    print('SOURCE_GEOMETRY_FROZEN',sha(root/'geometry.npz'),flush=True)
    spec=read(root/'serve-source.json');spec['geometry']={'path':str(root/'geometry.npz'),'sha256':sha(root/'geometry.npz')}
    save(root/'serve.json',spec);start=time.perf_counter()
    subprocess.run([sys.executable,str(root/'context_conditioned.py'),str(root/'serve.json'),str(root/'predictions')],check=True)
    child_s=time.perf_counter()-start;seal_path=root/'predictions/prediction-seal.json';seal_sha=sha(seal_path);seal=read(seal_path)
    print('SOURCE_CONTEXT_FORECASTS_SEALED',seal_sha,flush=True)
    # First full-context references and saved output projection reads follow complete sealing.
    projection=read(root/'reference-source.json')['projection']
    if sha(projection['path'])!=projection['sha256']:raise ValueError('Saved output projection changed')
    matrix=floats(np.load(projection['path'],allow_pickle=False)).astype(np.float64)
    canonical=Prefix(spec['store'],spec['manifestSHA256']);work=root/'reference-work';work.mkdir();refs={};ports={};ref_rows=[];scores=[];forecasts={};closures=[]
    for item in spec['prompts']:
        port=next(p for p in seal['ports'] if p['id']==item['id'])
        if sha(port['path'])!=port['sha256']:raise ValueError('Port changed')
        packet=load(port['path']);ports[item['id']]=packet;full,_=canonical.forward(item['prompt'])
        actual={**{k:full[k] for k in ['ids','mask','positions']},'q':full['qpos'][:,:128],'k':full['kpos'][:,:128],'v':full['v'][:,:128]}
        mismatch={k:int(np.count_nonzero(packet[k]!=actual[k])) for k in actual};target=exact_head(actual);refs[item['id']]=target
        path=work/(item['id']+'-exact.npz');np.savez(path,**target);ref_rows.append({'id':item['id'],'path':str(path),'sha256':sha(path),'canonicalPortMismatches':mismatch})
    omega=np.load(spec['omega']['path'],allow_pickle=False)
    with (out/'observations.jsonl').open('x') as f:
        for record in seal['rows']:
            if sha(record['path'])!=record['sha256']:raise ValueError('Forecast changed')
            pred=load(record['path']);forecasts[record['id'],record['method']]=pred;target=refs[record['id']];packet=ports[record['id']]
            err=pred['raw']-target['raw'];rounderr=floats(pred['rounded']).astype(float)-floats(target['rounded']).astype(float);transport=err@matrix.T;refproj=target['raw']@matrix.T
            for region,mask in [('all',np.ones(len(err),dtype=bool)),('active',packet['mask'].astype(bool)),('padding',~packet['mask'].astype(bool))]:
                if mask.any():scores.append({'id':record['id'],'method':record['method'],'region':region,'queries':int(mask.sum()),
                    'head':summary(pred['raw'][mask],target['raw'][mask]),'roundedHead':summary(floats(pred['rounded'][mask]),floats(target['rounded'][mask])),
                    'projectedErrorRms':rms(transport[mask]),'projectedReferenceRms':rms(refproj[mask]),'projectedRelativeRms':rms(transport[mask])/rms(refproj[mask])})
            rowvalues={n:rowrms(a) for n,a in [('headError',err),('headReference',target['raw']),('roundedHeadError',rounderr),('projectedError',transport),('projectedReference',refproj)]}
            for i in range(len(err)):
                f.write(json.dumps({'id':record['id'],'method':record['method'],'position':i,'active':bool(packet['mask'][i]),
                    **{n+'Rms':float(a[i]) for n,a in rowvalues.items()},**{n:float(pred[n][i]) for n in ['featureMaxMass','featureEntropy','logDenominator','hullViolation']},
                    'exactMaxWeight':float(target['maxWeight'][i]),'exactEntropy':float(target['entropy'][i])},allow_nan=False)+'\n')
            if record['method'].startswith('conditioned') and record['id'] in [PROMPTS[0]['id'],PROMPTS[-2]['id']]:
                direct_head=direct(packet,omega[:record['rank']],geometry)
                closures.append({'id':record['id'],'method':record['method'],'maxAbs':float(np.max(np.abs(direct_head-pred['raw'])))})
    pairs=[]
    with (out/'paired-observations.jsonl').open('x') as f:
        for a,b in r['pairs']:
            target=refs[b]['raw']-refs[a]['raw'];refproj=target@matrix.T;ma,mb=[ports[k]['mask'].astype(bool) for k in [a,b]]
            for method in METHODS:
                pred=forecasts[b,method]['raw']-forecasts[a,method]['raw'];err=pred-target;transport=err@matrix.T
                for region,mask in [('all',np.ones(len(err),dtype=bool)),('active_both',ma&mb),('padding_both',~ma&~mb),('mask_changed',ma!=mb)]:
                    if mask.any():pairs.append({'a':a,'b':b,'method':method,'region':region,'queries':int(mask.sum()),'head':summary(pred[mask],target[mask]),
                        'projectedRelativeRms':rms(transport[mask])/rms(refproj[mask]) if rms(refproj[mask]) else None})
                values={n:rowrms(v) for n,v in [('headError',err),('targetChange',target),('projectedError',transport),('projectedTargetChange',refproj)]}
                for i in range(len(err)):f.write(json.dumps({'a':a,'b':b,'method':method,'position':i,'activeA':bool(ma[i]),'activeB':bool(mb[i]),
                    **{n+'Rms':float(v[i]) for n,v in values.items()}},allow_nan=False)+'\n')
    if sha(seal_path)!=seal_sha:raise ValueError('Seal changed')
    save(out/'reference-receipt.json',{'rows':ref_rows,'projection':projection,'exactHeadCalls':len(refs),'directConditionedKernelCalls':len(closures),
        'fullLexicalAssemblies':len(refs),'originalWeightBytesRead':0,'nativeAttentionParity':'unmeasured; ideal FP64 mathematical reference'})
    for n,path in [('prediction-seal.json',seal_path),('request.json',root/'request.json'),('serve.json',root/'serve.json'),
        ('compile-receipt.json',root/'compile-receipt.json'),('geometry.npz',root/'geometry.npz'),('source-ids.npy',root/'source-ids.npy'),('omega.npy',Path(spec['omega']['path']))]:shutil.copyfile(path,out/n)
    save(out/'report.json',{'schema':'tce-source-conditioned-context-result-v1','requestSHA256':r['requestSHA256'],'scope':r['scope'],
        'sourceCompile':compile_receipt,'serving':seal,'predictionSealSHA256':seal_sha,'scores':scores,'pairedScores':pairs,'featureKernelClosures':closures,
        'reference':read(out/'reference-receipt.json'),'freshServingChildSeconds':child_s,'workerSeconds':time.perf_counter()-started,
        'observations':len(seal['rows'])*512,'pairedObservations':len(PAIRS)*len(METHODS)*512,
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'goalStatus':'Active. A source-frozen approximate contextual read, not full conditioning, own-state diffusion or zero-neural images.'})
    print('SOURCE_CONTEXT_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
