"""Finite raw-prompt context-state assay. Full attention exists only after seals."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

from encoder_atoms import np,Prefix,read,save,sha,identity,floats,words

HERE=Path(__file__).resolve().parent
PAIRS=[['objects-left','objects-right'],['forest-short','forest-expanded'],['order-a','order-b'],
       ['repeat-once','repeat-many'],['unicode-a','unicode-b'],['long-forward','long-reverse']]
PROMPTS=[
 {'id':'objects-left','prompt':'An amber glass bottle sits to the left of a violet ceramic bowl.'},
 {'id':'objects-right','prompt':'An amber glass bottle sits to the right of a violet ceramic bowl.'},
 {'id':'forest-short','prompt':'A fox rests beneath a pine tree at dawn.'},
 {'id':'forest-expanded','prompt':'A fox rests beneath a pine tree at dawn. Behind the tree, a silver bicycle leans against a mossy wall while mist covers the river.'},
 {'id':'order-a','prompt':'silver teapot, blue notebook, wooden spoon'},
 {'id':'order-b','prompt':'wooden spoon, blue notebook, silver teapot'},
 {'id':'repeat-once','prompt':'A tiny clock hangs in an empty station.'},
 {'id':'repeat-many','prompt':'A tiny clock, clock, clock, clock, clock hangs in an empty station, station, station.'},
 {'id':'unicode-a','prompt':'夜の喫茶店。白い猫、赤い椅子、緑の窓。 Café lumière ☕'},
 {'id':'unicode-b','prompt':'夜の喫茶店。緑の猫、白い椅子、赤い窓。 Café lumière ☕'},
 {'id':'long-forward','prompt':'copper lantern stone bridge purple banner narrow river '*90},
 {'id':'long-reverse','prompt':'narrow river purple banner stone bridge copper lantern '*90},
 {'id':'whitespace','prompt':'\n\tA   brass bell:\n\na folded map; a pear?!  '},
 {'id':'literal','prompt':'A poster displaying the literal symbols <think> and </think> above a painted door.'},
]
SMOKE=[{'id':'smoke-short','prompt':'A brass bell beside a folded map.'},
       {'id':'smoke-long','prompt':'brass bell folded map pear railway platform '*90}]
SCOPE='Raw text -> complete compiled lexical atoms -> one first-layer Qwen3 head0 positive-kernel context state; exact FP64 attention and original output-slice diagnostics offline only. Reduced model-derived computation, no full encoder or image.'


def stage(root,parent,mode):
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();collection=read(parent/'collection.json');prior=read(parent/'request.json')
    job=Api().json('GET','/api/jobs/'+collection['jobID'])
    if job['state']!='succeeded' or job['config']['requestSHA256']!=prior['requestSHA256'] or job['payload_custody']!=collection['custody']:
        raise ValueError('Lexical parent custody differs')
    for name in ['manifest.json','report.json']:
        if sha(parent/'result'/name)!=collection['inventory']['files'][name]['sha256']:raise ValueError('Parent compact result changed')
    manifest=read(parent/'result/manifest.json')
    if manifest['coverageCount']!=151936:raise ValueError('Complete vocabulary required')
    root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    source=read(HERE/'encoder-atoms-source.json')
    serving={'store':job['result']['diagnostics']['cwd']+'/atom-store','manifestSHA256':sha(parent/'result/manifest.json'),
             'head':0,'ranks':[16,32,64],'seed':6306,'repeats':7 if mode=='full' else 2,
             'prompts':PROMPTS if mode=='full' else SMOKE}
    save(payload/'serve-source.json',serving)
    save(payload/'reference-source.json',{'weightFile':{**source['weightFile'],'path':source['modelRoot']+'/'+source['weightFile']['path']},
         'tensor':'model.layers.0.self_attn.o_proj.weight','shape':[2560,4096],'dtype':'BF16','dataOffsets':[932587776,953559296],
         'head':0,'headDim':128,'scope':'Original head0 output slice, used only for ideal linear error transport after forecasts seal. Not complete native fused projection/residual.'})
    for name in ['context_statistics.py','context_statistics_study.py','encoder_atoms.py']:shutil.copyfile(HERE/name,payload/name)
    request={'schema':'tce-context-statistics-request-v1','mode':mode,'scope':SCOPE,'head':0,'kvHead':0,'ranks':[16,32,64],
             'seed':6306,'pairs':PAIRS if mode=='full' else [],'sourceHashes':{p.name:sha(p) for p in payload.iterdir()},
             'parent':{'jobID':job['job_id'],'requestSHA256':prior['requestSHA256'],'custody':job['payload_custody']},
             'backend':'numpy-fp64','device':'cpu','cpuThreads':1,'exactAttentionOnline':False,
             'featureConstruction':'Nested first16/32/64 rows of standard-normal PCG64(seed6306) matrix [64,128]. Frozen before targets; exp dot positive kernel with input norm and d**.25 scaling.',
             'referencePolicy':'Post-seal FP64 causal attention consuming BF16 lexical/RoPE operands, then BF16 head rounding. Not native SDPA/Flash parity.',
             'retention':'Complete ports, forecasts, state checkpoints and reference heads stay near compute. Collect compact metrics and manifests; original weights excluded from serving.'}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(payload/'request.json',request)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/'payload'/n)!=h:raise ValueError('Staged source changed')
    cmd=command()[:-5]+['python','-u','context_statistics_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-context-statistics-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
         config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'mode':r['mode'],'backend':'numpy','device':'cpu','head':0,'ranks':r['ranks'],'seq_len':512},
         timeout_s=600,detach=True,retry_on_kill=False,preflight=True,guarded=True,
         note='CPU1 raw-text context-state '+r['mode']+': head0,14strings full/2smoke, ranks16/32/64+uniform,512causal queries. Fixed features, no native/framework/projection online; seals then FP64 full-head refs and originalW_o slice error transport. Finite600s,no retries; retained lexical store host-local.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def exact_head(packet):
    q,k,v=[floats(packet[n]).astype(np.float64) for n in ['q','k','v']];n,d=q.shape
    logits=q@k.T/np.sqrt(d);allowed=(np.arange(n)[None,:]<=np.arange(n)[:,None])&packet['mask'].astype(bool)[None,:]
    logits[~allowed]=-np.inf;present=np.any(allowed,axis=1);weights=np.zeros_like(logits)
    a=logits[present];weights[present]=np.exp(a-a.max(1,keepdims=True));totals=weights.sum(1)
    probability=np.divide(weights,totals[:,None],out=np.zeros_like(weights),where=totals[:,None]>0)
    raw=probability@v;entropy=-np.sum(probability*np.log(probability,where=probability>0,out=np.zeros_like(probability)),axis=1)
    return {'raw':raw,'rounded':words(raw),'maxWeight':probability.max(1),'entropy':entropy}


def direct_feature_head(packet,omega):
    q,k,v=[floats(packet[n]).astype(np.float64) for n in ['q','k','v']];d=q.shape[1];x=q/(d**.25);y=k/(d**.25)
    lq=x@omega.T-.5*(x*x).sum(1)[:,None];lk=y@omega.T-.5*(y*y).sum(1)[:,None];out=[]
    for i in range(len(q)):
        selected=np.flatnonzero(packet['mask'][:i+1])
        if not len(selected):out.append(np.zeros(d));continue
        logfeatures=lq[i][None,:]+lk[selected];top=logfeatures.max(1)
        logs=top+np.log(np.exp(logfeatures-top[:,None]).sum(1));weight=np.exp(logs-logs.max());weight/=weight.sum()
        out.append(weight@v[selected])
    return np.stack(out)


def projection(spec,out):
    ref=spec['weightFile'];path=Path(ref['path']);t=time.perf_counter()
    if path.stat().st_size!=ref['bytes'] or sha(path)!=ref['sha256']:raise ValueError('Original source shard differs')
    with path.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(length));entry=header[spec['tensor']]
        if entry!={'dtype':spec['dtype'],'shape':spec['shape'],'data_offsets':spec['dataOffsets']}:raise ValueError('Output tensor header differs')
        rows=[];width=spec['shape'][1];d=spec['headDim']
        for i in range(spec['shape'][0]):
            f.seek(8+length+entry['data_offsets'][0]+2*(i*width+spec['head']*d));raw=f.read(d*2)
            if len(raw)!=d*2:raise ValueError('Truncated output slice')
            rows.append(np.frombuffer(raw,dtype='<u2').copy())
    matrix=np.stack(rows);p=out/'output-slice.npy';np.save(p,matrix,allow_pickle=False)
    return floats(matrix).astype(np.float64),{'path':str(p),'sha256':sha(p),'payloadBytes':matrix.nbytes,'source':spec,'seconds':time.perf_counter()-t}


def load(path):
    with np.load(path,allow_pickle=False) as d:return {k:d[k] for k in d.files}
def rms(value):return float(np.sqrt(np.mean(np.square(value))))
def summary(pred,target):
    error=pred-target;scale=rms(target);a=rms(pred)
    return {'rmsError':rms(error),'targetRms':scale,'relativeRms':rms(error)/scale if scale else None,
            'maxAbs':float(np.max(np.abs(error))),'cosine':float(np.mean(pred*target)/(a*scale)) if a*scale else None}
def rowrms(a):return np.sqrt(np.mean(a*a,axis=1))


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json')
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/n)!=h:raise ValueError('Source changed')
    start=time.perf_counter();out=root/'response-result';out.mkdir(exist_ok=False)
    spec=read(root/'serve-source.json');omega=np.random.Generator(np.random.PCG64(spec['seed'])).standard_normal((max(spec['ranks']),128))
    np.save(root/'omega.npy',omega,allow_pickle=False);spec['omega']={'path':str(root/'omega.npy'),'sha256':sha(root/'omega.npy')};save(root/'serve.json',spec)
    t=time.perf_counter();subprocess.run([sys.executable,str(root/'context_statistics.py'),str(root/'serve.json'),str(root/'predictions')],check=True)
    child_s=time.perf_counter()-t;seal_path=root/'predictions/prediction-seal.json';seal_sha=sha(seal_path);seal=read(seal_path)
    print('CONTEXT_STATE_SEALED',seal_sha,flush=True)
    # Exact attention and original model files are first evaluated/read AFTER complete forecasts seal.
    work=root/'reference-work';work.mkdir();matrix,parameter=projection(read(root/'reference-source.json'),work)
    canonical=Prefix(spec['store'],spec['manifestSHA256']);references={};ports={};reference_rows=[];closures=[];ref_start=time.perf_counter()
    for item in spec['prompts']:
        port=next(p for p in seal['ports'] if p['id']==item['id'])
        if sha(port['path'])!=port['sha256']:raise ValueError('Port changed')
        packet=load(port['path']);ports[item['id']]=packet;full,_=canonical.forward(item['prompt'])
        actual={**{n:full[n] for n in ['ids','mask','positions']},'q':full['qpos'][:,:128],'k':full['kpos'][:,:128],'v':full['v'][:,:128]}
        mismatch={n:int(np.count_nonzero(packet[n]!=actual[n])) for n in actual}
        target=exact_head(actual);references[item['id']]=target;p=work/(item['id']+'-exact.npz');np.savez(p,**target)
        reference_rows.append({'id':item['id'],'path':str(p),'sha256':sha(p),'canonicalPortMismatches':mismatch})
    forecasts={};scores=[];dependent_rows=0
    with (out/'observations.jsonl').open('x') as stream:
        for record in seal['rows']:
            if sha(record['path'])!=record['sha256']:raise ValueError('Forecast changed')
            prediction=load(record['path']);forecasts[record['id'],record['method']]=prediction;target=references[record['id']];packet=ports[record['id']]
            error=prediction['raw']-target['raw'];round_error=floats(prediction['rounded']).astype(float)-floats(target['rounded']).astype(float)
            transported=error@matrix.T;transported_round=round_error@matrix.T;reference_projection=target['raw']@matrix.T
            rows={n:rowrms(a) for n,a in [('headError',error),('headReference',target['raw']),('roundedHeadError',round_error),
                  ('projectedError',transported),('projectedRoundedError',transported_round),('projectedReference',reference_projection)]}
            for region,mask in [('all',np.ones(len(error),dtype=bool)),('active',packet['mask'].astype(bool)),('padding',~packet['mask'].astype(bool))]:
                if not mask.any():continue
                scores.append({'id':record['id'],'method':record['method'],'region':region,'queries':int(mask.sum()),
                               'head':summary(prediction['raw'][mask],target['raw'][mask]),
                               'roundedHead':summary(floats(prediction['rounded'][mask]),floats(target['rounded'][mask])),
                               'projectedErrorRms':rms(transported[mask]),'projectedReferenceRms':rms(reference_projection[mask]),
                               'projectedRelativeRms':rms(transported[mask])/rms(reference_projection[mask]) if rms(reference_projection[mask]) else None,
                               'projectedRoundedErrorRms':rms(transported_round[mask])})
            for i in range(len(error)):
                row={'id':record['id'],'method':record['method'],'position':i,'active':bool(packet['mask'][i]),'allowedKeys':int(packet['mask'][:i+1].sum()),
                     **{n+'Rms':float(a[i]) for n,a in rows.items()},'logDenominator':float(prediction['logDenominator'][i]) if not prediction['empty'][i] else None,
                     'featureMaxMass':float(prediction['featureMaxMass'][i]),'featureEntropy':float(prediction['featureEntropy'][i]),
                     'exactMaxWeight':float(target['maxWeight'][i]),'exactEntropy':float(target['entropy'][i]),'hullViolation':float(prediction['hullViolation'][i])}
                stream.write(json.dumps(row,allow_nan=False)+'\n');dependent_rows+=1
            if record['rank'] and record['id'] in [spec['prompts'][0]['id'],spec['prompts'][-1 if r['mode']=='smoke' else 10]['id']]:
                direct=direct_feature_head(packet,omega[:record['rank']]);closures.append({'id':record['id'],'method':record['method'],'maxAbs':float(np.max(np.abs(direct-prediction['raw']))),'rms':rms(direct-prediction['raw'])})
    pairs=[]
    with (out/'paired-observations.jsonl').open('x') as stream:
        for a,b in r['pairs']:
            target=references[b]['raw']-references[a]['raw'];target_projection=target@matrix.T
            for method in ['rank16','rank32','rank64','uniform']:
                pred=forecasts[b,method]['raw']-forecasts[a,method]['raw'];err=pred-target;projected=err@matrix.T
                ma,mb=[ports[n]['mask'].astype(bool) for n in [a,b]]
                for region,mask in [('all',np.ones(len(target),dtype=bool)),('active_both',ma&mb),('padding_both',~ma&~mb),('mask_changed',ma!=mb)]:
                    if not mask.any():continue
                    pairs.append({'a':a,'b':b,'method':method,'region':region,'queries':int(mask.sum()),'head':summary(pred[mask],target[mask]),
                                  'projectedErrorRms':rms(projected[mask]),'projectedReferenceRms':rms(target_projection[mask]),
                                  'projectedRelativeRms':rms(projected[mask])/rms(target_projection[mask]) if rms(target_projection[mask]) else None})
                e,t,pe,pt=[rowrms(x) for x in [err,target,projected,target_projection]]
                for i in range(len(target)):stream.write(json.dumps({'a':a,'b':b,'method':method,'position':i,'activeA':bool(ma[i]),'activeB':bool(mb[i]),
                    'headErrorRms':float(e[i]),'targetChangeRms':float(t[i]),'projectedErrorRms':float(pe[i]),'projectedTargetChangeRms':float(pt[i])},allow_nan=False)+'\n')
    if sha(seal_path)!=seal_sha:raise ValueError('Seal changed during scoring')
    for name,path in [('prediction-seal.json',seal_path),('request.json',root/'request.json'),('serve.json',root/'serve.json'),('omega.npy',root/'omega.npy')]:shutil.copyfile(path,out/name)
    save(out/'reference-receipt.json',{'rows':reference_rows,'projection':parameter,'exactHeadCalls':len(references),'fullLexicalAssemblies':len(references),
         'directFeatureKernelCalls':len(closures),'scoringSecondsIncludingPortChecks':time.perf_counter()-ref_start,
         'scope':'FP64 ideal causal attention on compiled BF16 operands. Output projection slice diagnostic uses original W after seals; no native attention/full encoder/image or fused-project parity.'})
    save(out/'report.json',{'schema':'tce-context-statistics-result-v1','requestSHA256':r['requestSHA256'],'scope':SCOPE,'serving':seal,
         'predictionSealSHA256':seal_sha,'scores':scores,'pairedScores':pairs,'featureKernelClosures':closures,
         'reference':read(out/'reference-receipt.json'),'freshServingChildSeconds':child_s,'workerSeconds':time.perf_counter()-start,
         'observations':dependent_rows,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
         'goalStatus':'Active; reduced approximate contextual computation only, not zero-neural arbitrary-prompt images.'})
    print('CONTEXT_STATISTICS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');p.add_argument('--mode',choices=['smoke','full'],default='smoke');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent,a.mode)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
