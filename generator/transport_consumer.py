"""Original-equation receiving-operation diagnostic, not a serving encoder.

Only head0 is predicted. Other heads are explicitly supplied exact mathematical
reference context. Original output/RMS/MLP weights are evaluated offline here.
"""
import argparse
import json
from pathlib import Path
import shutil
import struct
import sys
import time

from encoder_atoms import np,Prefix,read,save,sha,identity,floats,words,bf,NoNeural
from context_statistics_study import exact_head,summary,rowrms

HERE=Path(__file__).resolve().parent
ARMS=['zero','exact','nearest64','transport64']


def rms_norm(value,gamma,epsilon):
    value=floats(value);variance=np.mean(value*value,axis=1,keepdims=True,dtype=np.float32)
    normalized=bf(value/np.sqrt(variance+np.float32(epsilon)))
    return words(normalized*floats(gamma))


def mlp(value,gate,up,down):
    value=floats(value).astype(np.float64);g=bf(value@gate.T);u=bf(value@up.T)
    tail=np.exp(-np.abs(g));sigmoid=np.where(g>=0,1/(1+tail),tail/(1+tail));activated=bf(g*sigmoid)
    product=bf(activated*u);return words(product.astype(float)@down.T)


def consume(e,rest,head,matrix,gamma,epsilon,positions,weights):
    projected=words(rest+head@matrix.T);residual=words(floats(e)+floats(projected));normalized=rms_norm(residual,gamma,epsilon)
    written=mlp(normalized[positions],*weights);state=words(floats(residual[positions])+floats(written))
    return {'projection':projected,'residual':residual,'rms':normalized,'mlp':written,'state':state}


def parameters(spec):
    root=Path(spec['modelRoot']);entry=spec['weightFile'];path=root/entry['path']
    if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:raise ValueError('Original weight shard differs')
    config_path=root/'text_encoder/config.json'
    if sha(config_path)!=spec['assets']['text_encoder/config.json']:raise ValueError('Encoder configuration differs')
    config=read(config_path)
    if (config['hidden_size'],config['num_attention_heads'],config['num_key_value_heads'],config['head_dim'],config['hidden_act'])!=(2560,32,8,128,'silu'):raise ValueError('Unexpected original consumer geometry')
    names={'output':'model.layers.0.self_attn.o_proj.weight','gamma':'model.layers.0.post_attention_layernorm.weight',
        'gate':'model.layers.0.mlp.gate_proj.weight','up':'model.layers.0.mlp.up_proj.weight','down':'model.layers.0.mlp.down_proj.weight'}
    values={};records=[]
    with path.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        if length>16<<20:raise ValueError('Unbounded checkpoint header')
        header=json.loads(f.read(length))
        for key,name in names.items():
            item=header[name]
            if item['dtype']!='BF16':raise ValueError('Unexpected original parameter dtype')
            a,b=item['data_offsets'];count=int(np.prod(item['shape']))
            if b-a!=count*2 or a<0 or 8+length+b>entry['bytes']:raise ValueError('Invalid source tensor span')
            f.seek(8+length+a);raw=f.read(b-a)
            if len(raw)!=b-a:raise ValueError('Truncated source tensor')
            value=np.frombuffer(raw,dtype='<u2').reshape(item['shape']).copy();values[key]=value if key=='gamma' else floats(value).astype(np.float64)
            records.append({'key':key,'tensor':name,'header':item,'payloadBytes':len(raw),'payloadSHA256':__import__('hashlib').sha256(raw).hexdigest()})
    h=config['hidden_size'];width=config['intermediate_size']
    if values['output'].shape!=(h,4096) or values['gamma'].shape!=(h,) or values['gate'].shape!=(width,h) or values['up'].shape!=(width,h) or values['down'].shape!=(h,width):raise ValueError('Original matrix shapes differ')
    return values,config,{'source':spec,'tensors':records,'parameterPayloadBytes':sum(x['payloadBytes'] for x in records),
        'float64ResidentMatrixBytes':sum(x.nbytes for x in values.values()),'sourceShardBytesHashed':entry['bytes'],
        'numericPolicy':'FP64 mathematical attention/output/MLP contractions on BF16 operands, explicit BF16 boundaries; FP32 RMS variance and SiLU. Shared equation reference, not nativeCUDA/Flash/Torch numerical parity.'}


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,old,prior=checked_collection(parent);job=Api().json('GET','/api/jobs/'+c['jobID'])
    if job['state']!='succeeded' or job['payload_custody']!=c['custody'] or job['config']['requestSHA256']!=prior['requestSHA256']:raise ValueError('Prediction parent differs')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    save(p/'source.json',{'predictionSeal':{'path':job['result']['diagnostics']['cwd']+'/predictions/prediction-seal.json','sha256':sha(parent/'result/prediction-seal.json')},
        'serve':read(parent/'result/serve.json'),'references':read(parent/'result/reference-receipt.json')['rows'],'pairs':prior['pairs']})
    for n in ['transport_consumer.py','context_statistics_study.py','encoder_atoms.py','encoder-atoms-source.json']:shutil.copyfile(HERE/n,p/n)
    r={'schema':'tce-transport-consumer-request-v1','parentJobID':job['job_id'],'parentRequestSHA256':prior['requestSHA256'],
        'scope':'Offline receiving-operation diagnostic. Predicted head0 with reference other31heads, original output/RMS/MLP equations; not own-state complete encoder or zero-neural serving.',
        'arms':ARMS,'probePolicy':'Common mask-clock positions within eachpair:0,bothlastactive,min(maxactive,511),511. Output/residual/RMS all512rows; MLP/state only these predeclared clocks.',
        'primaryObservable':'Predicted head0-specific effect relative to head0 deletion, and its paired change; full endpoint/fullprompt changes secondary because supplied heads/embedding can dominate.',
        'numericPolicy':'BF16 exact mathematical head bank; FP64 otherheads output projection grouped separately then add head0 slice. Explicit BF16 writer/residual/RMS/MLP/state. Nativebackend parity unmeasured.',
        'sourceHashes':{f.name:sha(f) for f in p.iterdir()},'device':'cpu','backend':'numpy-fp64','cpuThreads':1}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from response_study import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for name,digest in r['sourceHashes'].items():
        if sha(root/'payload'/name)!=digest:raise ValueError('Staged source changed')
    cmd=command()[:-5]+['python','-u','transport_consumer.py','execute','.']
    # This source-qualified tokenizer environment is cached; no neural framework is imported.
    from run import command as prefix_command
    cmd=prefix_command()[:-5]+cmd[-5:];cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-query-transport-consumer-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512},timeout_s=600,
        detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 original-equation consumer:8promptseach32exactheads, predictedhead0 plus exactother31, output/residual/RMSall512, MLP4sourceclockrows.4arms/deletion-scaledmetrics. Original5GBshard hashed; no fullmodel/GPU/framework; expected under3min,finite600s,no retry.')
    result={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',result);return result


def loaded(item):
    if sha(item['path'])!=item['sha256']:raise ValueError('Sealed array changed')
    with np.load(item['path'],allow_pickle=False) as f:return {key:f[key] for key in f.files}


def effect_scores(pred,exact,zero,valid):
    valid=valid&np.isfinite(pred).all(1)&np.isfinite(exact).all(1)
    return {'queries':len(valid),'validQueries':int(valid.sum()),'invalidQueries':int(np.count_nonzero(~valid)),
        'head0Effect':summary(pred[valid]-zero[valid],exact[valid]-zero[valid]) if valid.any() else None,
        'endpoint':summary(pred[valid],exact[valid]) if valid.any() else None}


def execute(root):
    if NoNeural.roots.intersection(sys.modules):raise RuntimeError('Neural framework already imported')
    sys.meta_path.insert(0,NoNeural());root=Path(root).resolve();r=read(root/'request.json');start=time.perf_counter()
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for name,digest in r['sourceHashes'].items():
        if sha(root/name)!=digest:raise ValueError('Source changed')
    source=read(root/'source.json');seal_ref=source['predictionSeal']
    if sha(seal_ref['path'])!=seal_ref['sha256']:raise ValueError('Parent prediction seal changed')
    seal=read(seal_ref['path']);params,config,param_receipt=parameters(read(root/'encoder-atoms-source.json'));prefix=Prefix(source['serve']['store'],source['serve']['manifestSHA256'])
    out=root/'response-result';out.mkdir(exist_ok=False);work=root/'consumer-arrays';work.mkdir();positions={}
    for a,b in source['pairs']:
        aa,bb=[next(p['activeTokens'] for p in seal['ports'] if p['id']==name) for name in [a,b]]
        rows=sorted({0,aa-1,bb-1,min(max(aa,bb),511),511});positions[a]=positions[b]=np.asarray(rows,dtype=np.int64)
    outputs={};validity={};rows=[];scores=[];pairs=[];reference_checks=[];observations=0
    with (out/'observations.jsonl').open('x') as f:
        for item in source['serve']['prompts']:
            name=item['id'];record=next(z for z in seal['rows'] if z['id']==name);bundle=loaded(record);full,_=prefix.forward(item['prompt'])
            h=config['hidden_size'];heads=[]
            for head in range(32):
                kv=head//4;packet={'q':full['qpos'][:,head*128:(head+1)*128],'k':full['kpos'][:,kv*128:(kv+1)*128],
                    'v':full['v'][:,kv*128:(kv+1)*128],'mask':full['mask']}
                heads.append(exact_head(packet)['rounded'])
            bank=np.concatenate(heads,axis=1);exact0=floats(heads[0]).astype(float);target=loaded(next(z for z in source['references'] if z['id']==name))
            reference_checks.append({'id':name,'head0ReferenceWordMismatches':int(np.count_nonzero(heads[0]!=target['rounded']))})
            rest=floats(bank[:,128:]).astype(float)@params['output'][:,128:].T;matrix=params['output'][:,:128];probe=positions[name]
            context_path=work/(name+'-context.npz');np.savez(context_path,e=full['e'],headBank=bank,rest=rest,positions=probe,mask=full['mask'])
            results={};valid={}
            for arm in ARMS:
                mask=np.ones(512,dtype=bool) if arm in ['zero','exact'] else bundle[arm+'_rounded_valid'].astype(bool)
                head=np.zeros_like(exact0) if arm=='zero' else exact0 if arm=='exact' else floats(bundle[arm+'_rounded']).astype(float)
                results[arm]=consume(full['e'],rest,head,matrix,params['gamma'],config['rms_norm_eps'],probe,[params[k] for k in ['gate','up','down']]);valid[arm]=mask
            outputs[name]=results;validity[name]=valid
            path=work/(name+'-consumer.npz');np.savez(path,**{arm+'__'+key:value for arm,fields in results.items() for key,value in fields.items()},
                **{arm+'__valid':value for arm,value in valid.items()},positions=probe)
            rows.append({'id':name,'path':str(path),'sha256':sha(path),'context':{'path':str(context_path),'sha256':sha(context_path)},'probePositions':probe.tolist()})
            for arm in ['nearest64','transport64']:
                for stage in ['projection','residual','rms','mlp','state']:
                    pred,exact,zero=[floats(results[a][stage]).astype(float) for a in [arm,'exact','zero']];mask=valid[arm] if len(pred)==512 else valid[arm][probe]
                    scores.append({'id':name,'arm':arm,'stage':stage,**effect_scores(pred,exact,zero,mask)})
                    er,tr=[rowrms(z) for z in [pred-exact,exact-zero]];clocks=np.arange(512) if len(pred)==512 else probe
                    for i,pos in enumerate(clocks):
                        f.write(json.dumps({'id':name,'arm':arm,'stage':stage,'position':int(pos),'active':bool(full['mask'][pos]),'valid':bool(mask[i]),
                            'consumerErrorRms':float(er[i]),'head0EffectRms':float(tr[i])},allow_nan=False)+'\n');observations+=1
    for a,b in source['pairs']:
        for arm in ['nearest64','transport64']:
            for stage in ['projection','residual','rms','mlp','state']:
                exact,zero,pred=[floats(outputs[b][name][stage]).astype(float)-floats(outputs[a][name][stage]).astype(float) for name in ['exact','zero',arm]]
                valid=validity[a][arm]&validity[b][arm]
                if len(pred)!=512:valid=valid[positions[a]]
                pairs.append({'a':a,'b':b,'arm':arm,'stage':stage,**effect_scores(pred,exact,zero,valid)})
    if sha(seal_ref['path'])!=seal_ref['sha256']:raise ValueError('Parent seal changed')
    shutil.copyfile(seal_ref['path'],out/'prediction-seal.json');shutil.copyfile(root/'request.json',out/'request.json')
    save(out/'source-receipt.json',{'predictionSeal':seal_ref,'parentInputs':source,'parameters':param_receipt,
        'suppliedContext':'Other31heads are exact mathematical reference reads for each prompt, not predicted/native-free operands. E is compiled lexical input. No fullencoder/conditioning/image closure.'})
    save(out/'reference-receipt.json',{'rows':rows,'head0ReferenceChecks':reference_checks,
        'scope':'Original-equation offline consumer arrays. All512writer/residual/RMS rows; source-clock MLP/state probes only. Parent seal is inherited, not a separately forecast fullencoder.'})
    save(out/'report.json',{'schema':'tce-transport-consumer-result-v1','requestSHA256':r['requestSHA256'],'scope':r['scope'],'scores':scores,'pairedScores':pairs,
        'parameters':param_receipt,'reference':read(out/'reference-receipt.json'),'predictionSealSHA256':seal_ref['sha256'],
        'workerSeconds':time.perf_counter()-start,'observations':observations,'originalEquationWork':{'exactHeadEvaluations':32*len(outputs),
            'sharedOtherHeadOutputProjections':len(outputs),'head0SliceProjections':4*len(outputs),'mlpDenseProjections':12*len(outputs),'postAttentionRMSCalls':4*len(outputs)},
        'frameworkCalls':0,'originalLearnedEquationsEvaluated':True,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'goalStatus':'Active. Receiving-operation diagnostic with explicit reference context/original weights, not closedencoder or strictzero-neural image serving.'})
    print('TRANSPORT_CONSUMER_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
