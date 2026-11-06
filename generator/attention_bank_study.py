"""Seal complete predicted attention banks, then evaluate original consumers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from encoder_atoms import np,Prefix,read,save,sha,identity,floats,NoNeural
from attention_bank import packet,geometry,head_asset
from context_transport import METHODS
from context_statistics_study import exact_head,summary,rowrms
from context_transport_study import diagnostics
from transport_consumer import parameters,consume

HERE=Path(__file__).resolve().parent
PROMPTS=[{'id':'crane-behind','prompt':'An orange paper crane stands behind a silver key on a dark wooden table.'},
         {'id':'crane-before','prompt':'An orange paper crane stands before a silver key on a dark wooden table.'}]
STAGES=['projection','residual','rms','mlp','state']


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,j,r=checked_collection(parent)
    live=Api().json('GET','/api/jobs/'+c['jobID'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody'] or live['config']['requestSHA256']!=r['requestSHA256']:raise ValueError('Source parent identity differs')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir();spec=read(parent/'result/serve.json')
    shutil.copyfile(parent/'result/source-ids.npy',p/'source-ids.npy')
    save(p/'source.json',{'store':spec['store'],'manifestSHA256':spec['manifestSHA256'],'sampleIDsSHA256':sha(p/'source-ids.npy'),
        'landmarks':64,'candidates':4096,'seed':1909,'batch':64,'prompts':PROMPTS})
    for name in ['attention_bank.py','attention_bank_study.py','encoder_atoms.py','encoder-atoms-source.json',
                 'context_source_geometry.py','context_landmarks.py','context_transport.py','context_statistics.py',
                 'context_statistics_study.py','context_transport_study.py','transport_consumer.py']:shutil.copyfile(HERE/name,p/name)
    request={'schema':'tce-complete-attention-bank-request-v1','parentJobID':c['jobID'],'parentRequestSHA256':r['requestSHA256'],
        'scope':'One fresh raw-prompt pair, all32first-layer query heads predicted with no exact replacement; original first-block consumer and exact attention offline after whole-bank seal. Not zero-neural encoder/images.',
        'sourceRule':'Same2048sourceIDsx512positions. Eight KV covariance computations,64independent landmarks for each32query heads. PCG64seed1909 lexical4096+all512padpositions,50/50source mixture; deterministic farthest-first. No pooled landmarks/compressed coordinates.',
        'predictionRule':'Source-metric nearest reader plus actual-prefix centered Cov(v,y) displacement correction. Sequential head execution; full logical moment state charged. No clipping/exact fallback/reference head refresh.',
        'interfacePolicy':'Strings fixed in hashed source.json; IDs/masks sealed before source compilation; all512queries retained. Complete nearest/transport bank and all final moment-state hashes sealed before original parameter reads/exact heads.',
        'consumerPolicy':'Original full W_o over complete rounded bank, BF16 residual, FP32 RMS variance, original gated SiLU MLP and final residual on all512rows. Shared FP64/BF16 equation reference, nativebackend parity unmeasured.',
        'primaryObservable':'Paired firstblock state effect relative to attention-bank deletion, active/padding/maskchanged splits. Full endpoints, perhead linear output contributions and signed cancellation retained.',
        'arms':['zero','exact',*METHODS],'pairs':[[x['id'] for x in PROMPTS]],'sourceHashes':{f.name:sha(f) for f in p.iterdir()},
        'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':900}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    verify_request(root/'payload',r)
    cmd=command()[:-5]+['python','-u','attention_bank_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-complete-attention-bank-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512,'heads':32,'landmarks':64},
        timeout_s=r['timeoutSeconds'],detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 all32heads/8KVcovariances from2048sourceIDsx512; one fresh pair,64landmarks/head,sequential moment banks. Seal full predictions before64exactheads+4arms full512row output/RMS/MLP/state consumers. No reference-head refresh; original learned consumer offline. Host-local source store/weights; default admission sizing, expected under8min,finite900s,no retry.')
    result={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',result);return result


def verify_request(root,r):
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for name,h in r['sourceHashes'].items():
        if sha(root/name)!=h:raise ValueError('Staged source changed')


def loaded(record):
    if sha(record['path'])!=record['sha256']:raise ValueError('Sealed array changed')
    with np.load(record['path'],allow_pickle=False) as f:return {k:f[k] for k in f.files}


def regions(a,b=None):
    a=np.asarray(a,dtype=bool)
    if b is None:return [('all',np.ones(len(a),dtype=bool)),('active',a),('padding',~a)]
    b=np.asarray(b,dtype=bool)
    return [('all',np.ones(len(a),dtype=bool)),('active_both',a&b),('padding_both',~a&~b),('mask_changed',a!=b)]


def effect_scores(pred,exact,zero,selected,valid):
    use=selected&valid&np.isfinite(pred).all(1)&np.isfinite(exact).all(1)&np.isfinite(zero).all(1)
    return {'queries':int(selected.sum()),'validQueries':int(use.sum()),'invalidQueries':int(np.count_nonzero(selected&~use)),
        'attentionEffect':summary(pred[use]-zero[use],exact[use]-zero[use]) if use.any() else None,
        'endpoint':summary(pred[use],exact[use]) if use.any() else None}


def linear_head_effects(pred,exact,matrix,masks,d,valid):
    """Linear output-error decomposition before the shared final BF16 boundary."""
    error=pred-exact;total=error@matrix.T;rows=[];summed=np.zeros_like(total);region_energy={name:0. for name,mask in masks if mask.any()}
    for head in range(pred.shape[1]//d):
        part=error[:,head*d:(head+1)*d]@matrix[:,head*d:(head+1)*d].T;summed+=part
        for name,selected in masks:
            use=selected&valid
            if not use.any():continue
            energy=float(np.sum(part[use]*part[use]));region_energy[name]+=energy
            rows.append({'queryHead':head,'region':name,'queries':int(use.sum()),
                'head':summary(pred[use,head*d:(head+1)*d],exact[use,head*d:(head+1)*d]),
                'linearOutputErrorEnergy':energy,'linearOutputErrorRms':float(np.sqrt(np.mean(part[use]*part[use]))),
                'signedErrorDotTotal':float(np.sum(part[use]*total[use]))})
    cancellation=[]
    for name,selected in masks:
        use=selected&valid
        if not use.any():continue
        energy=float(np.sum(total[use]*total[use]));summed_energy=region_energy[name]
        cancellation.append({'region':name,'queries':int(use.sum()),'sumHeadErrorEnergies':summed_energy,'totalErrorEnergy':energy,
            'crossHeadSignedEnergy':energy-summed_energy,'sumHeadEnergyOverTotal':summed_energy/energy if energy else None})
    return {'heads':rows,'cancellation':cancellation,'linearSumMaxAbsClosure':float(np.max(np.abs(summed-total))),
        'scope':'Pre-BF16 linear W_o error contributions. Negative cross-head signed energy indicates cancellation; this is not an additive MLP/state decomposition.'}


def score_case(label,results,valid,masks,stream):
    scores=[];count=0
    for method in METHODS:
        for stage in STAGES:
            pred,exact,zero=[floats(results[a][stage]).astype(float) for a in [method,'exact','zero']]
            for region,selected in masks:
                if selected.any():scores.append({**label,'method':method,'stage':stage,'region':region,**effect_scores(pred,exact,zero,selected,valid[method])})
            error,target=[rowrms(z) for z in [pred-exact,exact-zero]]
            for i in range(len(pred)):
                stream.write(json.dumps({**label,'method':method,'stage':stage,'position':i,'valid':bool(valid[method][i]),
                    'consumerErrorRms':float(error[i]),'attentionEffectRms':float(target[i])},allow_nan=False)+'\n');count+=1
    return scores,count


def execute(root):
    if NoNeural.roots.intersection(sys.modules):raise RuntimeError('Neural framework already imported')
    sys.meta_path.insert(0,NoNeural());root=Path(root).resolve();r=read(root/'request.json');verify_request(root,r);started=time.perf_counter()
    source=read(root/'source.json');out=root/'response-result';out.mkdir(exist_ok=False)
    if sha(root/'source-ids.npy')!=source['sampleIDsSHA256']:raise ValueError('Source IDs changed')
    prefix=Prefix(source['store'],source['manifestSHA256']);interfaces={}
    for item in source['prompts']:
        text,ids,mask=prefix.tokenize(item['prompt']);interfaces[item['id']]={'ids':ids.tolist(),'mask':mask.tolist(),
            'renderedTextSHA256':hashlib.sha256(text.encode()).hexdigest(),'activeTokens':int(mask.sum())}
    del prefix
    save(root/'prompt-interfaces.json',{'requestSHA256':r['requestSHA256'],'prompts':interfaces,'scope':'Tokenization frozen before source compilation and all contextual references.'})
    compile_spec={**{k:source[k] for k in ['store','manifestSHA256','landmarks','candidates','seed','batch']},
        'sampleIDs':{'path':str(root/'source-ids.npy'),'sha256':source['sampleIDsSHA256']}}
    save(root/'compile.json',compile_spec);t=time.perf_counter()
    subprocess.run([sys.executable,str(root/'attention_bank.py'),'compile',str(root/'compile.json'),str(root/'compiled')],check=True)
    compile_s=time.perf_counter()-t
    serve_spec={**{k:source[k] for k in ['store','manifestSHA256','prompts']},
        'asset':{'path':str(root/'compiled/bank-assets.npz'),'sha256':sha(root/'compiled/bank-assets.npz')},
        'interfaces':{'path':str(root/'prompt-interfaces.json'),'sha256':sha(root/'prompt-interfaces.json')}}
    save(root/'serve.json',serve_spec);t=time.perf_counter()
    subprocess.run([sys.executable,str(root/'attention_bank.py'),'serve',str(root/'serve.json'),str(root/'predictions')],check=True)
    serve_s=time.perf_counter()-t;seal_path=root/'predictions/prediction-seal.json';seal_sha=sha(seal_path);seal=read(seal_path)
    print('COMPLETE_ATTENTION_BANKS_SEALED',seal_sha,flush=True)
    # Only now may original parameters and exact contextual references be read.
    t=time.perf_counter();params,config,param_receipt=parameters(read(root/'encoder-atoms-source.json'));parameter_s=time.perf_counter()-t
    prefix=Prefix(source['store'],source['manifestSHA256']);d,heads,groups=geometry(prefix.m);work=root/'reference-work';work.mkdir()
    with np.load(serve_spec['asset']['path'],allow_pickle=False) as f:asset={k:f[k] for k in f.files}
    results={};validity={};banks={};masks={};refrows=[];canonical=[];direct=[];scores=[];paired=[];linear=[];observations=0;timings=[]
    with (out/'observations.jsonl').open('x') as f:
        for item in source['prompts']:
            name=item['id'];record=next(z for z in seal['rows'] if z['id']==name);candidate=loaded(record);port=loaded(record['port']);full,_=prefix.forward(item['prompt'])
            canonical.append({'id':name,'mismatches':{k:int(np.count_nonzero(port[k]!=full[k])) for k in port}})
            t=time.perf_counter();exact=[]
            for head in range(heads):
                target=exact_head(packet(full,prefix.m,head));exact.append(target['rounded'])
                a=head_asset(asset,prefix.m,head);bundle={k:candidate[k][:,head*d:(head+1)*d] for k in [method+'_raw' for method in METHODS]}
                bundle['nearestIndex']=candidate['nearestIndex'][:,head]
                direct.append({'id':name,'queryHead':head,'rows':diagnostics(packet(full,prefix.m,head),a,bundle,target['raw'])})
            bank=np.concatenate(exact,axis=1);exact_s=time.perf_counter()-t;n=len(bank);bank_arms={'zero':np.zeros_like(bank),'exact':bank,**{method:candidate[method+'_rounded'] for method in METHODS}}
            valid={method:np.all(candidate[method+'_valid'],axis=1) for method in METHODS};t=time.perf_counter();consumed={}
            for arm,b in bank_arms.items():
                consumed[arm]=consume(full['e'],np.zeros((n,config['hidden_size'])),floats(b).astype(float),params['output'],params['gamma'],config['rms_norm_eps'],np.arange(n),[params[k] for k in ['gate','up','down']])
            consumer_s=time.perf_counter()-t;timings.append({'id':name,'exactHeadsAndDiagnosticsSeconds':exact_s,'allFourConsumerArmsSeconds':consumer_s})
            results[name]=consumed;validity[name]=valid;banks[name]=bank_arms;masks[name]=full['mask'].astype(bool)
            path=work/(name+'-consumer.npz');np.savez(path,**{arm+'__'+stage:value for arm,fields in consumed.items() for stage,value in fields.items()},
                exactBank=bank,**{method+'__valid':v for method,v in valid.items()},mask=full['mask'])
            refrows.append({'id':name,'path':str(path),'sha256':sha(path)})
            rows,count=score_case({'id':name,'kind':'endpoint'},consumed,valid,regions(masks[name]),f);scores+=rows;observations+=count
            for method in METHODS:
                linear.append({'id':name,'kind':'endpoint','method':method,**linear_head_effects(floats(bank_arms[method]).astype(float),floats(bank).astype(float),params['output'],regions(masks[name]),d,valid[method])})
            print('FULL_BLOCK_CONSUMED',name,flush=True)
        a,b=r['pairs'][0]
        # Scores below use FP64 differences of BF16 states; no extra rounding.
        valid={method:validity[a][method]&validity[b][method] for method in METHODS};pair_regions=regions(masks[a],masks[b])
        for method in METHODS:
            for stage in STAGES:
                pred,exact,zero=[floats(results[b][arm][stage]).astype(float)-floats(results[a][arm][stage]).astype(float) for arm in [method,'exact','zero']]
                for region,selected in pair_regions:
                    if selected.any():paired.append({'a':a,'b':b,'method':method,'stage':stage,'region':region,**effect_scores(pred,exact,zero,selected,valid[method])})
                er,tr=[rowrms(z) for z in [pred-exact,exact-zero]]
                for i in range(len(pred)):
                    f.write(json.dumps({'a':a,'b':b,'kind':'paired','method':method,'stage':stage,'position':i,
                        'activeA':bool(masks[a][i]),'activeB':bool(masks[b][i]),'valid':bool(valid[method][i]),'consumerErrorRms':float(er[i]),'attentionEffectRms':float(tr[i])},allow_nan=False)+'\n');observations+=1
            pred,exact=[floats(banks[b][arm]).astype(float)-floats(banks[a][arm]).astype(float) for arm in [method,'exact']]
            linear.append({'a':a,'b':b,'kind':'paired','method':method,**linear_head_effects(pred,exact,params['output'],pair_regions,d,valid[method])})
    if sha(seal_path)!=seal_sha:raise ValueError('Prediction seal changed')
    for name,path in [('prediction-seal.json',seal_path),('request.json',root/'request.json'),('source.json',root/'source.json'),('serve.json',root/'serve.json'),
                      ('source-ids.npy',root/'source-ids.npy'),('compile-receipt.json',root/'compiled/compile-receipt.json'),('prompt-interfaces.json',root/'prompt-interfaces.json')]:shutil.copyfile(path,out/name)
    save(out/'reference-receipt.json',{'rows':refrows,'canonicalPortChecks':canonical,'directMomentDiagnostics':direct,'parameters':param_receipt,
        'scope':'Post-whole-bank-seal original learned firstblock equations. No exact heads supplied to prediction arms. Everyconsumer stage covers all512rows. Shared math, not nativebackend parity.'})
    save(out/'linear-head-effects.json',{'cases':linear,'scope':'All head contributions and signed cancellation retained; pre-BF16 output error only.'})
    save(out/'report.json',{'schema':'tce-complete-attention-bank-result-v1','requestSHA256':r['requestSHA256'],'predictionSealSHA256':seal_sha,
        'scope':r['scope'],'serving':seal,'scores':scores,'pairedScores':paired,'timings':timings,'compileChildSeconds':compile_s,
        'servingChildSeconds':serve_s,'originalParameterReadSeconds':parameter_s,'workerSeconds':time.perf_counter()-started,'observations':observations,
        'originalEquationWork':{'exactHeadEvaluations':heads*len(results),'fullOutputProjections':4*len(results),'mlpDenseProjections':12*len(results),'rowsPerArm':512},
        'frameworkCalls':0,'originalLearnedConsumerEquationsEvaluated':True,'referenceHeadsSuppliedToPrediction':0,
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'goalStatus':'Active. Complete reduced first-layer attention bank plus original learned consumer; later contextual encoder/diffusion/decoder and strictzero-neural images remain open.'})
    print('COMPLETE_ATTENTION_BANK_STUDY_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
