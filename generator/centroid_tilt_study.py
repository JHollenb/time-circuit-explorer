"""Matched tail laws with frozen eligible and already-selected oracle supports."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from encoder_atoms import np,read,save,sha,identity,floats,words,NoNeural
from attention_bank import geometry,packet,head_asset
from attention_bank_study import loaded,verify_request,regions,effect_scores,STAGES
from context_statistics_study import summary,rowrms
from sparse_tilt import mixture
from sparse_tilt_study import logsum,restricted_read,case_scores
from centroid_tilt import same_support
from transport_consumer import parameters,consume

HERE=Path(__file__).resolve().parent
METHODS=['eligible-unit','eligible-centroid','oracle-unit','oracle-centroid']


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,j,r=checked_collection(parent,current_workers=False);api=Api();live=api.json('GET','/api/jobs/'+c['jobID'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody'] or live['config']['requestSHA256']!=r['requestSHA256']:raise ValueError('Sparse parent differs')
    ancestor=api.json('GET','/api/jobs/'+r['parentJobID'])
    if ancestor['state']!='succeeded' or ancestor['config']['requestSHA256']!=r['parentRequestSHA256']:raise ValueError('Complete-bank ancestor differs')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir();old=read(parent/'result/source.json');seal=read(parent/'result/prediction-seal.json');ref=read(parent/'result/reference-receipt.json')
    source={'parentSeal':{'path':j['result']['diagnostics']['cwd']+'/predictions/prediction-seal.json','sha256':sha(parent/'result/prediction-seal.json')},
        'asset':old['asset'],'ports':read(parent/'result/serve-source.json')['ports'],'geometry':old['geometry'],'pairs':old['pairs'],
        'primary':[x for x in seal['rows'] if x['budget']==4],'parentStates':[x for x in seal['states'] if x['budget']==4],
        'oracle':[x for x in ref['rows'] if x['kind']=='oracle-diagnostics' and x['budget']==4],
        'parentConsumers':[x for x in ref['rows'] if x['kind']=='consumers'],'references':old['references'],
        'parentConsumerParameterSource':old['parentConsumerParameterSource'],'completeBankRoot':ancestor['result']['diagnostics']['cwd']}
    save(p/'source.json',source)
    save(p/'serve-source.json',{k:source[k] for k in ['asset','ports','geometry']}|{'budget':4,'primarySupport':source['primary']})
    files=['centroid_tilt.py','centroid_tilt_study.py','sparse_tilt.py','sparse_tilt_study.py','attention_bank.py','attention_bank_study.py',
        'encoder_atoms.py','encoder-atoms-source.json','context_source_geometry.py','context_landmarks.py','context_transport.py',
        'context_statistics.py','context_statistics_study.py','context_transport_study.py','transport_consumer.py']
    for n in files:shutil.copyfile(HERE/n,p/n)
    request={'schema':'tce-tail-centroid-request-v1','parentJobID':c['jobID'],'parentRequestSHA256':r['requestSHA256'],
        'completeBankJobID':r['parentJobID'],'completeBankRequestSHA256':r['parentRequestSHA256'],'pairs':source['pairs'],'budget':4,
        'scope':'Retrospective saved pair, all32heads/512rows. Freeze old eligible/oracle support sets and routes; matched unit/centroid tail laws. No new rawtext validation, selector fit, fullencoder or zero-neural images.',
        'law':'Tail atom has original anchor mass,value mean,key mean; multiply by exp(delta dot(mu_tail-mu_full)). Retained atoms use exp(delta dot(y-mu_full)). Common centering cancels. Existing Z/N/Y, no new persistent state orCvy.',
        'sealPolicy':'Fresh child advances own eligible unit/centroid banks and verifies frozen support/routes before seal. Oracle support/target/original consumer parameter reads only afterward. No reranking of retained oracle indices.',
        'observables':'Paired fullbank/fullblock response; active/padding/before8/at8/after8; paired head rows by identical/changed valid-position sets, diagnostic not causal isolation. Exact/zero and parent unit consistency. Preserve all countertrends and raw errors.',
        'sourceHashes':{f.name:sha(f) for f in p.iterdir()},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':600}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    verify_request(root/'payload',r);cmd=command()[:-5]+['python','-u','centroid_tilt_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-tail-centroid-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512,'heads':32,'budget':4},
        timeout_s=r['timeoutSeconds'],detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 matched m4tail law,2savedportsx32headsx512queries. Eligible unit/centroid ownstate banks seal first; frozen old oracle indices afterward, no rerank. Then6original fullblock consumer arms/prompt, all512rows. Host-local immutable parents/5GBweight shard; default admission, expected under3min, finite600s,no retry.')
    result={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',result);return result


def direct_centroid(score,U,V,selected,delta):
    """Independent full-prefix construction of the same centroid mixture."""
    tail=np.ones(len(score),dtype=bool);tail[selected]=False;logs=list(score[selected]);keys=list(U[selected]);values=list(V[selected])
    if tail.any():
        logz=logsum(score[tail]);w=np.exp(score[tail]-logz)
        logs.append(logz);keys.append(w@U[tail]);values.append(w@V[tail])
    return mixture(np.asarray(logs)+np.asarray(keys)@delta,np.asarray(values))[0]


def frozen_reads(p,asset,primary,frozen):
    q,k,v=[floats(p[key]).astype(float) for key in ['q','k','v']];n,d=q.shape;u=k/d**.25-asset['keyCenter']
    result={a:np.zeros((n,d)) for a in ['unit','centroid']};closure={a:np.zeros(n) for a in ['unit','centroid']}
    for i in range(n):
        allowed=np.flatnonzero(p['mask'][:i+1]);positions=frozen[i];positions=positions[positions>=0]
        if len(set(positions))!=len(positions) or not set(positions).issubset(set(allowed)) or len(positions)!=min(len(allowed),frozen.shape[1]):raise ValueError('Frozen oracle support is not a causal set')
        if not len(allowed):continue
        U=u[allowed];V=v[allowed];ell=asset['landmarks'][primary['nearestIndex'][i]];score=U@ell
        w=np.exp(score-logsum(score));mu=w@U;delta=q[i]/d**.25-ell;t=(U-mu)@delta
        eligible=np.flatnonzero(np.isin(allowed,primary['supportPositions'][i]));chosen=np.flatnonzero(np.isin(allowed,positions))
        result['unit'][i]=restricted_read(score,t,V,chosen);result['centroid'][i]=direct_centroid(score,U,V,chosen,delta)
        closure['unit'][i]=np.max(np.abs(restricted_read(score,t,V,eligible)-primary['unit'][i]))
        closure['centroid'][i]=np.max(np.abs(direct_centroid(score,U,V,eligible,delta)-primary['centroid'][i]))
    return result,closure


def execute(root):
    if NoNeural.roots.intersection(sys.modules):raise RuntimeError('Neural framework imported')
    sys.meta_path.insert(0,NoNeural());root=Path(root).resolve();r=read(root/'request.json');verify_request(root,r);start=time.perf_counter()
    source=read(root/'source.json');out=root/'response-result';out.mkdir(exist_ok=False)
    if sha(source['parentSeal']['path'])!=source['parentSeal']['sha256']:raise ValueError('Sparse parent seal changed')
    tick=time.perf_counter();subprocess.run([sys.executable,str(root/'centroid_tilt.py'),str(root/'serve-source.json'),str(root/'predictions')],check=True)
    child_s=time.perf_counter()-tick;seal_path=root/'predictions/prediction-seal.json';seal_sha=sha(seal_path);seal=read(seal_path)
    print('ALL_ELIGIBLE_TAIL_LAWS_SEALED',seal_sha,flush=True)
    asset=loaded(source['asset']);m=source['geometry'];d,heads,_=geometry(m);work=root/'reference-work';work.mkdir()
    tick=time.perf_counter();params,config,param_receipt=parameters(read(root/'encoder-atoms-source.json'));parameter_s=time.perf_counter()-tick
    if param_receipt['source']!=source['parentConsumerParameterSource']:raise ValueError('Consumer source differs')
    all_results={};all_banks={};validity={};supports={};masks={};refrows=[];checks=[];scores=[];paired=[];head_scores=[];membership=[];timings=[];observations=0
    with (out/'observations.jsonl').open('x') as f:
        for item in source['ports']:
            name=item['id'];port=loaded(item);n=len(port['mask']);masks[name]=port['mask'].astype(bool);tick=time.perf_counter()
            primary={law:loaded(next(x for x in seal['rows'] if x['id']==name and x['method']=='eligible-'+law)) for law in ['unit','centroid']}
            old_primary=loaded(next(x for x in source['primary'] if x['id']==name));old_oracle=loaded(next(x for x in source['oracle'] if x['id']==name))
            old_consumers=loaded(next(x for x in source['parentConsumers'] if x['id']==name));parent=loaded(next(x for x in source['references'] if x['id']==name))
            banks={'eligible-'+law:primary[law]['rounded'] for law in primary};raw_oracle={law:np.zeros((n,heads*d)) for law in ['unit','centroid']};closure={law:np.zeros((n,heads)) for law in ['unit','centroid']}
            for head in range(heads):
                span=slice(head*d,(head+1)*d);p={'nearestIndex':primary['unit']['nearestIndex'][:,head],'supportPositions':primary['unit']['supportPositions'][:,head],
                    **{law:primary[law]['raw'][:,span] for law in primary}}
                diagnostic,residual=frozen_reads(packet(port,m,head),head_asset(asset,m,head),p,old_oracle['supportPositions'][:,head])
                for law in primary:raw_oracle[law][:,span]=diagnostic[law];closure[law][:,head]=residual[law]
            banks.update({'oracle-'+law:words(raw_oracle[law]) for law in primary});banks.update(exact=parent['exactBank'],zero=np.zeros_like(parent['exactBank']))
            valid={method:np.all(primary[method.split('-')[1]]['valid'],axis=1) if method.startswith('eligible') else np.isfinite(raw_oracle[method.split('-')[1]]).all(1) for method in METHODS}
            frozen={'eligible':primary['unit']['supportPositions'],'oracle':old_oracle['supportPositions']}
            path=work/(name+'-frozen-oracle.npz');np.savez(path,**{law+'_raw':a for law,a in raw_oracle.items()},supportPositions=frozen['oracle'],**{law+'_closure':a for law,a in closure.items()})
            refrows.append({'id':name,'kind':'frozen-oracle','path':str(path),'sha256':sha(path)})
            state_mismatch={}
            for head in range(heads):
                states=[loaded(next(x for x in source['parentStates'] if x['id']==name and x['queryHead']==head))]+[loaded(next(x for x in seal['states'] if x['id']==name and x['queryHead']==head and x['law']==law)) for law in primary]
                state_mismatch[str(head)]=sum(not np.array_equal(states[0][key],state[key]) for state in states[1:] for key in states[0])
            checks.append({'id':name,'eligibleUnitParentWordMismatches':int(np.count_nonzero(banks['eligible-unit']!=old_primary['rounded'])),
                'oracleUnitParentWordMismatches':int(np.count_nonzero(banks['oracle-unit']!=old_oracle['rounded'])),
                'directEligibleMaxAbsClosure':{law:float(a.max()) for law,a in closure.items()},'stateFieldMismatchesByHead':state_mismatch,'oracleSupportReranked':False})
            diagnostic_s=time.perf_counter()-tick;tick=time.perf_counter();results={}
            for method,bank in banks.items():results[method]=consume(port['e'],np.zeros((n,config['hidden_size'])),floats(bank).astype(float),params['output'],params['gamma'],config['rms_norm_eps'],np.arange(n),[params[k] for k in ['gate','up','down']])
            mismatch={method:{stage:int(np.count_nonzero(results[method][stage]!=(parent[method+'__'+stage] if method in ['exact','zero'] else old_consumers[('m4' if method.startswith('eligible') else 'oracle4')+'__'+stage]))) for stage in STAGES} for method in ['exact','zero','eligible-unit','oracle-unit']}
            checks.append({'id':name,'consumerParentWordMismatches':mismatch});timings.append({'id':name,'frozenDiagnosticsSeconds':diagnostic_s,'sixConsumerArmsSeconds':time.perf_counter()-tick})
            path=work/(name+'-consumers.npz');np.savez(path,**{method+'__'+stage:a for method,fields in results.items() for stage,a in fields.items()})
            refrows.append({'id':name,'kind':'consumers','path':str(path),'sha256':sha(path)})
            all_results[name]=results;all_banks[name]=banks;validity[name]=valid;supports[name]=frozen
            rows,count=case_scores({'id':name,'kind':'endpoint'},results,valid,regions(masks[name]),METHODS,f);scores+=rows;observations+=count
            print('MATCHED_TAIL_CONSUMERS_COMPLETE',name,flush=True)
        a,b=source['pairs'][0];clock=np.arange(len(masks[a]));pair_regions=regions(masks[a],masks[b])+[('before8',clock<8),('at8',clock==8),('after8',clock>8)]
        target=(floats(all_banks[b]['exact']).astype(float)-floats(all_banks[a]['exact']).astype(float)).reshape(-1,heads,d)
        for method in METHODS:
            cls=method.split('-')[0];same=same_support(supports[a][cls],supports[b][cls]);valid=validity[a][method]&validity[b][method]
            delta=(floats(all_banks[b][method]).astype(float)-floats(all_banks[a][method]).astype(float)).reshape(-1,heads,d)
            for region,selected in pair_regions:
                if not selected.any():continue
                use=selected&valid
                for head in [None,*range(heads)]:
                    pred,exact=(delta[use],target[use]) if head is None else (delta[use,head],target[use,head])
                    head_scores.append({'method':method,'queryHead':head,'region':region,'validQueries':int(use.sum()),'score':summary(pred,exact) if use.any() else None})
                for label,chosen in [('unchanged',same),('changed',~same)]:
                    rows=use[:,None]&chosen
                    membership.append({'method':method,'region':region,'membership':label,'headQueryRows':int(rows.sum()),'score':summary(delta[rows],target[rows]) if rows.any() else None})
            for stage in STAGES:
                pred,exact,zero=[floats(all_results[b][arm][stage]).astype(float)-floats(all_results[a][arm][stage]).astype(float) for arm in [method,'exact','zero']]
                for region,selected in pair_regions:
                    if selected.any():paired.append({'a':a,'b':b,'method':method,'stage':stage,'region':region,**effect_scores(pred,exact,zero,selected,valid)})
                er,tr=[rowrms(z) for z in [pred-exact,exact-zero]]
                for i in range(len(pred)):
                    f.write(json.dumps({'a':a,'b':b,'kind':'paired','method':method,'stage':stage,'position':i,'valid':bool(valid[i]),'consumerErrorRms':float(er[i]),'attentionEffectRms':float(tr[i])},allow_nan=False)+'\n');observations+=1
    if sha(seal_path)!=seal_sha:raise ValueError('Eligible seal changed')
    for name,path in [('request.json',root/'request.json'),('source.json',root/'source.json'),('serve-source.json',root/'serve-source.json'),('prediction-seal.json',seal_path)]:shutil.copyfile(path,out/name)
    save(out/'reference-receipt.json',{'rows':refrows,'checks':checks,'parameters':param_receipt,'scope':'Frozen old oracle supports, no reranking. Shared original fullblock equations after eligible seal, not nativebackend parity.'})
    save(out/'membership-diagnostics.json',{'rows':membership,'scope':'Paired head-query errors stratified by equality of valid position sets, not heap order. Diagnostic association, not causal isolation of membership.'})
    save(out/'report.json',{'schema':'tce-tail-centroid-result-v1','requestSHA256':r['requestSHA256'],'predictionSealSHA256':seal_sha,'scope':r['scope'],
        'serving':seal,'scores':scores,'pairedScores':paired,'pairedHeadScores':head_scores,'checks':checks,'timings':timings,
        'servingChildSeconds':child_s,'originalParameterReadSeconds':parameter_s,'workerSeconds':time.perf_counter()-start,'observations':observations,
        'originalEquationWork':{'fullOutputProjections':12,'mlpDenseProjections':36,'rowsPerArm':512},'frameworkCalls':0,'originalLearnedConsumerEquationsEvaluated':True,
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed','goalStatus':'Active. Matched retrospective reduced attention law with original learned consumer; no zero-neural image or freshrawprompt validation.'})
    print('TAIL_CENTROID_STUDY_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
