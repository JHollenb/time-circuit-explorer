"""Seal full-prefix selector contrasts, then inspect signed corrections/consumers."""
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
from context_statistics_study import summary,rowrms,exact_head
from sparse_tilt_study import case_scores
from centroid_tilt_study import direct_centroid
from selector_objective import METHODS as NEW,validate_support,tilt_measure,capture
from transport_consumer import parameters,consume

HERE=Path(__file__).resolve().parent
METHODS=NEW+['anchor','witness','oracle']


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,j,r=checked_collection(parent,current_workers=False);api=Api()
    live=api.json('GET','/api/jobs/'+c['jobID']);ancestor=api.json('GET','/api/jobs/'+r['completeBankJobID'])
    oracle_parent=api.json('GET','/api/jobs/'+r['parentJobID'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody'] or live['config']['requestSHA256']!=r['requestSHA256']:raise ValueError('Witness parent differs')
    if ancestor['state']!='succeeded' or ancestor['config']['requestSHA256']!=r['completeBankRequestSHA256']:raise ValueError('Complete-bank ancestor differs')
    if oracle_parent['state']!='succeeded' or oracle_parent['config']['requestSHA256']!=r['parentRequestSHA256']:raise ValueError('Frozen-support ancestor differs')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir();old=read(parent/'result/source.json');seal=read(parent/'result/prediction-seal.json');ref=read(parent/'result/reference-receipt.json')
    source={k:old[k] for k in ['asset','ports','geometry','pairs','references','parentConsumerParameterSource','completeBankRoot','parentOracle']}
    source.update(parentSeal={'path':j['result']['diagnostics']['cwd']+'/predictions/prediction-seal.json','sha256':sha(parent/'result/prediction-seal.json')},
        parentBanks=seal['rows'],parentConsumers=[x for x in ref['rows'] if x['kind']=='consumers'])
    save(p/'source.json',source);save(p/'serve-source.json',{k:source[k] for k in ['asset','ports','geometry']}|{'budget':4})
    files=['selector_objective.py','selector_objective_study.py','centroid_tilt.py','centroid_tilt_study.py','sparse_tilt.py','sparse_tilt_study.py',
        'attention_bank.py','attention_bank_study.py','encoder_atoms.py','encoder-atoms-source.json','context_source_geometry.py','context_landmarks.py',
        'context_transport.py','context_statistics.py','context_statistics_study.py','context_transport_study.py','transport_consumer.py']
    for name in files:shutil.copyfile(HERE/name,p/name)
    request={'schema':'tce-selector-objective-request-v1','parentJobID':c['jobID'],'parentRequestSHA256':r['requestSHA256'],
        'completeBankJobID':r['completeBankJobID'],'completeBankRequestSHA256':r['completeBankRequestSHA256'],'pairs':source['pairs'],'budget':4,
        'frozenSupportParentJobID':r['parentJobID'],'frozenSupportParentRequestSHA256':r['parentRequestSHA256'],
        'scope':'Retrospective supplied pair/all32heads/512rows. Full current causal prefix selectors, not compact serving, freshrawtext, fullencoder or zero-neural images.',
        'selectors':'Top4 current a=x dot u; top4 max(anchor s,current a), fixed source-key-mean gauge, zero offsets, earlier occurrence ties. Source anchors/routes and centroid tail fixed. Old correction-mass oracle support retained without reranking.',
        'sealPolicy':'Fresh child seals complete raw/BF16 banks and support before exact heads, historical banks/oracle arrays and original consumer parameters are read.',
        'observables':'Raw/rounded endpoint/paired heads and fullblock state, active/padding/before8/at8/after8, perhead countertrends. Positive/negative/log absolute centeredtilt mass capture, signed numerator vectors and partition corrections normalized by exact tilt partition. Undefined ratios remain explicit.',
        'interpretation':'Envelope improvement is a geometry lead, confounded with removing3source witnesses. Current vs envelope tests anchor protection; both vs frozen correctionmass support informs ranking objective. No outputoptimality/efficiency claim.',
        'sourceHashes':{f.name:sha(f) for f in p.iterdir()},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':600}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    verify_request(root/'payload',r);cmd=command()[:-5]+['python','-u','selector_objective_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-selector-objective-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512,'heads':32,'budget':4},
        timeout_s=r['timeoutSeconds'],detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 saved2ports x32heads x512queries. Seal2fullprefix selectors/top4 current vs anchor-current envelope under same centroid law. Then signed positive/negative tilt diagnostics,64exact heads and4original fullblock consumers/prompt; historical anchor/witness/oracle reused unchanged. Host-local immutable parents/5GBshard; default admission,expected under3min,finite600s,no retry. Not compact serving.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def strata(mask,other=None):
    clock=np.arange(len(mask))
    return regions(mask,other)+[('before8',clock<8),('at8',clock==8),('after8',clock>8)]


def describe(values):
    values=np.asarray(values);finite=np.isfinite(values);a=values[finite]
    return {'count':int(values.size),'finiteCount':int(finite.sum()),'undefinedCount':int(np.isnan(values).sum()),
        'negativeInfinityCount':int(np.isneginf(values).sum()),'positiveInfinityCount':int(np.isposinf(values).sum()),
        'zeroCount':int(np.count_nonzero(values==0)),'mean':float(a.mean()) if len(a) else None,
        'minimum':float(a.min()) if len(a) else None,'maximum':float(a.max()) if len(a) else None}


def diagnose(p,asset,candidates,exact,budget):
    q,k,v=[floats(p[key]).astype(float) for key in ['q','k','v']];n,d=q.shape;u=k/d**.25-asset['keyCenter'];fields={}
    def put(key,i,value):
        a=np.asarray(value)
        if key not in fields:fields[key]=np.full((n,)+a.shape,np.nan)
        fields[key][i]=value
    for i in range(n):
        allowed=np.flatnonzero(p['mask'][:i+1])
        if not len(allowed):continue
        route=int(candidates['current']['nearestIndex'][i]);ell=asset['landmarks'][route];U=u[allowed];V=v[allowed];s=U@ell;delta=q[i]/d**.25-ell
        measure=tilt_measure(s,U,V,delta)
        for key in ['logD','beta','positiveLogMass','negativeLogMass','positiveMassOverD','negativeMassOverD','anchorRead',
            'fullSignedNumeratorOverD','fullSignedPartitionOverD','partitionClosure','readClosure']:put('common__'+key,i,measure[key])
        put('common__exactRawClosure',i,float(np.max(np.abs(measure['beta']*measure['anchorRead']+measure['fullSignedNumeratorOverD']-exact[i]))))
        oracle=set(candidates['oracle']['supportPositions'][i]);oracle.discard(-1)
        for method,candidate in candidates.items():
            if int(candidate['nearestIndex'][i])!=route:raise ValueError('Source route changed')
            selected=validate_support(candidate['supportPositions'][i],allowed,budget)
            for key,value in capture(measure,selected,V).items():put(method+'__'+key,i,value)
            direct=direct_centroid(s,U,V,selected,delta)
            put(method+'__directCentroidMaxAbsClosure',i,float(np.max(np.abs(direct-candidate['raw'][i]))))
            put(method+'__oracleSupportOverlap',i,len(set(allowed[selected])&oracle)/len(selected))
            put(method+'__address8Retained',i,float(8 in allowed[selected]))
    return fields


def head_scores(label,banks,target,masks,valid,d,heads):
    rows=[]
    for method in METHODS:
        for precision in ['raw','rounded']:
            pred=banks[method][precision].reshape(-1,heads,d);exact=target[precision].reshape(-1,heads,d)
            for region,selected in masks:
                use=selected&valid[method]
                if not selected.any():continue
                for head in [None,*range(heads)]:
                    a,b=(pred[use],exact[use]) if head is None else (pred[use,head],exact[use,head])
                    rows.append({**label,'method':method,'precision':precision,'queryHead':head,'region':region,
                        'validQueries':int(use.sum()),'score':summary(a,b) if use.any() else None})
    return rows


def execute(root):
    if NoNeural.roots.intersection(sys.modules):raise RuntimeError('Neural framework already imported')
    sys.meta_path.insert(0,NoNeural());root=Path(root).resolve();r=read(root/'request.json');verify_request(root,r);start=time.perf_counter()
    source=read(root/'source.json');out=root/'response-result';out.mkdir(exist_ok=False)
    tick=time.perf_counter();subprocess.run([sys.executable,str(root/'selector_objective.py'),str(root/'serve-source.json'),str(root/'predictions')],check=True);child_s=time.perf_counter()-tick
    seal_path=root/'predictions/prediction-seal.json';seal_sha=sha(seal_path);seal=read(seal_path);print('ALL_SELECTOR_BANKS_SEALED',seal_sha,flush=True)
    asset=loaded(source['asset']);m=source['geometry'];d,heads,_=geometry(m);work=root/'reference-work';work.mkdir()
    tick=time.perf_counter();params,config,param_receipt=parameters(read(root/'encoder-atoms-source.json'));parameter_s=time.perf_counter()-tick
    if param_receipt['source']!=source['parentConsumerParameterSource']:raise ValueError('Original consumer source differs')
    all_results={};all_banks={};all_targets={};validity={};masks={};refrows=[];checks=[];scores=[];paired=[];head_rows=[];diagnostic_rows=[];timings=[];observations=0
    with (out/'observations.jsonl').open('x') as f:
        for item in source['ports']:
            name=item['id'];port=loaded(item);n=len(port['mask']);mask=port['mask'].astype(bool);masks[name]=mask;tick=time.perf_counter()
            pred={method:loaded(next(x for x in seal['rows'] if x['id']==name and x['method']==method)) for method in NEW}
            pred.update({method:loaded(next(x for x in source['parentBanks'] if x['id']==name and x['method']==method)) for method in ['anchor','witness']})
            oracle=loaded(next(x for x in source['parentOracle'] if x['id']==name));pred['oracle']={'raw':oracle['centroid_raw'],'rounded':words(oracle['centroid_raw']),
                'supportPositions':oracle['supportPositions'],'nearestIndex':pred['anchor']['nearestIndex'],'valid':np.ones((n,heads),dtype=bool)}
            parent=loaded(next(x for x in source['references'] if x['id']==name));old_consumer=loaded(next(x for x in source['parentConsumers'] if x['id']==name))
            fields={};exact_raw=np.empty((n,heads*d));exact_bits=np.empty((n,heads*d),dtype=np.uint16)
            for head in range(heads):
                span=slice(head*d,(head+1)*d);p=packet(port,m,head);exact=exact_head(p);exact_raw[:,span]=exact['raw'];exact_bits[:,span]=exact['rounded']
                candidates={method:{'raw':v['raw'][:,span],'supportPositions':v['supportPositions'][:,head],'nearestIndex':v['nearestIndex'][:,head]} for method,v in pred.items()}
                for key,value in diagnose(p,head_asset(asset,m,head),candidates,exact['raw'],r['budget']).items():
                    if key not in fields:fields[key]=np.empty((n,heads)+value.shape[1:])
                    fields[key][:,head]=value
                    if value.ndim==1:
                        for region,selected in strata(mask):
                            if selected.any():diagnostic_rows.append({'id':name,'queryHead':head,'field':key,'region':region,**describe(value[selected])})
                    else:
                        for region,selected in strata(mask):
                            if selected.any():diagnostic_rows.append({'id':name,'queryHead':head,'field':key+'Rms','region':region,**describe(rowrms(value)[selected])})
            path=work/(name+'-signed-diagnostics.npz');np.savez(path,**fields);refrows.append({'id':name,'kind':'signed-diagnostics','path':str(path),'sha256':sha(path)})
            path=work/(name+'-exact-heads.npz');np.savez(path,raw=exact_raw,rounded=exact_bits);refrows.append({'id':name,'kind':'exact-heads','path':str(path),'sha256':sha(path)})
            checks.append({'id':name,'exactParentWordMismatches':int(np.count_nonzero(exact_bits!=parent['exactBank'])),
                'routeMismatches':{method:int(np.count_nonzero(v['nearestIndex']!=pred['anchor']['nearestIndex'])) for method,v in pred.items()},
                'directCentroidMaxAbsClosure':{method:float(np.nanmax(fields[method+'__directCentroidMaxAbsClosure'])) for method in METHODS},
                'partitionClosure':float(np.nanmax(fields['common__partitionClosure'])),'signedReadClosure':float(np.nanmax(fields['common__readClosure'])),
                'signedReadVsExactRawClosure':float(np.nanmax(fields['common__exactRawClosure']))})
            diagnostic_s=time.perf_counter()-tick;tick=time.perf_counter();banks={method:{'raw':v['raw'],'rounded':floats(v['rounded']).astype(float)} for method,v in pred.items()}
            consumer_banks={method:pred[method]['rounded'] for method in NEW};consumer_banks.update(exact=parent['exactBank'],zero=np.zeros_like(parent['exactBank']));results={}
            for method,bank in consumer_banks.items():results[method]=consume(port['e'],np.zeros((n,config['hidden_size'])),floats(bank).astype(float),params['output'],params['gamma'],config['rms_norm_eps'],np.arange(n),[params[k] for k in ['gate','up','down']])
            checks.append({'id':name,'consumerParentWordMismatches':{method:{stage:int(np.count_nonzero(results[method][stage]!=old_consumer[method+'__'+stage])) for stage in STAGES} for method in ['exact','zero']}})
            for method in ['anchor','witness','oracle']:results[method]={stage:old_consumer[method+'__'+stage] for stage in STAGES}
            valid={method:np.all(pred[method]['valid'],axis=1)&np.isfinite(pred[method]['raw']).all(1) for method in METHODS}
            timings.append({'id':name,'postSealExactAndSignedDiagnosticsSeconds':diagnostic_s,'fourOriginalConsumerArmsSeconds':time.perf_counter()-tick})
            path=work/(name+'-consumers.npz');np.savez(path,**{method+'__'+stage:a for method,columns in results.items() for stage,a in columns.items()});refrows.append({'id':name,'kind':'consumers','path':str(path),'sha256':sha(path)})
            target={'raw':exact_raw,'rounded':floats(parent['exactBank']).astype(float)};all_results[name]=results;all_banks[name]=banks;all_targets[name]=target;validity[name]=valid
            label={'id':name,'kind':'endpoint'};rows,count=case_scores(label,results,valid,strata(mask),METHODS,f);scores+=rows;observations+=count
            head_rows+=head_scores(label,banks,target,strata(mask),valid,d,heads);print('SELECTOR_CONSUMERS_COMPLETE',name,flush=True)
        a,b=source['pairs'][0];pair_regions=strata(masks[a],masks[b]);valid={method:validity[a][method]&validity[b][method] for method in METHODS}
        delta={method:{precision:all_banks[b][method][precision]-all_banks[a][method][precision] for precision in ['raw','rounded']} for method in METHODS}
        target={precision:all_targets[b][precision]-all_targets[a][precision] for precision in ['raw','rounded']}
        head_rows+=head_scores({'a':a,'b':b,'kind':'paired'},delta,target,pair_regions,valid,d,heads)
        for method in METHODS:
            for stage in STAGES:
                pred,exact,zero=[floats(all_results[b][arm][stage]).astype(float)-floats(all_results[a][arm][stage]).astype(float) for arm in [method,'exact','zero']]
                for region,selected in pair_regions:
                    if selected.any():paired.append({'a':a,'b':b,'method':method,'stage':stage,'region':region,**effect_scores(pred,exact,zero,selected,valid[method])})
                er,tr=[rowrms(z) for z in [pred-exact,exact-zero]]
                for i in range(len(pred)):
                    f.write(json.dumps({'a':a,'b':b,'kind':'paired','method':method,'stage':stage,'position':i,'valid':bool(valid[method][i]),'consumerErrorRms':float(er[i]),'attentionEffectRms':float(tr[i])},allow_nan=False)+'\n');observations+=1
    if sha(seal_path)!=seal_sha:raise ValueError('Prediction seal changed')
    for name,path in [('request.json',root/'request.json'),('source.json',root/'source.json'),('serve-source.json',root/'serve-source.json'),('prediction-seal.json',seal_path)]:shutil.copyfile(path,out/name)
    save(out/'reference-receipt.json',{'rows':refrows,'checks':checks,'parameters':param_receipt,'scope':'Post-seal exact/signed fullprefix diagnostic and original learned consumer. Historical anchor/witness/oracle arrays/supports/consumers reused, no oracle rerank.'})
    save(out/'signed-diagnostics.json',{'rows':diagnostic_rows,'scope':'Correction w=p_anchor expm1(centered tilt), coefficients w/D. Captured fractions use log masses; undefined zero-sign ratios explicit. Raw log masses/signed numerator vectors and partition scalars retained in authenticated NPZ. RMS of signed vectors is a magnitude diagnostic, not fidelity certification.'})
    save(out/'report.json',{'schema':'tce-selector-objective-result-v1','requestSHA256':r['requestSHA256'],'predictionSealSHA256':seal_sha,'scope':r['scope'],
        'serving':seal,'scores':scores,'pairedScores':paired,'headScores':head_rows,'checks':checks,'timings':timings,'servingChildSeconds':child_s,
        'originalParameterReadSeconds':parameter_s,'workerSeconds':time.perf_counter()-start,'observations':observations,
        'originalEquationWork':{'fullOutputProjections':8,'mlpDenseProjections':24,'rowsPerArm':512,'historicalConsumers':'anchor/witness/oracle retained parent results, not recomputed'},
        'frameworkCalls':0,'originalLearnedConsumerEquationsEvaluated':True,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'goalStatus':'Active. Fullprefix diagnostic selectors and original learned receiving consumer, not fullencoder or arbitrary-prompt zero-neural images.'})
    print('SELECTOR_OBJECTIVE_STUDY_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
