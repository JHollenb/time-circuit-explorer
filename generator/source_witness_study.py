"""Source-only witness compilation, sealed selector contrast, original consumer."""
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
from sparse_tilt_study import logsum,log_abs_expm1,case_scores
from centroid_tilt_study import direct_centroid
from centroid_tilt import same_support
from source_witness import POLICY
from transport_consumer import parameters,consume

HERE=Path(__file__).resolve().parent
METHODS=['anchor','witness','oracle']


def stage(root,parent,complete_bank):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();complete_bank=Path(complete_bank).resolve()
    c,j,r=checked_collection(parent,current_workers=False);gc,gj,gr=checked_collection(complete_bank,current_workers=False);api=Api()
    for collection,request in [(c,r),(gc,gr)]:
        live=api.json('GET','/api/jobs/'+collection['jobID'])
        if live['state']!='succeeded' or live['payload_custody']!=collection['custody'] or live['config']['requestSHA256']!=request['requestSHA256']:raise ValueError('Source parent differs')
    if gc['jobID']!=r['completeBankJobID'] or gr['requestSHA256']!=r['completeBankRequestSHA256']:raise ValueError('Ancestor differs')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir();old=read(parent/'result/source.json');seal=read(parent/'result/prediction-seal.json');ref=read(parent/'result/reference-receipt.json');serve=read(complete_bank/'result/serve.json')
    source={k:old[k] for k in ['asset','ports','geometry','pairs','references','parentConsumerParameterSource','completeBankRoot']}
    source.update(parentSeal={'path':j['result']['diagnostics']['cwd']+'/predictions/prediction-seal.json','sha256':sha(parent/'result/prediction-seal.json')},
        parentBanks=[x for x in seal['rows'] if x['method']=='eligible-centroid'],parentStates=[x for x in seal['states'] if x['law']=='centroid'],
        parentConsumers=[x for x in ref['rows'] if x['kind']=='consumers'],parentOracle=[x for x in ref['rows'] if x['kind']=='frozen-oracle'])
    save(p/'source.json',source);save(p/'compile-source.json',{'asset':source['asset'],**{k:serve[k] for k in ['store','manifestSHA256']}})
    files=['source_witness.py','source_witness_study.py','centroid_tilt.py','centroid_tilt_study.py','sparse_tilt.py','sparse_tilt_study.py',
        'attention_bank.py','attention_bank_study.py','encoder_atoms.py','encoder-atoms-source.json','context_source_geometry.py','context_landmarks.py',
        'context_transport.py','context_statistics.py','context_statistics_study.py','context_transport_study.py','transport_consumer.py']
    for n in files:shutil.copyfile(HERE/n,p/n)
    request={'schema':'tce-source-witness-request-v1','parentJobID':c['jobID'],'parentRequestSHA256':r['requestSHA256'],
        'completeBankJobID':gc['jobID'],'completeBankRequestSHA256':gr['requestSHA256'],'pairs':source['pairs'],'budget':4,'policy':POLICY,
        'scope':'Retrospective saved pair/all32heads/512rows. Frozen source-cell witnesses change support priority; original anchor measure/centroid read/routing fixed. No freshrawtext/fullencoder/zero-neural image claim.',
        'compiler':'Original source candidates/anchors/keycov metric, canonical sourcecells, anchor plus3farthest sourcecell witnesses, repeatanchor degeneracy. Fixed sourcekeymean gauge and zerooffsets. No currentprompt queries, normnormalization or targetfit.',
        'stateLaw':'Separate fixed maxwitness priority rho from anchor score s; top4 by rho, earlieroccurrence wins ties. Every eviction/tail Z/N/Y/read uses s. Ownstate append-only tail; noCvy. Witness policy/source/clock bound on resume.',
        'sealPolicy':'Compiler then newchild anchor/witness banks, seal both complete predictions before exact references/oracle arrays/original parameter read. Retain old oracle-centroid result, no rerank or new oracle consumer.',
        'observables':'Paired head and original fullblock state,active/padding/at8/after8, supportoverlap/omitted normalizedabsolute tiltmass,address8,membership and sourcewitness counts/norms. Preserve zero/smalltargets/countertrends. Costs separate.',
        'sourceHashes':{f.name:sha(f) for f in p.iterdir()},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':600}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    verify_request(root/'payload',r);cmd=command()[:-5]+['python','-u','source_witness_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-source-witness-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512,'heads':32,'budget':4},
        timeout_s=r['timeoutSeconds'],detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 sourcecell4witness compiler from retained lexical atoms, fixed64anchors/head;2savedportsx32headsx512queries. Source-only priority/anchorweight separation, seal2ownstate candidatebanks before currentprefix diagnostics and4original fullblock consumers/prompt; oldoracle retained. Host-local immutable store/parents/5GBshard; default admission,expected under4min,finite600s,no retry.')
    result={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',result);return result


def diagnose(p,asset,candidates,oracle):
    q,k,v=[floats(p[key]).astype(float) for key in ['q','k','v']];n,d=q.shape;u=k/d**.25-asset['keyCenter']
    fields={method:{key:np.zeros(n) for key in ['directCentroidMaxAbsClosure','oracleSupportOverlap','omittedAbsoluteTiltMassFraction','address8Retained']} for method in candidates}
    for i in range(n):
        allowed=np.flatnonzero(p['mask'][:i+1])
        if not len(allowed):continue
        U=u[allowed];V=v[allowed];route=int(candidates['anchor']['nearestIndex'][i]);ell=asset['landmarks'][route];score=U@ell
        weight=np.exp(score-logsum(score));mu=weight@U;delta=q[i]/d**.25-ell;t=(U-mu)@delta
        logmass=score-logsum(score)+log_abs_expm1(t);total=logsum(logmass);mass=np.exp(logmass-total) if np.isfinite(total) else np.zeros(len(allowed))
        reference=set(oracle[i][oracle[i]>=0])
        for method,candidate in candidates.items():
            positions=candidate['supportPositions'][i];positions=positions[positions>=0]
            if len(set(positions))!=len(positions) or not set(positions).issubset(set(allowed)) or len(positions)!=min(4,len(allowed)):raise ValueError('Candidate support is not a causal occurrence set')
            selected=np.flatnonzero(np.isin(allowed,positions));direct=direct_centroid(score,U,V,selected,delta);omitted=np.ones(len(allowed),dtype=bool);omitted[selected]=False
            fields[method]['directCentroidMaxAbsClosure'][i]=float(np.max(np.abs(direct-candidate['raw'][i])))
            fields[method]['oracleSupportOverlap'][i]=len(set(positions)&reference)/len(reference)
            fields[method]['omittedAbsoluteTiltMassFraction'][i]=float(mass[omitted].sum());fields[method]['address8Retained'][i]=float(8 in positions)
    return fields


def execute(root):
    if NoNeural.roots.intersection(sys.modules):raise RuntimeError('Neural framework imported')
    sys.meta_path.insert(0,NoNeural());root=Path(root).resolve();r=read(root/'request.json');verify_request(root,r);start=time.perf_counter()
    source=read(root/'source.json');out=root/'response-result';out.mkdir(exist_ok=False)
    if sha(source['parentSeal']['path'])!=source['parentSeal']['sha256']:raise ValueError('Parent seal differs')
    tick=time.perf_counter();subprocess.run([sys.executable,str(root/'source_witness.py'),'compile',str(root/'compile-source.json'),str(root/'compiled')],check=True);compile_s=time.perf_counter()-tick
    spec={k:source[k] for k in ['ports','geometry']};spec['asset']={'path':str(root/'compiled/witness-assets.npz'),'sha256':sha(root/'compiled/witness-assets.npz')};save(root/'serve.json',spec)
    tick=time.perf_counter();subprocess.run([sys.executable,str(root/'source_witness.py'),'serve',str(root/'serve.json'),str(root/'predictions')],check=True);child_s=time.perf_counter()-tick
    seal_path=root/'predictions/prediction-seal.json';seal_sha=sha(seal_path);seal=read(seal_path);print('ALL_SOURCE_WITNESS_BANKS_SEALED',seal_sha,flush=True)
    asset=loaded(spec['asset']);m=source['geometry'];d,heads,_=geometry(m);work=root/'reference-work';work.mkdir()
    tick=time.perf_counter();params,config,param_receipt=parameters(read(root/'encoder-atoms-source.json'));parameter_s=time.perf_counter()-tick
    if param_receipt['source']!=source['parentConsumerParameterSource']:raise ValueError('Original consumer source differs')
    all_results={};all_banks={};validity={};supports={};masks={};refrows=[];checks=[];scores=[];paired=[];head_scores=[];membership=[];diagnostic_rows=[];timings=[];observations=0
    with (out/'observations.jsonl').open('x') as f:
        for item in source['ports']:
            name=item['id'];port=loaded(item);n=len(port['mask']);mask=port['mask'].astype(bool);masks[name]=mask;tick=time.perf_counter()
            pred={method:loaded(next(x for x in seal['rows'] if x['id']==name and x['method']==method)) for method in ['anchor','witness']}
            old=loaded(next(x for x in source['parentBanks'] if x['id']==name));old_oracle=loaded(next(x for x in source['parentOracle'] if x['id']==name));old_consumer=loaded(next(x for x in source['parentConsumers'] if x['id']==name));parent=loaded(next(x for x in source['references'] if x['id']==name))
            fields={};frozen={method:pred[method]['supportPositions'] for method in pred};frozen['oracle']=old_oracle['supportPositions']
            for head in range(heads):
                span=slice(head*d,(head+1)*d);candidate={method:{'nearestIndex':p['nearestIndex'][:,head],'supportPositions':p['supportPositions'][:,head],'raw':p['raw'][:,span]} for method,p in pred.items()}
                diagnostic=diagnose(packet(port,m,head),head_asset(asset,m,head),candidate,frozen['oracle'][:,head])
                for method,columns in diagnostic.items():
                    for key,values in columns.items():
                        namekey=method+'__'+key
                        if namekey not in fields:fields[namekey]=np.empty((n,heads))
                        fields[namekey][:,head]=values
                        for region,selected in regions(mask):
                            if selected.any():diagnostic_rows.append({'id':name,'queryHead':head,'method':method,'metric':key,'region':region,'mean':float(np.mean(values[selected])),'maximum':float(np.max(values[selected]))})
            state_mismatch=[]
            for head in range(heads):
                before=loaded(next(x for x in source['parentStates'] if x['id']==name and x['queryHead']==head));after=loaded(next(x for x in seal['states'] if x['id']==name and x['queryHead']==head and x['method']=='anchor'))
                state_mismatch.append(sum(not np.array_equal(before[k],after[k]) for k in before))
            checks.append({'id':name,'anchorParentWordMismatches':int(np.count_nonzero(pred['anchor']['rounded']!=old['rounded'])),
                'anchorParentSupportMismatches':int(np.count_nonzero(~same_support(pred['anchor']['supportPositions'],old['supportPositions']))),
                'routeMismatches':int(np.count_nonzero(pred['anchor']['nearestIndex']!=pred['witness']['nearestIndex'])),
                'parentRouteMismatches':int(np.count_nonzero(pred['anchor']['nearestIndex']!=old['nearestIndex'])),'anchorParentStateFieldMismatchesByHead':state_mismatch,
                'directCentroidMaxAbsClosure':{method:float(fields[method+'__directCentroidMaxAbsClosure'].max()) for method in pred}})
            path=work/(name+'-support-diagnostics.npz');np.savez(path,**fields);refrows.append({'id':name,'kind':'support-diagnostics','path':str(path),'sha256':sha(path)})
            diagnostic_s=time.perf_counter()-tick;tick=time.perf_counter();banks={method:p['rounded'] for method,p in pred.items()};banks.update(exact=parent['exactBank'],zero=np.zeros_like(parent['exactBank']));results={}
            for method,bank in banks.items():results[method]=consume(port['e'],np.zeros((n,config['hidden_size'])),floats(bank).astype(float),params['output'],params['gamma'],config['rms_norm_eps'],np.arange(n),[params[k] for k in ['gate','up','down']])
            checks.append({'id':name,'consumerParentWordMismatches':{method:{stage:int(np.count_nonzero(results[method][stage]!=(parent[method+'__'+stage] if method in ['exact','zero'] else old_consumer['eligible-centroid__'+stage]))) for stage in STAGES} for method in ['exact','zero','anchor']}})
            results['oracle']={stage:old_consumer['oracle-centroid__'+stage] for stage in STAGES};banks['oracle']=words(old_oracle['centroid_raw'])
            valid={method:np.all(pred[method]['valid'],axis=1) if method in pred else np.ones(n,dtype=bool) for method in METHODS};timings.append({'id':name,'supportDiagnosticsSeconds':diagnostic_s,'fourOriginalConsumerArmsSeconds':time.perf_counter()-tick})
            path=work/(name+'-consumers.npz');np.savez(path,**{method+'__'+stage:a for method,columns in results.items() for stage,a in columns.items()});refrows.append({'id':name,'kind':'consumers','path':str(path),'sha256':sha(path)})
            all_results[name]=results;all_banks[name]=banks;validity[name]=valid;supports[name]=frozen
            rows,count=case_scores({'id':name,'kind':'endpoint'},results,valid,regions(mask),METHODS,f);scores+=rows;observations+=count
            print('SOURCE_WITNESS_CONSUMERS_COMPLETE',name,flush=True)
        a,b=source['pairs'][0];clock=np.arange(len(masks[a]));pair_regions=regions(masks[a],masks[b])+[('before8',clock<8),('at8',clock==8),('after8',clock>8)]
        target=(floats(all_banks[b]['exact']).astype(float)-floats(all_banks[a]['exact']).astype(float)).reshape(-1,heads,d)
        for method in METHODS:
            same=same_support(supports[a][method],supports[b][method]);valid=validity[a][method]&validity[b][method]
            delta=(floats(all_banks[b][method]).astype(float)-floats(all_banks[a][method]).astype(float)).reshape(-1,heads,d)
            for region,selected in pair_regions:
                if not selected.any():continue
                use=selected&valid
                for head in [None,*range(heads)]:
                    pred,exact=(delta[use],target[use]) if head is None else (delta[use,head],target[use,head])
                    head_scores.append({'method':method,'queryHead':head,'region':region,'validQueries':int(use.sum()),'score':summary(pred,exact) if use.any() else None})
                for label,chosen in [('unchanged',same),('changed',~same)]:
                    rows=use[:,None]&chosen;membership.append({'method':method,'region':region,'membership':label,'headQueryRows':int(rows.sum()),'score':summary(delta[rows],target[rows]) if rows.any() else None})
            for stage in STAGES:
                pred,exact,zero=[floats(all_results[b][arm][stage]).astype(float)-floats(all_results[a][arm][stage]).astype(float) for arm in [method,'exact','zero']]
                for region,selected in pair_regions:
                    if selected.any():paired.append({'a':a,'b':b,'method':method,'stage':stage,'region':region,**effect_scores(pred,exact,zero,selected,valid)})
                er,tr=[rowrms(z) for z in [pred-exact,exact-zero]]
                for i in range(len(pred)):
                    f.write(json.dumps({'a':a,'b':b,'kind':'paired','method':method,'stage':stage,'position':i,'valid':bool(valid[i]),'consumerErrorRms':float(er[i]),'attentionEffectRms':float(tr[i])},allow_nan=False)+'\n');observations+=1
    if sha(seal_path)!=seal_sha:raise ValueError('Prediction seal changed')
    for name,path in [('request.json',root/'request.json'),('source.json',root/'source.json'),('compile-source.json',root/'compile-source.json'),('serve.json',root/'serve.json'),('prediction-seal.json',seal_path),('compile-receipt.json',root/'compiled/compile-receipt.json')]:shutil.copyfile(path,out/name)
    save(out/'reference-receipt.json',{'rows':refrows,'checks':checks,'parameters':param_receipt,'scope':'Post-seal fullprefix support diagnostics/original learned fullblock. Old oracle-centroid banks/consumers retained, no rerank or new oracle consumer.'})
    save(out/'support-diagnostics.json',{'rows':diagnostic_rows,'scope':'Perhead/region currentprefix diagnostic after seal. Omittedmass is fraction of absolute tilt mass, not probabilityerror bound; encoded0when totalmass0.'})
    save(out/'membership-diagnostics.json',{'rows':membership,'scope':'Valid-position-set identity across prompts. Diagnostic association, not causal membership intervention.'})
    save(out/'report.json',{'schema':'tce-source-witness-result-v1','requestSHA256':r['requestSHA256'],'predictionSealSHA256':seal_sha,'scope':r['scope'],
        'serving':seal,'scores':scores,'pairedScores':paired,'pairedHeadScores':head_scores,'checks':checks,'timings':timings,'compileChildSeconds':compile_s,
        'servingChildSeconds':child_s,'originalParameterReadSeconds':parameter_s,'workerSeconds':time.perf_counter()-start,'observations':observations,
        'originalEquationWork':{'fullOutputProjections':8,'mlpDenseProjections':24,'rowsPerArm':512,'oracleConsumer':'retained parent result, not recomputed'},
        'frameworkCalls':0,'originalLearnedConsumerEquationsEvaluated':True,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'goalStatus':'Active. Retrospective source-witness selector and original learned receiving consumer, not fullencoder or arbitrary-prompt zero-neural images.'})
    print('SOURCE_WITNESS_STUDY_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');p.add_argument('--complete-bank');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent,a.complete_bank)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
