"""Retrospective sparse joint-tilt test, with a distinct full-query-score oracle."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from encoder_atoms import np,read,save,sha,identity,floats,words,NoNeural
from attention_bank import packet,geometry,head_asset
from attention_bank_study import loaded,verify_request,regions,effect_scores,STAGES
from context_statistics_study import summary,rowrms
from transport_consumer import parameters,consume
from sparse_tilt import BUDGETS

HERE=Path(__file__).resolve().parent


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,j,r=checked_collection(parent);live=Api().json('GET','/api/jobs/'+c['jobID'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody'] or live['config']['requestSHA256']!=r['requestSHA256']:raise ValueError('Complete-bank parent changed')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir();seal=read(parent/'result/prediction-seal.json');ref=read(parent/'result/reference-receipt.json')
    source={'predictionSeal':{'path':j['result']['diagnostics']['cwd']+'/predictions/prediction-seal.json','sha256':sha(parent/'result/prediction-seal.json')},
        'asset':seal['asset'],'oldPredictions':seal['rows'],'references':ref['rows'],'pairs':r['pairs'],'geometry':{'headDim':128,'qWidth':4096,'kvWidth':1024},
        'parentConsumerParameterSource':ref['parameters']['source']}
    save(p/'source.json',source)
    save(p/'serve-source.json',{'asset':source['asset'],'ports':[{'id':z['id'],**z['port']} for z in seal['rows']],'geometry':source['geometry'],'budgets':BUDGETS})
    for n in ['sparse_tilt.py','sparse_tilt_study.py','attention_bank.py','attention_bank_study.py','encoder_atoms.py','encoder-atoms-source.json',
              'context_source_geometry.py','context_landmarks.py','context_transport.py','context_statistics.py','context_statistics_study.py','context_transport_study.py','transport_consumer.py']:shutil.copyfile(HERE/n,p/n)
    request={'schema':'tce-sparse-joint-tilt-request-v1','parentJobID':c['jobID'],'parentRequestSHA256':r['requestSHA256'],
        'scope':'Already-scored saved pair, all32heads/all512rows. Serving-eligible source-score topm and append-only positive tail, noCvy. Full-query-score oracle after primary seal, original learned consumer offline. No fresh generalization or zero-neural image claim.',
        'budgets':BUDGETS,'pairs':r['pairs'],'gauge':'t=(x-l) dot(y-mu_l), full live anchor keymean reconstructed from retained entries and positive tail.',
        'supportPolicy':'Fixed anchor scores l dot(y-sourcecenter); retain topm occurrences, earlier position wins ties. Evictions/rejections append to tail with original scores; no subtraction. Separate states per budget; no KVpayload dedup saving claimed.',
        'oraclePolicy':'Only after complete primary seal: rank abs(p_l expm1(t)) over every allowed causal key. No future keys or output-error fit. Same selected set updates numerator and partition; mass-ranked support is not output-optimal sparse capacity.',
        'consumerPolicy':'Recompute original fullblock on all512rows for all six sparse/oracle arms plus exact/zero consistency controls. Parent nearest/transport arrays retained as supplied references, not refreshed candidates. Shared FP64/BF16 equations, nativebackend parity unmeasured.',
        'primaryObservables':'Paired state attention-deletion effect, head response/cosines, active/padding/before8/at8/after8. Perhead diagnostics include support overlap/churn/address8retention/omitted absolute tilt mass/exponential remainder mass and direct restricted-law closure. Raw arrays retained nearcompute.',
        'sourceHashes':{f.name:sha(f) for f in p.iterdir()},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':600}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    verify_request(root/'payload',r);cmd=command()[:-5]+['python','-u','sparse_tilt_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-sparse-joint-tilt-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512,'heads':32,'budgets':BUDGETS},
        timeout_s=r['timeoutSeconds'],detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 retrospective2savedportsx32headsx512queries; m1/2/4 separate positive heap/tail states, seal primary banks, then full-query-score oracle/direct support diagnostics and8original fullblock consumer arms. NoCvy or freshprompt claim. Original5GBshard only after seal; defaultadmission, expected under3min,finite600s,no retry.')
    result={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',result);return result


def logsum(logs):
    if not len(logs):return -np.inf
    top=logs.max()
    return float(top+np.log(np.exp(logs-top).sum())) if np.isfinite(top) else -np.inf


def log_abs_expm1(t):
    result=np.full(t.shape,-np.inf);large=t>50.;other=(t!=0)&~large
    result[large]=t[large]+np.log1p(-np.exp(-t[large]));result[other]=np.log(np.abs(np.expm1(t[other])))
    return result


def restricted_read(score,t,values,selected):
    adjusted=score.copy();adjusted[selected]+=t[selected];weight=np.exp(adjusted-logsum(adjusted))
    return weight@values


def diagnose(p,asset,primary,budget):
    q,k,v=[floats(p[key]).astype(float) for key in ['q','k','v']];n,d=q.shape;u=k/d**.25-asset['keyCenter'];out=np.zeros((n,d));support=np.full((n,budget),-1,dtype=np.int64)
    fields={key:np.zeros(n) for key in ['restrictedLawMaxAbsClosure','supportOverlapFraction','eligibleAddress8','oracleAddress8',
        'eligibleOmittedAbsoluteTiltMass','oracleOmittedAbsoluteTiltMass','eligibleRemainderMassCaptured','oracleRemainderMassCaptured',
        'normalizedRemainderMass','eligibleCorrectionRms','oracleCorrectionRms']}
    for i in range(n):
        allowed=np.flatnonzero(p['mask'][:i+1])
        if not len(allowed):continue
        ell=asset['landmarks'][int(primary['nearestIndex'][i])];U=u[allowed];V=v[allowed];score=U@ell;logp=score-logsum(score);weight=np.exp(logp);mu=weight@U;H=weight@V
        t=(U-mu)@(q[i]/d**.25-ell);logmass=logp+log_abs_expm1(t);order=np.lexsort((allowed,-logmass));chosen=order[:budget]
        positions=primary['supportPositions'][i];kept=np.flatnonzero(np.isin(allowed,positions[positions>=0]));direct=restricted_read(score,t,V,kept);oracle=restricted_read(score,t,V,chosen)
        out[i]=oracle;support[i,:len(chosen)]=allowed[chosen];fields['restrictedLawMaxAbsClosure'][i]=float(np.max(np.abs(direct-primary['raw'][i])))
        fields['supportOverlapFraction'][i]=len(set(allowed[kept])&set(allowed[chosen]))/len(chosen) if len(chosen) else 0.
        fields['eligibleAddress8'][i]=float(8 in allowed[kept]);fields['oracleAddress8'][i]=float(8 in allowed[chosen])
        logtotal=logsum(logmass);absolute=np.exp(logmass-logtotal) if np.isfinite(logtotal) else np.zeros(len(logmass))
        tilt=logp+t;logZ=logsum(tilt);beta=np.exp(-logZ);remainder=np.exp(tilt-logZ)-beta*weight*(1+t);small=np.abs(t)<=1.
        remainder[small]=beta*weight[small]*(np.expm1(t[small])-t[small]);total=float(remainder.sum());fields['normalizedRemainderMass'][i]=total
        for label,selected,readout in [('eligible',kept,direct),('oracle',chosen,oracle)]:
            omitted=np.ones(len(allowed),dtype=bool);omitted[selected]=False;fields[label+'OmittedAbsoluteTiltMass'][i]=float(absolute[omitted].sum())
            fields[label+'RemainderMassCaptured'][i]=float(remainder[selected].sum()/total) if total else 0.
            fields[label+'CorrectionRms'][i]=float(np.sqrt(np.mean((readout-H)**2)))
    return {'raw':out,'rounded':words(out),'supportPositions':support,**fields}


def case_scores(label,results,valid,masks,methods,stream):
    scores=[];count=0
    for method in methods:
        for stage in STAGES:
            pred,exact,zero=[floats(results[a][stage]).astype(float) for a in [method,'exact','zero']]
            for region,selected in masks:
                if selected.any():scores.append({**label,'method':method,'stage':stage,'region':region,**effect_scores(pred,exact,zero,selected,valid[method])})
            er,tr=[rowrms(z) for z in [pred-exact,exact-zero]]
            for i in range(len(pred)):
                stream.write(json.dumps({**label,'method':method,'stage':stage,'position':i,'valid':bool(valid[method][i]),'consumerErrorRms':float(er[i]),'attentionEffectRms':float(tr[i])},allow_nan=False)+'\n');count+=1
    return scores,count


def execute(root):
    if NoNeural.roots.intersection(sys.modules):raise RuntimeError('Neural framework already imported')
    sys.meta_path.insert(0,NoNeural());root=Path(root).resolve();r=read(root/'request.json');verify_request(root,r);start=time.perf_counter()
    source=read(root/'source.json');out=root/'response-result';out.mkdir(exist_ok=False)
    if sha(source['predictionSeal']['path'])!=source['predictionSeal']['sha256']:raise ValueError('Parent seal differs')
    tick=time.perf_counter();subprocess.run([sys.executable,str(root/'sparse_tilt.py'),str(root/'serve-source.json'),str(root/'predictions')],check=True)
    child_s=time.perf_counter()-tick;seal_path=root/'predictions/prediction-seal.json';seal_sha=sha(seal_path);seal=read(seal_path)
    print('ALL_SPARSE_PRIMARY_BANKS_SEALED',seal_sha,flush=True)
    # Full-query-score support and original learned equations are reference-only.
    asset=loaded(source['asset']);m=source['geometry'];d,heads,groups=geometry(m);methods=['m'+str(x) for x in BUDGETS]+['oracle'+str(x) for x in BUDGETS]
    tick=time.perf_counter();params,config,param_receipt=parameters(read(root/'encoder-atoms-source.json'));parameter_s=time.perf_counter()-tick
    if param_receipt['source']!=source['parentConsumerParameterSource']:raise ValueError('Original consumer source differs')
    work=root/'reference-work';work.mkdir();all_results={};all_banks={};validity={};masks={};refrows=[];diagnostic_rows=[];head_scores=[];scores=[];paired=[];checks=[];observations=0;timings=[]
    with (out/'observations.jsonl').open('x') as f:
        for port_entry in read(root/'serve-source.json')['ports']:
            name=port_entry['id'];port=loaded(port_entry);mask=port['mask'].astype(bool);masks[name]=mask;n=len(mask)
            parent=loaded(next(x for x in source['references'] if x['id']==name));old=loaded(next(x for x in source['oldPredictions'] if x['id']==name));bank_arms={};diagnostics=[]
            valid={};tick=time.perf_counter()
            for budget in BUDGETS:
                pred=loaded(next(x for x in seal['rows'] if x['id']==name and x['budget']==budget));oracle=np.zeros((n,heads*d),dtype=np.uint16);raw_oracle=np.zeros((n,heads*d));fields={};oracle_support=np.zeros((n,heads,budget),dtype=np.int64)
                for head in range(heads):
                    span=slice(head*d,(head+1)*d);p=packet(port,m,head);primary={'nearestIndex':pred['nearestIndex'][:,head],'supportPositions':pred['supportPositions'][:,head],'raw':pred['raw'][:,span]}
                    diagnostic=diagnose(p,head_asset(asset,m,head),primary,budget);oracle[:,span]=diagnostic['rounded'];raw_oracle[:,span]=diagnostic['raw'];oracle_support[:,head]=diagnostic['supportPositions']
                    for key,value in diagnostic.items():
                        if value.ndim==1:
                            if key not in fields:fields[key]=np.empty((n,heads))
                            fields[key][:,head]=value
                    for region,selected in regions(mask):
                        if selected.any():diagnostic_rows.append({'id':name,'budget':budget,'queryHead':head,'region':region,
                            **{key:{'mean':float(np.mean(value[selected,head])),'maximum':float(np.max(value[selected,head]))} for key,value in fields.items()}})
                bank_arms['m'+str(budget)]=pred['rounded'];bank_arms['oracle'+str(budget)]=oracle;valid['m'+str(budget)]=np.all(pred['valid'],axis=1);valid['oracle'+str(budget)]=np.ones(n,dtype=bool)
                path=work/(name+'-m'+str(budget)+'-oracle.npz');np.savez(path,raw=raw_oracle,rounded=oracle,supportPositions=oracle_support,**fields)
                refrows.append({'id':name,'budget':budget,'kind':'oracle-diagnostics','path':str(path),'sha256':sha(path)})
                checks.append({'id':name,'budget':budget,'directRestrictedMaxAbsClosure':float(fields['restrictedLawMaxAbsClosure'].max()),
                    'routeMismatchesFromParent':int(np.count_nonzero(pred['nearestIndex']!=old['nearestIndex']))})
            oracle_s=time.perf_counter()-tick;tick=time.perf_counter();results={}
            bank_arms['exact']=parent['exactBank'];bank_arms['zero']=np.zeros_like(parent['exactBank'])
            for method,b in bank_arms.items():results[method]=consume(port['e'],np.zeros((n,config['hidden_size'])),floats(b).astype(float),params['output'],params['gamma'],config['rms_norm_eps'],np.arange(n),[params[k] for k in ['gate','up','down']])
            mismatch={arm:{stage:int(np.count_nonzero(results[arm][stage]!=parent[arm+'__'+stage])) for stage in STAGES} for arm in ['zero','exact']}
            checks.append({'id':name,'originalConsumerReferenceWordMismatches':mismatch})
            for method in ['nearest64','transport64']:results[method]={stage:parent[method+'__'+stage] for stage in STAGES};valid[method]=np.ones(n,dtype=bool)
            consumer_s=time.perf_counter()-tick;timings.append({'id':name,'oracleAndDirectDiagnosticsSeconds':oracle_s,'eightOriginalConsumerArmsSeconds':consumer_s})
            all_results[name]=results;all_banks[name]=bank_arms;validity[name]=valid
            path=work/(name+'-consumers.npz');np.savez(path,**{method+'__'+stage:value for method,fields in results.items() for stage,value in fields.items()},**{method+'__valid':v for method,v in valid.items()})
            refrows.append({'id':name,'kind':'consumers','path':str(path),'sha256':sha(path)})
            rows,count=case_scores({'id':name,'kind':'endpoint'},results,valid,regions(mask),methods,f);scores+=rows;observations+=count
            print('SPARSE_CONSUMERS_COMPLETE',name,flush=True)
        a,b=source['pairs'][0];clock=np.arange(len(masks[a]));pair_regions=regions(masks[a],masks[b])+[('before8',clock<8),('at8',clock==8),('after8',clock>8)]
        for method in methods+['nearest64','transport64']:
            use=validity[a][method]&validity[b][method]
            if method in methods:
                predicted=floats(all_banks[b][method]).astype(float)-floats(all_banks[a][method]).astype(float);target=floats(all_banks[b]['exact']).astype(float)-floats(all_banks[a]['exact']).astype(float)
                for region,selected in pair_regions:
                    if not selected.any():continue
                    selected=selected&use
                    for head in [None,*range(heads)]:
                        span=slice(None) if head is None else slice(head*d,(head+1)*d)
                        head_scores.append({'method':method,'queryHead':head,'region':region,'validQueries':int(selected.sum()),
                            'score':summary(predicted[selected,span],target[selected,span]) if selected.any() else None})
            for stage in STAGES:
                pred,exact,zero=[floats(all_results[b][arm][stage]).astype(float)-floats(all_results[a][arm][stage]).astype(float) for arm in [method,'exact','zero']]
                for region,selected in pair_regions:
                    if selected.any():paired.append({'a':a,'b':b,'method':method,'stage':stage,'region':region,**effect_scores(pred,exact,zero,selected,use)})
                if method in methods:
                    er,tr=[rowrms(z) for z in [pred-exact,exact-zero]]
                    for i in range(len(pred)):
                        f.write(json.dumps({'a':a,'b':b,'kind':'paired','method':method,'stage':stage,'position':i,'valid':bool(use[i]),'consumerErrorRms':float(er[i]),'attentionEffectRms':float(tr[i])},allow_nan=False)+'\n');observations+=1
    if sha(seal_path)!=seal_sha:raise ValueError('Primary seal changed')
    for name,path in [('request.json',root/'request.json'),('source.json',root/'source.json'),('serve-source.json',root/'serve-source.json'),('prediction-seal.json',seal_path)]:shutil.copyfile(path,out/name)
    save(out/'reference-receipt.json',{'rows':refrows,'checks':checks,'parameters':param_receipt,
        'scope':'Full-query-score oracle/direct restricted-law diagnostics after primaryseal. Original shared-math learned consumers all512rows; not nativebackend parity or serving.'})
    save(out/'support-diagnostics.json',{'rows':diagnostic_rows,'scope':'Perhead/region summaries; complete perquery support/correction arrays retained nearcompute. Oracle is current fullquery score access, not future keys or outputoptimal sparse support.'})
    save(out/'report.json',{'schema':'tce-sparse-joint-tilt-result-v1','requestSHA256':r['requestSHA256'],'predictionSealSHA256':seal_sha,'scope':r['scope'],
        'serving':seal,'scores':scores,'pairedScores':paired,'pairedHeadScores':head_scores,'checks':checks,'timings':timings,'servingChildSeconds':child_s,'originalParameterReadSeconds':parameter_s,'workerSeconds':time.perf_counter()-start,
        'observations':observations,'originalEquationWork':{'fullOutputProjections':16,'mlpDenseProjections':48,'rowsPerArm':512},'originalLearnedConsumerEquationsEvaluated':True,
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed','frameworkCalls':0,
        'goalStatus':'Active. Retrospective reduced attention law and original learned consumer; no freshrawtext validation, completeencoder or strictzero-neural images.'})
    print('SPARSE_JOINT_TILT_STUDY_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
