"""Compile source coordinates, seal a streaming bank, then measure its consumers."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from encoder_atoms import np,read,save,sha,identity,floats,words,NoNeural
from attention_bank import geometry,packet,head_asset
from attention_bank_study import loaded,verify_request,effect_scores,STAGES
from context_statistics_study import summary,rowrms
from selector_objective_study import strata,describe
from selector_objective import select
from sparse_tilt_study import case_scores
from tail_factorial import ordered_support
from tail_factorial_study import energy
from rank_one_tail import direct
from transport_consumer import parameters,consume

HERE=Path(__file__).resolve().parent
METHODS=['stream','direct','quadratic','00','11']


def stage(root,parent,complete):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();complete=Path(complete).resolve();c,j,r=checked_collection(parent,current_workers=False);cc,cj,cr=checked_collection(complete,current_workers=False);api=Api()
    for collection,request in [(c,r),(cc,cr)]:
        live=api.json('GET','/api/jobs/'+collection['jobID'])
        if live['state']!='succeeded' or live['payload_custody']!=collection['custody'] or live['config']['requestSHA256']!=request['requestSHA256']:raise ValueError('Authenticated ancestor differs')
    if cc['jobID']!=r['completeBankJobID'] or cr['requestSHA256']!=r['completeBankRequestSHA256']:raise ValueError('Complete-bank lineage differs')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir();old=read(parent/'result/source.json');seal=read(parent/'result/prediction-seal.json');ref=read(parent/'result/reference-receipt.json');base=j['result']['diagnostics']['cwd'];atoms=read(complete/'result/source.json')
    source={k:old[k] for k in ['asset','ports','geometry','pairs','parentConsumerParameterSource','completeBankRoot']}
    if ref['parameters']['source']!=source['parentConsumerParameterSource']:raise ValueError('Parent parameter identity differs')
    source.update(parentSeal={'path':base+'/predictions/prediction-seal.json','sha256':sha(parent/'result/prediction-seal.json')},parentBanks=seal['rows'],
        parentControls=[x for x in ref['rows'] if x['kind']=='control-banks'],parentConsumers=[x for x in ref['rows'] if x['kind']=='consumers'],store=atoms['store'],manifestSHA256=atoms['manifestSHA256'])
    save(p/'source.json',source);save(p/'compile-source.json',{k:source[k] for k in ['asset','store','manifestSHA256']});save(p/'serve-template.json',{k:source[k] for k in ['asset','ports','geometry']})
    files=sorted({name for name in r['sourceHashes'] if name.endswith('.py') or name=='encoder-atoms-source.json'}|{'rank_one_tail.py','rank_one_tail_study.py'})
    for name in files:shutil.copyfile(HERE/name,p/name)
    request={'schema':'tce-rank-one-tail-request-v1','parentJobID':c['jobID'],'parentRequestSHA256':r['requestSHA256'],'completeBankJobID':cc['jobID'],'completeBankRequestSHA256':cr['requestSHA256'],
        'pairs':source['pairs'],'budget':4,'scope':'Retrospective saved pair,32heads/512queries. Source-product rank1 residual score, actual streaming mixed moments, full centroid/currentquery exact top4 index. Not rawprompt/fullencoder/zero-neural image.',
        'sourcePolicy':'Immutable source key covariance and uncentered routed query-displacement second moment, pooled equally over4heads/GQAgroup. Symmetric resolved-support eigensystem, positive largest-coordinate sign. No promptfit/rank sweep.',
        'statePolicy':'All64anchors updated on admission; original anchor-score cache and absolute envelopes explicitly paid. Selected-only path or same-gauge downdate. FP64 engineering envelope fallback to same rank1 direct law, perrow recorded/counted, no silent clamp.',
        'sealPolicy':'Source compiler child reads source atoms only. Serving child reads geometry/basis/supplied ports only; complete banks and state snapshots sealed before historical output decoding, direct diagnostics and original parameters.',
        'consumerPolicy':'One streaming consumer/prompt. Directrank1 inherits only for complete identical BF16bank, else one additional counted diagnostic consumer/prompt. Authenticate fullquadratic/centroid/exact/zero parent input/parameter identities before inheritance; no fresh native parity.',
        'observables':'Head/endpoint/paired-state errors, perhead/active/padding/before8/at8/after8 and nofallback/allheadmoment strata. Signed rankloss/statearithmetic/lawloss/rounding energies. Independent postseal directrank1, support/order/routes checks, purestate versus fallback closure; no accuracy stopping gate.',
        'sourceHashes':{f.name:sha(f) for f in p.iterdir()},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':600}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request);save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    verify_request(root/'payload',r);cmd=command()[:-5]+['python','-u','rank_one_tail_study.py','execute','.'];cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-rank-one-tail-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512,'heads':32,'budget':4},timeout_s=r['timeoutSeconds'],detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU math1thread; source-only rank1 basis compilation then causal mixedmoments2savedportsx32headsx512rows/all64anchors, fullKV top4 exactselector. Seal all banks/states before directrank1 reference/old outputs/originalweights.2stream originalconsumers plus atmost2conditionaldirect consumers. Explicit normalization fallbacks/envelopes/scorecache. Hostlocal atomstore/ports/checkpoint,defaultadmission,expected under3min,finite600s,no retry; not zero-neural images.')
    result={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',result);return result


def reference(port,m,asset,basis,pred):
    """Recompute admission operands independently after the complete candidate seal."""
    d,heads,groups=geometry(m);hpg=heads//groups;n=len(port['mask']);raw=np.zeros((n,heads*d));support=np.full((n,heads,4),-1,dtype=np.int64);routes=np.empty((n,heads),dtype=np.int64)
    for g in range(groups):
        p=packet(port,m,g*hpg);U=floats(p['k']).astype(float)/d**.25-asset['keyCenters'][g];V=floats(p['v']).astype(float);ell=asset['landmarks'][g*hpg:(g+1)*hpg];rank=ell.shape[1]
        scores=np.stack([(ell.reshape(-1,d)@u).reshape(hpg,rank) for u in U],axis=2);z=np.array([basis['b'][g]@u for u in U]);root=asset['metricRoots'][g];lm=ell@root
        for h in range(g*hpg,(g+1)*hpg):
            local=h-g*hpg;x=floats(packet(port,m,h)['q']).astype(float)/d**.25
            for i in range(n):
                r=int(np.argmin(np.sum((x[i]@root-lm[local])**2,axis=1)));routes[i,h]=r;allowed=np.flatnonzero(port['mask'][:i+1])
                if not len(allowed):continue
                chosen=select(U[allowed]@x[i],allowed,4);support[i,h,:len(chosen)]=allowed[chosen]
                # Read the candidate's declared route/support to isolate its state arithmetic.
                r=int(pred['nearestIndex'][i,h]);selected=ordered_support(pred['supportPositions'][i,h],allowed,4);delta=x[i]-ell[local,r]
                raw[i,h*d:(h+1)*d]=direct(scores[local,r,allowed],U[allowed],V[allowed],z[allowed],selected,delta,float(basis['a'][g]@delta))
    return {'raw':raw,'rounded':words(raw),'supportPositions':support,'nearestIndex':routes}


def head_scores(label,banks,masks,d,heads):
    rows=[]
    for method in METHODS:
        for precision in ['raw','rounded']:
            a=banks[method][precision].reshape(-1,heads,d);b=banks['11'][precision].reshape(-1,heads,d)
            for region,use in masks:
                if not use.any():continue
                for h in [None,*range(heads)]:rows.append({**label,'method':method,'precision':precision,'region':region,'queryHead':h,'score':summary(a[use] if h is None else a[use,h],b[use] if h is None else b[use,h])})
    return rows


def decompositions(label,banks,masks,d,heads):
    err=banks['stream']['rounded']-banks['11']['rounded'];terms={'sourceRank':banks['direct']['raw']-banks['quadratic']['raw'],'stateArithmetic':banks['stream']['raw']-banks['direct']['raw'],
        'fullQuadraticLaw':banks['quadratic']['raw']-banks['11']['raw'],'rounding':err-(banks['stream']['raw']-banks['11']['raw'])};err=err.reshape(-1,heads,d);terms={k:a.reshape(-1,heads,d) for k,a in terms.items()};rows=[]
    for region,use in masks:
        if not use.any():continue
        for h in [None,*range(heads)]:
            part=lambda a:a[use] if h is None else a[use,h]
            rows.append({**label,'region':region,'queryHead':h,**energy(part(err),{k:part(a) for k,a in terms.items()})})
    return rows


def closure(pred,ref,d,heads):
    err=(pred['raw']-ref['raw']).reshape(-1,heads,d);changed=(pred['rounded']!=ref['rounded']).reshape(-1,heads,d);fallback=pred['fallbackReason']!=0;selected=pred['selectedOnly'].astype(bool);rows=[]
    for label,use in [('all',np.ones_like(fallback)),('pureMoment',~fallback&~selected),('selectedOnly',selected),('fallback',fallback)]:
        for h in [None,*range(heads)]:
            pick=use if h is None else use[:,h];a=err[pick] if h is None else err[:,h][pick];bits=changed[pick] if h is None else changed[:,h][pick]
            rows.append({'region':label,'queryHead':h,'headQueryCount':int(pick.sum()),'rawMaxAbs':float(np.max(np.abs(a))) if a.size else None,'rawRms':float(np.sqrt(np.mean(a*a))) if a.size else None,'roundedWordMismatches':int(bits.sum())})
    return rows


def execute(root):
    if NoNeural.roots.intersection(sys.modules):raise RuntimeError('Neural framework imported')
    sys.meta_path.insert(0,NoNeural());root=Path(root).resolve();r=read(root/'request.json');verify_request(root,r);start=time.perf_counter();source=read(root/'source.json');out=root/'response-result';out.mkdir(exist_ok=False)
    tick=time.perf_counter();subprocess.run([sys.executable,str(root/'rank_one_tail.py'),'compile',str(root/'compile-source.json'),str(root/'compiled')],check=True);compile_s=time.perf_counter()-tick
    basis_entry={'path':str(root/'compiled/rank-one-basis.npz'),'sha256':sha(root/'compiled/rank-one-basis.npz')};save(root/'serve-source.json',read(root/'serve-template.json')|{'basis':basis_entry})
    tick=time.perf_counter();subprocess.run([sys.executable,str(root/'rank_one_tail.py'),'serve',str(root/'serve-source.json'),str(root/'predictions')],check=True);child_s=time.perf_counter()-tick
    seal_path=root/'predictions/prediction-seal.json';seal_sha=sha(seal_path);seal=read(seal_path);print('ALL_RANK_ONE_BANKS_AND_STATES_SEALED',seal_sha,flush=True)
    if sha(source['parentSeal']['path'])!=source['parentSeal']['sha256']:raise ValueError('Parent seal differs')
    asset=loaded(source['asset']);basis=loaded(basis_entry);m=source['geometry'];d,heads,_=geometry(m);work=root/'reference-work';work.mkdir()
    tick=time.perf_counter();params,config,param_receipt=parameters(read(root/'encoder-atoms-source.json'));parameter_s=time.perf_counter()-tick
    if param_receipt['source']!=source['parentConsumerParameterSource']:raise ValueError('Original parameter source differs')
    all_banks={};all_results={};masks={};pure={};checks=[];refrows=[];scores=[];paired=[];hs=[];decomposed=[];diagnostics=[];timings=[];observations=0;consumer_arms=0
    with (out/'observations.jsonl').open('x') as f:
        for item in source['ports']:
            name=item['id'];port=loaded(item);n=len(port['mask']);mask=port['mask'].astype(bool);masks[name]=mask;tick=time.perf_counter();pred=loaded(next(x for x in seal['rows'] if x['id']==name));ref=reference(port,m,asset,basis,pred)
            quad=loaded(next(x for x in source['parentBanks'] if x['id']==name));old=loaded(next(x for x in source['parentControls'] if x['id']==name));cons=loaded(next(x for x in source['parentConsumers'] if x['id']==name))
            if not np.isfinite(pred['raw']).all() or not np.array_equal(words(pred['raw']),pred['rounded']):raise ValueError('Finite/BF16 candidate mechanics differ')
            check={'id':name,'independentSupportOrOrderMismatches':int(np.count_nonzero(pred['supportPositions']!=ref['supportPositions'])),'independentRouteMismatches':int(np.count_nonzero(pred['nearestIndex']!=ref['nearestIndex'])),
                'parentSupportOrOrderMismatches':int(np.count_nonzero(pred['supportPositions']!=quad['supportPositions'])),'parentRouteMismatches':int(np.count_nonzero(pred['nearestIndex']!=quad['nearestIndex'])),'streamVsDirect':closure(pred,ref,d,heads)}
            path=work/(name+'-direct-rank1.npz');np.savez(path,**ref);refrows.append({'id':name,'kind':'direct-rank1','path':str(path),'sha256':sha(path)});diag_s=time.perf_counter()-tick;tick=time.perf_counter()
            raw={'stream':pred['raw'],'direct':ref['raw'],'quadratic':quad['raw'],**{arm:old[arm+'__raw'] for arm in ['00','11']}};bits={'stream':pred['rounded'],'direct':ref['rounded'],'quadratic':quad['rounded'],**{arm:old[arm+'__rounded'] for arm in ['00','11']}}
            results={method:{stage:cons[method+'__'+stage] for stage in STAGES} for method in ['quadratic','00','11','zero']};computed=[]
            for method in ['stream','direct']:
                if method=='direct' and np.array_equal(bits['direct'],bits['stream']):results[method]=results['stream'];continue
                results[method]=consume(port['e'],np.zeros((n,config['hidden_size'])),floats(bits[method]).astype(float),params['output'],params['gamma'],config['rms_norm_eps'],np.arange(n),[params[k] for k in ['gate','up','down']]);consumer_arms+=1;computed.append(method)
            results['exact']=results['11'];check.update(newOriginalConsumerArms=computed,inheritedConsumerArms=['quadratic','00','11','zero']+(['direct=stream'] if 'direct' not in computed else []));checks.append(check)
            path=work/(name+'-consumers.npz');np.savez(path,**{method+'__'+stage:results[method][stage] for method in METHODS+['zero'] for stage in STAGES});refrows.append({'id':name,'kind':'consumers','path':str(path),'sha256':sha(path)})
            timings.append({'id':name,'postSealDirectDiagnosticsSeconds':diag_s,'originalConsumersAndSerializationSeconds':time.perf_counter()-tick});banks={method:{'raw':raw[method],'rounded':floats(bits[method]).astype(float)} for method in METHODS};all_banks[name]=banks;all_results[name]=results
            nofallback=np.all(pred['fallbackReason']==0,axis=1);moment=nofallback&np.all(pred['selectedOnly']==0,axis=1);pure[name]=(nofallback,moment);sm=strata(mask)+[('noFallback',nofallback),('allHeadsMoment',moment)];label={'id':name,'kind':'endpoint'};valid={method:np.ones(n,dtype=bool) for method in METHODS}
            rows,count=case_scores(label,results,valid,sm,METHODS,f);scores+=rows;observations+=count;hs+=head_scores(label,banks,sm,d,heads);decomposed+=decompositions(label,banks,sm,d,heads)
            for key,value in pred.items():
                if value.shape!=(n,heads) or key=='nearestIndex':continue
                for region,use in strata(mask):
                    if not use.any():continue
                    for h in [None,*range(heads)]:diagnostics.append({'id':name,'field':key,'region':region,'queryHead':h,**describe(value[use] if h is None else value[use,h])})
            print('RANK_ONE_CONSUMERS_COMPLETE',name,flush=True)
        a,b=source['pairs'][0];sm=strata(masks[a],masks[b])+[('noFallback',pure[a][0]&pure[b][0]),('allHeadsMoment',pure[a][1]&pure[b][1])];label={'a':a,'b':b,'kind':'paired'};banks={method:{p:all_banks[b][method][p]-all_banks[a][method][p] for p in ['raw','rounded']} for method in METHODS};hs+=head_scores(label,banks,sm,d,heads);decomposed+=decompositions(label,banks,sm,d,heads)
        for method in METHODS:
            for stage in STAGES:
                pred,exact,zero=[floats(all_results[b][arm][stage]).astype(float)-floats(all_results[a][arm][stage]).astype(float) for arm in [method,'11','zero']]
                for region,use in sm:
                    if use.any():paired.append({**label,'method':method,'stage':stage,'region':region,**effect_scores(pred,exact,zero,use,np.ones(len(pred),dtype=bool))})
                er,tr=[rowrms(z) for z in [pred-exact,exact-zero]]
                for i in range(len(pred)):f.write(json.dumps({**label,'method':method,'stage':stage,'position':i,'consumerErrorRms':float(er[i]),'attentionEffectRms':float(tr[i])},allow_nan=False)+'\n');observations+=1
    if sha(seal_path)!=seal_sha:raise ValueError('Prediction seal changed')
    for name in ['request.json','source.json','compile-source.json','serve-template.json','serve-source.json']:shutil.copyfile(root/name,out/name)
    shutil.copyfile(seal_path,out/'prediction-seal.json');shutil.copyfile(root/'compiled/compile-receipt.json',out/'compile-receipt.json')
    save(out/'reference-receipt.json',{'rows':refrows,'checks':checks,'parameters':param_receipt,'scope':'Postseal directrank1 diagnostics; inherited authenticated same-input fullquadratic/centroid/exact/zero controls. Only stream/conditionaldirect original consumers newly executed. No new exact-head evaluations.'})
    save(out/'state-diagnostics.json',{'rows':diagnostics,'scope':'Dependent perquery/head engineering conditioning/radius/fallback diagnostics, not scientific gates or formal error certification.'})
    save(out/'report.json',{'schema':'tce-rank-one-tail-result-v1','requestSHA256':r['requestSHA256'],'predictionSealSHA256':seal_sha,'scope':r['scope'],'serving':seal,'scores':scores,'pairedScores':paired,'headScores':hs,'headDecompositions':decomposed,
        'checks':checks,'timings':timings,'compilationChildSeconds':compile_s,'servingChildSeconds':child_s,'originalParameterReadSeconds':parameter_s,'workerSeconds':time.perf_counter()-start,'observations':observations,
        'originalEquationWork':{'newConsumerArms':consumer_arms,'fullOutputProjections':consumer_arms,'mlpDenseProjections':3*consumer_arms,'rowsPerArm':512},'newMathematicalExactHeadReferences':0,'postSealDirectRankOneHeadReferences':heads*len(source['ports']),
        'frameworkCalls':0,'originalLearnedConsumerEquationsEvaluated':True,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed','goalStatus':'Active. A supplied-operand first-block component experiment; arbitrary-prompt zero-neural images remain open.'});print('RANK_ONE_STUDY_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');p.add_argument('--complete');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent,a.complete)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
