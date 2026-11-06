"""Seal one quadratic tail bank, then measure exact diagnostics/original consumers."""
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
from context_statistics_study import rowrms
from selector_objective_study import strata,describe
from sparse_tilt_study import case_scores
from positive_tail_diagnostics import METHODS,VECTORS,SCALARS,diagnose_head,head_rows,decompositions
from transport_consumer import parameters,consume

HERE=Path(__file__).resolve().parent


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,j,r=checked_collection(parent,current_workers=False);api=Api()
    live=api.json('GET','/api/jobs/'+c['jobID']);ancestor=api.json('GET','/api/jobs/'+r['completeBankJobID'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody'] or live['config']['requestSHA256']!=r['requestSHA256']:raise ValueError('Factorial parent differs')
    if ancestor['state']!='succeeded' or ancestor['config']['requestSHA256']!=r['completeBankRequestSHA256']:raise ValueError('Complete-bank ancestor differs')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir();old=read(parent/'result/source.json');seal=read(parent/'result/prediction-seal.json');ref=read(parent/'result/reference-receipt.json');base=j['result']['diagnostics']['cwd']
    source={k:old[k] for k in ['asset','ports','geometry','pairs','references','parentConsumerParameterSource','completeBankRoot']}
    if ref['parameters']['source']!=source['parentConsumerParameterSource']:raise ValueError('Parent parameter identity differs')
    source.update(parentSeal={'path':base+'/predictions/prediction-seal.json','sha256':sha(parent/'result/prediction-seal.json')},parentBanks=[x for x in seal['rows'] if x['arm'] in ['00','11']],parentConsumers=[x for x in ref['rows'] if x['kind']=='consumers'])
    save(p/'source.json',source);save(p/'serve-source.json',{k:source[k] for k in ['asset','ports','geometry']}|{'frozenSupports':[x for x in source['parentBanks'] if x['arm']=='00'],'budget':4})
    files=['positive_tail.py','positive_tail_diagnostics.py','positive_tail_study.py','tail_factorial.py','tail_factorial_study.py','selector_objective.py','selector_objective_study.py',
        'centroid_tilt.py','centroid_tilt_study.py','sparse_tilt.py','sparse_tilt_study.py','attention_bank.py','attention_bank_study.py','encoder_atoms.py','encoder-atoms-source.json',
        'context_source_geometry.py','context_landmarks.py','context_transport.py','context_statistics.py','context_statistics_study.py','context_transport_study.py','transport_consumer.py']
    for name in files:shutil.copyfile(HERE/name,p/name)
    request={'schema':'tce-positive-tail-request-v1','parentJobID':c['jobID'],'parentRequestSHA256':r['requestSHA256'],'completeBankJobID':r['completeBankJobID'],'completeBankRequestSHA256':r['completeBankRequestSHA256'],
        'pairs':source['pairs'],'budget':4,'scope':'Retrospective supplied pair/all32heads/512rows. Frozen currentquery top4/order/routes, fullprefix directional quadratic tail law. Not compact serving, freshrawtext/fullencoder or zero-neural image.',
        'law':'t=(U_tail-muTail)@delta; r2=((t+1)^2+1)/2>0. Same anchor-weighted positive measure for tailmass and tiltedvalue. Logr2/logmasses canonical, measured meanT retained, no target fit/covariance tensor/lowrank state.',
        'sealPolicy':'Fresh candidate-only child decodes frozen support/order/routes only; no exact-tail diagnostics, old output decoding or original parameters before complete raw/BF16 candidate seal.',
        'consumerPolicy':'One new quadratic consumer/prompt; inherit centroid/exact/zero only after sameboundports/sourceparameter identity and complete reconstructed control BF16bank identity. Mismatch consumers newly evaluated and counted, candidate never refreshed.',
        'observables':'Endpoint/paired heads/receivingstate, allperhead/activepadding/before8/at8/after8; signed mass/value/controlarithmetic/rounding energies and crossenergies. Exact/quadratic/anchor tilted mass on t<-1, centeringmean/variance, Gaussian logkappa-meanT-varT/2 and value-Cov(V,t) residuals; no extra Gaussianconsumer.',
        'sourceHashes':{f.name:sha(f) for f in p.iterdir()},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':600}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    verify_request(root/'payload',r);cmd=command()[:-5]+['python','-u','positive_tail_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-positive-tail-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512,'heads':32,'budget':4},
        timeout_s=r['timeoutSeconds'],detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 two savedportsx32headsx512queries; seal ONEpositivequadratic bank perprompt before exacttail/controloutputs/originalweights. Fixed orderedtop4/routes, directional moments only/no higherorder tensor. Then64exacthead references and2new original512row consumers total, extra mismatchedcontrols separatelycounted. Hostlocal immutable operands/5GBshard,defaultadmission,expected under3min,finite600s,no retry. Diagnostic law,not zero-neural images.')
    result={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',result);return result


def execute(root):
    if NoNeural.roots.intersection(sys.modules):raise RuntimeError('Neural framework imported')
    sys.meta_path.insert(0,NoNeural());root=Path(root).resolve();r=read(root/'request.json');verify_request(root,r);start=time.perf_counter();source=read(root/'source.json');out=root/'response-result';out.mkdir(exist_ok=False)
    tick=time.perf_counter();subprocess.run([sys.executable,str(root/'positive_tail.py'),str(root/'serve-source.json'),str(root/'predictions')],check=True);child_s=time.perf_counter()-tick
    seal_path=root/'predictions/prediction-seal.json';seal_sha=sha(seal_path);seal=read(seal_path);print('ALL_POSITIVE_TAIL_BANKS_SEALED',seal_sha,flush=True)
    asset=loaded(source['asset']);m=source['geometry'];d,heads,_=geometry(m);work=root/'reference-work';work.mkdir()
    tick=time.perf_counter();params,config,param_receipt=parameters(read(root/'encoder-atoms-source.json'));parameter_s=time.perf_counter()-tick
    if param_receipt['source']!=source['parentConsumerParameterSource']:raise ValueError('Original parameter source differs')
    all_banks={};all_fields={};all_results={};masks={};checks=[];refrows=[];scores=[];paired=[];head_scores=[];decomposed=[];diagnostics=[];timings=[];observations=0;consumer_arms=0
    with (out/'observations.jsonl').open('x') as f:
        for item in source['ports']:
            name=item['id'];port=loaded(item);n=len(port['mask']);mask=port['mask'].astype(bool);masks[name]=mask;tick=time.perf_counter();pred=loaded(next(x for x in seal['rows'] if x['id']==name))
            old={arm:loaded(next(x for x in source['parentBanks'] if x['id']==name and x['arm']==arm)) for arm in ['00','11']}
            raw={method:np.empty((n,heads*d)) for method in METHODS};fields={}
            for h in range(heads):
                span=slice(h*d,(h+1)*d);frozen={key:old['00'][key][:,h] for key in ['supportPositions','nearestIndex']}
                banks,diag=diagnose_head(packet(port,m,h),head_asset(asset,m,h),frozen,r['budget'])
                for method in METHODS:raw[method][:,span]=banks[method]
                for key,value in diag.items():
                    if key not in fields:fields[key]=np.empty((n,heads*d) if key in VECTORS else (n,heads))
                    fields[key][:,span if key in VECTORS else h]=value
            check={'id':name,'candidateRawBitMismatches':int(np.count_nonzero(raw['quadratic'].view(np.uint64)!=pred['raw'].view(np.uint64))),
                'candidateRoundedWordMismatches':int(np.count_nonzero(words(raw['quadratic'])!=pred['rounded'])),
                'frozenSupportOrOrderMismatches':int(np.count_nonzero(pred['supportPositions']!=old['00']['supportPositions'])),
                'routeMismatches':int(np.count_nonzero(pred['nearestIndex']!=old['00']['nearestIndex'])),
                'parentControlWordMismatches':{arm:int(np.count_nonzero(words(raw[arm])!=old[arm]['rounded'])) for arm in ['00','11']},
                'parentControlRawBitMismatches':{arm:int(np.count_nonzero(raw[arm].view(np.uint64)!=old[arm]['raw'].view(np.uint64))) for arm in ['00','11']}}
            if any(check[k] for k in ['candidateRawBitMismatches','candidateRoundedWordMismatches','frozenSupportOrOrderMismatches','routeMismatches']):raise ValueError('Candidate replay/support identity differs')
            path=work/(name+'-tail-diagnostics.npz');np.savez(path,**fields);refrows.append({'id':name,'kind':'tail-diagnostics','path':str(path),'sha256':sha(path)})
            path=work/(name+'-controls.npz');np.savez(path,**{arm+'__'+precision:(raw[arm] if precision=='raw' else words(raw[arm])) for arm in ['00','11'] for precision in ['raw','rounded']});refrows.append({'id':name,'kind':'control-banks','path':str(path),'sha256':sha(path)})
            diagnostic_s=time.perf_counter()-tick;tick=time.perf_counter();old_consumer=loaded(next(x for x in source['parentConsumers'] if x['id']==name));results={};computed=[];inherited=[]
            for method in METHODS:
                if method in ['00','11'] and check['parentControlWordMismatches'][method]==0:
                    results[method]={stage:old_consumer[method+'__'+stage] for stage in STAGES};inherited.append(method)
                else:
                    bank=pred['rounded'] if method=='quadratic' else words(raw[method]);results[method]=consume(port['e'],np.zeros((n,config['hidden_size'])),floats(bank).astype(float),params['output'],params['gamma'],config['rms_norm_eps'],np.arange(n),[params[k] for k in ['gate','up','down']]);consumer_arms+=1;computed.append(method)
            results['zero']={stage:old_consumer['zero__'+stage] for stage in STAGES};results['exact']=results['11'];check.update(inheritedConsumerArms=inherited+['zero'],newOriginalConsumerArms=computed);checks.append(check)
            path=work/(name+'-consumers.npz');np.savez(path,**{method+'__'+stage:results[method][stage] for method in METHODS+['zero'] for stage in STAGES});refrows.append({'id':name,'kind':'consumers','path':str(path),'sha256':sha(path)})
            timings.append({'id':name,'postSealExactAndTailDiagnosticsSeconds':diagnostic_s,'originalConsumerAndSerializationSeconds':time.perf_counter()-tick})
            banks={method:{'raw':pred['raw'] if method=='quadratic' else raw[method],'rounded':floats(pred['rounded'] if method=='quadratic' else words(raw[method])).astype(float)} for method in METHODS};all_banks[name]=banks;all_fields[name]=fields;all_results[name]=results
            label={'id':name,'kind':'endpoint'};sm=strata(mask);valid={method:np.ones(n,dtype=bool) for method in METHODS}
            rows,count=case_scores(label,results,valid,sm,METHODS,f);scores+=rows;observations+=count;head_scores+=head_rows(label,banks,sm,d,heads);decomposed+=decompositions(label,banks,fields,sm,d,heads)
            for key,value in fields.items():
                vector=key in VECTORS;a=value.reshape(n,heads,d) if vector else value
                for region,use in sm:
                    if not use.any():continue
                    for h in [None,*range(heads)]:
                        z=a[use] if h is None else a[use,h]
                        if vector:z=np.sqrt(np.mean(z*z,axis=-1))
                        diagnostics.append({'id':name,'field':key+('Rms' if vector else ''),'region':region,'queryHead':h,**describe(z)})
            print('POSITIVE_TAIL_CONSUMER_COMPLETE',name,flush=True)
        a,b=source['pairs'][0];sm=strata(masks[a],masks[b]);label={'a':a,'b':b,'kind':'paired'};banks={method:{precision:all_banks[b][method][precision]-all_banks[a][method][precision] for precision in ['raw','rounded']} for method in METHODS};fields={key:all_fields[b][key]-all_fields[a][key] for key in VECTORS}
        head_scores+=head_rows(label,banks,sm,d,heads);decomposed+=decompositions(label,banks,fields,sm,d,heads)
        for method in METHODS:
            for stage in STAGES:
                pred,exact,zero=[floats(all_results[b][arm][stage]).astype(float)-floats(all_results[a][arm][stage]).astype(float) for arm in [method,'11','zero']]
                for region,use in sm:
                    if use.any():paired.append({**label,'method':method,'stage':stage,'region':region,**effect_scores(pred,exact,zero,use,np.ones(len(pred),dtype=bool))})
                er,tr=[rowrms(z) for z in [pred-exact,exact-zero]]
                for i in range(len(pred)):
                    f.write(json.dumps({**label,'method':method,'stage':stage,'position':i,'consumerErrorRms':float(er[i]),'attentionEffectRms':float(tr[i])},allow_nan=False)+'\n');observations+=1
    if sha(seal_path)!=seal_sha:raise ValueError('Prediction seal changed')
    for name,path in [('request.json',root/'request.json'),('source.json',root/'source.json'),('serve-source.json',root/'serve-source.json'),('prediction-seal.json',seal_path)]:shutil.copyfile(path,out/name)
    save(out/'reference-receipt.json',{'rows':refrows,'checks':checks,'parameters':param_receipt,'exactHeadReferenceEvaluations':heads*len(source['ports']),'scope':'All exact/centroid/negative-tail/Gaussian diagnostics and original learned consumers occur after candidate seal. Controls inherited only under complete input identity; extra consumers counted.'})
    save(out/'tail-diagnostics.json',{'rows':diagnostics,'scope':'Perhead and pooled dependent head/query descriptions; missing tails/zero-effect ratios explicitly undefined. Positive directional moment capacity, not compact state or universal Gaussian law.'})
    save(out/'report.json',{'schema':'tce-positive-tail-result-v1','requestSHA256':r['requestSHA256'],'predictionSealSHA256':seal_sha,'scope':r['scope'],'serving':seal,'scores':scores,'pairedScores':paired,'headScores':head_scores,'headDecompositions':decomposed,
        'checks':checks,'timings':timings,'servingChildSeconds':child_s,'originalParameterReadSeconds':parameter_s,'workerSeconds':time.perf_counter()-start,'observations':observations,
        'originalEquationWork':{'newConsumerArms':consumer_arms,'fullOutputProjections':consumer_arms,'mlpDenseProjections':3*consumer_arms,'rowsPerArm':512},'mathematicalExactHeadReferences':heads*len(source['ports']),
        'frameworkCalls':0,'originalLearnedConsumerEquationsEvaluated':True,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed','goalStatus':'Active. Suppliedport positive tail-law test with original learned receiving consumer offline; arbitrary-prompt zero-neural images remain open.'})
    print('POSITIVE_TAIL_STUDY_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
