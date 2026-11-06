"""Frozen-support tail mass/value factorial and actual receiving-state response."""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from encoder_atoms import np,read,save,sha,identity,floats,NoNeural
from attention_bank import geometry
from attention_bank_study import loaded,verify_request,regions,effect_scores,STAGES
from context_statistics_study import summary,rowrms
from selector_objective_study import strata,describe
from sparse_tilt_study import case_scores
from tail_factorial import ARMS,VECTORS
from transport_consumer import parameters,consume

HERE=Path(__file__).resolve().parent


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,j,r=checked_collection(parent,current_workers=False);api=Api()
    live=api.json('GET','/api/jobs/'+c['jobID']);ancestor=api.json('GET','/api/jobs/'+r['completeBankJobID'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody'] or live['config']['requestSHA256']!=r['requestSHA256']:raise ValueError('Selector parent differs')
    if ancestor['state']!='succeeded' or ancestor['config']['requestSHA256']!=r['completeBankRequestSHA256']:raise ValueError('Complete-bank ancestor differs')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir();old=read(parent/'result/source.json');seal=read(parent/'result/prediction-seal.json');ref=read(parent/'result/reference-receipt.json');base=j['result']['diagnostics']['cwd']
    source={k:old[k] for k in ['asset','ports','geometry','pairs','references','parentConsumerParameterSource','completeBankRoot']}
    source.update(parentSeal={'path':base+'/predictions/prediction-seal.json','sha256':sha(parent/'result/prediction-seal.json')},
        parentBanks=[x for x in seal['rows'] if x['method']=='current'],parentConsumers=[x for x in ref['rows'] if x['kind']=='consumers'])
    save(p/'source.json',source);save(p/'serve-source.json',{k:source[k] for k in ['asset','ports','geometry']}|{'frozenSupports':source['parentBanks'],'budget':4})
    files=['tail_factorial.py','tail_factorial_study.py','selector_objective.py','selector_objective_study.py','centroid_tilt.py','centroid_tilt_study.py','sparse_tilt.py','sparse_tilt_study.py',
        'attention_bank.py','attention_bank_study.py','encoder_atoms.py','encoder-atoms-source.json','context_source_geometry.py','context_landmarks.py',
        'context_transport.py','context_statistics.py','context_statistics_study.py','context_transport_study.py','transport_consumer.py']
    for name in files:shutil.copyfile(HERE/name,p/name)
    request={'schema':'tce-tail-factorial-request-v1','parentJobID':c['jobID'],'parentRequestSHA256':r['requestSHA256'],'completeBankJobID':r['completeBankJobID'],
        'completeBankRequestSHA256':r['completeBankRequestSHA256'],'pairs':source['pairs'],'budget':4,
        'scope':'Retrospective suppliedport pair/all32heads/512rows. Frozen currentquery top4/order/routes; fullprefix tailmass/value oracle repairs. Not compact serving, freshrawtext/fullencoder or zero-neural image.',
        'law':'2x2 positive mixtures: centroid/exact currenttailmass crossed with anchor/exact tiltedtailvalue, one common unnormalized gauge. Existing00 directcentroid and11 exacthead arithmetic separately canonical; smoothfactor vectors and controlarithmetic defects retained. Stablelogs, emptytailundefinedlogKappa.',
        'sealPolicy':'Freshchild loads only supportPositions/nearestIndex members from oldbank inputs, not raw/rounded output members. Seals all4completebanks and rawdiagnosticvectors before controloutputs/reference/originalparameter reads. Wholefile custody hashes read bytes without decoding outputmembers.',
        'consumerPolicy':'Only2newhybrid consumers/prompt by default,4output/12MLP projections. Inherit00/11 only after completeBF16bank identity and sameboundports/sourceparams; unmatchedcontrols evaluated additionally and charged. Zero inherited under same source/port identity.',
        'observables':'Endpoint/paired raw+BF16heads/receivingstate, perhead/activepadding/before8/at8/after8. Orderedmass/value signedvectors and crossenergy, controlarithmetic+rounding defects, factorialinteraction and downstreamcrossresponse, logKappa/tailweight/emptytail counts. No accuracy stopgate.',
        'sourceHashes':{f.name:sha(f) for f in p.iterdir()},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':600}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    verify_request(root/'payload',r);cmd=command()[:-5]+['python','-u','tail_factorial_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-tail-factorial-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512,'heads':32,'budget':4},
        timeout_s=r['timeoutSeconds'],detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 frozen-currenttop4 tailmass/value2x2,2savedportsx32headsx512queries. Seal4fullbanks/rawsignedvectors beforeoldoutputs/weights. Then2newhybrid originalblock consumers/prompt; unmatched00/11controls only ifbankidentityfails, allwork charged. Host-local immutable ports/source/5GBshard,defaultadmission,expected under3min,finite600s,no retry. Diagnostic fullprefix repair,notserving.')
    result={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',result);return result


def energy(error,terms):
    e=float(np.mean(error*error));diagonal={k:float(np.mean(a*a)) for k,a in terms.items()}
    cross={a+'__'+b:float(2*np.mean(terms[a]*terms[b])) for a,b in itertools.combinations(terms,2)};reconstructed=sum(terms.values())
    return {'errorEnergy':e,'errorRms':e**.5,'termEnergies':diagonal,'signedCrossEnergies':cross,
        'termEnergiesOverError':{k:a/e if e else None for k,a in diagonal.items()},'signedCrossEnergiesOverError':{k:a/e if e else None for k,a in cross.items()},
        'vectorClosureMaxAbs':float(np.max(np.abs(error-reconstructed))),'energyClosure':e-sum(diagonal.values())-sum(cross.values())}


def head_rows(label,banks,masks,d,heads):
    rows=[]
    for arm in ARMS:
        for precision in ['raw','rounded']:
            pred=banks[arm][precision].reshape(-1,heads,d);target=banks['11'][precision].reshape(-1,heads,d)
            for region,chosen in masks:
                if not chosen.any():continue
                for h in [None,*range(heads)]:
                    a,b=(pred[chosen],target[chosen]) if h is None else (pred[chosen,h],target[chosen,h])
                    rows.append({**label,'arm':arm,'precision':precision,'queryHead':h,'region':region,'score':summary(a,b)})
    return rows


def decomposition_rows(label,banks,diag,masks,d,heads):
    raw_error=banks['00']['raw']-banks['11']['raw'];rounded_error=banks['00']['rounded']-banks['11']['rounded']
    terms={k:diag[k] for k in ['mass','value','controlArithmetic']};terms['rounding']=rounded_error-raw_error
    terms={k:a.reshape(-1,heads,d) for k,a in terms.items()};err=rounded_error.reshape(-1,heads,d);out=[]
    smooth_interaction=diag['interaction'].reshape(-1,heads,d)
    raw_interaction=(banks['11']['raw']-banks['10']['raw']-banks['01']['raw']+banks['00']['raw']).reshape(-1,heads,d)
    rounded_interaction=(banks['11']['rounded']-banks['10']['rounded']-banks['01']['rounded']+banks['00']['rounded']).reshape(-1,heads,d)
    for region,chosen in masks:
        if not chosen.any():continue
        for h in [None,*range(heads)]:
            use=lambda a:a[chosen] if h is None else a[chosen,h]
            out.append({**label,'region':region,'queryHead':h,**energy(use(err),{k:use(a) for k,a in terms.items()}),
                'smoothInteractionEnergy':float(np.mean(use(smooth_interaction)**2)),
                'canonicalRawInteractionEnergy':float(np.mean(use(raw_interaction)**2)),
                'canonicalRoundedInteractionEnergy':float(np.mean(use(rounded_interaction)**2))})
    return out


def consumer_interactions(label,results,masks,stream,decoded=False):
    out=[];observations=0
    for stage in STAGES:
        a={arm:results[arm][stage] if decoded else floats(results[arm][stage]).astype(float) for arm in ARMS+['zero']};mass=a['00']-a['10'];value=a['10']-a['11'];err=a['00']-a['11'];interaction=a['11']-a['10']-a['01']+a['00'];effect=a['11']-a['zero']
        for region,chosen in masks:
            if not chosen.any():continue
            er=float(np.sqrt(np.mean(effect[chosen]**2)));cross=float(np.sqrt(np.mean(interaction[chosen]**2)))
            out.append({**label,'stage':stage,'region':region,'interactionRms':cross,'exactAttentionEffectRms':er,
                'interactionOverExactAttentionEffect':cross/er if er else None,'orderedConsumerError':energy(err[chosen],{'massFirst':mass[chosen],'valueAfterMass':value[chosen]})})
        mr,vr,cr,er=[rowrms(x) for x in [mass,value,interaction,err]];signed=2*np.mean(mass*value,axis=1)
        for i in range(len(err)):
            stream.write(json.dumps({**label,'kind':'consumer-factorial','stage':stage,'position':i,'massFirstRms':float(mr[i]),'valueAfterMassRms':float(vr[i]),'interactionRms':float(cr[i]),'baselineErrorRms':float(er[i]),'orderedSignedCrossEnergy':float(signed[i])},allow_nan=False)+'\n');observations+=1
    return out,observations


def execute(root):
    if NoNeural.roots.intersection(sys.modules):raise RuntimeError('Neural framework imported')
    sys.meta_path.insert(0,NoNeural());root=Path(root).resolve();r=read(root/'request.json');verify_request(root,r);start=time.perf_counter();source=read(root/'source.json');out=root/'response-result';out.mkdir(exist_ok=False)
    t=time.perf_counter();subprocess.run([sys.executable,str(root/'tail_factorial.py'),str(root/'serve-source.json'),str(root/'predictions')],check=True);child_s=time.perf_counter()-t
    seal_path=root/'predictions/prediction-seal.json';seal_sha=sha(seal_path);seal=read(seal_path);print('ALL_TAIL_FACTORIAL_BANKS_SEALED',seal_sha,flush=True)
    t=time.perf_counter();params,config,param_receipt=parameters(read(root/'encoder-atoms-source.json'));parameter_s=time.perf_counter()-t
    if param_receipt['source']!=source['parentConsumerParameterSource']:raise ValueError('Original parameter source differs')
    d,heads,_=geometry(source['geometry']);work=root/'reference-work';work.mkdir();all_banks={};all_diag={};all_results={};masks={};checks=[];refrows=[];scores=[];paired=[];head_scores=[];decompositions=[];interactions=[];diagnostics=[];timings=[];observations=0;consumer_arms=0
    with (out/'observations.jsonl').open('x') as f:
        for item in source['ports']:
            name=item['id'];port=loaded(item);n=len(port['mask']);mask=port['mask'].astype(bool);masks[name]=mask;tick=time.perf_counter()
            pred={arm:loaded(next(x for x in seal['rows'] if x['id']==name and x['arm']==arm)) for arm in ARMS};diag=loaded(next(x for x in seal['diagnostics'] if x['id']==name))
            old=loaded(next(x for x in source['parentBanks'] if x['id']==name));parent=loaded(next(x for x in source['references'] if x['id']==name));old_consumer=loaded(next(x for x in source['parentConsumers'] if x['id']==name))
            check={'id':name,'currentControlWordMismatches':int(np.count_nonzero(pred['00']['rounded']!=old['rounded'])),
                'exactControlWordMismatches':int(np.count_nonzero(pred['11']['rounded']!=parent['exactBank'])),
                'frozenSupportOrOrderMismatches':{arm:int(np.count_nonzero(pred[arm]['supportPositions']!=old['supportPositions'])) for arm in ARMS},
                'routeMismatches':{arm:int(np.count_nonzero(pred[arm]['nearestIndex']!=old['nearestIndex'])) for arm in ARMS},
                'maxClosure':{key:float(np.nanmax(diag[key])) for key in ['telescopeClosure','interactionClosure','centroidControlClosure','exactControlClosure']}}
            results={};inherited=[];computed=[]
            for arm in ARMS:
                if arm in ['00','11'] and check['currentControlWordMismatches' if arm=='00' else 'exactControlWordMismatches']==0:
                    results[arm]={stage:old_consumer[('current' if arm=='00' else 'exact')+'__'+stage] for stage in STAGES};inherited.append(arm)
                else:
                    results[arm]=consume(port['e'],np.zeros((n,config['hidden_size'])),floats(pred[arm]['rounded']).astype(float),params['output'],params['gamma'],config['rms_norm_eps'],np.arange(n),[params[k] for k in ['gate','up','down']]);computed.append(arm);consumer_arms+=1
            results['zero']={stage:old_consumer['zero__'+stage] for stage in STAGES};results['exact']=results['11'];check.update(inheritedConsumerArms=inherited+['zero'],newOriginalConsumerArms=computed);checks.append(check)
            path=work/(name+'-consumers.npz');np.savez(path,**{arm+'__'+stage:results[arm][stage] for arm in ARMS+['zero'] for stage in STAGES});refrows.append({'id':name,'kind':'consumers','path':str(path),'sha256':sha(path)})
            timings.append({'id':name,'postSealControlsAndOriginalConsumersSeconds':time.perf_counter()-tick})
            banks={arm:{'raw':p['raw'],'rounded':floats(p['rounded']).astype(float)} for arm,p in pred.items()};all_banks[name]=banks;all_diag[name]=diag;all_results[name]=results
            label={'id':name,'kind':'endpoint'};strata_masks=strata(mask);valid={arm:np.isfinite(pred[arm]['raw']).all(1) for arm in ARMS}
            rows,count=case_scores(label,results,valid,strata_masks,ARMS,f);scores+=rows;observations+=count
            head_scores+=head_rows(label,banks,strata_masks,d,heads);decompositions+=decomposition_rows(label,banks,diag,strata_masks,d,heads)
            rows,count=consumer_interactions(label,results,strata_masks,f);interactions+=rows;observations+=count
            for key in ['tailWeight','exactTailWeight','logKappa','hasTail','telescopeClosure','interactionClosure','centroidControlClosure','exactControlClosure']:
                for region,chosen in strata_masks:
                    if chosen.any():diagnostics.append({'id':name,'field':key,'region':region,**describe(diag[key][chosen])})
            print('TAIL_FACTORIAL_CONSUMERS_COMPLETE',name,flush=True)
        a,b=source['pairs'][0];pair_masks=strata(masks[a],masks[b]);banks={arm:{precision:all_banks[b][arm][precision]-all_banks[a][arm][precision] for precision in ['raw','rounded']} for arm in ARMS}
        diag={key:all_diag[b][key]-all_diag[a][key] for key in VECTORS};label={'a':a,'b':b,'kind':'paired'}
        head_scores+=head_rows(label,banks,pair_masks,d,heads);decompositions+=decomposition_rows(label,banks,diag,pair_masks,d,heads)
        results={arm:{stage:(floats(all_results[b][arm][stage]).astype(float)-floats(all_results[a][arm][stage]).astype(float)) for stage in STAGES} for arm in ARMS+['zero']}
        # Paired consumer differences are float arrays: do not reinterpret them as BF16 words.
        for arm in ARMS:
            for stage in STAGES:
                pred,exact,zero=[results[x][stage] for x in [arm,'11','zero']]
                for region,chosen in pair_masks:
                    if chosen.any():paired.append({'a':a,'b':b,'method':arm,'stage':stage,'region':region,**effect_scores(pred,exact,zero,chosen,np.ones(len(pred),dtype=bool))})
                er,tr=[rowrms(x) for x in [pred-exact,exact-zero]]
                for i in range(len(pred)):
                    f.write(json.dumps({**label,'arm':arm,'stage':stage,'position':i,'consumerErrorRms':float(er[i]),'attentionEffectRms':float(tr[i])},allow_nan=False)+'\n');observations+=1
        # The same interaction routine accepts an explicit already-decoded path below.
        rows,count=consumer_interactions(label,results,pair_masks,f,decoded=True);interactions+=rows;observations+=count
    if sha(seal_path)!=seal_sha:raise ValueError('Prediction seal changed')
    for name,path in [('request.json',root/'request.json'),('source.json',root/'source.json'),('serve-source.json',root/'serve-source.json'),('prediction-seal.json',seal_path)]:shutil.copyfile(path,out/name)
    save(out/'reference-receipt.json',{'rows':refrows,'checks':checks,'parameters':param_receipt,'scope':'Post-seal original receiving consumer;00/11/zero inherited only under same source/ports and matchingcompletebank; unmatchedcontrols newly evaluated and counted.'})
    save(out/'report.json',{'schema':'tce-tail-factorial-result-v1','requestSHA256':r['requestSHA256'],'predictionSealSHA256':seal_sha,'scope':r['scope'],'serving':seal,
        'scores':scores,'pairedScores':paired,'headScores':head_scores,'headDecompositions':decompositions,'consumerInteractions':interactions,'tailDiagnostics':diagnostics,
        'checks':checks,'timings':timings,'servingChildSeconds':child_s,'originalParameterReadSeconds':parameter_s,'workerSeconds':time.perf_counter()-start,'observations':observations,
        'originalEquationWork':{'newConsumerArms':consumer_arms,'fullOutputProjections':consumer_arms,'mlpDenseProjections':3*consumer_arms,'rowsPerArm':512},
        'frameworkCalls':0,'originalLearnedConsumerEquationsEvaluated':True,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'goalStatus':'Active. Frozen-support fullprefix tail mass/value diagnostic and original learned receiving consumer, not fullencoder or arbitrary-prompt zero-neural images.'})
    print('TAIL_FACTORIAL_STUDY_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
