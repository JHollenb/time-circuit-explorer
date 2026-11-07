"""Post-exit original-bank AV, allocation identity and own-state response rows."""
import json
from pathlib import Path
import numpy as np
from encoder_atoms import read, sha
from paired_fields import load, retain, metrics
from image_dual_analysis import NAMES, effect


def execute(root):
    root=Path(root);out=root/'response-result';r,s,targets=[read(root/n) for n in ['request.json','inputs.json','targets.json']]
    actor=read(out/'actor.json');seal=read(out/'actor-seal.json')
    if seal['actorSHA256']!=sha(out/'actor.json') or actor['compilerSealSHA256']!=s['compilerSealSHA256']:
        raise ValueError('Actor must seal and exit before reference access')
    a=actor['allocate']['fields'];v=actor['value']['fields'];old=s['matchedControlFields']
    matched={n:a[n]['payloadSHA256']==old[n]['payloadSHA256'] for n in
             ['imageNumerator','textLogits','textK_hybrid','logits','probability','peak','partition']}
    matched['actualImageModLNInput']=actor['image_modln']['fields']['x']['payloadSHA256']==s['controlModLN']['payloadSHA256']
    probability=load(a['probability']);bank=load(s['banks']['Q']['V']);bank[:,r['support']]=load(s['selected']['V'])
    bank_h=bank[0].transpose(1,0,2);reference=probability@bank_h
    ref_record=retain(root/'arrays/diagnostics','sameAllocationNativeBankH',reference)
    raw=load(v['H_FP64']);admitted=load(v['H']).reshape(1,4,24,128)[0].transpose(1,0,2)
    candidate={**actor['source']['fields'],**actor['next_query']['fields']}
    arrays={n:{f:load(x[f]) for f in NAMES} for n,x in targets.items()}
    arrays['primary']={f:load(candidate[f]) for f in NAMES}
    scores,observations={},[]
    for field in NAMES:
        pp,qq=arrays['PP'][field],arrays['QQ'][field];response=qq-pp
        scores[field]={n+'Response':metrics(arrays[n][field]-pp,response) for n in ['primary','retained','stale']}
        scores[field].update(primaryEndpoint=metrics(arrays['primary'][field],qq),zeroResponse=metrics(np.zeros_like(response),response))
        for j,row in enumerate(r['support']):
            observations.append({'field':field,'panel':'whole-row','row':row,
                **{n+'Response':metrics((arrays[n][field]-pp)[0,j],response[0,j]) for n in ['primary','retained','stale']}})
        if field in ['nextQProjected','nextQNormalized','nextQ']:
            for j,row in enumerate(r['support']):
                for h in range(24):
                    observations.append({'field':field,'panel':'query-head','row':row,'head':h,
                        **{n+'Response':metrics((arrays[n][field]-pp)[0,j,h],response[0,j,h]) for n in ['primary','retained','stale']}})
    head=lambda x:x.reshape(1,4,24,128)[0].transpose(1,0,2)
    pp,qq=head(arrays['PP']['H']),head(arrays['QQ']['H']);response=qq-pp
    old_raw=load(old['H_FP64']);old_admitted=head(arrays['retained']['H'])
    parts={'newAdmission':admitted-raw,'deferredVsNativeBank':raw-reference,
           'allocationOrContraction':reference-old_raw,'negativeOldAdmission':old_raw-old_admitted}
    retained={n:retain(root/'arrays/decomposition',n,x) for n,x in parts.items()}
    total=admitted-old_admitted;telescope=sum(parts.values())
    for h in range(24):
        for j,row in enumerate(r['support']):
            observations.append({'field':'H_value_policy','panel':'query-head-policy','row':row,'head':h,
                **{n:effect(x[h,j],response[h,j]) for n,x in parts.items()},'total':effect(total[h,j],response[h,j])})
    with (out/'observations.jsonl').open('x') as stream:
        for row in observations:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    execution=read(out/'actor-execution.json')
    return {'scores':scores,'matchedControlPayloads':matched,
        'diagnostics':{'deferredRawVsSameAllocationNativeV':metrics(raw,reference),
            'newAdmittedVsRetainedAdmitted':metrics(admitted,old_admitted),
            'nativeBankAVVsRetainedRaw':metrics(reference,old_raw),
            'valueAttributionApplicability':'matched-allocation' if all(matched.values()) else 'allocation-input-mismatch',
            'scope':'Native V per-row BF16 projection policy and deferred FP64 reassociation differ jointly; no isolated rounding-cause claim.'},
        'headDifferenceTelescope':{**{n:effect(x,response) for n,x in parts.items()},'total':effect(total,response),
            'telescopingRoundoffMaxAbs':float(np.abs(telescope-total).max())},
        'headDifferenceCrossEnergy':{a:{b:float(np.mean(x*y)) for b,y in parts.items()} for a,x in parts.items()},
        'retainedDecomposition':retained,'sameAllocationNativeBankH':ref_record,'observations':len(observations),
        'diagnosticMatrixOperations':[{'operation':'post-seal-native-bank-AV','inputShape':list(probability.shape),
            'rightShape':list(bank_h.shape),'dtype':'float64','device':'cpu','nominalMACs':18874368}],
        'workByPhase':{'actor':execution['nominalMatrixMACs'],'postSealBankAV':18874368},
        'totalNewMatrixMACs':execution['nominalMatrixMACs']+18874368,
        'distinctBF16ParameterBytes':sum(x['bytes'] for x in execution['loadedParameters']),
        'primaryCandidate':'Same cycle58 live allocation, deferred head-specific V projections, own native MLP and next query',
        'scope':'One supplied P/Q source pair and four demanded queries. Original states/ModLN/maps/duals/textkeys/retainedPnorm/native references remain paid. No full V or selected V numerical input before seal; post-seal original-bank AV only, no extra consumer,fit,suffix,image or arbitrary-prompt zero-neural completion.'}
