"""Post-seal source operand comparison and actual own-state response evidence."""
import json
from pathlib import Path
import numpy as np
from encoder_atoms import read,sha
from paired_fields import load,retain,metrics
from image_dual_analysis import NAMES


def execute(root):
    root=Path(root);out=root/'response-result';r,s,targets=[read(root/n) for n in ['request.json','inputs.json','targets.json']]
    actor=read(out/'actor.json')
    if read(out/'actor-seal.json')['actorSHA256']!=sha(out/'actor.json'):raise ValueError('Actor must seal before target access')
    derived=actor['source_inputs']['fields'];query=actor['query']['fields'];keys=actor['keys']['fields'];allocation=actor['allocate']['fields']
    actual_operands={n:derived[n] for n in ['R','txtMod','ropeCos','ropeSin']};actual_operands['Q']=query['Q']
    comparisons={n:{'payloadExact':v['payloadSHA256']==s['selected'][n]['payloadSHA256'],
                    'metrics':metrics(load(v),load(s['selected'][n]))} for n,v in actual_operands.items()}
    native_k=load(s['banks']['Q']['K']);native_k[:,r['support']]=load(s['selected']['K'])
    own_k=load(keys['K']);own_q=load(query['Q']);old_q=load(s['selected']['Q'])
    old_actor=read(s['sourceReaderParentRecords']['actor.json']['path'])
    if sha(s['sourceReaderParentRecords']['actor.json']['path'])!=s['sourceReaderParentRecords']['actor.json']['sha256']:
        raise ValueError('Sealed assisted actor metadata differs')
    assisted_probability=old_actor['allocate']['fields']['probability']
    probability=load(allocation['probability']);prior_p=load(assisted_probability)
    candidate={**actor['source']['fields'],**actor['next_query']['fields']}
    arrays={n:{f:load(x[f]) for f in NAMES} for n,x in targets.items()}
    arrays['primary']={f:load(candidate[f]) for f in NAMES}
    scores,observations={},[]
    for field in NAMES:
        pp,qq=arrays['PP'][field],arrays['QQ'][field];response=qq-pp
        scores[field]={n+'Response':metrics(arrays[n][field]-pp,response) for n in ['primary','assisted','retained','stale']}
        scores[field].update(primaryEndpoint=metrics(arrays['primary'][field],qq),zeroResponse=metrics(np.zeros_like(response),response))
        for j,row in enumerate(r['support']):
            observations.append({'field':field,'panel':'whole-row','row':row,
                **{n+'Response':metrics((arrays[n][field]-pp)[0,j],response[0,j]) for n in ['primary','assisted','retained','stale']}})
        if field in ['nextQProjected','nextQNormalized','nextQ']:
            for j,row in enumerate(r['support']):
                for h in range(24):
                    observations.append({'field':field,'panel':'query-head','row':row,'head':h,
                        **{n+'Response':metrics((arrays[n][field]-pp)[0,j,h],response[0,j,h]) for n in ['primary','assisted','retained','stale']}})
    for j in range(1536):
        for h in range(24):
            observations.append({'field':'K','panel':'key-head-input','row':j,'head':h,'stream':'text' if j<512 else 'image',
                'nativeComparison':metrics(own_k[0,j,h],native_k[0,j,h])})
    for j,row in enumerate(r['support']):
        for h in range(24):
            observations.append({'field':'allocation','panel':'query-head-input','row':row,'head':h,
                'queryVsArchived':metrics(own_q[0,j,h],old_q[0,j,h]),
                'probabilityVsAssisted':metrics(probability[h,j],prior_p[h,j])})
    with (out/'observations.jsonl').open('x') as stream:
        for row in observations:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    execution=read(out/'actor-execution.json')
    return {'scores':scores,'derivedSourceOperandComparisons':comparisons,
        'diagnostics':{'ownKeysVsNativeHybrid':metrics(own_k,native_k),'probabilityVsAssisted':metrics(probability,prior_p),
            'imageModLNMatchesAssisted':actor['image_modln']['fields']['x']['payloadSHA256']==old_actor['image_modln']['fields']['x']['payloadSHA256'],
            'textModLNMatchesAssisted':actor['text_modln']['fields']['x']['payloadSHA256']==old_actor['text_modln']['fields']['x']['payloadSHA256'],
            'splicedTextMatchesAssisted':actor['splice']['fields']['x']['payloadSHA256']==old_actor['splice']['fields']['x']['payloadSHA256'],
            'scope':'Actual source-derived Q/K and changed live allocation are retained even if finite/input differences occur. No assertion of native attention/V policy parity.'},
        'observations':len(observations),'workByPhase':{'actor':execution['nominalMatrixMACs'],'postSealAnalysis':0},
        'totalNewMatrixMACs':execution['nominalMatrixMACs'],
        'distinctBF16ParameterBytes':sum(x['bytes'] for x in execution['loadedParameters']),
        'primaryCandidate':'Source-state Pquery/QhybridK→liveallocation→deferredV→ownnativeMLP/nextquery',
        'scope':'One source pair/four demanded queries. Actor numerical inputs are original source states/context and13learned tensors; no archived selected operands/Q/K/V/duals/Pnorm/oldoutputs. Native prefixes/clockcontext/maps remain paid. No new fullV/referencecampaign/fit/extraconsumer/suffix/image or arbitrary-prompt zero-neural completion.'}
