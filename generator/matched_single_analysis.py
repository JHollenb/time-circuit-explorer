"""Both actors have exited; analyze signed common-product omission, then old exits."""
from pathlib import Path
import json
import shutil
import sys
import time
from encoder_atoms import read,save,sha,identity,np
from paired_fields import load,retain,metrics,payload

ENERGIES=['omittedDiagonalEnergy','omittedWriteEnergy','omittedCrossChannelEnergy','selectedDiagonalEnergy','selectedWriteEnergy','selectedCrossChannelEnergy','selectedOmittedCrossEnergy','fullWriteEnergy']


def rows(value):
    if value.ndim>=3 and value.shape[0]==1:return value[0].reshape(value.shape[1],-1)
    return value.reshape(-1,1) if value.ndim==1 else value.reshape(1,-1)


def execute(root):
    import torch
    import diffusers
    from matched_single import common_write,energy_panel
    from matched_single_ledger import Ledger
    from complete_masked_study import profile
    from flux_clock_study import verify,source_checks,assets_unchanged
    from flux_clock import header,snapshot
    from projected_complement import load_selected
    from matched_single_worker import PARAMETERS
    root=Path(root).resolve();out=root/'response-result';r=read(root/'request.json');s=read(root/'inputs.json');verify(root,r);started=time.perf_counter();cpu=time.process_time()
    c=read(out/'candidate.json');e=read(out/'execution.json');seal=read(out/'candidate-seal.json');seal_sha=sha(out/'candidate-seal.json')
    if seal!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json'),'sparseSealSHA256':sha(out/'seal-sparse.json'),'denseSealSHA256':sha(out/'seal-dense.json')}:raise ValueError('Both-actor release differs')
    for mode in ['sparse','dense']:
        if read(out/('seal-'+mode+'.json'))!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/('replay-'+mode+'.json')),'executionSHA256':sha(out/('execution-'+mode+'.json'))}:raise ValueError('Local release differs')
    numeric=profile(torch,4)
    if numeric!=s['numericalProfile']:raise ValueError('Diagnostic numerical profile differs')
    torch.cuda.reset_peak_memory_stats();model,assets=source_checks(s,diffusers);entries,hr=header(model/'transformer/diffusion_pytorch_model.safetensors')
    weights,loaded=load_selected(model/'transformer/diffusion_pytorch_model.safetensors',[PARAMETERS[1]],entries)
    if loaded!=[s['originalParameters'][1]]:raise ValueError('Original down source differs')
    down=weights[PARAMETERS[1]][:,3072:];manifest=e['replays']['sparse']['parameterManifestSHA256']
    ledger=Ledger(r['requestSHA256'],manifest);products=[]
    def product(name,a,b):
        if a.dtype!=torch.float64 or b.dtype!=torch.float64 or a.device.type!='cuda' or b.device.type!='cuda' or torch.is_autocast_enabled('cuda') or torch.backends.cuda.matmul.allow_tf32:raise ValueError('FP64 diagnostic contraction policy differs')
        value=torch.matmul(a,b);products.append({'operator':name,'leftShape':list(a.shape),'rightShape':list(b.shape),'outputShape':list(value.shape),
            'dtype':str(value.dtype),'device':value.device.type,'nominalMACs':value.numel()*a.shape[-1]});return value
    inputs={'admittedProduct':load(c['fields']['dense_product'],'cuda'),'originalDown':down,'ownSupport':load(c['fields']['sparse_selection_support'],'cuda'),
        'sparseExit':load(c['fields']['sparse_exit'],'cuda'),'denseExit':load(c['fields']['dense_exit'],'cuda'),'gate':load(c['fields']['dense_gate'],'cuda')}
    source_hashes={n:payload(v) for n,v in inputs.items()}
    def operation(v):
        p=v['admittedProduct'].double();d=v['originalDown'].double();g=v['gate'].double();support=v['ownSupport']
        writes=common_write(p,d,support,product);energies,channels=energy_panel(p,d,support,writes['fullWrite'],writes['selectedWrite'],g)
        delta=v['sparseExit'].double()-v['denseExit'].double();gated=g*writes['omittedWrite'];remainder=delta+gated
        return {**writes,'channelMagnitude':torch.stack([channels[n] for n in ['all','text','image']]),
            'energyTable':torch.tensor([[row[n] for n in ENERGIES] for row in energies],dtype=torch.float64,device=p.device),
            'deltaExit':delta,'gatedOmission':gated,'finiteRemainder':remainder,'accountClosure':delta-(-gated+remainder)}
    outputs={n:{'shape':[1,1536,3072],'dtype':'torch.float64'} for n in ['fullWrite','selectedWrite','omittedWrite','deltaExit','gatedOmission','finiteRemainder','accountClosure']}
    outputs.update(channelMagnitude={'shape':[3,9216],'dtype':'torch.float64'},energyTable={'shape':[6,8],'dtype':'torch.float64'})
    with torch.inference_mode():
        got=ledger.run('anatomy',inputs,operation,outputs,{'candidateSealSHA256':seal_sha,'sparseProducerFrameID':e['replays']['sparse']['typedActs'][0]['outputFrame']['frame_id'],
            'denseProducerFrameID':e['replays']['dense']['typedActs'][0]['outputFrame']['frame_id'],'policy':'FP64 common admitted dense product and original down; no alternate support or new model consumer.'})
        anatomy={n:retain(root/'arrays/anatomy',n,value) for n,value in got.items()}
    if snapshot(torch)!=numeric or source_hashes!={n:payload(v) for n,v in inputs.items()}:raise ValueError('Diagnostic source/numeric closure differs')
    if len(products)!=2 or sum(x['nominalMACs'] for x in products)!=54358179840:raise ValueError('Diagnostic matrix census differs')
    assets_unchanged(assets)
    save(out/'anatomy.json',{'fields':anatomy,'energyColumns':ENERGIES,'energyRows':[{'stratum':name,'gated':gated} for name in ['all','text','image'] for gated in [False,True]],'candidateSealSHA256':seal_sha})
    save(out/'anatomy-execution.json',{'products':products,'typedActs':ledger.records,'timings':ledger.timings,'loadedParameters':loaded,'originalDownPayloadSHA256':source_hashes['originalDown'],
        'originalDownShape':list(down.shape),'parameterManifestSHA256':manifest,'numericalProfile':numeric,'candidateSealSHA256':seal_sha,'sourceInputClosure':True,'header':hr,'assets':assets,
        'wallSecondsAfterImportsBeforeComparisons':time.perf_counter()-started,'CPUSecondsAfterImportsBeforeComparisons':time.process_time()-cpu,
        'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved()})
    # Release numerical diagnostic residency before CPU descriptions and archived comparisons.
    del got,inputs,down,weights;torch.cuda.empty_cache()
    comparisons={};differences={};observations=0;t=read(root/'targets.json')
    with (out/'observations.jsonl').open('x') as stream:
        def compare(name,a,b,payload_exact=None):
            nonlocal observations
            error=a-b;comparisons[name]={'metrics':metrics(a,b),'numericExact':bool(np.array_equal(a,b)),'payloadExact':payload_exact,'signedMeanError':float(error.mean()),'minError':float(error.min()),'maxError':float(error.max())}
            ar,br=rows(a),rows(b)
            for index in range(len(ar)):
                stream.write(json.dumps({'comparison':name,'row':index,**metrics(ar[index],br[index]),'signedMeanError':float((ar[index]-br[index]).mean())},sort_keys=True,allow_nan=False)+'\n');observations+=1
            return error
        for mode,targets in t['fields'].items():
            for name,target in targets.items():
                field=c['fields'][mode+'_'+name];error=compare('archive_'+mode+'_'+name,load(field),load(target),field['payloadSHA256']==target['payloadSHA256'])
                if name in ['textOut','imageOut']:differences['archive_'+mode+'_'+name]=retain(root/'arrays/differences','archive_'+mode+'_'+name,error)
        support=load(c['fields']['sparse_selection_support']).astype(np.int64)
        common=['normalized','queryRaw','keyRaw','valueRaw','gateRaw','activation','attentionQuery','attentionKey','attentionValue','imageAttention','attentionRead','preResidualUpdate','textOut','imageOut','exit','alpha','textRead','gate']
        for name in common:
            a_item,b_item=c['fields']['sparse_'+name],c['fields']['dense_'+name]
            a,b=load(a_item),load(b_item);error=compare('operand_'+name,a,b,a_item['payloadSHA256']==b_item['payloadSHA256'])
            if name in ['exit','preResidualUpdate']:differences['operand_'+name]=retain(root/'arrays/differences','operand_'+name,error)
        for name in ['upRaw','product']:
            a,b=load(c['fields']['sparse_'+name]),load(c['fields']['dense_'+name])[...,support];compare('operand_'+name,a,b)
    magnitude=load(anatomy['channelMagnitude']);scores=load(c['fields']['sparse_selection_score']);ranking=load(c['fields']['sparse_selection_rankedIndices']).astype(np.int64)
    order=np.lexsort((np.arange(9216),-scores))
    if not np.array_equal(ranking,order) or not np.array_equal(support,np.sort(order[:2304])):raise ValueError('Own stable score/support ordering differs')
    selected=np.zeros(9216,dtype=bool);selected[support]=True
    def correlation(a,b):
        ac,bc=a-a.mean(),b-b.mean();den=float(np.linalg.norm(ac)*np.linalg.norm(bc));return float(ac@bc/den) if den else None
    def average_ranks(x):
        order=np.argsort(x,kind='stable');ranks=np.empty(len(x),dtype=np.float64);start=0
        while start<len(x):
            end=start+1
            while end<len(x) and x[order[end]]==x[order[start]]:end+=1
            ranks[order[start:end]]=(start+end-1)/2;start=end
        return ranks
    alignment={}
    with (out/'channel-observations.jsonl').open('x') as stream:
        for stratum_index,stratum in enumerate(['all','text','image']):
            values=magnitude[stratum_index];ratios=np.divide(values,scores,out=np.zeros_like(values),where=scores!=0)
            alignment[stratum]={'scoreMagnitudePearson':correlation(scores,values),'scoreMagnitudeSpearman':correlation(average_ranks(scores),average_ranks(values)),
                'selectedMagnitudeSum':float(values[selected].sum()),'omittedMagnitudeSum':float(values[~selected].sum()),'zeroScoreChannels':int(np.count_nonzero(scores==0)),
                'admittedMagnitudeOverRoundedScore':{'min':float(ratios.min()),'median':float(np.median(ratios)),'max':float(ratios.max()),'scope':'Zero-score rows use ratio0 and are separately counted. Not a certified real bound.'}}
            for channel in range(9216):stream.write(json.dumps({'stratum':stratum,'channel':channel,'selected':bool(selected[channel]),'score':float(scores[channel]),'admittedMagnitude':float(values[channel]),'admittedMagnitudeOverScore':float(ratios[channel])},sort_keys=True,allow_nan=False)+'\n')
    energies=load(anatomy['energyTable']);energy_rows=[{**row,**dict(zip(ENERGIES,map(float,energies[index])))} for index,row in enumerate(read(out/'anatomy.json')['energyRows'])]
    delta=load(anatomy['deltaExit']);omitted=load(anatomy['gatedOmission']);remainder=load(anatomy['finiteRemainder']);closure=load(anatomy['accountClosure']);incoming=np.concatenate([load(s['sourceFields']['text']),load(s['sourceFields']['image'])],1);dense_exit=load(c['fields']['dense_exit'])
    decomposition={}
    with (out/'anatomy-observations.jsonl').open('x') as stream:
        for name,part in [('all',slice(None)),('text',slice(0,512)),('image',slice(512,None))]:
            de,om,re,wr=delta[:,part],omitted[:,part],remainder[:,part],(dense_exit-incoming)[:,part]
            de_norm=float(np.linalg.norm(de));re_norm=float(np.linalg.norm(re));om_norm=float(np.linalg.norm(om));write_norm=float(np.linalg.norm(wr))
            decomposition[name]={'sparseWriteVsDenseWrite':metrics(wr+de,wr),'actualDeltaRelativeDenseWrite':de_norm/write_norm if write_norm else None,'negativeOmissionVsActualDelta':metrics(-om,de),
                'finiteRemainderRelativeRMS':re_norm/de_norm if de_norm else None,'gatedOmissionRelativeRMS':om_norm/de_norm if de_norm else None,
                'deltaEnergy':float(np.sum(de*de)),'gatedOmissionEnergy':float(np.sum(om*om)),'finiteRemainderEnergy':float(np.sum(re*re)),'negativeOmissionRemainderCrossEnergy':float(-2*np.sum(om*re)),
                'closureMaxAbs':float(np.max(np.abs(closure[:,part])))}
        for row in range(1536):
            de,om,re=delta[0,row],omitted[0,row],remainder[0,row]
            stream.write(json.dumps({'row':row,'stratum':'text' if row<512 else 'image','actualDeltaRMS':float(np.sqrt(np.mean(de*de))),
                'gatedOmissionRMS':float(np.sqrt(np.mean(om*om))),'finiteRemainderRMS':float(np.sqrt(np.mean(re*re))),'omissionDeltaCosine':metrics(-om,de)['cosine'],
                'negativeOmissionRemainderCrossEnergy':float(-2*np.sum(om*re))},sort_keys=True,allow_nan=False)+'\n')
    if sha(out/'candidate-seal.json')!=seal_sha or sha(out/'candidate.json')!=seal['candidateSHA256'] or sha(out/'execution.json')!=seal['executionSHA256']:raise ValueError('Actors changed after target release')
    save(out/'differences.json',differences)
    total=sum(x['totalMatrixMACs'] for x in e['replays'].values())+sum(x['nominalMACs'] for x in products)
    if total!=r['prospectiveMatrixMACs']:raise ValueError('Whole diagnostic work account differs')
    save(out/'report.json',{'schema':'tce-matched-single-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal_sha,'comparisons':comparisons,'energyRows':energy_rows,'alignment':alignment,'decomposition':decomposition,
        'observations':observations,'channelObservations':27648,'anatomyRowObservations':1536,'totalMatrixMACs':total,'FP64DownMACs':54358179840,'newSparseReplays':1,'newDenseReplays':1,'newImages':0,'newPrefixCalls':0,'newVAECalls':0,'newHeadCalls':0,
        'analysisSecondsAfterImports':time.perf_counter()-started,'mechanics_status':'complete','instrument_status':'common-input-local-operator-anatomy','trend_status':'All signed omission/cancellation/finite-residual and replay rows retained; no effect-size gate','terminal_status':'unassessed',
        'scope':'One shared incoming clock0single0, exactfrozen72sparse/70densemoment operators, fixedquarter. Diagnostic denseadmittedP/down signedwrite, not a newserving producer or finalimagecausal attribution. Originalmaps/states/norms/historicalprefix paid; arbitrary-promptzero-neuralopen.'})
    for name in ['request.json','inputs.json','targets.json','resource-basis.json','frozen-plan.md']:shutil.copyfile(root/name,out/name)
    print('MATCHED_SINGLE_ANALYZED',flush=True)


if __name__=='__main__':execute(sys.argv[1])
