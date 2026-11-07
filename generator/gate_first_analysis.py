"""Compare sealed terminal/head output to existing parent; selection audit after exit."""
import json
from pathlib import Path
import shutil
import sys
import time
from encoder_atoms import read,save,sha,np
from paired_fields import load,retain,metrics


def execute(root):
    root=Path(root).resolve();out=root/'response-result';tick=time.perf_counter();r=read(root/'request.json');s=read(root/'inputs.json');t=read(root/'targets.json')
    seal=read(out/'candidate-seal.json');seal_sha=sha(out/'candidate-seal.json');c=read(out/'candidate.json');e=read(out/'execution.json')
    if seal!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')}:raise ValueError('Candidate release changed')
    incoming=load(t['sourceCombined'])[:,512:];scores={};differences={};observations=0;better=worse=0
    with (out/'observations.jsonl').open('x') as stream:
        for name,field in [('image','image_ownImage'),('velocity','head_ownVelocity')]:
            a=load(c['fields'][field]);b=load(t['fields'][name]);error=a-b
            scores[name]={'endpoint':metrics(a,b),'payloadExact':c['fields'][field]['payloadSHA256']==t['fields'][name]['payloadSHA256'],
                'signedMeanError':float(error.mean()),'minError':float(error.min()),'maxError':float(error.max())}
            if name=='image':scores[name].update(parentWrite=metrics(a-incoming,b-incoming),holdWrite=metrics(np.zeros_like(b),b-incoming))
            differences[name]=retain(root/'arrays/differences',name,error)
            for axis,label in [(1,'image-row'),(2,'channel')]:
                for index in range(a.shape[axis]):
                    av,bv=(a[:,index],b[:,index]) if axis==1 else (a[:,:,index],b[:,:,index])
                    row={'field':name,'panel':label,'index':index,**metrics(av,bv),'signedMeanError':float((av-bv).mean())}
                    if name=='image' and axis==1:
                        iv=incoming[:,index];row['parentWrite']=metrics(av-iv,bv-iv);row['holdWrite']=metrics(np.zeros_like(iv),bv-iv)
                        better+=row['parentWrite']['rms']<row['holdWrite']['rms'];worse+=row['parentWrite']['rms']>row['holdWrite']['rms']
                    stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');observations+=1
    scores['image']['rowsVersusHold']={'improve':int(better),'worsen':int(worse),'equal':1024-int(better)-int(worse)}
    channels=load(c['fields']['selection_score']);ranking=load(c['fields']['selection_rankedIndices']).astype(np.int64);support=load(c['fields']['selection_support']).astype(np.int64)
    expected=np.lexsort((np.arange(9216),-channels))
    if not np.array_equal(ranking,expected) or not np.array_equal(support,np.sort(expected[:2304])):raise ValueError('Stable source-only rank/support mechanics differ')
    up=load(c['fields']['sourceMaps_upNorm']);down=load(c['fields']['sourceMaps_downNorm']);energy=load(c['fields']['selection_weightedGateEnergy'])
    selected=np.zeros(9216,dtype=bool);selected[support]=True;rank_positions=np.empty(9216,dtype=np.int64);rank_positions[ranking]=np.arange(9216)
    with (out/'selection-observations.jsonl').open('x') as stream:
        for index in range(9216):
            stream.write(json.dumps({'panel':'source-only-gate-selection','channel':index,'rank':int(rank_positions[index]),'selected':bool(selected[index]),'scoreFP32':float(channels[index]),
                'upNormFP32':float(up[index]),'downNormFP32':float(down[index]),'weightedGateEnergyFP32':float(energy[index])},sort_keys=True,allow_nan=False)+'\n')
    omitted=float(load(c['fields']['selection_omittedScoreSum']));kept=float(load(c['fields']['selection_selectedScoreSum']));mod=load(s['sourceFields']['mod']);amplification=float(np.max(np.abs(mod[...,6144:])))
    selection={'selectedChannels':len(support),'totalChannels':9216,'stableRankAndAscendingSupportVerified':True,'support':support.tolist(),'scoreMin':float(channels.min()),'scoreMedian':float(np.median(channels)),'scoreMax':float(channels.max()),
        'omittedScoreSumFP32':omitted,'selectedScoreSumFP32':kept,'doubleSumOfRoundedOmittedScores':float(channels[~selected].sum()),'residualGateMaxAbs':amplification,'residualGateTimesOmittedEstimate':amplification*omitted,
        'scope':'Exact ordering of admitted FP32 scores verified after seal. Rounded score sums are diagnostic estimates, not certified upper bounds or quality gates. Real bound ignores signed cancellation; no full U used to select or analyze omitted channels.'}
    save(out/'selection-summary.json',selection)
    if sha(out/'candidate-seal.json')!=seal_sha or sha(out/'candidate.json')!=seal['candidateSHA256'] or sha(out/'execution.json')!=seal['executionSHA256']:raise ValueError('Candidate changed after target release')
    save(out/'differences.json',differences)
    save(out/'report.json',{'schema':'tce-gate-first-terminal-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal_sha,'scores':scores,'selection':selection,'observations':observations,'selectionObservations':9216,
        'newMatrixMACs':e['totalMatrixMACs'],'affineMACs':e['affineMACs'],'attentionMACs':e['attentionMACs'],'nonMatrixWork':e['nonMatrixWork'],
        'matchedParentTerminalAndHeadMACs':145376673792,'avoidedMatrixMACs':43486543872,'newNativeReferenceCalls':0,'newOriginalHeadCalls':1,'newReducedTerminalCalls':1,'newImages':0,'newEncoderCalls':0,'newVAECalls':0,'newPrefixCalls':0,
        'originalParameterBytes':e['originalParameterBytes'],'analysisSeconds':time.perf_counter()-tick,'mechanics_status':'complete','instrument_status':'complete-terminal-head-and-selection-comparison','trend_status':'Full signed finite discrepancy retained, no effect-size or bound-quality gate','terminal_status':'unassessed',
        'scope':'One paid clock3 terminal source from70 own trajectory. Fullcurrentgate plus fixedquarter selectedU/down, completeimageattention and actualownhead. Parent70 terminalImage/velocity onlyafterseal+exit; no new image/reference/fit/sweep. Originalmaps/prefixpaid; changedGEMMgeometry and omittedchannels combined. Not wholetrajectory, native equivalence or arbitrary-promptzero-neural generation.'})
    for name in ['request.json','inputs.json','targets.json','resource-basis.json','frozen-plan.md']:shutil.copyfile(root/name,out/name)
    print('GATE_FIRST_COMPARED',flush=True)


if __name__=='__main__':execute(sys.argv[1])
