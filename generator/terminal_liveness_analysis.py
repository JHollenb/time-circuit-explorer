"""Saved-only target comparison after the reduced producer seals and exits."""
import json
from pathlib import Path
import shutil
import sys
import time
from encoder_atoms import read,save,sha,np
from paired_fields import load,retain,metrics


def execute(root):
    root=Path(root).resolve();out=root/'response-result';tick=time.perf_counter();r=read(root/'request.json')
    seal=read(out/'candidate-seal.json');seal_sha=sha(out/'candidate-seal.json');candidate=read(out/'candidate.json');execution=read(out/'execution.json');targets=read(root/'targets.json')
    if seal!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')}:raise ValueError('Candidate release changed')
    incoming=load(targets['sourceCombined'])[:,512:];scores={};differences={};observations=0;rows_better=rows_worse=0
    with (out/'observations.jsonl').open('x') as stream:
        for name in ['image','velocity']:
            a=load(candidate['fields'][name]);b=load(targets['fields'][name]);error=a-b
            scores[name]={'endpoint':metrics(a,b),'payloadExact':candidate['fields'][name]['payloadSHA256']==targets['fields'][name]['payloadSHA256'],
                          'signedMeanError':float(error.mean()),'minError':float(error.min()),'maxError':float(error.max())}
            if name=='image':
                scores[name]['nativeWrite']=metrics(a-incoming,b-incoming);scores[name]['holdWrite']=metrics(np.zeros_like(b),b-incoming)
            differences[name]=retain(root/'arrays/differences',name,error)
            for axis,label in [(1,'image-row'),(2,'channel')]:
                for index in range(a.shape[axis]):
                    av,bv=(a[:,index],b[:,index]) if axis==1 else (a[:,:,index],b[:,:,index])
                    row={'field':name,'panel':label,'index':index,**metrics(av,bv),'signedMeanError':float((av-bv).mean())}
                    if name=='image' and axis==1:
                        iv=incoming[:,index];row['nativeWrite']=metrics(av-iv,bv-iv);row['holdWrite']=metrics(np.zeros_like(iv),bv-iv)
                        rows_better+=row['nativeWrite']['rms']<row['holdWrite']['rms'];rows_worse+=row['nativeWrite']['rms']>row['holdWrite']['rms']
                    stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');observations+=1
    scores['image']['rowsVersusHold']={'improve':int(rows_better),'worsen':int(rows_worse),'equal':1024-int(rows_better)-int(rows_worse)}
    if sha(out/'candidate-seal.json')!=seal_sha or sha(out/'candidate.json')!=seal['candidateSHA256'] or sha(out/'execution.json')!=seal['executionSHA256']:raise ValueError('Candidate changed during target comparison')
    save(out/'differences.json',differences)
    save(out/'report.json',{'schema':'tce-terminal-liveness-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal_sha,'scores':scores,
        'observations':observations,'newMatrixMACs':execution['totalMatrixMACs'],'workByPhase':{'candidate':execution['totalMatrixMACs'],'postSealAnalysis':0},
        'referenceBlockAndHeadMatrixMACs':203358732288,'avoidedAffineMACs':53150220288,'avoidedAttentionMACs':4831838208,'avoidedMatrixMACs':57982058496,
        'newNativeReferenceCalls':0,'newOriginalHeadCalls':1,'newReconstructedBlockCalls':1,'newImages':0,'newEncoderCalls':0,'newVAECalls':0,'originalParameterBytes':execution['originalParameterBytes'],
        'analysisSeconds':time.perf_counter()-tick,'mechanics_status':'complete','instrument_status':'complete-state-and-velocity-comparison',
        'trend_status':'measured finite-arithmetic differences retained; no effect-size gate','terminal_status':'unassessed',
        'scope':'One known native terminalsource. Physical dead textQ/MLP/output elimination with allcurrentKV and ownimage consumed by originalhead. Originallearned maps, sourceprefix/context/capture and failedpredecessor remainpaid; no independentfullmodel parity/image/latencygain/arbitrary-promptzero-neural claim.'})
    for name in ['request.json','inputs.json','targets.json','parameter-artifacts.json','resource-basis.json','frozen-plan.md','execution-intake.md']:
        shutil.copyfile(root/name,out/name)
    print('TERMINAL_LIVENESS_COMPARED',flush=True)


if __name__=='__main__':execute(sys.argv[1])
