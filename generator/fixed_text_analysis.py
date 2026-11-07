"""Compare one sealed own-state image against its already-sealed native history."""
import json
from pathlib import Path
import shutil
import sys
import time
from encoder_atoms import read,save,sha,np
from paired_fields import load,retain,metrics


def rows(value,name):
    if value.ndim==3 and name!='pngRGB':return value[0]
    if name=='pngRGB':return value.reshape(value.shape[0],-1)
    if name=='rgbFloat':return value[0].reshape(value.shape[1],-1)
    if value.ndim==4:return value[0].transpose(1,0,2).reshape(value.shape[-2],-1)
    raise ValueError('Unknown scalar observation geometry')


def execute(root):
    from PIL import Image
    root=Path(root).resolve();out=root/'response-result';started=time.perf_counter();r=read(root/'request.json');s=read(root/'inputs.json');t=read(root/'targets.json')
    c=read(out/'candidate.json');e=read(out/'execution.json');seal=read(out/'candidate-seal.json');sealed_sha=sha(out/'candidate-seal.json')
    if seal!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json'),'imageSHA256':sha(c['image']['path'])}:raise ValueError('Complete candidate release differs')
    scores={};differences={};count=0
    with (out/'observations.jsonl').open('x') as stream:
        for name,reference in t['fields'].items():
            a,b=load(c['fields'][name]),load(reference);error=a-b
            scores[name]={'metrics':metrics(a,b),'payloadExact':c['fields'][name]['payloadSHA256']==reference['payloadSHA256'],
                'signedMeanError':float(error.mean()),'minError':float(error.min()),'maxError':float(error.max())}
            if not name.startswith('C'):differences[name]=retain(root/'arrays/differences',name,error)
            ar,br=rows(a,name),rows(b,name)
            for index in range(len(ar)):
                stream.write(json.dumps({'field':name,'panel':'row','index':index,**metrics(ar[index],br[index]),'signedMeanError':float((ar[index]-br[index]).mean())},sort_keys=True,allow_nan=False)+'\n');count+=1
    curve=[];old_native=load(s['noise']['packed']);old_candidate=old_native
    for k in range(4):
        a,b=load(c['fields']['latent'+str(k+1)]),load(t['fields']['latent'+str(k+1)])
        curve.append({'step':k,'latent':scores['latent'+str(k+1)]['metrics'],'velocity':scores['velocity'+str(k)]['metrics'],
            'commitWrite':metrics(a-old_candidate,b-old_native)})
        old_candidate,old_native=a,b
    for role,record,field in [('candidate',c['image'],c['fields']['pngRGB']),('native',t['image'],t['fields']['pngRGB'])]:
        if sha(record['path'])!=record['sha256'] or not np.array_equal(np.asarray(Image.open(record['path']).convert('RGB')).astype(np.float64),load(field)):
            raise ValueError('PNG file and actual pixel-array join differs')
        shutil.copyfile(record['path'],out/(role+'-A.png'))
    if sha(out/'candidate-seal.json')!=sealed_sha or sha(out/'candidate.json')!=seal['candidateSHA256'] or sha(out/'execution.json')!=seal['executionSHA256']:raise ValueError('Candidate changed after target release')
    save(out/'differences.json',differences)
    save(out/'report.json',{'schema':'tce-fixed-text-memory-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':sealed_sha,'scores':scores,'trajectory':curve,
        'observations':count,'work':{'affineMACs':e['affineMACs'],'attentionMACs':e['attentionMACs'],'convolutionMACs':e['convolutionMACs'],
            'modelMatrixAndConvolutionMACs':e['affineMACs']+e['attentionMACs']+e['convolutionMACs'],'avoidedTextPathMatrixMACs':5798205849600,'incrementalAvoidedMatrixMACsBeyondTerminal':5566277615616,
            'newNativeReferences':0,'newEncoderCalls':0,'newNoiseDraws':0,'ownDenoisers':4,'ownSchedulerCommits':4,'originalVAEDecodes':1,'newCandidateImages':1},
        'analysisSeconds':time.perf_counter()-started,'mechanics_status':'complete','instrument_status':'complete-own-trajectory-native-comparison',
        'trend_status':'Full signed finite trajectory and pixel discrepancy retained; no effect-size gate','terminal_status':'unassessed',
        'scope':'One known input/seed from saved nativeE/noise. Fixed C(E) at every joint/single block, physically omitted textQ/attentionout/MLP; all currentKV/image, learnedclock, head and VAEwork paid. No oracle intermediate refresh, native shadow/reference, fit/sweep, matchedspeedup or arbitrary-promptzero-neural claim.'})
    for name in ['request.json','inputs.json','targets.json','intake.json','resource-basis.json','frozen-plan.md']:
        shutil.copyfile(root/name,out/name)
    print('OWN_FIXED_TEXT_MEMORY_COMPARED',flush=True)


if __name__=='__main__':execute(sys.argv[1])
