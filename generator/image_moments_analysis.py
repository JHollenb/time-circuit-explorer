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
    partitions=[]
    with (out/'partition-observations.jsonl').open('x') as stream:
        for block in e['blockRecords']:
            if block['alphaField'] is None:continue
            a=load(c['fields'][block['alphaField']])
            if a.shape!=(1,24,512):raise ValueError('Saved partition mass shape differs')
            for head in range(24):
                values=a[0,head];row={'step':block['step'],'kind':block['kind'],'index':block['index'],'head':head,'panel':'own-text-attention-mass',
                    'mean':float(values.mean()),'min':float(values.min()),'max':float(values.max()),'p05':float(np.quantile(values,.05)),'median':float(np.median(values)),'p95':float(np.quantile(values,.95))}
                partitions.append(row);stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    save(out/'partition-summary.json',{'headRows':len(partitions),'scope':'Descriptive own current text partition mass; not a native mass comparison, quality gate or causal attribution.'})
    moment_rows=[]
    with (out/'moment-observations.jsonl').open('x') as stream:
        for block in e['blockRecords']:
            if not block['moments']:continue
            values={n:load(c['fields'][item['field']]) for n,item in block['moments'].items()}
            for head in range(24):
                row={'step':block['step'],'kind':block['kind'],'index':block['index'],'head':head,'panel':'own-key-moments',
                    'meanNorm':float(np.linalg.norm(values['mean'][0,head])),'covarianceTrace':float(np.trace(values['covariance'][0,head])),
                    'covarianceFrobenius':float(np.linalg.norm(values['covariance'][0,head]))}
                for name in ['logZI','logZT','quadratic']:
                    x=values[name][0,head];row[name]={'mean':float(x.mean()),'min':float(x.min()),'max':float(x.max())}
                moment_rows.append(row);stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    save(out/'moment-summary.json',{'headRows':len(moment_rows),'scope':'Own current post-RoPE full population key moments and approximate logZI; no exact finite-bank/native partition error or quality gate.'})
    if sha(s['priorTrajectoryReportPath'])!=s['priorTrajectoryReportSHA256']:raise ValueError('Prior report changed')
    shutil.copyfile(s['priorTrajectoryReportPath'],out/'prior-report.json')
    save(out/'differences.json',differences)
    save(out/'report.json',{'schema':'tce-image-partition-moments-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':sealed_sha,'scores':scores,'trajectory':curve,
        'observations':count,'work':{'affineMACs':e['affineMACs'],'attentionMACs':e['attentionMACs'],'convolutionMACs':e['convolutionMACs'],
            'modelMatrixAndConvolutionMACs':e['affineMACs']+e['attentionMACs']+e['convolutionMACs'],'avoidedMatrixMACs':483032825856,'explicitTextMACs':e['explicitTextMACs'],'nativeSDPAMACs':e['nativeSDPAMACs'],
            'newNativeReferences':0,'newEncoderCalls':0,'newNoiseDraws':0,'ownDenoisers':4,'ownSchedulerCommits':4,'originalVAEDecodes':1,'newCandidateImages':1},
        'analysisSeconds':time.perf_counter()-started,'mechanics_status':'complete','instrument_status':'complete-own-trajectory-native-comparison',
        'trend_status':'Full signed finite trajectory and pixel discrepancy retained; no effect-size gate','terminal_status':'unassessed',
        'scope':'One known input/seed from saved nativeE/noise. Own live coupled text/image states, full post-RoPE image-key population moments approximate logZI; TT scores/logsumexp/conditional AV times own alpha before BF16output; native image attention, finaltextwrite omitted. Originalmaps/E/noise/decoder paid; no shadow/reference/fit/sweep/cheapscalar/zero-neural claim. Common-incoming-QKV realidentity doesnotdecompose diverged trajectories.'})
    for name in ['request.json','inputs.json','targets.json','intake.json','resource-basis.json','frozen-plan.md']:
        shutil.copyfile(root/name,out/name)
    print('IMAGE_MOMENTS_COMPARED',flush=True)


if __name__=='__main__':execute(sys.argv[1])
