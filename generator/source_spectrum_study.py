"""Stage and execute a source-only SVD through one guarded CPU analysis lease."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from encoder_atoms import np,read,save,sha,identity,guard
from paired_fields import load,retain
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['source_spectrum.py','source_spectrum_study.py','source_response.py','encoder_atoms.py','paired_fields.py','flux_clock_study.py']


def stage(root):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    from mdb.job_support import one_lease_scheduling_contract
    parent=HERE.parent/'artifacts/source-response-v2';c,j,old=checked_collection(parent,current_workers=False);api=Api();live=api.json('GET','/api/jobs/'+j['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody']:raise ValueError('Response parent differs')
    p0=parent/'result';fit=read(p0/'fit.json');served=read(p0/'serving.json');s=read(p0/'source.json');report=read(p0/'report.json');base=j['result']['diagnostics']['cwd']
    if report['candidateSealSHA256']!=sha(p0/'candidate-seal.json') or read(parent/'retained-array-verification-v2.json')['candidateSealSHA256']!=report['candidateSealSHA256']:raise ValueError('Parent release/custody differs')
    pairs=[]
    for pair in fit['discoveryPairs']:
        row=next(x for x in served['rows'] if x['id']==pair['id'] and x['method']=='response-only')
        if pair['X']['payloadSHA256']!=row['fields']['X']['payloadSHA256'] or pair['mask']['payloadSHA256']!=row['fields']['mask']['payloadSHA256']:raise ValueError('Fit/serving source operands differ')
        pairs.append({**pair,'oldPrediction':row['prediction']})
    root=Path(root).resolve();root.mkdir(exist_ok=False,parents=True);p=root/'payload';p.mkdir()
    for n in OWNED:
        if n in old['ownedSources'] and sha(HERE/n)!=old['sourceHashes'][n]:raise ValueError('Qualified helper changed')
        shutil.copyfile(HERE/n,p/n)
    shutil.copyfile(HERE.parent/'docs/plans/source-response-spectrum.md',p/'frozen-plan.md')
    canary=next(x for x in served['rows'] if x['population']=='evaluation')['prediction']['path']
    source={'pairs':pairs,'scale':fit['sourceScale'],'ridge':fit['ridge'],'expectedGeometry':[1888,3072,128],
        'protectedRoots':[base,*s['protectedRoots']], 'deniedCanaries':[canary,read(p0/'targets.json')[next(x['id'] for x in s['pairs'] if x['population']=='evaluation')][1]['path']],
        'parentRecords':{n:{'path':base+'/response-result/'+n,'sha256':sha(p0/n)} for n in ['fit.json','serving.json','report.json','coefficient-seal.json','candidate-seal.json']}}
    save(p/'source.json',source)
    contract=one_lease_scheduling_contract(model_free=True,model_loads=0,logical_operations=1,logical_branches=1,logical_checkpoint_replays=0,extra={'task_family':'analysis'})
    r={'schema':'tce-source-spectrum-request-v1','parentJobID':j['job_id'],'parentRequestSHA256':old['requestSHA256'],'ownedSources':OWNED,
        'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'scheduling_contract':contract,'needs':{'cpu':True},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':300,
        'protocol':'One thin SVD of sealed discoveryX/Y; lambda .01/scales fixed. Reconstruct existing ridge source prediction, continuous mode/target/residual/coefficient amplification; full source vectors retained. No new predictor,heldlabels,rank/gain/regularization sweep,model/reader/VAE/image/noise. No neural frameworks.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same job')
    cmd=command();cmd=cmd[:cmd.index('--with')]+['--with','numpy==2.5.4','--with','psutil>=5.9']
    i=cmd.index('uv');cmd[i:i]=['PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    cmd+=['python','-u','source_spectrum_study.py','execute','.']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-source-spectrum-v1',task='analysis',model=None,payload=root/'payload',pin='beast',needs=r['needs'],gpu=False,
        mrun_config={k:r[k] for k in ['schema','requestSHA256','backend','device','cpuThreads','scheduling_contract']},timeout_s=r['timeoutSeconds'],detach=True,retry_on_kill=False,preflight=True,
        note='CPU1 fixed sourceX1888x3072/Y1888x128,one thinFP64SVD,lambda.01. Reconstruct existing ridge prediction;retain fullspectrum/targetenergy/attenuation/coeffamplification/U/V/sourcevectors. No rank/gain/lambda sweep,new predictor,heldlabels,neuralframework/model/H0/VAE/images. Intent-first resources;priorcompatible CPUjob833.9MBRSS,0VRAM. Expected<2min/300scap,noretries;beastlocal arrays.')
    result={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',result);return result


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'source.json');guard()
    from source_response import quarantine
    from source_spectrum import analyse
    fields=[s['scale'],*[pair[n] for pair in s['pairs'] for n in ['X','Y','mask','oldPrediction']]]
    q=quarantine(root/'source.json',s,[str(root/'request.json'),str(root/'frozen-plan.md'),*[x['path'] for x in fields]],[root/'arrays',root/'response-result'])
    tick=time.perf_counter();cpu=time.process_time();out=root/'response-result';out.mkdir();scale=load(s['scale'])
    if scale.shape!=(3328,) or not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Inherited feature normalization differs')
    xs=[];ys=[];preds=[]
    for pair in s['pairs']:
        selected=load(pair['mask']).astype(bool)
        if selected.shape!=(1024,) or selected.sum()!=944:raise ValueError('Frozen discovery mask differs')
        xs.append(load(pair['X']));ys.append(load(pair['Y']));preds.append(load(pair['oldPrediction']))
    x,y,p=np.concatenate(xs),np.concatenate(ys),np.concatenate(preds)
    if [len(x),x.shape[1],y.shape[1]]!=s['expectedGeometry'] or s['ridge']!=.01:raise ValueError('Frozen source dimensions/objective differ')
    arrays,modes,summary=analyse(x,y,p,s['ridge']);retained={n:retain(root/'arrays',n,v) for n,v in arrays.items()}
    save(out/'fields.json',retained)
    with (out/'modes.jsonl').open('x') as stream:
        for row in modes:stream.write(json.dumps(row,allow_nan=False)+'\n')
    save(out/'analysis-seal.json',{'schema':'tce-source-spectrum-seal-v1','requestSHA256':r['requestSHA256'],'fields':retained,'modesSHA256':sha(out/'modes.jsonl')})
    for n in ['request.json','source.json','frozen-plan.md']:shutil.copyfile(root/n,out/n)
    save(out/'report.json',{'schema':'tce-source-spectrum-result-v1','requestSHA256':r['requestSHA256'],'analysisSealSHA256':sha(out/'analysis-seal.json'),'summary':summary,
        'wallSecondsIncludingSourceRetention':time.perf_counter()-tick,'cpuSecondsIncludingSourceRetention':time.process_time()-cpu,
        'numpyVersion':np.__version__,'pythonVersion':sys.version,'frameworksLoaded':[],'fileQuarantine':q,
        'work':{'SVDs':1,'newPredictors':0,'newLearnedFits':0,'originalModelEvaluations':0,'newH0Computations':0,'newImages':0},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limits':['Only fixed discovery X/Y and sealed original source predictions; no new predictor or feedback/evaluation access.',
            'Thin U is square here. Its near-zero complement is not evidence of informative source span; numerical-rank ambiguity remains explicit.',
            'Old learned H0/native label production and fitted response products remain paid. Reconstruction is a target-assisted analysis, not arbitrary-prompt zero-neural serving.',
            'No modes, gains or regularization selected; SVD/dual-solve finite differences retained as numerical observations.']})
    print('SOURCE_SPECTRUM_DONE',sha(out/'analysis-seal.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(json.dumps(v))
