"""Stage/run/collect the frozen, saved-only zero-network decoder-chart assay."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
CONTEXTS=['60666.I0','60666.I1','60667.I0','60667.I1']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def stage(root,source):
    source=Path(source).resolve();root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    transport=read(source.parent.parent/'report.json')
    if sha(source/'files.json')!=transport['files_manifest_sha256']:raise ValueError('Source inventory differs from retained transport report')
    files=read(source/'files.json')
    for name in ['rows.json','request.json']:
        if sha(source/name)!=files[name]:raise ValueError('Source metadata differs: '+name)
    rows=read(source/'rows.json');source_request=read(source/'request.json')
    calibration={c:[] for c in CONTEXTS};evaluation=[]
    for row in rows:
        context=f'{row["pair"]}.I{row["bit"]}';alpha=row['alpha'];name=row['png'];path=source/name
        if context not in CONTEXTS or alpha not in [0,.05,.1,.25,.5,1]:raise ValueError('Source curve differs')
        if sha(path)!=row['png_sha256'] or files[name]!=row['png_sha256']:raise ValueError('Source image digest differs')
        role='calibration' if alpha in [0,.25,1] else 'evaluation';dest=payload/role/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(path,dest)
        item={'path':role+'/'+name,'sha256':sha(path),'context':context,'alpha':alpha}
        if role=='calibration':calibration[context].append(item)
        else:evaluation.append(item)
    for values in calibration.values():values.sort(key=lambda x:x['alpha'])
    if len(evaluation)!=12 or any([x['alpha'] for x in a]!=[0,.25,1] for a in calibration.values()):raise ValueError('Calibration/held split differs')
    for name in ['response_chart.py','response_study.py']:shutil.copyfile(HERE/name,payload/name)
    save(payload/'compile.json',{'calibration':calibration})
    save(payload/'serve.json',{'alphas':[.05,.1,.5,.075]})
    plan={'schema':'tce-zero-network-chart-request-v1','primary':'piecewise','comparisons':['quadratic','copy','nearest','endpoint','transfer'],
        'calibration':calibration,'evaluation':evaluation,'novelAlpha':.075,'eyeBox':source_request['config']['observers']['windows']['eyes'],
        'backgroundBox':source_request['config']['observers']['windows']['background'],
        'source':{'job':'job-93f45da60821','requestSHA256':source_request['request_sha256'],'filesSHA256':sha(source/'files.json'),'transportSHA256':sha(source.parent.parent/'report.json'),'sourceArchiveSHA256':transport['archive_sha256'],'rowsSHA256':sha(source/'rows.json'),'checkpoint':source_request['config']['checkpoint_sha256'],
                  'historicalCost':{'loads':1,'vaeImages':24,'nativeSchedulerChecks':4,'newDenoiserCalls':0},'curves':source_request['config']['sources']},
        'scope':'Retrospective held-file isolation. Supplied Euler-defined native-state paths; compiled alpha-to-RGB, not raw prompt/seed generation. No new neural evaluation during compilation, serving or scoring.'}
    plan['sourceHashes']={p.name:sha(p) for p in payload.iterdir() if p.is_file()};plan['requestSHA256']=identity(plan)
    save(payload/'request.json',plan);save(root/'request.json',plan)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':plan['requestSHA256']}
def command():
    return ['/usr/bin/env','PYTHONUNBUFFERED=1','PYTHONNOUSERSITE=1','PYTHONPATH=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
            'UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache','UV_LINK_MODE=symlink','uv','run','--no-project','--offline','--python','3.14.6','--with','numpy>=2.0','--with','pillow>=10.0',
            'python','-u','response_study.py','execute','.']
def submit(root):
    from mrun.client.submit import launch
    root=Path(root).resolve();request=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    job=launch(command(),experiment='tce-zero-network-decoder-chart-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
               config={'schema':'tce-zero-network-assay-v1','requestSHA256':request['requestSHA256'],'backend':'numpy','device':'cpu','plannedNeuralForwards':0},
               timeout_s=240,detach=True,retry_on_kill=False,preflight=True,guarded=True,
               note='Saved-only decoder response chart:12 calibration/12 held PNGs,4 dependent curves. CPU1, zero neural/network/weights. Compile then fresh serving, seal before scoring. Finite240s, no retries; no GPU or model acquisition.')
    result={'jobID':str(job),'requestSHA256':request['requestSHA256']};save(root/'job.json',result);return result
def execute(root):
    root=Path(root).resolve();started=time.perf_counter();request=read(root/'request.json');out=root/'response-result';out.mkdir(exist_ok=False)
    if identity({k:v for k,v in request.items() if k!='requestSHA256'})!=request['requestSHA256']:raise ValueError('Request changed')
    for name,digest in request['sourceHashes'].items():
        if sha(root/name)!=digest:raise ValueError('Source changed: '+name)
    subprocess.run([sys.executable,str(root/'response_chart.py'),'compile','--request',str(root/'compile.json'),'--bundle',str(out/'chart.npz'),'--receipt',str(out/'compile-receipt.json')],check=True,cwd=root)
    subprocess.run([sys.executable,str(root/'response_chart.py'),'serve','--request',str(root/'serve.json'),'--bundle',str(out/'chart.npz'),'--output',str(out/'images'),'--receipt',str(out/'serve-receipt.json')],check=True,cwd=root)
    serving=read(out/'serve-receipt.json')
    seal={'schema':'tce-prediction-seal-v1','requestSHA256':request['requestSHA256'],'bundleSHA256':sha(out/'chart.npz'),
          'servingReceiptSHA256':sha(out/'serve-receipt.json'),'images':{r['path']:r['sha256'] for r in serving['rows']}}
    save(out/'prediction-seal.json',seal);seal_digest=sha(out/'prediction-seal.json')
    print('TCE_RESPONSE_SEALED '+seal_digest,flush=True)
    result=score(root,out,request,serving)
    if sha(out/'prediction-seal.json')!=seal_digest or sha(out/'serve-receipt.json')!=seal['servingReceiptSHA256']:raise ValueError('Seal changed during scoring')
    save(out/'report.json',{'schema':'tce-zero-network-response-v1','requestSHA256':request['requestSHA256'],'source':request['source'],'primary':'piecewise',
         'compile':read(out/'compile-receipt.json'),'serving':serving,'scores':result,'predictionSealSHA256':seal_digest,'workerWallSeconds':time.perf_counter()-started,
         'physicalWork':{'newEncoderForwards':0,'newDenoiserForwards':0,'newVAEForwards':0,'newNeuralStudentForwards':0,'checkpointBytesRead':0},
         'mechanics_status':'complete','terminal_status':'unassessed','scope':request['scope']})
    print('TCE_RESPONSE_COMPLETE',flush=True)
def score(root,out,request,serving):
    import numpy as np
    from PIL import Image
    def image(path):return np.asarray(Image.open(path).convert('RGB')).astype(np.float64)
    def rms(value):return float(np.sqrt(np.mean(value*value)))
    def region(value,box):x0,y0,x1,y1=box;return value[y0:y1,x0:x1]
    def compare(pred,target,scale):
        error=pred-target;den=rms(scale);a=rms(pred);b=rms(target)
        return {'mae':float(np.abs(error).mean()),'rms':rms(error),'predictedRms':a,'targetRms':b,'normalizerRms':den,'relativeRms':rms(error)/den if den else None,
                'cosine':float(np.mean(pred*target)/(a*b)) if a*b else None}
    base={c:image(root/items[0]['path']) for c,items in request['calibration'].items()};actual={};predicted={};rows=[]
    for item in request['evaluation']:
        path=root/item['path']
        if sha(path)!=item['sha256']:raise ValueError('Evaluation digest changed')
        actual[item['context'],item['alpha']]=image(path)
    for row in serving['rows']:
        path=out/'images'/row['path']
        if sha(path)!=row['sha256']:raise ValueError('Prediction changed after seal')
        pred=image(path);context,alpha=row['context'],row['alpha'];predicted[context,alpha,row['method']]=pred
        if (context,alpha) not in actual:continue
        target=actual[context,alpha];effect=target-base[context]
        record={k:row[k] for k in ['context','alpha','method','clippedFraction']}
        record.update(rgb=compare(pred,target,effect),eye=compare(region(pred,request['eyeBox']),region(target,request['eyeBox']),region(effect,request['eyeBox'])),
                      background=compare(region(pred,request['backgroundBox']),region(target,request['backgroundBox']),region(effect,request['backgroundBox'])))
        record['edit']=compare(pred-base[context],effect,effect);rows.append(record)
    pairs=[]
    for seed in ['60666','60667']:
        for alpha in [.05,.1,.5]:
            actual_effect=actual[seed+'.I1',alpha]-actual[seed+'.I0',alpha]
            for method in request['comparisons']+[request['primary']]:
                pred_effect=predicted[seed+'.I1',alpha,method]-predicted[seed+'.I0',alpha,method]
                pairs.append({'seed':seed,'alpha':alpha,'method':method,'full':compare(pred_effect,actual_effect,actual_effect),
                              'eye':compare(region(pred_effect,request['eyeBox']),region(actual_effect,request['eyeBox']),region(actual_effect,request['eyeBox']))})
    return {'rows':rows,'pairedEffects':pairs,'evaluationImages':12,'dependentCurves':4,'knownSeeds':2,'novelAccuracy':'unmeasured'}
def collect(root):
    from mrun.client.api import Api
    from run import inventory_code
    root=Path(root).resolve();job=Api().json('GET','/api/jobs/'+read(root/'job.json')['jobID']);result=job.get('result') or {};custody=job.get('payload_custody') or {}
    if job['state']!='succeeded' or result.get('returncode')!=0:raise ValueError('Job not successfully terminal: '+job['state'])
    if job.get('config',{}).get('requestSHA256')!=read(root/'request.json')['requestSHA256']:raise ValueError('Request binding differs')
    if not job.get('admission') or custody.get('required') is not True:raise ValueError('Guarded custody absent')
    phases=[custody.get(k) or {} for k in ['declared','sealed','executed']]
    if not phases[0].get('sha256') or any(any(p.get(k)!=phases[0].get(k) for k in ['sha256','size_bytes']) for p in phases) or phases[2].get('host')!='beast':raise ValueError('Payload custody differs')
    cwd=result['diagnostics']['cwd']
    if cwd!='/mnt/hdd8tb/mrun-work/'+job['job_id']:raise ValueError('Unexpected worker path')
    remote=cwd+'/response-result'
    code=inventory_code(remote).replace('size>8<<20 or total>32<<20','size>32<<20 or total>128<<20').replace('len(files)>50','len(files)>200')
    def inventory():return json.loads(subprocess.run(['ssh','-T','beast','python3','-'],input=code,text=True,capture_output=True,check=True,timeout=30).stdout)
    before=inventory();local=root/'result';local.mkdir(exist_ok=True)
    subprocess.run(['rsync','-a','--protect-args','beast:'+remote+'/',str(local)+'/'],check=True,capture_output=True,timeout=90)
    after=inventory();actual={str(p.relative_to(local)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in local.rglob('*') if p.is_file()}
    if before!=after or actual!=before['files']:raise ValueError('Remote/local result inventory differs')
    if read(local/'report.json')['requestSHA256']!=read(root/'request.json')['requestSHA256']:raise ValueError('Collected report request differs')
    save(root/'collection.json',{'jobID':job['job_id'],'custody':custody,'result':result,'inventory':before});save(root/'scheduler-job.json',job)
    return {'result':str(local),'jobID':job['job_id']}
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run',type=Path);p.add_argument('--source',type=Path);args=p.parse_args()
    result=stage(args.run,args.source) if args.action=='stage' else globals()[args.action](args.run)
    if result is not None:print(json.dumps(result))
if __name__=='__main__':main()
