"""Finite mrun transaction: compile -> fresh raw text serving -> seal -> reference."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

from encoder_atoms import np,read,save,sha,identity,floats

HERE=Path(__file__).resolve().parent
SMOKE=['blue sphere and orange cube.','cube. orange and sphere blue']
HELD=[
    {'id':'empty','prompt':''},
    {'id':'objects','prompt':'A blue glass sphere beside an orange cube on a white background.'},
    {'id':'reordered','prompt':'On a white background, an orange cube beside a blue glass sphere.'},
    {'id':'repeated','prompt':'blue blue blue sphere, sphere; orange cube. blue sphere orange cube.'},
    {'id':'spacing','prompt':'  A\tblue\nsphere...  beside: an orange cube?!  '},
    {'id':'unicode','prompt':'雨の京都、青いガラスの球。 café naïve — 🌧️ 🟦'},
    {'id':'special','prompt':'A sign literally reading <|im_start|>user<|im_end|> with <tool_response> painted below.'},
    {'id':'truncated','prompt':('blue sphere orange cube white background; '*110)+' a final red pyramid.'},
]
SCOPE='Complete-vocabulary token-local Qwen3 E/input-RMS/QKV/QK-RMS memoization; raw text assembly + positions. No contextual attention, later encoder, denoiser, decoder or image capability.'


def stage(root,coverage,calibration=None):
    root=Path(root).resolve();root.mkdir(exist_ok=False,parents=True);payload=root/'payload';payload.mkdir()
    spec=read(HERE/'encoder-atoms-source.json');spec.update(coverage=coverage,smokePrompts=SMOKE)
    save(payload/'compile.json',spec)
    prompts=HELD if coverage=='full' else [{'id':'smoke'+str(i),'prompt':p} for i,p in enumerate(SMOKE)]
    save(payload/'raw-prompts.json',{'prompts':prompts,'repeats':7})
    for name in ['encoder_atoms.py','encoder_atoms_native.py','encoder_atoms_study.py']:
        shutil.copyfile(HERE/name,payload/name)
    measured=None
    if calibration is not None:
        parent=Path(calibration).resolve();job=read(parent/'scheduler-job.json');collection=read(parent/'collection.json')
        report=read(parent/'result/report.json')
        if (job['state']!='succeeded' or job['job_id']!=collection['jobID'] or job['payload_custody']!=collection['custody']
            or sha(parent/'result/report.json')!=collection['inventory']['files']['report.json']['sha256']):
            raise ValueError('Calibration custody differs')
        batches=(151936+511)//512
        estimate=job['result']['elapsed_s']-report['compile']['atomCompileSeconds']+report['compile']['atomCompileSeconds']*batches/report['compile']['physicalBatches']
        measured={'jobID':job['job_id'],'requestSHA256':report['requestSHA256'],'custody':job['payload_custody'],
                  'sampledPeakRssMB':job['result']['peak_rss_mb'],'sampledPeakVramMB':job['result']['peak_vram_mb'],
                  'compileHighWaterRSSKiB':report['compile']['highWaterRSSKiB'],
                  'reservation':{**job['reservation'],'cpu_threads':1,'disk_gb':4,'est_wall_s':float(np.ceil(estimate))},
                  'reason':'Reuse admitted smoke RAM/VRAM profile; fixed512-row streamed compilation keeps working set bounded. CPU math is explicitly1thread. Full store+forecasts+refs need <4GB disk; estimate scales measured smoke batch time conservatively, not a speed claim.'}
    request={'schema':'tce-raw-prompt-atoms-request-v1','coverage':coverage,'scope':SCOPE,'resourceCalibration':measured,
             'sourceHashes':{p.name:sha(p) for p in payload.iterdir()},
             'physicalBatchGeometry':[1,512,2560],'compilerTokenRows':151936 if coverage=='full' else 'frozen512lowIDs+293specialIDs+smokeIDs',
             'offlineNativePrimitiveEvaluation':True,'onlineOriginalDenseProjections':0,
             'onlineLearnedFunctionMemoization':True,'fullEncoderForwards':0,
             'outputRetention':'All151936atomrows and complete forecast/reference operands remain near compute with hash-bound manifests. Collect compact measurements/row diagnostics only.'}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(payload/'request.json',request)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256'],'coverage':coverage}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root=Path(root).resolve();request=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect this handle')
    if identity({k:v for k,v in request.items() if k!='requestSHA256'})!=request['requestSHA256']:raise ValueError('Request changed')
    for name,digest in request['sourceHashes'].items():
        if sha(root/'payload'/name)!=digest:raise ValueError('Staged source changed')
    resources={}
    if request['coverage']=='full':
        from mrun.client.api import Api
        measured=request.get('resourceCalibration')
        if not measured:raise ValueError('Full compilation requires a collected measured smoke calibration')
        parent=Api().json('GET','/api/jobs/'+measured['jobID'])
        if parent['state']!='succeeded' or parent['payload_custody']!=measured['custody'] or parent['config']['requestSHA256']!=measured['requestSHA256']:
            raise ValueError('Calibration scheduler identity differs')
        r=measured['reservation'];resources={'ram_mb':r['ram_mb'],'vram_mb':r['vram_mb'],'disk_gb':r['disk_gb'],'cpu_threads':r['cpu_threads'],'est_wall_s':r['est_wall_s']}
    cmd=command()[:-5]+['python','-u','encoder_atoms_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-raw-prompt-atoms-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cuda':True},
               config={'schema':request['schema'],'requestSHA256':request['requestSHA256'],'coverage':request['coverage'],
                       'backend':'native-cuda-bf16-prefix-offline-numpy-serving','device':'cuda','batch_size':1,'seq_len':512,'fullEncoderForwards':0},
               timeout_s=600,detach=True,retry_on_kill=False,preflight=True,guarded=True,**resources,
               note='Raw-prompt lexical atoms '+request['coverage']+': CUDA BF16 token-local prefix B1x512, 7 primitive calls/batch; no full encoder/denoiser/VAE. Stream E+QKV shards, fresh CPU1 serving+3interface controls seals before native prefix refs. Finite600s,no retries; host-local checkpoint.')
    value={'jobID':str(job),'requestSHA256':request['requestSHA256']};save(root/'job.json',value);return value


def metric(a,b):
    delta=floats(a).astype(np.float64)-floats(b).astype(np.float64);base=floats(b).astype(np.float64)
    rms=float(np.sqrt(np.mean(delta*delta)));scale=float(np.sqrt(np.mean(base*base)))
    return {'wordMismatches':int(np.count_nonzero(a!=b)),'scalars':a.size,'rms':rms,'referenceRms':scale,
            'relativeRms':rms/scale if scale else None,'maxAbs':float(np.max(np.abs(delta)))}


def execute(root):
    root=Path(root).resolve();request=read(root/'request.json')
    if identity({k:v for k,v in request.items() if k!='requestSHA256'})!=request['requestSHA256']:raise ValueError('Request changed')
    for name,digest in request['sourceHashes'].items():
        if sha(root/name)!=digest:raise ValueError('Staged source changed')
    started=time.perf_counter();out=root/'response-result';out.mkdir(exist_ok=False);store=root/'atom-store'
    subprocess.run([sys.executable,str(root/'encoder_atoms_native.py'),'compile',str(root/'compile.json'),str(store)],check=True)
    compile_receipt=read(store/'compile-receipt.json');manifest_sha=sha(store/'manifest.json')
    spec={**read(root/'raw-prompts.json'),'store':str(store),'manifestSHA256':manifest_sha};save(root/'serve.json',spec)
    serve0=time.perf_counter();subprocess.run([sys.executable,str(root/'encoder_atoms.py'),str(root/'serve.json'),str(root/'prefix-predictions')],check=True)
    serve_s=time.perf_counter()-serve0;seal_path=root/'prefix-predictions/prediction-seal.json';seal_sha=sha(seal_path);seal=read(seal_path)
    print('RAW_PROMPT_PREFIX_SEALED',seal_sha,flush=True)
    # No native prefix references are evaluated/opened until all primary/control forecasts seal.
    subprocess.run([sys.executable,str(root/'encoder_atoms_native.py'),'reference',str(root/'compile.json'),str(root/'prefix-references'),'--serve',str(root/'serve.json')],check=True)
    native=read(root/'prefix-references/reference-receipt.json');targets={item['id']:item for item in native['rows']};scores=[]
    with (out/'observations.jsonl').open('x') as observations:
        for row in seal['rows']:
            if sha(row['path'])!=row['sha256']:raise ValueError('Forecast changed after seal')
            target=targets[row['id']]
            if sha(target['path'])!=target['sha256']:raise ValueError('Native reference changed')
            with np.load(row['path'],allow_pickle=False) as a,np.load(target['path'],allow_pickle=False) as b:
                fields={key:metric(a[key],b[key]) for key in ['e','q','k','v','qpos','kpos']}
                scores.append({'id':row['id'],'control':row['control'],'fields':fields,
                               'tokenIDMismatches':int(np.count_nonzero(a['ids']!=b['ids'])),
                               'maskMismatches':int(np.count_nonzero(a['mask']!=b['mask'])),
                               'positionMismatches':int(np.count_nonzero(a['positions']!=b['positions'])),
                               'renderedTextEqual':row['renderedTextSHA256']==target['renderedTextSHA256']})
                if row['control']=='primary':
                    for i in range(512):
                        record={'id':row['id'],'position':i,'tokenID':int(a['ids'][i]),'active':bool(a['mask'][i]),
                                'fields':{key:metric(a[key][i],b[key][i]) for key in fields}}
                        observations.write(__import__('json').dumps(record,allow_nan=False)+'\n')
    if sha(seal_path)!=seal_sha:raise ValueError('Seal changed during references')
    for name,path in [('manifest.json',store/'manifest.json'),('compile-receipt.json',store/'compile-receipt.json'),
                      ('prediction-seal.json',seal_path),('reference-receipt.json',root/'prefix-references/reference-receipt.json'),
                      ('request.json',root/'request.json'),('raw-prompts.json',root/'raw-prompts.json')]:shutil.copyfile(path,out/name)
    save(out/'report.json',{'schema':'tce-raw-prompt-atoms-result-v1','requestSHA256':request['requestSHA256'],'scope':SCOPE,
         'compile':compile_receipt,'serving':seal,'freshServingProcessSeconds':serve_s,'nativeReference':native,'scores':scores,
         'predictionSealSHA256':seal_sha,'manifestSHA256':manifest_sha,'workerWallSeconds':time.perf_counter()-started,
         'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
         'goalStatus':'active; no contextual prompt conditioning or image generation; cached learned-function outputs remain.'})
    print('RAW_PROMPT_ATOMS_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--coverage',choices=['smoke','full'],default='smoke');p.add_argument('--calibration');a=p.parse_args()
    if a.action=='stage':result=stage(a.run,a.coverage,a.calibration)
    elif a.action=='collect':
        from response_study import collect
        result=collect(a.run)
    else:result=globals()[a.action](a.run)
    if result is not None:print(__import__('json').dumps(result))


if __name__=='__main__':main()
