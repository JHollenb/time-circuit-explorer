"""Stage, schedule, inspect, and collect bounded prediction-only jobs through mrun.

Use the workspace's common exec launcher, or an environment with mrun installed.
Credentials stay in the scheduler client's environment and never enter payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import time

from worker import validate_input,sha,verify_sources

HERE=Path(__file__).resolve().parent
MODEL='/mnt/big/llm-models/FLUX.2-klein-4B'
PACKAGES=['torch==2.13.0','diffusers==0.39.0','transformers==5.14.1','accelerate==1.14.0','safetensors>=0.5','numpy>=2.0','psutil>=5.9','pillow>=10.0','pyyaml>=6.0']


def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)


def command():
    result=['/usr/bin/env','PYTHONUNBUFFERED=1','PYTHONNOUSERSITE=1','PYTHONPATH=',
        'UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache','UV_LINK_MODE=symlink','TORCH_HOME=/mnt/hdd8tb/mdb/torch',
        'TMPDIR=/mnt/hdd8tb/mdb/tmp','CUDA_CACHE_PATH=/mnt/hdd8tb/mdb/cuda-cache',
        'HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','DIFFUSERS_OFFLINE=1','TOKENIZERS_PARALLELISM=false',
        'uv','run','--no-project','--offline','--python','3.14.6']
    for requirement in PACKAGES:result.extend(['--with',requirement])
    return result+['python','-u','worker.py','--request','request.json']


def stage(run_dir,samples):
    manifest=verify_sources()
    if not 1<=len(samples)<=2:raise ValueError('One finite lease accepts one or two serial images.')
    samples=[dict(zip(['prompt','seed'],validate_input(x['prompt'],x['seed']))) for x in samples]
    run_dir=Path(run_dir).resolve();run_dir.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(__file__),run_dir/'launcher.py')
    payload=run_dir/'payload';payload.mkdir()
    names=['worker.py','kernel-manifest.json',*manifest['files']]
    for name in names:
        dest=payload/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/name,dest)
    request={'schema':'tce-flux-prediction-request-v1','config':{'modelPath':MODEL,'output':'prediction-result','samples':samples},
        'sourceHashes':{name:sha(payload/name) for name in names},'launcherSHA256':sha(Path(__file__)),
        'model':'flux2-klein-4b-distilled','backend':'diffusers','device':'cuda','dtype':'bfloat16','batch_size':1,'height':512,'width':512,
        'task_family':'forward','executionContract':{'nativeTransformerLoaded':False,'nativeDenoiserCalls':0,'predictedTransitionsPerImage':112,'payloadCustodyRequired':True,
            'nativePromptEncodingsPerImage':1,'nativeVAEDecodesPerImage':1,'cpuThreads':4,'expectedWallSeconds':500*len(samples)+500,
            'retention':'final latent, PNG, per-stage resources, pass hashes and bounded progress; no intermediate activations'},
        'resourceCalibration':{'job':'job-4350531b1dcd','rssMB':8509.1,'vramMB':12936,
            'scope':'Historical full validation harness. New generator releases encoder after encoding, omits transformer residency and trajectory capture. New actual peaks reported separately.'},
        'resourceException':{'ramMB':14336,'vramMB':13000,'cpuThreads':4,'diskGB':1,
            'reason':'Intent-first attempt rejected by an impossible generalized-history15360MB VRAM ask (guard ceiling16896 >15557 physical-minus-margin). Reuse historical admitted14336MBRAM/13000MBVRAM profile, measuredRSS8509/VRAM12936. Release encoder after encoding as historical worker did; native transformer and2GBtrajectory omitted. Subsequent encoder reload included in encode time. Actual4 CPU threads; output below1MB perimage, pretrained constants external.'}}
    request['requestSHA256']=identity(request);save(run_dir/'request.json',request);save(payload/'request.json',request)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[*names,'request.json','preflight.v1.json'],
        'path_token_allowlist':[MODEL,'prediction-result']})
    return request


def submit(run_dir):
    from mrun.client.submit import launch
    root=Path(run_dir).resolve();request=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('This request already has a job; inspect or collect it instead.')
    if identity({k:v for k,v in request.items() if k!='requestSHA256'})!=request['requestSHA256']:raise ValueError('Request identity changed')
    if sha(Path(__file__))!=request['launcherSHA256']:raise ValueError('Launcher changed; stage a new request')
    for name,digest in request['sourceHashes'].items():
        if sha(root/'payload'/name)!=digest:raise ValueError('Staged source changed: '+name)
    total=len(request['config']['samples']);cmd=command()
    job_id=launch(cmd,experiment='time-circuit-flux-prediction-only-v1',model=request['model'],task='forward',payload=root/'payload',
        prefer_host='beast',gpu=True,needs={'cuda':True},config=request,ram_mb=14336,vram_mb=13000,cpu_threads=4,disk_gb=1,est_wall_s=500*total+500,
        timeout_s=1500 if total==1 else 1800,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note=f'Prediction-only Klein4B B1 512: {total} serial image(s), {112*total} NumPy transitions, native encoder/VAE only. No native transformer or comparison. CPU4; finite lease, no retries. Save PNG/final bits/resources; stop on source, parity, or finite-value errors.')
    result={'jobID':str(job_id),'requestSHA256':request['requestSHA256'],'runDirectory':str(root)}
    save(root/'job.json',result);return result


def api_client():
    from mrun.client.api import Api
    return Api()


def status(run_dir,*,include_job=False):
    root=Path(run_dir).resolve();job_id=read(root/'job.json')['jobID'];api=api_client();job=api.json('GET','/api/jobs/'+job_id)
    result={'jobID':job_id,'state':job['state'],'host':job.get('assigned_host'),'requestSHA256':read(root/'request.json')['requestSHA256']}
    _,body,_=api.request('GET','/api/jobs/'+job_id+'/logs')
    events=[]
    for line in body.decode(errors='replace').splitlines():
        if 'TCE_EVENT ' not in line:continue
        try:events.append(json.loads(line.split('TCE_EVENT ',1)[1]))
        except (ValueError,TypeError):continue
    result['progress']=events[-1] if events else None
    result['completedTransitions']=max([int(x.get('completed',0)) for x in events if x.get('kind')=='transition'] or [0])
    if job['state'] in ['failed','killed','cancelled','timed_out','timeout']:
        result['error']=(job.get('result') or {}).get('failure',{}).get('message','Generation did not finish; partial evidence is retained.')
    if include_job:result['_job']=job
    return result


def inventory_code(remote):
    return '''import hashlib,json,os,stat
from pathlib import Path
root=Path(ROOT)
if root.is_symlink() or not root.is_dir():raise ValueError("Missing/unsafe result directory")
files={};total=0
for path in sorted(root.rglob("*")):
 if path.is_symlink():raise ValueError("Symlink result")
 if path.is_dir():continue
 if not stat.S_ISREG(path.stat().st_mode):raise ValueError("Nonregular result")
 size=path.stat().st_size;total+=size
 if size>8<<20 or total>32<<20 or len(files)>50:raise ValueError("Collection bound exceeded")
 files[str(path.relative_to(root))]={"bytes":size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps({"files":files,"bytes":total},sort_keys=True))
'''.replace('ROOT',repr(remote),1)


def collect(run_dir):
    root=Path(run_dir).resolve();snapshot=status(root,include_job=True);job=snapshot['_job'];job_id=snapshot['jobID'];request=read(root/'request.json')
    if job['state']!='succeeded':raise ValueError('Collection requires a succeeded job: '+job['state'])
    if job.get('assigned_host')!='beast' or not re.fullmatch(r'job-[A-Za-z0-9_-]+',job_id):raise ValueError('Unknown collection host/job')
    if identity({k:v for k,v in request.items() if k!='requestSHA256'})!=request['requestSHA256']:raise ValueError('Request identity changed')
    if sha(root/'launcher.py')!=request['launcherSHA256']:raise ValueError('Archived request-bound launcher differs')
    original_command=runpy.run_path(str(root/'launcher.py'))['command']()
    normalized_config=dict(job.get('config') or {});guarded_id=normalized_config.pop('mrun_guarded_request_sha256',None)
    if guarded_id is not None and (not re.fullmatch(r'[0-9a-f]{64}',guarded_id) or not (job.get('needs') or {}).get('payload_custody_v2')):raise ValueError('Invalid guarded wrapper identity')
    if normalized_config!={**request,'retry_on_kill':False} or job.get('cmd')!=original_command:raise ValueError('Scheduler normalized config/command differs from staged request')
    result=job.get('result') or {};custody=job.get('payload_custody') or {};executed=result.get('executed_payload') or custody.get('executed')
    payload=custody.get('sealed') or {}
    if result.get('returncode')!=0 or not re.fullmatch(r'[0-9a-f]{64}',payload.get('sha256','')):raise ValueError('Successful exit/sealed payload identity absent')
    for phase in ['declared','sealed']:
        if any((custody.get(phase) or {}).get(key)!=payload.get(key) for key in ['sha256','size_bytes']):raise ValueError('mrun declared/sealed payload differs')
    if executed and any(executed.get(key)!=payload.get(key) for key in ['sha256','size_bytes']):raise ValueError('mrun executed payload differs')
    if executed and executed.get('host')!=job['assigned_host']:raise ValueError('Executed payload host differs from assigned host')
    if request['executionContract'].get('payloadCustodyRequired') and (custody.get('required') is not True or not job.get('admission') or guarded_id is None):raise ValueError('Required guarded admission/custody absent')
    if (custody.get('required') or request['executionContract'].get('payloadCustodyRequired')) and not executed:raise ValueError('Required executed-payload seal absent')
    cwd=result.get('diagnostics',{}).get('cwd')
    if cwd!='/mnt/hdd8tb/mrun-work/'+job_id:raise ValueError('Unexpected authenticated worker directory')
    remote=cwd+'/prediction-result'
    def inventory(path):
        return json.loads(subprocess.run(['ssh','-T','beast','python3','-'],input=inventory_code(path),text=True,capture_output=True,check=True,timeout=30).stdout)
    expected={str(p.relative_to(root/'payload')):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (root/'payload').rglob('*') if p.is_file()}
    if read(root/'payload/request.json')!=request:raise ValueError('Staged payload request differs')
    for name,digest in request['sourceHashes'].items():
        if expected[name]['sha256']!=digest:raise ValueError('Staged source differs: '+name)
    source_before=inventory(cwd)
    if any(source_before['files'].get(name)!=entry for name,entry in expected.items()):raise ValueError('Authenticated worker input files differ from staged payload')
    def generated_paths(snapshot):
        paths=[name for name in snapshot['files'] if name not in expected]
        if any(name!='job.log' and not name.startswith('prediction-result/') and not re.fullmatch(r'(?:kernels/)?__pycache__/[^/]+\.pyc',name) for name in paths):raise ValueError('Unexpected extra file in authenticated worker directory')
        return paths
    generated=generated_paths(source_before)
    before=inventory(remote);local=root/'result';local.mkdir(exist_ok=True)
    subprocess.run(['rsync','-a','--protect-args','--max-size=8m','beast:'+remote+'/',str(local)+'/'],check=True,timeout=120,capture_output=True)
    after=inventory(remote);source_after=inventory(cwd)
    generated_paths(source_after)
    actual={str(p.relative_to(local)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in local.rglob('*') if p.is_file()}
    if before!=after or actual!=before['files']:raise ValueError('Remote-before/after and collected inventories differ')
    if any(source_after['files'].get(name)!=entry for name,entry in expected.items()):raise ValueError('Authenticated worker input files changed during collection')
    report=read(local/'report.json')
    if report['requestSHA256']!=request['requestSHA256'] or report['sourceManifestSHA256']!=sha(root/'payload/kernel-manifest.json') or len(report['samples'])!=len(request['config']['samples']):raise ValueError('Result/request/source binding differs')
    if report['counts']!={'nativeDenoiser':0,'promptEncodings':len(report['samples']),'vaeDecodes':len(report['samples']),'predictedTransitions':112*len(report['samples'])} or report['nativeTransformerLoaded']:raise ValueError('Completed prediction-only census differs')
    for sample in report['samples']:
        for key in ['image','latent']:
            if before['files'][sample[key]['path']]['sha256']!=sample[key]['sha256']:raise ValueError('Sample digest differs')
    receipt={'schema':'tce-generator-collection-v2','jobID':job_id,'requestSHA256':request['requestSHA256'],
        'payload':{key:payload[key] for key in ['sha256','size_bytes']},'executedPayloadSeal':executed,'payloadCustodyRequired':bool(custody.get('required')),
        'payloadCustody':custody,
        'sourceAudit':{'authority':'Authenticated post-run input hashes before/after collection; these are not a retroactive execution-time attestation.','inputs':expected,'workerInventoryBefore':source_before,'workerInventoryAfter':source_after,'generatedPaths':generated},
        'remoteInventory':before,'reportSHA256':sha(local/'report.json'),'schedulerResult':result,
        'scope':'Succeeded scheduler request/command, matching declared/sealed payload, explicit executed-seal availability, and authenticated SSH input/result inventories. No new model execution during collection.'}
    if not (root/'collection.json').exists():save(root/'collection.json',receipt)
    if not (root/'collection-v2.json').exists():save(root/'collection-v2.json',receipt)
    if not (root/'scheduler-job.json').exists():save(root/'scheduler-job.json',job)
    return {'jobID':job_id,'resultDirectory':str(local),'images':len(report['samples']),'report':str(local/'report.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['stage','submit','status','collect']);parser.add_argument('run_dir',type=Path)
    parser.add_argument('--prompt');parser.add_argument('--seed',type=int,default=113);parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    if args.action=='stage':
        baseline=verify_sources()['baseline'];samples=[{'prompt':args.prompt or baseline['prompt'],'seed':args.seed}]
        if args.benchmark:samples=[{'prompt':baseline['prompt'],'seed':113},{'prompt':'A small blue glass sphere beside an orange cube on a plain white background. Two separate objects, simple studio product photograph.','seed':2718}]
        result=stage(args.run_dir,samples);result={'requestSHA256':result['requestSHA256'],'runDirectory':str(args.run_dir.resolve())}
    else:result=globals()[args.action](args.run_dir)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
