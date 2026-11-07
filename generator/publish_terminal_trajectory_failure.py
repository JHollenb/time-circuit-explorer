"""Preserve the first trajectory's native-SDPA keyword instrumentation failure."""
import base64
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read,save,sha,identity


def execute(root,destination):
    from mrun.client.api import Api
    root=Path(root).resolve();r=read(root/'request.json');job=Api().json('GET','/api/jobs/'+read(root/'job.json')['jobID'])
    if job['state']!='failed' or job['config']['requestSHA256']!=r['requestSHA256'] or identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Failed request identity differs')
    custody=job['payload_custody']
    if not custody['required'] or not job['admission'] or any(any(custody[p][n]!=custody['declared'][n] for n in ['sha256','size_bytes']) for p in ['sealed','executed']):raise ValueError('Guarded custody differs')
    for n,h in r['sourceHashes'].items():
        if sha(root/'payload'/n)!=h:raise ValueError('Frozen failed payload differs')
    code='''import base64,json,sys,hashlib
from pathlib import Path
p=Path(json.load(sys.stdin)['root']);files={}
for q in [p/'job.log',*(p/'response-result').rglob('*')]:
 if not q.is_file():continue
 if q.is_symlink() or q.stat().st_size>2<<20:raise ValueError('Unexpected failed result geometry')
 raw=q.read_bytes();files[str(q.relative_to(p))]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'base64':base64.b64encode(raw).decode()}
print(json.dumps({'files':files,'arrays':[str(x.relative_to(p)) for x in (p/'arrays').rglob('*') if x.is_file()],'candidateImages':[str(x.relative_to(p)) for x in p.glob('*.png')]}))'''
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','beast','python3 -c '+shlex.quote(code)],input=json.dumps({'root':job['result']['diagnostics']['cwd']}),text=True,capture_output=True,check=True,timeout=30)
    actual=json.loads(proc.stdout)
    if actual['arrays'] or actual['candidateImages'] or set(actual['files'])!={'job.log','response-result/toy-mechanics.json'}:raise ValueError('Inspect unexpected partial scientific artifacts')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    for name,item in actual['files'].items():
        target=out/Path(name).name;target.write_bytes(base64.b64decode(item['base64']))
        if sha(target)!=item['sha256']:raise ValueError('Failed file transfer differs')
    if sha(out/'job.log')!=job['result']['artifacts']['job_log']['sha256']:raise ValueError('Authoritative complete log differs')
    for n in ['request.json','frozen-plan.md','resource-basis.json']:shutil.copyfile(root/'payload'/n,out/n)
    save(out/'failure.json',{'schema':'tce-terminal-trajectory-failure-v1','jobID':job['job_id'],'requestSHA256':r['requestSHA256'],'state':job['state'],
        'payloadCustody':custody,'admission':job['admission'],'result':job['result'],'completeLogSHA256':sha(out/'job.log'),
        'partialResultFiles':['toy-mechanics.json'],'observedArrays':actual['arrays'],'observedImages':actual['candidateImages'],
        'cause':'Native attention dispatch calls SDPA with query/key/value keywords; v1 measurement wrapper required q/k/v positional names. First denoiser aborts at first joint attention.',
        'partialWork':'Original model/VAE loaded; clock/context/image projections and first joint pre-attention work reached. No durable complete denoiser/work ledger, so exact completed MACs unavailable.',
        'mechanics_status':'failed-at-instrument-call','instrument_status':'keyword-API-mismatch','terminal_status':'unassessed',
        'scope':'Tiny toy mechanics result retained; no candidate model arrays, sealed trajectory, final decode or image. An instrument failure, not a scientific null.'})
    save(out/'publication.json',{'publisherSourceSHA256':sha(__file__),'source':str(root),'sourceHashes':r['sourceHashes'],'files':{p.name:sha(p) for p in out.iterdir()}})
    return {'jobID':job['job_id'],'completeLogSHA256':sha(out/'job.log'),'partialMetadataFiles':1,'candidateImages':0}


if __name__=='__main__':
    import sys
    print(json.dumps(execute(sys.argv[1],sys.argv[2])))
