"""Preserve the first terminal capture's pinned-wheel persistence failure."""
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
    if job['state']!='failed' or job['config']['requestSHA256']!=r['requestSHA256'] or identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:
        raise ValueError('Expected failed original request')
    custody=job['payload_custody']
    if not custody['required'] or not job['admission'] or any(any(custody[p][n]!=custody['declared'][n] for n in ['sha256','size_bytes']) for p in ['sealed','executed']):raise ValueError('Guarded custody differs')
    for n,h in r['sourceHashes'].items():
        if sha(root/'payload'/n)!=h:raise ValueError('Frozen failed payload differs')
    code='''import base64,json,sys
from pathlib import Path
p=Path(json.load(sys.stdin)['root']);raw=(p/'job.log').read_bytes()
print(json.dumps({'log':base64.b64encode(raw).decode(),'resultFiles':[str(x.relative_to(p)) for x in (p/'response-result').rglob('*') if x.is_file()],
 'storeFiles':[str(x.relative_to(p)) for x in (p/'terminal-store').rglob('*') if x.is_file()],'arrayFiles':[str(x.relative_to(p)) for x in (p/'arrays').rglob('*') if x.is_file()]}))'''
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','beast','python3 -c '+shlex.quote(code)],input=json.dumps({'root':job['result']['diagnostics']['cwd']}),text=True,capture_output=True,check=True,timeout=30)
    actual=json.loads(proc.stdout)
    if actual['resultFiles'] or actual['storeFiles'] or actual['arrayFiles']:raise ValueError('Inspect unexpected partial artifacts before publication')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);(out/'job.log').write_bytes(base64.b64decode(actual['log']))
    if sha(out/'job.log')!=job['result']['artifacts']['job_log']['sha256']:raise ValueError('Complete scheduler log identity differs')
    for n in ['request.json','frozen-plan.md','resource-basis.json']:shutil.copyfile(root/'payload'/n,out/n)
    save(out/'failure.json',{'schema':'tce-terminal-capture-failure-v1','jobID':job['job_id'],'requestSHA256':r['requestSHA256'],
        'state':job['state'],'payloadCustody':custody,'admission':job['admission'],'result':job['result'],'completeLogSHA256':sha(out/'job.log'),
        'observedEmptyResult':actual['resultFiles'],'observedEmptyTerminalStore':actual['storeFiles'],'observedNoArrays':actual['arrayFiles'],
        'cause':'Pinned saturn_pub0.4.1 wheel has no shared_store module; development source did. Failure at first persist before any terminal cut or numerical artifact sealed.',
        'prefixWork':'Traceback reaches single19 source persist after the four-joint/nineteen-single loop. This is code/trace inference; live counters were not durable and exact completed arithmetic is not separately measured.',
        'mechanics_status':'failed-at-persistence','instrument_status':'package-mismatch-no-sealed-capture','terminal_status':'unassessed',
        'scope':'Failed native prefix attempt retained immutably; no terminal source, target, velocity, reduced producer or image result. No model/scientific null.'})
    save(out/'publication.json',{'publisherSourceSHA256':sha(__file__),'source':str(root),'sourceHashes':r['sourceHashes'],
        'files':{p.name:sha(p) for p in out.iterdir()}})
    return {'jobID':job['job_id'],'completeLogSHA256':sha(out/'job.log'),'newNumericalArtifacts':0}


if __name__=='__main__':
    import sys
    print(json.dumps(execute(sys.argv[1],sys.argv[2])))
