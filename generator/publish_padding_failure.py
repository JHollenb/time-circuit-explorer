"""Retain the terminal guard failure and its completed source-only compiler."""
import argparse
import base64
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read, save, sha, identity
from publish_source_context import REMOTE


def publish(root, destination):
    from mrun.client.api import Api
    root = Path(root).resolve(); request = read(root/'request.json')
    job = Api().json('GET', '/api/jobs/'+read(root/'job.json')['jobID'])
    if job['state'] != 'failed' or job['config']['requestSHA256'] != request['requestSHA256']:
        raise ValueError('Terminal failed request required')
    if identity({k:v for k,v in request.items() if k!='requestSHA256'}) != request['requestSHA256']:
        raise ValueError('Request differs')
    custody = job['payload_custody']
    if not custody['required'] or not job['admission']: raise ValueError('Guarded admission required')
    for phase in ['sealed','executed']:
        for k in ['sha256','size_bytes']:
            if custody[phase][k] != custody['declared'][k]: raise ValueError('Payload differs')
    for n,h in request['sourceHashes'].items():
        if sha(root/'payload'/n) != h: raise ValueError('Frozen source differs')
    remote = job['result']['diagnostics']['cwd']
    code = '''import base64,json,sys
from pathlib import Path
p=Path(json.load(sys.stdin)['root']);names=['compile-receipt.json','job.log']
print(json.dumps({'files':{n:base64.b64encode((p/n).read_bytes()).decode() for n in names},
'absent':[n for n in ['reader-seal.json','encoder-candidate.json','images-candidate.json','candidate-seal.json','native-write-diagnostic.json'] if not (p/n).exists()],
'resultFiles':[x.name for x in (p/'response-result').iterdir()]}))'''
    proc = subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(code)],
        input=json.dumps({'root':remote}), text=True, capture_output=True, check=True, timeout=60)
    data=json.loads(proc.stdout)
    if len(data['absent']) != 5 or data['resultFiles']: raise ValueError('Failure scope differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    for n,v in data['files'].items(): (out/n).write_bytes(base64.b64decode(v))
    receipt=read(out/'compile-receipt.json')
    if receipt['requestSHA256'] != request['requestSHA256'] or sha(out/'compile-receipt.json') not in (out/'job.log').read_text():
        raise ValueError('Executed compiler receipt differs')
    if sha(out/'job.log') != job['result']['artifacts']['job_log']['sha256']: raise ValueError('Scheduler log differs')
    source=read(root/'payload/inputs.json')
    items=[*receipt['fields'].values(),*receipt['sourceAtomFiles']]
    items += [{'path':remote+'/'+n,'sha256':sha(out/n)} for n in ['compile-receipt.json','job.log']]
    spec={'allowedRoots':[remote,source['lexicalStore']],'items':items}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps(spec),text=True,capture_output=True,check=True,timeout=180)
    verification=json.loads(proc.stdout);verification.update(jobID=job['job_id'],verifierSourceSHA256=sha(__file__))
    save(out/'retained-array-verification.json',verification)
    shutil.copyfile(root/'request.json',out/'request.json')
    shutil.copyfile(root/'payload/frozen-plan.md',out/'frozen-plan.md')
    save(out/'failure.json',{'schema':'tce-padding-mlp-failure-v1','jobID':job['job_id'],'requestSHA256':request['requestSHA256'],
        'state':job['state'],'payloadCustody':custody,'admission':job['admission'],'result':job['result'],
        'absentRecords':data['absent'],'resultFiles':data['resultFiles'],'newImages':0,
        'mechanics_status':'failed-before-reader','instrument_status':'source-compiler-complete-only','terminal_status':'unassessed',
        'cause':'Outer target guard blocked frozen reader integrity hashing, before head formation. Repaired v2 driver authenticates before the reader installs its own guard.',
        'scope':'Paid source compilation retained; no reader, encoder/image candidate, native comparison or scientific null.'})
    save(out/'files.json',{f.name:sha(f) for f in out.iterdir()})
    return {'files':len(verification['files']),'bytesVerified':verification['totalBytesVerified'],'path':str(out)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('destination');a=p.parse_args();print(json.dumps(publish(a.run,a.destination)))
