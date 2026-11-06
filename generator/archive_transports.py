"""Move explicitly selected completed-run transport copies to verified fleet storage.

Extracted evidence and custody records stay local. Every removal follows an exact
remote checksum, a restoration pointer, and a fresh local identity check. Byte I/O
only; this is not a model runner or an automatic workspace-wide deletion policy.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from encoder_atoms import read,save,sha,identity

STORE='/mnt/hdd8tb/cold-storage/mbp1/time-circuit-explorer/transports'


def ssh(code):
    return subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3','-'],
                          input=code,text=True,capture_output=True,check=True,timeout=180).stdout


def stage(root,sources):
    from mrun.client.api import Api
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);items=[];api=Api()
    for directory in sources:
        directory=Path(directory).resolve()
        if not directory.is_relative_to(Path('/Users/jakeholl/domains/artifacts')):raise ValueError('Outside declared artifact root')
        job=read(directory/'job.json');live=api.json('GET','/api/jobs/'+job['job_id'])
        if live['state'] not in ['succeeded','failed','killed','cancelled','timed_out']:raise ValueError('Source job is not terminal')
        if not (directory/'extracted').is_dir():raise ValueError('Keep extracted evidence local before moving transport copies')
        for name,receipt,size_key,sha_key in [('archive.tar.gz','archive-custody.json','archive_bytes','archive_sha256'),
                                             ('worker.log.gz','log-custody.json','compressed_bytes','compressed_sha256')]:
            path=directory/name;custody=read(directory/receipt);s=path.stat()
            if path.is_symlink() or not path.is_file() or not custody.get('complete') or custody['job_id']!=job['job_id']:
                raise ValueError('Source custody invalid')
            if s.st_size!=custody[size_key] or sha(path)!=custody[sha_key]:raise ValueError('Transport no longer matches custody')
            items.append({'path':str(path),'bytes':s.st_size,'sha256':custody[sha_key],
                          'statIdentity':[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns],'jobID':job['job_id'],
                          'custodyPath':str(directory/receipt),'custodySHA256':sha(directory/receipt),
                          'remote':STORE+'/sha256/'+custody[sha_key][:2]+'/'+custody[sha_key]})
    plan={'schema':'tce-transport-cleanup-plan-v1','items':items,'localFreeBytesBefore':shutil.disk_usage(root).free,
          'bytesToArchive':sum(x['bytes'] for x in items),'createdAtUnix':time.time(),
          'scope':'Explicit completed-run archive/log transport copies only; all extracted scientific artifacts and original receipts remain local.'}
    plan['planSHA256']=identity(plan);save(root/'plan.json',plan)
    return {'plan':str(root/'plan.json'),'files':len(items),'bytes':plan['bytesToArchive']}


def archive(root):
    root=Path(root).resolve();plan=read(root/'plan.json')
    if identity({k:v for k,v in plan.items() if k!='planSHA256'})!=plan['planSHA256']:raise ValueError('Cleanup plan changed')
    if (root/'result.json').exists():raise ValueError('Already completed')
    rows=[]
    for i,item in enumerate(plan['items']):
        path=Path(item['path']);pointer=path.with_name(path.name+'.cold.json')
        if not path.exists():
            if not pointer.exists() or read(pointer)['sha256']!=item['sha256']:raise ValueError('Missing unarchived source')
        else:
            s=path.stat()
            if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]!=item['statIdentity'] or sha(path)!=item['sha256']:raise ValueError('Source changed since plan')
            temporary=STORE+'/incoming/'+item['sha256']
            ssh('from pathlib import Path\nPath('+repr(temporary)+').parent.mkdir(parents=True,exist_ok=True)\nPath('+repr(item['remote'])+').parent.mkdir(parents=True,exist_ok=True)\n')
            subprocess.run(['rsync','-a','--partial','--protect-args',str(path),'beast:'+temporary],check=True,timeout=900)
            # A final name becomes authoritative only after its complete bytes verify.
            verify='''import hashlib,json,os
from pathlib import Path
p=Path(TEMP);final=Path(FINAL)
def digest(x):
 h=hashlib.sha256()
 with x.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
if p.is_symlink() or p.stat().st_size!=SIZE or digest(p)!=SHA:raise ValueError('Cold copy mismatch')
if final.exists():
 if final.is_symlink() or final.stat().st_size!=SIZE or digest(final)!=SHA:raise ValueError('Existing cold object corrupt')
 p.unlink()
else:
 with p.open('rb') as f:os.fsync(f.fileno())
 p.rename(final)
print(json.dumps({'path':str(final),'sha256':SHA,'bytes':SIZE,'verified':True}))
'''.replace('TEMP',repr(temporary)).replace('FINAL',repr(item['remote'])).replace('SIZE',str(item['bytes'])).replace('SHA',repr(item['sha256']))
            remote=json.loads(ssh(verify))
            record={'schema':'tce-cold-transport-pointer-v1',**item,'remoteHost':'beast','remoteVerification':remote,
                    'restoreCommand':['rsync','-a','--protect-args','beast:'+item['remote'],str(path)],'planSHA256':plan['planSHA256']}
            if not pointer.exists():save(pointer,record)
            elif read(pointer)!=record:raise ValueError('Restoration pointer differs')
            s=path.stat()
            if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]!=item['statIdentity'] or sha(path)!=item['sha256']:raise ValueError('Local copy changed during transfer')
            if sha(item['custodyPath'])!=item['custodySHA256']:raise ValueError('Custody changed during transfer')
            path.unlink()
        rows.append(read(pointer));print('TRANSPORT_ARCHIVED',i+1,len(plan['items']),item['bytes'],flush=True)
        with (root/'events.jsonl').open('a') as f:f.write(json.dumps({'path':str(path),'sha256':item['sha256'],'remote':item['remote'],'localRemoved':True})+'\n')
    result={'schema':'tce-transport-cleanup-result-v1','planSHA256':plan['planSHA256'],'rows':rows,
            'archivedBytes':sum(x['bytes'] for x in rows),'files':len(rows),'localFreeBytesBefore':plan['localFreeBytesBefore'],
            'localFreeBytesAfter':shutil.disk_usage(root).free,'scope':plan['scope'],'status':'complete'}
    save(root/'result.json',result);return {k:result[k] for k in ['files','archivedBytes','localFreeBytesBefore','localFreeBytesAfter','status']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','archive']);p.add_argument('run');p.add_argument('--source',action='append',default=[]);a=p.parse_args()
    print(stage(a.run,a.source) if a.action=='stage' else archive(a.run))
