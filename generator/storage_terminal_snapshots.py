"""Reversible complete terminal snapshots, including failed runs; raw outcomes unchanged."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import time
from encoder_atoms import read,save,sha,identity

WORKSPACE=Path(__file__).resolve().parents[2]
ROOTS=[WORKSPACE/'artifacts',WORKSPACE/'time-circuit-explorer/artifacts']
COLD='/mnt/hdd8tb/cold-storage/mbp1/time-circuit-explorer/transports/sha256'


def signature(p):
    s=p.stat()
    if p.is_symlink() or not stat.S_ISREG(s.st_mode):raise ValueError('Expected ordinary transport file')
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]


def remote(code,value,timeout=180):
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(code)],input=json.dumps(value),text=True,capture_output=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def stage(root,parents):
    from mrun.client.api import Api
    root=Path(root).resolve();items=[];api=Api()
    for parent in map(lambda p:Path(p).resolve(),parents):
        if not any(parent.is_relative_to(x) for x in ROOTS) or not (parent/'extracted').is_dir() or not any((parent/'extracted').rglob('*')):raise ValueError('Expected retained extracted evidence under artifact roots')
        for name,receipt,digest_key,size_key in [('archive.tar.gz','archive-custody.json','archive_sha256','archive_bytes'),('worker.log.gz','log-custody.json','compressed_sha256','compressed_bytes')]:
            p=parent/name;cp=parent/receipt;c=read(cp);job=api.json('GET','/api/jobs/'+c['job_id'])
            if c.get('complete') is not True or c.get('error') or job['state'] not in ['succeeded','failed'] or (parent/(name+'.cold.json')).exists():raise ValueError('Transport is not a fresh completed custody copy')
            sig=signature(p);h=sha(p)
            if h!=c[digest_key] or sig[2]!=c[size_key] or signature(p)!=sig:raise ValueError('Local original custody/hash differs')
            items.append({'path':str(p),'bytes':sig[2],'sha256':h,'statIdentity':sig,'custodyPath':str(cp),'custodySHA256':sha(cp),'jobID':c['job_id'],'jobStateAtPlan':job['state'],'extractedRoot':str(parent/'extracted'),'remote':COLD+'/'+h[:2]+'/'+h})
    value={'schema':'tce-transport-cleanup-plan-v1','createdAtUnix':time.time(),'items':items,'bytesToArchive':sum(x['bytes'] for x in items),'localFreeBytesBefore':shutil.disk_usage(WORKSPACE).free,'scope':'Explicit complete-custody terminal snapshot archive/log copies, including failed outcomes. Terminal failure is not a byte-integrity failure. Full cold verification/restoration pointers precede local unlink; extracted science and original outcome/custody unchanged.'};value['planSHA256']=identity(value)
    root.mkdir(exist_ok=False,parents=True);save(root/'plan.json',value);return {'files':len(items),'bytes':value['bytesToArchive']}


VERIFY="""import hashlib,json,os,stat,sys
from pathlib import Path
s=json.load(sys.stdin);p=Path(s['path']);root=Path(s['root'])
if not p.is_relative_to(root) or p.is_symlink() or not stat.S_ISREG(p.stat().st_mode):raise ValueError('Unsafe cold file')
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1<<20),b''):h.update(b)
if h.hexdigest()!=s['sha256'] or p.stat().st_size!=s['bytes']:raise ValueError('Cold full-byte identity differs')
dest=Path(s.get('destination',str(p)))
if dest!=p:
 if not dest.is_relative_to(root):raise ValueError('Unsafe destination')
 if dest.exists():
  if dest.is_symlink() or not stat.S_ISREG(dest.stat().st_mode):raise ValueError('Unsafe existing cold object')
  g=hashlib.sha256()
  with dest.open('rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):g.update(b)
  if g.hexdigest()!=s['sha256'] or dest.stat().st_size!=s['bytes']:raise ValueError('Conflicting immutable cold object')
 else:os.link(p,dest)
 p.unlink();p=dest
print(json.dumps({'path':str(p),'bytes':p.stat().st_size,'sha256':s['sha256'],'verified':True}))
"""


def execute(root):
    from mrun.client.api import Api
    root=Path(root).resolve();plan=read(root/'plan.json');api=Api()
    if identity({k:v for k,v in plan.items() if k!='planSHA256'})!=plan['planSHA256'] or (root/'result.json').exists():raise ValueError('Plan changed or completed')
    rows=[]
    with (root/'events.jsonl').open('x') as events:
        def event(kind,**values):events.write(json.dumps({'kind':kind,'timestamp':time.time(),**values})+'\n');events.flush();os.fsync(events.fileno())
        for item in plan['items']:
            p=Path(item['path']);cp=Path(item['custodyPath']);pointer=Path(str(p)+'.cold.json')
            if signature(p)!=item['statIdentity'] or sha(p)!=item['sha256'] or sha(cp)!=item['custodySHA256'] or api.json('GET','/api/jobs/'+item['jobID'])['state']!=item['jobStateAtPlan'] or not Path(item['extractedRoot']).is_dir():raise ValueError('Fresh source/custody/terminal identity differs')
            temporary=item['remote']+'.partial-'+str(os.getpid())
            remote("import json,sys;from pathlib import Path;s=json.load(sys.stdin);p=Path(s['path']);r=Path(s['root']);assert p.is_relative_to(r);p.parent.mkdir(parents=True,exist_ok=True);print('{}')",{'path':temporary,'root':COLD})
            subprocess.run(['rsync','-a','--protect-args',str(p),'beast:'+temporary],check=True,capture_output=True,timeout=180)
            verified=remote(VERIFY,{'path':temporary,'destination':item['remote'],'root':COLD,'sha256':item['sha256'],'bytes':item['bytes']})
            value={'schema':'tce-cold-transport-pointer-v1',**item,'planSHA256':plan['planSHA256'],'remoteHost':'beast','remoteVerification':verified,'restoreCommand':['rsync','-a','--protect-args','beast:'+item['remote'],str(p)]}
            with pointer.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
            directory=os.open(pointer.parent,os.O_RDONLY)
            try:os.fsync(directory)
            finally:os.close(directory)
            if read(pointer)!=value or signature(p)!=item['statIdentity'] or sha(cp)!=item['custodySHA256']:raise ValueError('Durable pointer/fresh local identity differs')
            event('verified-before-unlink',path=str(p),remote=item['remote'],sha256=item['sha256'],bytes=item['bytes']);p.unlink();event('local-transport-unlinked',path=str(p));rows.append(value)
    value={'schema':'tce-transport-cleanup-result-v1','planSHA256':plan['planSHA256'],'files':len(rows),'archivedBytes':sum(x['bytes'] for x in rows),'localFreeBytesBefore':plan['localFreeBytesBefore'],'localFreeBytesAfter':shutil.disk_usage(WORKSPACE).free,'rows':rows,'scope':'Full local/remote SHA verified; content-addressed cold bytes and durable restore pointers preserved. Existing extracted evidence/custody untouched. Shared-volume free-space change is not physical attribution.'};save(root/'result.json',value);return value


def audit(root):
    checks=[];candidates=[]
    for base in ROOTS:
        for p in base.rglob('*.cold.json'):
            x=read(p)
            if x.get('schema')!='tce-cold-transport-pointer-v1':continue
            if Path(x['path']).exists() or sha(x['custodyPath'])!=x['custodySHA256']:raise ValueError('Cold pointer/local custody differs')
            c=read(x['custodyPath']);h=c.get('archive_sha256',c.get('compressed_sha256'))
            if h!=x['sha256'] or c['job_id']!=x['jobID']:raise ValueError('Original custody join differs')
            checks.append({'pointer':str(p),'remote':x['remote'],'bytes':x['bytes'],'sha256':x['sha256'],'custodySHA256':x['custodySHA256']})
        for pattern in ['archive.tar.gz','worker.log.gz']:
            for p in base.rglob(pattern):
                if p.is_file() and not p.is_symlink() and p.stat().st_size>=100000000:candidates.append({'path':str(p),'bytes':p.stat().st_size})
    code="""import json,stat,sys
from pathlib import Path
s=json.load(sys.stdin);rows=[]
for x in s:
 p=Path(x['remote']);v=p.stat();ok=not p.is_symlink() and stat.S_ISREG(v.st_mode)
 if not ok or v.st_size!=x['bytes']:raise ValueError('Cold metadata identity differs')
 rows.append({'remote':str(p),'bytes':v.st_size,'regular':True})
print(json.dumps(rows))
"""
    metadata=remote(code,checks);root=Path(root);root.mkdir(exist_ok=False,parents=True);value={'schema':'tce-storage-audit-v1','timestampUnix':time.time(),'artifactRoots':[str(x) for x in ROOTS],'pointerChecks':checks,'remoteMetadataChecks':metadata,'remoteCheckScope':'Fresh regular-file/size checks only; historical cold objects not fully byte-rehashed in this audit. New transaction has its own full-byte checks.','candidateFiles':len(candidates),'candidateLogicalBytes':sum(x['bytes'] for x in candidates),'candidates':sorted(candidates,key=lambda x:x['path']),'minimumBytes':100000000,'localFreeBytesObserved':shutil.disk_usage(WORKSPACE).free,'auditorSHA256':sha(__file__),'scope':'Read-only transport name/size review and pointer/custody joins. Candidates are not deletion certificates or unique physical bytes.'};save(root/'result.json',value);return {'pointers':len(checks),'candidates':len(candidates),'candidateLogicalBytes':value['candidateLogicalBytes']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','execute','audit']);p.add_argument('root');p.add_argument('parents',nargs='*');a=p.parse_args();print(json.dumps(stage(a.root,a.parents) if a.action=='stage' else globals()[a.action](a.root)))
