"""Lossless current-byte transport archival without retroactive scientific custody."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from encoder_atoms import read,save,sha,identity
from storage_terminal_snapshots import ROOTS,COLD,signature,remote,VERIFY as VERIFY_TEMPLATE


VERIFY = VERIFY_TEMPLATE.replace("print(json.dumps(", "with p.open('rb') as f:os.fsync(f.fileno())\ndirectory=os.open(p.parent,os.O_RDONLY)\ntry:os.fsync(directory)\nfinally:os.close(directory)\nprint(json.dumps(")

def stage(root,audit):
    audit=Path(audit).resolve();observed=read(audit);items=[]
    for row in observed['candidates']:
        p=Path(row['path']).resolve()
        if p.name not in ['archive.tar.gz','worker.log.gz'] or not any(p.is_relative_to(x) for x in ROOTS): raise ValueError('Unexpected local transport')
        if Path(str(p)+'.cold.json').exists(): raise ValueError('Already has restoration pointer')
        sig=signature(p);digest=sha(p)
        if sig[2]!=row['bytes'] or signature(p)!=sig: raise ValueError('Snapshot identity changed during hashing')
        metadata={str(x):sha(x) for x in p.parent.glob('*.json') if x.is_file() and not x.is_symlink()}
        items.append({'path':str(p),'bytes':sig[2],'sha256':digest,'statIdentity':sig,'neighborMetadataSHA256':metadata,
            'remote':COLD+'/'+digest[:2]+'/'+digest,'originalScientificCustody':'Not certified by this operation. Existing receipts, missing fields and outcomes remain unchanged.'})
    value={'schema':'tce-local-byte-snapshot-plan-v1','createdAtUnix':time.time(),'auditPath':str(audit),'auditSHA256':sha(audit),
        'items':items,'bytesToArchive':sum(x['bytes'] for x in items),'localFreeBytesBefore':shutil.disk_usage(ROOTS[0]).free,
        'scope':'Move exact present local transport bytes reversibly; no claim that they are complete original model artifacts or that historical scientific custody is repaired. Full cold SHA and durable restore pointer precede local unlink; all neighbor metadata and extracted science retained.'}
    value['planSHA256']=identity(value);root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',value)
    return {'files':len(items),'bytes':value['bytesToArchive']}


def execute(root):
    root=Path(root).resolve();plan=read(root/'plan.json')
    if identity({k:v for k,v in plan.items() if k!='planSHA256'})!=plan['planSHA256'] or (root/'result.json').exists(): raise ValueError('Plan changed or completed')
    rows=[]
    with (root/'events.jsonl').open('x') as events:
        def event(kind,**values):
            events.write(json.dumps({'kind':kind,'timestamp':time.time(),**values})+'\n');events.flush();os.fsync(events.fileno())
        for item in plan['items']:
            p=Path(item['path']);pointer=Path(str(p)+'.cold.json')
            def unchanged():
                return signature(p)==item['statIdentity'] and all(sha(Path(n))==h for n,h in item['neighborMetadataSHA256'].items())
            if not unchanged() or sha(p)!=item['sha256']: raise ValueError('Fresh source/metadata differs')
            temporary=item['remote']+'.partial-local-snapshot-'+str(os.getpid())
            remote("import json,sys;from pathlib import Path;s=json.load(sys.stdin);p=Path(s['path']);r=Path(s['root']);assert p.is_relative_to(r);p.parent.mkdir(parents=True,exist_ok=True);print('{}')",{'path':temporary,'root':COLD})
            subprocess.run(['rsync','-a','--protect-args',str(p),'beast:'+temporary],check=True,capture_output=True,timeout=180)
            verified=remote(VERIFY,{'path':temporary,'destination':item['remote'],'root':COLD,'sha256':item['sha256'],'bytes':item['bytes']})
            event('remote-file-and-directory-fsynced',remote=item['remote'],sha256=item['sha256'])
            value={'schema':'tce-cold-local-snapshot-pointer-v1',**item,'planSHA256':plan['planSHA256'],'remoteHost':'beast','remoteVerification':verified,
                'restoreCommand':['rsync','-a','--protect-args','beast:'+item['remote'],str(p)]}
            with pointer.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
            directory=os.open(pointer.parent,os.O_RDONLY)
            try:os.fsync(directory)
            finally:os.close(directory)
            if read(pointer)!=value or not unchanged() or sha(p)!=item['sha256']: raise ValueError('Durable pointer/fresh local byte identity differs')
            event('verified-before-unlink',path=str(p),sha256=item['sha256'],bytes=item['bytes'],remote=item['remote']);p.unlink();event('local-transport-unlinked',path=str(p));rows.append(value)
    value={'schema':'tce-local-byte-snapshot-result-v1','planSHA256':plan['planSHA256'],'files':len(rows),'archivedBytes':sum(x['bytes'] for x in rows),
        'localFreeBytesBefore':plan['localFreeBytesBefore'],'localFreeBytesAfter':shutil.disk_usage(ROOTS[0]).free,'rows':rows,'executorSHA256':sha(__file__),'remoteDurabilityVerifiedBeforeUnlink':True,'remoteVerifierSHA256':hashlib.sha256(VERIFY.encode()).hexdigest(),
        'scope':'Remote file and containing directory fsynced; current local and remote full-byte SHA verified before unlink. Exact cold bytes/restoration pointers retained. No retroactive scientific attestation; all original metadata/extracted evidence/outcomes unchanged. Shared-volume free-space change is not exclusive physical attribution.'}
    save(root/'result.json',value);return {'files':len(rows),'bytes':value['archivedBytes']}


def audit(root):
    rows=[]
    for base in ROOTS:
        for p in base.rglob('*.cold.json'):
            x=read(p)
            if x.get('schema')!='tce-cold-local-snapshot-pointer-v1':continue
            if Path(x['path']).exists() or any(sha(Path(n))!=h for n,h in x['neighborMetadataSHA256'].items()): raise ValueError('Local pointer/neighbor metadata differs')
            # Full remote hash is deliberate for these newly archived, uncertified-origin snapshots.
            witness=remote(VERIFY,{'path':x['remote'],'root':COLD,'sha256':x['sha256'],'bytes':x['bytes']})
            rows.append({'pointer':str(p),'pointerSHA256':sha(p),'remoteFullByteWitness':witness})
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    value={'schema':'tce-local-byte-snapshot-audit-v1','timestampUnix':time.time(),'pointers':len(rows),'bytes':sum(x['remoteFullByteWitness']['bytes'] for x in rows),'rows':rows,'auditorSHA256':sha(__file__),
        'scope':'New current-byte snapshot pointer/neighbor metadata/full cold SHA checks. Distinct from historical scientific custody or older terminal-transport pointer metadata audit.'}
    save(root/'result.json',value);return {'pointers':len(rows),'bytes':value['bytes']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','execute','audit']);p.add_argument('root');p.add_argument('--audit');a=p.parse_args()
    print(json.dumps(stage(a.root,a.audit) if a.action=='stage' else globals()[a.action](a.root)))
