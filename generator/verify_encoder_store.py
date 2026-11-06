"""Authenticated, bounded byte-I/O check of retained atoms and operand artifacts.

No model or scientific computation. Hash large files remotely rather than copying them.
"""
import argparse
from pathlib import Path
import subprocess

from encoder_atoms import read,save,sha

CODE='''import hashlib,json,stat
from pathlib import Path
root=Path(ROOT)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
manifest=root/'atom-store/manifest.json'
if digest(manifest)!=MANIFEST:raise ValueError('Retained manifest changed')
m=json.loads(manifest.read_text());items=[m[k] for k in ['coverage','rowHashes','rope','tokenizer','template']]+m['shards']
inventory={}
for item in items:
 p=root/'atom-store'/item['path']
 if p.parent!=root/'atom-store' or p.is_symlink() or not stat.S_ISREG(p.stat().st_mode):raise ValueError('Unsafe store member')
 size=p.stat().st_size
 if size!=item['bytes'] or digest(p)!=item['sha256']:raise ValueError('Store member changed: '+p.name)
 inventory['atom-store/'+p.name]={'bytes':size,'sha256':item['sha256']}
for directory,receipt in [('prefix-predictions','prediction-seal.json'),('prefix-references','reference-receipt.json')]:
 p=root/directory/receipt
 if digest(p)!=RECEIPTS[directory]:raise ValueError('Operand receipt changed')
 for row in json.loads(p.read_text())['rows']:
  p=Path(row['path'])
  if p.parent!=root/directory or p.is_symlink() or not stat.S_ISREG(p.stat().st_mode):raise ValueError('Unsafe operand member')
  if digest(p)!=row['sha256']:raise ValueError('Operand changed')
  inventory[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':row['sha256']}
print(json.dumps({'schema':'tce-retained-encoder-store-byte-verification-v1','root':str(root),'manifestSHA256':MANIFEST,
 'coverageCount':m['coverageCount'],'files':inventory,'totalBytesVerified':sum(v['bytes'] for v in inventory.values()),
 'mode':'Authenticated SSH byte hashing only; no model execution. Receipt and manifest identities originate in collected guarded job result.'},sort_keys=True))
'''


def verify(root):
    import json
    root=Path(root).resolve();job=read(root/'scheduler-job.json');result=root/'result';report=read(result/'report.json')
    code=CODE.replace('ROOT',repr(job['result']['diagnostics']['cwd'])).replace('MANIFEST',repr(report['manifestSHA256']))
    code=code.replace('RECEIPTS',repr({'prefix-predictions':sha(result/'prediction-seal.json'),'prefix-references':sha(result/'reference-receipt.json')}))
    raw=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3','-'],input=code,text=True,capture_output=True,check=True,timeout=180)
    value=json.loads(raw.stdout);value['verifierSourceSHA256']=sha(__file__)
    save(root/'retained-store-verification.json',value)
    return {k:value[k] for k in ['root','coverageCount','totalBytesVerified']}|{'files':len(value['files'])}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();print(verify(a.run))
