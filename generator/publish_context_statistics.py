"""Publish authenticated context-state evidence with byte-lossless gzip streams."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess

from encoder_atoms import read,save,sha,identity


def checked_collection(root):
    root=Path(root).resolve();c=read(root/'collection.json');j=read(root/'scheduler-job.json');r=read(root/'request.json')
    if j['state']!='succeeded' or j['job_id']!=c['jobID'] or j['payload_custody']!=c['custody'] or j['config']['requestSHA256']!=r['requestSHA256']:
        raise ValueError('Collected job identity differs')
    custody=c['custody']
    if not custody.get('required') or not j.get('admission'):raise ValueError('Guarded admission required')
    for phase in ['sealed','executed']:
        if any(custody[phase].get(k)!=custody['declared'].get(k) for k in ['sha256','size_bytes']):raise ValueError('Payload custody differs')
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,item in c['inventory']['files'].items():
        p=root/'result'/n
        if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Collected result changed: '+n)
    for n,h in r['sourceHashes'].items():
        if sha(root/'payload'/n)!=h:raise ValueError('Staged source changed: '+n)
        if n.endswith('.py') and sha(Path(__file__).parent/n)!=h:raise ValueError('Current worker differs: '+n)
    if read(root/'result/report.json')['requestSHA256']!=r['requestSHA256']:raise ValueError('Report request differs')
    return c,j,r


VERIFY='''import json,hashlib,stat,sys
from pathlib import Path
spec=json.load(sys.stdin);root=Path(spec['root']);inventory={}
for item in spec['items']:
 p=Path(item['path'])
 if p.is_symlink() or not stat.S_ISREG(p.stat().st_mode) or not p.is_relative_to(root):raise ValueError('Unsafe retained path')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 if h.hexdigest()!=item['sha256']:raise ValueError('Retained bytes changed: '+str(p))
 inventory[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps({'root':str(root),'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'mode':'Authenticated SSH byte hashing only; no scientific/model computation.'}))
'''


def verify_retained(root):
    root=Path(root).resolve();c,j,r=checked_collection(root);base=j['result']['diagnostics']['cwd'];p=root/'result'
    seal=read(p/'prediction-seal.json');ref=read(p/'reference-receipt.json')
    items=[*seal['rows'],*seal['ports'],*ref['rows'],ref['projection'],seal['omega']]
    items.extend({'path':base+'/'+n,'sha256':sha(p/source)} for n,source in [
        ('predictions/prediction-seal.json','prediction-seal.json'),('response-result/reference-receipt.json','reference-receipt.json')])
    # Data stays on stdin; command text is fixed, shell-quoted code, without source-path interpolation.
    import shlex
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast',
        'python3 -c '+shlex.quote(VERIFY)],input=json.dumps({'root':base,'items':items}),text=True,capture_output=True,check=True,timeout=180)
    value=json.loads(proc.stdout);value.update(schema='tce-retained-context-arrays-v1',verifierSourceSHA256=sha(__file__),
        parentSealSHA256=sha(p/'prediction-seal.json'),jobID=j['job_id'])
    save(root/'retained-array-verification.json',value)
    return {'files':len(value['files']),'bytes':value['totalBytesVerified']}


def publish(root,destination):
    root=Path(root).resolve();c,j,r=checked_collection(root);out=Path(destination).resolve();source=root/'result'
    out.mkdir(parents=True,exist_ok=False);transforms=[]
    for p in source.iterdir():
        if not p.is_file():raise ValueError('Unexpected result directory')
        if p.suffix=='.jsonl':
            target=out/(p.name+'.gz');data=p.read_bytes();target.write_bytes(gzip.compress(data,compresslevel=9,mtime=0))
            if gzip.decompress(target.read_bytes())!=data:raise ValueError('Lossless stream round-trip differs')
            transforms.append({'source':p.name,'sourceBytes':len(data),'sourceSHA256':sha(p),'published':target.name,
                'publishedBytes':target.stat().st_size,'publishedSHA256':sha(target),'conversion':'gzip only; decompressed bytes exactly equal source, verified; no row/value/schema changes.'})
        else:shutil.copyfile(p,out/p.name)
    retained=root/'retained-array-verification.json'
    if (source/'prediction-seal.json').exists():
        v=read(retained)
        if v['parentSealSHA256']!=sha(source/'prediction-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Retained verification differs')
        shutil.copyfile(retained,out/retained.name)
    save(out/'provenance.json',{'schema':'tce-context-evidence-public-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],
        'payloadCustody':c['custody'],'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],
        'remoteResultRoot':j['result']['diagnostics']['cwd']+'/response-result','losslessStreamTransports':transforms,
        'publication':'All collected result files retained; JSONL transported as byte-lossless gzip. Large sealed heads/state/ports/reference slice remain near compute, separately hash-verified for primary assay.',
        'goalStatus':'Active. Context-state approximation and retrospective diagnostics do not establish arbitrary-prompt zero-neural images.'})
    save(out/'files.json',{str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()})
    return {'path':str(out),'files':len(list(out.iterdir())),'bytes':sum(p.stat().st_size for p in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify-retained','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(verify_retained(a.run) if a.action=='verify-retained' else publish(a.run,a.destination))
