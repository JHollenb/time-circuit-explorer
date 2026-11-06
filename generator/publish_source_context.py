"""Source-conditioned publication and bounded two-generation byte verification."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import shlex
import subprocess

from encoder_atoms import read,save,sha
from publish_context_statistics import checked_collection

REMOTE='''import json,hashlib,stat,sys
from pathlib import Path
s=json.load(sys.stdin);inventory={}
for item in s['items']:
 p=Path(item['path'])
 if p.is_symlink() or not stat.S_ISREG(p.stat().st_mode) or not any(p.is_relative_to(Path(r)) for r in s['allowedRoots']):raise ValueError('Unsafe retained member')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 if h.hexdigest()!=item['sha256']:raise ValueError('Retained bytes changed')
 inventory[str(p)]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps({'files':inventory,'totalBytesVerified':sum(v['bytes'] for v in inventory.values()),'allowedRoots':s['allowedRoots']}))
'''


def verify(root):
    from mrun.client.api import Api
    root=Path(root).resolve();c,j,r=checked_collection(root);base=j['result']['diagnostics']['cwd'];source=root/'result'
    parent=Api().json('GET','/api/jobs/'+r['parentJobID'])
    if parent['state']!='succeeded' or not parent['payload_custody']['required']:raise ValueError('Parent no longer authenticated')
    seal=read(source/'prediction-seal.json');ref=read(source/'reference-receipt.json')
    items=[*seal['rows'],*seal['ports'],*ref['rows'],ref['projection'],seal['omega'],seal['geometry']]
    for path,name in [('predictions/prediction-seal.json','prediction-seal.json'),('response-result/reference-receipt.json','reference-receipt.json'),('compile-receipt.json','compile-receipt.json'),('source-ids.npy','source-ids.npy')]:
        items.append({'path':base+'/'+path,'sha256':sha(source/name)})
    spec={'allowedRoots':[base,parent['result']['diagnostics']['cwd']],'items':items}
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps(spec),text=True,capture_output=True,check=True,timeout=180)
    value=json.loads(result.stdout);value.update(schema='tce-source-context-retained-bytes-v1',jobID=j['job_id'],
        predictionSealSHA256=sha(source/'prediction-seal.json'),verifierSourceSHA256=sha(__file__),
        mode='Authenticated SSH byte hashing only; inherited omega/output slice restricted to authenticated parent root; no scientific/model computation.')
    save(root/'retained-array-verification.json',value)
    return {'files':len(value['files']),'bytes':value['totalBytesVerified']}


def publish(root,destination):
    root=Path(root).resolve();c,j,r=checked_collection(root);source=root/'result';v=read(root/'retained-array-verification.json')
    if v['predictionSealSHA256']!=sha(source/'prediction-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Verification differs')
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False);transforms=[]
    for p in source.iterdir():
        if not p.is_file():raise ValueError('Unexpected result directory')
        if p.suffix=='.jsonl':
            data=p.read_bytes();target=out/(p.name+'.gz');target.write_bytes(gzip.compress(data,compresslevel=9,mtime=0))
            if gzip.decompress(target.read_bytes())!=data:raise ValueError('Lossless stream roundtrip differs')
            transforms.append({'source':p.name,'sourceBytes':len(data),'sourceSHA256':sha(p),'published':target.name,
                'publishedBytes':target.stat().st_size,'publishedSHA256':sha(target),'conversion':'Byte-lossless gzip; exact decompression verified, no row/value changes.'})
        else:shutil.copyfile(p,out/p.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-source-context-public-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],
        'payloadCustody':c['custody'],'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],
        'remoteResultRoot':j['result']['diagnostics']['cwd']+'/response-result','losslessStreamTransports':transforms,
        'publication':'All compact result files preserved, JSONL as byte-lossless gzip. Source geometry/IDs/features are public; complete sealed heads/ports/reference arrays remain near compute with independent byte verification.',
        'goalStatus':'Active; first-head approximate contextual computation only, no complete encoder or images.'})
    save(out/'files.json',{p.name:sha(p) for p in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(p.stat().st_size for p in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(verify(a.run) if a.action=='verify' else publish(a.run,a.destination))
