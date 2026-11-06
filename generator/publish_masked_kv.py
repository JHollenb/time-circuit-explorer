"""Verify retained own-state fields and publish lossless compact reports."""
import argparse
import gzip
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read, save, sha
from publish_context_statistics import checked_collection
from publish_source_context import REMOTE


def verify(root):
    root=Path(root).resolve();c,j,r=checked_collection(root,current_workers=False);p=root/'result';base=j['result']['diagnostics']['cwd']
    source=read(p/'source.json');fields=read(p/'retained-fields.json');items=[port for row in fields['rows']for port in row['ports'].values()]
    items+=source['lexicalSeal']['rows']
    spec=read(root/'payload/serve.json');items.append({'path':spec['store']+'/manifest.json','sha256':spec['manifestSHA256']})
    items += [{'path':base+'/response-result/'+f.name,'sha256':sha(f)}for f in p.iterdir()]
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots':[base,spec['store']],'items':items}),text=True,capture_output=True,check=True,timeout=120)
    value=json.loads(proc.stdout);value.update(schema='tce-masked-kv-retained-bytes-v1',jobID=j['job_id'],
         predictionSealSHA256=sha(p/'prediction-seal.json'),verifierSourceSHA256=sha(__file__),
         scope='Independent full-byte hashes of every retained candidate/control/native field, lexical source packets, compact reports and inherited atom manifest. No parameter or local tensor hydration.')
    save(root/'retained-array-verification.json',value)
    return {'files':len(value['files']),'bytes':value['totalBytesVerified']}


def publish(root,destination):
    root=Path(root).resolve();c,j,r=checked_collection(root,current_workers=False);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['predictionSealSHA256']!=sha(p/'prediction-seal.json')or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Byte verification differs')
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False);transforms=[]
    for f in p.iterdir():
        if f.suffix=='.jsonl':
            data=f.read_bytes();dest=out/(f.name+'.gz');dest.write_bytes(gzip.compress(data,mtime=0))
            if gzip.decompress(dest.read_bytes())!=data:raise ValueError('Lossless stream differs')
            transforms.append({'source':f.name,'sourceSHA256':sha(f),'sourceBytes':len(data),'published':dest.name,'publishedSHA256':sha(dest),'publishedBytes':dest.stat().st_size})
        else:shutil.copyfile(f,out/f.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-masked-kv-public-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],
         'payloadCustody':c['custody'],'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreams':transforms,
         'scope':'Two fresh raw strings from compiled E through own H1/H2; original learned tails remain. Same-H1 full/zero controls isolate formation. Native chain starts compiled E, not an independent native tokenizer/embedding. No image or zero-neural encoder claim.'})
    save(out/'files.json',{f.name:sha(f)for f in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(f.stat().st_size for f in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(verify(a.run)if a.action=='verify'else publish(a.run,a.destination)))
