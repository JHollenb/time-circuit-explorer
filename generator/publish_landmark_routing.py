"""Retain saved-only routing observations and explicitly oracle capacity evidence."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import shlex
import subprocess

from encoder_atoms import read,save,sha
from publish_context_statistics import checked_collection
from publish_source_context import REMOTE


def verify(root):
    from mrun.client.api import Api
    root=Path(root).resolve();c,j,r=checked_collection(root);base=j['result']['diagnostics']['cwd'];source=root/'result'
    parent=Api().json('GET','/api/jobs/'+r['parentJobID'])
    if parent['state']!='succeeded' or parent['config']['requestSHA256']!=r['parentRequestSHA256'] or not parent['payload_custody']['required']:raise ValueError('Parent differs')
    seal=read(source/'prediction-seal.json');ref=read(source/'reference-receipt.json');capacity=read(source/'capacity-receipt.json')
    items=[*seal['rows'],*seal['sourcePortHashes'],seal['asset'],*ref['rows'],*ref['parentPredictions'],ref['projection'],capacity['arrays']]
    for name in ['prediction-seal.json','capacity-receipt.json','reference-receipt.json']:items.append({'path':base+'/response-result/'+name,'sha256':sha(source/name)})
    spec={'allowedRoots':[base,parent['result']['diagnostics']['cwd'],ref['projection']['path']],'items':items}
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps(spec),text=True,capture_output=True,check=True,timeout=180)
    value=json.loads(result.stdout);value.update(schema='tce-landmark-routing-retained-bytes-v1',jobID=j['job_id'],
        predictionSealSHA256=sha(source/'prediction-seal.json'),verifierSourceSHA256=sha(__file__),
        sharedRemoteVerifierSHA256=sha(Path(__file__).parent/'publish_source_context.py'),
        mode='Authenticated SSH byte hashing only. Current/parent roots and exact inherited output-slice file; includes oracle capacity arrays, no scientific/model computation.')
    save(root/'retained-array-verification.json',value);return {'files':len(value['files']),'bytes':value['totalBytesVerified']}


def publish(root,destination):
    root=Path(root).resolve();c,j,r=checked_collection(root);source=root/'result';v=read(root/'retained-array-verification.json')
    if v['predictionSealSHA256']!=sha(source/'prediction-seal.json') or v['verifierSourceSHA256']!=sha(__file__) or v['sharedRemoteVerifierSHA256']!=sha(Path(__file__).parent/'publish_source_context.py'):raise ValueError('Verification differs')
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False);transforms=[]
    for p in source.iterdir():
        if not p.is_file():raise ValueError('Unexpected collected directory')
        if p.suffix=='.jsonl':
            data=p.read_bytes();target=out/(p.name+'.gz');target.write_bytes(gzip.compress(data,compresslevel=9,mtime=0))
            if gzip.decompress(target.read_bytes())!=data:raise ValueError('Lossless stream differs')
            transforms.append({'source':p.name,'sourceBytes':len(data),'sourceSHA256':sha(p),'published':target.name,
                'publishedBytes':target.stat().st_size,'publishedSHA256':sha(target),'conversion':'Byte-lossless gzip, exact decompression verified.'})
        else:shutil.copyfile(p,out/p.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-landmark-routing-public-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],
        'payloadCustody':c['custody'],'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],
        'losslessStreamTransports':transforms,'remoteResultRoot':j['result']['diagnostics']['cwd']+'/response-result',
        'publication':'All compact reports/raw observations preserved; full routing forecasts and target-assisted capacity weights/arrays remain near compute, hash-verified.',
        'claimBoundary':'Retrospective development bandwidth and oracle capacity diagnosis. No fresh-input validation, serving oracle weights, fullencoder or image claim.'})
    save(out/'files.json',{p.name:sha(p) for p in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(p.stat().st_size for p in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(verify(a.run) if a.action=='verify' else publish(a.run,a.destination))
