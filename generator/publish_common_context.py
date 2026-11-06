"""Compact transductive-bank publication; complete kernel/heads stay near compute."""
import argparse
import gzip
import json
from pathlib import Path
import shlex
import shutil
import subprocess

from encoder_atoms import read,save,sha
from publish_context_statistics import checked_collection
from publish_source_context import REMOTE


def verify(root):
    from mrun.client.api import Api
    root=Path(root).resolve();c,j,r=checked_collection(root);base=j['result']['diagnostics']['cwd'];p=root/'result'
    parent=Api().json('GET','/api/jobs/'+r['parentJobID'])
    if parent['state']!='succeeded' or parent['config']['requestSHA256']!=r['parentRequestSHA256']:raise ValueError('Parent differs')
    seal=read(p/'prediction-seal.json');ref=read(p/'reference-receipt.json');compiler=read(p/'compiler.json')
    items=[*seal['rows'],seal['factors'],seal['packets'],compiler['commonBank'],*ref['rows'],ref['projection']]
    for remote,local in [('predictions/prediction-seal.json','prediction-seal.json'),('response-result/reference-receipt.json','reference-receipt.json'),('response-result/compiler.json','compiler.json')]:
        items.append({'path':base+'/'+remote,'sha256':sha(p/local)})
    spec={'allowedRoots':[base,parent['result']['diagnostics']['cwd'],ref['projection']['path']],'items':items}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps(spec),text=True,capture_output=True,check=True,timeout=180)
    value=json.loads(proc.stdout);value.update(schema='tce-common-context-retained-bytes-v1',jobID=j['job_id'],
        predictionSealSHA256=sha(p/'prediction-seal.json'),verifierSourceSHA256=sha(__file__),
        sharedRemoteVerifierSHA256=sha(Path(__file__).parent/'publish_source_context.py'),
        mode='Authenticated SSH byte hashing only. Current/parent roots and one exact hash-bound ancestor output-slice file; no scientific/model computation.')
    save(root/'retained-array-verification.json',value)
    return {'files':len(value['files']),'bytes':value['totalBytesVerified']}


def publish(root,destination):
    root=Path(root).resolve();c,j,r=checked_collection(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['predictionSealSHA256']!=sha(p/'prediction-seal.json') or v['verifierSourceSHA256']!=sha(__file__) or v['sharedRemoteVerifierSHA256']!=sha(Path(__file__).parent/'publish_source_context.py'):raise ValueError('Retained verification differs')
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False);transforms=[]
    for source in p.iterdir():
        if not source.is_file():raise ValueError('Unexpected result directory')
        if source.suffix=='.jsonl':
            raw=source.read_bytes();target=out/(source.name+'.gz');target.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless roundtrip differs')
            transforms.append({'source':source.name,'sourceBytes':len(raw),'sourceSHA256':sha(source),'published':target.name,
                'publishedBytes':target.stat().st_size,'publishedSHA256':sha(target),'conversion':'Byte-lossless gzip; exact decompression verified.'})
        else:shutil.copyfile(source,out/source.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-common-context-public-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],
        'payloadCustody':c['custody'],'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],
        'remoteResultRoot':j['result']['diagnostics']['cwd']+'/response-result','losslessStreamTransports':transforms,
        'publication':'All compact collected results preserved. Factor banks and occurrence packets public; full unmasked kernel/atom bank and sealed heads remain near compute, hash-verified.',
        'claimBoundary':'All eight prompts fit the factors. No unseen-query feature rule, raw-prompt serving, full encoder or zero-neural image claim.'})
    save(out/'files.json',{z.name:sha(z) for z in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(z.stat().st_size for z in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(verify(a.run) if a.action=='verify' else publish(a.run,a.destination))
