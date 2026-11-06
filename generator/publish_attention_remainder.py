"""Publish the retrospective remainder diagnostic with verified retained arrays."""
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
    root=Path(root).resolve();c,j,r=checked_collection(root);p=root/'result';base=j['result']['diagnostics']['cwd']
    parent=Api().json('GET','/api/jobs/'+r['parentJobID'])
    if parent['state']!='succeeded' or parent['config']['requestSHA256']!=r['parentRequestSHA256'] or not parent['payload_custody']['required']:raise ValueError('Complete-bank parent differs')
    source=read(p/'source.json');ref=read(p/'reference-receipt.json')
    items=[source['predictionSeal'],source['asset'],*source['predictions'],*[z['port'] for z in source['predictions']],*source['references'],*ref['rows']]
    for n in ['request.json','source.json','report.json','reference-receipt.json']:items.append({'path':base+'/response-result/'+n,'sha256':sha(p/n)})
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots':[base,parent['result']['diagnostics']['cwd']],'items':items}),text=True,capture_output=True,check=True,timeout=240)
    value=json.loads(result.stdout);value.update(schema='tce-attention-remainder-retained-bytes-v1',jobID=j['job_id'],
        parentPredictionSealSHA256=source['predictionSeal']['sha256'],verifierSourceSHA256=sha(__file__),
        sharedRemoteVerifierSHA256=sha(Path(__file__).parent/'publish_source_context.py'),
        scope='Authenticated full byte hashing of diagnostic terms/coefficients, complete bank/port/reference inputs and source asset. Shared parent bytes counted; no original checkpoint used or local tensor hydration.')
    save(root/'retained-array-verification.json',value)
    return {'files':len(value['files']),'bytes':value['totalBytesVerified']}


def publish(root,destination):
    root=Path(root).resolve();c,j,r=checked_collection(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['parentPredictionSealSHA256']!=read(p/'source.json')['predictionSeal']['sha256'] or v['verifierSourceSHA256']!=sha(__file__) or v['sharedRemoteVerifierSHA256']!=sha(Path(__file__).parent/'publish_source_context.py'):raise ValueError('Verification differs')
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False);transforms=[]
    for path in p.iterdir():
        if not path.is_file():raise ValueError('Unexpected result directory')
        if path.suffix=='.jsonl':
            data=path.read_bytes();target=out/(path.name+'.gz');target.write_bytes(gzip.compress(data,compresslevel=9,mtime=0))
            if gzip.decompress(target.read_bytes())!=data:raise ValueError('Lossless stream differs')
            transforms.append({'source':path.name,'sourceBytes':len(data),'sourceSHA256':sha(path),'published':target.name,
                'publishedBytes':target.stat().st_size,'publishedSHA256':sha(target)})
        else:shutil.copyfile(path,out/path.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-attention-remainder-public-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],
        'payloadCustody':c['custody'],'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreamTransports':transforms,
        'remoteResultRoot':j['result']['diagnostics']['cwd']+'/response-result',
        'publication':'All compact results and lossless observations retained; complete remainder/address arrays remain near compute with independent byte verification.',
        'claimBoundary':'Retrospective exact exponential-remainder and one-address controls on the already-scored complete bank. No new predictor, target-free correction or strictzero-neural image claim. No original checkpoint/projection evaluation.'})
    save(out/'files.json',{path.name:sha(path) for path in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(path.stat().st_size for path in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(verify(a.run) if a.action=='verify' else publish(a.run,a.destination))
