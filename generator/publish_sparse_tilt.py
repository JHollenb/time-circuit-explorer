"""Verified retained-byte publication for sparse heap/tail experiments."""
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
    if parent['state']!='succeeded' or parent['config']['requestSHA256']!=r['parentRequestSHA256'] or not parent['payload_custody']['required']:raise ValueError('Parent differs')
    source=read(p/'source.json');seal=read(p/'prediction-seal.json');ref=read(p/'reference-receipt.json');model=ref['parameters']['source'];weight=model['weightFile']
    wp=str(Path(model['modelRoot'])/weight['path']);cp=model['modelRoot']+'/text_encoder/config.json'
    items=[source['predictionSeal'],source['asset'],*source['oldPredictions'],*[x['port'] for x in source['oldPredictions']],*source['references'],
        *seal['rows'],*seal['states'],*ref['rows'],{'path':wp,'sha256':weight['sha256']},{'path':cp,'sha256':model['assets']['text_encoder/config.json']}]
    for n in ['request.json','source.json','prediction-seal.json','report.json','reference-receipt.json','support-diagnostics.json']:
        items.append({'path':base+'/response-result/'+n,'sha256':sha(p/n)})
    value=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots':[base,parent['result']['diagnostics']['cwd'],wp,cp],'items':items}),text=True,capture_output=True,check=True,timeout=240)
    value=json.loads(value.stdout);value.update(schema='tce-sparse-joint-tilt-retained-bytes-v1',jobID=j['job_id'],predictionSealSHA256=sha(p/'prediction-seal.json'),
        verifierSourceSHA256=sha(__file__),sharedRemoteVerifierSHA256=sha(Path(__file__).parent/'publish_source_context.py'),
        scope='Authenticated full-byte hashing of all candidate banks/final states, full-query-score oracle arrays, original-consumer outputs and parent operands. Includes original checkpoint and overlapping parents; no local tensor hydration.')
    save(root/'retained-array-verification.json',value);return {'files':len(value['files']),'bytes':value['totalBytesVerified']}


def publish(root,destination):
    root=Path(root).resolve();c,j,r=checked_collection(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['predictionSealSHA256']!=sha(p/'prediction-seal.json') or v['verifierSourceSHA256']!=sha(__file__) or v['sharedRemoteVerifierSHA256']!=sha(Path(__file__).parent/'publish_source_context.py'):raise ValueError('Verification differs')
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False);transforms=[]
    for path in p.iterdir():
        if not path.is_file():raise ValueError('Unexpected result directory')
        if path.suffix=='.jsonl':
            data=path.read_bytes();target=out/(path.name+'.gz');target.write_bytes(gzip.compress(data,compresslevel=9,mtime=0))
            if gzip.decompress(target.read_bytes())!=data:raise ValueError('Lossless stream differs')
            transforms.append({'source':path.name,'sourceBytes':len(data),'sourceSHA256':sha(path),'published':target.name,'publishedBytes':target.stat().st_size,'publishedSHA256':sha(target)})
        else:shutil.copyfile(path,out/path.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-sparse-joint-tilt-public-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],
        'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreamTransports':transforms,'remoteResultRoot':j['result']['diagnostics']['cwd']+'/response-result',
        'publication':'All compact reports and lossless consumer observations retained; full states/support/correction/consumer arrays stay near compute with independent byte checks.',
        'claimBoundary':'Retrospective serving-eligible sparse law on saved ports, separate current full-query-score oracle and original learned fullblock consumer. Not freshrawtext validation, fullencoder or strictzero-neural images.'})
    save(out/'files.json',{path.name:sha(path) for path in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(path.stat().st_size for path in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(verify(a.run) if a.action=='verify' else publish(a.run,a.destination))
