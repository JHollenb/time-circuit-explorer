"""Authenticated publication for transported heads and explicit reference consumers."""
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


def verify(root,mode):
    from mrun.client.api import Api
    root=Path(root).resolve();c,j,r=checked_collection(root);base=j['result']['diagnostics']['cwd'];p=root/'result'
    parent=Api().json('GET','/api/jobs/'+r['parentJobID'])
    if parent['state']!='succeeded' or parent['config']['requestSHA256']!=r['parentRequestSHA256'] or not parent['payload_custody']['required']:raise ValueError('Parent differs')
    parent_base=parent['result']['diagnostics']['cwd'];seal=read(p/'prediction-seal.json');ref=read(p/'reference-receipt.json');source=read(p/'source-receipt.json')
    if mode=='head':
        items=[*seal['rows'],*seal['ports'],seal['transportAsset'],seal['interfaces'],*ref['rows'],ref['projection'],source['landmarkAsset'],
            {'path':parent_base+'/response-result/compile-receipt.json','sha256':source['compiledSourceReceiptSHA256']},
            {'path':base+'/predictions/prediction-seal.json','sha256':sha(p/'prediction-seal.json')}]
        allowed=[base,parent_base,ref['projection']['path']]
    elif mode=='consumer':
        model=source['parameters']['source'];weight=model['weightFile'];weight_path=str(Path(model['modelRoot'])/weight['path']);config_path=model['modelRoot']+'/text_encoder/config.json'
        items=[*seal['rows'],*seal['ports'],seal['transportAsset'],seal['interfaces'],*ref['rows'],*[z['context'] for z in ref['rows']],
            source['predictionSeal'],{'path':weight_path,'sha256':weight['sha256']},
            {'path':config_path,'sha256':model['assets']['text_encoder/config.json']}]
        allowed=[base,parent_base,weight_path,config_path]
    else:raise ValueError('Unknown publication mode')
    for name in ['reference-receipt.json','source-receipt.json','prediction-seal.json']:items.append({'path':base+'/response-result/'+name,'sha256':sha(p/name)})
    spec={'allowedRoots':allowed,'items':items}
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps(spec),text=True,capture_output=True,check=True,timeout=240)
    value=json.loads(result.stdout);value.update(schema='tce-query-transport-retained-bytes-v1',jobID=j['job_id'],mode=mode,
        predictionSealSHA256=sha(p/'prediction-seal.json'),verifierSourceSHA256=sha(__file__),sharedRemoteVerifierSHA256=sha(Path(__file__).parent/'publish_source_context.py'),
        verificationScope='Authenticated SSH full byte hashing, no scientific/model computation. Consumer mode includes original checkpoint/config as declared reference assistance; no local hydration.')
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
            transforms.append({'source':path.name,'sourceBytes':len(data),'sourceSHA256':sha(path),'published':target.name,
                'publishedBytes':target.stat().st_size,'publishedSHA256':sha(target),'conversion':'Byte-lossless gzip; exact decompression verified.'})
        else:shutil.copyfile(path,out/path.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-query-transport-public-v1','mode':v['mode'],'jobID':j['job_id'],'requestSHA256':r['requestSHA256'],
        'payloadCustody':c['custody'],'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreamTransports':transforms,
        'remoteResultRoot':j['result']['diagnostics']['cwd']+'/response-result',
        'publication':'Every compact result preserved; complete state/head/reference-consumer arrays remain near compute with independent byte checks.',
        'claimBoundary':'Head mode predicts one contextual read from rawtext. Consumer mode evaluates original learned equations with other31heads supplied as references. Neither is complete conditioning or strictzero-neural image generation.'})
    save(out/'files.json',{path.name:sha(path) for path in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(path.stat().st_size for path in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');p.add_argument('--mode',choices=['head','consumer']);a=p.parse_args()
    print(verify(a.run,a.mode) if a.action=='verify' else publish(a.run,a.destination))
