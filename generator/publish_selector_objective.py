"""Authenticate complete retained selector/correction arrays before publication."""
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
    root=Path(root).resolve();c,j,r=checked_collection(root);p=root/'result';base=j['result']['diagnostics']['cwd'];api=Api();parents={}
    for key in ['parent','completeBank','frozenSupportParent']:
        job=api.json('GET','/api/jobs/'+r[key+'JobID'])
        if job['state']!='succeeded' or job['config']['requestSHA256']!=r[key+'RequestSHA256'] or not job['payload_custody']['required']:raise ValueError('Source ancestor differs')
        parents[key]=job['result']['diagnostics']['cwd']
    source=read(p/'source.json');seal=read(p/'prediction-seal.json');ref=read(p/'reference-receipt.json');model=ref['parameters']['source'];weight=model['weightFile']
    wp=str(Path(model['modelRoot'])/weight['path']);cp=model['modelRoot']+'/text_encoder/config.json'
    if source['completeBankRoot']!=parents['completeBank']:raise ValueError('Complete-bank root differs')
    items=[source['parentSeal'],source['asset'],*source['ports'],*source['parentBanks'],*source['parentConsumers'],*source['parentOracle'],*source['references'],
        *seal['rows'],*ref['rows'],{'path':wp,'sha256':weight['sha256']},{'path':cp,'sha256':model['assets']['text_encoder/config.json']}]
    for path in p.iterdir():
        if not path.is_file():raise ValueError('Unexpected result directory')
        items.append({'path':base+'/response-result/'+path.name,'sha256':sha(path)})
    value=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots':[base,*parents.values(),wp,cp],'items':items}),text=True,capture_output=True,check=True,timeout=240)
    value=json.loads(value.stdout);value.update(schema='tce-selector-objective-retained-bytes-v1',jobID=j['job_id'],predictionSealSHA256=sha(p/'prediction-seal.json'),
        verifierSourceSHA256=sha(__file__),sharedRemoteVerifierSHA256=sha(Path(__file__).parent/'publish_source_context.py'),
        scope='Complete-byte hashing of new supports/raw/rounded banks, signed numerator vectors/log mass/partition diagnostics, exact references and original consumers, historical supports/banks/consumers and inherited checkpoint. Overlapping inherited bytes, not unique new artifacts; no local tensor hydration.')
    save(root/'retained-array-verification.json',value);return {'files':len(value['files']),'bytes':value['totalBytesVerified']}


def publish(root,destination):
    root=Path(root).resolve();c,j,r=checked_collection(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['predictionSealSHA256']!=sha(p/'prediction-seal.json') or v['verifierSourceSHA256']!=sha(__file__) or v['sharedRemoteVerifierSHA256']!=sha(Path(__file__).parent/'publish_source_context.py'):raise ValueError('Verification differs')
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False);transforms=[]
    for path in p.iterdir():
        if not path.is_file():raise ValueError('Unexpected result directory')
        if path.suffix=='.jsonl' or path.name=='signed-diagnostics.json':
            data=path.read_bytes();target=out/(path.name+'.gz');target.write_bytes(gzip.compress(data,compresslevel=9,mtime=0))
            if gzip.decompress(target.read_bytes())!=data:raise ValueError('Lossless stream differs')
            transforms.append({'source':path.name,'sourceBytes':len(data),'sourceSHA256':sha(path),'published':target.name,'publishedBytes':target.stat().st_size,'publishedSHA256':sha(target)})
        else:shutil.copyfile(path,out/path.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-selector-objective-public-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],
        'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreamTransports':transforms,'remoteResultRoot':j['result']['diagnostics']['cwd']+'/response-result',
        'publication':'All compact reports/lossless observations retained; full new and historical support/head/diagnostic/consumer arrays stay near compute with independent complete byte checks.',
        'claimBoundary':'Retrospective complete causal-prefix selector diagnostic, not compact ownstate serving. Original learned receiving consumer offline; inherited correctionmass oracle indices unchanged. No freshrawprompt/fullencoder or strictzero-neural image.'})
    save(out/'files.json',{path.name:sha(path) for path in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(path.stat().st_size for path in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(verify(a.run) if a.action=='verify' else publish(a.run,a.destination))
