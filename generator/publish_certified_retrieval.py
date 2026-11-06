"""Retain complete indexed retrieval evidence without local array hydration."""
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
    root=Path(root).resolve();c,j,r=checked_collection(root);p=root/'result';base=j['result']['diagnostics']['cwd'];api=Api();parents=[]
    for key in ['parent','completeBank']:
        job=api.json('GET','/api/jobs/'+r[key+'JobID'])
        if job['state']!='succeeded' or job['config']['requestSHA256']!=r[key+'RequestSHA256'] or not job['payload_custody']['required']:raise ValueError('Source ancestor differs')
        parents.append(job['result']['diagnostics']['cwd'])
    source=read(p/'source.json');seals=[read(p/n) for n in ['prediction-seal.json','fullscan-seal.json']]
    if source['completeBankRoot']!=parents[1]:raise ValueError('Geometry root differs')
    items=[source['parentSeal'],source['parentReport'],source['asset'],*source['ports'],*source['parentBanks'],*source['parentConsumers']]
    for seal in seals:items.extend(seal['rows']+seal['states'])
    for path in p.iterdir():
        if not path.is_file():raise ValueError('Unexpected result directory')
        items.append({'path':base+'/response-result/'+path.name,'sha256':sha(path)})
    value=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots':[base,*parents],'items':items}),text=True,capture_output=True,check=True,timeout=180)
    value=json.loads(value.stdout);value.update(schema='tce-certified-retrieval-retained-bytes-v1',jobID=j['job_id'],predictionSealSHA256=sha(p/'prediction-seal.json'),
        fullscanSealSHA256=sha(p/'fullscan-seal.json'),verifierSourceSHA256=sha(__file__),sharedRemoteVerifierSHA256=sha(Path(__file__).parent/'publish_source_context.py'),
        scope='Complete-byte hashing of certified/fullscan banks, causal indexes, perquery counters, source geometry/ports and inherited parent forecasts/consumers. No original checkpoint used this run, no local tensor hydration. Inherited original learned consumer remains historical assistance.')
    save(root/'retained-array-verification.json',value);return {'files':len(value['files']),'bytes':value['totalBytesVerified']}


def publish(root,destination):
    root=Path(root).resolve();c,j,r=checked_collection(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['predictionSealSHA256']!=sha(p/'prediction-seal.json') or v['fullscanSealSHA256']!=sha(p/'fullscan-seal.json') or v['verifierSourceSHA256']!=sha(__file__) or v['sharedRemoteVerifierSHA256']!=sha(Path(__file__).parent/'publish_source_context.py'):raise ValueError('Verification differs')
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False);transforms=[]
    for path in p.iterdir():
        if path.suffix=='.jsonl':
            data=path.read_bytes();target=out/(path.name+'.gz');target.write_bytes(gzip.compress(data,compresslevel=9,mtime=0))
            if gzip.decompress(target.read_bytes())!=data:raise ValueError('Lossless stream differs')
            transforms.append({'source':path.name,'sourceBytes':len(data),'sourceSHA256':sha(path),'published':target.name,'publishedBytes':target.stat().st_size,'publishedSHA256':sha(target)})
        else:shutil.copyfile(path,out/path.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-certified-retrieval-public-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],
        'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreamTransports':transforms,'remoteResultRoot':j['result']['diagnostics']['cwd']+'/response-result',
        'publication':'All compact reports/lossless observations retained; full indexes/counters/banks and inherited consumers remain near compute with independent complete byte hashes.',
        'claimBoundary':'One retrospective selection-cost assay. Source-model-derived keys/queries and fullprefix centroid arithmetic remain. No smaller/faster attention claim from coordinateproducts alone, no new originalconsumer evaluations, no fullencoder or strictzero-neural image.'})
    save(out/'files.json',{path.name:sha(path) for path in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(path.stat().st_size for path in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(verify(a.run) if a.action=='verify' else publish(a.run,a.destination))
