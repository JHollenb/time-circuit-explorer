"""Authenticate retained source/state bytes and publish compact rank-one evidence."""
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
    source=read(p/'source.json');seal=read(p/'prediction-seal.json');ref=read(p/'reference-receipt.json');compiled=read(p/'compile-receipt.json');model=ref['parameters']['source'];weight=model['weightFile'];wp=str(Path(model['modelRoot'])/weight['path']);cp=model['modelRoot']+'/text_encoder/config.json'
    if source['completeBankRoot']!=parents[1]:raise ValueError('Geometry root differs')
    items=[source['parentSeal'],source['asset'],*source['ports'],*source['parentBanks'],*source['parentControls'],*source['parentConsumers'],seal['basis'],*compiled['sourceFiles'],*seal['rows'],*seal['states'],*ref['rows'],
        {'path':base+'/compiled/basis-diagnostics.npz','sha256':compiled['diagnosticsSHA256']},{'path':base+'/compiled/compile-receipt.json','sha256':sha(p/'compile-receipt.json')},
        {'path':wp,'sha256':weight['sha256']},{'path':cp,'sha256':model['assets']['text_encoder/config.json']}]
    for path in p.iterdir():
        if not path.is_file():raise ValueError('Unexpected result directory')
        items.append({'path':base+'/response-result/'+path.name,'sha256':sha(path)})
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],input=json.dumps({'allowedRoots':[base,*parents,source['store'],wp,cp],'items':items}),text=True,capture_output=True,check=True,timeout=240)
    v=json.loads(proc.stdout);v.update(schema='tce-rank-one-tail-retained-bytes-v1',jobID=j['job_id'],predictionSealSHA256=sha(p/'prediction-seal.json'),verifierSourceSHA256=sha(__file__),sharedRemoteVerifierSHA256=sha(Path(__file__).parent/'publish_source_context.py'),
        scope='Complete byte hashes of source geometry/touched atom shards, compiled basis/source diagnostics, streaming banks/final states, directrank1 references, inherited controls/consumers, new consumers and original checkpoint. Overlapping ancestor files, not unique servingstate; no Mac tensor hydration.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified']}


def publish(root,destination):
    root=Path(root).resolve();c,j,r=checked_collection(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['predictionSealSHA256']!=sha(p/'prediction-seal.json') or v['verifierSourceSHA256']!=sha(__file__) or v['sharedRemoteVerifierSHA256']!=sha(Path(__file__).parent/'publish_source_context.py'):raise ValueError('Verification differs')
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False);transforms=[]
    for path in p.iterdir():
        if path.suffix=='.jsonl' or path.name=='state-diagnostics.json':
            data=path.read_bytes();target=out/(path.name+'.gz');target.write_bytes(gzip.compress(data,compresslevel=9,mtime=0))
            if gzip.decompress(target.read_bytes())!=data:raise ValueError('Lossless stream differs')
            transforms.append({'source':path.name,'sourceBytes':len(data),'sourceSHA256':sha(path),'published':target.name,'publishedBytes':target.stat().st_size,'publishedSHA256':sha(target),'conversion':'Byte-lossless gzip only; round-trip checked.'})
        else:shutil.copyfile(path,out/path.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-rank-one-tail-public-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreamTransports':transforms,
        'remoteResultRoot':j['result']['diagnostics']['cwd']+'/response-result','publication':'All compact reports/observations retained, bulky scalar descriptions byte-lossless gzip. Full basis/sourcegeometry/streaming states/banks/consumers remain near compute with complete byte checks.',
        'claimBoundary':'Retrospective source-defined rank1 streaming mixed-moment assay. Full supplied learned operands/currentquery KV selection and original receiving computations remain paid. No freshrawprompt/fullencoder/zero-neural image claim.'})
    save(out/'files.json',{path.name:sha(path) for path in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(path.stat().st_size for path in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args();print(verify(a.run) if a.action=='verify' else publish(a.run,a.destination))
