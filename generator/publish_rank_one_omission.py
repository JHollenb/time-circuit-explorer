"""Authenticate retained diagnostic arrays and publish lossless compact evidence."""
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
    root=Path(root).resolve();c,j,r=checked_collection(root);p=root/'result';base=j['result']['diagnostics']['cwd'];source=read(p/'source.json');seal=read(p/'analysis-seal.json');api=Api();parents=[]
    for key in ['parent','fullQuadratic']:
        job=api.json('GET','/api/jobs/'+r[key+'JobID'])
        if job['state']!='succeeded' or job['config']['requestSHA256']!=r[key+'RequestSHA256'] or not job['payload_custody']['required']:raise ValueError('Ancestor differs')
        parents.append(job['result']['diagnostics']['cwd'])
    if source['roots'][:2]!=parents:raise ValueError('Ancestor roots differ')
    items=[source['parentSeal'],source['asset'],source['basis'],*source['ports'],*source['rankBanks'],*source['parentBanks'],*source['parentControls'],*seal['rows']]
    items += [{'path':base+'/response-result/'+path.name,'sha256':sha(path)} for path in p.iterdir()]
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],input=json.dumps({'allowedRoots':[base,*source['roots']],'items':items}),text=True,capture_output=True,check=True,timeout=180)
    v=json.loads(proc.stdout);v.update(schema='tce-rank-one-omission-retained-bytes-v1',jobID=j['job_id'],analysisSealSHA256=sha(p/'analysis-seal.json'),verifierSourceSHA256=sha(__file__),sharedRemoteVerifierSHA256=sha(Path(__file__).parent/'publish_source_context.py'),scope='Complete byte hashes of diagnostic banks/signedvectors/scalars and inherited ports/basis/geometry/rank1/fullquadratic/exactcontrol banks. No original checkpoint read or Mac tensor hydration.');save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified']}


def publish(root,destination):
    root=Path(root).resolve();c,j,r=checked_collection(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['analysisSealSHA256']!=sha(p/'analysis-seal.json') or v['verifierSourceSHA256']!=sha(__file__) or v['sharedRemoteVerifierSHA256']!=sha(Path(__file__).parent/'publish_source_context.py'):raise ValueError('Byte verification differs')
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False);transforms=[]
    for path in p.iterdir():
        if path.suffix=='.jsonl' or path.name=='scalar-diagnostics.json':
            data=path.read_bytes();target=out/(path.name+'.gz');target.write_bytes(gzip.compress(data,compresslevel=9,mtime=0))
            if gzip.decompress(target.read_bytes())!=data:raise ValueError('Lossless round-trip differs')
            transforms.append({'source':path.name,'sourceSHA256':sha(path),'sourceBytes':len(data),'published':target.name,'publishedSHA256':sha(target),'publishedBytes':target.stat().st_size,'conversion':'Byte-lossless gzip, exact round-trip checked.'})
        else:shutil.copyfile(path,out/path.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json');save(out/'provenance.json',{'schema':'tce-rank-one-omission-public-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreamTransports':transforms,
        'publication':'All compact reports/observations retained; complete canonicalbanks/signedvectors/scalars remain nearcompute with fullbyte checks. Retrospective analysis, not serving or heldout prediction.',
        'claimBoundary':'Frozen rank1 omission diagnosis; zero new original projections/consumers/neural calls. Does not establish a new representation, laterstate/encoder/image capability.'});save(out/'files.json',{path.name:sha(path) for path in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(path.stat().st_size for path in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args();print(verify(a.run) if a.action=='verify' else publish(a.run,a.destination))
