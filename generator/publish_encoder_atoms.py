"""Publish compact immutable evidence; keep the multi-GB lexical table near compute."""
import argparse
from pathlib import Path
import shutil

from encoder_atoms import read,save,sha,identity


def publish(root,destination):
    root=Path(root).resolve();out=Path(destination).resolve();source=root/'result'
    collection=read(root/'collection.json');job=read(root/'scheduler-job.json');request=read(root/'request.json')
    if (job['state']!='succeeded' or job['job_id']!=collection['jobID'] or job['payload_custody']!=collection['custody']
        or job['config']['requestSHA256']!=request['requestSHA256']):raise ValueError('Collection/scheduler identity differs')
    custody=collection['custody']
    if custody.get('required') is not True or not job.get('admission'):raise ValueError('Guarded admission required')
    for phase in ['sealed','executed']:
        if any(custody[phase].get(k)!=custody['declared'].get(k) for k in ['sha256','size_bytes']):raise ValueError('Payload custody differs')
    if identity({k:v for k,v in request.items() if k!='requestSHA256'})!=request['requestSHA256']:raise ValueError('Request changed')
    for name,item in collection['inventory']['files'].items():
        path=source/name
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('Collected result changed')
    for name in ['encoder_atoms.py','encoder_atoms_native.py','encoder_atoms_study.py']:
        if sha(Path(__file__).parent/name)!=request['sourceHashes'][name]:raise ValueError('Current worker source differs from full run: '+name)
    report=read(source/'report.json')
    if report['requestSHA256']!=request['requestSHA256']:raise ValueError('Report differs')
    retained=read(root/'retained-store-verification.json')
    if retained['manifestSHA256']!=report['manifestSHA256'] or retained['verifierSourceSHA256']!=sha(Path(__file__).parent/'verify_encoder_store.py'):
        raise ValueError('Retained store verification differs')
    out.mkdir(parents=True,exist_ok=False)
    names=['report.json','manifest.json','compile-receipt.json','prediction-seal.json','reference-receipt.json',
           'request.json','raw-prompts.json','observations.jsonl']
    for name in names:shutil.copyfile(source/name,out/name)
    shutil.copyfile(root/'retained-store-verification.json',out/'retained-store-verification.json')
    save(out/'provenance.json',{'schema':'tce-raw-prompt-atoms-public-v1','jobID':job['job_id'],
         'requestSHA256':request['requestSHA256'],'payloadCustody':custody,'schedulerResult':collection['result'],
         'completeCollectedInventory':collection['inventory'],'remoteResultRoot':job['result']['diagnostics']['cwd']+'/response-result',
         'retainedNearCompute':'atom-store/, prefix-predictions/, prefix-references/ beside response-result; exact table shards/rows/indices are bound by manifest, complete forecasts and references by their seals/receipts.',
         'publication':'Unmodified compact result files. No learned full-vocabulary table or complete prefix arrays copied into Git.',
         'goalStatus':'Active. Lexical operand assembly is not arbitrary-prompt image generation or a representation-independent absence of learned computation.'})
    save(out/'files.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    return {'path':str(out),'files':len(list(out.iterdir())),'bytes':sum(p.stat().st_size for p in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('destination');a=p.parse_args();print(publish(a.run,a.destination))
