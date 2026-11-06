"""Publish a verified global-head bundle or its complete saved-only control report."""
import argparse
import json
from pathlib import Path
import shutil

from response_study import read,save,sha


def publish(run,destination,receiver=None):
    run,destination=Path(run).resolve(),Path(destination).resolve()
    request,custody,report=[read(run/p) for p in ['request.json','collection.json','result/report.json']]
    if request['requestSHA256']!=report['requestSHA256']:raise ValueError('Publication request mismatch')
    for name,item in custody['inventory']['files'].items():
        path=run/'result'/name
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('Collected bytes changed')
    is_head=report['schema']=='tce-global-head-report-v1'
    names=['report.json','prediction-seal.json','compile-receipt.json','reader-head.npz','fixed-head.npz','attention-cache.npz','observations.jsonl','serve.json'] if is_head else ['report.json','candidate-seal.json','probabilities.npz']
    if is_head:names += [r['path'] for r in report['serving']['rows'] if r['method']=='all']
    destination.mkdir(parents=True,exist_ok=False)
    for name in names:shutil.copyfile(run/'result'/name,destination/name)
    shutil.copyfile(run/'request.json',destination/'request.json')
    samples=[]
    derived={}
    if is_head:
        receiver=Path(receiver).resolve()
        parent=read(receiver/'collection.json')
        if parent['jobID']!=request['parent']['jobID']:raise ValueError('Wrong receiver collection')
        for packet in read(run/'result/serve.json')['packets']:
            name=Path(packet['path']).name;path=receiver/'result'/name
            if sha(path)!=packet['sha256'] or parent['inventory']['files'][name]['sha256']!=packet['sha256']:
                raise ValueError('Emitted sample changed')
            shutil.copyfile(path,destination/name)
            samples.append({'id':packet['id'],'path':name,'sha256':packet['sha256'],'shape':packet['shape']})
        save(destination/'sample-states.json',{'schema':'tce-global-head-samples-v1','parentJobID':parent['jobID'],'packets':samples,
             'scope':'First packet supplies baseline for computational controls; second is emitted midpoint state. Serving receives opaque state packets, no dose.'})
    else:
        with (destination/'observations.jsonl').open('x') as stream:
            for i,row in enumerate(report['scores']):
                stream.write(json.dumps({'step':i,**{k:row[k] for k in ['method','profile','view','region']},
                     'metrics':{k:v for k,v in row.items() if k not in ['method','profile','view','region']},
                     'sourceReportSHA256':sha(destination/'report.json')},allow_nan=False)+'\n')
        derived={'observations.jsonl':{'sha256':sha(destination/'observations.jsonl'),'rows':len(report['scores']),
                 'method':'Lossless per-score projection from the collected report; not a new model measurement.'}}
    save(destination/'provenance.json',{'schema':'tce-global-head-publication-v1','jobID':custody['jobID'],'requestSHA256':request['requestSHA256'],
         'reportSHA256':sha(destination/'report.json'),'payloadCustody':custody['custody'],'inventory':custody['inventory'],
         'derived':derived,
         'result':{k:v for k,v in custody['result'].items() if k!='diagnostics'},
         'scope':'Verified collected subset with exact banks and sample states for head serving, or exact saved-only diagnostic metadata. Omitted arrays remain hash-bound in complete remote/local collections; no general prompt/image capability.'})
    save(destination/'files.json',{p.name:sha(p) for p in sorted(destination.iterdir()) if p.is_file()})
    return {'published':str(destination),'bytes':sum(p.stat().st_size for p in destination.iterdir())}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('destination',type=Path);parser.add_argument('--receiver',type=Path)
    args=parser.parse_args();print(json.dumps(publish(args.run,args.destination,args.receiver)))
