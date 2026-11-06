"""Take a small public custody copy of the two saved-only diagnostics."""
import argparse
import json
from pathlib import Path
import shutil

from response_study import read,save,sha

def publish(closure,span,source,destination):
    closure=Path(closure).resolve();span=Path(span).resolve();source=Path(source).resolve();out=Path(destination).resolve()
    runs=[]
    for run in [closure,span]:
        request=read(run/'request.json');report=read(run/'result/report.json');custody=read(run/'collection.json')
        if report['requestSHA256']!=request['requestSHA256']:raise ValueError('Request binding differs')
        inventory=custody['inventory']['files']
        for name,item in inventory.items():
            if sha(run/'result'/name)!=item['sha256']:raise ValueError('Collected result changed: '+name)
        runs.append({'jobID':custody['jobID'],'requestSHA256':request['requestSHA256'],'reportSHA256':sha(run/'result/report.json'),
                     'payloadCustody':custody['custody'],'result':{k:v for k,v in custody['result'].items() if k!='diagnostics'},'inventory':custody['inventory']})
    out.mkdir(parents=True,exist_ok=False)
    for run,prefix in [(closure,''),(span,'span-')]:
        shutil.copyfile(run/'result/report.json',out/(prefix+'report.json'));shutil.copyfile(run/'request.json',out/(prefix+'request.json'))
    for name in ['prediction-seal.json']:
        shutil.copyfile(closure/'result'/name,out/name)
    for name in ['calibration.json','evaluation.json']:shutil.copyfile(closure/'payload'/name,out/name)
    seal=read(out/'prediction-seal.json');images=out/'images';images.mkdir()
    for row in seal['rows']:
        if 'png' in row:
            path=closure/'result'/row['png']
            if sha(path)!=row['pngSHA256']:raise ValueError('Sealed PNG changed')
            shutil.copyfile(path,images/row['png'])
    cal=read(out/'calibration.json');evaluation=read(out/'evaluation.json')
    refs=next(x for x in cal['rows'] if x['id']=='rgb')['calibration']+[next(x for x in evaluation['rows'] if x['id']=='rgb')['target']]
    for ref in refs:
        path=source/Path(ref['path']).name
        if sha(path)!=ref['sha256']:raise ValueError('Native RGB differs')
        shutil.copyfile(path,images/path.name)
    save(out/'provenance.json',{'schema':'tce-prompt-closure-publication-v1','runs':runs,
        'scope':'Authenticated collected subsets, original bytes preserved. Source native arrays remain hash-bound on Beast. Predictor seal contains concrete full-array hashes, scorer reconstructs them before comparing. Not an own-state rollout or general generator.'})
    rows=read(out/'report.json')['scores']['rows']
    with (out/'observations.jsonl').open('x') as f:
        for i,row in enumerate(rows):
            metrics={k:v for k,v in row.items() if isinstance(v,(int,float)) or v is None}
            f.write(json.dumps({'step':i,'boundary':row['id'],'method':row['method'],'metrics':metrics,'reportSHA256':runs[0]['reportSHA256']},allow_nan=False)+'\n')
    with (out/'observations-span.jsonl').open('x') as f:
        for i,row in enumerate(read(out/'span-report.json')['rows']):
            metrics={k:v for k,v in row.items() if isinstance(v,(int,float)) or v is None}
            f.write(json.dumps({'step':i,'boundary':row['id'],'targetAssisted':True,'metrics':metrics,'reportSHA256':runs[1]['reportSHA256']},allow_nan=False)+'\n')
    save(out/'files.json',{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps({'published':str(out),'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('closure',type=Path);p.add_argument('span',type=Path);p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();publish(a.closure,a.span,a.source,a.destination)
