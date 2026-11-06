"""Package saved assay bytes for the static browser demo; no model execution."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def export(run,destination):
    run=Path(run).resolve();out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1]/'docs/results/zero-network-response'
    report=read(source/'report.json');request=read(source/'request.json');analysis=read(source/'analysis.json');seal=read(source/'prediction-seal.json')
    if sha(run/'result/report.json')!=sha(source/'report.json'):raise ValueError('Collected report differs')
    if sha(source/'chart.npz')!=seal['bundleSHA256']:raise ValueError('Chart identity differs')
    contexts=[]
    with np.load(source/'chart.npz',allow_pickle=False) as bundle:
        for name in bundle.files:
            pixels=bundle[name]
            if pixels.shape!=(3,512,512,3) or pixels.dtype!=np.uint8:raise ValueError('Chart geometry differs')
            raw=pixels.tobytes();path=name+'.rgb.gz';(out/path).write_bytes(gzip.compress(raw,mtime=0))
            contexts.append({'id':name,'path':path,'bytes':(out/path).stat().st_size,'rawBytes':len(raw),'sha256':sha(out/path),
                             'rawSHA256':hashlib.sha256(raw).hexdigest()})
    held=[]
    for item in request['evaluation']:
        original=run/'payload'/item['path']
        if sha(original)!=item['sha256']:raise ValueError('Historical target differs')
        name=Path(item['path']).name;shutil.copyfile(original,out/name)
        held.append({**{k:item[k] for k in ['context','alpha','sha256']},'path':name})
    expected=[]
    for row in report['serving']['rows']:
        if row['method']!='piecewise':continue
        path=run/'result/images'/row['path']
        if sha(path)!=seal['images'][row['path']]:raise ValueError('Recorded prediction differs')
        pixels=np.asarray(Image.open(path).convert('RGB'))
        expected.append({'context':row['context'],'alpha':row['alpha'],'rgbSHA256':hashlib.sha256(pixels.tobytes()).hexdigest()})
        if row['alpha']==request['novelAlpha']:shutil.copyfile(path,out/row['path'])
    for name in ['report.json','analysis.json','request.json','provenance.json']:shutil.copyfile(source/name,out/name)
    data={'schema':'tce-browser-response-v1','width':512,'height':512,'anchors':[0,.25,1],'contexts':contexts,'held':held,'expectedPrimaryPixels':expected,
          'jobID':'job-2e98c6786986','requestSHA256':request['requestSHA256'],'reportSHA256':sha(source/'report.json'),
          'timing':analysis['timing'],'serving':{k:report['serving'][k] for k in ['loadSeconds','wallSeconds','peakRssBytes','runtime']},
          'compileSeconds':report['compile']['seconds'],'bundleBytes':report['compile']['bundleBytes'],'workerWallSeconds':report['workerWallSeconds'],
          'sourceCost':report['source']['historicalCost'],'physicalWork':report['physicalWork'],
          'scores':[r for r in report['scores']['rows'] if r['method']=='piecewise'],'interactionRows':[r for r in analysis['interactionRows'] if r['method']=='piecewise'],
          'groups':analysis['groups'],'scope':report['scope']}
    (out/'response.json').write_text(json.dumps(data,sort_keys=True,indent=2,allow_nan=False))
    (out/'files.json').write_text(json.dumps({p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},sort_keys=True,indent=2))
    print(json.dumps({'destination':str(out),'contexts':len(contexts),'heldImages':len(held),'assetsBytes':sum(p.stat().st_size for p in out.iterdir())}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();export(a.run,a.destination)
