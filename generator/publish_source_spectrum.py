"""Byte custody, compact publication and source-spectrum mechanics audit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read,save,sha,identity
from publish_source_native import checked
from publish_source_context import REMOTE


def verify(root):
    root,c,j,r=checked(root);p=root/'result';s=read(p/'source.json');report=read(p/'report.json');fields=read(p/'fields.json')
    if report['analysisSealSHA256']!=sha(p/'analysis-seal.json') or read(p/'analysis-seal.json')['fields']!=fields:raise ValueError('Analysis release differs')
    items=[s['scale'],*s['parentRecords'].values(),*fields.values()]
    for pair in s['pairs']:items += [pair[n] for n in ['X','Y','mask','oldPrediction']]
    base=j['result']['diagnostics']['cwd'];items += [{'path':base+'/response-result/'+x.name,'sha256':sha(x)} for x in p.iterdir()]
    unique={x['path']:x for x in items}
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots':[base,*s['protectedRoots']],'items':list(unique.values())}),text=True,capture_output=True,check=True,timeout=180)
    v=json.loads(result.stdout);v.update(schema='tce-source-spectrum-bytes-v1',jobID=j['job_id'],analysisSealSHA256=sha(p/'analysis-seal.json'),verifierSourceSHA256=sha(__file__),
        scope='Complete supplied discovery X/Y/masks/scale/old prediction, full SVD/mode/reconstruction fields and compact records. Byte hashing only; arrays remain near compute.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified']}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['analysisSealSHA256']!=sha(p/'analysis-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Byte verification differs')
    out=Path(destination);out.mkdir(exist_ok=False,parents=True);streams=[]
    for x in p.iterdir():
        if x.suffix=='.jsonl':
            raw=x.read_bytes();to=out/(x.name+'.gz');to.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(to.read_bytes())!=raw:raise ValueError('Lossless mode stream differs')
            streams.append({'source':x.name,'sourceSHA256':sha(x),'sourceBytes':len(raw),'published':to.name,'publishedSHA256':sha(to),'publishedBytes':to.stat().st_size})
        else:shutil.copyfile(x,out/x.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-source-spectrum-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],
        'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreams':streams,
        'scope':'One fixed-source SVD/ridge reconstruction. Target-assisted discovery analysis; no new predictor, selection, original model or image. Old learned operands/labels/products remain paid.'})
    save(out/'files.json',{x.name:sha(x) for x in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    p=Path(source);r=read(p/'request.json');s=read(p/'source.json');report=read(p/'report.json');seal=read(p/'analysis-seal.json');fields=read(p/'fields.json');v=read(p/'retained-array-verification.json');prov=read(p/'provenance.json')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Published member differs')
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Executed source differs')
    if report['analysisSealSHA256']!=sha(p/'analysis-seal.json') or v['analysisSealSHA256']!=sha(p/'analysis-seal.json') or seal['fields']!=fields:raise ValueError('Complete analysis release differs')
    if s['expectedGeometry']!=[1888,3072,128] or s['ridge']!=.01 or len(s['pairs'])!=2 or any(not x['id'].startswith('discovery-') for x in s['pairs']):raise ValueError('Frozen discovery contract differs')
    if report['frameworksLoaded'] or len(report['fileQuarantine']['deniedCanaries'])!=2:raise ValueError('Framework/target quarantine absent')
    expected={'U':[1888,1888],'sigma':[1888],'Vt':[1888,3072],'targetModes':[1888,128],'reconstructedPrediction':[1888,128],'originalPredictionDifference':[1888,128]}
    for n,shape in expected.items():
        if fields[n]['shape']!=shape:raise ValueError('Thin SVD/reconstruction geometry differs')
    def custody(item):
        if v['files'][item['path']]['sha256']!=item['sha256']:raise ValueError('Full byte custody absent')
    for item in [s['scale'],*fields.values()]:custody(item)
    for pair in s['pairs']:
        for n in ['X','Y','mask','oldPrediction']:custody(pair[n])
    raw=gzip.decompress((p/'modes.jsonl.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest()!=seal['modesSHA256'] or prov['losslessStreams'][0]['sourceSHA256']!=seal['modesSHA256']:raise ValueError('Lossless spectrum differs')
    rows=[json.loads(x) for x in raw.splitlines()]
    if len(rows)!=1888 or [x['position'] for x in rows]!=list(range(1888)) or any(rows[i]['singularValue']<rows[i+1]['singularValue'] for i in range(1887)):raise ValueError('Complete ordered spectrum differs')
    if report['work']!={'SVDs':1,'newPredictors':0,'newLearnedFits':0,'originalModelEvaluations':0,'newH0Computations':0,'newImages':0}:raise ValueError('Declared physical work differs')
    return {'schema':'tce-source-spectrum-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'sourcePairs':2,'retainedFields':len(fields),'completeModes':len(rows),
        'analysisSealSHA256':sha(p/'analysis-seal.json'),'scope':'Source/release/fullbyte/thinSVDgeometry/losslessmode/frameworkquarantine mechanics only; no rank, information sufficiency, transfer or generator certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source),indent=2))
