"""Full byte custody and lossless publication of a fixed extension audit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read,save,sha,identity,np
from publish_source_native import checked
from publish_source_context import REMOTE


def items(p):
    s=read(p/'source.json');g=read(p/'geometry.json');ref=read(p/'reference-source.json')
    return [s['inheritedScale'],*s['factors'].values(),*s['parentRecords'].values(),*[x for pair in s['pairs'] for x in pair['fields'].values()],
        *[x for fs in g['fields'].values() for x in fs.values()],ref['report'],*ref['predictions'].values()]


def verify(root):
    root,c,j,r=checked(root);p=root/'result';s=read(p/'source.json');seal=sha(p/'geometry-seal.json');base=j['result']['diagnostics']['cwd']
    if read(p/'report.json')['geometrySealSHA256']!=seal or read(p/'geometry-seal.json')['geometryReceiptSHA256']!=sha(p/'geometry.json'):raise ValueError('Geometry release differs')
    values=items(p)+[{'path':base+'/response-result/'+x.name,'sha256':sha(x)} for x in p.iterdir()];unique={x['path']:x for x in values}
    v=json.loads(subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots':[base,*s['protectedRoots']],'items':list(unique.values())}),text=True,capture_output=True,check=True,timeout=180).stdout)
    v.update(schema='tce-extension-geometry-bytes-v1',jobID=j['job_id'],geometrySealSHA256=seal,verifierSourceSHA256=sha(__file__),
        scope='Complete supplied factors/scale/normalized inputs/masks/old forecast references, new A/Yhat/D arrays and compact records. Byte hashing only, no native response target access or array hydration. Historical native production remains paid.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified']}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['geometrySealSHA256']!=sha(p/'geometry-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Byte verification differs')
    out=Path(destination);out.mkdir(exist_ok=False,parents=True);streams=[]
    for x in p.iterdir():
        if x.suffix=='.jsonl':
            raw=x.read_bytes();to=out/(x.name+'.gz');to.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(to.read_bytes())!=raw:raise ValueError('Lossless geometry stream differs')
            streams.append({'source':x.name,'sourceSHA256':sha(x),'sourceBytes':len(raw),'published':to.name,'publishedSHA256':sha(to),'publishedBytes':to.stat().st_size})
        else:shutil.copyfile(x,out/x.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-extension-geometry-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],'schedulerResult':c['result'],
        'completeCollectedInventory':c['inventory'],'losslessStreams':streams,'scope':'Retrospective fixed-map geometry; no new predictor, fit, original model, image or causal conditioning test. Old learned factors/labels/state remain paid.'})
    save(out/'files.json',{x.name:sha(x) for x in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    from source_response import mask
    p=Path(source);r=read(p/'request.json');s=read(p/'source.json');g=read(p/'geometry.json');seal=read(p/'geometry-seal.json');v=read(p/'retained-array-verification.json');report=read(p/'report.json');prov=read(p/'provenance.json');ref=read(p/'reference-source.json')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Publication member differs')
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Executed worker differs')
    if report['geometrySealSHA256']!=sha(p/'geometry-seal.json') or v['geometrySealSHA256']!=sha(p/'geometry-seal.json') or seal['geometryReceiptSHA256']!=sha(p/'geometry.json') or seal['fields']!=g['fields']:raise ValueError('Geometry seal chain differs')
    for item in items(p):
        if v['files'][item['path']]['sha256']!=item['sha256']:raise ValueError('Full byte custody absent')
    if g['frameworksLoaded'] or len(g['fileQuarantine']['deniedCanaries'])!=2 or len(s['pairs'])!=6:raise ValueError('Geometry cohort/quarantine differs')
    expected={x['id']:x for x in s['pairs']};streams={}
    for name,key in [('queries','queryStreamSHA256'),('modes','modeStreamSHA256')]:
        raw=gzip.decompress((p/(name+'.jsonl.gz')).read_bytes())
        if hashlib.sha256(raw).hexdigest()!=seal[key] or next(x for x in prov['losslessStreams'] if x['source']==name+'.jsonl')['sourceSHA256']!=seal[key]:raise ValueError('Lossless stream differs')
        streams[name]=[json.loads(x) for x in raw.splitlines()]
    for key,pair in expected.items():
        qs=[x for x in streams['queries'] if x['id']==key];ms=[x for x in streams['modes'] if x['id']==key]
        if [x['selectedRowIndex'] for x in qs]!=list(range(944)) or [x['imageQueryIndex'] for x in qs]!=np.flatnonzero(mask(pair['location'])).tolist() or [x['position'] for x in ms]!=list(range(1888)):raise ValueError('Complete query/image-index/mode coverage differs')
        fs=g['fields'][key]
        if {k:x['shape'] for k,x in fs.items()}!={'inverseCoordinates':[944,1888],'reconstructedPrediction':[944,128],'modalContraction':[1888,128]}:raise ValueError('Retained geometry dimensions differ')
    if len(streams['queries'])!=g['queryRecords'] or len(streams['modes'])!=g['modeRecords'] or g['queryRecords']!=5664 or g['modeRecords']!=11328:raise ValueError('Record coverage differs')
    prior=Path(__file__).resolve().parent.parent/'docs/results/minimum-response';old=read(prior/'report.json');served=read(prior/'serving.json')
    if ref['predictions']!={x['id']:x['prediction'] for x in served['rows'] if x['population']!='discovery'}:raise ValueError('Old forecast references differ')
    for a in report['accounts']:
        if a['oldResponseAccount']!=next(x for x in old['accounts'] if x['id']==a['id'] and x['method']=='minimum-norm'):raise ValueError('Old scalar account differs')
    if report['nominalLeadingMACs']!=68439244800 or report['work']!={'SVDs':0,'newFits':0,'newPredictors':0,'fixedPredictionReconstructions':6,'originalModelEvaluations':0,'newH0Computations':0,'newImages':0}:raise ValueError('Declared fixed work differs')
    return {'schema':'tce-extension-geometry-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'pairs':6,'retainedArrays':18,'queryRecords':5664,'modeRecords':11328,
        'geometrySealSHA256':sha(p/'geometry-seal.json'),'maxQueryEnergyClosureAbs':max(x['maxQueryEnergyClosureAbs'] for x in report['accounts']),
        'scope':'Source/release/fullbyte/frameworkquarantine/completequery/imageindex/modecoverage/oldforecastjoins mechanics only; no geometry quality or causal/generator certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source),indent=2))
