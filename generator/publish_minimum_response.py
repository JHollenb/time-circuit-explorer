"""Complete byte custody and lossless publication of one endpoint forecast."""
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


def items(p):
    s=read(p/'source.json');fit=read(p/'fit.json');sc=read(p/'scoring-source.json');served=read(p/'serving.json');cs=read(p/'coefficient-seal.json')
    values=[s['scale'],*s['factors'].values(),*s['parentRecords'].values(),cs['baselineField'],*fit['fields'].values(),*sc['targets'].values()]
    for row in sc['controls']+served['rows']:values += [row['prediction'],*row['fields'].values()]
    return values


def verify(root):
    root,c,j,r=checked(root);p=root/'result';s=read(p/'source.json');seal=sha(p/'candidate-seal.json')
    if read(p/'report.json')['candidateSealSHA256']!=seal or read(p/'coefficient-seal.json')['fitReceiptSHA256']!=sha(p/'fit.json'):raise ValueError('Forecast release differs')
    base=j['result']['diagnostics']['cwd'];values=items(p)+[{'path':base+'/response-result/'+x.name,'sha256':sha(x)} for x in p.iterdir()];unique={x['path']:x for x in values}
    v=json.loads(subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots':[base,*s['protectedRoots']],'items':list(unique.values())}),text=True,capture_output=True,check=True,timeout=180).stdout)
    v.update(schema='tce-minimum-response-bytes-v1',jobID=j['job_id'],candidateSealSHA256=seal,verifierSourceSHA256=sha(__file__),
        scope='All supplied retained factors/scale, normalized inputs/masks, old response targets/controls, baseline/new coefficients/new predictions and compact records. Byte hashing only. Native H0/label production has its historical parent custody; arrays remain near compute.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified']}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Byte verification differs')
    out=Path(destination);out.mkdir(exist_ok=False,parents=True);streams=[]
    for x in p.iterdir():
        if x.suffix=='.jsonl':
            raw=x.read_bytes();to=out/(x.name+'.gz');to.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(to.read_bytes())!=raw:raise ValueError('Lossless stream differs')
            streams.append({'source':x.name,'sourceSHA256':sha(x),'published':to.name,'publishedSHA256':sha(to),'sourceBytes':len(raw),'publishedBytes':to.stat().st_size})
        else:shutil.copyfile(x,out/x.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-minimum-response-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],'schedulerResult':c['result'],
        'completeCollectedInventory':c['inventory'],'losslessStreams':streams,'scope':'One retrospective allmode minimum-norm learned response map;8new/24reused controls, not completevelocity or arbitrary-prompt zero-neural images. Original operands/labels/factorization remain paid.'})
    save(out/'files.json',{x.name:sha(x) for x in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    p=Path(source);r=read(p/'request.json');s=read(p/'source.json');sc=read(p/'scoring-source.json');fit=read(p/'fit.json');served=read(p/'serving.json');cs=read(p/'coefficient-seal.json');seal=read(p/'candidate-seal.json');v=read(p/'retained-array-verification.json');report=read(p/'report.json');prov=read(p/'provenance.json')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Publication member differs')
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Executed source differs')
    if report['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or seal['servingSHA256']!=sha(p/'serving.json') or cs['fitReceiptSHA256']!=sha(p/'fit.json') or seal['coefficientSealSHA256']!=sha(p/'coefficient-seal.json'):raise ValueError('Release chain differs')
    if seal['rows']!=served['rows'] or fit['fields']!=cs['fields'] or cs['restoredFrame']!=cs['baselineFrame'] or cs['sourceFrame']!=cs['baselineFrame'] or cs['fittedFrame']['parent_frame_id']!=cs['sourceFrame']['frame_id']:raise ValueError('Typed coefficient lineage differs')
    if [fit[k] for k in ['modes','features','outputs','regularization','intercept','gain','allModesUsed']]!=[1888,3072,128,0,False,1,True]:raise ValueError('Fixed minimum-norm law differs')
    if fit['sourceFactors']!=s['factors'] or fit['sourceScale']!=s['scale'] or served['sourceScale']!=s['scale']:raise ValueError('Inherited factors/scale differ')
    prior=HERE.parent/'docs/results/source-response';spectrum=HERE.parent/'docs/results/source-spectrum'
    if sc['controls']!=read(prior/'serving.json')['rows'] or sc['targets']!={k:x['target'] for k,x in read(prior/'diagnostic-fields.json').items()} or s['factors']!={k:x for k,x in read(spectrum/'fields.json').items() if k in ['Vt','sigma','targetModes']}:raise ValueError('Fixed controls/targets/factors differ from parent publication')
    for item in items(p):
        if v['files'][item['path']]['sha256']!=item['sha256']:raise ValueError('Complete byte custody absent')
    for actor in [fit,served]:
        if actor['frameworksLoaded'] or len(actor['fileQuarantine']['deniedCanaries'])!=2:raise ValueError('Actor framework/target exclusion absent')
    expected={x['id']:x for x in s['pairs']}
    if len(served['rows'])!=8 or len(report['accounts'])!=32 or len(sc['controls'])!=24:raise ValueError('Logical response coverage differs')
    for row in served['rows']:
        if row['fields']!=expected[row['id']]['fields'] or row['prediction']['shape']!=[944,128] or row['applicableQueries']!=944 or row['inapplicableQueries']:raise ValueError('Archived source/query coverage differs')
    data=gzip.decompress((p/'observations.jsonl.gz').read_bytes())
    if hashlib.sha256(data).hexdigest()!=prov['losslessStreams'][0]['sourceSHA256'] or len(data.splitlines())!=report['observations'] or report['observations']!=30208:raise ValueError('Lossless row coverage differs')
    if report['work']!={'SVDs':0,'newLearnedCoefficientMaps':1,'newLearnedResponseProducts':8,'reusedControlResponses':24,'originalModelEvaluations':0,'newH0Computations':0,'newImages':0}:raise ValueError('Declared physical work differs')
    closure=max(abs(x['errorEnergy']-x['predictionEnergy']-x['targetEnergy']+x['twicePredictionTarget']) for x in report['accounts'])
    return {'schema':'tce-minimum-response-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'newResponses':8,'reusedControlResponses':24,'observations':30208,'allModesUsed':1888,'maxEnergyClosureAbs':closure,
        'candidateSealSHA256':sha(p/'candidate-seal.json'),'scope':'Exact source/factor/control/release/fullbyte/quarantine/typedlineage/losslesscoverage mechanics only. No interpolation, transfer or generator quality gate.'}


HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source),indent=2))
