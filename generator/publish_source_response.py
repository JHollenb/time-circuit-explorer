"""Byte verification and lossless publication for the response-only contrast."""
import argparse
import gzip
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read,save,sha,identity
from publish_source_native import checked
from publish_source_context import REMOTE

BYTE_RECEIPT='retained-array-verification-v2.json'


def verify(root):
    root,c,j,r=checked(root);p=root/'result';s=read(p/'source.json');t=read(p/'targets.json');fit=read(p/'fit.json');served=read(p/'serving.json');cs=read(p/'coefficient-seal.json')
    if read(p/'report.json')['candidateSealSHA256']!=sha(p/'candidate-seal.json') or cs['fitReceiptSHA256']!=sha(p/'fit.json'):raise ValueError('Response release differs')
    items=[s['scale'],s['oldWeight'],cs['baselineField'],*s['parentRecords'].values(),*fit['fields'].values()]
    for pair in s['pairs']:
        for side in ['A','B']:items+=list(pair[side].values())
    for fs in t.values():items+=fs
    for pair in fit['discoveryPairs']:items += [pair[n] for n in ['X','Y','mask']]
    for row in served['rows']:items += [row['prediction'],*row['fields'].values()]
    for fs in read(p/'diagnostic-fields.json').values():items += list(fs.values())
    base=j['result']['diagnostics']['cwd'];items += [{'path':base+'/response-result/'+x.name,'sha256':sha(x)} for x in p.iterdir()]
    unique={x['path']:x for x in items};roots=[base,*s['protectedRoots']]
    v=json.loads(subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots':roots,'items':list(unique.values())}),text=True,capture_output=True,check=True,timeout=180).stdout)
    v.update(schema='tce-source-response-bytes-v1',jobID=j['job_id'],candidateSealSHA256=sha(p/'candidate-seal.json'),verifierSourceSHA256=sha(__file__),
        scope='All supplied H0/Z/previous/scale/old coefficients, old native labels, complete new normalized features/coefficients/masks/raw responses/targets and compact records. Byte hashing only; arrays remain near compute.')
    save(root/BYTE_RECEIPT,v);return {'files':len(v['files']),'bytes':v['totalBytesVerified']}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/BYTE_RECEIPT)
    if v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Byte verification differs')
    out=Path(destination);out.mkdir(exist_ok=False,parents=True);streams=[]
    for x in p.iterdir():
        if x.suffix=='.jsonl':
            raw=x.read_bytes();to=out/(x.name+'.gz');to.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(to.read_bytes())!=raw:raise ValueError('Lossless stream differs')
            streams.append({'source':x.name,'sourceSHA256':sha(x),'published':to.name,'publishedSHA256':sha(to),'sourceBytes':len(raw),'publishedBytes':to.stat().st_size})
        else:shutil.copyfile(x,out/x.name)
    shutil.copyfile(root/BYTE_RECEIPT,out/'retained-array-verification.json')
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification-before-scope-correction.json')
    save(out/'provenance.json',{'schema':'tce-source-response-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],
        'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreams':streams,
        'scope':'Retrospective raw-response objective/transfer discriminator. No original model/H0/VAE/image evaluations;16 served learned affine products plus one post-fit product remain paid. Old learned operands/labels are preparation. Not a full velocity or arbitrary-prompt zero-neural generator.'})
    save(out/'files.json',{x.name:sha(x) for x in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    p=Path(source);r=read(p/'request.json');s=read(p/'source.json');fit=read(p/'fit.json');served=read(p/'serving.json');seal=read(p/'candidate-seal.json');cs=read(p/'coefficient-seal.json');v=read(p/'retained-array-verification.json');report=read(p/'report.json');prov=read(p/'provenance.json')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Published member differs')
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Executed source differs')
    if report['candidateSealSHA256']!=sha(p/'candidate-seal.json') or v['candidateSealSHA256']!=sha(p/'candidate-seal.json') or seal['servingSHA256']!=sha(p/'serving.json') or cs['fitReceiptSHA256']!=sha(p/'fit.json'):raise ValueError('Release chain differs')
    if seal['rows']!=served['rows'] or cs['restoredFrame']!=cs['baselineFrame'] or cs['sourceFrame']!=cs['baselineFrame'] or cs['fittedFrame']['parent_frame_id']!=cs['sourceFrame']['frame_id']:raise ValueError('Typed immutable coefficient lineage differs')
    if fit['rows']!=1888 or fit['features']!=3072 or fit['ridge']!=.01 or fit['intercept'] or fit['gain']!=1:raise ValueError('Fixed response objective differs')
    if fit['sourceScale']!=s['scale'] or served['sourceScale']!=s['scale'] or served['oldWeight']!=s['oldWeight']:raise ValueError('Inherited numerical operands differ')
    for actor in [fit,served]:
        if actor['frameworksLoaded'] or len(actor['fileQuarantine']['deniedCanaries'])!=2:raise ValueError('Fresh actor quarantine absent')
    def custody(item):
        if v['files'][item['path']]['sha256']!=item['sha256']:raise ValueError('Full byte custody absent')
    for item in [s['scale'],s['oldWeight'],cs['baselineField'],*fit['fields'].values()]:custody(item)
    for pair in fit['discoveryPairs']:
        for n in ['X','Y','mask']:custody(pair[n])
    zeros=set()
    for row in served['rows']:
        if row['declaredQueries']!=944 or row['applicableQueries']!=944 or row['inapplicableQueries']:raise ValueError('Unchanged-query applicability differs')
        for item in [row['prediction'],*row['fields'].values()]:custody(item)
        if row['method']=='zero':zeros.add(row['prediction']['path'])
    if len(zeros)!=1 or len(served['rows'])!=24 or len(report['accounts'])!=24:raise ValueError('Physical zero reuse/logical response coverage differs')
    data=gzip.decompress((p/'observations.jsonl.gz').read_bytes())
    if __import__('hashlib').sha256(data).hexdigest()!=prov['losslessStreams'][0]['sourceSHA256'] or len(data.splitlines())!=report['observations']:raise ValueError('Lossless observation coverage differs')
    max_closure=max(abs(x['errorEnergy']-x['predictionEnergy']-x['referenceRms']**2+x['twicePredictionTarget']) for x in report['accounts'])
    return {'schema':'tce-source-response-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'discoveryPairs':2,'feedbackPairs':2,'evaluationPairs':4,'rawResponses':24,
        'physicallySharedZeroArrays':len(zeros),'observations':report['observations'],'maxEnergyClosureAbs':max_closure,'candidateSealSHA256':sha(p/'candidate-seal.json'),
        'scope':'Source/fixedobjective/quarantine/typedcoefficientlineage/fullbytes/losslessrows mechanics; no response quality, independence or generator certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source),indent=2))
