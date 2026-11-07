"""Full byte custody and lossless ordered text-write response publication."""
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
    s=read(p/'inputs.json');v=read(p/'velocity-seal.json');e=read(p/'execution.json');t=read(p/'targets.json')
    return [s['clock'],s['textDonor'],*s['donors'].values(),*[x['z'] for x in s['receivers'].values()],*s['commonFiles'].values(),*s['parentRecords'].values(),*s['expectedNativeFields'].values(),
        *[{'path':str(Path(s['modelRoot'])/n),'sha256':h} for n,h in s['imageAssets'].items()],*t['fields'].values(),*v['fields'].values(),*read(p/'derived-fields.json').values(),*e['nativeFields'].values()]


def verify(root):
    root,c,j,r=checked(root);p=root/'result';s=read(p/'inputs.json');base=j['result']['diagnostics']['cwd'];seal=sha(p/'velocity-seal.json')
    if read(p/'report.json')['velocitySealSHA256']!=seal:raise ValueError('New velocity release differs')
    values=items(p)+[{'path':base+'/response-result/'+x.name,'sha256':sha(x)} for x in p.iterdir()];unique={}
    for x in values:
        if x['path'] in unique and unique[x['path']]['sha256']!=x['sha256']:raise ValueError('Conflicting byte identity')
        unique[x['path']]=x
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],input=json.dumps({'allowedRoots':[base,*s['protectedRoots'],s['clock']['path']],'items':list(unique.values())}),text=True,capture_output=True,timeout=180)
    if proc.returncode:raise RuntimeError(proc.stderr)
    v=json.loads(proc.stdout);v.update(schema='tce-text-crossing-bytes-v1',jobID=j['job_id'],velocitySealSHA256=seal,verifierSourceSHA256=sha(__file__),
        scope='Full original transformer assets, receiver/common/clock/image+text donors, qualified parent records/native fields, four old and two new velocities, all seven derived fields and compact records. SSH byte hashing only; arrays remain near compute.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified']}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['velocitySealSHA256']!=sha(p/'velocity-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Byte verification differs')
    out=Path(destination);out.mkdir(exist_ok=False,parents=True);streams=[]
    for x in p.iterdir():
        if x.suffix=='.jsonl':
            raw=x.read_bytes();to=out/(x.name+'.gz');to.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(to.read_bytes())!=raw:raise ValueError('Lossless query stream differs')
            streams.append({'source':x.name,'sourceSHA256':sha(x),'sourceBytes':len(raw),'published':to.name,'publishedSHA256':sha(to),'publishedBytes':to.stat().st_size})
        else:shutil.copyfile(x,out/x.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-text-crossing-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],'schedulerResult':c['result'],'completeCollectedInventory':c['inventory'],'losslessStreams':streams,
        'scope':'Two native first-text-write/output plus whole-image/input clamp interventions; ordered signed response decomposition with all learned work paid. No causal percentages, cheap operator, images or arbitrary-prompt zero-neural claim.'})
    save(out/'files.json',{x.name:sha(x) for x in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    from source_response import mask
    p=Path(source);r=read(p/'request.json');s=read(p/'inputs.json');seal=read(p/'velocity-seal.json');e=read(p/'execution.json');report=read(p/'report.json');v=read(p/'retained-array-verification.json');prov=read(p/'provenance.json');targets=read(p/'targets.json')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Publication member differs')
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request identity differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Executed worker differs')
    if report['velocitySealSHA256']!=sha(p/'velocity-seal.json') or v['velocitySealSHA256']!=sha(p/'velocity-seal.json') or seal['requestSHA256']!=r['requestSHA256']:raise ValueError('Velocity seal chain differs')
    for x in items(p):
        if v['files'][x['path']]['sha256']!=x['sha256']:raise ValueError('Full byte custody absent')
    if sorted(seal['fields'])!=['A','B'] or any(x['shape']!=[1,1024,128] or x['logicalDtype']!='torch.bfloat16' for x in seal['fields'].values()):raise ValueError('Complete new velocity fields differ')
    expected=[('A','text'),('A','image'),('B','text'),('B','image')]
    if [(x['cell'],x['kind']) for x in seal['writes']]!=expected or [(x['cell'],x['kind']) for x in seal['frames']]!=expected:raise ValueError('Two-boundary write/frame coverage differs')
    prior={}
    for w,f in zip(seal['writes'],seal['frames']):
        kind=w['kind'];descriptor=s['textDonor'] if kind=='text' else s['donors'][w['carrier']];native_name='textWriteFields' if kind=='text' else 'nativeH0Fields';shape=[1,512 if kind=='text' else 1024,3072]
        if w['donor']!=descriptor or w['consumedPayloadSHA256']!=descriptor['payloadSHA256'] or w['nativeField']!=e['nativeFields'][native_name]:raise ValueError('Exact donor/native consumption differs')
        if w['shape']!=shape or w['dtype']!='torch.bfloat16' or w['recipientStride']!=w['consumedStride'] or w['recipientStorageOffset']!=w['consumedStorageOffset'] or w['consumedStoragePointer'] in [w['recipientStoragePointer'],w['donorStoragePointer']]:raise ValueError('Replacement geometry/storage differs')
        ev=w['evidence'];a=f['sourceFrame'];b=f['outputFrame']
        if ev['input_frame_id']!=a['frame_id'] or ev['output_frame_id']!=b['frame_id'] or b['parent_frame_id']!=a['frame_id'] or b['program_fingerprint']!=r['requestSHA256'] or ev['act']!='tce.text_crossing.replace_'+kind+':v1':raise ValueError('Typed Act/Frame lineage differs')
        if a['slots']['donor']['fingerprint']!=descriptor['payloadSHA256'] or a['slots']['native']['fingerprint']!=w['nativeField']['payloadSHA256'] or ev['metrics']['consumedPayloadSHA256']!=descriptor['payloadSHA256'] or ev['metrics']['separateStorage'] is not True:raise ValueError('Typed content evidence differs')
        if kind=='text':prior[w['cell']]=b['frame_id']
        elif f['earlierTextOutputFrameID']!=prior[w['cell']] or a['metadata']['earlierTextOutputFrameID']!=prior[w['cell']]:raise ValueError('Text-before-image temporal binding differs')
    for n,x in e['nativeFields'].items():
        if x['payloadSHA256']!=s['expectedNativeFields'][n]['payloadSHA256']:raise ValueError('Original native receiver path differs')
    if len(e['loadedParameters'])!=169 or e['parameterMapIdentity']!=s['parameterMapIdentity'] or e['parameterMapIdentity']!=identity({x['name']:x['payloadSHA256'] for x in e['loadedParameters']}):raise ValueError('Original parameter map differs')
    if any(e[k] is not True for k in ['parameterIdentityVersionClosure','bufferPayloadClosure','inputPayloadVersionClosure','consumerRNGClosure']):raise ValueError('Native mechanics closure absent')
    q=e['fileQuarantine']
    if len(q['deniedCanaries'])!=3 or set(q['postSealVelocityReadFiles'])!={x['path'] for x in targets['fields'].values()} or set(q['primaryAllowedReadFiles'])&set(q['postSealVelocityReadFiles']) or q['releaseSHA256BeforeTargetReads']!=sha(p/'velocity-seal.json'):raise ValueError('Pre/post-seal target quarantine differs')
    if e['numericalProfile']!=s['numericalProfile'] or e['attentionProcessors']!=s['attentionProcessors'] or e['receiverPrefixHashes']!=s['expectedReceiverPrefix']:raise ValueError('Qualified profile/prefix differs')
    for cell in ['A','B']:
        if e['counts'][cell]!={'textHeads':1,'textClamps':1,'imageClamps':1} or sum(x['cell']==cell for x in e['linears'])!=109 or sum(x['cell']==cell for x in e['attentionCalls'])!=25:raise ValueError('Native census differs')
    if report['nominalAffineMACs']!=9448195620864 or report['nominalAffineMACs']!=sum(x['nominalMACs'] for x in e['linears']) or report['nominalQKAVMACs']!=724775731200 or report['nominalQKAVMACs']!=sum(x['nominalQKAVMACs'] for x in e['attentionCalls']):raise ValueError('Work ledger differs')
    expected_work={'modelLoads':1,'originalTransformerCalls':2,'recipientH0Computations':2,'recipientTextProjectionComputations':2,'imageClamps':2,'textClamps':2,'affineCalls':218,'attentionDispatches':50,'newDonorFormation':0,'newEncoderCalls':0,'newFits':0,'newSchedulerCommits':0,'newVAECalls':0,'newImages':0}
    if report['work']!=expected_work:raise ValueError('Declared physical work differs')
    raw=gzip.decompress((p/'observations.jsonl.gz').read_bytes());rows=[json.loads(x) for x in raw.splitlines()]
    if hashlib.sha256(raw).hexdigest()!=prov['losslessStreams'][0]['sourceSHA256'] or len(rows)!=1024 or report['observations']!=1024 or [x['imageQueryIndex'] for x in rows]!=list(range(1024)) or [x['sourceNonlocal'] for x in rows]!=mask('bottom-right').tolist():raise ValueError('Actual-address query coverage differs')
    fs=read(p/'derived-fields.json')
    if set(fs)!={'responseR0','responseR1','responseR1TextR0','originalInteraction','remainingReceiverDifference','firstTextWriteEffect','closure'} or any(x['shape']!=[1,1024,128] or x['storedDtype']!='float64' for x in fs.values()) or [x['queries'] for x in report['accounts']]!=[1024,944]:raise ValueError('Ordered full-field/scope coverage differs')
    parent=Path(__file__).parent.parent/'docs/results/carrier-crossing'
    if targets['velocitySealSHA256']!=sha(parent/'velocity-seal.json') or targets['fields']!=read(parent/'velocity-seal.json')['fields']:raise ValueError('Old velocity target lineage differs')
    for old,new in zip(read(parent/'report.json')['accounts'],report['accounts']):
        if old['referenceRms']!=new['responseR0Rms'] or old['candidateRms']!=new['responseR1Rms']:raise ValueError('Old response reconstruction differs')
    return {'schema':'tce-text-crossing-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'newVelocities':2,'typedWrites':4,'completeDerivedFields':7,'queryRecords':1024,'sourceNonlocalQueries':944,'velocitySealSHA256':sha(p/'velocity-seal.json'),
        'maxClosureAbs':max(x['maxClosureAbs'] for x in report['accounts']),'scope':'Frozen request/source/seal/fullbyte/typed two-boundary temporal+layout+storage/native census/prefix/target quarantine/query coverage/oldresponse lineage mechanics. Closure retained as observed; no scientific threshold, causal shares or generator certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args();print(json.dumps(publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source),indent=2))
