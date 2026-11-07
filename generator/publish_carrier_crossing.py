"""Full-byte custody and lossless mechanics of the native 2x2 crossing."""
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
    s=read(p/'inputs.json');v=read(p/'velocity-seal.json');e=read(p/'execution.json')
    return [s['clock'],*s['donors'].values(),*[x['z'] for x in s['receivers'].values()],*s['commonFiles'].values(),*s['parentRecords'].values(),
        *[{'path':str(Path(s['modelRoot'])/n),'sha256':h} for n,h in s['imageAssets'].items()],
        *v['fields'].values(),*read(p/'derived-fields.json').values(),*[x for name in ['nativeH0Fields','textHeadFields','textWriteFields'] for x in e[name].values()]]


def verify(root):
    root,c,j,r=checked(root);p=root/'result';s=read(p/'inputs.json');base=j['result']['diagnostics']['cwd'];seal=sha(p/'velocity-seal.json')
    if read(p/'report.json')['velocitySealSHA256']!=seal:raise ValueError('Velocity release differs')
    values=items(p)+[{'path':base+'/response-result/'+x.name,'sha256':sha(x)} for x in p.iterdir()];unique={}
    for x in values:
        if x['path'] in unique and unique[x['path']]['sha256']!=x['sha256']:raise ValueError('Conflicting byte identity')
        unique[x['path']]=x
    roots=[base,*s['protectedRoots'],s['clock']['path']]
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots':roots,'items':list(unique.values())}),text=True,capture_output=True,timeout=180)
    if proc.returncode:raise RuntimeError(proc.stderr)
    v=json.loads(proc.stdout)
    v.update(schema='tce-carrier-crossing-bytes-v1',jobID=j['job_id'],velocitySealSHA256=seal,verifierSourceSHA256=sha(__file__),
        scope='Full original transformer assets, actual receiver Z/common pages/clock/donors, parent records, four complete velocities, native receiver/text fields, six derived fields and compact records. SSH byte hashing only; arrays remain near compute.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified']}


def publish(root,destination):
    root,c,j,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['velocitySealSHA256']!=sha(p/'velocity-seal.json') or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Byte verification differs')
    out=Path(destination);out.mkdir(exist_ok=False,parents=True);streams=[]
    for x in p.iterdir():
        if x.suffix=='.jsonl':
            raw=x.read_bytes();to=out/(x.name+'.gz');to.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(to.read_bytes())!=raw:raise ValueError('Lossless crossing stream differs')
            streams.append({'source':x.name,'sourceSHA256':sha(x),'sourceBytes':len(raw),'published':to.name,'publishedSHA256':sha(to),'publishedBytes':to.stat().st_size})
        else:shutil.copyfile(x,out/x.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'provenance.json',{'schema':'tce-carrier-crossing-publication-v1','jobID':j['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':c['custody'],'schedulerResult':c['result'],
        'completeCollectedInventory':c['inventory'],'losslessStreams':streams,
        'scope':'Four native supplied-carrier interventions; all original learned transformer work paid. No natural-branch parity, image, cheap predictor or arbitrary-prompt zero-neural claim.'})
    save(out/'files.json',{x.name:sha(x) for x in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


def audit(source):
    from source_response import mask
    p=Path(source);r=read(p/'request.json');s=read(p/'inputs.json');seal=read(p/'velocity-seal.json');e=read(p/'execution.json');report=read(p/'report.json');v=read(p/'retained-array-verification.json');prov=read(p/'provenance.json')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Publication member differs')
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request identity differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]:raise ValueError('Executed worker differs')
    if report['velocitySealSHA256']!=sha(p/'velocity-seal.json') or v['velocitySealSHA256']!=sha(p/'velocity-seal.json') or seal['requestSHA256']!=r['requestSHA256']:raise ValueError('Velocity seal chain differs')
    for x in items(p):
        if v['files'][x['path']]['sha256']!=x['sha256']:raise ValueError('Full byte custody absent')
    expected=['R0A','R0B','R1A','R1B']
    if sorted(seal['fields'])!=expected or any(x['shape']!=[1,1024,128] or x['logicalDtype']!='torch.bfloat16' for x in seal['fields'].values()):raise ValueError('Complete native velocities differ')
    if [x['cell'] for x in seal['writes']]!=expected or [x['cell'] for x in seal['frames']]!=expected:raise ValueError('Crossing write/frame coverage differs')
    for w,f in zip(seal['writes'],seal['frames']):
        if w['donor']!=s['donors'][w['carrier']] or w['consumedPayloadSHA256']!=w['donor']['payloadSHA256']:raise ValueError('Complete consumed donor differs')
        if w['shape']!=[1,1024,3072] or w['dtype']!='torch.bfloat16' or w['recipientStride']!=w['consumedStride'] or w['recipientStorageOffset']!=w['consumedStorageOffset']:raise ValueError('Replacement geometry differs')
        if w['consumedStoragePointer'] in [w['recipientStoragePointer'],w['donorStoragePointer']] or w['nativeH0']!=e['nativeH0Fields'][w['receiver']]:raise ValueError('Storage/receiver binding differs')
        ev=w['evidence'];a=f['sourceFrame'];b=f['outputFrame']
        if ev['input_frame_id']!=a['frame_id'] or ev['output_frame_id']!=b['frame_id'] or b['parent_frame_id']!=a['frame_id'] or b['program_fingerprint']!=r['requestSHA256']:raise ValueError('Typed frame lineage differs')
        if ev['act']!='tce.replace_first_joint_image_carrier:v1' or b['act']!=ev['act'] or a['slots']['donorH0']['fingerprint']!=w['donor']['payloadSHA256'] or a['slots']['nativeH0']['fingerprint']!=w['nativeH0']['payloadSHA256']:raise ValueError('Typed Act/input content differs')
        if ev['metrics']['consumedPayloadSHA256']!=w['consumedPayloadSHA256'] or ev['metrics']['separateStorage'] is not True:raise ValueError('Typed consumed payload differs')
    if len(e['loadedParameters'])!=169 or e['parameterMapIdentity']!=identity({x['name']:x['payloadSHA256'] for x in e['loadedParameters']}):raise ValueError('Original parameter map differs')
    for flag in ['parameterIdentityVersionClosure','bufferPayloadClosure','consumerRNGClosure']:
        if e[flag] is not True:raise ValueError('Native closure absent')
    if e['numericalProfile']!=s['numericalProfile'] or e['attentionProcessors']!=s['attentionProcessors'] or len(e['fileQuarantine']['deniedCanaries'])!=2:raise ValueError('Profile/quarantine differs')
    for cell in expected:
        if e['counts'][cell]!={'clamps':1,'textWrites':1} or sum(x['cell']==cell for x in e['linears'])!=109 or sum(x['cell']==cell for x in e['attentionCalls'])!=25:raise ValueError('Native call census differs')
    if report['nominalAffineMACs']!=18896391241728 or report['nominalAffineMACs']!=sum(x['nominalMACs'] for x in e['linears']) or report['nominalQKAVMACs']!=sum(x['nominalQKAVMACs'] for x in e['attentionCalls']):raise ValueError('Work ledger differs')
    if report['work']!={'modelLoads':1,'originalTransformerCalls':4,'recipientH0Computations':4,'carrierClamps':4,'affineCalls':436,'attentionDispatches':100,'newDonorFormation':0,'newEncoderCalls':0,'newSchedulerCommits':0,'newVAECalls':0,'newImages':0,'newFits':0}:raise ValueError('Declared work differs')
    raw=gzip.decompress((p/'observations.jsonl.gz').read_bytes());rows=[json.loads(x) for x in raw.splitlines()];stream=prov['losslessStreams'][0]
    if hashlib.sha256(raw).hexdigest()!=stream['sourceSHA256'] or len(rows)!=1024 or report['observations']!=1024 or [x['imageQueryIndex'] for x in rows]!=list(range(1024)) or [x['sourceNonlocal'] for x in rows]!=mask('bottom-right').tolist():raise ValueError('Lossless actual query coverage differs')
    derived=read(p/'derived-fields.json')
    if set(derived)!={'responseR0','responseR1','receiverEffectA','interaction','diagonalChange','closure'} or any(x['shape']!=[1,1024,128] or x['storedDtype']!='float64' for x in derived.values()):raise ValueError('Complete factorial fields differ')
    if [x['queries'] for x in report['accounts']]!=[1024,944] or not np.isfinite([x for x in report['energy'].values()]).all():raise ValueError('Factorial account differs')
    return {'schema':'tce-carrier-crossing-mechanics-v1','ownedSourcePins':len(r['ownedSources']),'cells':4,'completeVelocities':4,'completeDerivedFields':6,'queryRecords':1024,'sourceNonlocalQueries':944,
        'velocitySealSHA256':sha(p/'velocity-seal.json'),'nominalAffineMACs':report['nominalAffineMACs'],'nominalQKAVMACs':report['nominalQKAVMACs'],
        'scope':'Request/source/seal/full-byte/whole-input stride+offset+storage custody/native census/typed receipt/query coverage mechanics only. Worker verifies within-receiver prefix/nativeH0/text identity and immutable inputs. No scientific interaction threshold or generator certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source),indent=2))
