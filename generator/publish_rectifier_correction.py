"""Full bytes, source/Down/typed isolation joins and lossless scalar publication."""
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
from publish_sparse_text_kv import NPY_BYTES
from publish_terminal_trajectory import structured
from rectifier_correction import abis
from rectifier_correction_analysis import analysis_abis
from rectifier_correction_study import source_bindings


def items(p):
    s=read(p/'inputs.json');c=read(p/'candidate.json');a=read(p/'analysis-fields.json')
    return [*s['sourceFields'].values(),*s['comparisonFields'].values(),*c['fields'].values(),*a['fields'].values(),
        *[{'path':s['parentRoot']+'/response-result/'+n,'sha256':h} for n,h in s['parentRecords'].items()]]


def verify(root):
    from mrun.client.api import Api
    root,col,job,r=checked(root);p=root/'result';s=read(p/'inputs.json');live=Api().json('GET','/api/jobs/'+s['parentJobID'])
    if live['state']!='succeeded' or live['payload_custody']!=s['parentCustody'] or live['result']['diagnostics']['cwd']!=s['parentRoot']:raise ValueError('Parent scheduler custody differs')
    base=job['result']['diagnostics']['cwd'];values=items(p)+[{'path':base+'/response-result/'+x.name,'sha256':sha(x)} for x in p.iterdir()];unique={}
    for x in values:
        if x['path'] in unique and unique[x['path']]['sha256']!=x['sha256']:raise ValueError('Conflicting identities')
        unique[x['path']]={**unique.get(x['path'],{}),**x}
    spec={'items':list(unique.values()),'allowedRoots':[base,*s['sourceRoots']]}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(NPY_BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=180)
    if proc.returncode:raise RuntimeError(proc.stderr)
    v=json.loads(proc.stdout);v.update(schema='tce-rectifier-correction-bytes-v1',jobID=job['job_id'],candidateSealSHA256=sha(p/'candidate-seal.json'),verifierSourceSHA256=sha(__file__),byteTemplateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
        scope='Complete admitted/comparison/producer/analysis NPY bytes and parent metadata. OriginalD ancestry checked through frozen78/77/76 audits. Rectification/means/Down/newexpression identities source-audited/tiny-tested; no independent real-array numerical replay by byte verifier.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified'],'NPYPayloadWitnesses':len(v['npyPayloadWitnesses'])}


def publish(root,destination):
    root,col,job,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Executed verifier/release differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for x in p.iterdir():
        if x.suffix=='.jsonl':
            raw=x.read_bytes();target=out/(x.name+'.gz');target.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless stream differs')
            streams.append({'source':x.name,'sourceSHA256':sha(x),'sourceBytes':len(raw),'published':target.name,'publishedSHA256':sha(target),'publishedBytes':target.stat().st_size})
        else:shutil.copyfile(x,out/x.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json');save(out/'scheduler.json',{k:job[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-rectifier-correction-publication-v1','jobID':job['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':col['custody'],'completeCollectedInventory':col['inventory'],'losslessStreams':streams,
        'scope':'One fixed same-state rectifier/learnedDown producer and post-exit analysis. OriginalG/Up/L/maps/source/historypaid; no newstate/image/economicalcompiler/arbitrarypromptzero-neuralresult.'})
    save(out/'files.json',{x.name:sha(x) for x in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


def audit(p):
    from publish_odd_gate import audit as parent_audit
    p=Path(p);r,s,c,e,a,report,v=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json','analysis-fields.json','report.json','retained-array-verification.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Publication differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]:raise ValueError('Executed source differs:'+n)
    if v['verifierSourceSHA256']!=sha(__file__) or v['byteTemplateSourceSHA256']!=sha(Path(__file__).with_name('publish_sparse_text_kv.py')):raise ValueError('Verifier source differs')
    parent=Path(__file__).resolve().parent.parent/'docs/results/odd-gate';parent_audit(parent);ps=read(parent/'inputs.json');pe=read(parent/'execution.json')
    if any(sha(parent/n)!=h for n,h in s['parentRecords'].items()) or (s['sourceFields'],s['comparisonFields'])!=source_bindings(parent) or s['originalDLineage']!=ps['originalDLineage']:raise ValueError('Source/target/map ancestry differs')
    if s['parentSealSHA256']!=sha(parent/'candidate-seal.json') or s['parentOutputFrameID']!=pe['typedActs'][0]['outputFrame']['frame_id']:raise ValueError('Typed parent release differs')
    for x in items(p):
        if v['files'][x['path']]['sha256']!=x['sha256'] or (x.get('payloadSHA256') and v['npyPayloadWitnesses'][x['path']]['payloadSHA256']!=x['payloadSHA256']):raise ValueError('Full-byte witness absent')
    seal={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json')}
    if read(p/'candidate-seal.json')!=seal or any(x!=sha(p/'candidate-seal.json') for x in [report['candidateSealSHA256'],v['candidateSealSHA256'],a['candidateSealSHA256']]):raise ValueError('Producer seal differs')
    manifest=identity({'parentSealSHA256':s['parentSealSHA256'],'sourceFields':s['sourceFields'],'helperSHA256':r['sourceHashes']['rectifier_correction.py']})
    if e['sourceManifestSHA256']!=manifest or not e['sourceRNGClosure'] or e['numericalProfile']!=s['numericalProfile']:raise ValueError('Source/profile closure differs')
    if {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in c['fields'].items()}!=abis() or {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in a['fields'].items()}!=analysis_abis():raise ValueError('Nine/twelve ABIs differ')
    if len(e['typedActs'])!=1:raise ValueError('One producer Act required')
    act=e['typedActs'][0];src,dst,ev=act['sourceFrame'],act['outputFrame'],act['evidence']
    if src['frame_id']!=ev['input_frame_id'] or dst['frame_id']!=ev['output_frame_id'] or dst['parent_frame_id']!=src['frame_id'] or dst['program_fingerprint']!=r['requestSHA256'] or src['metadata']['sourceManifestSHA256']!=manifest:raise ValueError('Typed lineage differs')
    if {n:x['fingerprint'] for n,x in src['slots'].items()}!={n:x['payloadSHA256'] for n,x in s['sourceFields'].items()}:raise ValueError('Seven actual input bindings differ')
    if ev['metrics']['outputPayloads']!={n:x['payloadSHA256'] for n,x in c['fields'].items()} or any(dst['slots'][n]['fingerprint']!=structured(x) for n,x in c['fields'].items()):raise ValueError('Typed output bindings differ')
    guard=e['fileQuarantine'];allowed={x['path'] for x in s['sourceFields'].values()};denied=set(s['deniedNumericalInputs'])
    if set(guard['allowedFiles'])!=allowed or set(guard['deniedCanaries'])!=denied or allowed&denied or not set(guard['successfulFileOpenCounts']).issubset(allowed) or not {x['path'] for x in s['comparisonFields'].values()}.issubset(denied):raise ValueError('Actual target quarantine differs')
    exited=read(p/'producer-exit.json')
    if exited['returncode']!=0 or exited['candidateSealSHA256']!=sha(p/'candidate-seal.json') or exited['requestSHA256']!=r['requestSHA256']:raise ValueError('Producer exit-before-analysis differs')
    ops=e['operations']
    if [(x['name'],x['inputShape'],x['weightShape'],x['outputShape'],x['dtype'],x['device']) for x in ops]!=[('rectifierDown',[1,1536,9216],[3072,9216],[1,1536,3072],'torch.float64','cuda')]:raise ValueError('One actual Down differs')
    op=ops[0]
    if (op['inputPayloadSHA256'],op['weightPayloadSHA256'],op['outputPayloadSHA256'])!=(c['fields']['rectifierProduct']['payloadSHA256'],s['sourceFields']['weightD']['payloadSHA256'],c['fields']['rectifierOutput']['payloadSHA256']):raise ValueError('Physical Down operands differ')
    if sum(x['nominalMACs'] for x in ops)!=43486543872 or e['totalMatrixMACs']!=43486543872 or report['totalMatrixMACs']!=43486543872 or e['newRectifierElements']!=14155776 or report['rectifierElements']!=14155776 or any(e[n]!=0 for n in ['newGateProjections','newUpProjections','newOriginalCheckpointTensorLoads','newNativeReferences','newImages','newHeadCalls','newVAECalls']) or any(e[n]!=1 for n in ['newDownProjections','savedOriginalDownMapLoads']) or e['savedOriginalDownMapBytes']!=226492416:raise ValueError('Complete paid work differs')
    stream=read(p/'provenance.json')['losslessStreams'][0];raw=gzip.decompress((p/stream['published']).read_bytes())
    if hashlib.sha256(raw).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless row hash differs')
    rows=[json.loads(x) for x in raw.splitlines()]
    if len(rows)!=1536 or [x['row'] for x in rows]!=list(range(1536)):raise ValueError('Complete row coverage differs')
    return {'schema':'tce-rectifier-correction-mechanics-v1','typedActs':1,'sourceInputs':7,'producerABIs':9,'analysisABIs':12,'physicalFP64Projections':1,'totalMatrixMACs':43486543872,'rectifierElements':14155776,'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'observations':1536,
        'scope':'Source/parentD/typed/targetisolation/exit/physicalwork/fullbytes. Rectifier/means/contraction/newexpression arithmetic source-audited/tiny-tested,notindependentlyreal-arraynumericallyreplayed. No nativefiniteblock/image/economical/zero-neuralcertification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('root');p.add_argument('--destination');a=p.parse_args();print(json.dumps(publish(a.root,a.destination) if a.action=='publish' else globals()[a.action](a.root),indent=2))
