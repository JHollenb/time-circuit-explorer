"""Full byte custody, source isolation, physical work and lossless publication."""
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
from mean_gate import abis,SOURCE_NAMES,COMPARISON_NAMES
from mean_gate_analysis import analysis_abis


def items(p):
    s=read(p/'inputs.json');c=read(p/'candidate.json');a=read(p/'analysis-fields.json')
    return [*s['sourceFields'].values(),*s['comparisonFields'].values(),*c['fields'].values(),*a['fields'].values(),
        *[{'path':s['parentRoot']+'/response-result/'+n,'sha256':h} for n,h in s['parentRecords'].items()]]


def verify(root):
    from mrun.client.api import Api
    root,col,job,r=checked(root);p=root/'result';s=read(p/'inputs.json');parent=Api().json('GET','/api/jobs/'+s['parentJobID'])
    if parent['state']!='succeeded' or parent['payload_custody']!=s['parentCustody'] or parent['result']['diagnostics']['cwd']!=s['parentRoot']: raise ValueError('Paid source custody differs')
    base=job['result']['diagnostics']['cwd'];values=items(p)+[{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()];unique={}
    for x in values:
        if x['path'] in unique and unique[x['path']]['sha256']!=x['sha256']: raise ValueError('Conflicting source identities')
        unique[x['path']]={**unique.get(x['path'],{}),**x}
    spec={'items':list(unique.values()),'allowedRoots':[base,*s['sourceRoots']]}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(NPY_BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=180)
    if proc.returncode: raise RuntimeError(proc.stderr)
    v=json.loads(proc.stdout);v.update(schema='tce-mean-gate-bytes-v1',jobID=job['job_id'],candidateSealSHA256=sha(p/'candidate-seal.json'),
        verifierSourceSHA256=sha(__file__),byteTemplateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
        scope='All six admitted fields, four quarantined comparisons, six producer and fifteen analysis arrays, parent metadata and complete compact result. SSH performs bytes/NPY geometry/payload checks only. Parent76 separately authenticates original D promotion lineage; no independent real-array numeric replay of reductions/projection.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified'],'NPYPayloadWitnesses':len(v['npyPayloadWitnesses'])}


def publish(root,destination):
    root,col,job,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'): raise ValueError('Executed verifier/release differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for n in p.iterdir():
        if n.suffix=='.jsonl':
            raw=n.read_bytes();dst=out/(n.name+'.gz');dst.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(dst.read_bytes())!=raw: raise ValueError('Lossless stream publication differs')
            streams.append({'source':n.name,'sourceSHA256':sha(n),'sourceBytes':len(raw),'published':dst.name,'publishedSHA256':sha(dst),'publishedBytes':dst.stat().st_size})
        else: shutil.copyfile(n,out/n.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'scheduler.json',{k:job[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-mean-gate-publication-v1','jobID':job['job_id'],'requestSHA256':r['requestSHA256'],
        'payloadCustody':col['custody'],'completeCollectedInventory':col['inventory'],'losslessStreams':streams,
        'scope':'One paid own-state mean-gate projection; post-exit changing/reassociation term and full/centered diagnostic, not correction application/continuation/image/economical producer/zero-neural result.'})
    save(out/'files.json',{n.name:sha(n) for n in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(n.stat().st_size for n in out.iterdir())}


def audit(p):
    from publish_gated_covariance import audit as parent_audit
    p=Path(p);r,s,c,e,report,v,a=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json','report.json','retained-array-verification.json','analysis-fields.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']: raise ValueError('Request changed')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h: raise ValueError('Published file differs')
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]: raise ValueError('Executed source changed:'+n)
    if v['verifierSourceSHA256']!=sha(__file__) or v['byteTemplateSourceSHA256']!=sha(Path(__file__).with_name('publish_sparse_text_kv.py')): raise ValueError('Executed verifier changed')
    parent=Path(__file__).resolve().parent.parent/'docs/results/own-state-gated-covariance';parent_audit(parent)
    ps=read(parent/'inputs.json');pc=read(parent/'candidate.json');pe=read(parent/'execution.json');pv=read(parent/'retained-array-verification.json')
    if any(sha(parent/n)!=h for n,h in s['parentRecords'].items()) or s['parentSealSHA256']!=sha(parent/'candidate-seal.json'): raise ValueError('Parent snapshot differs')
    if s['parentOutputFrameID']!=pe['typedActs'][0]['outputFrame']['frame_id']: raise ValueError('Parent output lineage differs')
    if s['sourceFields']!={n:(ps['sourceFields'][n] if n in ps['sourceFields'] else pc['fields'][n]) for n in SOURCE_NAMES} or s['comparisonFields']!={n:pc['fields'][n] for n in COMPARISON_NAMES}: raise ValueError('Paid input or postrelease comparison descriptor differs')
    if s['originalDLineage']!={'promotedDescriptor':pc['fields']['weightD'],'sourceView':pe['sourceViews']['D'],'originalParameterWitness':pv['originalParameterWitnesses'][pe['sourceViews']['D']['sourceParameter']]}: raise ValueError('Original D provenance differs')
    for item in items(p):
        if v['files'][item['path']]['sha256']!=item['sha256'] or (item.get('payloadSHA256') and v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']): raise ValueError('Full byte witness absent')
    seal={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json')}
    if read(p/'candidate-seal.json')!=seal or any(x!=sha(p/'candidate-seal.json') for x in [report['candidateSealSHA256'],v['candidateSealSHA256'],a['candidateSealSHA256']]): raise ValueError('Producer release differs')
    manifest=identity({'parentSealSHA256':s['parentSealSHA256'],'sourceFields':s['sourceFields'],'helperSHA256':r['sourceHashes']['mean_gate.py']})
    if manifest!=e['sourceManifestSHA256'] or not e['sourceRNGClosure'] or e['numericalProfile']!=s['numericalProfile']: raise ValueError('Source/profile closure differs')
    if {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in c['fields'].items()}!=abis() or {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in a['fields'].items()}!=analysis_abis(): raise ValueError('Explicit six producer/fifteen analysis ABIs differ')
    if len(e['typedActs'])!=1: raise ValueError('Exactly one producer Act required')
    act=e['typedActs'][0];src,dst,ev=act['sourceFrame'],act['outputFrame'],act['evidence']
    if src['frame_id']!=ev['input_frame_id'] or dst['frame_id']!=ev['output_frame_id'] or dst['parent_frame_id']!=src['frame_id'] or dst['program_fingerprint']!=r['requestSHA256'] or src['metadata']['sourceManifestSHA256']!=manifest: raise ValueError('Typed producer lineage differs')
    if ev['metrics']['outputPayloads']!={n:x['payloadSHA256'] for n,x in c['fields'].items()} or any(dst['slots'][n]['fingerprint']!=structured(x) for n,x in c['fields'].items()): raise ValueError('Typed producer output binding differs')
    if {n:x['fingerprint'] for n,x in act['sourceFrame']['slots'].items()}!={n:x['payloadSHA256'] for n,x in s['sourceFields'].items()}: raise ValueError('Source-to-producer join differs')
    meta=act['sourceFrame']['metadata']
    if meta['parentSealSHA256']!=s['parentSealSHA256'] or meta['parentOutputFrameID']!=s['parentOutputFrameID'] or meta['fixedAddress']!={'clock':3,'singleBlock':0}: raise ValueError('Fixed source metadata differs')
    guard=e['fileQuarantine'];allowed={x['path'] for x in s['sourceFields'].values()};denied=set(s['deniedNumericalInputs'])
    if set(guard['allowedFiles'])!=allowed or set(guard['deniedCanaries'])!=denied or allowed&denied or not set(guard['successfulFileOpenCounts']).issubset(allowed) or not {x['path'] for x in s['comparisonFields'].values()}.issubset(denied): raise ValueError('Producer quarantine differs')
    exited=read(p/'producer-exit.json')
    if exited['returncode']!=0 or exited['candidateSealSHA256']!=sha(p/'candidate-seal.json') or exited['requestSHA256']!=r['requestSHA256']: raise ValueError('Producer exit-before-analysis differs')
    if len(e['operations'])!=1: raise ValueError('Exactly one physical Down required')
    op=e['operations'][0]
    if (op['name'],op['inputShape'],op['weightShape'],op['outputShape'],op['dtype'],op['device'])!=('meanGateDown',[1,1536,9216],[3072,9216],[1,1536,3072],'torch.float64','cuda'): raise ValueError('Physical projection geometry differs')
    if op['inputPayloadSHA256']!=c['fields']['linearProduct']['payloadSHA256'] or op['weightPayloadSHA256']!=s['sourceFields']['weightD']['payloadSHA256'] or op['outputPayloadSHA256']!=c['fields']['linearOutput']['payloadSHA256']: raise ValueError('Physical input/map/result binding differs')
    if any(x!=43486543872 for x in [op['nominalMACs'],e['totalMatrixMACs'],report['totalMatrixMACs']]) or any(e[n]!=0 for n in ['newUpProjections','newOriginalCheckpointTensorLoads','newNativeReferences','newImages','newHeadCalls','newVAECalls']) or e['savedOriginalDownMapLoads']!=1 or e['savedOriginalDownMapBytes']!=226492416: raise ValueError('Actual complete work differs')
    streams=read(p/'provenance.json')['losslessStreams']
    if len(streams)!=1: raise ValueError('One complete row stream required')
    stream=streams[0];raw=gzip.decompress((p/stream['published']).read_bytes())
    if hashlib.sha256(raw).hexdigest()!=stream['sourceSHA256']: raise ValueError('Lossless rows differ')
    rows=[json.loads(x) for x in raw.splitlines()]
    if len(rows)!=1536 or [x['row'] for x in rows]!=list(range(1536)): raise ValueError('Complete signed row observations differ')
    return {'schema':'tce-mean-gate-mechanics-v1','typedActs':1,'sourceInputs':6,'producerABIs':6,'analysisABIs':15,'physicalDowns':1,
        'totalMatrixMACs':43486543872,'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'observations':1536,
        'scope':'Source custody/parent D lineage/isolation/typedjoins/actualwork/full bytes. Reduction/product/FP64 contraction semantics source-audited/tiny-tested, not independently numerically replayed on real arrays. No image/continuation/economical/zero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('root');p.add_argument('--destination');a=p.parse_args()
    print(json.dumps(publish(a.root,a.destination) if a.action=='publish' else globals()[a.action](a.root),sort_keys=True))
