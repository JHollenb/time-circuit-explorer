"""Complete bytes, original gate slice, isolated inputs and physical work joins."""
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
from odd_gate import SOURCE_NAMES,COMPARISON_NAMES,PARAMETER,abis
from odd_gate_analysis import analysis_abis

BYTES=NPY_BYTES.split('print(json.dumps')[0]+r'''
with Path(s['checkpoint']).open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));base=8+n
 item=s['loaded'][0];entry=header[item['name']];a,b=entry['data_offsets'];f.seek(base+a);raw=f.read(b-a)
 if entry['dtype']!='BF16' or entry['shape']!=item['shape'] or [a,b]!=item['dataOffsets'] or b-a!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['payloadSHA256']:raise ValueError('Original fused gate source differs')
 derived=raw[9216*3072*2:18432*3072*2];v=s['gateView']
 if v['rows']!=[9216,18432] or v['shape']!=[9216,3072] or len(derived)!=v['bytes'] or hashlib.sha256(derived).hexdigest()!=v['payloadSHA256']:raise ValueError('Physical gate view differs')
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,'originalParameterWitness':item,'gateSourceViewWitness':v}))
'''


def items(p):
    s=read(p/'inputs.json');c=read(p/'candidate.json');a=read(p/'analysis-fields.json')
    return [*s['sourceFields'].values(),*s['comparisonFields'].values(),*c['fields'].values(),*a['fields'].values(),
        *[{'path':str(Path(s['modelRoot'])/n),'sha256':h} for n,h in s['imageAssets'].items()],
        *[{'path':s['parentRoot']+'/response-result/'+n,'sha256':h} for n,h in s['parentRecords'].items()],
        *[{'path':s['covarianceRoot']+'/response-result/'+n,'sha256':h} for n,h in s['covarianceRecords'].items()]]


def verify(root):
    from mrun.client.api import Api
    root,col,job,r=checked(root);p=root/'result';s=read(p/'inputs.json');e=read(p/'execution.json');api=Api()
    for prefix in ['parent','covariance']:
        j=api.json('GET','/api/jobs/'+s[prefix+'JobID'])
        if j['state']!='succeeded' or j['payload_custody']!=s[prefix+'Custody'] or j['result']['diagnostics']['cwd']!=s[prefix+'Root']:raise ValueError('Paid source custody differs')
    base=job['result']['diagnostics']['cwd'];values=items(p)+[{'path':base+'/response-result/'+n.name,'sha256':sha(n)} for n in p.iterdir()];unique={}
    for x in values:
        if x['path'] in unique and unique[x['path']]['sha256']!=x['sha256']:raise ValueError('Conflicting input identities')
        unique[x['path']]={**unique.get(x['path'],{}),**x}
    spec={'items':list(unique.values()),'allowedRoots':[base,s['modelRoot'],*s['sourceRoots']],'checkpoint':str(Path(s['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),'loaded':e['loadedParameters'],'gateView':e['gateSourceView']}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=240)
    if proc.returncode:raise RuntimeError(proc.stderr)
    v=json.loads(proc.stdout);v.update(schema='tce-odd-gate-bytes-v1',jobID=job['job_id'],candidateSealSHA256=sha(p/'candidate-seal.json'),verifierSourceSHA256=sha(__file__),byteTemplateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
        scope='Complete admitted/comparison/producer/analysis NPY files and metadata; original fusedBF16parameter/gate rowview bytes. Promotions, contraction/means and odd/even product arithmetic source-audited/tiny-tested, not independently numerically replayed by byte verifier.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified'],'NPYPayloadWitnesses':len(v['npyPayloadWitnesses'])}


def publish(root,destination):
    root,col,job,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['candidateSealSHA256']!=sha(p/'candidate-seal.json'):raise ValueError('Executed verifier/release differs')
    out=Path(destination);out.mkdir(parents=True,exist_ok=False);streams=[]
    for n in p.iterdir():
        if n.suffix=='.jsonl':
            raw=n.read_bytes();dst=out/(n.name+'.gz');dst.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(dst.read_bytes())!=raw:raise ValueError('Lossless publication differs')
            streams.append({'source':n.name,'sourceSHA256':sha(n),'sourceBytes':len(raw),'published':dst.name,'publishedSHA256':sha(dst),'publishedBytes':dst.stat().st_size})
        else:shutil.copyfile(n,out/n.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json');save(out/'scheduler.json',{k:job[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-odd-gate-publication-v1','jobID':job['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':col['custody'],'completeCollectedInventory':col['inventory'],'losslessStreams':streams,
        'scope':'One fixed paid quadratic producer and post-exit H/F product-only account; no new native block/image/economical compiler/zero-neural result.'})
    save(out/'files.json',{n.name:sha(n) for n in out.iterdir()});return {'files':len(list(out.iterdir())),'bytes':sum(n.stat().st_size for n in out.iterdir())}


def audit(p):
    from publish_mean_gate import audit as parent_audit
    p=Path(p);r,s,c,e,a,report,v=[read(p/n) for n in ['request.json','inputs.json','candidate.json','execution.json','analysis-fields.json','report.json','retained-array-verification.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Publication changed')
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]:raise ValueError('Executed source changed:'+n)
    if v['verifierSourceSHA256']!=sha(__file__) or v['byteTemplateSourceSHA256']!=sha(Path(__file__).with_name('publish_sparse_text_kv.py')):raise ValueError('Verifier source changed')
    root=Path(__file__).resolve().parent.parent;parent=root/'docs/results/mean-gate';cov=root/'docs/results/own-state-gated-covariance';parent_audit(parent)
    ps=read(parent/'inputs.json');pc=read(parent/'candidate.json');pe=read(parent/'execution.json');pa=read(parent/'analysis-fields.json');cs=read(cov/'inputs.json');ce=read(cov/'execution.json')
    for directory,key in [(parent,'parentRecords'),(cov,'covarianceRecords')]:
        if any(sha(directory/n)!=h for n,h in s[key].items()):raise ValueError('Parent snapshot differs')
    expected={n:ps['sourceFields'][n] for n in SOURCE_NAMES if n not in ['normalized','linearOutput']};expected.update(normalized=cs['sourceFields']['normalized'],linearOutput=pc['fields']['linearOutput'])
    compares={n:ps['comparisonFields'][n] for n in ['centeredOutput','policyOutput','totalOutput']};compares.update(activation=ps['sourceFields']['activation'],activationMean=pc['fields']['activationMean'],changingProduct=pa['fields']['changingProduct'])
    if s['sourceFields']!=expected or s['comparisonFields']!=compares or s['originalDLineage']!=ps['originalDLineage'] or s['originalParameters']!=[next(x for x in ce['loadedParameters'] if x['name']==PARAMETER)]:raise ValueError('Source/target/original map lineage differs')
    if s['parentSealSHA256']!=sha(parent/'candidate-seal.json') or s['covarianceSealSHA256']!=sha(cov/'candidate-seal.json') or s['parentOutputFrameID']!=pe['typedActs'][0]['outputFrame']['frame_id']:raise ValueError('Typed parent release differs')
    for item in items(p):
        if v['files'][item['path']]['sha256']!=item['sha256'] or (item.get('payloadSHA256') and v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']):raise ValueError('Full byte witness absent')
    seal={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(p/'candidate.json'),'executionSHA256':sha(p/'execution.json')}
    if read(p/'candidate-seal.json')!=seal or any(x!=sha(p/'candidate-seal.json') for x in [report['candidateSealSHA256'],v['candidateSealSHA256'],a['candidateSealSHA256']]):raise ValueError('Producer release differs')
    manifest=identity({'parentSealSHA256':s['parentSealSHA256'],'covarianceSealSHA256':s['covarianceSealSHA256'],'sourceFields':s['sourceFields'],'originalParameters':s['originalParameters'],'helperSHA256':r['sourceHashes']['odd_gate.py']})
    if e['sourceManifestSHA256']!=manifest or e['loadedParameters']!=s['originalParameters'] or not e['sourceParameterRNGClosure'] or e['numericalProfile']!=s['numericalProfile']:raise ValueError('Source parameter/profile closure differs')
    if {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in c['fields'].items()}!=abis() or {n:{'shape':x['shape'],'dtype':x['logicalDtype']} for n,x in a['fields'].items()}!=analysis_abis():raise ValueError('Eleven/fourteen output ABIs differ')
    if len(e['typedActs'])!=1:raise ValueError('One producer Act required')
    act=e['typedActs'][0];src,dst,ev=act['sourceFrame'],act['outputFrame'],act['evidence']
    if src['frame_id']!=ev['input_frame_id'] or dst['frame_id']!=ev['output_frame_id'] or dst['parent_frame_id']!=src['frame_id'] or dst['program_fingerprint']!=r['requestSHA256'] or src['metadata']['sourceManifestSHA256']!=manifest:raise ValueError('Typed lineage differs')
    if ev['metrics']['outputPayloads']!={n:x['payloadSHA256'] for n,x in c['fields'].items()} or any(dst['slots'][n]['fingerprint']!=structured(x) for n,x in c['fields'].items()):raise ValueError('Typed output binding differs')
    inputs={n:x['payloadSHA256'] for n,x in s['sourceFields'].items()};inputs.update(G=e['gateSourceView']['payloadSHA256'],originalGateParameter=s['originalParameters'][0]['payloadSHA256'])
    if {n:x['fingerprint'] for n,x in src['slots'].items()}!=inputs or src['metadata']['gateSourceView']!=e['gateSourceView'] or v['originalParameterWitness']!=s['originalParameters'][0] or v['gateSourceViewWitness']!=e['gateSourceView']:raise ValueError('Physical original gate binding differs')
    guard=e['fileQuarantine'];allowed={x['path'] for x in s['sourceFields'].values()}|{str(Path(s['modelRoot'])/n) for n in s['imageAssets']};denied=set(s['deniedNumericalInputs'])
    if set(guard['allowedFiles'])!=allowed or set(guard['deniedCanaries'])!=denied or allowed&denied or not set(guard['successfulFileOpenCounts']).issubset(allowed) or not {x['path'] for x in s['comparisonFields'].values()}.issubset(denied):raise ValueError('Actual A/error quarantine differs')
    exited=read(p/'producer-exit.json')
    if exited['returncode']!=0 or exited['candidateSealSHA256']!=sha(p/'candidate-seal.json') or exited['requestSHA256']!=r['requestSHA256']:raise ValueError('Producer exit-before-analysis differs')
    ops=e['operations'];wanted=[('fullGate',[1,1536,3072],[9216,3072],[1,1536,9216]),('quadraticDown',[1,1536,9216],[3072,9216],[1,1536,3072])]
    if [(x['name'],x['inputShape'],x['weightShape'],x['outputShape']) for x in ops]!=wanted or any(x['dtype']!='torch.float64' or x['device']!='cuda' for x in ops):raise ValueError('Two physical projections differ')
    for op,inp,weight,out in [(ops[0],c['fields']['normalizedFP64'],c['fields']['weightG'],c['fields']['gateReference']),(ops[1],c['fields']['quadraticProduct'],s['sourceFields']['weightD'],c['fields']['quadraticOutput'])]:
        if (op['inputPayloadSHA256'],op['weightPayloadSHA256'],op['outputPayloadSHA256'])!=(inp['payloadSHA256'],weight['payloadSHA256'],out['payloadSHA256']):raise ValueError('Physical input/map/output binding differs')
    if sum(x['nominalMACs'] for x in ops)!=86973087744 or e['totalMatrixMACs']!=86973087744 or report['totalMatrixMACs']!=86973087744 or any(e[n]!=0 for n in ['newUpProjections','newNativeReferences','newImages','newHeadCalls','newVAECalls']) or any(e[n]!=1 for n in ['newGateProjections','newDownProjections','newOriginalCheckpointTensorLoads','savedOriginalDownMapLoads']) or e['savedOriginalDownMapBytes']!=226492416:raise ValueError('Complete paid work differs')
    streams=read(p/'provenance.json')['losslessStreams']
    if len(streams)!=1:raise ValueError('Complete row stream required')
    stream=streams[0];raw=gzip.decompress((p/stream['published']).read_bytes())
    if hashlib.sha256(raw).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless row hash differs')
    rows=[json.loads(x) for x in raw.splitlines()]
    if len(rows)!=1536 or [x['row'] for x in rows]!=list(range(1536)):raise ValueError('Complete row coverage differs')
    return {'schema':'tce-odd-gate-mechanics-v1','typedActs':1,'sourceInputs':9,'producerABIs':11,'analysisABIs':14,'physicalFP64Projections':2,'totalMatrixMACs':86973087744,'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),'observations':1536,
        'scope':'Original gate view/parentD/inputisolation/exit/typed/physicalwork/bytes. Finite promotions/contractions/means/product identities source-audited/tiny-tested, not independently real-array numerically replayed. No image/economical/zero-neural certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('root');p.add_argument('--destination');a=p.parse_args();print(json.dumps(publish(a.root,a.destination) if a.action=='publish' else globals()[a.action](a.root),sort_keys=True))
