"""Full saved-array bytes and scalar publication; no tensor arithmetic here."""
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
from gated_even_bound_study import bindings,PARENTS


def verify(root):
    from mrun.client.api import Api
    root,col,job,r=checked(root);p=root/'result';s=read(p/'inputs.json');report=read(p/'report.json');api=Api()
    for item in s['parents'].values():
        parent=api.json('GET','/api/jobs/'+item['jobID'])
        if parent['state']!='succeeded' or parent['payload_custody']!=item['custody'] or parent['result']['diagnostics']['cwd']!=item['root']:raise ValueError('Source scheduler custody differs')
    base=job['result']['diagnostics']['cwd'];items=[*s['fields'].values(),report['rowBounds']]
    items += [{'path':item['root']+'/response-result/'+n,'sha256':h} for item in s['parents'].values() for n,h in item['records'].items()]
    items += [{'path':base+'/response-result/'+x.name,'sha256':sha(x)} for x in p.iterdir()]
    unique={}
    for item in items:
        if item['path'] in unique and item['sha256']!=unique[item['path']]['sha256']:raise ValueError('Conflicting byte identities')
        unique[item['path']]={**unique.get(item['path'],{}),**item}
    spec={'items':list(unique.values()),'allowedRoots':[base,*s['sourceRoots']]}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(NPY_BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=180)
    if proc.returncode:raise RuntimeError(proc.stderr)
    v=json.loads(proc.stdout);v.update(schema='tce-gated-even-bound-bytes-v1',jobID=job['job_id'],reportSHA256=sha(p/'report.json'),verifierSourceSHA256=sha(__file__),templateSourceSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
        scope='Complete saved input/map/product/output NPY bytes and metadata/parent records, new rowbound array. Norm/inequality/interval arithmetic source-audited and exact rational/Decimal toy-tested; no independent real-array numerical replay, no cuBLAS rounding attestation.')
    save(root/'retained-array-verification.json',v)
    return {'files':len(v['files']),'bytes':v['totalBytesVerified'],'NPYPayloadWitnesses':len(v['npyPayloadWitnesses'])}


def publish(root,destination):
    root,col,job,r=checked(root);p=root/'result';v=read(root/'retained-array-verification.json')
    if v['verifierSourceSHA256']!=sha(__file__) or v['reportSHA256']!=sha(p/'report.json'):raise ValueError('Executed byte verification differs')
    out=Path(destination);out.mkdir(exist_ok=False,parents=True);streams=[]
    for x in p.iterdir():
        if x.suffix=='.jsonl':
            raw=x.read_bytes();target=out/(x.name+'.gz');target.write_bytes(gzip.compress(raw,mtime=0))
            if gzip.decompress(target.read_bytes())!=raw:raise ValueError('Lossless stream differs')
            streams.append({'source':x.name,'sourceSHA256':sha(x),'sourceBytes':len(raw),'published':target.name,'publishedSHA256':sha(target),'publishedBytes':target.stat().st_size})
        else:shutil.copyfile(x,out/x.name)
    shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    save(out/'scheduler.json',{k:job[k] for k in ['job_id','state','reservation','admission','payload_custody','result']})
    save(out/'provenance.json',{'schema':'tce-gated-even-bound-publication-v1','jobID':job['job_id'],'requestSHA256':r['requestSHA256'],'payloadCustody':col['custody'],'completeCollectedInventory':col['inventory'],'losslessStreams':streams,
        'scope':'Saved-only conditional operator attribution. Zero new learned products/evaluations, but original maps/source/H/previous projections/history paid. No serving generator, stateupdate or image.'})
    save(out/'files.json',{x.name:sha(x) for x in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(x.stat().st_size for x in out.iterdir())}


def audit(p):
    from publish_odd_gate import audit as parent_audit
    from gated_even_bound import MODEL
    p=Path(p);r,s,report,v=[read(p/n) for n in ['request.json','inputs.json','report.json','retained-array-verification.json']]
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n,h in read(p/'files.json').items():
        if sha(p/n)!=h:raise ValueError('Publication changed')
    for n in r['ownedSources']:
        if sha(Path(__file__).with_name(n))!=r['sourceHashes'][n]:raise ValueError('Executed source changed:'+n)
    root=Path(__file__).resolve().parent.parent;parent_audit(root/'docs/results/odd-gate')
    dirs={name:root/'artifacts'/folder/'result' for name,folder in PARENTS.items()}
    if s['fields']!=bindings(dirs['quadratic'],dirs['linear'],dirs['covariance']):raise ValueError('Exact retained-array ancestry differs')
    for name,item in s['parents'].items():
        if any(sha(dirs[name]/n)!=h for n,h in item['records'].items()):raise ValueError('Source metadata differs')
    if v['verifierSourceSHA256']!=sha(__file__) or v['templateSourceSHA256']!=sha(Path(__file__).with_name('publish_sparse_text_kv.py')) or v['reportSHA256']!=sha(p/'report.json'):raise ValueError('Fullbyte verifier/report differs')
    for item in [*s['fields'].values(),report['rowBounds']]:
        if v['files'][item['path']]['sha256']!=item['sha256'] or v['npyPayloadWitnesses'][item['path']]['payloadSHA256']!=item['payloadSHA256']:raise ValueError('Fullarray witness absent')
    if sha(p/'frozen-plan.md')!=r['sourceHashes']['frozen-plan.md'] or report['numericalModel']!=MODEL or report['numericalDiagnostics']['historicalContractionModelAttested'] is not False:raise ValueError('Mathematical scope differs')
    expected={'newMatrixMACs':0,'newLearnedProjections':0,'newNeuralEvaluations':0,'newCheckpointLoads':0,'newImages':0,'newFullProductArrays':0,'newRetainedArrays':1}
    if report['physicalWork']!=expected or report['rowBounds']['shape']!=[1536,10]:raise ValueError('Physical work/output ABI differs')
    provenance=read(p/'provenance.json');stream=provenance['losslessStreams'][0];raw=gzip.decompress((p/stream['published']).read_bytes())
    if hashlib.sha256(raw).hexdigest()!=stream['sourceSHA256']:raise ValueError('Lossless observations differ')
    rows=[json.loads(line) for line in raw.splitlines()]
    if [x['row'] for x in rows]!=list(range(1536)) or any(x['stratum']!=('text' if x['row']<512 else 'image') for x in rows) or report['retainedAdverseTextRows']!=[rows[3],rows[35]]:raise ValueError('Complete scalar observations differ')
    if report['requestSHA256']!=r['requestSHA256']:raise ValueError('Result/request binding differs')
    return {'schema':'tce-gated-even-bound-mechanics-v1','sourceInputs':len(s['fields']),'observations':len(rows),'retainedArrays':1,'fullByteFiles':len(v['files']),'NPYPayloadWitnesses':len(v['npyPayloadWitnesses']),
        'newLearnedMatrixMACs':0,'scope':'Source/operand ancestry/fullbytes/outputABI/lossless rows/work/source audit joins. Exact rational/Decimal tiny references separate; actual contraction model unverified, all eta conclusions conditional. No semantic/image/generalization/zero-neural serving certification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish','audit']);p.add_argument('source');p.add_argument('destination',nargs='?');a=p.parse_args()
    value=publish(a.source,a.destination) if a.action=='publish' else globals()[a.action](a.source)
    print(json.dumps(value,indent=2))
