"""Metadata-only census of source-bound retained FLUX StateCuts."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
from encoder_atoms import read,save,sha

REMOTE=r'''
import hashlib,json,sys
from pathlib import Path
s=json.load(sys.stdin);root=Path(s['root']);rows=[];matches=[]
for name,expected in s['cuts'].items():
 p=root/name
 if not p.is_file() or p.is_symlink():raise ValueError('Cut must be an ordinary retained file')
 raw=p.read_bytes()
 if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:raise ValueError('Original cut bytes differ')
 cut=json.loads(raw);point=cut['execution_point'];rows.append({'cutID':p.stem,'path':str(p),'sha256':expected['sha256'],'bytes':len(raw),'point':point,'parent':cut.get('parent'),'slots':list(cut['slots'])})
 if point.get('phase')=='single' and point.get('index')==19:matches.append({'path':str(p),'sha256':expected['sha256'],'cut':cut})
census={}
for row in rows:
 point=row['point'];key=str((point.get('logical_step'),point.get('phase'),point.get('index'),point.get('next_operation')));census[key]=census.get(key,0)+1
print(json.dumps({'rows':rows,'pointCensus':census,'terminalSingle19Cuts':matches,'scope':'Read-only JSON metadata and full-byte hashes; no tensor loads or numerical/model computation.'}))
'''


def execute(destination):
    from mrun.client.api import Api
    workspace=Path(__file__).resolve().parents[2]
    family=workspace/'artifacts/mdb-computation-family-20261107/fresh-family-02'
    p=family/'compact';original=read(family/'scheduler-job.json');job=Api().json('GET','/api/jobs/'+original['job_id'])
    if job['state']!='succeeded' or job['payload_custody']!=original['payload_custody']:raise ValueError('Source job custody differs')
    manifest=read(p/'files.json');cuts={n:v for n,v in manifest['files'].items() if n.startswith('state-store/cuts/')}
    source=read(Path(__file__).resolve().parent.parent/'docs/results/source-reader/inputs.json')
    remote=str(Path(source['sourceCuts']['P']['cutFile']['path']).parents[2])
    if source['sourceJobID']!=job['job_id'] or source['sourceCustody']!=job['payload_custody']:raise ValueError('Qualified source-root lineage differs')
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps({'root':remote,'cuts':cuts}),text=True,capture_output=True,timeout=90)
    if proc.returncode:raise RuntimeError(proc.stderr)
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False)
    save(out/'state-cut-census.json',{'schema':'tce-terminal-liveness-intake-v1','sourceJobID':job['job_id'],'sourceCustody':job['payload_custody'],
        'sourceRoot':remote,'sourceMetadata':{n:sha(p/n) for n in ['files.json','state-index.json','source-roots.json','native-contract.json']},
        'intakeSourceSHA256':sha(__file__),**json.loads(proc.stdout)})
    return {'sourceJobID':job['job_id'],'cutFilesVerified':len(cuts),'terminalSingle19Cuts':len(read(out/'state-cut-census.json')['terminalSingle19Cuts'])}


if __name__=='__main__':print(json.dumps(execute(sys.argv[1])))
