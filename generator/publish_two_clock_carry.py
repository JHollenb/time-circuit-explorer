"""Complete byte custody for the saved-only radial-response diagnostic."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read,save,sha
from publish_context_statistics import checked_collection,publish
from publish_source_context import REMOTE


def verify(root):
    root=Path(root).resolve();c,j,r=checked_collection(root)
    p=root/'result';s=read(p/'inputs.json');base=j['result']['diagnostics']['cwd']
    if read(p/'report.json')['candidateSealSHA256']!=r['candidateSealSHA256']:raise ValueError('Primary release differs')
    items=list(s['primaryRecords'].values())
    for lanes in s['fields'].values():
        for fs in lanes.values():items+=list(fs.values())
    for fs in read(p/'fields.json').values():items+=list(fs.values())
    items += [{'path':base+'/response-result/'+f.name,'sha256':sha(f)} for f in p.iterdir()]
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps({'allowedRoots':[base,*s['allowedRoots']],'items':items}),text=True,capture_output=True,check=True,timeout=180)
    v=json.loads(proc.stdout);v.update(schema='tce-two-clock-carry-bytes-v1',jobID=j['job_id'],candidateSealSHA256=r['candidateSealSHA256'],
        verifierSourceSHA256=sha(__file__),scope='Complete authenticated NN/HN Z3/v3/Z4 and independently saved finite remainders, all30newFP64vectors and compact primary/addon records; no source weights or new images. Scientific arrays remain near compute.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified']}


def publication(root,destination):
    root=Path(root);r=read(root/'request.json');v=read(root/'retained-array-verification.json')
    if v['candidateSealSHA256']!=r['candidateSealSHA256'] or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Byte verification differs')
    publish(root,destination);out=Path(destination);shutil.copyfile(root/'retained-array-verification.json',out/'retained-array-verification.json')
    (out/'files.json').unlink();save(out/'files.json',{f.name:sha(f) for f in out.iterdir()})
    return {'files':len(list(out.iterdir())),'bytes':sum(f.stat().st_size for f in out.iterdir())}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(verify(a.run) if a.action=='verify' else publication(a.run,a.destination)))
