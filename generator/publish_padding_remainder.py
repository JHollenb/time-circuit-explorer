"""Byte-authenticated publication of the immutable postseal MLP diagnostic."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
from encoder_atoms import read,save,sha
from publish_context_statistics import checked_collection,publish
from publish_source_context import REMOTE


def verify(root):
    root=Path(root).resolve();c,j,r=checked_collection(root,current_workers=False)
    for n in r['ownedSources']:
        if sha(Path(__file__).parent/n)!=r['sourceHashes'][n]: raise ValueError('Executed source differs')
    p=root/'result';s=read(p/'inputs.json');base=j['result']['diagnostics']['cwd'];primary=s['primaryRoot'];source=s['source']
    if read(p/'report.json')['candidateSealSHA256']!=r['candidateSealSHA256']:raise ValueError('Primary release differs')
    items=[*s['compiled']['fields'].values(),*s['primaryRecords'].values()]
    for group in [read(p/'fields.json'),s['encoder']['firstMLPFields'],s['diagnostic']['fields']]:
        for fs in group.values():items+=list(fs.values())
    items += [{'path':base+'/response-result/'+f.name,'sha256':sha(f)} for f in p.iterdir()]
    spec={'allowedRoots':[base,primary,source['lexicalStore']],'items':items}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(REMOTE)],
        input=json.dumps(spec),text=True,capture_output=True,check=True,timeout=180)
    v=json.loads(proc.stdout);v.update(schema='tce-padding-remainder-bytes-v1',jobID=j['job_id'],candidateSealSHA256=r['candidateSealSHA256'],
        verifierSourceSHA256=sha(__file__),scope='Complete primary own normalized/residual/write inputs, compiler arrays and native postseal diagnostic, eight new smooth/linear/curvature/finite arrays, all compact primary and addon records. Original checkpoint reads authenticated by worker receipts.')
    save(root/'retained-array-verification.json',v);return {'files':len(v['files']),'bytes':v['totalBytesVerified']}


def publication(root,destination):
    root=Path(root);r=read(root/'request.json');v=read(root/'retained-array-verification.json')
    if v['candidateSealSHA256']!=r['candidateSealSHA256'] or v['verifierSourceSHA256']!=sha(__file__):raise ValueError('Byte verification differs')
    result=publish(root,destination)
    # Generic publisher binds custody and all source/stream bytes. This receipt
    # additionally binds the complete saved receivers and addon output arrays.
    out=Path(destination);save(out/'retained-array-verification.json',v)
    save(out/'diagnostic-scope.json',{'primaryJobID':r['parentJobID'],'primaryRequestSHA256':r['parentRequestSHA256'],
        'candidateSealSHA256':r['candidateSealSHA256'],'scope':'Additive postseal diagnostic only. No new image, candidate change, native BF16 MLP or serving shortcut.'})
    (out/'files.json').unlink();save(out/'files.json',{f.name:sha(f) for f in out.iterdir()})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','publish']);p.add_argument('run');p.add_argument('destination',nargs='?');a=p.parse_args()
    print(json.dumps(verify(a.run) if a.action=='verify' else publication(a.run,a.destination)))
