"""Authenticate one already-sealed complete image comparator by bytes only."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
from encoder_atoms import read,save,sha
from publish_sparse_text_kv import NPY_BYTES


def execute(destination):
    from publish_complete_masked import checked
    from mrun.client.api import Api
    root=Path(__file__).resolve().parent.parent;parent,c,j,r=checked(root/'artifacts/complete-masked-image-v1');p=parent/'result'
    s=read(p/'source.json');enc=read(p/'encoder-full.json');images=read(p/'images-full.json');seal=read(p/'full-seal.json');prior=read(parent/'retained-array-verification.json')
    if sha(p/'images-full.json')!='2c75863cba8c8cde576fe70f344b8673c7d90109c107570bcf2a66282ca91c1f' or sha(p/'full-seal.json')!='d666935bd61e43950ff3d74b8605fb9e823c77da4496ef325b61a822ddce3a1b':raise ValueError('Named native comparator changed')
    if seal['encoderSHA256']!=sha(p/'encoder-full.json') or seal['imagesSHA256']!=sha(p/'images-full.json') or seal['encoderFields']['A']!=enc['fields']['A'] or seal['imageFields']['A']!=images['fields']['A']:
        raise ValueError('Native comparator seal differs')
    noise_job=Api().json('GET','/api/jobs/'+s['noiseParentJobID'])
    if noise_job['state']!='succeeded' or noise_job['config']['requestSHA256']!=s['noiseParentRequestSHA256']:raise ValueError('Original noise lineage differs')
    latest=read(root/'docs/results/terminal-liveness/inputs.json');model=s['lexicalSource']['modelRoot']
    if model!=latest['modelRoot'] or images['numericalProfile']!=latest['numericalProfile'] or {'torch':images['torchVersion'],'diffusers':images['diffusersVersion']}!=latest['versions']:raise ValueError('Terminal/native image compatibility differs')
    if any(latest['nativeSources'].get(n)!=h for n,h in s['nativeSources'].items()):raise ValueError('Original native source overlap differs')
    if any(latest['imageAssets'][n]!=h for n,h in s['imageAssets'].items() if n in latest['imageAssets']):raise ValueError('Original terminal checkpoint/config differs')
    base=j['result']['diagnostics']['cwd'];items=[enc['fields']['A']['E'],*images['fields']['A'].values(),images['images']['A']]+[s['noise'][n] for n in ['raw','packed','ids']]
    for item in items:
        name=item['path']
        if name not in prior['files'] and item==images['images']['A']:name=base+'/response-result/full-A.png'
        if prior['files'][name]['sha256']!=item['sha256']:raise ValueError('Prior complete-byte custody differs')
    items += [{'path':str(Path(model)/n),'sha256':h} for n,h in s['imageAssets'].items()]
    items += [{'path':base+'/response-result/'+n,'sha256':sha(p/n)} for n in ['source.json','encoder-full.json','images-full.json','full-seal.json','request.json']]
    spec={'allowedRoots':[base,s['noiseParentRoot'],model],'items':items}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','beast','python3 -c '+shlex.quote(NPY_BYTES)],input=json.dumps(spec),text=True,capture_output=True,timeout=240)
    if proc.returncode:raise RuntimeError(proc.stderr)
    out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False);bytes_receipt=json.loads(proc.stdout)
    result={'schema':'tce-terminal-trajectory-intake-v1','case':'A','nativeJobID':j['job_id'],'nativeCustody':c['custody'],'noiseJobID':noise_job['job_id'],'noiseCustody':noise_job['payload_custody'],
        'modelRoot':model,'E':enc['fields']['A']['E'],'noise':s['noise'],'targets':{'fields':images['fields']['A'],'image':images['images']['A']},'schedule':images['schedules']['A'],
        'sourceRecords':{n:sha(p/n) for n in ['source.json','encoder-full.json','images-full.json','full-seal.json','request.json']},'priorByteVerificationSHA256':sha(parent/'retained-array-verification.json'),
        'nativeSources':latest['nativeSources'],'imageAssets':s['imageAssets'],'versions':latest['versions'],'numericalProfile':latest['numericalProfile'],'attentionProcessors':images['attentionProcessors'],
        'intakeSourceSHA256':sha(__file__),'byteTemplateSourceSHA256':sha(Path(__file__).with_name('publish_sparse_text_kv.py')),'completeByteVerification':bytes_receipt,
        'scope':'Full raw-byte/NPY/header/custody intake only. No tensor values hydrated locally or model/scientific numerical calculation; one historical comparator, no new native reference.'}
    save(out/'intake.json',result)
    return {'nativeJobID':j['job_id'],'files':len(bytes_receipt['files']),'bytes':bytes_receipt['totalBytesVerified'],'NPYPayloadWitnesses':len(bytes_receipt['npyPayloadWitnesses'])}


if __name__=='__main__':print(json.dumps(execute(sys.argv[1]),indent=2))
