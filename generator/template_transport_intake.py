"""Fresh six-parent full-byte/header intake, with three explicitly admitted templates."""
import json
from pathlib import Path
import shlex
import subprocess
from encoder_atoms import read,save,sha
from own_successor_intake import BYTES
from projected_complement import fields
from dense_corner_study import validate_corner_metadata

TEMPLATES={'gu_basis':([2,3072,128],'9a19e4c262e770f878d2e3156fb11321f49404d2aefeafed129debbd7a71e467'),
 'gu_projectedGU':([2,128,18432],'dd26d8ba4bd4df753a93980615fb019d731156555b5247a95f5192bbfc43ed69'),
 'rowBasis':([1,1536,128],'1ef7c350f15904a6e58d8f1489f2df774d143ca16271027c5fa422a5049e973f')}


def validate_metadata(i):
    validate_corner_metadata(i)
    root=Path(__file__).resolve().parent.parent
    p=root/'artifacts/dense-corner-v1/result'
    for name in ['request.json','inputs.json','candidate.json','execution.json','candidate-seal.json','producer-exit.json','report.json']:
        remote=i['parents']['89']['root']+'/response-result/'+name
        if sha(p/name)!=i['metadataFiles'][remote]:raise ValueError('Authenticated89metadata differs')
    s=read(p/'inputs.json');c=read(p/'candidate.json')['fields']
    if i['sourceFields']!=s['sourceFields'] or i['originalParameters']!=s['originalParameters'] or i['comparisonFields']['denseOwnOutgoing']!=c['exit']:
        raise ValueError('Dense-own comparator association differs')
    templates=read(root/'artifacts/matched-parent-v1/result/candidate.json')['fields']
    for name,(shape,digest) in TEMPLATES.items():
        item=i['templateFields'][name]
        if item!=templates[name] or item['shape']!=shape or item['logicalDtype']!='torch.float64' or item['sha256']!=digest:
            raise ValueError('Original retained template differs:'+name)


def execute(destination):
    from mrun.client.api import Api
    from publish_source_native import checked
    root=Path(__file__).resolve().parent.parent
    parent,col,j89,r89=checked(root/'artifacts/dense-corner-v1');p=parent/'result'
    base=read(p/'intake.json')
    if sha(p/'intake.json')!=read(p/'inputs.json')['intakeSHA256']:raise ValueError('Original intake locator differs')
    jobs={**base['parents'],'89':{'jobID':j89['job_id'],'custody':j89['payload_custody'],'root':j89['result']['diagnostics']['cwd']}}
    api=Api()
    for stored in jobs.values():
        actual=api.json('GET','/api/jobs/'+stored['jobID'])
        if actual['state']!='succeeded' or actual['payload_custody']!=stored['custody'] or actual['result']['diagnostics']['cwd']!=stored['root']:
            raise ValueError('Original successful custody differs')
    c88=read(root/'artifacts/matched-parent-v1/result/candidate.json')['fields']
    templates={n:c88[n] for n in TEMPLATES}
    comparison={**base['comparisonFields'],'denseOwnOutgoing':read(p/'candidate.json')['fields']['exit']}
    metadata={**base['metadataFiles'],**{jobs['89']['root']+'/response-result/'+n:sha(p/n) for n in
        ['request.json','inputs.json','candidate.json','execution.json','candidate-seal.json','producer-exit.json','report.json']}}
    values=[*base['sourceFields'].values(),*templates.values(),*list(fields(comparison)),
            *[{'path':n,'sha256':h} for n,h in metadata.items()]]
    unique={x['path']:x for x in values}
    spec={'allowedRoots':[j['root'] for j in jobs.values()],'items':list(unique.values()),
          'checkpoint':str(Path(base['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),'parameters':base['originalParameters']}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],
                        input=json.dumps(spec),text=True,capture_output=True,timeout=240)
    if proc.returncode:raise RuntimeError(proc.stderr)
    value=json.loads(proc.stdout)
    value.update({k:base[k] for k in ['sourceFields','originalParameters','modelRoot','sourceConfig','frozenConsumerSHA256']})
    value.update(schema='tce-template-transport-intake-v1',comparisonFields=comparison,templateFields=templates,
                 parents=jobs,metadataFiles=metadata,verifierSHA256=sha(__file__),
                 byteTemplateSHA256=sha(Path(__file__).with_name('own_successor_intake.py')),
                 scope='Fresh successful70/73/86/87/88/89 custody,complete source/template/comparison/metadata bytes and original four parameters. SSH bytes/headers only,not numerical replay.')
    validate_metadata(value);save(Path(destination),value)
    return {'files':len(value['files']),'NPY':len(value['npyPayloadWitnesses']),'fileBytes':value['totalBytesVerified'],'parameterBytes':sum(x['bytes'] for x in value['originalParameters'])}


if __name__=='__main__':
    import sys
    print(json.dumps(execute(sys.argv[1])))
