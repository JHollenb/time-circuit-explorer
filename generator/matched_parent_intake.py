"""Fresh byte/header-only custody for the four authenticated predecessor jobs."""
import json
from pathlib import Path
import shlex
import subprocess
from encoder_atoms import read,save,sha
from own_successor_intake import BYTES,PARAMETERS


def execute(destination):
    from mrun.client.api import Api
    from publish_source_native import checked
    root=Path(__file__).resolve().parent.parent
    previous,collection,j87,request=checked(root/'artifacts/own-successor-v1')
    directories={'87':previous/'result','86':root/'artifacts/receiver-weighted-row-output-v1/result',
                 '70':root/'artifacts/image-partition-moments-v1/result','73':root/'artifacts/matched-single-write-v1/result'}
    jobs={'87':j87,**{tag:read(root/('docs/results/'+name+'/scheduler.json')) for tag,name in
                       [('86','receiver-weighted-row-output'),('70','image-partition-moments'),('73','matched-single-write')]}}
    api=Api()
    for stored in jobs.values():
        actual=api.json('GET','/api/jobs/'+stored['job_id'])
        if actual['state']!='succeeded' or actual['payload_custody']!=stored['payload_custody'] or actual['result']['diagnostics']['cwd']!=stored['result']['diagnostics']['cwd']:
            raise ValueError('Original successful identity/custody differs')
    c70=read(directories['70']/'candidate.json')['fields']
    c86=read(directories['86']/'candidate.json')['fields']
    c87=read(directories['87']/'candidate.json')['fields']
    s87=read(directories['87']/'inputs.json');s70=read(directories['70']/'inputs.json')
    e70=read(directories['70']/'execution.json');e87=read(directories['87']/'execution.json')
    block=next(x for x in e70['blockRecords'] if (x['kind'],x['index'],x['step'])==('single',1,0))
    dense=read(directories['73']/'replay-dense.json')['fields'];s73=read(directories['73']/'inputs.json')
    source={n:c70['b_single0_'+n+'Out0'] for n in ['text','image']}
    source.update({n:c70[key] for n,key in [('mod','singleMod0'),('cosine','cosine0'),('sine','sine0')]})
    comparison={'ownIncoming':{n:c86[n+'Out'] for n in ['text','image']},
                'ownOutgoing':{n:c87[n+'Out'] for n in ['text','image']},
                'parentOutgoing':{n:c70['b_single1_'+n+'Out0'] for n in ['text','image']}}
    for n in ['text','image']:
        if len({source[n]['payloadSHA256'],dense[n+'Out']['payloadSHA256'],block[n+'InputPayloadSHA256']})!=1:
            raise ValueError('Parent70/73/single1 input join differs')
        if comparison['parentOutgoing'][n]['payloadSHA256']!=block[n+'OutputPayloadSHA256'] or comparison['ownIncoming'][n]!=s87['sourceFields'][n]:
            raise ValueError('Saved own/parent corner join differs')
    for n in ['mod','cosine','sine']:
        if source[n]!=s87['sourceFields'][n] or source[n]['payloadSHA256']!=s73['sourceFields'][n]['payloadSHA256']:
            raise ValueError('Shared clock/RoPE differs')
    parameters=[next(x for x in e70['loadedTransformerParameters'] if x['name']==name) for name in PARAMETERS]
    if parameters!=e87['loadedParameters'] or sum(x['bytes'] for x in parameters)!=245367296:
        raise ValueError('Original four-map identity differs')
    records={'87':['request.json','inputs.json','candidate.json','execution.json','candidate-seal.json','producer-exit.json','report.json'],
             '86':['request.json','inputs.json','candidate.json','execution.json','candidate-seal.json','producer-exit.json','report.json'],
             '70':['request.json','inputs.json','candidate.json','execution.json','candidate-seal.json','report.json'],
             '73':['inputs.json','replay-dense.json','execution-dense.json','seal-dense.json']}
    metadata={str(Path(jobs[tag]['result']['diagnostics']['cwd'])/'response-result'/name):sha(directory/name)
              for tag,directory in directories.items() for name in records[tag]}
    values=[*source.values(),*(v for port in comparison.values() for v in port.values()),
            *[{'path':name,'sha256':digest} for name,digest in metadata.items()]]
    spec={'allowedRoots':[x['result']['diagnostics']['cwd'] for x in jobs.values()],'items':values,
          'checkpoint':str(Path(s70['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'),'parameters':parameters}
    proc=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast','python3 -c '+shlex.quote(BYTES)],
                        input=json.dumps(spec),text=True,capture_output=True,timeout=240)
    if proc.returncode:raise RuntimeError(proc.stderr)
    value=json.loads(proc.stdout)
    value.update(schema='tce-matched-parent-intake-v1',sourceFields=source,comparisonFields=comparison,
                 originalParameters=parameters,modelRoot=s70['modelRoot'],
                 parents={tag:{'jobID':j['job_id'],'custody':j['payload_custody'],'root':j['result']['diagnostics']['cwd']} for tag,j in jobs.items()},
                 metadataFiles=metadata,sourceConfig={n:s87[n] for n in ['nativeSources','versions','imageAssets','numericalProfile']},
                 verifierSHA256=sha(__file__),byteTemplateSHA256=sha(Path(__file__).with_name('own_successor_intake.py')),
                 scope='Fresh successful70/73/86/87 identity/custody,complete eleven NPY and metadata bytes,four original parameters;SSH bytes/headers only,not numerical tensor replay.')
    save(Path(destination),value)
    return {'files':len(value['files']),'NPY':len(value['npyPayloadWitnesses']),'fileBytes':value['totalBytesVerified'],'parameterBytes':sum(x['bytes'] for x in parameters)}


if __name__=='__main__':
    import sys
    print(json.dumps(execute(sys.argv[1])))
