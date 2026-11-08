"""Fresh byte-only intake for the own-state single1 successor boundary."""
import json
from pathlib import Path
import shlex
import subprocess
from encoder_atoms import read, save, sha
from publish_sparse_text_kv import NPY_BYTES

PREFIX = 'single_transformer_blocks.1.'
PARAMETERS = [PREFIX+'attn.'+name for name in
              ['to_qkv_mlp_proj.weight', 'to_out.weight', 'norm_q.weight', 'norm_k.weight']]

BYTES = NPY_BYTES.split('print(json.dumps')[0] + r'''
p=Path(s['checkpoint']);parameters=[]
with p.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];head=json.loads(f.read(n));base=8+n
 for item in s['parameters']:
  entry=head[item['name']];a,b=entry['data_offsets']
  if entry['dtype']!='BF16' or entry['shape']!=item['shape'] or [a,b]!=item['dataOffsets'] or b-a!=item['bytes']:raise ValueError('Original parameter header differs')
  f.seek(base+a);remaining=b-a;h=hashlib.sha256()
  while remaining:
   piece=f.read(min(1<<20,remaining))
   if not piece:raise ValueError('Truncated original parameter')
   h.update(piece);remaining-=len(piece)
  if h.hexdigest()!=item['payloadSHA256']:raise ValueError('Original parameter bytes differ')
  parameters.append({'name':item['name'],'bytes':b-a,'payloadSHA256':h.hexdigest()})
print(json.dumps({'files':inventory,'totalBytesVerified':sum(x['bytes'] for x in inventory.values()),'npyPayloadWitnesses':witnesses,'originalParameters':parameters}))
'''


def execute(destination):
    from mrun.client.api import Api
    from publish_source_native import checked
    root = Path(__file__).resolve().parent.parent
    previous, collection, job, request = checked(root/'artifacts/receiver-weighted-row-output-v1')
    p86 = previous/'result'
    p70 = root/'artifacts/image-partition-moments-v1/result'
    p73 = root/'artifacts/matched-single-write-v1/result'
    j70 = read(root/'docs/results/image-partition-moments/scheduler.json')
    j73 = read(root/'docs/results/matched-single-write/scheduler.json')
    api = Api()
    for stored in [job, j70, j73]:
        actual = api.json('GET', '/api/jobs/'+stored['job_id'])
        if actual['state'] != 'succeeded' or actual['payload_custody'] != stored['payload_custody']:
            raise ValueError('Successful original custody differs')
    c86, c70 = read(p86/'candidate.json')['fields'], read(p70/'candidate.json')['fields']
    e70, s70, s73 = read(p70/'execution.json'), read(p70/'inputs.json'), read(p73/'inputs.json')
    dense = read(p73/'replay-dense.json')['fields']
    block = next(x for x in e70['blockRecords'] if (x['kind'], x['index'], x['step']) == ('single', 1, 0))
    source = {n: c86[n+'Out'] for n in ['text', 'image']}
    source.update({n: c70[field] for n, field in [('mod','singleMod0'), ('cosine','cosine0'), ('sine','sine0')]})
    comparison = {port: {n: c70['b_single'+str(index)+'_'+n+'Out0'] for n in ['text','image']}
                  for port, index in [('incoming',0), ('outgoing',1)]}
    for n in ['text', 'image']:
        if len({comparison['incoming'][n]['payloadSHA256'], dense[n+'Out']['payloadSHA256'], block[n+'InputPayloadSHA256']}) != 1:
            raise ValueError('70/73 incoming boundary differs')
        if comparison['outgoing'][n]['payloadSHA256'] != block[n+'OutputPayloadSHA256']:
            raise ValueError('70 successor record differs')
    for n in ['mod','cosine','sine']:
        if source[n]['payloadSHA256'] != s73['sourceFields'][n]['payloadSHA256']:
            raise ValueError('Clock/position source differs')
    parameters = [next(x for x in e70['loadedTransformerParameters'] if x['name'] == name) for name in PARAMETERS]
    if sum(x['bytes'] for x in parameters) != 245367296:
        raise ValueError('Four-parameter census differs')
    jobs = {'86':job, '70':j70, '73':j73}
    records = {'86':['request.json','inputs.json','candidate.json','execution.json','candidate-seal.json','producer-exit.json','report.json'],
               '70':['request.json','inputs.json','candidate.json','execution.json','candidate-seal.json','report.json'],
               '73':['inputs.json','replay-dense.json','execution-dense.json','seal-dense.json']}
    metadata = {str(Path(jobs[tag]['result']['diagnostics']['cwd'])/'response-result'/name): sha(directory/name)
                for tag, directory in [('86',p86),('70',p70),('73',p73)] for name in records[tag]}
    values = [*source.values(), *(x for port in comparison.values() for x in port.values()),
              *[{'path':n,'sha256':h} for n,h in metadata.items()]]
    spec = {'allowedRoots':[j['result']['diagnostics']['cwd'] for j in jobs.values()], 'items':values,
            'checkpoint':str(Path(s70['modelRoot'])/'transformer/diffusion_pytorch_model.safetensors'), 'parameters':parameters}
    result = subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','beast',
                             'python3 -c '+shlex.quote(BYTES)], input=json.dumps(spec), text=True, capture_output=True, timeout=240)
    if result.returncode:
        raise RuntimeError(result.stderr)
    value = json.loads(result.stdout)
    value.update(schema='tce-own-successor-intake-v1', sourceFields=source, comparisonFields=comparison,
                 originalParameters=parameters, modelRoot=s70['modelRoot'],
                 parents={tag:{'jobID':j['job_id'],'custody':j['payload_custody'],'root':j['result']['diagnostics']['cwd']} for tag,j in jobs.items()},
                 metadataFiles=metadata, sourceConfig={n:s70[n] for n in ['nativeSources','versions','imageAssets','numericalProfile']},
                 verifierSHA256=sha(__file__), byteTemplateSHA256=sha(Path(__file__).with_name('publish_sparse_text_kv.py')),
                 scope='Fresh complete original source/parent/successor NPY and metadata bytes, four original tensor bytes and authoritative successful custody. SSH bytes/headers only, no model or scientific-array numerical replay.')
    save(Path(destination), value)
    return {'files':len(value['files']), 'NPY':len(value['npyPayloadWitnesses']),
            'fileBytes':value['totalBytesVerified'], 'parameterBytes':sum(x['bytes'] for x in parameters)}


if __name__ == '__main__':
    import sys
    print(json.dumps(execute(sys.argv[1])))
