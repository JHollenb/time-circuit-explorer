"""Raw-byte support inventory of authenticated post-joint0 states; no tensor math."""
import json
from pathlib import Path
import shlex
import subprocess
from encoder_atoms import read, save, sha

BYTES = r'''
import hashlib,json,struct,sys
from pathlib import Path
s=json.load(sys.stdin);records=s['sources'];support=s['support'];files={};rows={}
def tensor(page):
 p=Path(page['store'])/'blobs'/(page['metadata']['tensor_blob']+'.safetensors')
 raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 if digest!=page['metadata']['tensor_blob'] or page['files'][str(p)]!=digest:raise ValueError('Source blob bytes differ')
 n=struct.unpack('<Q',raw[:8])[0];h=json.loads(raw[8:8+n]);entry=h[page['metadata']['encoded']['name']]
 d=page['declaredDescriptor'];shape=entry['shape'];start,end=entry['data_offsets'];payload=raw[8+n+start:8+n+end]
 if entry['dtype']!='BF16' or shape!=d['shape'] or len(shape)!=3 or shape[0]!=1 or d['stride']!=[shape[1]*shape[2],shape[2],1]:raise ValueError('Source row byte layout differs')
 ph=hashlib.sha256(payload).hexdigest()
 if ph!=d['sha256'] or len(payload)!=shape[1]*shape[2]*2:raise ValueError('Source raw payload differs')
 files[str(p)]={'sha256':digest,'bytes':len(raw),'payloadSHA256':ph,'payloadBytes':len(payload),'shape':shape}
 return payload,shape
for slot in ['text','image']:
 p,shape=tensor(records['P']['pages'][slot]);q,other=tensor(records['Q']['pages'][slot])
 if shape!=other:raise ValueError('Source shapes differ')
 width=shape[-1]*2;changed=[];counts=[];po=hashlib.sha256();qo=hashlib.sha256()
 for j in range(shape[1]):
  a,b=p[j*width:(j+1)*width],q[j*width:(j+1)*width]
  count=sum(a[k:k+2]!=b[k:k+2] for k in range(0,width,2))
  counts.append(count)
  if count:changed.append(j)
  if slot=='image' or j not in support:po.update(a);qo.update(b)
 outside=[j for j in changed if slot=='image' or j not in support]
 rows[slot]={'shape':shape,'changedRows':changed,'changedBF16WordsPerRow':counts,
             'changedOutsideSupportRows':outside,'outsideSupportPBytesSHA256':po.hexdigest(),'outsideSupportQBytesSHA256':qo.hexdigest(),
             'outsideSupportByteIdentical':po.hexdigest()==qo.hexdigest()}
print(json.dumps({'files':files,'rows':rows,'contextDescriptorsExact':records['P']['pages']['context']['declaredDescriptor']==records['Q']['pages']['context']['declaredDescriptor'],
                  'scope':'Full raw BF16 word/row comparison; SSH byte I/O only, no neural or numerical tensor computation.'}))
'''


def execute(root):
    from mrun.client.api import Api
    root = Path(root).resolve()
    meta = read(root / 'source-metadata.json')
    family = Path(__file__).resolve().parents[2] / 'artifacts/mdb-computation-family-20261107/fresh-family-02'
    original = read(family / 'scheduler-job.json')
    job = Api().json('GET', '/api/jobs/' + original['job_id'])
    if job['state'] != 'succeeded' or job['payload_custody'] != original['payload_custody']:
        raise ValueError('Original source scheduler custody differs')
    for n, digest in meta['familyMetadataSHA256'].items():
        if sha(family / 'compact' / n) != digest:
            raise ValueError('Original source metadata differs')
    actual = read(family / 'compact/native-source-writers.json')['sphere-cube-plain-2027']['sources']
    if meta['origins'] != {n: x['state'] for n, x in actual.items()}:
        raise ValueError('Actual post-joint0 source origins differ')
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3 -c ' + shlex.quote(BYTES)],
                            input=json.dumps({'sources': meta['sources'], 'support': [4, 11, 12, 13]}), text=True, capture_output=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr)
    receipt = {'schema': 'tce-propagated-context-intake-v1', 'sourceMetadataSHA256': sha(root / 'source-metadata.json'),
               'verifierSourceSHA256': sha(__file__), 'sourceJobID': job['job_id'], 'sourceCustody': job['payload_custody'],
               'support': [4, 11, 12, 13], **json.loads(result.stdout)}
    receipt['contextContrastApplicable'] = any(not x['outsideSupportByteIdentical'] for x in receipt['rows'].values())
    save(root / 'byte-inventory.json', receipt)
    return {'sourceJobID': job['job_id'], 'applicable': receipt['contextContrastApplicable'],
            'contextExact': receipt['contextDescriptorsExact'],
            'changedRows': {n: len(x['changedRows']) for n, x in receipt['rows'].items()},
            'changedOutsideSupportRows': {n: len(x['changedOutsideSupportRows']) for n, x in receipt['rows'].items()}}


if __name__ == '__main__':
    import sys
    print(json.dumps(execute(sys.argv[1]), indent=2))
