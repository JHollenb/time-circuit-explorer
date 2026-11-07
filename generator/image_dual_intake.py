"""Authenticated byte-only preparation of a native text-key prefix on its host."""
import json
import shlex
import subprocess

POSITION_BYTES = r'''import hashlib,json,struct,sys
from pathlib import Path
s=json.load(sys.stdin);page=s['page'];raw=(Path(page['store'])/'blobs'/(page['metadata']['tensor_blob']+'.safetensors')).read_bytes()
assert hashlib.sha256(raw).hexdigest()==page['metadata']['tensor_blob']
length=struct.unpack('<Q',raw[:8])[0];header=json.loads(raw[8:8+length]);parts=[]
declared={x['key']['value']:x['value'] for x in page['declaredDescriptor']['items']}['rope']['items']
for n,d in zip(['tensor_1','tensor_2'],declared):
 e=header[n];assert e['dtype']=='F32' and e['shape']==[1536,128]
 start,end=e['data_offsets'];payload=raw[8+length+start:8+length+end]
 assert hashlib.sha256(payload).hexdigest()==d['sha256'];parts.append(payload)
axes=[]
for a in range(4):
 rows=[b''.join(x[(j*128+a*32)*4:(j*128+(a+1)*32)*4] for x in parts) for j in range(1536)]
 im=rows[512:];text=rows[:512]
 axes.append({'axis':a,'imageDistinctByteGroups':len(set(im)),'textDistinctByteGroups':len(set(text)),
  'imageConstant':all(x==im[0] for x in im),'imageHeightMembershipExact':all(x==im[(j//32)*32] for j,x in enumerate(im)),
  'imageWidthMembershipExact':all(x==im[j%32] for j,x in enumerate(im))})
print(json.dumps({'sourceBlobSHA256':page['metadata']['tensor_blob'],'axes':axes,
 'scope':'Exact stored F32 row-byte equality only; no floating-point decoding, tensor arithmetic or neural evaluation.'}))
'''

SLICE = r'''import ast,hashlib,json,struct,sys
from pathlib import Path
s=json.load(sys.stdin);source=s['source'];p=Path(source['path']);data=p.read_bytes()
digest=lambda b:hashlib.sha256(b).hexdigest()
assert digest(data)==source['sha256'] and data[:8]==b'\x93NUMPY\x01\x00'
length=struct.unpack('<H',data[8:10])[0];offset=10+length
header=ast.literal_eval(data[10:offset].decode('latin1').strip())
assert header=={'descr':'<u2','fortran_order':False,'shape':(1,1536,24,128)}
assert source['shape']==[1,1536,24,128] and source['logicalDtype']=='torch.bfloat16'
assert digest(data[offset:])==source['payloadSHA256']
count=512*24*128*2;payload=data[offset:offset+count]
text=repr({'descr':'<u2','fortran_order':False,'shape':(1,512,24,128)})
raw=text.encode('latin1');raw+=b' '*((-10-len(raw)-1)%64)+b'\n'
result=b'\x93NUMPY\x01\x00'+struct.pack('<H',len(raw))+raw+payload
out=Path(s['destination']);out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():assert out.read_bytes()==result
else:
 with out.open('xb') as f:f.write(result)
print(json.dumps({'path':str(out),'sha256':digest(result),'payloadSHA256':digest(payload),
 'bytes':len(result),'shape':[1,512,24,128],'storedDtype':'uint16','logicalDtype':'torch.bfloat16',
 'source':source,'sourceHeaderSHA256':digest(data[:offset]),'sourceDataOffset':offset,
 'selectedByteOffset':offset,'selectedPayloadBytes':count,'sourcePayloadBytes':len(data)-offset,
 'policy':'Full-file byte hashing; decode only the NPY header. Copy contiguous text-key bytes; image keys never numerically decoded.'}))
'''


def prepare(source, destination):
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'beast', 'python3 -c ' + shlex.quote(SLICE)],
        input=json.dumps({'source': source, 'destination': destination}), text=True,
        capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


def inventory_positions(page):
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'beast', 'python3 -c ' + shlex.quote(POSITION_BYTES)],
        input=json.dumps({'page': page}), text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout)
