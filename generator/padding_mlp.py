"""Source weights and a full-input smooth SwiGLU Jacobian; no eager model imports."""
import hashlib
import json
from pathlib import Path
import struct
from encoder_atoms import np,read,sha


def smooth_numpy(x,G,U,D):
    g=G@x;u=U@x;s=np.exp(-np.logaddexp(0.,-g));return D@(g*s*u)


def jacobian_numpy(x,G,U,D):
    g=G@x;u=U@x;s=np.exp(-np.logaddexp(0.,-g));return (D*(u*s*(1+g*(1-s)))[None,:])@G+(D*(g*s)[None,:])@U


def smooth_jacobian_torch(x,G,U,D):
    import torch
    g=G@x;u=U@x;s=torch.sigmoid(g);phi=g*s
    return D@(phi*u),(D*(u*s*(1+g*(1-s)))[None,:])@G+(D*phi[None,:])@U,g,u


def shared_active(cache,key,receiver,compute,payload):
    h=payload(receiver)
    if key in cache:
        old,value=cache[key]
        if old!=h:raise ValueError('Policies have different actual active receiver')
        return value,True
    value=compute();cache[key]=(h,value);return value,False


class SourceWeights:
    def __init__(self,source):
        self.source=source;self.model=Path(source['lexicalSource']['modelRoot']);self.files={};self.reads=[];self.checks=[]
        for item in source['qwenShards']:
            p=Path(item['path']);before=self.signature(p)
            if before[2]!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Original encoder shard differs')
            with p.open('rb') as f:
                size=struct.unpack('<Q',f.read(8))[0]
                if not 0<size<16<<20:raise ValueError('Invalid source header')
                data=f.read(size)
            if hashlib.sha256(data).hexdigest()!=item['header_sha256']:raise ValueError('Source header differs')
            self.files[str(p)]={'header':json.loads(data),'offset':size+8,'signature':before};self.checks.append({**item,'verifiedSignature':before})
        self.index=read(self.model/'text_encoder/model.safetensors.index.json')['weight_map']

    @staticmethod
    def signature(p):
        s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

    def parameter(self,name,row=None):
        from conditioning_demand import parameter_from_words
        p=self.model/'text_encoder'/self.index[name];f=self.files[str(p)];entry=f['header'][name]
        if self.signature(p)!=f['signature'] or entry['dtype']!='BF16':raise ValueError('Parameter source differs')
        lo,hi=entry['data_offsets'];shape=entry['shape']
        if hi-lo!=int(np.prod(shape))*2:raise ValueError('Parameter geometry differs')
        if row is not None:
            if len(shape)!=2 or not 0<=row<shape[0]:raise ValueError('Invalid embedding row')
            lo+=row*shape[1]*2;hi=lo+shape[1]*2;shape=[1,shape[1]]
        with p.open('rb') as stream:stream.seek(f['offset']+lo);data=stream.read(hi-lo)
        if len(data)!=hi-lo:raise ValueError('Truncated source')
        self.reads.append({'name':name,'row':row,'bytes':len(data),'payloadSHA256':hashlib.sha256(data).hexdigest(),'file':str(p)})
        return parameter_from_words(np.frombuffer(data,dtype='<u2'),shape,'cuda')

    def unchanged(self):
        for path,x in self.files.items():
            if self.signature(Path(path))!=x['signature']:raise ValueError('Source mutated')
