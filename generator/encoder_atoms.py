"""Raw text -> compiled Qwen3 layer-zero operands. No neural runtime at serving.

The table memoizes a learned, token-local function. It is not a contextual encoder.
BF16 values travel as uint16 words; position arithmetic preserves eager rounding.
"""
import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def words(value):
    value=np.asarray(value,dtype=np.float32)
    if not np.isfinite(value).all():raise ValueError('Nonfinite prefix operand')
    bits=value.view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype('<u2')
def floats(value):return (np.asarray(value,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
def bf(value):return floats(words(value))


class NoNeural(importlib.abc.MetaPathFinder):
    roots={'torch','transformers','diffusers','tensorflow','jax','onnxruntime'}
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in self.roots:raise ImportError('Neural runtime prohibited during atom serving')


def guard():
    if NoNeural.roots.intersection(sys.modules):raise RuntimeError('Serving must start without a neural runtime')
    sys.meta_path.insert(0,NoNeural())
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=str(args[0])
            if '/mnt/big/llm-models/' in path or path.endswith(('.safetensors','.pt','.pth')):
                raise PermissionError('Original model access prohibited during atom serving')
        if event in {'socket.connect','socket.bind'}:raise PermissionError('Network prohibited during atom serving')
    sys.addaudithook(audit)


def checked(root,item):
    path=Path(root)/item['path']
    if path.is_symlink() or sha(path)!=item['sha256']:raise ValueError('Compiled asset changed: '+str(path))
    return path


class AtomStore:
    def __init__(self,root,expected):
        self.root=Path(root);path=self.root/'manifest.json'
        if sha(path)!=expected:raise ValueError('Atom manifest identity differs')
        self.manifest=read(path);m=self.manifest
        self.ids=np.load(checked(self.root,m['coverage']),allow_pickle=False)
        self.hashes=np.load(checked(self.root,m['rowHashes']),allow_pickle=False)
        if self.ids.dtype!=np.dtype('<u4') or self.hashes.shape!=(len(self.ids),32) or self.hashes.dtype!=np.uint8:
            raise ValueError('Invalid compiled index')
        if len(self.ids)==0 or np.any(self.ids[1:]<=self.ids[:-1]) or self.ids[-1]>=m['vocabSize']:
            raise ValueError('Invalid token domain')
        self.index=np.full(m['vocabSize'],-1,dtype=np.int32);self.index[self.ids]=np.arange(len(self.ids))
        self.shards={};self.cache={};self.rowBytesRead=0;self.rowsRead=0

    def gather(self,ids):
        ids=np.asarray(ids)
        if np.any(ids<0) or np.any(ids>=len(self.index)):raise ValueError('Token ID outside source vocabulary')
        loc=self.index[ids];missing=np.unique(ids[loc<0])
        if len(missing):raise ValueError('Uncovered token IDs: '+','.join(map(str,missing)))
        result=[];m=self.manifest
        for token,row in zip(ids,loc):
            token=int(token);row=int(row)
            if token not in self.cache:
                shard,offset=divmod(row,m['chunkRows'])
                if shard not in self.shards:
                    item=m['shards'][shard];path=self.root/item['path']
                    if path.is_symlink() or path.stat().st_size!=item['bytes']:raise ValueError('Unsafe atom shard')
                    array=np.load(path,mmap_mode='r',allow_pickle=False)
                    if array.dtype!=np.dtype('<u2') or array.shape!=tuple(item['shape']):raise ValueError('Invalid atom shard')
                    self.shards[shard]=array
                value=self.shards[shard][offset].copy()
                if hashlib.sha256(value.tobytes()).digest()!=self.hashes[row].tobytes():raise ValueError('Atom row checksum differs')
                self.cache[token]=value;self.rowBytesRead+=value.nbytes;self.rowsRead+=1
            result.append(self.cache[token])
        return np.stack(result)


class Prefix:
    def __init__(self,root,manifestSHA256):
        from tokenizers import Tokenizer
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        self.store=AtomStore(root,manifestSHA256);m=self.store.manifest;self.m=m
        self.tokenizer=Tokenizer.from_file(str(checked(root,m['tokenizer'])))
        p=m['tokenPolicy']
        self.tokenizer.enable_padding(length=m['sequenceLength'],direction=p['paddingSide'],pad_id=p['padID'],pad_token=p['padToken'])
        self.tokenizer.enable_truncation(max_length=m['sequenceLength'],direction=p['truncationSide'],strategy='longest_first')
        env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=['jinja2.ext.loopcontrols'])
        env.filters['tojson']=lambda value,**kwargs:json.dumps(value,ensure_ascii=False,**kwargs)
        def refuse(message):raise ValueError(message)
        env.globals['raise_exception']=refuse
        self.template=env.from_string(checked(root,m['template']).read_text())
        self.rope=np.load(checked(root,m['rope']),allow_pickle=False)
        if self.rope.shape!=(2,m['sequenceLength'],m['headDim']) or self.rope.dtype!=np.dtype('<u2'):
            raise ValueError('Invalid positional table')

    def tokenize(self,prompt,generation=True):
        if not isinstance(prompt,str):raise TypeError('Raw text required')
        text=self.template.render(messages=[{'role':'user','content':prompt}],tools=None,documents=None,
                                  add_generation_prompt=generation,enable_thinking=False,
                                  bos_token=None,eos_token=self.m['tokenPolicy']['eosToken'],pad_token=self.m['tokenPolicy']['padToken'])
        encoded=self.tokenizer.encode(text,add_special_tokens=True)
        return text,np.asarray(encoded.ids,dtype=np.int64),np.asarray(encoded.attention_mask,dtype=np.int64)

    def forward(self,prompt,control='primary'):
        text,ids,mask=self.tokenize(prompt,generation=control!='wrong_template')
        if control=='wrong_token':
            ids=ids.copy();position=min(4,int(mask.sum())-1)
            choices=self.store.ids;where=int(np.searchsorted(choices,ids[position]))
            ids[position]=choices[(where+1)%len(choices)]
        atoms=self.store.gather(ids);m=self.m;h=m['hiddenSize'];qwidth=m['qWidth'];kwidth=m['kvWidth']
        e,q,k,v=np.split(atoms,[h,h+qwidth,h+qwidth+kwidth],axis=-1)
        positions=np.arange(m['sequenceLength'],dtype=np.int64)
        if control=='wrong_position':positions=np.roll(positions,1)
        cos,sin=[floats(x[positions])[:,None,:] for x in self.rope]
        def rotate(value):
            value=floats(value).reshape(len(ids),-1,m['headDim']);half=m['headDim']//2
            turn=np.concatenate([-value[...,half:],value[...,:half]],axis=-1)
            return words(bf(value*cos)+bf(turn*sin)).reshape(len(ids),-1)
        return {'ids':ids,'mask':mask,'positions':positions,'e':e,'q':q,'k':k,'v':v,'qpos':rotate(q),'kpos':rotate(k)},text


def serve(spec_path,out):
    guard();spec=read(spec_path);out=Path(out);out.mkdir(exist_ok=False)
    started=time.perf_counter();cpu=time.process_time();init=time.perf_counter()
    prefix=Prefix(spec['store'],spec['manifestSHA256']);init_s=time.perf_counter()-init
    rows=[]
    for item in spec['prompts']:
        for control in ['primary','wrong_token','wrong_position','wrong_template']:
            first=time.perf_counter();values,text=prefix.forward(item['prompt'],control);first_s=time.perf_counter()-first
            path=out/(item['id']+'-'+control+'.npz');np.savez(path,**values)
            samples=[]
            if control=='primary':
                for _ in range(spec['repeats']):
                    t=time.perf_counter();prefix.forward(item['prompt']);samples.append(time.perf_counter()-t)
            rows.append({'id':item['id'],'control':control,'path':str(path),'sha256':sha(path),
                         'renderedTextSHA256':hashlib.sha256(text.encode()).hexdigest(),
                         'firstArithmeticSeconds':first_s,'warmSeconds':samples,'warmMedianSeconds':float(np.median(samples)) if samples else None,
                         'activeTokens':int(values['mask'].sum()),'uniqueTokenIDs':int(len(np.unique(values['ids']))),
                         'logicalAtomBytesPerEvaluation':prefix.m['sequenceLength']*prefix.m['rowWidth']*2})
    seal={'schema':'tce-raw-prefix-seal-v1','manifestSHA256':spec['manifestSHA256'],'rows':rows,
          'promptSpecSHA256':sha(spec_path),'initializationSeconds':init_s,'servingSecondsAfterImports':time.perf_counter()-started,
          'cpuSecondsAfterImports':time.process_time()-cpu,'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'uniqueCompiledRowsRead':prefix.store.rowsRead,'compiledRowPayloadBytesRead':prefix.store.rowBytesRead,
          'coverageCount':len(prefix.store.ids),'vocabSize':prefix.m['vocabSize'],'frameworkCalls':0,
          'originalDenseProjections':0,'learnedFunctionMemoization':True,'contextualAttentionEvaluations':0,
          'scope':'Token-local lexical operands and eager-BF16 RoPE only; no contextual encoding or image. Warm calls reuse checksum-verified lexical rows; first call is not a disk-cache eviction experiment.'}
    save(out/'prediction-seal.json',seal)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');p.add_argument('output');args=p.parse_args();serve(args.spec,args.output)
