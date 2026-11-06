"""Compiled, context-bound RGB response. No neural networks or target images."""
import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import resource
import sys
import time

DENIED={'torch','diffusers','transformers','tensorflow','jax','onnxruntime'}
class NoNetworks(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in DENIED:raise ImportError('Neural runtime prohibited: '+fullname)
if any(name.split('.')[0] in DENIED for name in sys.modules):raise RuntimeError('Neural runtime already imported')
sys.meta_path.insert(0,NoNetworks())
import numpy as np
from PIL import Image, __version__ as pillow_version

METHODS=['piecewise','quadratic','copy','nearest','endpoint','transfer']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
def weights(alpha,method):
    if not np.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('Dose outside compiled domain [0,1]')
    if method=='copy':return [1,0,0]
    if method=='nearest':return [int(i==min(range(3),key=lambda j:abs(alpha-[0,.25,1][j]))) for i in range(3)]
    if method=='endpoint':return [1-alpha,0,alpha]
    if method=='quadratic':return [(alpha-.25)*(alpha-1)/.25,alpha*(alpha-1)/(-.1875),alpha*(alpha-.25)/.75]
    if method=='piecewise':return [1-4*alpha,4*alpha,0] if alpha<=.25 else [0,(1-alpha)/.75,(alpha-.25)/.75]
    raise ValueError('Unknown response law')
def evaluate(anchors,alpha,method):
    return sum(np.float32(w)*anchors[i] for i,w in enumerate(weights(alpha,method)))
def encode(value):return np.rint(np.clip(value,0,255)).astype(np.uint8)
def guard():
    opened=[]
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=Path(args[0]).resolve();opened.append(str(path))
            if 'evaluation' in path.parts or path.suffix in {'.safetensors','.pt','.pth','.bin'}:raise PermissionError('Forbidden input: '+str(path))
        if event in {'socket.connect','socket.bind'}:raise PermissionError('Network access prohibited')
    sys.addaudithook(audit);return opened
def compile_chart(request,bundle):
    opened=guard();started=time.perf_counter();spec=json.loads(Path(request).read_text());arrays={}
    for name,items in spec['calibration'].items():
        values=[]
        for item in items:
            p=Path(item['path'])
            if sha(p)!=item['sha256']:raise ValueError('Calibration digest differs')
            image=np.asarray(Image.open(p).convert('RGB'))
            if image.shape!=(512,512,3):raise ValueError('Calibration image geometry differs')
            values.append(image)
        arrays[name]=np.stack(values)
    np.savez_compressed(bundle,**arrays)
    return {'seconds':time.perf_counter()-started,'bundleSHA256':sha(bundle),'bundleBytes':Path(bundle).stat().st_size,'opened':sorted(set(opened))}
def serve(bundle,spec_path,output):
    opened=guard();started=time.perf_counter();spec=json.loads(Path(spec_path).read_text());out=Path(output);out.mkdir(exist_ok=False)
    with np.load(bundle,allow_pickle=False) as archive:arrays={name:archive[name].astype(np.float32) for name in archive.files}
    loaded=time.perf_counter();rows=[]
    for name,anchors in arrays.items():
        partner=name.replace('60666','OTHER').replace('60667','60666').replace('OTHER','60667')
        for alpha in spec['alphas']:
            for method in METHODS:
                begin,cpu=time.perf_counter(),time.process_time()
                value=evaluate(anchors,alpha,method) if method!='transfer' else anchors[0]+evaluate(arrays[partner],alpha,'piecewise')-arrays[partner][0]
                clipped=float(np.mean((value<0)|(value>255)));pixels=encode(value)
                compute=time.perf_counter()-begin;cpu=time.process_time()-cpu
                path=out/f'{name}.a{alpha:g}.{method}.png';Image.fromarray(pixels).save(path)
                rows.append({'context':name,'alpha':alpha,'method':method,'path':path.name,'sha256':sha(path),'computeSeconds':compute,'cpuSeconds':cpu,'computeAndSaveSeconds':time.perf_counter()-begin,'clippedFraction':clipped})
    if any(name.split('.')[0] in DENIED for name in sys.modules):raise RuntimeError('Neural runtime entered serving')
    return {'schema':'tce-zero-network-serving-v1','bundleSHA256':sha(bundle),'loadSeconds':loaded-started,'wallSeconds':time.perf_counter()-started,'peakRssBytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'runtime':{'python':sys.version,'numpy':np.__version__,'pillow':pillow_version},'neuralRuntimeImports':[],'neuralForwardCalls':0,'checkpointWeightsRead':0,'forwardAccounting':'Source-audited arithmetic and import/file audit; not a hardware counter.','rows':rows,'opened':sorted(set(opened)),'scope':'Context and dose to compiled RGB arithmetic. Native history supplied offline; no general latent decoder or prompt encoder.'}
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['compile','serve']);p.add_argument('--request');p.add_argument('--bundle',required=True);p.add_argument('--output');p.add_argument('--receipt',required=True);args=p.parse_args()
    result=compile_chart(args.request,args.bundle) if args.mode=='compile' else serve(args.bundle,args.request,args.output)
    write(args.receipt,result)
if __name__=='__main__':main()
