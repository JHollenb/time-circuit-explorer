"""Portable bounded global attention read; input is state, output is one head field."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from response_study import read,save,sha


def render(bundle,state,output):
    bundle,state,output=[Path(p).resolve() for p in [bundle,state,output]]
    files=read(bundle/'files.json');samples=read(bundle/'sample-states.json')
    needed=['sample-states.json','compile-receipt.json','reader-head.npz','fixed-head.npz','attention-cache.npz']
    baseline=samples['packets'][0];needed.append(baseline['path'])
    for name in needed:
        if sha(bundle/name)!=files[name]:raise ValueError('Published head bundle changed: '+name)
    compiled=read(bundle/'compile-receipt.json');output.mkdir(parents=True,exist_ok=False)
    spec={key:{'path':str(bundle/name),'sha256':compiled['bundles'][name]['sha256']} for key,name in [('reader','reader-head.npz'),('fixed','fixed-head.npz'),('cache','attention-cache.npz')]}
    spec.update(packets=[{'id':'baseline','path':str(bundle/baseline['path']),'sha256':baseline['sha256'],'shape':baseline['shape']},
                         {'id':'input','path':str(state),'sha256':sha(state),'shape':baseline['shape']}],
                eps=1e-6,repeats=1,scope=compiled['scope'],sourceRoots=[str(bundle),str(state.parent),str(output)])
    save(output/'serve.json',spec)
    subprocess.run([sys.executable,str(Path(__file__).with_name('global_attention.py')),'serve',str(output/'serve.json'),str(output)],check=True)
    seal=read(output/'prediction-seal.json');row=next(r for r in seal['rows'] if r['method']=='all' and r['id']=='input')
    return {'output':str(output/row['path']),'sha256':row['sha256'],'scope':compiled['scope']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',required=True,type=Path);parser.add_argument('--state',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(render(args.bundle,args.state,args.output)))
