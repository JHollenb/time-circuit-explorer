"""Vendor the byte-pinned public equations and parameter-only recipes, never activations."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main(source_run, target):
    data=source_run/'extracted/flux-initial-noise-result'
    request=json.loads((source_run/'request.json').read_text())
    names=['flux_reentry_reference.py','flux_transition_reference.py','flux_step_boundary_reference.py','flux_mlp_replacement.py']
    (target/'kernels').mkdir(parents=True,exist_ok=True)
    files={}
    for name in names:
        src=source_run/'payload'/name
        if digest(src)!=request['source_hashes'][name]:raise ValueError('Qualified source changed: '+name)
        dest=target/'kernels'/name;dest.write_bytes(src.read_bytes());files['kernels/'+name]=digest(dest)
    b=json.loads((data/'binding.json').read_text())
    keys={
        'project':['parameters','axes_dims_rope','rope_theta','timestep_channels'],
        'readout':['parameters','eps'],
        **{'joint'+str(i):['parameters','heads','layernorm_eps','qk_eps'] for i in range(5)}}
    result={key:{k:b[key][k] for k in fields} for key,fields in keys.items()}
    result['final_blocks']={key:{k:value[k] for k in ['parameters','heads','layernorm_eps','qk_eps']} for key,value in b['final_blocks'].items()}
    result['text_rows']=b['text_rows']
    for spec in [result['project'],result['readout'],*[result['joint'+str(i)] for i in range(5)],*result['final_blocks'].values()]:
        spec['parameters']={k:None if v is None else v['parameter_tensor'] for k,v in spec['parameters'].items()}
    def save(name,value):
        path=target/name;path.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n');files[name]=digest(path)
    save('bindings.json',result)
    recipes=json.loads((data/'parameter-artifacts.json').read_text())
    save('parameter-artifacts.json',recipes)
    # Keep the original standalone recipe validator/retriever alongside its manifest.
    (target/'retrieve_weights.py').write_bytes((data/'retrieve_weights.py').read_bytes())
    files['retrieve_weights.py']=digest(target/'retrieve_weights.py')
    source=json.loads((data/'source.json').read_text())
    save('checkpoint-files.json',{k:v for k,v in source['checkpoint_files'].items() if not k.startswith('.') and k!='room-studio-identity.json'})
    def plain(n):
        if n['kind']=='scalar':return n['value']
        if n['kind']=='array':return n
        if n['kind']=='mapping':return{k:plain(v) for k,v in n['items'].items()}
        return[plain(v) for v in n['items']]
    program=plain(json.loads((data/'transaction/full/forecasts/passes.json').read_text())['predictions'])
    origin=b['origins']['full']['inputs']
    baseline={'prompt':source['context']['prompt'],'seed':113,'initialBF16':origin['latent']['descriptor']['sha256'],
              'conditioningBF16':origin['conditioning']['descriptor']['sha256'],
              'terminalFP32':program['passes'][-1]['commit']['latent']['content_sha256'],
              'pngSHA256':digest(data/'full.predicted.png')}
    (target/'kernel-manifest.json').write_text(json.dumps({'schema':1,'upstream':'https://github.com/JHollenb/saturn-pub',
        'sourceCampaign':source['job_id'],'sourceRequest':request['request_sha256'],'files':files,'baseline':baseline,
        'scope':'Byte-identical qualified NumPy equations; parameter-only bindings and exact retrieval recipes. No observed activations or prior forecast arrays are included.'},indent=2)+'\n')
    print(json.dumps({'files':len(files),'parameterRecipes':len(recipes['artifacts']),'target':str(target)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    args=parser.parse_args();main(args.source_run.resolve(),Path(__file__).resolve().parent)
