"""Stage the source-bound saved-array diagnostic and execute it through mrun."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from response_study import sha,read,save,identity,command,collect

HERE=Path(__file__).resolve().parent
CELLS=['c0-o0','c1-o0','c0-o1','c1-o1']
SCOPE='Retrospective same-seed four-prompt boundary diagnostic. Every boundary uses three native calibration histories independently; not an own-state rollout, raw-prompt compiler or arbitrary image generator. Supplied token-grid transport; no global equivariance claim.'
def stage(root,source):
    root=Path(root).resolve();run=Path(source).resolve();data=run/'extracted/flux-prompt-factorial-result';collection=read(run/'collection.json')
    inv=read(run/'collection-local-inventory.json')
    if sha(run/'collection-local-inventory.json')!=collection['bindings']['collection-local-inventory.json']:raise ValueError('Collection inventory identity differs')
    for name in ['source.json','summary.json','binding.json']:
        if sha(data/name)!=inv['files'][name]['sha256']:raise ValueError('Source metadata changed: '+name)
    original=read(data/'source.json');binding=read(data/'binding.json');summary=read(data/'summary.json');remote=collection['remote_path']
    def slim(ref,kind='array'):
        path=ref['path'];digest=ref['sha256']
        if inv['files'][path]['sha256']!=digest:raise ValueError('Array reference differs from collected inventory')
        return {'path':remote+'/'+path,'sha256':digest,'encoding':ref.get('encoding'),'shape':ref.get('stored_shape',[512,512,3]),'kind':kind}
    def checked_jsonl(name):
        if sha(data/name)!=inv['files'][name]['sha256']:raise ValueError('Native trace changed')
        return [json.loads(line) for line in (data/name).read_text().splitlines()]
    native={c:checked_jsonl('native/'+c+'.jsonl') for c in CELLS};cal=[];evaluation=[]
    def add(name,space,field,refs,step=None,local=None):
        cal.append({'id':name,'space':space,'field':field,'step':step,'local':local,'calibration':[slim(refs[c],space if space=='rgb' else 'array') for c in CELLS[:3]]})
        evaluation.append({'id':name,'target':slim(refs[CELLS[3]],space if space=='rgb' else 'array')})
    add('conditioning','text','conditioning',{c:original['encodings'][c]['conditioning'] for c in CELLS})
    for i in range(112):
        if i%28 not in [0,5,25,26,27]:continue
        fields=['image','text'] if i%28 in [0,5,25] else ['noise'] if i%28==26 else ['latent']
        for field in fields:
            refs={c:native[c][i]['outputs'][field]['archive'] for c in CELLS}
            add(f'step{i//28}.local{i%28}.{field}','text' if field=='text' else 'tokens',field,refs,i//28,i%28)
    add('rgb','rgb','rgb',{c:{**summary['rows'][c+'.native']['image'],'stored_shape':[512,512,3]} for c in CELLS})
    ids={c:binding['origins'][c]['inputs']['img_ids'] for c in CELLS}
    if len({r['sha256'] for r in ids.values()})!=1:raise ValueError('Image coordinates differ across prompt cells')
    root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    save(payload/'calibration.json',{'sourceRoot':remote,'ids':slim(ids[CELLS[0]]),'rows':cal,'scope':SCOPE})
    save(payload/'evaluation.json',{'rows':evaluation})
    for name in ['prompt_closure.py','prompt_closure_study.py','response_chart.py','response_study.py']:shutil.copyfile(HERE/name,payload/name)
    request={'schema':'tce-prompt-closure-request-v1','sourceJobID':collection['job_id'],'sourceCollectionSHA256':sha(run/'collection.json'),
        'sourceInventorySHA256':sha(run/'collection-local-inventory.json'),'checkpoint':original['checkpoint_sha256'],'sourceRoot':remote,
        'conditions':original['conditions'],'rows':len(cal),'scope':SCOPE,'methods':['transported','additive','copy_order','copy_color','mirror_color','vertical'],
        'sourceHashes':{p.name:sha(p) for p in payload.iterdir() if p.is_file()}}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(payload/'request.json',request)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'rows':len(cal),'requestSHA256':request['requestSHA256']}
def submit(root):
    from mrun.client.submit import launch
    root=Path(root).resolve();request=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    cmd=[x.replace('response_study.py','prompt_closure_study.py') for x in command()]
    job=launch(cmd,experiment='tce-prompt-composition-closure-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':'tce-prompt-closure-v1','requestSHA256':request['requestSHA256'],'backend':'numpy','device':'cpu','plannedNeuralForwards':0},
        timeout_s=300,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='Saved-only34boundary prompt-composition assay. Host-local hash-bound native archives;3calibration/1target. Seal6laws before target reads; no neural/weight math, CPU1, noGPU. Finite300s, no retry. Not an own-state rollout.')
    result={'jobID':str(job),'requestSHA256':request['requestSHA256']};save(root/'job.json',result);return result
def execute(root):
    root=Path(root).resolve();start=time.perf_counter();request=read(root/'request.json');out=root/'response-result';out.mkdir(exist_ok=False)
    if identity({k:v for k,v in request.items() if k!='requestSHA256'})!=request['requestSHA256']:raise ValueError('Request changed')
    for name,digest in request['sourceHashes'].items():
        if sha(root/name)!=digest:raise ValueError('Source changed: '+name)
    subprocess.run([sys.executable,str(root/'prompt_closure.py'),'predict',str(root/'calibration.json'),str(out)],check=True,cwd=root)
    from prompt_closure import score
    result=score(root/'calibration.json',root/'evaluation.json',out)
    save(out/'report.json',{'schema':'tce-prompt-composition-report-v1','requestSHA256':request['requestSHA256'],'jobScope':request['scope'],
        'scores':result,'workerWallSeconds':time.perf_counter()-start,'newNeuralEvaluations':0,'denseLearnedWeightMath':False,
        'mechanics_status':'complete','terminal_status':'unassessed'})
    print('PROMPT_CLOSURE_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run',type=Path);p.add_argument('--source',type=Path);a=p.parse_args()
    result=stage(a.run,a.source) if a.action=='stage' else globals()[a.action](a.run)
    if result is not None:print(json.dumps(result))
