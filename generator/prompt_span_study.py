"""Target-assisted capacity of the supplied two-delta plane; not serving."""
import argparse
import json
from pathlib import Path
import shutil
import time

from response_study import read,sha,save,identity,command,collect

def project_span(d1,d2,m):
    from prompt_closure import np,rms,ratio
    ds=[d1,d2];g=np.array([[np.mean(x*y) for y in ds] for x in ds],np.float64)
    b=np.array([np.mean(x*m) for x in ds],np.float64);values,vectors=np.linalg.eigh(g)
    cutoff=2*np.finfo(np.float64).eps*max(float(values[-1]),0);keep=values>cutoff
    inverse=np.zeros_like(values);inverse[keep]=1/values[keep]
    gamma=vectors@(inverse*(vectors.T@b));residual=m-gamma[0]*d1-gamma[1]*d2
    return residual,{'gram':g.tolist(),'gramEigenvalues':values.tolist(),'rank':int(keep.sum()),'cutoff':cutoff,
        'gramCondition':float(values[-1]/values[0]) if keep.all() else None,'gamma':gamma.tolist(),
        'adjustedCoefficients':(gamma+1).tolist(),'mixedRms':rms(m),'unexplainedRms':rms(residual),
        'unexplainedOverMixed':ratio(rms(residual),rms(m)),'residualBasisDots':[float(np.mean(residual*x)) for x in ds]}

def stage(root,parent):
    root=Path(root).resolve();parent=Path(parent).resolve();original=read(parent/'request.json');root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    for name in ['calibration.json','evaluation.json','prompt_closure.py','response_chart.py','response_study.py']:
        source=parent/'payload'/name
        if sha(source)!=original['sourceHashes'][name]:raise ValueError('Parent staged source changed')
        shutil.copyfile(source,payload/name)
    shutil.copyfile(Path(__file__),payload/Path(__file__).name)
    request={'schema':'tce-prompt-span-request-v1','parentRequestSHA256':original['requestSHA256'],'parentReportSHA256':sha(parent/'result/report.json'),
        'sourceJobID':original['sourceJobID'],'scope':'Target-assisted retrospective two-delta linear-plane capacity at independent native boundaries. No predictor, neural evaluation, or own-state rollout.',
        'resourceProfile':{'ramMB':2048,'source':'job-3ff4ee580893 measured 495.4MB peak on identical 34-boundary geometry. This job adds 2x2 Gram reductions, retains no candidate history; conservative 4x measured headroom. Default4096MB queued run cancelled before execution, not retried.'},
        'sourceHashes':{p.name:sha(p) for p in payload.iterdir()}}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(payload/'request.json',request)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}
def submit(root):
    from mrun.client.submit import launch
    root=Path(root).resolve();request=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    cmd=[x.replace('response_study.py','prompt_span_study.py') for x in command()]
    job=launch(cmd,experiment='tce-prompt-span-capacity-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},ram_mb=request['resourceProfile']['ramMB'],
        config={'schema':'tce-prompt-span-v1','requestSHA256':request['requestSHA256'],'backend':'numpy','device':'cpu','plannedNeuralForwards':0},
        timeout_s=180,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='Target-assisted34boundary2Dspan capacity. Saved arrays;2x2Gram, no model/weights. CPU1/2GiB from measured495MB samegeometry +4x headroom. NoGPU, finite180s, no retry. Analysis only, not serving/rollout.')
    result={'jobID':str(job),'requestSHA256':request['requestSHA256']};save(root/'job.json',result);return result
def execute(root):
    from prompt_closure import load,np,rms,ratio
    root=Path(root).resolve();started=time.perf_counter();request=read(root/'request.json');out=root/'response-result';out.mkdir(exist_ok=False)
    if identity({k:v for k,v in request.items() if k!='requestSHA256'})!=request['requestSHA256']:raise ValueError('Request changed')
    for name,digest in request['sourceHashes'].items():
        if sha(root/name)!=digest:raise ValueError('Source changed')
    spec=read(root/'calibration.json');targets=read(root/'evaluation.json');rows=[]
    for row,target in zip(spec['rows'],targets['rows'],strict=True):
        if row['id']!=target['id']:raise ValueError('Address differs')
        b,c,o=[load(ref) for ref in row['calibration']];t=load(target['target']);m=t-o-c+b
        residual,metrics=project_span(c-b,o-b,m)
        rows.append({**{k:row[k] for k in ['id','space','field','step','local']},**metrics,
            'nativeColorEffectRms':rms(t-o),'unexplainedOverNativeColorEffect':ratio(rms(residual),rms(t-o))})
        print('SPAN '+row['id'],flush=True)
    save(out/'report.json',{'schema':'tce-prompt-span-capacity-report-v1','requestSHA256':request['requestSHA256'],'rows':rows,
        'workerWallSeconds':time.perf_counter()-started,'targetAssisted':True,'newNeuralEvaluations':0,'mechanics_status':'complete','terminal_status':'unassessed','scope':request['scope']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run',type=Path);p.add_argument('--parent',type=Path);a=p.parse_args()
    result=stage(a.run,a.parent) if a.action=='stage' else globals()[a.action](a.run)
    if result is not None:print(json.dumps(result))
