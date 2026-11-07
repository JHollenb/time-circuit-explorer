"""One guarded artifact-only lease; geometry seals before reference access."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import np,read,save,sha,identity,guard
from paired_fields import load
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['extension_geometry.py','extension_geometry_study.py','source_response.py','encoder_atoms.py','paired_fields.py','flux_clock_study.py']


def stage(root):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    from mdb.job_support import one_lease_scheduling_contract
    parent=HERE.parent/'artifacts/minimum-response-v1';spectrum=HERE.parent/'artifacts/source-spectrum-v1';api=Api();c,j,old=checked_collection(parent,current_workers=False);sc,sj,sr=checked_collection(spectrum,current_workers=False)
    for coll,job in [(c,j),(sc,sj)]:
        live=api.json('GET','/api/jobs/'+job['job_id'])
        if live['state']!='succeeded' or live['payload_custody']!=coll['custody']:raise ValueError('Qualified parent differs')
    p0=parent/'result';p1=spectrum/'result';source0=read(p0/'source.json');fields=read(p1/'fields.json');served=read(p0/'serving.json')
    for run,p,key,name in [(parent,p0,'candidateSealSHA256','candidate-seal.json'),(spectrum,p1,'analysisSealSHA256','analysis-seal.json')]:
        if read(p/'report.json')[key]!=sha(p/name) or read(run/'retained-array-verification.json')[key]!=sha(p/name):raise ValueError('Parent release/bytes differ')
    if source0['factors']!={k:fields[k] for k in source0['factors']}:raise ValueError('Retained decomposition differs')
    pairs=[{k:x[k] for k in ['id','population','location','fields']} for x in served['rows'] if x['population']!='discovery']
    if len(pairs)!=6 or len({x['id'] for x in pairs})!=6:raise ValueError('Held-role population differs')
    root=Path(root).resolve();root.mkdir(exist_ok=False,parents=True);p=root/'payload';p.mkdir()
    for n in OWNED:
        for qualified in [old,sr]:
            if n in qualified['ownedSources'] and sha(HERE/n)!=qualified['sourceHashes'][n]:raise ValueError('Qualified helper changed')
        shutil.copyfile(HERE/n,p/n)
    shutil.copyfile(HERE.parent/'docs/plans/minimum-response-extension-geometry.md',p/'frozen-plan.md');base=j['result']['diagnostics']['cwd'];sb=sj['result']['diagnostics']['cwd']
    source={'pairs':pairs,'factors':{k:fields[k] for k in ['Vt','sigma','targetModes','responseFilter']},'inheritedScale':source0['scale'],
        'protectedRoots':[base,sb,*source0['protectedRoots']],'deniedCanaries':[served['rows'][-1]['prediction']['path'],read(p0/'scoring-source.json')['targets'][pairs[-1]['id']]['path']],
        'parentRecords':{label+'/'+n:{'path':remote+'/response-result/'+n,'sha256':sha(local/n)} for label,remote,local,names in [
            ('minimum',base,p0,['report.json','serving.json','candidate-seal.json','coefficient-seal.json']),('spectrum',sb,p1,['fields.json','analysis-seal.json','report.json'])] for n in names}}
    save(p/'source.json',source);save(p/'reference-source.json',{'report':source['parentRecords']['minimum/report.json'],'predictions':{x['id']:x['prediction'] for x in served['rows'] if x['population']!='discovery'}})
    contract=one_lease_scheduling_contract(model_free=True,model_loads=0,logical_operations=6,logical_branches=6,logical_checkpoint_replays=0,extra={'task_family':'analysis'})
    r={'schema':'tce-extension-geometry-request-v1','parentJobID':j['job_id'],'parentRequestSHA256':old['requestSHA256'],'ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},
        'scheduling_contract':contract,'needs':{'cpu':True},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':300,
        'protocol':'Six archived heldX; allmode Z=XV,A=Z/sigma,Xparallel=ZVt,r=X-Xparallel,Yhat=AC,D=AtYhat. Retainqueryenergies/leverage/signedmodepower/crossclosure/actualimageindices. Sealgeometry beforeoldforecast/erroraccountaccess. No SVD/fit/newpredictor/select/rank/gain/regularization sweep,T/U,model/H0/VAE/images. 68.439BleadingMAC.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r);save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same job')
    cmd=command();cmd=cmd[:cmd.index('--with')]+['--with','numpy==2.5.4','--with','psutil>=5.9'];i=cmd.index('uv');cmd[i:i]=['PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1'];cmd+=['python','-u','extension_geometry_study.py','execute','.']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-extension-geometry-v1',task='analysis',model=None,payload=root/'payload',pin='beast',needs=r['needs'],gpu=False,
        mrun_config={k:r[k] for k in ['schema','requestSHA256','backend','device','cpuThreads','scheduling_contract']},timeout_s=r['timeoutSeconds'],detach=True,retry_on_kill=False,preflight=True,
        note='CPU1 saved6heldX944x3072/V1888modes/C128outputs.68.439BleadingMAC geometry audit,retainA/Yhat/D/queryenergies/signedmodalpower. Geometryseal beforeoldforecasts/errorjoins. No newSVD/fit/predictor/sweep/model/H0/VAE/images/frameworks;no U/T. Intent-first resources,priorCPU800MBRSS/0VRAM;beastlocalarrays,expected<2min/300scap,noretries. Retrospective dependentroles;no missingconditioncausality.')
    v={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',v);return v


def execute(root):
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'source.json');guard();out=root/'response-result';out.mkdir();start=time.perf_counter();released={'value':False}
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and not released['value']:
            p=Path(os.fsdecode(args[0])).resolve()
            if (p.suffix in {'.npy','.npz'} and not p.is_relative_to(root)) or p==root/'reference-source.json' or p.name=='report.json' and not p.is_relative_to(root):raise PermissionError('Reference access before geometry release')
    sys.addaudithook(audit)
    spec={**s,'output':str(root/'geometry'),'arrays':str(root/'arrays')};save(root/'geometry-spec.json',spec);tick=time.perf_counter();subprocess.run([sys.executable,'-u',str(root/'extension_geometry.py'),str(root/'geometry-spec.json')],check=True,timeout=240);phase=time.perf_counter()-tick
    g=read(root/'geometry/geometry.json')
    for fs in g['fields'].values():
        for item in fs.values():load(item)
    save(root/'geometry-seal.json',{'schema':'tce-extension-geometry-seal-v1','requestSHA256':r['requestSHA256'],'geometryReceiptSHA256':sha(root/'geometry/geometry.json'),'fields':g['fields'],
        'queryStreamSHA256':sha(root/'geometry/queries.jsonl'),'modeStreamSHA256':sha(root/'geometry/modes.jsonl')})
    sealed=sha(root/'geometry-seal.json');print('EXTENSION_GEOMETRY_SEALED',sealed,flush=True);released['value']=True;ref=read(root/'reference-source.json')
    if sha(ref['report']['path'])!=ref['report']['sha256']:raise ValueError('Old response report differs')
    old=read(ref['report']['path']);accounts=[]
    for pair in g['pairs']:
        pred=load(g['fields'][pair['id']]['reconstructedPrediction']);prior=load(ref['predictions'][pair['id']]);e=pred-prior
        account=next(x for x in old['accounts'] if x['id']==pair['id'] and x['method']=='minimum-norm')
        accounts.append({**pair,'oldResponseAccount':account,'reconstructionDifference':{'maxAbs':float(np.max(np.abs(e))),'errorEnergy':float(np.sum(e*e)),'relativeRMS':float(np.sqrt(np.sum(e*e)/np.sum(prior*prior)))}})
    if sha(root/'geometry-seal.json')!=sealed:raise ValueError('Geometry release changed')
    for n in ['request.json','source.json','reference-source.json','frozen-plan.md','geometry-seal.json']:shutil.copyfile(root/n,out/n)
    for x in (root/'geometry').iterdir():shutil.copyfile(x,out/x.name)
    save(out/'report.json',{'schema':'tce-extension-geometry-result-v1','requestSHA256':r['requestSHA256'],'geometrySealSHA256':sealed,'accounts':accounts,'geometryActorPhaseSeconds':phase,'workerWallSeconds':time.perf_counter()-start,
        'nominalLeadingMACs':sum(x['nominalLeadingMACs'] for x in accounts),'work':{'SVDs':0,'newFits':0,'newPredictors':0,'fixedPredictionReconstructions':6,'originalModelEvaluations':0,'newH0Computations':0,'newImages':0},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limits':['Saved input/source-factor geometry only; old forecasts/erroraccounts open after seal. No new native targets or causal conditioning test.',
            'Large inverse-mode coordinates amplify in-span response; orthogonal residual is ignored by B0 and describes source-unconstrained extension freedom.',
            'Allmodes retained, no new fit/predictor/gain/rank/regularization selection. Floatingpoint closure/discrepancies retained as observations.',
            'Old learned H0/labels/factors/control arithmetic remain paid; source row leverage1 is an exact-algebra reference, not a new measured assay.',
            'Retrospective paired roles/queries are dependent; seed and location remain confounded. No arbitrary-prompt zero-neural generation claim.']})
    print('EXTENSION_GEOMETRY_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(json.dumps(v))
