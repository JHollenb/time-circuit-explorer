"""Saved-only bandwidth and response-capacity diagnosis; no fresh-input claim."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from encoder_atoms import np,read,save,sha,identity,guard
from context_landmarks import stream,routes
from context_statistics_study import summary,rowrms

HERE=Path(__file__).resolve().parent
SCALES=[1.,1/16,1/256]


def simplex(value):
    ordered=np.sort(value,axis=1)[:,::-1];cumulative=np.cumsum(ordered,axis=1)-1
    count=np.sum(ordered-cumulative/np.arange(1,value.shape[1]+1)>0,axis=1)-1
    theta=cumulative[np.arange(len(value)),count]/(count+1)
    return np.maximum(value-theta[:,None],0.)


def capacity(readers,target,initial,iterations=512):
    """Target-assisted signed-affine lower bound and attained convex fit, not serving."""
    difference=(readers[1:]-readers[0]).T;U,S,Vt=np.linalg.svd(difference,full_matrices=False)
    tolerance=max(difference.shape)*np.finfo(float).eps*(S[0] if len(S) else 0.)
    keep=S>tolerance;centered=target-readers[0];coeff=(centered@U[:,keep]/S[keep])@Vt[keep]
    affine_weights=np.concatenate([(1-coeff.sum(1))[:,None],coeff],axis=1);affine=affine_weights@readers
    gram=readers@readers.T;b=target@readers.T;L=float(np.linalg.eigvalsh(gram)[-1]);w=initial.copy();curve=[]
    for step in range(iterations):
        if L>0:w=simplex(w-(w@gram-b)/L)
        if step+1 in [1,16,128,iterations]:curve.append({'iteration':step+1,'fit':summary(w@readers,target)})
    predicted=w@readers;error=predicted-target;gradient=w@gram-b
    gap=np.maximum(np.sum(gradient*w,axis=1)-gradient.min(1),0.);objective=.5*np.sum(error*error,axis=1)
    return {'affine':affine,'affineWeights':affine_weights,'convex':predicted,'convexWeights':w,'fwGap':gap,
        'objective':objective,'objectiveLowerBound':np.maximum(objective-gap,0.)}, {
        'scope':'Target-assisted per-query fit of padding-pair response vectors. Signed affine relaxation lower bound; attained512-step convex fit is an upper bound, not a certified optimum or deployable route.',
        'affineRank':int(keep.sum()),'singularValues':S.tolist(),'rankRoundoffTolerance':float(tolerance),
        'affine':summary(affine,target),'affineMaximumNegativeWeightMass':float(np.max(-np.minimum(affine_weights,0.).sum(1))),
        'convex':summary(predicted,target),'iterations':iterations,'iterationCurve':curve,
        'maximumWeightSumError':float(np.max(np.abs(w.sum(1)-1))),'minimumWeight':float(w.min()),
        'meanFrankWolfeObjectiveGap':float(gap.mean()),'meanObjective':float(objective.mean()),
        'meanObjectiveLowerBound':float(np.maximum(objective-gap,0.).mean())}


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,old,prior=checked_collection(parent)
    job=Api().json('GET','/api/jobs/'+c['jobID'])
    if job['state']!='succeeded' or job['payload_custody']!=c['custody'] or job['config']['requestSHA256']!=prior['requestSHA256']:raise ValueError('Parent differs')
    seal=read(parent/'result/prediction-seal.json');ref=read(parent/'result/reference-receipt.json')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    save(p/'source.json',{'ports':seal['ports'],'predictions':seal['rows'],'references':ref['rows'],'projection':ref['projection'],
        'asset':seal['landmarkAsset'],'pairs':prior['pairs'],'parentSealSHA256':sha(parent/'result/prediction-seal.json')})
    for n in ['landmark_routing_diagnostics.py','context_landmarks.py','context_source_geometry.py','context_statistics.py','context_statistics_study.py','encoder_atoms.py']:
        shutil.copyfile(HERE/n,p/n)
    r={'schema':'tce-landmark-routing-diagnostic-request-v1','parentJobID':job['job_id'],'parentRequestSHA256':prior['requestSHA256'],
        'scope':'Retrospective development diagnosis on the8already-scored strings. Fixed bandwidth scales plus explicit target-assisted padding-response capacity; no held-out serving/generalization/image claim.',
        'bandwidthSquaredScales':SCALES,'convexFitIterations':512,
        'capacity':'For same-query padding positions only, response matrix rows are64landmark hB-hA vectors. Signed affine relaxation and per-query simplex-constrained fit read target post-seal. Retain convergence/gap without claiming optimum.',
        'sourceHashes':{f.name:sha(f) for f in p.iterdir()},'device':'cpu','backend':'numpy-fp64','cpuThreads':1}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from response_study import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/'payload'/n)!=h:raise ValueError('Staged source changed')
    cmd=[s.replace('response_study.py','landmark_routing_diagnostics.py') for s in command()]
    job=launch(cmd,experiment='tce-landmark-routing-diagnosis-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu'},timeout_s=600,
        detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 saved8ports,64fixed landmarks,3bandwidth scales, seal then paired scores and target-assisted3padding-pair affine/512step simplex fits. No model/projections/tokenizer/native/GPU. Expected under30s,finite600s,no retry; host-local saved arrays.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def load(item):
    if sha(item['path'])!=item['sha256']:raise ValueError('Saved input differs')
    with np.load(item['path'],allow_pickle=False) as f:return {k:f[k] for k in f.files}


def execute(root):
    guard();root=Path(root).resolve();r=read(root/'request.json');source=read(root/'source.json');start=time.perf_counter()
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/n)!=h:raise ValueError('Source changed')
    asset=load(source['asset']);ports={p['id']:load(p) for p in source['ports']};work=root/'predictions';work.mkdir();records=[];predictions={};route_rows=[];closure=[]
    for name,packet in ports.items():
        first_state=None
        for scale in r['bandwidthSquaredScales']:
            a={**asset,'bandwidthSquared':asset['bandwidthSquared']*scale};output,state=stream(packet,a);predictions[name,scale]=output
            if first_state is None:first_state=state
            closure.append({'id':name,'scale':scale,'stateMaxAbs':max(float(np.max(np.abs(state[k]-first_state[k]))) for k in ['M','z','maximum'])})
            path=work/(name+'-'+str(scale)+'.npz');np.savez(path,**output);records.append({'id':name,'scale':scale,'path':str(path),'sha256':sha(path)})
            for region,mask in [('all',np.ones(len(packet['mask']),dtype=bool)),('active',packet['mask'].astype(bool)),('padding',~packet['mask'].astype(bool))]:
                if mask.any():route_rows.append({'id':name,'scale':scale,'region':region,'queries':int(mask.sum()),
                    **{key:{'median':float(np.median(output[key][mask])),'min':float(output[key][mask].min()),'max':float(output[key][mask].max())}
                       for key in ['routeEntropy','routeMaxMass','nearestDistanceSquared']}})
    out=root/'response-result';out.mkdir();save(out/'prediction-seal.json',{'rows':records,'asset':source['asset'],'scope':r['scope'],'sourcePortHashes':source['ports']})
    seal_sha=sha(out/'prediction-seal.json');print('ROUTING_DIAG_SEALED',seal_sha,flush=True)
    targets={p['id']:load(p) for p in source['references']};old={p['id']:load(p) for p in source['predictions']}
    from encoder_atoms import floats
    projection=source['projection']
    if sha(projection['path'])!=projection['sha256']:raise ValueError('Projection changed')
    matrix=floats(np.load(projection['path'],allow_pickle=False)).astype(float);scores=[];pairs=[];capacities=[];capacity_arrays={};rows=0
    with (out/'observations.jsonl').open('x') as f:
        for (name,scale),bundle in predictions.items():
            pred=bundle['landmark64_raw'];target=targets[name]['raw'];mask=ports[name]['mask'].astype(bool)
            for region,selected in [('all',np.ones(len(mask),dtype=bool)),('active',mask),('padding',~mask)]:
                if selected.any():scores.append({'id':name,'scale':scale,'region':region,'head':summary(pred[selected],target[selected])})
        for a,b in source['pairs']:
            target=targets[b]['raw']-targets[a]['raw'];ma,mb=[ports[k]['mask'].astype(bool) for k in [a,b]]
            for scale in r['bandwidthSquaredScales']:
                pred=predictions[b,scale]['landmark64_raw']-predictions[a,scale]['landmark64_raw'];er,tr=[rowrms(z) for z in [pred-target,target]]
                for region,selected in [('all',np.ones(len(ma),dtype=bool)),('active_both',ma&mb),('padding_both',~ma&~mb),('mask_changed',ma!=mb)]:
                    if selected.any():pairs.append({'a':a,'b':b,'scale':scale,'region':region,'head':summary(pred[selected],target[selected]),'projection':summary(pred[selected]@matrix.T,target[selected]@matrix.T)})
                for i in range(len(ma)):
                    f.write(json.dumps({'a':a,'b':b,'scale':scale,'position':i,'activeA':bool(ma[i]),'activeB':bool(mb[i]),
                        'headErrorRms':float(er[i]),'targetChangeRms':float(tr[i])},allow_nan=False)+'\n');rows+=1
            selected=~ma&~mb
            if not selected.any():continue
            pa,pb=[ports[n] for n in [a,b]]
            if not np.array_equal(pa['q'][selected],pb['q'][selected]):raise ValueError('Pair padding queries differ')
            # Padded reads admit no new keys. The stored final landmark statistics therefore give the exact64reader response vectors.
            if ma[np.flatnonzero(selected)[0]:].any() or mb[np.flatnonzero(selected)[0]:].any():raise ValueError('Padding is not a terminal suffix')
            ha=old[a]['state_M']/old[a]['state_z'][:,None];hb=old[b]['state_M']/old[b]['state_z'][:,None];D=hb-ha
            q=floats(pa['q'][selected]).astype(float)/pa['q'].shape[1]**.25;w,_,_=routes(q,asset)
            fixed=w@D;reference=old[b]['landmark64_raw'][selected]-old[a]['landmark64_raw'][selected]
            fit,record=capacity(D,target[selected],w,r['convexFitIterations']);prefix=a.replace('-','_')
            for key,value in fit.items():capacity_arrays[prefix+'__'+key]=value
            capacity_arrays[prefix+'__readerChanges']=D;capacity_arrays[prefix+'__target']=target[selected];capacity_arrays[prefix+'__positions']=np.flatnonzero(selected)
            capacities.append({'a':a,'b':b,'paddingQueries':int(selected.sum()),'fixedRouteResponseMaxAbsClosure':float(np.max(np.abs(fixed-reference))),
                'originalSmooth':summary(fixed,target[selected]),'nearest':summary(old[b]['nearest64_raw'][selected]-old[a]['nearest64_raw'][selected],target[selected]),
                **record})
    path=root/'capacity-fits.npz';np.savez(path,**capacity_arrays)
    save(out/'capacity-receipt.json',{'rows':capacities,'arrays':{'path':str(path),'sha256':sha(path)},
        'scope':'Post-seal oracle fits, not serving assets. All padding rows within a pair share the same64response-vector bank but have separate target-assisted weights.'})
    if sha(out/'prediction-seal.json')!=seal_sha:raise ValueError('Seal changed')
    save(out/'reference-receipt.json',{'rows':source['references'],'parentPredictions':source['predictions'],'projection':projection,'scope':'Existing saved references/predictions, opened after new bandwidth candidates sealed.'})
    shutil.copyfile(root/'request.json',out/'request.json')
    save(out/'report.json',{'schema':'tce-landmark-routing-diagnostic-result-v1','requestSHA256':r['requestSHA256'],'scope':r['scope'],
        'scores':scores,'pairedScores':pairs,'routes':route_rows,'capacities':capacities,'stateClosures':closure,
        'predictionSealSHA256':seal_sha,'workerSeconds':time.perf_counter()-start,'observations':rows,
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed','goalStatus':'Active; development routing/capacity diagnosis, no new held-out or image result.'})
    print('LANDMARK_ROUTING_DIAG_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
