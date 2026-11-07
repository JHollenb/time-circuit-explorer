"""Saved-only source-fixed radial-response test; no learned maps or images."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
from encoder_atoms import np,read,save,sha,identity,guard
from paired_fields import load,retain,metrics
from masked_kv_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['two_clock_carry.py','encoder_atoms.py','paired_fields.py','masked_kv_study.py']


def energy_account(actual,terms):
    actual=np.asarray(actual,dtype=np.float64);terms={k:np.asarray(v,dtype=np.float64) for k,v in terms.items()}
    if any(v.shape!=actual.shape for v in terms.values()):raise ValueError('Term geometry differs')
    names=list(terms);error=actual-sum(terms.values(),np.zeros_like(actual))
    return {'actualEnergy':float(np.sum(actual*actual)),'termEnergy':{k:float(np.sum(v*v)) for k,v in terms.items()},
        'twiceCrossTerms':[{'a':a,'b':b,'value':float(2*np.sum(terms[a]*terms[b]))} for i,a in enumerate(names) for b in names[i+1:]],
        'closureRMS':float(np.sqrt(np.mean(error*error))),'closureMaxAbs':float(np.max(np.abs(error))),'scalars':int(actual.size)}


def assay(z_n,v_n,y_n,e_n,z_h,v_h,y_h,e_h,sigma):
    values=[np.asarray(x,dtype=np.float64) for x in [z_n,v_n,y_n,e_n,z_h,v_h,y_h,e_h]]
    if not values[0].size or any(x.shape!=values[0].shape or not np.isfinite(x).all() for x in values):raise ValueError('Finite matching fields required')
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('Positive finite final sigma required')
    z_n,v_n,y_n,e_n,z_h,v_h,y_h,e_h=values
    d=z_h-z_n;dv=v_h-v_n;w=-sigma*dv;r=e_h-e_n;o=y_h-y_n;pred=d/sigma
    dd=float(np.sum(d*d));ww=float(np.sum(w*w));oo=float(np.sum(o*o));lam=float(np.sum(d*w)/dd) if dd else None
    fields={'incomingCarry':d,'deltaVelocity':dv,'radialVelocityPrediction':pred,'actualWrite':w,
        'finiteRemainder':r,'outgoingDifference':o,'cleanEstimateDifference':d+w,'radialVelocityError':pred-dv}
    projection=None
    if lam is not None:
        parallel=lam*d;perp=w-parallel;fields.update(writeParallel=parallel,writeOrthogonal=perp)
        projection={'lambda':lam,'radialCancellationStrength':-lam,'parallelEnergy':float(np.sum(parallel*parallel)),
            'orthogonalEnergy':float(np.sum(perp*perp)),'orthogonalToIncomingDot':float(np.sum(perp*d)),
            'orthogonalWriteFractionOfEnergy':float(np.sum(perp*perp)/ww) if ww else None,
            'factoredAccount':energy_account(o,{'remainingRadialCarry':(1+lam)*d,'orthogonalWrite':perp,'finiteRemainder':r})}
    account={'incomingRMS':float(np.sqrt(dd/d.size)),'outgoingRMS':float(np.sqrt(oo/o.size)),
        'outgoingIncomingGain':float(np.sqrt(oo/dd)) if dd else None,
        'cleanEstimateIncomingGain':float(np.linalg.norm(d+w)/np.sqrt(dd)) if dd else None,
        'writeVsCancellationCosine':float(np.sum(-d*w)/np.sqrt(dd*ww)) if dd and ww else None,
        'radialVelocityPredictionVsActual':metrics(pred,dv),'projection':projection,
        'carryWriteFiniteAccount':energy_account(o,{'incomingCarry':d,'actualWrite':w,'finiteRemainder':r}),
        'zeroIncomingNorm':dd==0}
    return fields,account


def stage(root):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    parent=HERE.parent/'artifacts/two-clock-v1';c,j,r=checked_collection(parent,current_workers=False)
    fresh=Api().json('GET','/api/jobs/'+j['job_id'])
    if fresh['state']!='succeeded' or fresh['payload_custody']!=c['custody']:raise ValueError('Primary terminal custody differs')
    p0=parent/'result';im=read(p0/'images.json');targets=read(p0/'targets.json');finite=read(p0/'finite-fields.json');s=read(p0/'inputs.json');report=read(p0/'report.json')
    if report['candidateSealSHA256']!=sha(p0/'candidate-seal.json') or s['schedule']['sigmas'][4]!=0:raise ValueError('Released final-step contract differs')
    root=Path(root).resolve();root.mkdir(exist_ok=False,parents=True);p=root/'payload';p.mkdir()
    for n in OWNED:
        if n in r['ownedSources'] and sha(HERE/n)!=r['sourceHashes'][n]:raise ValueError('Primary helper changed')
        shutil.copyfile(HERE/n,p/n)
    shutil.copyfile(HERE.parent/'docs/plans/two-clock-carry.md',p/'frozen-plan.md')
    inputs={}
    for case in ['00','10']:
        inputs[case]={}
        for lane in ['NN','HN']:
            fs=targets['NN']['fields'][case] if lane=='NN' else im['fields']['HN-'+case]
            inputs[case][lane]={**{n:fs[n] for n in ['latent3','velocity3','latent4']},'epsilon3':finite[case]['clock3'+lane]}
    base=j['result']['diagnostics']['cwd']
    save(p/'inputs.json',{'fields':inputs,'sigma3':s['schedule']['sigmas'][3],'primaryJobID':j['job_id'],'primaryRoot':base,
        'allowedRoots':[base,s['nativeParentRoot']],
        'primaryRecords':{n:{'path':base+'/response-result/'+n,'sha256':sha(p0/n)} for n in ['images.json','targets.json','finite-fields.json','candidate-seal.json','report.json']}})
    q={'schema':'tce-two-clock-carry-request-v1','parentJobID':j['job_id'],'parentRequestSHA256':r['requestSHA256'],
        'parentReportSHA256':sha(p0/'report.json'),'candidateSealSHA256':sha(p0/'candidate-seal.json'),
        'ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'device':'cpu','backend':'numpy-fp64',
        'cpuThreads':1,'timeoutSeconds':180,'protocol':'Postseal saved-only D+W+independentR and source-fixed deltaVHat=D/sigma3. Diagnostic projection only; no fit/weights/learnedmaps/newimages or new neural evaluation.'}
    q['requestSHA256']=identity(q);save(root/'request.json',q);save(p/'request.json',q)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':q['requestSHA256']}


def submit(root):
    from response_study import command
    from mrun.client.submit import launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same job')
    cmd=command();cmd[-3]='two_clock_carry.py'
    job=launch(cmd,experiment='tce-two-clock-carry-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'requestSHA256':r['requestSHA256'],'specimen':'saved-two-clock-NN-HN','device':'cpu','backend':'numpy-fp64','cpu_threads':1,'task_family':'saved-algebra'},
        timeout_s=180,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='Saved-only postseal NN/HN D+W+independentfiniteR; frozenradial deltaV=D/sigma3, retrospectiveparallel/orthogonal projection.2contexts+dependentcolor,3072tokenrows/fullvectors.0newmodelcalls/weights/images/fit. CPU1 actualNumPy, automaticreservation,noRAM/VRAM/CPU/wallguesses,180scap,noretries; hostlocalarrays.')
    v={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',v);return v


def execute(root):
    guard();root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json');start=time.perf_counter();cpu=time.process_time()
    out=root/'response-result';out.mkdir(exist_ok=False)
    for item in s['primaryRecords'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Released primary record changed')
    values={c:{lane:{n:load(item) for n,item in fs.items()} for lane,fs in lanes.items()} for c,lanes in s['fields'].items()}
    if any(a.shape!=(1,1024,128) for lanes in values.values() for fs in lanes.values() for a in fs.values()):raise ValueError('Actual retained geometry differs')
    values['color']={lane:{n:values['10'][lane][n]-values['00'][lane][n] for n in fs} for lane,fs in values['00'].items()}
    rows=[];fields={};observations=0;arithmetic=0.
    with (out/'observations.jsonl').open('x') as stream:
        for domain,lanes in values.items():
            args=[lanes[lane][n] for lane in ['NN','HN'] for n in ['latent3','velocity3','latent4','epsilon3']]
            tick=time.perf_counter();vectors,account=assay(*args,s['sigma3']);arithmetic+=time.perf_counter()-tick
            fields[domain]={n:retain(root/'arrays'/domain,n,a) for n,a in vectors.items()}
            rows.append({'domain':domain,'scope':'dependent color contrast, not a single-state Jacobian' if domain=='color' else 'one measured incoming displacement',**account})
            for position in range(1024):
                _,local=assay(*[a[0,position] for a in args],s['sigma3'])
                stream.write(json.dumps({'domain':domain,'position':position,'projectionScope':'token-local diagnostic',**local},allow_nan=False)+'\n');observations+=1
    for item in s['primaryRecords'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Primary changed during assay')
    for n in ['request.json','inputs.json','frozen-plan.md']:shutil.copyfile(root/n,out/n)
    save(out/'fields.json',fields);save(out/'report.json',{'schema':'tce-two-clock-carry-result-v1','requestSHA256':r['requestSHA256'],
        'primaryJobID':s['primaryJobID'],'candidateSealSHA256':r['candidateSealSHA256'],'sigma3':s['sigma3'],'rows':rows,'observations':observations,
        'threeGlobalAssaysArithmeticSeconds':arithmetic,'workerSecondsAfterImports':time.perf_counter()-start,'cpuSecondsAfterImports':time.process_time()-cpu,
        'numpyVersion':np.__version__,'threadEnvironment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
        'work':{'newEncoderEvaluations':0,'newDenoiserEvaluations':0,'newVAEEvaluations':0,'newImages':0,'learnedMapEvaluations':0,'sourceWeightBytesRead':0,'fittedServingCoefficients':0},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limitations':['Retrospective authenticated native NN/HN states and velocities are supplied paid inputs, not predicted for an unseen prompt.',
            'Fixed radial hypothesis and diagnostic projection are distinct; projection is not an eligible serving fit.',
            'Two known contexts/one seed and one displacement each; no global contraction, semantic quality or arbitrary-prompt zero-neural claim.']})
    print('TWO_CLOCK_CARRY_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(json.dumps(v))
