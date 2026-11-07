"""One CPU lease: fixed factors -> coefficient Act -> sealed forecasts -> old scores."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from encoder_atoms import np,read,save,sha,identity
from paired_fields import load,retain,metrics
from flux_clock_study import verify

HERE=Path(__file__).resolve().parent
OWNED=['minimum_response.py','minimum_response_study.py','source_response.py','encoder_atoms.py','paired_fields.py','masked_kv_study.py','flux_clock_study.py']
WHEEL='mdb_runtime-0.1.0-py3-none-any.whl'


def stage(root):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    from mdb.job_support import one_lease_scheduling_contract
    parent=HERE.parent/'artifacts/source-spectrum-v1';response=HERE.parent/'artifacts/source-response-v2';api=Api()
    c,j,old=checked_collection(parent,current_workers=False);rc,rj,rr=checked_collection(response,current_workers=False)
    for coll,job in [(c,j),(rc,rj)]:
        live=api.json('GET','/api/jobs/'+job['job_id'])
        if live['state']!='succeeded' or live['payload_custody']!=coll['custody']:raise ValueError('Qualified parent differs')
    p0=parent/'result';p1=response/'result';sf=read(p0/'fields.json');source0=read(p0/'source.json');served=read(p1/'serving.json');source1=read(p1/'source.json')
    for run,p,seal in [(parent,p0,'analysis-seal.json'),(response,p1,'candidate-seal.json')]:
        v=read(run/('retained-array-verification.json' if run==parent else 'retained-array-verification-v2.json'))
        if read(p/'report.json')[('analysisSealSHA256' if run==parent else 'candidateSealSHA256')]!=sha(p/seal) or v[('analysisSealSHA256' if run==parent else 'candidateSealSHA256')]!=sha(p/seal):raise ValueError('Parent release/bytes differ')
    rows=[x for x in served['rows'] if x['method']=='response-only'];pairs=[{k:x[k] for k in ['id','population','location','fields','applicableQueries','inapplicableQueries']} for x in rows]
    if len(pairs)!=8 or len({x['id'] for x in pairs})!=8 or any(x['applicableQueries']!=944 or x['inapplicableQueries'] for x in pairs):raise ValueError('Archived response population differs')
    for pair in source0['pairs']:
        row=next(x for x in pairs if x['id']==pair['id'])
        if pair['X']['payloadSHA256']!=row['fields']['X']['payloadSHA256'] or pair['mask']['payloadSHA256']!=row['fields']['mask']['payloadSHA256']:raise ValueError('Decomposition/forecast source differs')
    if source0['scale']!=served['sourceScale']:raise ValueError('Inherited scale differs')
    root=Path(root).resolve();root.mkdir(exist_ok=False,parents=True);p=root/'payload';p.mkdir()
    for n in OWNED:
        for qualified in [old,rr]:
            if n in qualified['ownedSources'] and sha(HERE/n)!=qualified['sourceHashes'][n]:raise ValueError('Qualified helper changed')
        shutil.copyfile(HERE/n,p/n)
    shutil.copyfile(response/'payload'/WHEEL,p/WHEEL);shutil.copyfile(HERE.parent/'docs/plans/minimum-norm-source-response.md',p/'frozen-plan.md')
    base=j['result']['diagnostics']['cwd'];rb=rj['result']['diagnostics']['cwd'];diag=read(p1/'diagnostic-fields.json');canary=next(x for x in served['rows'] if x['population']=='evaluation')
    source={'factors':{k:sf[k] for k in ['Vt','sigma','targetModes']},'scale':source0['scale'],'pairs':pairs,
        'protectedRoots':[base,rb,*source1['protectedRoots']],'deniedCanaries':[diag[canary['id']]['target']['path'],canary['prediction']['path']],
        'parentRecords':{label+'/'+n:{'path':remote+'/response-result/'+n,'sha256':sha(local/n)} for label,remote,local,names in [
            ('spectrum',base,p0,['report.json','fields.json','analysis-seal.json','source.json']),
            ('response',rb,p1,['report.json','serving.json','diagnostic-fields.json','candidate-seal.json','coefficient-seal.json'])] for n in names}}
    save(p/'source.json',source);save(p/'scoring-source.json',{'targets':{k:v['target'] for k,v in diag.items()},'controls':served['rows']})
    contract=one_lease_scheduling_contract(model_free=True,model_loads=0,logical_branches=32,logical_operations=9,logical_checkpoint_replays=0,
        extra={'task_family':'analysis','controlPlaneImports':'Current canonical Frames/Acts import Torch; no model construction/evaluation. Fresh coefficient/forecast actors exclude neural frameworks.'})
    r={'schema':'tce-minimum-response-request-v1','parentJobID':j['job_id'],'responseParentJobID':rj['job_id'],'parentRequestSHA256':old['requestSHA256'],
        'ownedSources':OWNED,'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'scheduling_contract':contract,'needs':{'cpu':True},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':300,
        'protocol':'One allmode minimum-norm B from retainedV/sigma/C, inheritedscale/nointercept/unitgain. Sealcoeff ->freshactor8archivedinputresponses ->allseal ->oldtargets/24fixedcontrols. No newSVD/rank/gain/regularization sweep/model/H0/VAE/image/noise. Retrospective seed/locationconfounded.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256'],'pairs':len(pairs)}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same job')
    cmd=command();cmd=cmd[:cmd.index('--with')]
    for req in ['torch==2.13.0','numpy==2.5.4','psutil>=5.9','./'+WHEEL]:cmd+=['--with',req]
    i=cmd.index('uv');cmd[i:i]=['PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    cmd+=['python','-u','minimum_response_study.py','execute','.']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-minimum-response-v1',task='analysis',model=None,payload=root/'payload',pin='beast',needs=r['needs'],gpu=False,
        mrun_config={k:r[k] for k in ['schema','requestSHA256','backend','device','cpuThreads','scheduling_contract']},timeout_s=r['timeoutSeconds'],detach=True,retry_on_kill=False,preflight=True,
        note='CPU1 saved Vt1888x3072/sigma/C1888x128 ->one allmode minimum-normB;seal ->freshactor8rawresponses ->seal ->oldtargets/24controls. .742Bfit+2.970BserveMAC;no newSVD/sweep/model/H0/VAE/images. Frames importTorch,0model loads/evals;actors forbidframeworks/targets. Intent-first resources;priorcompatibleCPU833.9MBRSS/0VRAM;beastlocal arrays,expected<2min/300scap,noretries. Historicalroles/seedlocationconfounded.')
    v={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',v);return v


def score(root,out,start,phases):
    s=read(root/'source.json');sc=read(root/'scoring-source.json');served=read(root/'served/serving.json');r=read(root/'request.json');seal_sha=sha(root/'candidate-seal.json')
    accounts=[];outputs=[];observations=0;targets={k:load(x) for k,x in sc['targets'].items()};rows=served['rows']+sc['controls']
    with (out/'observations.jsonl').open('x') as stream:
        for row in rows:
            pred=load(row['prediction']);target=targets[row['id']];label={k:row[k] for k in ['id','population','location','method']};e=pred-target
            energy={'predictionEnergy':float(np.mean(pred*pred)),'targetEnergy':float(np.mean(target*target)),'twicePredictionTarget':float(2*np.mean(pred*target)),'errorEnergy':float(np.mean(e*e))}
            accounts.append({**label,**metrics(pred,target),**energy,'applicableQueries':row['applicableQueries'],'inapplicableQueries':row['inapplicableQueries']})
            for index,(a,b) in enumerate(zip(pred,target)):
                stream.write(json.dumps({**label,'position':index,**metrics(a,b)},allow_nan=False)+'\n');observations+=1
            for index in range(128):outputs.append({**label,'coordinate':index,**metrics(pred[:,index],target[:,index])})
    if sha(root/'candidate-seal.json')!=seal_sha:raise ValueError('Candidate seal changed')
    for n in ['request.json','source.json','scoring-source.json','frozen-plan.md','coefficient-seal.json','candidate-seal.json']:shutil.copyfile(root/n,out/n)
    shutil.copyfile(root/'fit/fit.json',out/'fit.json');shutil.copyfile(root/'served/serving.json',out/'serving.json');save(out/'output-accounts.json',outputs);fit=read(root/'fit/fit.json')
    save(out/'report.json',{'schema':'tce-minimum-response-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal_sha,'accounts':accounts,'observations':observations,'phases':phases,
        'workerSecondsAfterImports':time.perf_counter()-start,'fitCost':{k:fit[k] for k in ['coefficientNominalMACs','coefficientBytes','coefficientEnergy','modeCoefficientEnergy']},
        'servingNominalMACs':sum(x['nominalMACs'] for x in served['rows']),'work':{'SVDs':0,'newLearnedCoefficientMaps':1,'newLearnedResponseProducts':8,'reusedControlResponses':24,'originalModelEvaluations':0,'newH0Computations':0,'newImages':0},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limits':['Retrospective supplied learned states/labels; not raw-prompt serving, complete velocity or a zero-learned generator.',
            'Allmode source interpolation is expected algebraically; numerical rank/source fitting do not certify transferable information.',
            'One untuned regularization endpoint, not a rank/gain/regularization sweep. Seed/location confounded and paired roles/queries dependent.',
            'Original H0/native labels and previous factorization/control prediction remain paid. New learned affine arithmetic explicitly charged.',
            'Canonical Frame/Act control imports Torch without model evaluation; fresh numerical actors prohibit frameworks/models/network/targets.']})


def execute(root):
    from mdb.titan import Frame,ActSpec,ActRegistry
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'source.json');start=time.perf_counter();phases=[];out=root/'response-result';out.mkdir();future={'released':False}
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.suffix in {'.safetensors','.pt','.pth'}:raise PermissionError('No model bytes in coefficient control plane')
            if not future['released'] and ((p.suffix in {'.npy','.npz'} and not p.is_relative_to(root)) or p==root/'scoring-source.json'):raise PermissionError('Inputs/old labels belong to isolated actors until all forecasts seal')
    sys.addaudithook(audit)
    fs={k:s[k] for k in ['factors','scale','protectedRoots','deniedCanaries']};fs['output']=str(root/'fit');save(root/'fit-spec.json',fs)
    registry=ActRegistry();act=ActSpec('tce.minimum_norm_response','v1',reads=('factorContract',),writes=('coefficients',),write_kinds={'coefficients':'tensor-artifact'},
        write_abis={'coefficients':{'shape':[3072,128],'dtype':'float64'}},numerical_contract='allmode-FP64-retained-factor-minimum-norm-nointercept-unitgain')
    def propose(values):
        if values['factorContract']!=fs:raise ValueError('Factor source contract differs')
        tick=time.perf_counter();subprocess.run([sys.executable,'-u',str(root/'minimum_response.py'),'form',str(root/'fit-spec.json')],check=True,timeout=180);phases.append({'action':'form','wallSecondsIncludingFreshProcess':time.perf_counter()-tick})
        info=read(root/'fit/fit.json');return {'coefficients':info['fields']['weight']},{'fitReceiptSHA256':sha(root/'fit/fit.json'),'newSVDs':0,'gain':1,'allModesUsed':True}
    registry.register(act,'isolated-factor-response-form',propose);zero=retain(root/'arrays/baseline','coefficients',np.zeros((3072,128),np.float64))
    source=Frame.from_values({'factorContract':fs,'coefficients':zero},kinds={'factorContract':'source-artifact-contract','coefficients':'tensor-artifact'},abis={'coefficients':{'shape':[3072,128],'dtype':'float64'}},program_fingerprint=r['requestSHA256'])
    fitted,ev=registry.run(act.identity,'isolated-factor-response-form',source);info=read(root/'fit/fit.json');restored=source
    if restored.fingerprint!=source.fingerprint:raise ValueError('Immutable baseline reselection differs')
    save(root/'coefficient-seal.json',{'schema':'tce-minimum-response-coefficient-seal-v1','fitReceiptSHA256':sha(root/'fit/fit.json'),'fields':info['fields'],'baselineField':zero,
        'sourceFrame':source.to_dict(),'fittedFrame':fitted.to_dict(),'baselineFrame':source.to_dict(),'restoredFrame':restored.to_dict(),'fitEvidence':ev.to_dict(),'scope':'Immutable coefficient branches, not mutable optimizer rollback.'})
    ss={k:s[k] for k in ['pairs','scale','protectedRoots','deniedCanaries']};ss.update(coefficientSeal=str(root/'coefficient-seal.json'),fitReceipt=str(root/'fit/fit.json'),output=str(root/'served'));save(root/'serve-spec.json',ss)
    tick=time.perf_counter();subprocess.run([sys.executable,'-u',str(root/'minimum_response.py'),'serve',str(root/'serve-spec.json')],check=True,timeout=180);phases.append({'action':'serve','wallSecondsIncludingFreshProcess':time.perf_counter()-tick});served=read(root/'served/serving.json')
    if len(served['rows'])!=8 or served['coefficientSealSHA256']!=sha(root/'coefficient-seal.json'):raise ValueError('Forecast coverage/seal differs')
    for row in served['rows']:load(row['prediction'])
    save(root/'candidate-seal.json',{'schema':'tce-minimum-response-seal-v1','requestSHA256':r['requestSHA256'],'coefficientSealSHA256':sha(root/'coefficient-seal.json'),'servingSHA256':sha(root/'served/serving.json'),'rows':served['rows']})
    print('MINIMUM_RESPONSES_SEALED',sha(root/'candidate-seal.json'),flush=True);future['released']=True;score(root,out,start,phases);print('MINIMUM_RESPONSE_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(json.dumps(v))
