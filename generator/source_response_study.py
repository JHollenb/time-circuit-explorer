"""One model-free lease: response fit, fresh serving, seal, then old labels."""
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
OWNED=['source_response.py','source_response_study.py','encoder_atoms.py','paired_fields.py','masked_kv_study.py','flux_clock_study.py']
WHEEL='mdb_runtime-0.1.0-py3-none-any.whl'


def stage(root):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    from mdb.job_support import one_lease_scheduling_contract
    parent=HERE.parent/'artifacts/source-native-reader-v1';c,j,old=checked_collection(parent,current_workers=False);api=Api();live=api.json('GET','/api/jobs/'+j['job_id'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody']:raise ValueError('Source-native parent differs')
    p0=parent/'result';im=read(p0/'images.json');corp=read(p0/'corpus.json');fit=read(p0/'fit.json');report=read(p0/'report.json')
    if report['candidateSealSHA256']!=sha(p0/'candidate-seal.json') or read(parent/'retained-array-verification.json')['candidateSealSHA256']!=report['candidateSealSHA256']:raise ValueError('Parent release/complete bytes differ')
    reads={x['id']:x for x in im['reads']};records={x['id']:x for x in corp['records']};pairs=[];targets={}
    feedback=[]
    for row in corp['records']:
        if row['population']=='feedback' and row['kind']=='remote-edit':feedback.append([row['context']+'.canonical',row['id']])
    if len(feedback)!=2:raise ValueError('Same-location feedback differs')
    for pop,ids in [('discovery',corp['discoveryPairs']),('feedback',feedback),('evaluation',corp['evaluationPairs'])]:
        for a,b in ids:
            label=pop+'-'+records[a]['context'];pairs.append({'id':label,'population':pop,'location':'top-left' if pop=='evaluation' else 'bottom-right',
                'A':{k:reads[a][k] for k in ['z','previous','H0']},'B':{k:reads[b][k] for k in ['z','previous','H0']}})
            targets[label]=[records[a]['target'],records[b]['target']]
    root=Path(root).resolve();root.mkdir(exist_ok=False,parents=True);p=root/'payload';p.mkdir()
    for n in OWNED:
        if n in old['ownedSources'] and sha(HERE/n)!=old['sourceHashes'][n]:raise ValueError('Qualified helper changed')
        shutil.copyfile(HERE/n,p/n)
    shutil.copyfile(parent/'payload'/WHEEL,p/WHEEL);shutil.copyfile(HERE.parent/'docs/plans/source-native-response-writer.md',p/'frozen-plan.md')
    base=j['result']['diagnostics']['cwd'];source={'scale':fit['fields']['scale'],'oldWeight':fit['fields']['weight'],'pairs':pairs,
        'protectedRoots':[base,*corp['sourceRoots']], 'deniedCanaries':[targets[pairs[-1]['id']][1]['path'],str(Path(base)/'reader-00.png')],
        'parentRecords':{n:{'path':base+'/response-result/'+n,'sha256':sha(p0/n)} for n in ['report.json','images.json','fit.json','coefficient-seal.json','candidate-seal.json']}}
    save(p/'source.json',source);save(p/'targets.json',targets)
    contract=one_lease_scheduling_contract(model_free=True,model_loads=0,logical_branches=24,logical_operations=25,logical_checkpoint_replays=0,
        extra={'task_family':'analysis','controlPlaneImports':'Current canonical Frame/Act imports Torch; no model construction/evaluation. Numerical fit/response actors prohibit neural frameworks.'})
    r={'schema':'tce-source-response-request-v1','parentJobID':j['job_id'],'parentRequestSHA256':old['requestSHA256'],'ownedSources':OWNED,
        'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'scheduling_contract':contract,'needs':{'cpu':True},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':300,
        'protocol':'One discovery response-only dual ridge .01/nointercept/unitgain, inherited H0 scales. Sealcoeff before fresh actor emits rawFP64 response-only/originalBH/zero on2discovery2feedback4evaluationpairs. Sealallresponses before old evaluationlabels; no new learned reader,parameter bank,encoder,VAE,image ornoise. Seed/locationconfounded.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':r['requestSHA256'],'pairs':len(pairs)}


def submit(root):
    from run import command
    from mdb.job_support import guarded_mrun_launch
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same job')
    cmd=command();cmd=cmd[:cmd.index('--with')]
    for req in ['torch==2.13.0','numpy>=2.0','psutil>=5.9','./'+WHEEL]:cmd+=['--with',req]
    i=cmd.index('uv');cmd[i:i]=['PYTHONDONTWRITEBYTECODE=1','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    cmd+=['python','-u','source_response_study.py','execute','.']
    job=guarded_mrun_launch(cmd,request=r,experiment='tce-source-response-v1',task='analysis',model=None,payload=root/'payload',pin='beast',needs=r['needs'],gpu=False,
        mrun_config={k:r[k] for k in ['schema','requestSHA256','backend','device','cpuThreads','scheduling_contract']},
        timeout_s=r['timeoutSeconds'],detach=True,retry_on_kill=False,preflight=True,
        note='CPU1 saved H0 only.2discoverypairs/1888rows/3072features ->one dual FP64ridge .01/nointercept/unitgain; sealcoeff ->fresh response actor24raw outputs ->allseal ->old2feedback4evalpairs. CurrentFrames importTorch but0model loads/evals;actors forbidframeworks. No new H0/images/native/VAE. Intent-first admission,beastlocal arrays,expected<2min/300scap,no retries;seed/locationconfounded.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def score(root,out,started,phases):
    from source_response import mask
    root=Path(root);s=read(root/'source.json');targets=read(root/'targets.json');served=read(root/'served/serving.json');seal_sha=sha(root/'candidate-seal.json');r=read(root/'request.json')
    native={};diag={};accounts=[];outputs=[];observations=0
    for pair in s['pairs']:
        row=next(x for x in served['rows'] if x['id']==pair['id']);selected=load(row['fields']['mask']).astype(bool)
        a,b=[load(x) for x in targets[pair['id']]];native[pair['id']]=(b-a)[0,selected]
        diag[pair['id']]={'target':retain(root/'arrays/diagnostic'/pair['id'],'target',native[pair['id']])}
    with (out/'observations.jsonl').open('x') as stream:
        for row in served['rows']:
            pred=load(row['prediction']);target=native[row['id']];label={k:row[k] for k in ['id','population','location','method']};error=pred-target
            energy={'predictionEnergy':float(np.mean(pred*pred)),'targetEnergy':float(np.mean(target*target)),
                'twicePredictionTarget':float(2*np.mean(pred*target)),'errorEnergy':float(np.mean(error*error))}
            accounts.append({**label,**metrics(pred,target),**energy,'applicableQueries':row['applicableQueries'],'inapplicableQueries':row['inapplicableQueries']})
            for index,(a,b) in enumerate(zip(pred,target)):
                stream.write(json.dumps({**label,'position':index,**metrics(a,b)},allow_nan=False)+'\n');observations+=1
            for index in range(128):outputs.append({**label,'coordinate':index,**metrics(pred[:,index],target[:,index])})
    if sha(root/'candidate-seal.json')!=seal_sha:raise ValueError('Candidate seal changed')
    for n in ['request.json','source.json','targets.json','frozen-plan.md','coefficient-seal.json','candidate-seal.json']:shutil.copyfile(root/n,out/n)
    shutil.copyfile(root/'fit/fit.json',out/'fit.json');shutil.copyfile(root/'served/serving.json',out/'serving.json');save(out/'diagnostic-fields.json',diag);save(out/'output-accounts.json',outputs)
    fit=read(root/'fit/fit.json')
    save(out/'report.json',{'schema':'tce-source-response-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal_sha,'accounts':accounts,'observations':observations,
        'phases':phases,'workerSecondsAfterImports':time.perf_counter()-started,'fitCost':{k:fit[k] for k in ['gramNominalMACs','coefficientNominalMACs','estimatedSolveFLOPs','postFitNominalMACs','coefficientBytes']},
        'servingNominalMACs':sum(x['nominalMACs'] for x in served['rows']),'work':{'newModelEvaluations':0,'newH0Reads':0,'newImages':0,'closedFormFits':1,'rawResponses':24},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'limits':['Retrospective supplied H0/native velocity labels, not raw-prompt generation or a complete velocity producer. Original source-reader/label production remains paid.',
            'Both methods produce raw FP64 responses before output quantization; inherited endpoint-fit B_H remains the control.',
            'Same-location feedback and top-left evaluation change both seed and location, so transfer heterogeneity is not isolated location causality.',
            'Control process imports Torch only for current Saturn Frame/Act; fresh NumPy actors prohibit all neural frameworks/models/network.']})


def execute(root):
    from mdb.titan import Frame,ActSpec,ActRegistry
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'source.json');targets=read(root/'targets.json');start=time.perf_counter();phases=[];out=root/'response-result';out.mkdir()
    future={'released':False}
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.suffix in {'.safetensors','.pt','.pth'}:raise PermissionError('No model bytes in response control plane')
            if p.suffix in {'.npy','.npz'} and not p.is_relative_to(root) and not future['released']:raise PermissionError('Old target/input arrays belong only in isolated declared actors until release')
    sys.addaudithook(audit)
    fit_pairs=[{**p,'targets':targets[p['id']]} for p in s['pairs'] if p['population']=='discovery']
    fs={'pairs':fit_pairs,'scale':s['scale'],'output':str(root/'fit'),'protectedRoots':s['protectedRoots'],'deniedCanaries':s['deniedCanaries']};save(root/'fit-spec.json',fs)
    registry=ActRegistry();act=ActSpec('tce.response_only_fit','v1',reads=('discoveryContract',),writes=('coefficients',),write_kinds={'coefficients':'tensor-artifact'},write_abis={'coefficients':{'shape':[3072,128],'dtype':'float64'}},numerical_contract='one-FP64-dual-ridge-no-intercept-unitgain')
    def propose(values):
        if values['discoveryContract']!=fs:raise ValueError('Discovery source contract differs')
        tick=time.perf_counter();subprocess.run([sys.executable,'-u',str(root/'source_response.py'),'fit',str(root/'fit-spec.json')],check=True,timeout=180);phases.append({'action':'fit','wallSecondsIncludingFreshProcess':time.perf_counter()-tick})
        info=read(root/'fit/fit.json');return {'coefficients':info['fields']['weight']},{'fitReceiptSHA256':sha(root/'fit/fit.json'),'epochs':0,'gain':1}
    registry.register(act,'isolated-discovery-response-fit',propose)
    zero=retain(root/'arrays/baseline','coefficients',np.zeros((3072,128),np.float64))
    source=Frame.from_values({'discoveryContract':fs,'coefficients':zero},kinds={'discoveryContract':'source-artifact-contract','coefficients':'tensor-artifact'},
        abis={'coefficients':{'shape':[3072,128],'dtype':'float64'}},program_fingerprint=r['requestSHA256'])
    fitted,ev=registry.run(act.identity,'isolated-discovery-response-fit',source);info=read(root/'fit/fit.json');baseline=source
    if zero['payloadSHA256']!=info['fields']['baseline']['payloadSHA256']:raise ValueError('Fresh actor zero baseline differs')
    restored=baseline
    if restored.fingerprint!=baseline.fingerprint:raise ValueError('Immutable baseline selection differs')
    save(root/'coefficient-seal.json',{'schema':'tce-source-response-coefficient-seal-v1','fitReceiptSHA256':sha(root/'fit/fit.json'),'fields':info['fields'],'baselineField':zero,'sourceFrame':source.to_dict(),'fittedFrame':fitted.to_dict(),'baselineFrame':baseline.to_dict(),'restoredFrame':restored.to_dict(),'fitEvidence':ev.to_dict(),'scope':'Immutable coefficient branches, not mutable optimizer rollback.'})
    ss={'pairs':s['pairs'],'scale':s['scale'],'oldWeight':s['oldWeight'],'coefficientSeal':str(root/'coefficient-seal.json'),'fitReceipt':str(root/'fit/fit.json'),'output':str(root/'served'),'protectedRoots':s['protectedRoots'],'deniedCanaries':s['deniedCanaries']};save(root/'serve-spec.json',ss)
    tick=time.perf_counter();subprocess.run([sys.executable,'-u',str(root/'source_response.py'),'serve',str(root/'serve-spec.json')],check=True,timeout=180);phases.append({'action':'serve','wallSecondsIncludingFreshProcess':time.perf_counter()-tick})
    served=read(root/'served/serving.json')
    if len(served['rows'])!=24 or served['coefficientSealSHA256']!=sha(root/'coefficient-seal.json'):raise ValueError('Response coverage/seal differs')
    for row in served['rows']:
        for item in [row['prediction'],*row['fields'].values()]:load(item)
    save(root/'candidate-seal.json',{'schema':'tce-source-response-seal-v1','requestSHA256':r['requestSHA256'],'servingSHA256':sha(root/'served/serving.json'),'coefficientSealSHA256':sha(root/'coefficient-seal.json'),'rows':served['rows']})
    print('SOURCE_RESPONSES_SEALED',sha(root/'candidate-seal.json'),flush=True);future['released']=True
    score(root,out,start,phases);print('SOURCE_RESPONSE_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(json.dumps(v))
