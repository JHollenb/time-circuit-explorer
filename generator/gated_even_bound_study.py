"""One finite saved-only CPU assay, with no model runtime or learned product."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from encoder_atoms import read,save,sha,identity

HERE=Path(__file__).resolve().parent
OWNED=['gated_even_bound.py','gated_even_bound_study.py','test_gated_even_bound.py','encoder_atoms.py','paired_fields.py']
PLAN='docs/plans/gated-even-operator-bound.md'
PARENTS={'quadratic':'odd-gate-v1','linear':'mean-gate-v1','covariance':'own-state-gated-covariance-v1'}


def bindings(q,l,c):
    qs=read(q/'inputs.json');qc=read(q/'candidate.json')['fields'];qa=read(q/'analysis-fields.json')['fields'];lc=read(l/'candidate.json')['fields'];la=read(l/'analysis-fields.json')['fields'];cc=read(c/'candidate.json')['fields']
    return {'D':qs['sourceFields']['weightD'],'gate':qs['sourceFields']['gate'],'support':qs['sourceFields']['selection_support'],
        'R':cc['centeredProduct'],'L':lc['linearProduct'],'B':la['changingProduct'],'T':qc['quadraticProduct'],'H':qa['evenProduct'],'F':qa['finiteProduct'],'closure78':qa['productClosure'],
        'YR':cc['centeredOutput'],'YL':lc['linearOutput'],'YT':qc['quadraticOutput'],'remaining':qa['remainingOutput'],
        'gatedRemaining':qa['gatedRemaining'],'gatedQuadratic':qc['gatedQuadratic']}


def stage(root):
    from mrun.client.api import Api
    from publish_source_native import checked
    from publish_odd_gate import audit
    api=Api();parents={};dirs={}
    audit(HERE.parent/'docs/results/odd-gate')
    for name,folder in PARENTS.items():
        directory,col,job,r=checked(HERE.parent/'artifacts'/folder);live=api.json('GET','/api/jobs/'+job['job_id'])
        if live['state']!='succeeded' or live['payload_custody']!=col['custody']:raise ValueError('Retained source custody differs')
        dirs[name]=directory/'result'
        parents[name]={'jobID':job['job_id'],'custody':col['custody'],'root':job['result']['diagnostics']['cwd'],
            'records':{n:sha(dirs[name]/n) for n in ['inputs.json','candidate.json','execution.json','candidate-seal.json','report.json','analysis-fields.json'] if (dirs[name]/n).exists()}}
        for n in OWNED:
            if n in r['ownedSources'] and sha(HERE/n)!=r['sourceHashes'][n]:raise ValueError('Frozen reused source differs')
    fields=bindings(dirs['quadratic'],dirs['linear'],dirs['covariance'])
    witnesses=[read(HERE.parent/'artifacts'/folder/'retained-array-verification.json') for folder in PARENTS.values()]
    for item in fields.values():
        if not any(v['files'].get(item['path'],{}).get('sha256')==item['sha256'] and v['npyPayloadWitnesses'].get(item['path'],{}).get('payloadSHA256')==item['payloadSHA256'] for v in witnesses):raise ValueError('Saved source full-byte witness absent')
    root=Path(root).resolve();root.mkdir(exist_ok=False,parents=True);payload=root/'payload';payload.mkdir()
    for n in OWNED:shutil.copyfile(HERE/n,payload/n)
    shutil.copyfile(HERE.parent/PLAN,payload/'frozen-plan.md')
    source={'fields':fields,'parents':parents,'sourceRoots':sorted({str(Path(x['path']).parent.parent.parent) for x in fields.values()}|{x['root'] for x in parents.values()}),
        'originalDLineage':read(dirs['quadratic']/'inputs.json')['originalDLineage'],
        'preStageReviewSnapshots':{n:sha(HERE.parent/n) for n in ['docs/results/odd-gate-evidence/advisor.json','docs/results/research-review-cycle78/advisor-survey.json','docs/results/storage-audit-cycle78/publication.json']}}
    save(payload/'inputs.json',source)
    host=next(x for x in api.json('GET','/api/hosts') if x['name']=='beast')
    save(payload/'resource-basis.json',{'sourceFileBytes':sum(x['bytes'] for x in fields.values()),'sourceFiles':len(fields),'memoryPolicy':'Read-only mmap FP64 products/map and BF16gate; rowwise norm scans,64output-channel map blocks, no full newproduct/output matrices. One retained1536-row scalar table and compact metadata.',
        'liveHost':{k:host[k] for k in ['name','active_jobs','admission','telemetry','ram_total_mb']},'measuredParent':{'jobID':parents['quadratic']['jobID'],'schedulerSeconds':43.15,'sampledRSSMB':2894.6,'scope':'Parent CUDA two learned products and fullanalysis, not a saved-only runtime estimate.'},
        'admission':'Intent-first saved-only CPU analysis, no hand-reserved RAM/VRAM. Hard host binding required by authenticated host-local retained arrays. Finite240s, one lease/no retries; disk output below8MB.'})
    r={'schema':'tce-gated-even-bound-request-v1','ownedSources':OWNED,'sourceHashes':{p.name:sha(p) for p in payload.iterdir()},'device':'cpu','backend':'numpy','dtype':'float64','timeoutSeconds':240,
        'plannedMatrixMACs':0,'plannedNeuralEvaluations':0,'fixedAddress':{'clock':3,'singleBlock':0},
        'protocol':'One retained-H operator attribution bound. Allrows/all-text-image, both productclosures/formation allowances, selected residue, historical contraction andgrouping envelope, directcheck, conditional norm/magnitude/signedalignment intervals. No learnedproduct/activation/model/fit/consumer/image. All sourceformation/originalmaps/historypaid.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(payload/'request.json',r)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'root':str(root),'requestSHA256':r['requestSHA256'],'sourceFiles':len(fields),'sourceFileBytes':sum(x['bytes'] for x in fields.values())}


def verify(root,r):
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/n)!=h:raise ValueError('Frozen source/input changed:'+n)


def submit(root):
    from mrun.client.submit import launch
    from response_study import command
    root=Path(root).resolve();r=read(root/'request.json');verify(root/'payload',r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same handle')
    cmd=command()[:-5]+['python','-u','gated_even_bound_study.py','execute','.'];cmd.insert(1,'PYTHONDONTWRITEBYTECODE=1')
    job=launch(cmd,experiment='tce-gated-even-bound-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={k:r[k] for k in ['schema','requestSHA256','device','backend','dtype','plannedMatrixMACs','plannedNeuralEvaluations']},
        timeout_s=240,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='Saved-only clock3single0 H/F residual-gated operator bounds:16authenticated arrays,read-onlymmap/rowwise scans/normreductions,zero newlearnedproducts/neural evals. ConditionalFP64/cuBLASmodelexplicit,bothclosures/selectedresidue/Qseparate/all1536rows. CPU1,intentfirst admission,240scap,noretry. No HDown/model/activation/head/image/fit/sweep; historicalworkpaid.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def execute(root):
    from encoder_atoms import guard,floats,np
    guard()
    from gated_even_bound import Bounds,MODEL,map_norms,row_record,summarize
    from paired_fields import retain
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json');started=time.perf_counter()
    fields={}
    for name,item in s['fields'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError('Saved input hash differs')
        a=np.load(item['path'],mmap_mode='r',allow_pickle=False)
        if list(a.shape)!=item['shape'] or str(a.dtype)!=item['storedDtype'] or hashlib.sha256(memoryview(a).cast('B')).hexdigest()!=item['payloadSHA256']:raise ValueError('Saved input ABI/payload differs')
        fields[name]=floats(a).astype(np.float64) if item['logicalDtype']=='torch.bfloat16' else a
    n=1536;m=9216;d=3072
    if fields['D'].shape!=(d,m) or fields['gate'].shape!=(1,1,d) or fields['support'].shape!=(2304,):raise ValueError('Receiver ABI differs')
    for name in ['R','L','B','T','H','F','closure78']:
        if fields[name].shape!=(1,n,m) or fields[name].dtype!=np.float64:raise ValueError('Product ABI differs')
    for name in ['YR','YL','YT','remaining','gatedRemaining','gatedQuadratic']:
        if fields[name].shape!=(1,n,d) or fields[name].dtype!=np.float64:raise ValueError('Output ABI differs')
    support=np.asarray(fields['support'],dtype=np.int64)
    if len(np.unique(support))!=2304 or support.min()<0 or support.max()>=m:raise ValueError('Support identity invalid')
    b=Bounds();omitted,maps=map_norms(fields['D'],fields['gate'].reshape(-1),support,b)
    out=root/'response-result';out.mkdir();rows=[]
    with (out/'observations.jsonl').open('x') as stream:
        for i in range(n):
            row=row_record(fields,i,omitted,maps,b);rows.append(row);stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    panels={name:summarize(rows[lo:hi],b) for name,lo,hi in [('all',0,n),('text',0,512),('image',512,n)]}
    columns=['remainingLower','remainingUpper','quadraticLower','quadraticUpper','etaUpper','evenLower','evenUpper','alignmentLower','alignmentUpper','directAlternativeEtaUpper']
    table=np.array([[*x['remainingNormInterval'],*x['quadraticNormInterval'],x['etaUpper'],*x['evenOutputNormInterval'],*x['quadraticEvenInnerInterval'],x['directAlternativeEtaUpper']] for x in rows],dtype=np.float64)
    retained=retain(root/'arrays','rowBounds',table)
    for name,item in s['fields'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError('Saved input mutation')
    for name in ['inputs.json','request.json','resource-basis.json','frozen-plan.md']:shutil.copyfile(root/name,out/name)
    save(out/'report.json',{'schema':'tce-gated-even-bound-result-v1','requestSHA256':r['requestSHA256'],'fixedAddress':r['fixedAddress'],'panels':panels,
        'mapNormUpper':maps,'numericalModel':MODEL,'numericalDiagnostics':b.status(),'rowBounds':retained,'rowBoundColumns':columns,'observations':n,
        'retainedAdverseTextRows':[rows[3],rows[35]],'workerWallSeconds':time.perf_counter()-started,
        'physicalWork':{'newMatrixMACs':0,'newLearnedProjections':0,'newNeuralEvaluations':0,'newCheckpointLoads':0,'newImages':0,'newFullProductArrays':0,'newRetainedArrays':1},
        'mechanics_status':'complete','instrument_status':'saved-number-operator-envelope-conditional-FP64','trend_status':'All conditional magnitude/alignment intervals retained without effect gate','terminal_status':'unassessed',
        'scope':'Attribution to exact projection of retainedH, not native/ideal activation recovery. Qseparate. Norm/operator inequality exact-real; numeric intervals/contraction envelope conditional, cuBLASnotattested. No actual H/FDown/stateupdate/consumer/image or economical arbitrarypromptzero-neuralproducer; originalmaps/source/historypaid.'})
    print('GATED_EVEN_BOUND_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('root');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        value=collect(a.root)
    else:value=globals()[a.action](a.root)
    print(json.dumps(value))
