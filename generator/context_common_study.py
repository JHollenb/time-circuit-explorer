"""One oracle finite-bank SVD, followed by actual signed prefix-statistic reads."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from encoder_atoms import np,read,save,sha,identity,guard,floats
from context_common_bank import bank,probability,restrict
from context_statistics_study import summary,rms,rowrms

HERE=Path(__file__).resolve().parent


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,old,prior=checked_collection(parent)
    job=Api().json('GET','/api/jobs/'+c['jobID'])
    if job['state']!='succeeded' or job['payload_custody']!=c['custody'] or job['config']['requestSHA256']!=prior['requestSHA256']:raise ValueError('Parent differs')
    seal=read(parent/'result/prediction-seal.json');ref=read(parent/'result/reference-receipt.json')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir()
    save(p/'source.json',{'ports':seal['ports'],'references':ref['rows'],'projection':ref['projection'],
        'historicalReport':{'path':job['result']['diagnostics']['cwd']+'/response-result/report.json','sha256':sha(parent/'result/report.json')},
        'pairs':prior['pairs'],'parentJobID':job['job_id'],'parentSealSHA256':sha(parent/'result/prediction-seal.json')})
    for n in ['context_common_study.py','context_common_bank.py','context_statistics_study.py','encoder_atoms.py']:shutil.copyfile(HERE/n,p/n)
    request={'schema':'tce-common-bank-context-request-v1','parentJobID':job['job_id'],'parentRequestSHA256':prior['requestSHA256'],
        'scope':'Target-assisted transductive common-bank signed-rank capacity on all eight saved prompts. No unseen-query rule, held-out serving or image claim.',
        'bankPolicy':'Deduplicate queries and active keys by(tokenID,absolutePosition), checking duplicateBF16Q andKV equality. Retain actual occurrence lists, values, masks and padding queries.',
        'kernelPolicy':'One FP64 unmasked exp(QK/sqrt128), globally row normalized. One actual thin SVD, unique bank rows equally weighted. A=U sqrt(S), B=V sqrt(S); no clipping.',
        'ranks':[16,32,64,128],'repeats':5,'primaryObservable':'Four paired-change errors/cosines/raw scales and output-slice transport, split active/padding. Endpoints and matrix norm are secondary.',
        'signedReadPolicy':'No epsilon/fallback/clipping. Denominator0/nonfinite reads remain invalid per row; preserve raw heads, masks, signed mass and cancellation. Other rows/ranks continue.',
        'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'sourceHashes':{f.name:sha(f) for f in p.iterdir()}}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from response_study import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/'payload'/n)!=h:raise ValueError('Staged source changed')
    cmd=[v.replace('response_study.py','context_common_study.py') for v in command()]
    job=launch(cmd,experiment='tce-common-bank-context-rank-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','ranks':r['ranks']},timeout_s=600,
        detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 saved-only8ports common lexical bank: one stable unmasked kernel/thinSVD, ranks16/32/64/128. Fresh signed prefix state seals before archived head/W_o scoring. No clip/checkpoint/framework/GPU; transductive capacity only. Finite600s,no retries, host-local inputs.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def load(ref):
    if sha(ref['path'])!=ref['sha256']:raise ValueError('Saved source changed')
    with np.load(ref['path'],allow_pickle=False) as f:return {k:f[k] for k in f.files}


def nullable(value):return float(value) if np.isfinite(value) else None


def score(pred,target,mask,matrix,valid):
    finite=valid&np.isfinite(pred).all(1)&np.isfinite(target).all(1);selected=mask&finite
    result={'queries':int(mask.sum()),'validQueries':int(selected.sum()),'invalidQueries':int(np.count_nonzero(mask&~finite)),
        'coverage':'all' if np.all(finite[mask]) else 'finite-subset-only','head':None,'projectedRelativeRms':None}
    if selected.any():
        result['head']=summary(pred[selected],target[selected]);a=pred[selected]@matrix.T;b=target[selected]@matrix.T
        result['projection']=summary(a,b);result['projectedRelativeRms']=result['projection']['relativeRms']
    return result


def execute(root):
    guard();root=Path(root).resolve();r=read(root/'request.json');source=read(root/'source.json')
    if identity({k:v for k,v in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request changed')
    for n,h in r['sourceHashes'].items():
        if sha(root/n)!=h:raise ValueError('Staged source changed')
    start=time.perf_counter();out=root/'response-result';out.mkdir(exist_ok=False);work=root/'reference-work';work.mkdir()
    packets={p['id']:load(p) for p in source['ports']};atoms,maps,duplicates=bank(packets);t=time.perf_counter();P,kernel=probability(atoms['q'],atoms['k']);kernel_s=time.perf_counter()-t
    np.savez(work/'common-bank.npz',**atoms,P=P)
    if min(P.shape)<max(r['ranks']):raise ValueError('Prespecified ranks exceed common bank')
    t=time.perf_counter();U,S,Vt=np.linalg.svd(P,full_matrices=False);svd_s=time.perf_counter()-t
    maxrank=max(r['ranks']);A=U[:,:maxrank]*np.sqrt(S[:maxrank]);B=Vt[:maxrank].T*np.sqrt(S[:maxrank])
    full_reconstruction=(U*S)@Vt;full_error=float(np.linalg.norm(full_reconstruction-P)/np.linalg.norm(P));del full_reconstruction
    np.savez(root/'factors.npz',A=A,B=B,singularValues=S)
    np.savez(root/'packets.npz',**{name+'__'+key:z for name,p in maps.items() for key,z in p.items()})
    spec={'ids':list(maps),'ranks':r['ranks'],'repeats':r['repeats'],
        'factors':{'path':str(root/'factors.npz'),'sha256':sha(root/'factors.npz')},
        'packets':{'path':str(root/'packets.npz'),'sha256':sha(root/'packets.npz')}}
    save(root/'serve.json',spec);t=time.perf_counter()
    subprocess.run([sys.executable,str(root/'context_common_bank.py'),str(root/'serve.json'),str(root/'predictions')],check=True)
    child_s=time.perf_counter()-t;seal_path=root/'predictions/prediction-seal.json';seal_sha=sha(seal_path);seal=read(seal_path)
    print('COMMON_BANK_FORECASTS_SEALED',seal_sha,flush=True)
    references={p['id']:load(p) for p in source['references']}
    projection=source['projection']
    if sha(projection['path'])!=projection['sha256']:raise ValueError('Saved W_o slice changed')
    matrix=floats(np.load(projection['path'],allow_pickle=False)).astype(np.float64)
    exacts={};exact_rows=[]
    for name,p in maps.items():
        h,rows=restrict(P,p);exacts[name]=(h,rows);target=references[name]['raw'];finite=np.isfinite(h).all(1)
        exact_rows.append({'id':name,'queries':len(h),'invalidRestrictedReads':int(np.count_nonzero(~finite)),
            'referenceMaxAbs':float(np.max(np.abs(h[finite]-target[finite]))) if finite.any() else None,
            'minExactPrefixMass':min(z['denominator'] for z in rows),'maxExactPrefixMass':max(z['denominator'] for z in rows)})
    scores=[];closure=[];kernel_errors=[];predictions={};signed=[];row_count=0
    with (out/'observations.jsonl').open('x') as f:
        for rank in r['ranks']:
            Pr=A[:,:rank]@B[:,:rank].T
            kernel_errors.append({'rank':rank,'relativeFrobenius':float(np.linalg.norm(Pr-P)/np.linalg.norm(P)),
                'singularTailRelativeFrobenius':float(np.sqrt(np.sum(S[rank:]**2)/np.sum(S**2))),
                'negativeMatrixEntries':int(np.count_nonzero(Pr<0)),'negativeMatrixMass':float(-Pr[Pr<0].sum())})
            for name,p in maps.items():
                record=next(z for z in seal['rows'] if z['id']==name and z['rank']==rank);pred=load(record);predictions[name,rank]=pred
                direct,diagnostics=restrict(Pr,p);target=references[name]['raw'];valid=pred['valid'];direct_valid=np.isfinite(direct).all(1);common=valid&direct_valid
                closure.append({'id':name,'rank':rank,'jointValidRows':int(common.sum()),'validMaskMismatches':int(np.count_nonzero(valid!=direct_valid)),
                    'maxAbs':float(np.max(np.abs(pred['raw'][common]-direct[common]))) if common.any() else None,
                    'summary':summary(pred['raw'][common],direct[common]) if common.any() else None})
                ma=p['mask'].astype(bool)
                for region,mask in [('all',np.ones(len(ma),dtype=bool)),('active',ma),('padding',~ma)]:
                    if mask.any():
                        z={'id':name,'rank':rank,'region':region,**score(pred['raw'],target,mask,matrix,valid)}
                        z['directMatrix']=score(direct,target,mask,matrix,direct_valid)
                        z['rounded']=score(floats(pred['rounded']),floats(references[name]['rounded']),mask,matrix,pred['rounded_valid']);scores.append(z)
                signed.append({'id':name,'rank':rank,'negativePrefixRows':sum(z['negativeEntries']>0 for z in diagnostics),
                    'nonpositiveDenominators':sum(z['nonpositiveDenominator'] for z in diagnostics),
                    'minimumCancellationRatio':min(z['cancellationRatio'] for z in diagnostics),'maxNegativeProbabilityMass':max((z['negativeProbabilityMass'] for z in diagnostics if z['negativeProbabilityMass'] is not None),default=None)})
                error=pred['raw']-target;er=rowrms(error);tr=rowrms(target)
                for i,z in enumerate(diagnostics):
                    f.write(json.dumps({'id':name,'rank':rank,'position':i,'active':bool(ma[i]),'streamValid':bool(valid[i]),
                        'headErrorRms':nullable(er[i]),'headReferenceRms':nullable(tr[i]),'exactPrefixMass':exacts[name][1][i]['denominator'],
                        'streamDenominator':nullable(pred['denominator'][i]),**z},allow_nan=False)+'\n');row_count+=1
    pairs=[];pair_rows=0
    with (out/'paired-observations.jsonl').open('x') as f:
        for a,b in source['pairs']:
            target=references[b]['raw']-references[a]['raw'];ma,mb=[maps[n]['mask'].astype(bool) for n in [a,b]]
            for rank in r['ranks']:
                x,y=[predictions[n,rank] for n in [a,b]];pred=y['raw']-x['raw'];valid=x['valid']&y['valid']
                for region,mask in [('all',np.ones(len(ma),dtype=bool)),('active_both',ma&mb),('padding_both',~ma&~mb),('mask_changed',ma!=mb)]:
                    if mask.any():pairs.append({'a':a,'b':b,'rank':rank,'region':region,**score(pred,target,mask,matrix,valid)})
                error=rowrms(pred-target);scale=rowrms(target)
                for i in range(len(ma)):
                    f.write(json.dumps({'a':a,'b':b,'rank':rank,'position':i,'activeA':bool(ma[i]),'activeB':bool(mb[i]),'valid':bool(valid[i]),
                        'headErrorRms':nullable(error[i]),'targetChangeRms':nullable(scale[i])},allow_nan=False)+'\n');pair_rows+=1
    if sha(seal_path)!=seal_sha:raise ValueError('Forecast seal changed during scoring')
    hist=source['historicalReport']
    if sha(hist['path'])!=hist['sha256']:raise ValueError('Historical control report changed')
    old=read(hist['path'])
    save(out/'historical-controls.json',{'source':hist,'scope':'Unmodified score rows from the previously sealed source-frozen assay, not new candidate executions.',
        'scores':old['scores'],'pairedScores':old['pairedScores']})
    save(out/'compiler.json',{'scope':r['scope'],'queryBankRows':len(atoms['q']),'keyBankRows':len(atoms['k']),
        'queryOccurrences':sum(len(z['qmap']) for z in maps.values()),'activeKeyOccurrences':sum(int(z['mask'].sum()) for z in maps.values()),
        'duplicatesVerified':duplicates,'unmaskedKernelBytes':P.nbytes,'kernel':kernel,'kernelSeconds':kernel_s,'svdSeconds':svd_s,
        'fullSVDReconstructionRelativeFrobenius':full_error,'singularValues':S.tolist(),'rankDiagnostics':kernel_errors,
        'commonBank':{'path':str(work/'common-bank.npz'),'sha256':sha(work/'common-bank.npz')},
        'learnedFiniteBankFactors':True,'newFrameworkCalls':0,'originalCheckpointBytesRead':0})
    save(out/'reference-receipt.json',{'rows':source['references'],'projection':projection,'exactRestrictions':exact_rows,'signedDiagnostics':signed,
        'directMatrixStreamingClosures':closure,'scope':'Saved native-qualified mathematical head targets; no native attention or image. Exact common-bank restriction is an instrument check; truncated signed-kernel capacity is transductive.'})
    for name,path in [('request.json',root/'request.json'),('prediction-seal.json',seal_path),('serve.json',root/'serve.json'),
        ('factors.npz',root/'factors.npz'),('packets.npz',root/'packets.npz')]:shutil.copyfile(path,out/name)
    save(out/'report.json',{'schema':'tce-common-bank-context-result-v1','requestSHA256':r['requestSHA256'],'scope':r['scope'],
        'pairedScores':pairs,'scores':scores,'compiler':read(out/'compiler.json'),'reference':read(out/'reference-receipt.json'),
        'serving':seal,'predictionSealSHA256':seal_sha,'freshServingChildSeconds':child_s,'workerSeconds':time.perf_counter()-start,
        'observations':row_count,'pairedObservations':pair_rows,'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
        'goalStatus':'Active. Signed oracle finite-bank capacity only; no unseen-query feature rule, full encoder or zero-neural images.'})
    print('COMMON_BANK_CONTEXT_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
