"""One source-fixed retrieval-cost assay; consume no original model parameters."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from encoder_atoms import np,read,save,sha,identity,NoNeural
from attention_bank import geometry
from attention_bank_study import loaded,verify_request,regions

HERE=Path(__file__).resolve().parent


def stage(root,parent):
    from publish_context_statistics import checked_collection
    from mrun.client.api import Api
    root=Path(root).resolve();parent=Path(parent).resolve();c,j,r=checked_collection(parent,current_workers=False);api=Api()
    live=api.json('GET','/api/jobs/'+c['jobID']);ancestor=api.json('GET','/api/jobs/'+r['completeBankJobID'])
    if live['state']!='succeeded' or live['payload_custody']!=c['custody'] or live['config']['requestSHA256']!=r['requestSHA256']:raise ValueError('Selector parent differs')
    if ancestor['state']!='succeeded' or ancestor['config']['requestSHA256']!=r['completeBankRequestSHA256']:raise ValueError('Source geometry ancestor differs')
    root.mkdir(parents=True,exist_ok=False);p=root/'payload';p.mkdir();old=read(parent/'result/source.json');seal=read(parent/'result/prediction-seal.json');ref=read(parent/'result/reference-receipt.json');base=j['result']['diagnostics']['cwd']
    source={k:old[k] for k in ['asset','ports','geometry','pairs','completeBankRoot']}
    source.update(parentSeal={'path':base+'/predictions/prediction-seal.json','sha256':sha(parent/'result/prediction-seal.json')},
        parentBanks=[x for x in seal['rows'] if x['method']=='current'],parentConsumers=[x for x in ref['rows'] if x['kind']=='consumers'],
        parentReport={'path':base+'/response-result/report.json','sha256':sha(parent/'result/report.json')})
    save(p/'source.json',source);save(p/'serve-source.json',{k:source[k] for k in ['asset','ports','geometry']}|{'budget':4,'block':16})
    files=['certified_retrieval.py','certified_retrieval_study.py','selector_objective.py','centroid_tilt.py','centroid_tilt_study.py','sparse_tilt.py','sparse_tilt_study.py',
        'attention_bank.py','attention_bank_study.py','encoder_atoms.py','context_source_geometry.py','context_landmarks.py','context_transport.py',
        'context_statistics.py','context_statistics_study.py','context_transport_study.py','transport_consumer.py']
    for name in files:shutil.copyfile(HERE/name,p/name)
    request={'schema':'tce-certified-retrieval-request-v1','parentJobID':c['jobID'],'parentRequestSHA256':r['requestSHA256'],
        'completeBankJobID':r['completeBankJobID'],'completeBankRequestSHA256':r['completeBankRequestSHA256'],'pairs':source['pairs'],'budget':4,'block':16,
        'scope':'Retrospective suppliedports, all32heads/512queries. Causal actualkey/value index, source-fixed scorebounds, unchanged centroid read. Selection-cost assay, not compact encoder or zero-neural image.',
        'policy':'8blocks16/sourcecovariance diagonal order/coordinateindex ties. Initial4full candidate dots, then progressive block dots. At completion f is the accumulated full progressive dot, no duplicate hidden dot. eta=gamma4d bounds both completed-progressive and old GEMV errors. Outward tailnorms, strict certified prune, overlapping finalmembership/order or unsupportedrange falls back to original wholeprefix GEMV.',
        'finiteContract':'IEEEbinary64 normal/finite roundtonearest at most4d rounded dotoperations, d128. Nonzeroinputmagnitude in[2^-400,2^400]; otherwise fullfallback. Positive outward square/sum/sqrt/multiply; reverse tails avoid subtraction. Zero interval endpoints may be outward subnormal.',
        'sealPolicy':'Certified child complete bank/support/index/counters seal first; independent fresh fullscan child afterward. Historical forecasts/report/consumer hashes read only after certified seal. Consumer evidence inherited only for identical complete BF16 input banks; no new originalweight access/consumer evaluation.',
        'observables':'Complete support/order/raw/BF16 comparisons. Score products plus norms/bounds/sorts/fallbacks/indexbytes and phase wall/CPU time. Perhead/activepadding/at8after8. Fullprefix centroid read charged separately. One frozen protocol, retain poorpruning/slowresult.',
        'sourceHashes':{f.name:sha(f) for f in p.iterdir()},'device':'cpu','backend':'numpy-fp64','cpuThreads':1,'timeoutSeconds':600}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(p/'request.json',request)
    save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[f.name for f in p.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    verify_request(root/'payload',r);cmd=command()[:-5]+['python','-u','certified_retrieval_study.py','execute','.']
    cmd[cmd.index('uv'):cmd.index('uv')]=['OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1']
    job=launch(cmd,experiment='tce-certified-retrieval-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
        config={'schema':r['schema'],'requestSHA256':r['requestSHA256'],'backend':'numpy','device':'cpu','seq_len':512,'heads':32,'budget':4,'block':16},
        timeout_s=r['timeoutSeconds'],detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='CPU1 certifiedtop4 search,2savedportsx32headsx512queries; sourcecov coordinateorder/8blocks16, outward bounds and oldwholeprefix fallback. Seal completecandidate first, independentfullscan comparison second. Charge full actualKVindex/norm/bound/sort/read costs; identicalbank allows inheritedconsumer, no newweight/model evaluation. Host-local immutable parents,defaultadmission,expected under5min,finite600s,no retry.')
    value={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',value);return value


def products(counts):
    shape=counts['queries'].shape
    return sum((counts.get(k,np.zeros(shape,dtype=np.int64)) for k in ['initialCandidateFullDotProducts','progressiveScoreProducts','wholePrefixScoreProducts']))


def execute(root):
    if NoNeural.roots.intersection(sys.modules):raise RuntimeError('Neural framework already imported')
    sys.meta_path.insert(0,NoNeural());root=Path(root).resolve();r=read(root/'request.json');verify_request(root,r);start=time.perf_counter();out=root/'response-result';out.mkdir(exist_ok=False);source=read(root/'source.json')
    seals={};child_times={}
    for mode in ['certified','full']:
        tick=time.perf_counter();subprocess.run([sys.executable,str(root/'certified_retrieval.py'),mode,str(root/'serve-source.json'),str(root/mode)],check=True);child_times[mode]=time.perf_counter()-tick
        path=root/mode/'prediction-seal.json';seals[mode]=read(path);print('RETRIEVAL_BANK_SEALED',mode,sha(path),flush=True)
        shutil.copyfile(path,out/('prediction-seal.json' if mode=='certified' else 'fullscan-seal.json'))
    # No old forecast or consumer tensor becomes an operand of either child.
    if sha(source['parentSeal']['path'])!=source['parentSeal']['sha256'] or sha(source['parentReport']['path'])!=source['parentReport']['sha256']:raise ValueError('Historical parent changed')
    parent_report=read(source['parentReport']['path']);d,heads,_=geometry(source['geometry']);checks=[];strata_rows=[];observations=0;can_inherit=True
    with (out/'observations.jsonl').open('x') as stream:
        for item in source['ports']:
            name=item['id'];port=loaded(item);old=loaded(next(x for x in source['parentBanks'] if x['id']==name));banks={mode:loaded(next(x for x in seals[mode]['rows'] if x['id']==name)) for mode in seals}
            check={'id':name,'comparisons':{}}
            for label,a,b in [('certified-vs-full',banks['certified'],banks['full']),('full-vs-parent',banks['full'],old),('certified-vs-parent',banks['certified'],old)]:
                check['comparisons'][label]={key:int(np.count_nonzero(a[key]!=b[key])) for key in ['rounded','supportPositions','nearestIndex']}
                check['comparisons'][label]['rawBitMismatches']=int(np.count_nonzero(a['raw'].view(np.uint64)!=b['raw'].view(np.uint64)))
                check['comparisons'][label]['rawMaxAbs']=float(np.max(np.abs(a['raw']-b['raw'])))
            identical=check['comparisons']['certified-vs-parent']['rounded']==0;can_inherit=can_inherit and identical;check['consumerInputBankIdentical']=identical;checks.append(check)
            clock=np.arange(len(port['mask']));masks=regions(port['mask'])+[('at8',clock==8),('after8',clock>8)]
            for mode in seals:
                counts=loaded(next(x for x in seals[mode]['states'] if x['id']==name and x['kind']=='query-counters'));score_products=products(counts)
                zero=np.zeros_like(counts['queries']);certified_counts=counts.get('certifiedQueries',zero);pruned_counts=counts.get('prunedKeys',zero)
                for region,selected in masks:
                    if not selected.any():continue
                    for head in [None,*range(heads)]:
                        use=counts['queries'][selected] if head is None else counts['queries'][selected,head]
                        fields={k:int((a[selected] if head is None else a[selected,head]).sum()) for k,a in counts.items() if k!='fallbackReason'}
                        actual=int((score_products[selected] if head is None else score_products[selected,head]).sum());baseline=fields['fullScanScoreProducts']
                        strata_rows.append({'id':name,'mode':mode,'region':region,'queryHead':head,'headQueries':int(use.sum()),'scoreProducts':actual,'scoreProductRatio':actual/baseline if baseline else None,'counts':fields})
                for i in range(len(port['mask'])):
                    for h in range(heads):
                        stream.write(json.dumps({'id':name,'mode':mode,'position':i,'queryHead':h,'active':bool(port['mask'][i]),'scoreProducts':int(score_products[i,h]),
                            'fullScanScoreProducts':int(counts['fullScanScoreProducts'][i,h]),'certified':int(certified_counts[i,h]),
                            'prunedKeys':int(pruned_counts[i,h]),'fallbackReason':int(counts['fallbackReason'][i,h])},allow_nan=False)+'\n');observations+=1
    support_mismatch=sum(c['comparisons']['certified-vs-full']['supportPositions'] for c in checks)
    parent_support_mismatch=sum(c['comparisons']['certified-vs-parent']['supportPositions'] for c in checks)
    for name in ['request.json','source.json','serve-source.json']:shutil.copyfile(root/name,out/name)
    for mode in seals:
        if sha(root/mode/'prediction-seal.json')!=sha(out/('prediction-seal.json' if mode=='certified' else 'fullscan-seal.json')):raise ValueError('Sealed forecast changed')
    # Bind inherited arrays without evaluating the original learned consumer again.
    for item in source['parentConsumers']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Inherited consumer changed')
    save(out/'report.json',{'schema':'tce-certified-retrieval-result-v1','requestSHA256':r['requestSHA256'],'predictionSealSHA256':sha(out/'prediction-seal.json'),
        'fullscanSealSHA256':sha(out/'fullscan-seal.json'),'scope':r['scope'],'seals':seals,'childSeconds':child_times,'checks':checks,'strata':strata_rows,
        'inheritedConsumerEvidence':{'inputBanksIdentical':can_inherit,'parentReport':source['parentReport'],'consumerArrays':source['parentConsumers'],
            'scores':[x for x in parent_report['scores'] if x['method']=='current'] if can_inherit else [],
            'pairedScores':[x for x in parent_report['pairedScores'] if x['method']=='current'] if can_inherit else [],
            'scope':'Historical original learned consumer, reused only for bit-identical complete BF16 input bank; not new consumer/model evaluation.'},
        'observations':observations,'workerSeconds':time.perf_counter()-start,'frameworkCalls':0,'originalDenseProjections':0,'newOriginalConsumerEvaluations':0,
        'currentProfileSupportAndOrderIdentical':support_mismatch==0,'oldSelectorSupportAndOrderIdentical':parent_support_mismatch==0,
        'mechanics_status':'complete' if support_mismatch==0 else 'certificate-support-mismatch','instrument_status':'executed','terminal_status':'unassessed',
        'goalStatus':'Active. One retrieval-cost assay within model-derived attention; no completeencoder/ownstateimage/strictzero-neural claim.'})
    print('CERTIFIED_RETRIEVAL_STUDY_DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');p.add_argument('--parent');a=p.parse_args()
    if a.action=='stage':value=stage(a.run,a.parent)
    elif a.action=='collect':
        from response_study import collect
        value=collect(a.run)
    else:value=globals()[a.action](a.run)
    if value is not None:print(json.dumps(value))
