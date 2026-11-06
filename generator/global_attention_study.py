"""Finite compiled global-consumer experiment, with post-seal saved-QKV references."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from response_study import read,save,sha,identity,command,collect

HERE=Path(__file__).resolve().parent
SCOPE=('Actual32-row BF16 producer state -> reduced QKV -> one complete joint2 head0 read. '
       'Frozen supplied remainder:512text+992image rows;32changed queries read all1536keys. '
       'FP64 mathematical attention consumer with explicit BF16 boundaries; not native Flash parity, '
       'raw-prompt generation, output projection, denoising or RGB.')


def stage(root,parent,context):
    from mrun.client.api import Api
    root,parent,context=[Path(p).resolve() for p in [root,parent,context]]
    prior,custody=read(parent/'request.json'),read(parent/'collection.json')
    job=Api().json('GET','/api/jobs/'+custody['jobID'])
    if job['state']!='succeeded' or job['config']['requestSHA256']!=prior['requestSHA256'] or job['payload_custody']!=custody['custody']:
        raise ValueError('Parent receiver not authenticated')
    inventory=custody['inventory']['files']
    for name,item in inventory.items():
        path=parent/'result'/name
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('Parent result changed: '+name)
    if identity({k:v for k,v in prior.items() if k!='requestSHA256'})!=prior['requestSHA256']:
        raise ValueError('Parent request changed')
    remote=job['result']['diagnostics']['cwd']+'/response-result'
    def pref(name):return {'path':remote+'/'+name,'sha256':inventory[name]['sha256']}
    source=prior['producer']['sources'][1]
    if sha(context/'files.json')!=source['inventorySHA256']:
        raise ValueError('Context inventory does not match authenticated chain')
    files=read(context/'files.json')
    def cref(name):
        if sha(context/name)!=files[name]:raise ValueError('Context metadata changed: '+name)
        return {'path':source['remoteRoot']+'/'+name,'sha256':files[name]}
    cutname='store/cuts/'+Path(prior['boundary']['cut']['path']).name
    if cref(cutname)['sha256']!=prior['boundary']['cut']['sha256']:
        raise ValueError('Cut binding differs')
    cut=read(context/cutname)
    if cut['execution_point']['next_operation']!='joint_block.2':raise ValueError('Wrong next operation')
    def page(slot):
        name='store/pages/'+cut['slots'][slot]+'.json';cref(name);return read(context/name)
    def tensor_ref(p,encoded,desc):
        return {**cref('store/blobs/'+p['tensor_blob']+'.safetensors'),'tensor':encoded['name'],
                'shape':desc['shape'],'dtype':{'torch.bfloat16':'BF16','torch.float32':'F32'}[desc['dtype']],
                'tensorSHA256':desc['sha256']}
    operands={}
    for slot in ['image','text']:
        p=page(slot);operands[slot]=tensor_ref(p,p['encoded'],p['value'])
    p=page('context');desc={i['key']['value']:i['value'] for i in p['value']['items']}
    for name,field in [('imageMod','img_mod'),('textMod','txt_mod')]:
        operands[name]=tensor_ref(p,p['encoded']['items'][field],desc[field])
    for i,name in enumerate(['cos','sin']):
        operands[name]=tensor_ref(p,p['encoded']['items']['rope']['items'][i],desc['rope']['items'][i])
    kernel=read(HERE/'kernel-manifest.json')
    for name in ['parameter-artifacts.json','retrieve_weights.py','bindings.json','kernels/flux_reentry_reference.py']:
        if sha(HERE/name)!=kernel['files'][name]:raise ValueError('Pinned operator/recipe source changed')
    parameters=read(HERE/'parameter-artifacts.json')
    suffixes=['to_q.weight','to_k.weight','to_v.weight','add_q_proj.weight','add_k_proj.weight','add_v_proj.weight',
              'norm_q.weight','norm_k.weight','norm_added_q.weight','norm_added_k.weight']
    parameters['artifacts']=[a for a in parameters['artifacts'] if a['source']['tensor'] in ['transformer_blocks.2.attn.'+s for s in suffixes]]
    if len(parameters['artifacts'])!=10:raise ValueError('Fixed-bank parameter inventory differs')
    original=read(parent/'payload/compile.json')
    compile_spec={'schema':'tce-global-head-compile-spec-v1','operands':operands,'parameters':parameters,
                  'basis':pref('basis.npy'),'banks':{n:pref('bank-'+n+'.npz') for n in ['q','k','v']},
                  'checkpointRoot':original['checkpointRoot'],'checkpointFile':original['checkpointFile'],
                  'sourceRoots':[*original['sourceRoots'],remote], 'selectedRows':prior['selectedRows'],
                  'head':0,'headDim':128,'eps':1e-6,'scope':SCOPE,
                  'numericSource':{'path':'kernels/flux_reentry_reference.py','sha256':kernel['files']['kernels/flux_reentry_reference.py']}}
    parent_spec=read(parent/'result/serve.json');parent_report=read(parent/'result/report.json')
    packets=[];evaluation=[]
    for item in read(parent/'result/packet-map.json'):
        if item['alpha'] not in [0,.5]:continue
        packet=next(p for p in parent_spec['packets'] if p['id']==item['id'])
        packets.append({**pref(Path(packet['path']).name),'id':packet['id'],'shape':packet['shape']})
        native={r['id']:r for r in parent_report['native']['records']}
        predicted=next(r for r in parent_report['serving']['rows'] if r['id']==packet['id'])
        evaluation.append({'id':packet['id'],'alpha':item['alpha'],
                           'sameInput':pref(native[item['id']]['path']), 'nativeWrite':pref(native[item['nativeID']]['path']),
                           'priorPrediction':pref(predicted['output'])})
    if [x['alpha'] for x in evaluation]!=[0,.5]:raise ValueError('Packet order differs')
    root.mkdir(parents=True,exist_ok=False);payload=root/'payload';payload.mkdir()
    save(payload/'compile.json',compile_spec)
    save(payload/'packet-source.json',{'packets':packets})
    save(payload/'evaluation.json',{'rows':evaluation,'scope':'Saved native CPU QKV from original receiver assay, propagated through shared NumPy headnorm/RoPE/ideal attention. Not native attention output or CUDA parity.'})
    for name in ['global_attention.py','global_attention_study.py','state_receiver.py','mlp_polynomial.py','response_chart.py','prompt_closure.py','response_study.py','retrieve_weights.py','state_receiver_study.py']:
        shutil.copyfile(HERE/name,payload/name)
    request={'schema':'tce-global-head-request-v1','scope':SCOPE,'head':0,'headDim':128,'timingRepeats':31,
             'parent':{'jobID':job['job_id'],'requestSHA256':prior['requestSHA256'],'collectionSHA256':sha(parent/'collection.json'),
                       'payloadCustody':job['payload_custody'],'contextSource':source,'boundary':prior['boundary']},
             'selectedRows':prior['selectedRows'],'newNativeFrameworkCalls':0,'offlineOriginalDensePreparation':True,
             'sourceHashes':{p.name:sha(p) for p in payload.iterdir()}}
    request['requestSHA256']=identity(request);save(root/'request.json',request);save(payload/'request.json',request)
    save(payload/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'run':str(root),'requestSHA256':request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    root=Path(root).resolve();request=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    cmd=[v.replace('response_study.py','global_attention_study.py') for v in command()]
    job=launch(cmd,experiment='tce-compiled-global-head-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cpu':True},
               config={'schema':request['schema'],'requestSHA256':request['requestSHA256'],'backend':'numpy','device':'cpu',
                       'newNeuralFrameworkCalls':0,'offlineOriginalDenseHeadProjections':6,'head':0,'changedRows':32,'allQueries':1536},
               timeout_s=300,detach=True,retry_on_kill=False,preflight=True,guarded=True,
               note='Actual32row states drive globaljoint2head0:1536queries/keys. CPU1 compile fixed992image+512text QKV, zero selected-row refresh;3reducedmaps online.2packetsx31timings+6controls; seal then saved-QKV/full-attention references. No nativeNN/GPU. Finite300s,no retries.')
    result={'jobID':str(job),'requestSHA256':request['requestSHA256']};save(root/'job.json',result);return result


def execute(root):
    from global_attention import np,load_bundle,prepare,merge,full_attention,quantize
    from state_receiver_study import decompose
    from mlp_polynomial import rms,ratio
    root=Path(root).resolve();request=read(root/'request.json')
    if identity({k:v for k,v in request.items() if k!='requestSHA256'})!=request['requestSHA256']:raise ValueError('Request changed')
    for name,digest in request['sourceHashes'].items():
        if sha(root/name)!=digest:raise ValueError('Staged source changed')
    output=root/'response-result';output.mkdir(exist_ok=False);started=time.perf_counter()
    compile0=time.perf_counter()
    subprocess.run([sys.executable,str(root/'global_attention.py'),'compile',str(root/'compile.json'),str(output)],check=True)
    compile_s=time.perf_counter()-compile0;print('GLOBAL_HEAD_COMPILED',flush=True)
    compiled=read(output/'compile-receipt.json');original=read(root/'compile.json')
    serving={name:{'path':str(output/file),'sha256':compiled['bundles'][file]['sha256']} for name,file in [('reader','reader-head.npz'),('fixed','fixed-head.npz'),('cache','attention-cache.npz')]}
    serving.update(packets=read(root/'packet-source.json')['packets'],eps=original['eps'],repeats=request['timingRepeats'],
                   scope=SCOPE,sourceRoots=[*original['sourceRoots'],str(root)])
    save(output/'serve.json',serving);serve0=time.perf_counter()
    subprocess.run([sys.executable,str(root/'global_attention.py'),'serve',str(output/'serve.json'),str(output)],check=True)
    serve_s=time.perf_counter()-serve0;seal_hash=sha(output/'prediction-seal.json');seal=read(output/'prediction-seal.json')
    print('GLOBAL_HEAD_SEALED '+seal_hash,flush=True)
    # Native-derived QKV arrays and prior predictions first open AFTER the complete forecast seal.
    evaluation=read(root/'evaluation.json')
    reader,fixed,cache=[load_bundle(serving[n]) for n in ['reader','fixed','cache']]
    def load(ref):
        if sha(ref['path'])!=ref['sha256']:raise ValueError('Sealed or native-derived array changed')
        with np.load(ref['path'],allow_pickle=False) as data:return {k:data[k] for k in data.files}
    forecasts={}
    for row in seal['rows']:
        forecasts[row['method'],row['id']]=load({'path':str(output/row['path']),'sha256':row['sha256']})
    references={};closures=[];reference_records=[];same_profile=[];ref0=time.perf_counter()
    for item in evaluation['rows']:
        predicted=forecasts['all',item['id']]
        own_fields={n:predicted['changed_'+n] for n in ['q','k','v']}
        direct=full_attention(merge(fixed,cache['changed'],own_fields))
        closures.append({'id':item['id'],'rawMaxAbs':float(np.max(np.abs(direct-predicted['raw']))),
                         'rawRms':rms(direct-predicted['raw']),
                         'roundedChangedScalars':int(np.count_nonzero(quantize(direct)!=predicted['rounded']))})
        for arm in ['sameInput','nativeWrite']:
            fields=load(item[arm]);raw={n:fields[n][0,:,:128] for n in ['q','k','v']}
            prepared=prepare(raw,reader,original['eps'])
            head=full_attention(merge(fixed,cache['changed'],prepared))
            path=output/(arm+'-'+item['id']+'.npz');np.savez(path,raw=head,rounded=quantize(head))
            references[arm,item['id']]={'raw':head,'rounded':quantize(head)}
            reference_records.append({'arm':arm,'id':item['id'],'path':path.name,'sha256':sha(path)})
        prior=load(item['priorPrediction'])
        prior_fields=prepare({n:prior[n][0,:,:128] for n in ['q','k','v']},reader,original['eps'])
        same_profile.append({'id':item['id'],**{n+'MaxAbs':float(np.max(np.abs(own_fields[n]-prior_fields[n]))) for n in ['q','k','v']}})
    ref_s=time.perf_counter()-ref0
    ids=[item['id'] for item in evaluation['rows']];baseline=ids[0]
    n=compiled['textRows']+compiled['imageRows'];changed=cache['changed']
    regions={'all':np.arange(n),'text':np.arange(compiled['textRows']), 'changedImage':changed,
             'fixedImage':np.setdiff1d(np.arange(compiled['textRows'],n),changed)}
    aggregate,rows,controls,interactions=[],[],[],[]
    for item in evaluation['rows']:
        pid=item['id'];p,c,t=forecasts['all',pid],references['sameInput',pid],references['nativeWrite',pid]
        for view in ['raw','rounded']:
            effect=t[view].astype(np.float64)-references['nativeWrite',baseline][view]
            own=p[view].astype(np.float64)-forecasts['all',baseline][view]
            consumer=c[view].astype(np.float64)-references['sameInput',baseline][view]
            for comparison in ['absolute','paired']:
                values=(p[view].astype(np.float64),c[view].astype(np.float64),t[view].astype(np.float64)) if comparison=='absolute' else (own,consumer,effect)
                for name,index in regions.items():
                    aggregate.append({'alpha':item['alpha'],'view':view,'comparison':comparison,'region':name,
                                      **decompose(*(v[index] for v in values),effect[index])})
            if item['alpha']!=0:
                for i in range(n):
                    rows.append({'query':i,'stream':'text' if i<compiled['textRows'] else 'image','selectedQuery':bool(i in changed),
                                 'view':view,'alpha':item['alpha'],**decompose(own[i],consumer[i],effect[i],effect[i])})
    pid=ids[1]
    for view in ['raw','rounded']:
        base=forecasts['all',baseline][view].astype(np.float64)
        effect=references['nativeWrite',pid][view].astype(np.float64)-references['nativeWrite',baseline][view]
        primary=forecasts['all',pid][view].astype(np.float64)-base
        for row in seal['rows']:
            if row['method']=='all':continue
            delta=forecasts[row['method'],pid][view].astype(np.float64)-base
            for name,index in regions.items():
                a,b=delta[index],effect[index]
                controls.append({'method':row['method'],'view':view,'region':name,'deltaRms':rms(a),
                                 'errorOverNativeEffect':ratio(rms(a-b),rms(b)), 'signedCosineToNativeEffect':ratio(float(np.mean(a*b)),rms(a)*rms(b)),
                                 'errorOverPrimaryEffect':ratio(rms(a-primary[index]),rms(primary[index]))})
        deltas={method:forecasts[method,pid][view].astype(np.float64)-base for method in ['keys_only','values_only','queries_only','keys_values']}
        for name,value in [('keyValue',deltas['keys_values']-deltas['keys_only']-deltas['values_only']),
                           ('querySources',primary-deltas['queries_only']-deltas['keys_values'])]:
            interactions.append({'view':view,'interaction':name,'rms':rms(value),'overPrimaryEffect':ratio(rms(value),rms(primary))})
    with (output/'observations.jsonl').open('x') as stream:
        for row in rows:stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    for row in seal['rows']:
        if sha(output/row['path'])!=row['sha256']:raise ValueError('Scoring changed primary forecast')
    if sha(output/'prediction-seal.json')!=seal_hash:raise ValueError('Scoring mutated seal')
    save(output/'report.json',{'schema':'tce-global-head-report-v1','requestSHA256':request['requestSHA256'],'scope':SCOPE,
         'compile':compiled,'serving':seal,'predictionSealSHA256':seal_hash,'compileFreshProcessSeconds':compile_s,
         'serveFreshProcessSeconds':serve_s,'postSealReferenceSeconds':ref_s,'workerWallSeconds':time.perf_counter()-started,
         'referenceScope':evaluation['scope'],'references':reference_records,'sameArithmeticClosure':closures,
         'recomputedVsPriorReader':same_profile,'aggregate':aggregate,'controls':controls,'interactions':interactions,
         'observations':{'path':'observations.jsonl','sha256':sha(output/'observations.jsonl'),'rows':len(rows)},
         'newNeuralFrameworkCalls':0,'postSealFullAttentionEvaluations':6,'offlineOriginalDenseHeadProjections':6,
         'onlineOriginalDenseProjections':0,'onlineLearnedReducedMapEvaluationsPerPrimary':3,
         'mechanics_status':'complete','instrument_status':'source-bound mathematical attention consumer; no native attention parity claim',
         'terminal_status':'unassessed'})
    print('GLOBAL_HEAD_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','execute','collect']);parser.add_argument('run',type=Path)
    parser.add_argument('--parent',type=Path);parser.add_argument('--context',type=Path);args=parser.parse_args()
    result=stage(args.run,args.parent,args.context) if args.action=='stage' else globals()[args.action](args.run)
    if result is not None:print(json.dumps(result))
