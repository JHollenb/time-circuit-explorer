"""Source-bound row9 finite crossover: two original QKV units, no image."""
import gzip
import hashlib
import os
from pathlib import Path
import resource
import shutil
import time
import numpy as np
from contextual_writer_program import read,save,sha
from contextual_writer_worker import floats,metric
from contrast_writer_worker import Consumer,load_port
from finite_consumer_crossover import construct,decompose,norm2,STATE_FIELDS,VECTOR_FIELDS,POLICY,SCORING

BRANCHES = {'Xno':{'base':'a','meaning':'native-baseline-own-response-after-admission'},'Xon':{'base':'p','meaning':'own-baseline-native-response-after-admission'}}


def words(value):return (np.asarray(value,'<f4').view('<u4') >> 16).astype('<u2')


def load_state(item):
    path=Path(item['path'])
    if sha(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:raise ValueError('Stored state changed')
    value=np.load(path,allow_pickle=False) if item['format']=='npy' else np.fromfile(path,'<u2').reshape(item['shape'])
    if value.dtype!=np.dtype('<u2') or list(value.shape)!=item['shape'] or hashlib.sha256(value.tobytes()).hexdigest()!=item['payloadSHA256']:raise ValueError('Stored state payload/ABI differs')
    return floats(value)[0].astype(np.float64)


def execute(root):
    from finite_consumer_crossover_study import verify,WORK
    import torch
    from saturn.titan import Frame,ActSpec,ActRegistry
    from saturn.tensor_hash import tensor_bytes_sha256
    root=Path(root).resolve();req=read(root/'request.json');verify(root,req);start=time.perf_counter();cpu=time.process_time();torch.set_num_threads(1)
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Expected one numerical CPU thread')
    out=root/'response-result';out.mkdir();folder=root/'arrays';folder.mkdir();position=req['position'];source=read(root/'qualified-parent-inputs.json')
    if position!=9 or req['cases']!=['00','10'] or req['physicalWork']!=WORK or source['parameters']!=req['parameters'] or source['numericalProfile']!=req['numericalProfile']:raise ValueError('Fixed contract differs')
    authenticated=[]
    for path,x in req['remoteInputs'].items():
        if Path(path).stat().st_size!=x['bytes'] or sha(path)!=x['sha256']:raise ValueError('Qualified input changed')
        authenticated.append({'path':path,**x})
    for key in req['cases']:
        packet=req['packets'][key]
        for label,h in [('ids',source['cases'][key]['tokenIDsSHA256']),('mask',source['cases'][key]['maskSHA256'])]:
            if hashlib.sha256(np.asarray(packet[label],'<i8').tobytes()).hexdigest()!=h:raise ValueError('Qualified packet binding differs')
    if any(req['packets'][k]['mask'][9]!=1 for k in req['cases']) or req['packets']['00']['ids'][9]!=req['packets']['10']['ids'][9]:raise ValueError('Row9 is not unchanged active token')
    states={name:load_state(item) for name,item in req['endpoints'].items()};manifest={}
    def tensor(value,bf16=False):
        return torch.from_numpy(words(value).copy()).view(torch.bfloat16) if bf16 else torch.from_numpy(np.ascontiguousarray(value,dtype='<f8'))
    def retain(name,value):
        bf16=value.dtype==torch.bfloat16;data=value.view(torch.uint16).cpu().numpy() if bf16 else value.cpu().numpy();path=folder/(name+('.bf16' if bf16 else '.npy'))
        if bf16:path.write_bytes(data.tobytes())
        else:np.save(path,data,allow_pickle=False)
        item={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'payloadSHA256':tensor_bytes_sha256(value),'shape':list(value.shape),
              'dtype':str(value.dtype),'storedDtype':data.dtype.str,'format':'raw' if bf16 else 'npy'}
        if hashlib.sha256(data.tobytes()).hexdigest()!=item['payloadSHA256']:raise ValueError('Actual retained tensor payload differs')
        manifest[name]=item;return item
    values={name:tensor(value[None],True) for name,value in states.items()};before={k:tensor_bytes_sha256(v) for k,v in values.items()}
    if before!={k:v['payloadSHA256'] for k,v in req['endpoints'].items()}:raise ValueError('Actual construction inputs differ')
    parent=Frame.from_values(values,program_fingerprint=req['requestSHA256'],metadata={'position':position,'source':'authenticated-stored-diagonal-H1'})
    names=('Xno','Xon',*STATE_FIELDS);spec=ActSpec('finite_consumer.crossover','v1',reads=tuple(values),writes=names,numerical_contract=POLICY)
    def build(inputs):
        arrays={k:floats(v.view(torch.uint16).numpy())[0].astype(np.float64) for k,v in inputs.items()};result=construct(*(arrays[k] for k in ('a','b','p','q')),position)
        return {k:tensor(v[None] if k in ('Xno','Xon') else v,k in ('Xno','Xon')) for k,v in result.items()},{'position':position,'crossedStates':2,'newLinears':0}
    registry=ActRegistry();registry.register(spec,'finite-admission',build);tick=time.perf_counter();built,evidence=registry.run(spec.identity,'finite-admission',parent);construction_time=time.perf_counter()-tick
    after={k:tensor_bytes_sha256(v) for k,v in values.items()}
    if before!=after:raise ValueError('Construction input mutation')
    for name in names:retain(name,built.values[name])
    crossed={k:floats(built.values[k].view(torch.uint16).numpy())[0].astype(np.float64) for k in ('Xno','Xon')}
    construction={'input':parent.to_dict(),'output':built.to_dict(),'spec':spec.to_dict(),'evidence':evidence.to_dict(),'inputPayloadBefore':before,'inputPayloadAfter':after,
                  'rowFields':{k:{'normSquared':norm2(built.values[k].numpy()),'nonzeroCoordinates':int(np.count_nonzero(built.values[k].numpy()))} for k in STATE_FIELDS},'seconds':construction_time,'branches':BRANCHES}
    save(out/'typed-construction.json',construction)
    consumer=Consumer(root,source,req);outputs={};old_outputs={}
    for name,record in req['reusedConsumers'].items():
        if record['inputPayloadSHA256']!=before[name]:raise ValueError('Stored diagonal consumer input differs')
        old_outputs[name]={p:load_port(x) for p,x in record['fields'].items()}
    for branch in BRANCHES:
        item=manifest[branch];outputs[branch]=consumer.run(branch,item)
        for port,x in consumer.rows[-1]['fields'].items():manifest[branch+'-'+port]={**x,'dtype':'torch.bfloat16','storedDtype':'<u2','format':'npy'}
        consumer.rows[-1].update({'branch':branch,'semantics':BRANCHES[branch]});consumer.typed[-1].update({'branch':branch,'semantics':BRANCHES[branch]})
    parameter_after={k:tensor_bytes_sha256(v) for k,v in consumer.parent.values.items()}
    if len(consumer.rows)!=2 or parameter_after!=consumer.before or consumer.parent.fingerprint!=consumer.parent_id:raise ValueError('Consumer count or immutable parameters differ')
    if {x['path']:x['sha256'] for x in consumer.assets}!={x['path']:x['sha256'] for x in req['selectedParameterAssets']}:raise ValueError('Actual selected-shard authentication differs')
    save(out/'consumer-fields.json',{'rows':consumer.rows,'parameters':source['parameters'],'parameterAssets':consumer.assets,'reused':req['reusedConsumers']})
    save(out/'typed-consumers.json',{'spec':consumer.spec.to_dict(),'rows':consumer.typed,'unchangedWeightParent':consumer.parent.to_dict(),'parameterPayloadBefore':consumer.before,'parameterPayloadAfter':parameter_after})
    analyses=[];typed_analysis=[]
    for port,width in [('H1',2560),('Q',4096),('K',1024),('V',1024)]:
        arrays={**states,**crossed} if port=='H1' else {**{k:v[port] for k,v in old_outputs.items()},**{k:v[port] for k,v in outputs.items()}}
        inputs={k:tensor(v[None],True) for k,v in arrays.items()};hashes={k:tensor_bytes_sha256(v) for k,v in inputs.items()}
        frame=Frame.from_values(inputs,program_fingerprint=req['requestSHA256'],metadata={'position':position,'port':port,'source':'actual-full-geometry-output-payloads'})
        analysis_spec=ActSpec('finite_consumer.signed_analysis','v1',reads=tuple(inputs),writes=VECTOR_FIELDS,numerical_contract=SCORING);captured={}
        def analyze(inputs):
            data={k:floats(v.view(torch.uint16).numpy())[0,position].astype(np.float64) for k,v in inputs.items()}
            vectors,report=decompose(*(data[k] for k in ('a','b','p','q','Xno','Xon')));captured.update(report)
            return {k:tensor(v) for k,v in vectors.items()},{'position':position,'port':port,'width':width,'learnedOperators':0}
        analysis_registry=ActRegistry();analysis_registry.register(analysis_spec,'FP64-signed-paths',analyze);tick=time.perf_counter();result,receipt=analysis_registry.run(analysis_spec.identity,'FP64-signed-paths',frame);elapsed=time.perf_counter()-tick
        if hashes!={k:tensor_bytes_sha256(v) for k,v in inputs.items()}:raise ValueError('Analysis inputs mutated')
        for name in VECTOR_FIELDS:retain(port+'-'+name,result.values[name])
        analyses.append({'port':port,'seconds':elapsed,**captured});typed_analysis.append({'port':port,'spec':analysis_spec.to_dict(),'input':frame.to_dict(),'output':result.to_dict(),'evidence':receipt.to_dict(),'inputPayloadBefore':hashes,'inputPayloadAfter':hashes})
    summaries=[];controls=[];row_count=0
    with gzip.open(out/'observations.jsonl.gz','xt') as stream:
        import json
        for branch in BRANCHES:
            for port in ('Q','K','V'):
                pred=outputs[branch][port];base=old_outputs[BRANCHES[branch]['base']][port]
                changed=np.any(pred!=base,axis=-1);controls.append({'branch':branch,'port':port,'unmodifiedRows':511,'unmodifiedRowsChanged':int(changed[np.arange(512)!=position].sum())})
                for selection,take in [('all',np.ones(512,bool)),('intervened',np.arange(512)==position),('unmodified',np.arange(512)!=position)]:
                    summaries.append({'branch':branch,'port':port,'selection':selection,'rows':int(take.sum()),**metric(pred[take],base[take])})
                for i in range(512):stream.write(json.dumps({'branch':branch,'port':port,'position':i,'intervened':i==position,**metric(pred[i],base[i])},allow_nan=False)+'\n');row_count+=1
    if row_count!=WORK['comparisonRows']:raise ValueError('Actual comparison domain differs')
    save(out/'analysis.json',analyses);save(out/'typed-analysis.json',typed_analysis);save(out/'array-manifest.json',manifest)
    for name in req['retainedMetadata']:shutil.copyfile(root/name,out/name)
    members=['typed-construction.json','consumer-fields.json','typed-consumers.json','analysis.json','typed-analysis.json','array-manifest.json','observations.jsonl.gz']
    save(out/'analysis-seal.json',{name:sha(out/name) for name in members})
    save(out/'report.json',{'schema':req['schema'],'requestSHA256':req['requestSHA256'],'position':position,'branches':BRANCHES,'physicalWork':WORK,'analyses':analyses,'summaries':summaries,'controls':controls,
         'comparisonRows':row_count,'unmodifiedRowControlRows':3066,'authenticatedRemoteInputs':authenticated,'analysisSealSHA256':sha(out/'analysis-seal.json'),'finitePolicy':POLICY,'scoring':SCORING,
         'workerWallSecondsAfterImportsAndRequestVerification':time.perf_counter()-start,'cpuSeconds':time.process_time()-cpu,'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'parameterLoadSeconds':consumer.load_seconds,'selectedParameterBytes':sum(x['bytes'] for x in source['parameters']),'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'torchVersion':torch.__version__,
         'numpyVersion':np.__version__,'numericalThreads':torch.get_num_threads(),'cudaDeviceName':torch.cuda.get_device_name(),'cudaVersion':torch.version.cuda,
         'numericalProfile':source['numericalProfile'],'mechanics_status':'completed','instrument_status':'inspect-unmodified-row-controls-and-finite-signed-paths','trend_status':'retain-both-crosses-all-ports-and-cancellation',
         'terminal_status':'arbitrary-prompt-zero-neural-images-unachieved','limitations':['Stored targets/ownstates/parameters are supplied research inputs,not serving.','One discovery causalrow,reusedprefixnotreplication.','Cross terms include admission effects;no pure baseline/direction shares.','No fit,attention,MLP,fullencoder,denoiser,VAE,image or fresh prompt.','Historical and six new original learned linears remain paid.']})
    print('FINITE_CONSUMER_CROSSOVER_COMPLETE',flush=True)
