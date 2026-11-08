"""Six full finite operands; gamma-only loading, no affine materialization."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import struct
import time
import numpy as np
from contextual_writer_program import read,save,sha
from contextual_writer_worker import floats,metric
from coupled_normalization import STATES,EDGES,CONTRASTS,ANALYSIS_NAMES,SOURCE_POLICY,EXPLANATORY_POLICY,SCORING,analyze

BRANCHES={'Xno':'a','Xon':'p'}


def load_words(item):
    path=Path(item['path'])
    if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('Stored H1 bytes differ')
    data=np.load(path,allow_pickle=False) if item['format']=='npy' else np.fromfile(path,'<u2').reshape(item['shape'])
    if data.dtype!=np.dtype('<u2') or list(data.shape)!=[1,512,2560] or not data.flags.c_contiguous or hashlib.sha256(data.tobytes()).hexdigest()!=item['payloadSHA256']:raise ValueError('Actual H1 ABI/payload differs')
    return data


def gamma_only(req,torch):
    item=req['gammaParameter'];asset=req['selectedParameterAsset'];path=Path(item['file']);tick=time.perf_counter()
    if str(path)!=asset['path'] or path.stat().st_size!=asset['bytes'] or sha(path)!=asset['sha256']:raise ValueError('Full selected shard differs')
    hash_seconds=time.perf_counter()-tick
    with path.open('rb') as stream:
        size=struct.unpack('<Q',stream.read(8))[0]
        if size!=asset['header_bytes'] or size>16<<20:raise ValueError('Selected header size differs')
        header=stream.read(size)
        if hashlib.sha256(header).hexdigest()!=asset['header_sha256']:raise ValueError('Selected header differs')
        entry=json.loads(header)[item['name']];lo,hi=entry['data_offsets']
        if entry['dtype']!='BF16' or entry['shape']!=[2560] or hi-lo!=5120 or item['bytes']!=5120:raise ValueError('Selected gamma ABI differs')
        stream.seek(8+size+lo);data=stream.read(hi-lo)
    if len(data)!=5120 or hashlib.sha256(data).hexdigest()!=item['payloadSHA256']:raise ValueError('Selected gamma payload differs')
    value=torch.from_numpy(np.frombuffer(data,'<u2').copy()).view(torch.bfloat16).cuda();torch.cuda.synchronize()
    return value,{'fullShardAuthenticationSeconds':hash_seconds,'gammaLoadAndHeaderSeconds':time.perf_counter()-tick-hash_seconds,
                  'authenticatedShardBytes':asset['bytes'],'materializedParameterBytes':5120,'materializedParameterNames':[item['name']]}


def configure(torch,req):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    profile={'bf16ReducedPrecisionReduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'cublasWorkspaceConfig':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'cudaMatmulTF32':torch.backends.cuda.matmul.allow_tf32,
        'cudnnBenchmark':torch.backends.cudnn.benchmark,'cudnnDeterministic':torch.backends.cudnn.deterministic,'cudnnTF32':torch.backends.cudnn.allow_tf32,
        'deterministicAlgorithms':torch.are_deterministic_algorithms_enabled(),'float32MatmulPrecision':torch.get_float32_matmul_precision(),'threads':torch.get_num_threads()}
    if profile!=req['numericalProfile'] or torch.__version__!=req['torchVersion'] or req['rmsNormEpsilon']!=1e-6:raise ValueError('Qualified numerical profile differs')
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('Numerical thread contract differs')
    return profile


def execute(root):
    from coupled_normalization_study import verify,identity,WORK
    import torch
    from saturn.titan import Frame,ActSpec,ActRegistry
    from saturn.tensor_hash import tensor_bytes_sha256
    root=Path(root).resolve();req=read(root/'request.json');verify(root,req);start=time.perf_counter();cpu=time.process_time();profile=configure(torch,req)
    if req['physicalWork']!=WORK or req['states']!=list(STATES) or req['position']!=9 or identity(req['edges'])!=identity(EDGES) or identity(req['contrasts'])!=identity(CONTRASTS):raise ValueError('Fixed family differs')
    if req['sourcePolicy']!=SOURCE_POLICY or req['explanatoryPolicy']!=EXPLANATORY_POLICY or req['scoring']!=SCORING:raise ValueError('Numerical policy differs')
    out=root/'response-result';out.mkdir();folder=root/'arrays';folder.mkdir();manifest={};timing={};tick=time.perf_counter()
    words={k:load_words(req['endpoints'][k]) for k in STATES};timing['stateAuthenticationAndReadSeconds']=time.perf_counter()-tick
    values={k:torch.from_numpy(v.copy()).view(torch.bfloat16) for k,v in words.items()}
    def retain(name,value):
        value=value.detach().cpu().contiguous();bf16=value.dtype==torch.bfloat16;data=value.view(torch.uint16).numpy() if bf16 else value.numpy()
        path=folder/(name+('.bf16' if bf16 else '.npy'))
        if bf16:path.write_bytes(data.tobytes())
        else:np.save(path,data,allow_pickle=False)
        payload=tensor_bytes_sha256(value)
        if hashlib.sha256(data.tobytes()).hexdigest()!=payload:raise ValueError('Retained actual tensor payload differs')
        item={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'payloadSHA256':payload,'shape':list(value.shape),
              'dtype':str(value.dtype),'storedDtype':data.dtype.str,'format':'raw' if bf16 else 'npy'}
        manifest[name]=item;return item
    gamma,load=gamma_only(req,torch);gamma_hash=tensor_bytes_sha256(gamma)
    if gamma_hash!=req['gammaParameter']['payloadSHA256']:raise ValueError('Actual CUDA gamma payload differs')
    retain('gamma',gamma);parent=Frame.from_values({'normWeight':gamma},program_fingerprint=req['requestSHA256']);parent_id=parent.fingerprint
    spec=ActSpec('coupled_normalization.source_operand','v1',reads=('H1','normWeight'),writes=('Z',),numerical_contract=SOURCE_POLICY,
                 write_abis={'Z':{'shape':[1,512,2560],'dtype':'bfloat16'}})
    def normalize(inputs):
        h=inputs['H1'];work=h.float();work=work*torch.rsqrt(work.pow(2).mean(-1,keepdim=True)+req['rmsNormEpsilon'])
        norm=inputs['normWeight']*work.to(h.dtype)
        if norm.dtype!=torch.bfloat16 or list(norm.shape)!=[1,512,2560] or not norm.is_contiguous() or not torch.isfinite(norm).all():raise ValueError('Finite full operand ABI differs')
        return {'Z':norm},{'normalizationUnits':1,'rows':512,'coordinates':1310720,'originalLinears':0,'nominalMatrixMACs':0,'QKVMatricesMaterialized':0,
                           'inputShape':list(h.shape),'inputStride':list(h.stride()),'outputStride':list(norm.stride()),'dtype':str(norm.dtype)}
    registry=ActRegistry();registry.register(spec,'original-normalization-only',normalize);typed=[];operands={}
    for key in STATES:
        tick=time.perf_counter();h=values[key].cuda();torch.cuda.synchronize();transfer=time.perf_counter()-tick
        frame=parent.with_updates({'H1':h},act='six-actual-H1:v1',implementation='authenticated',metadata={'state':key})
        before={k:tensor_bytes_sha256(v) for k,v in frame.values.items()};tick=time.perf_counter();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record()
        with torch.inference_mode():result,evidence=registry.run(spec.identity,'original-normalization-only',frame)
        end.record();torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        after={k:tensor_bytes_sha256(v) for k,v in frame.values.items()}
        if before!=after or before['H1']!=req['endpoints'][key]['payloadSHA256']:raise ValueError('Actual normalization input mutation/binding differs')
        tick=time.perf_counter();operands[key]=result.values['Z'].cpu();retain('Z-'+key,operands[key]);capture=time.perf_counter()-tick
        typed.append({'state':key,'spec':spec.to_dict(),'input':frame.to_dict(),'output':result.to_dict(),'evidence':evidence.to_dict(),
                      'inputPayloadBefore':before,'inputPayloadAfter':after,'hostToDeviceSeconds':transfer,'actSecondsIncludingHashes':elapsed,
                      'cudaEventMillisecondsIncludingActHashWork':begin.elapsed_time(end),'outputTransferAndRetentionSeconds':capture})
        print('NORMALIZED',key,flush=True)
    after=tensor_bytes_sha256(gamma)
    if after!=gamma_hash or parent.fingerprint!=parent_id:raise ValueError('Immutable gamma parent changed')
    save(out/'typed-normalization.json',{'rows':typed,'weightParent':parent.to_dict(),'parameterPayloadBefore':gamma_hash,'parameterPayloadAfter':after})
    analysis_inputs={**{'H-'+k:v for k,v in values.items()},**{'Z-'+k:v for k,v in operands.items()},'gamma':gamma.cpu()}
    hashes={k:tensor_bytes_sha256(v) for k,v in analysis_inputs.items()};frame=Frame.from_values(analysis_inputs,program_fingerprint=req['requestSHA256'],metadata={'position':9,'source':'six-full-H1-six-full-Z-gamma'})
    analysis_spec=ActSpec('coupled_normalization.explanation','v1',reads=tuple(analysis_inputs),writes=ANALYSIS_NAMES,numerical_contract=SCORING+' '+EXPLANATORY_POLICY);captured={}
    def explanation(inputs):
        def row(value):return floats(value.view(torch.uint16).numpy())[0,9].astype(np.float64)
        vectors,report=analyze({k:row(inputs['H-'+k]) for k in STATES},{k:row(inputs['Z-'+k]) for k in STATES},
                               floats(inputs['gamma'].view(torch.uint16).numpy()).astype(np.float64),req['rmsNormEpsilon']);captured.update(report)
        return {k:torch.from_numpy(np.ascontiguousarray(v,dtype='<f8')) for k,v in vectors.items()},{'position':9,'width':2560,'originalLinears':0,'retainedRowFields':79}
    analysis_registry=ActRegistry();analysis_registry.register(analysis_spec,'FP64-coupled-explanation',explanation);tick=time.perf_counter()
    result,evidence=analysis_registry.run(analysis_spec.identity,'FP64-coupled-explanation',frame);timing['analysisActSecondsIncludingHashes']=time.perf_counter()-tick
    if hashes!={k:tensor_bytes_sha256(v) for k,v in analysis_inputs.items()}:raise ValueError('Analysis full inputs mutated')
    for key in ANALYSIS_NAMES:retain(key,result.values[key])
    save(out/'typed-analysis.json',{'spec':analysis_spec.to_dict(),'input':frame.to_dict(),'output':result.to_dict(),'evidence':evidence.to_dict(),'inputPayloadBefore':hashes,'inputPayloadAfter':hashes})
    save(out/'analysis.json',captured);summaries=[];controls=[];count=0;tick=time.perf_counter()
    with gzip.open(out/'observations.jsonl.gz','xt') as stream:
        for branch,base in BRANCHES.items():
            pred=floats(operands[branch].view(torch.uint16).numpy())[0].astype(np.float64);target=floats(operands[base].view(torch.uint16).numpy())[0].astype(np.float64)
            changed=np.any(pred!=target,axis=-1);controls.append({'branch':branch,'unmodifiedRows':511,'unmodifiedRowsChanged':int(changed[np.arange(512)!=9].sum())})
            for selection,take in [('all',np.ones(512,bool)),('intervened',np.arange(512)==9),('unmodified',np.arange(512)!=9)]:summaries.append({'branch':branch,'selection':selection,'rows':int(take.sum()),**metric(pred[take],target[take])})
            for i in range(512):stream.write(json.dumps({'branch':branch,'position':i,'intervened':i==9,**metric(pred[i],target[i])},allow_nan=False)+'\n');count+=1
    timing['comparisonAndRetentionSeconds']=time.perf_counter()-tick
    if count!=1024 or len(manifest)!=86:raise ValueError('Complete output domain differs')
    save(out/'array-manifest.json',manifest)
    for name in req['retainedMetadata']:shutil.copyfile(root/name,out/name)
    members=['typed-normalization.json','typed-analysis.json','analysis.json','array-manifest.json','observations.jsonl.gz']
    save(out/'analysis-seal.json',{name:sha(out/name) for name in members})
    save(out/'report.json',{'schema':req['schema'],'requestSHA256':req['requestSHA256'],'position':9,'physicalWork':WORK,'sourcePolicy':SOURCE_POLICY,'explanatoryPolicy':EXPLANATORY_POLICY,
        'scoring':SCORING,'summaries':summaries,'controls':controls,'comparisonRows':count,'unmodifiedRowControlRows':1022,'analysisSealSHA256':sha(out/'analysis-seal.json'),
        'gammaLoad':load,'timing':timing,'workerWallSecondsAfterImportsAndRequestVerification':time.perf_counter()-start,'cpuSeconds':time.process_time()-cpu,
        'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'torchVersion':torch.__version__,
        'numpyVersion':np.__version__,'numericalProfile':profile,'numericalThreads':torch.get_num_threads(),'cudaDeviceName':torch.cuda.get_device_name(),'cudaVersion':torch.version.cuda,
        'retainedNumericFileBytes':sum(x['bytes'] for x in manifest.values()),'mechanics_status':'completed','instrument_status':'inspect-controls-finite-residuals-and-signed-terms',
        'trend_status':'retain-both-paths-without-proxy-threshold','terminal_status':'arbitrary-prompt-zero-neural-images-unachieved',
        'limitations':['Supplied stored107 states and original gamma are research inputs,not serving.','New source-policy reconstruction; unavailable107 normalized intermediates cannot establish direct parity.',
                       'One discovery causal row; no independent specimen.','rho includes explanatory FP64 rounding; signed components are not additive causal percentages.',
                       'Pre-affine geometry cannot determine unique QKV attribution;107 QKV scalars reused only.','Six original normalization/gamma evaluations remain paid neural arithmetic;zero new affineMACs is not zero neural work.','No fit,newprompt,QKV,attention,MLP,suffix or image.']})
    print('COUPLED_NORMALIZATION_COMPLETE',flush=True)
