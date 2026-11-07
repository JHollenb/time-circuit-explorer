"""One resident four-matrix image-local consumer, no attention or full model."""
import os
from pathlib import Path
import sys
import time
from encoder_atoms import read,save,sha,identity
from paired_fields import load,retain,payload


def execute(root):
    from flux_clock_study import verify,source_checks,assets_unchanged
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json');out=root/'response-result';out.mkdir()
    allowed={str(Path(s['modelRoot'])/n) for n in s['imageAssets']}|{x['path'] for x in s['donors'].values()}|{s['modulation']['path']}|{x['z']['path'] for x in s['receivers'].values()};protected=[Path(x).resolve() for x in s['protectedRoots']]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(root):return
        if str(p) not in allowed and (p.suffix in {'.npy','.npz','.safetensors','.pt','.pth'} or any(p.is_relative_to(x) for x in protected)):raise PermissionError('Undeclared first-consumer operand/target/model')
    sys.addaudithook(audit)
    for p in s['deniedCanaries']:
        try:open(p,'rb')
        except PermissionError:continue
        raise RuntimeError('First-consumer target quarantine inactive')
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from complete_masked_study import profile
    from flux_clock import header,selected_load,snapshot
    from native_reader_corpus import load as historical
    from mdb.titan import Frame,ActSpec,ActRegistry,ProvidedFingerprint
    from first_consumer import SELECTED,make_local,path
    start=time.perf_counter();cpu=time.process_time();numerical=profile(torch,4)
    if numerical!=s['numericalProfile']:raise ValueError('Qualified profile differs')
    model_root,assets=source_checks(s,diffusers);weights=model_root/'transformer/diffusion_pytorch_model.safetensors';entries,head=header(weights)
    if len(entries)!=169 or {n:entries[n]['shape'] for n in SELECTED}!=SELECTED:raise ValueError('Original selected header differs')
    torch.cuda.reset_peak_memory_stats()
    with torch.device('meta'):model=make_local(torch,native)
    loaded=selected_load(model,weights,sorted(SELECTED),entries,'cuda');model.eval().requires_grad_(False)
    if loaded!=s['selectedParameters'] or sum(x['bytes'] for x in loaded)!=189530112:raise ValueError('Exact four-parameter payload admission differs')
    def census():return {n:[x.data_ptr(),x._version,list(x.shape),str(x.dtype)] for n,x in model.named_parameters()}
    before=census();mod=load(s['modulation'],'cuda');mod_before=payload(mod);parts=native.Flux2Modulation.split(mod,2);block=model.transformer_blocks[0]
    zs={k:historical(x['z'],'cuda') for k,x in s['receivers'].items()};donors={k:load(x,'cuda') for k,x in s['donors'].items()};layouts={};geometry=s['carrierGeometry']
    for k,x in donors.items():
        shape=geometry['shape'];stride=geometry['stride'];offset=geometry['storageOffset'];size=offset+1+sum((n-1)*v for n,v in zip(shape,stride));backing=torch.empty(size,dtype=x.dtype,device=x.device);v=backing.as_strided(shape,stride,offset);v.copy_(x)
        if payload(v)!=s['donors'][k]['payloadSHA256'] or v.untyped_storage().data_ptr()==x.untyped_storage().data_ptr():raise ValueError('Supplied projection geometry/payload differs')
        layouts[k]=v
    inputs={**{'Z.'+k:v for k,v in zs.items()},**{'H.'+k:v for k,v in donors.items()},**{'layout.'+k:v for k,v in layouts.items()},'modulation':mod}
    def input_census():return {k:[v.untyped_storage().data_ptr(),v._version,payload(v)] for k,v in inputs.items()}
    input_before=input_census();phase={};linears=[];primary={'shared':{},'embeddings':{},'constants':{},'cells':{},'views':{},'frames':[]};cells={};active={};current={};timings=[]
    names={v.data_ptr():n for n,v in model.named_parameters()};old_linear=torch.nn.functional.linear;old_attention=native.dispatch_attention_fn
    def linear(x,w,bias=None):
        name=names.get(w.data_ptr())
        if name not in SELECTED or bias is not None:raise ValueError('Unadmitted affine parameter/bias')
        linears.append({**phase,'parameter':name,'inputShape':list(x.shape),'inputStride':list(x.stride()),'inputStorageOffset':x.storage_offset(),'weightShape':list(w.shape),'nominalMACs':x.numel()//x.shape[-1]*w.numel()});return old_linear(x,w,bias)
    def deny_attention(*args,**kwargs):raise ValueError('Attention is outside local consumer contract')
    def capture(name,value):
        if name in active:raise ValueError('Repeated local boundary')
        active[name]=retain(root/'arrays/cells'/phase['cell'],name,value)
    def fused_hook(module,args,y):
        capture('fused',y);half=y.shape[-1]//2;primary['views'][phase['cell']]={k:{'parentField':'fused','lastAxisSlice':[lo,hi],'shape':[1,1024,9216],'payloadSHA256':payload(y[...,lo:hi])} for k,lo,hi in [('gateOperand',0,half),('upOperand',half,2*half)]}
    hooks=[block.ff.linear_in.register_forward_hook(fused_hook),block.ff.act_fn.gate_fn.register_forward_hook(lambda m,args,y:capture('activation',y)),block.ff.act_fn.register_forward_hook(lambda m,args,y:capture('product',y))]
    registry=ActRegistry();act=ActSpec('tce.first_image_consumer','v1',reads=('embedding','gatedCarrier','modulation'),writes=('imageExit',),write_kinds={'imageExit':'tensor'},write_abis={'imageExit':{'shape':[1,1024,3072],'dtype':'torch.bfloat16'}},numerical_contract='pinned-native-image-local-source-order-bf16')
    def operation(values):
        if payload(values['modulation'])!=s['modulation']['payloadSHA256']:raise ValueError('Typed modulation binding differs')
        y=path(block,values['embedding'],values['gatedCarrier'],parts,capture);return {'imageExit':y},{'capturedImageExitPayloadSHA256':payload(y),'originalMLPPaths':1}
    registry.register(act,'selected-native-image-path',operation);setup=None;rng_cpu=torch.get_rng_state().clone();rng_cuda=[v.clone() for v in torch.cuda.get_rng_state_all()]
    try:
        torch.nn.functional.linear=linear;native.dispatch_attention_fn=deny_attention
        with torch.inference_mode():
            embeddings={};writes={}
            for rr,z in zs.items():
                phase.clear();phase.update(role='input-projection',receiver=rr);value=model.x_embedder(z)
                if payload(value)!=s['expectedPrefixes'][rr]['hidden_states']:raise ValueError('Reconstructed embedding differs from archived native prefix')
                embeddings[rr]=value;primary['embeddings'][rr]=retain(root/'arrays/shared',rr+'.embedding',value)
            for h,x in layouts.items():
                phase.clear();phase.update(role='carrier-projection',carrier=h);projected=block.attn.to_out[1](block.attn.to_out[0](x));gated=parts[0][2]*projected;writes[h]=gated
                primary['shared'][h]={'projection':retain(root/'arrays/shared',h+'.projection',projected),'gatedCarrier':retain(root/'arrays/shared',h+'.gatedCarrier',gated)}
            primary['constants']['onePlusScale']=retain(root/'arrays/shared','onePlusScale',1+parts[1][1]);setup=time.perf_counter()-start
            for rr in ['R0','R1']:
                for h in ['A','B']:
                    cell=rr+h;phase.clear();phase.update(role='local-image-path',cell=cell,receiver=rr,carrier=h);active={}
                    if snapshot(torch)!=numerical:raise ValueError('Native profile changed')
                    values={'embedding':embeddings[rr],'gatedCarrier':writes[h],'modulation':mod};fp={k:ProvidedFingerprint.content_sha256(payload(v)) for k,v in values.items()};abi={k:{'shape':list(v.shape),'dtype':str(v.dtype),'stride':list(v.stride()),'storageOffset':v.storage_offset()} for k,v in values.items()}
                    frame=Frame.from_values(values,kinds={k:'tensor' for k in values},abis=abi,slot_fingerprints=fp,program_fingerprint=r['requestSHA256'],metadata={**phase,'receiverSource':s['receivers'][rr],'carrierSource':s['donors'][h],'modulationSource':s['modulation']})
                    torch.cuda.synchronize();tick=time.perf_counter();ct=time.process_time();result,evidence=registry.run(act.identity,'selected-native-image-path',frame);torch.cuda.synchronize();timings.append({**phase,'wallSecondsInstrumented':time.perf_counter()-tick,'cpuSecondsInstrumented':time.process_time()-ct})
                    expected={'residualCommit','normalization','normScaled','mlpInput','fused','activation','product','mlpWrite','gatedMLP','imageExit'}
                    if set(active)!=expected or payload(result.values['imageExit'])!=active['imageExit']['payloadSHA256']:raise ValueError('Complete native boundary capture differs')
                    primary['cells'][cell]=active;primary['frames'].append({'cell':cell,'sourceFrame':frame.to_dict(),'outputFrame':result.to_dict(),'evidence':evidence.to_dict()});print('FIRST_CONSUMER_CELL',cell,flush=True)
    finally:
        torch.nn.functional.linear=old_linear;native.dispatch_attention_fn=old_attention
        for hook in hooks:hook.remove()
    if census()!=before or input_census()!=input_before or payload(mod)!=mod_before:raise ValueError('Original local model/input closure changed')
    if not torch.equal(rng_cpu,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise ValueError('Local consumer RNG closure changed')
    if len(linears)!=12 or sum(x['nominalMACs'] for x in linears)!=368025010176:raise ValueError('Local affine census differs')
    assets_unchanged(assets);save(out/'primary.json',primary);save(out/'primary-seal.json',{'schema':'tce-first-consumer-primary-seal-v1','requestSHA256':r['requestSHA256'],'primarySHA256':sha(out/'primary.json')});seal=sha(out/'primary-seal.json');print('FIRST_CONSUMER_SEALED',seal,flush=True)
    from first_consumer_analysis import execute as analyze
    analysis=analyze(root,primary,out)
    if sha(out/'primary-seal.json')!=seal or sha(out/'primary.json')!=read(out/'primary-seal.json')['primarySHA256']:raise ValueError('Primary release changed')
    save(out/'execution.json',{'schema':'tce-first-consumer-execution-v1','loadedParameters':loaded,'header':head,'assets':assets,'numericalProfile':numerical,'parameterInputVersionPayloadClosure':True,'consumerRNGClosure':True,'expectedPrefixHashes':s['expectedPrefixes'],'carrierGeometry':geometry,
        'linears':linears,'timings':timings,'setupIncludingSharedProjectionsSecondsAfterImports':setup,'workerWallSecondsAfterImports':time.perf_counter()-start,'workerCPUSecondsAfterImports':time.process_time()-cpu,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),'fileQuarantine':{'allowedReadFiles':sorted(allowed),'deniedCanaries':s['deniedCanaries'],'scope':'Trusted-code audit hook, no historical velocity targets ever admitted.'}})
    import shutil
    for n in ['request.json','inputs.json','resource-basis.json','frozen-plan.md']:shutil.copyfile(root/n,out/n)
    save(out/'report.json',{'schema':'tce-first-consumer-result-v1','requestSHA256':r['requestSHA256'],'primarySealSHA256':seal,**analysis,'nominalAffineMACs':sum(x['nominalMACs'] for x in linears),
        'work':{'selectedOriginalMatrices':4,'selectedParameterBytes':189530112,'originalInputProjections':2,'sharedOriginalCarrierProjections':2,'originalMLPPaths':4,'affineCalls':12,'originalLayerNormCalls':4,'originalSiLUCalls':4,'fullTransformerCalls':0,'attentionCalls':0,'textPathCalls':0,'newClockNetworks':0,'newEncoders':0,'newFits':0,'newSchedulerCommits':0,'newVAECalls':0,'newImages':0},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed','limits':['Local source-order consumer on supplied states/carriers. Prefix hashes qualify input projection, not independent native full-block-exit parity.',
            'Stage/finite-change factorization uses measured operands, not predicted unseen effects. No whole-future mediation or sole missing cause inferred.',
            'Original four-matrix/norm/MLP learned work and old donor/clock/state preparation remain paid. No cheap complete velocity, pixel decoder or arbitrary-prompt zero-neural generation.',
            'FP64 references approximate real arithmetic; native normalization/activation/product/commit policy differences are retained separately. No extra down-projection of attribution terms.',
            'Instrumented time includes capture/hash/I/O/diagnostic arithmetic; no matched serving speedup claim.']});print('FIRST_CONSUMER_DONE',flush=True)
