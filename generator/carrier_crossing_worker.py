"""Four ordinary native transformer calls with a typed complete-carrier clamp."""
import json
import os
from pathlib import Path
import sys
import time
from encoder_atoms import np,read,save,sha,identity
from paired_fields import load,retain,payload,metrics


def execute(root):
    from flux_clock_study import verify,source_checks,assets_unchanged
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json');out=root/'response-result';out.mkdir()
    allowed={str(Path(s['modelRoot'])/n) for n in s['imageAssets']}|{x['path'] for x in s['donors'].values()}|{s['clock']['path']}
    for x in s['common'].values():allowed.update(x['files'])
    allowed.update(x['z']['path'] for x in s['receivers'].values());allowed={str(Path(x).resolve()) for x in allowed};protected=[Path(x).resolve() for x in s['protectedRoots']]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(root):return
        if str(p) not in allowed and (p.suffix in {'.npy','.npz','.safetensors','.pt','.pth'} or any(p.is_relative_to(x) for x in protected)):raise PermissionError('Undeclared crossing input/model/target')
    sys.addaudithook(audit)
    for p in s['deniedCanaries']:
        try:open(p,'rb')
        except PermissionError:continue
        raise RuntimeError('Native crossing target quarantine inactive')
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from diffusers import Flux2Transformer2DModel
    from complete_masked_study import profile
    from flux_clock import snapshot,header,selected_load
    from native_reader_corpus import load as historical
    from mdb.titan import Frame,ActSpec,ActRegistry,ProvidedFingerprint
    from carrier_crossing import replace_geometry,factorial
    start=time.perf_counter();cpu=time.process_time();numerical=profile(torch,4)
    if numerical!=s['numericalProfile']:raise ValueError('Qualified numerical profile differs')
    model,assets=source_checks(s,diffusers);entries,h=header(model/'transformer/diffusion_pytorch_model.safetensors');config=read(model/'transformer/config.json')
    if len(entries)!=169 or config['num_attention_heads']!=24 or config.get('guidance_embeds') is not False:raise ValueError('Original transformer closure differs')
    torch.cuda.reset_peak_memory_stats()
    with torch.device('meta'):transformer=Flux2Transformer2DModel.from_config(config)
    loaded=selected_load(transformer,model/'transformer/diffusion_pytorch_model.safetensors',sorted(entries),entries,'cuda');transformer.eval().requires_grad_(False)
    def processors():return sorted({type(m.processor).__name__ for m in transformer.modules() if hasattr(m,'processor')})
    if processors()!=s['attentionProcessors'] or transformer.is_cache_enabled or getattr(transformer,'_hf_hook',None) is not None:raise ValueError('Native attention/cache profile differs')
    def census():return {n:[v.data_ptr(),v._version,list(v.shape),str(v.dtype)] for n,v in [*transformer.named_parameters(),*transformer.named_buffers()]}
    before=census();buffers={n:payload(v) for n,v in transformer.named_buffers()};parameter_id=identity({x['name']:x['payloadSHA256'] for x in loaded})
    common={k:historical(x,'cuda') for k,x in s['common'].items()};clock=load(s['clock'],'cuda');donors={k:load(x,'cuda') for k,x in s['donors'].items()};donor_before={k:[x.untyped_storage().data_ptr(),x._version,payload(x)] for k,x in donors.items()}
    receivers={k:historical(x['z'],'cuda') for k,x in s['receivers'].items()};receiver_before={k:[x.untyped_storage().data_ptr(),x._version,payload(x)] for k,x in receivers.items()};common_before={k:payload(x) for k,x in common.items()};clock_before=payload(clock)
    phase={};linears=[];attention=[];writes=[];frames=[];timings=[];fields={};native_heads={};text_heads={};text_writes={};prefix_contracts={};counts={};current={};registry=ActRegistry()
    act=ActSpec('tce.replace_first_joint_image_carrier','v1',reads=('nativeH0','donorH0','receiverContract'),writes=('consumedH0',),write_kinds={'consumedH0':'tensor'},
        write_abis={'consumedH0':{'shape':[1,1024,3072],'dtype':'torch.bfloat16'}},numerical_contract='complete-carrier-replacement-native-stride-offset-preserved')
    def operation(values):
        x=values['nativeH0'];donor=values['donorH0'];contract=values['receiverContract']
        if contract!=current['contract'] or payload(donor)!=s['donors'][phase['carrier']]['payloadSHA256']:raise ValueError('Carrier/receiver binding differs')
        value=replace_geometry(x,donor)
        if payload(value)!=s['donors'][phase['carrier']]['payloadSHA256']:raise ValueError('Actual consumed carrier differs')
        return {'consumedH0':value},{'nativeH0PayloadSHA256':payload(x),'consumedPayloadSHA256':payload(value),'recipientStride':list(x.stride()),'consumedStride':list(value.stride()),
            'recipientStorageOffset':x.storage_offset(),'consumedStorageOffset':value.storage_offset(),'separateStorage':True}
    registry.register(act,'native-projection-input-clamp',operation)
    def capture_unique(table,key,name,value):
        if key not in table:table[key]=retain(root/'arrays/receiver'/key,name,value)
        elif payload(value)!=table[key]['payloadSHA256']:raise ValueError('Unpatched receiver state changed between carrier arms')
        return table[key]
    def clamp(module,args):
        if len(args)!=1 or list(args[0].shape)!=[1,1024,3072] or args[0].dtype!=torch.bfloat16:raise ValueError('Projection clamp boundary differs')
        x=args[0];rr=phase['receiver'];counts[phase['cell']]['clamps']+=1;native_field=capture_unique(native_heads,rr,'nativeH0',x)
        vals={'nativeH0':x,'donorH0':donors[phase['carrier']],'receiverContract':current['contract']}
        fp={k:ProvidedFingerprint.content_sha256(identity(v) if k=='receiverContract' else payload(v)) for k,v in vals.items()}
        abi={k:{'shape':list(v.shape),'dtype':str(v.dtype),'stride':list(v.stride()),'storageOffset':v.storage_offset()} for k,v in vals.items() if k!='receiverContract'}
        frame=Frame.from_values(vals,kinds={k:'tensor' if k!='receiverContract' else 'receiver-contract' for k in vals},abis=abi,slot_fingerprints=fp,program_fingerprint=r['requestSHA256'],metadata=dict(phase))
        result,ev=registry.run(act.identity,'native-projection-input-clamp',frame);value=result.values['consumedH0'];evidence=ev.to_dict()
        writes.append({**phase,'nativeH0':native_field,'donor':s['donors'][phase['carrier']],'consumedPayloadSHA256':payload(value),'shape':list(value.shape),'dtype':str(value.dtype),
            'recipientStride':list(x.stride()),'consumedStride':list(value.stride()),'recipientStorageOffset':x.storage_offset(),'consumedStorageOffset':value.storage_offset(),'recipientStoragePointer':x.untyped_storage().data_ptr(),'donorStoragePointer':donors[phase['carrier']].untyped_storage().data_ptr(),'consumedStoragePointer':value.untyped_storage().data_ptr(),'evidence':evidence})
        frames.append({'cell':phase['cell'],'sourceFrame':frame.to_dict(),'outputFrame':result.to_dict()});return (value,)
    def text_pre(module,args):capture_unique(text_heads,phase['receiver'],'textHead',args[0]);counts[phase['cell']]['textWrites']+=1
    def text_post(module,args,y):capture_unique(text_writes,phase['receiver'],'textWrite',y)
    def block_pre(module,args,kwargs):
        hashes={k:payload(v) for k,v in kwargs.items() if isinstance(v,torch.Tensor)};rr=phase['receiver']
        if rr in prefix_contracts and prefix_contracts[rr]!=hashes:raise ValueError('First-block receiver prefix differs')
        prefix_contracts[rr]=hashes
    weight_names={v.data_ptr():n for n,v in transformer.named_parameters()};old_linear=torch.nn.functional.linear;old_attention=native.dispatch_attention_fn
    def linear(x,w,bias=None):
        name=weight_names.get(w.data_ptr())
        if name is None:raise ValueError('Unbound original affine weight')
        linears.append({**phase,'parameter':name,'inputShape':list(x.shape),'weightShape':list(w.shape),'nominalMACs':x.numel()//x.shape[-1]*w.numel()});return old_linear(x,w,bias)
    def dispatch(q,k,v,*args,**kwargs):
        attention.append({**phase,'queryShape':list(q.shape),'keyShape':list(k.shape),'valueShape':list(v.shape),'nominalQKAVMACs':2*q.shape[0]*q.shape[1]*k.shape[1]*q.shape[2]*q.shape[3]});return old_attention(q,k,v,*args,**kwargs)
    hooks=[transformer.transformer_blocks[0].attn.to_out[0].register_forward_pre_hook(clamp),transformer.transformer_blocks[0].attn.to_add_out.register_forward_pre_hook(text_pre),
        transformer.transformer_blocks[0].attn.to_add_out.register_forward_hook(text_post),transformer.transformer_blocks[0].register_forward_pre_hook(block_pre,with_kwargs=True)]
    setup=time.perf_counter()-start;rng_cpu=torch.get_rng_state().clone();rng_cuda=[x.clone() for x in torch.cuda.get_rng_state_all()]
    try:
        torch.nn.functional.linear=linear;native.dispatch_attention_fn=dispatch
        with torch.inference_mode():
            for rr in ['R0','R1']:
                for carrier in ['A','B']:
                    cell=rr+carrier;phase.clear();phase.update(cell=cell,receiver=rr,carrier=carrier);counts[cell]={'clamps':0,'textWrites':0};z=receivers[rr]
                    current['contract']={'receiver':rr,'z':s['receivers'][rr]['z'],'common':s['common'],'clock':s['clock'],'parameterMapIdentity':parameter_id,'numericalProfile':numerical}
                    if snapshot(torch)!=numerical:raise ValueError('Native numerical profile changed')
                    torch.cuda.synchronize();tick=time.perf_counter();ct=time.process_time();ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True);ev0.record()
                    with transformer.cache_context('cond'):
                        velocity=transformer(hidden_states=z,timestep=clock,guidance=None,encoder_hidden_states=common['conditioning'],txt_ids=common['txt_ids'],img_ids=common['img_ids'],joint_attention_kwargs=None,return_dict=False)[0]
                    ev1.record();torch.cuda.synchronize();timings.append({**phase,'wallSecondsInstrumented':time.perf_counter()-tick,'cpuSecondsInstrumented':time.process_time()-ct,'cudaEventIntervalSecondsInstrumented':ev0.elapsed_time(ev1)/1000})
                    if list(velocity.shape)!=[1,1024,128] or velocity.dtype!=torch.bfloat16:raise ValueError('Native velocity ABI differs')
                    fields[cell]=retain(root/'arrays/velocity',cell,velocity)
                    if counts[cell]!={'clamps':1,'textWrites':1}:raise ValueError('Declared clamp/text-write count differs')
                    print('CARRIER_CROSSING_CELL',cell,fields[cell]['payloadSHA256'],flush=True)
    finally:
        torch.nn.functional.linear=old_linear;native.dispatch_attention_fn=old_attention
        for hook in hooks:hook.remove()
    if census()!=before or buffers!={n:payload(v) for n,v in transformer.named_buffers()} or processors()!=s['attentionProcessors']:raise ValueError('Original model identity/version/buffer closure changed')
    if not torch.equal(rng_cpu,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise ValueError('Consumer RNG closure changed')
    if donor_before!={k:[x.untyped_storage().data_ptr(),x._version,payload(x)] for k,x in donors.items()} or receiver_before!={k:[x.untyped_storage().data_ptr(),x._version,payload(x)] for k,x in receivers.items()} or common_before!={k:payload(x) for k,x in common.items()} or clock_before!=payload(clock):raise ValueError('Donor/receiver/common inputs mutated')
    if len(linears)!=436 or len(attention)!=100:raise ValueError('Declared native computation census differs')
    assets_unchanged(assets)
    save(root/'velocity-seal.json',{'schema':'tce-carrier-crossing-velocity-seal-v1','requestSHA256':r['requestSHA256'],'fields':fields,'writes':writes,'frames':frames})
    seal=sha(root/'velocity-seal.json');print('CARRIER_CROSSING_SEALED',seal,flush=True)
    cells={k:load(v) for k,v in fields.items()};derived,energy=factorial(cells);derived_fields={n:retain(root/'arrays/derived',n,x) for n,x in derived.items()}
    from source_response import mask
    selected=mask('bottom-right');accounts=[];observations=0
    with (out/'observations.jsonl').open('x') as stream:
        for scope,selection in [('all',np.ones(1024,bool)),('source-nonlocal',selected)]:
            a=derived['responseR1'][0,selection];b=derived['responseR0'][0,selection];accounts.append({'scope':scope,'queries':int(selection.sum()),**metrics(a,b)})
        for index in range(1024):
            stream.write(json.dumps({'imageQueryIndex':index,'sourceNonlocal':bool(selected[index]),**metrics(derived['responseR1'][0,index],derived['responseR0'][0,index])},allow_nan=False)+'\n');observations+=1
    if sha(root/'velocity-seal.json')!=seal:raise ValueError('Four-velocity release changed')
    save(out/'derived-fields.json',derived_fields);save(out/'execution.json',{'schema':'tce-carrier-crossing-execution-v1','loadedParameters':loaded,'parameterMapIdentity':parameter_id,'assets':assets,'numericalProfile':numerical,'attentionProcessors':processors(),'parameterIdentityVersionClosure':True,'bufferPayloadClosure':True,'consumerRNGClosure':True,
        'linears':linears,'attentionCalls':attention,'nativeH0Fields':native_heads,'textHeadFields':text_heads,'textWriteFields':text_writes,'receiverPrefixHashes':prefix_contracts,'counts':counts,'timings':timings,'setupSecondsAfterImports':setup,'workerWallSecondsAfterImports':time.perf_counter()-start,'workerCPUSecondsAfterImports':time.process_time()-cpu,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),'fileQuarantine':{'allowedReadFiles':sorted(allowed),'deniedCanaries':s['deniedCanaries'],'scope':'Trusted-code audit hook, not adversarial sandbox.'}})
    for n in ['request.json','inputs.json','resource-basis.json','frozen-plan.md','velocity-seal.json']:shutil_copy(root/n,out/n)
    save(out/'report.json',{'schema':'tce-carrier-crossing-result-v1','requestSHA256':r['requestSHA256'],'velocitySealSHA256':seal,'accounts':accounts,'energy':energy,'observations':observations,
        'nominalAffineMACs':sum(x['nominalMACs'] for x in linears),'nominalQKAVMACs':sum(x['nominalQKAVMACs'] for x in attention),'work':{'modelLoads':1,'originalTransformerCalls':4,'recipientH0Computations':4,'carrierClamps':4,'affineCalls':len(linears),'attentionDispatches':len(attention),'newDonorFormation':0,'newEncoderCalls':0,'newSchedulerCommits':0,'newVAECalls':0,'newImages':0,'newFits':0},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed','limits':['Four supplied-carrier interventions; selective-SDPA donors do not imply natural native parity. Hybrid states may be outside natural distribution.',
            'Interaction measures receiver dependence on this intervention domain, not missing natural H0 information, sole missing local latent state or a cheaper producer.',
            'All original transformer/clock/text/image attention is paid, including discarded recipient H0. Historical learned donors/conditioning/prefixes remain paid.',
            'Two receivers/one donor pair are dependent contrasts, not broad arbitrary-prompt coverage; no full image or zero-neural generator claim.',
            'Wall/CPU/CUDA-event intervals include instrumentation, hashing and captures; no matched runtime speedup claim.']})
    print('CARRIER_CROSSING_DONE',flush=True)


def shutil_copy(a,b):
    import shutil
    shutil.copyfile(a,b)
