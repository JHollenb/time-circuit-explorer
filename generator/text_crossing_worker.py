"""Two original denoisers with typed text-output and image-input clamps."""
import json
import os
from pathlib import Path
import sys
import time
from encoder_atoms import np,read,save,sha,identity
from paired_fields import load,retain,payload


def execute(root):
    from flux_clock_study import verify,source_checks,assets_unchanged
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json');out=root/'response-result';out.mkdir();scoring={'open':False}
    allowed={str(Path(s['modelRoot'])/n) for n in s['imageAssets']}|{x['path'] for x in s['donors'].values()}|{s['clock']['path'],s['textDonor']['path']}
    for x in s['common'].values():allowed.update(x['files'])
    allowed.update(x['z']['path'] for x in s['receivers'].values());allowed={str(Path(x).resolve()) for x in allowed};protected=[Path(x).resolve() for x in s['protectedRoots']];target_files=set()
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(root):return
        if scoring['open'] and str(p) in target_files:return
        if str(p) not in allowed and (p.suffix in {'.npy','.npz','.safetensors','.pt','.pth'} or any(p.is_relative_to(x) for x in protected)):raise PermissionError('Undeclared text-crossing input/target/model')
    sys.addaudithook(audit)
    for p in s['deniedCanaries']:
        try:open(p,'rb')
        except PermissionError:continue
        raise RuntimeError('Pre-seal velocity/model quarantine inactive')
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from diffusers import Flux2Transformer2DModel
    from complete_masked_study import profile
    from flux_clock import snapshot,header,selected_load
    from native_reader_corpus import load as historical
    from mdb.titan import Frame,ActSpec,ActRegistry,ProvidedFingerprint
    from carrier_crossing import replace_geometry
    from text_crossing import decompose,account
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
    if parameter_id!=s['parameterMapIdentity']:raise ValueError('Original parameter payload map differs')
    common={k:historical(x,'cuda') for k,x in s['common'].items()};clock=load(s['clock'],'cuda');donors={k:load(x,'cuda') for k,x in s['donors'].items()};text_donor=load(s['textDonor'],'cuda');z=historical(s['receivers']['R1']['z'],'cuda')
    all_inputs={**{'common.'+k:v for k,v in common.items()},**{'donor.'+k:v for k,v in donors.items()},'clock':clock,'textDonor':text_donor,'receiver':z}
    def input_census():return {k:[x.untyped_storage().data_ptr(),x._version,payload(x)] for k,x in all_inputs.items()}
    input_before=input_census();phase={};linears=[];attention=[];writes=[];frames=[];timings=[];fields={};captures={};counts={};current={};prefix={};registry=ActRegistry();acts={}
    for kind,shape in [('text',[1,512,3072]),('image',[1,1024,3072])]:
        act=ActSpec('tce.text_crossing.replace_'+kind,'v1',reads=('native','donor','receiverContract'),writes=('consumed',),write_kinds={'consumed':'tensor'},write_abis={'consumed':{'shape':shape,'dtype':'torch.bfloat16'}},numerical_contract='whole-tensor-replacement-native-stride-offset-independent-storage')
        def operation(values,kind=kind):
            x=values['native'];donor=values['donor'];descriptor=s['textDonor'] if kind=='text' else s['donors'][phase['carrier']]
            if values['receiverContract']!=current['contract'] or payload(donor)!=descriptor['payloadSHA256']:raise ValueError('Typed receiver/donor binding differs')
            value=replace_geometry(x,donor)
            if payload(value)!=descriptor['payloadSHA256']:raise ValueError('Exact consumed replacement differs')
            return {'consumed':value},{'nativePayloadSHA256':payload(x),'consumedPayloadSHA256':payload(value),'separateStorage':True}
        registry.register(act,'native-boundary-clamp',operation);acts[kind]=act
    def capture(name,value):
        if name not in captures:captures[name]=retain(root/'arrays/receiver',name,value)
        elif payload(value)!=captures[name]['payloadSHA256']:raise ValueError('Unpatched native path changed between A/B')
        if captures[name]['payloadSHA256']!=s['expectedNativeFields'][name]['payloadSHA256']:raise ValueError('Native receiver path differs from qualified parent')
        return captures[name]
    def replace(kind,x):
        expected=[1,512 if kind=='text' else 1024,3072]
        if list(x.shape)!=expected or x.dtype!=torch.bfloat16:raise ValueError('Declared replacement ABI differs')
        if kind=='image' and current.get('textFrameID') is None:raise ValueError('Text output must be fixed before image projection')
        descriptor=s['textDonor'] if kind=='text' else s['donors'][phase['carrier']];donor=text_donor if kind=='text' else donors[phase['carrier']];native_field=capture('textWriteFields' if kind=='text' else 'nativeH0Fields',x)
        vals={'native':x,'donor':donor,'receiverContract':current['contract']};fp={k:ProvidedFingerprint.content_sha256(identity(v) if k=='receiverContract' else payload(v)) for k,v in vals.items()};abi={k:{'shape':list(v.shape),'dtype':str(v.dtype),'stride':list(v.stride()),'storageOffset':v.storage_offset()} for k,v in vals.items() if k!='receiverContract'}
        metadata={**phase,'kind':kind,'earlierTextOutputFrameID':current.get('textFrameID')};frame=Frame.from_values(vals,kinds={k:'receiver-contract' if k=='receiverContract' else 'tensor' for k in vals},abis=abi,slot_fingerprints=fp,program_fingerprint=r['requestSHA256'],metadata=metadata)
        result,ev=registry.run(acts[kind].identity,'native-boundary-clamp',frame);value=result.values['consumed'];evidence=ev.to_dict()
        writes.append({**phase,'kind':kind,'nativeField':native_field,'donor':descriptor,'consumedPayloadSHA256':payload(value),'shape':list(value.shape),'dtype':str(value.dtype),
            'recipientStride':list(x.stride()),'consumedStride':list(value.stride()),'recipientStorageOffset':x.storage_offset(),'consumedStorageOffset':value.storage_offset(),'recipientStoragePointer':x.untyped_storage().data_ptr(),'donorStoragePointer':donor.untyped_storage().data_ptr(),'consumedStoragePointer':value.untyped_storage().data_ptr(),'evidence':evidence})
        frames.append({**metadata,'sourceFrame':frame.to_dict(),'outputFrame':result.to_dict()});counts[phase['cell']][kind+'Clamps']+=1
        if kind=='text':current['textFrameID']=evidence['output_frame_id']
        return value
    def image_pre(module,args):
        if len(args)!=1:raise ValueError('Image projection argument count differs')
        return (replace('image',args[0]),)
    def text_pre(module,args):capture('textHeadFields',args[0]);counts[phase['cell']]['textHeads']+=1
    def text_post(module,args,y):return replace('text',y)
    def block_pre(module,args,kwargs):
        hashes={k:payload(v) for k,v in kwargs.items() if isinstance(v,torch.Tensor)}
        if hashes!=s['expectedReceiverPrefix'] or prefix and prefix!=hashes:raise ValueError('Native receiver prefix differs')
        prefix.update(hashes)
    weight_names={v.data_ptr():n for n,v in transformer.named_parameters()};old_linear=torch.nn.functional.linear;old_attention=native.dispatch_attention_fn
    def linear(x,w,bias=None):
        name=weight_names.get(w.data_ptr())
        if name is None:raise ValueError('Unbound original affine weight')
        linears.append({**phase,'parameter':name,'inputShape':list(x.shape),'weightShape':list(w.shape),'nominalMACs':x.numel()//x.shape[-1]*w.numel()});return old_linear(x,w,bias)
    def dispatch(q,k,v,*args,**kwargs):
        attention.append({**phase,'queryShape':list(q.shape),'keyShape':list(k.shape),'valueShape':list(v.shape),'nominalQKAVMACs':2*q.shape[0]*q.shape[1]*k.shape[1]*q.shape[2]*q.shape[3]});return old_attention(q,k,v,*args,**kwargs)
    hooks=[transformer.transformer_blocks[0].attn.to_out[0].register_forward_pre_hook(image_pre),transformer.transformer_blocks[0].attn.to_add_out.register_forward_pre_hook(text_pre),transformer.transformer_blocks[0].attn.to_add_out.register_forward_hook(text_post),transformer.transformer_blocks[0].register_forward_pre_hook(block_pre,with_kwargs=True)]
    setup=time.perf_counter()-start;rng_cpu=torch.get_rng_state().clone();rng_cuda=[x.clone() for x in torch.cuda.get_rng_state_all()]
    try:
        torch.nn.functional.linear=linear;native.dispatch_attention_fn=dispatch
        with torch.inference_mode():
            for carrier in ['A','B']:
                phase.clear();phase.update(cell=carrier,receiver='R1',text='T0',carrier=carrier);counts[carrier]={'textHeads':0,'textClamps':0,'imageClamps':0}
                current['contract']={'receiver':s['receivers']['R1'],'textDonor':s['textDonor'],'common':s['common'],'clock':s['clock'],'parameterMapIdentity':parameter_id,'numericalProfile':numerical};current['textFrameID']=None
                if snapshot(torch)!=numerical:raise ValueError('Native numerical profile changed')
                torch.cuda.synchronize();tick=time.perf_counter();ct=time.process_time();ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True);ev0.record()
                with transformer.cache_context('cond'):velocity=transformer(hidden_states=z,timestep=clock,guidance=None,encoder_hidden_states=common['conditioning'],txt_ids=common['txt_ids'],img_ids=common['img_ids'],joint_attention_kwargs=None,return_dict=False)[0]
                ev1.record();torch.cuda.synchronize();timings.append({**phase,'wallSecondsInstrumented':time.perf_counter()-tick,'cpuSecondsInstrumented':time.process_time()-ct,'cudaEventIntervalSecondsInstrumented':ev0.elapsed_time(ev1)/1000})
                if list(velocity.shape)!=[1,1024,128] or velocity.dtype!=torch.bfloat16:raise ValueError('Native velocity ABI differs')
                fields[carrier]=retain(root/'arrays/velocity',carrier,velocity)
                if counts[carrier]!={'textHeads':1,'textClamps':1,'imageClamps':1}:raise ValueError('Two-boundary hook census differs')
                print('TEXT_CROSSING_CELL',carrier,fields[carrier]['payloadSHA256'],flush=True)
    finally:
        torch.nn.functional.linear=old_linear;native.dispatch_attention_fn=old_attention
        for hook in hooks:hook.remove()
    if census()!=before or buffers!={n:payload(v) for n,v in transformer.named_buffers()} or processors()!=s['attentionProcessors'] or input_census()!=input_before:raise ValueError('Original model/input closure changed')
    if not torch.equal(rng_cpu,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise ValueError('Consumer RNG closure changed')
    if len(linears)!=218 or len(attention)!=50:raise ValueError('Original computation census differs')
    assets_unchanged(assets);save(root/'velocity-seal.json',{'schema':'tce-text-crossing-velocity-seal-v1','requestSHA256':r['requestSHA256'],'fields':fields,'writes':writes,'frames':frames});seal=sha(root/'velocity-seal.json');print('TEXT_CROSSING_SEALED',seal,flush=True)
    targets=read(root/'targets.json');target_files.update(str(Path(x['path']).resolve()) for x in targets['fields'].values());scoring['open']=True
    old={k:load(x) for k,x in targets['fields'].items()};new={k:load(x) for k,x in fields.items()};derived=decompose(old,new);derived_fields={n:retain(root/'arrays/derived',n,x) for n,x in derived.items()}
    from source_response import mask
    selected=mask('bottom-right');accounts=[]
    with (out/'observations.jsonl').open('x') as stream:
        for scope,selection in [('all',np.ones(1024,bool)),('source-nonlocal',selected)]:accounts.append({'scope':scope,'queries':int(selection.sum()),**account({k:x[0,selection] for k,x in derived.items()})})
        for index in range(1024):stream.write(json.dumps({'imageQueryIndex':index,'sourceNonlocal':bool(selected[index]),**account({k:x[0,index] for k,x in derived.items()})},allow_nan=False)+'\n')
    if sha(root/'velocity-seal.json')!=seal:raise ValueError('New velocity release changed')
    save(out/'derived-fields.json',derived_fields);save(out/'execution.json',{'schema':'tce-text-crossing-execution-v1','loadedParameters':loaded,'parameterMapIdentity':parameter_id,'assets':assets,'numericalProfile':numerical,'attentionProcessors':processors(),'parameterIdentityVersionClosure':True,'bufferPayloadClosure':True,'inputPayloadVersionClosure':True,'consumerRNGClosure':True,
        'linears':linears,'attentionCalls':attention,'nativeFields':captures,'receiverPrefixHashes':prefix,'counts':counts,'timings':timings,'setupSecondsAfterImports':setup,'workerWallSecondsAfterImports':time.perf_counter()-start,'workerCPUSecondsAfterImports':time.process_time()-cpu,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),
        'fileQuarantine':{'primaryAllowedReadFiles':sorted(allowed),'deniedCanaries':s['deniedCanaries'],'postSealVelocityReadFiles':sorted(target_files),'releaseSHA256BeforeTargetReads':seal,'scope':'Trusted-code audit hook, not adversarial sandbox. Only old velocity payloads opened after new seal.'}})
    import shutil
    for n in ['request.json','inputs.json','targets.json','resource-basis.json','frozen-plan.md','velocity-seal.json']:shutil.copyfile(root/n,out/n)
    save(out/'report.json',{'schema':'tce-text-crossing-result-v1','requestSHA256':r['requestSHA256'],'velocitySealSHA256':seal,'accounts':accounts,'observations':1024,'nominalAffineMACs':sum(x['nominalMACs'] for x in linears),'nominalQKAVMACs':sum(x['nominalQKAVMACs'] for x in attention),
        'work':{'modelLoads':1,'originalTransformerCalls':2,'recipientH0Computations':2,'recipientTextProjectionComputations':2,'imageClamps':2,'textClamps':2,'affineCalls':218,'attentionDispatches':50,'newDonorFormation':0,'newEncoderCalls':0,'newFits':0,'newSchedulerCommits':0,'newVAECalls':0,'newImages':0},
        'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed','limits':['Supplied-carrier hybrids, one ordered co-carrier contrast/two dependent outcomes; no natural-manifold or universal causal-share claim.',
            'Remaining receiver difference includes later text-image feedback. Signed terms may cancel; no additive causal percentages.',
            'All original transformer/clock/text/image work is paid, including discarded native image read and text projection. Old donors/conditioning/velocities remain paid.',
            'No fit, cheap receiver-conditioned operator, full image, arbitrary-prompt coverage or zero-neural generation is established.',
            'Instrumented wall/CPU/CUDA-event intervals include captures/hash/hooks; no matched serving speedup claim.']});print('TEXT_CROSSING_DONE',flush=True)
