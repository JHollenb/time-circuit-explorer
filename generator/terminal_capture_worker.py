"""Paid native suffix capture: post-joint0 → complete single19 input and velocity."""
from pathlib import Path
import sys
import time
from encoder_atoms import read,save,sha,identity


def execute(root):
    import torch
    import diffusers
    from mdb.titan import Frame,ActSpec,ActRegistry,ProvidedFingerprint
    from saturn_pub.store import LocalStore
    from terminal_capture_cut import persist
    from paired_fields import payload,retain
    from flux_clock_study import verify,source_checks,assets_unchanged
    from flux_clock import header,snapshot
    from complete_masked_study import profile
    from projected_complement import load_selected
    from terminal_capture_guard import quarantine
    import diffusers.models.transformers.transformer_flux2 as native
    root=Path(root).resolve();r,s=read(root/'request.json'),read(root/'inputs.json');verify(root,r)
    out=root/'response-result';out.mkdir();started=time.perf_counter();cpu=time.process_time()
    source=s['sourceCut'];allowed=[str(Path(s['modelRoot'])/n) for n in s['imageAssets']]
    allowed+=[source['cutFile']['path']]+[n for page in source['pages'].values() for n in page['files']]
    accesses=quarantine(root,allowed,s['deniedNumericalInputs'],root/'terminal-store')
    numerical=profile(torch,4)
    if numerical!=s['numericalProfile']:raise ValueError('Native numerical profile differs')
    model,assets=source_checks(s,diffusers);config=read(model/'transformer/config.json')
    if config!=s['modelConfig']:raise ValueError('Original transformer configuration differs')
    checkpoint=model/'transformer/diffusion_pytorch_model.safetensors';entries,hr=header(checkpoint)
    original=LocalStore(source['store']).load(source['cutID'],device='cuda')
    if original.fingerprint!=source['cutID'] or original.manifest()!=source['cut']['manifest']:raise ValueError('Source complete StateCut differs')
    state=original.payload
    if (state['step'],state['phase'],state['cursor'])!=(2,'joint',1):raise ValueError('Declared prefix source differs')
    original_state_descriptor=identity(__import__('saturn_pub.values',fromlist=['describe']).describe(state))
    recipes={x['source']['tensor']:x['source'] for x in s['originalParameterArtifacts']}
    loaded,operations,attention,records,timings=[],[],[],[],[];stage='';active_weights={}
    old_linear=torch.nn.functional.linear;old_dispatch=native.dispatch_attention_fn
    rng,rng_cuda=torch.get_rng_state(),torch.cuda.get_rng_state_all();torch.cuda.reset_peak_memory_stats()
    def linear(x,w,bias=None):
        name=active_weights.get(w.data_ptr())
        if name is None or bias is not None:raise ValueError('Undeclared native affine')
        operations.append({'stage':stage,'parameter':name,'inputShape':list(x.shape),'weightShape':list(w.shape),
            'nominalMACs':x.numel()//x.shape[-1]*w.numel(),'dtype':str(x.dtype),'device':x.device.type})
        return old_linear(x,w,bias)
    def dispatch(q,k,v,*args,**kwargs):
        attention.append({'stage':stage,'QShape':list(q.shape),'KShape':list(k.shape),'VShape':list(v.shape),
            'nominalQKAndAVMACs':2*q.numel()*k.shape[1],'backend':kwargs.get('backend'),'maskIsNone':kwargs.get('attn_mask') is None})
        return old_dispatch(q,k,v,*args,**kwargs)
    def module(kind,index):
        prefix=('transformer_blocks.' if kind=='joint' else 'single_transformer_blocks.')+str(index)+'.'
        names=sorted(n for n in entries if n.startswith(prefix));weights,receipt=load_selected(checkpoint,names,entries)
        for x in receipt:
            rec=recipes[x['name']]
            if x['payloadSHA256']!=rec['payload_sha256'] or x['shape']!=rec['shape'] or x['dataOffsets']!=rec['data_offsets']:raise ValueError('Native parameter recipe differs')
        weights={n:v.to('cuda') for n,v in weights.items()}
        with torch.device('meta'):
            block=(native.Flux2TransformerBlock if kind=='joint' else native.Flux2SingleTransformerBlock)(dim=3072,num_attention_heads=24,attention_head_dim=128,mlp_ratio=3.,eps=1e-6,bias=False)
        block.load_state_dict({n[len(prefix):]:v for n,v in weights.items()},strict=True,assign=True);block.eval().requires_grad_(False)
        actual={prefix+n:v for n,v in block.named_parameters()}
        if set(actual)!=set(names) or any(v.device.type!='cuda' or v.dtype!=torch.bfloat16 for v in actual.values()):raise ValueError('Original module parameter closure/device differs')
        loaded.extend(receipt);return block,actual
    def run(name,inputs,operation,outputs,weights):
        nonlocal stage,active_weights
        registry=ActRegistry()
        stage=name;active_weights={v.data_ptr():n for n,v in weights.items()};inputs={**inputs,**weights}
        initial={n:payload(v) for n,v in inputs.items()}
        frame=Frame.from_values(inputs,kinds={n:'tensor' for n in inputs},
            abis={n:{'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype)} for n,v in inputs.items()},
            slot_fingerprints={n:ProvidedFingerprint.content_sha256(payload(v)) for n,v in inputs.items()},
            program_fingerprint=r['requestSHA256'],metadata={'phase':name,'originalParameterNames':list(weights)})
        abis={n:{'shape':shape,'dtype':'torch.bfloat16'} for n,shape in outputs.items()}
        spec=ActSpec('tce.terminal_capture_'+name,'v1',reads=tuple(inputs),writes=tuple(outputs),write_kinds={n:'tensor' for n in outputs},write_abis=abis,
            numerical_contract='Original native BF16 modules; paid prefix capture, not reduced or zero-neural execution')
        got={}
        def act(v):
            torch.cuda.synchronize();tick=time.perf_counter();result=operation(v);torch.cuda.synchronize()
            timings.append({'stage':name,'secondsBeforeFingerprintCapture':time.perf_counter()-tick});got.update(result)
            if {n:{'shape':list(t.shape),'dtype':str(t.dtype)} for n,t in result.items()}!=abis:raise ValueError('Native output ABI differs')
            return result,{'outputPayloads':{n:payload(t) for n,t in result.items()}}
        registry.register(spec,name,act);after,ev=registry.run(spec.identity,name,frame)
        if initial!={n:payload(v) for n,v in inputs.items()}:raise ValueError('Native input/parameter mutation')
        records.append({'stage':name,'sourceFrame':frame.to_dict(),'outputFrame':after.to_dict(),'evidence':ev.to_dict()})
        return got
    image,text=state['image'],state['text'];c=state['context'];captured={}
    try:
        torch.nn.functional.linear=linear;native.dispatch_attention_fn=dispatch
        with torch.inference_mode():
            for index in range(1,5):
                block,weights=module('joint',index)
                got=run('joint'+str(index),dict(image=image,text=text,imgMod=c['img_mod'],txtMod=c['txt_mod'],cosine=c['rope'][0],sine=c['rope'][1]),
                    lambda v:dict(zip(['text','image'],block(hidden_states=v['image'],encoder_hidden_states=v['text'],temb_mod_img=v['imgMod'],temb_mod_txt=v['txtMod'],image_rotary_emb=(v['cosine'],v['sine']),joint_attention_kwargs=None))),
                    {'text':[1,512,3072],'image':[1,1024,3072]},weights)
                image,text=got['image'],got['text'];del got,block,weights
            combined=torch.cat([text,image],1)
            for index in range(20):
                if index==19:
                    complete={**state,'image':combined[:,512:],'text':combined[:,:512],'phase':'single','cursor':19}
                    captured['source']=persist(complete,original,root/'terminal-store',r['requestSHA256'],'single',19,original.fingerprint)
                    captured['sourceFields']={'combined':retain(root/'arrays/source','combined',combined)}
                block,weights=module('single',index)
                got=run('single'+str(index),dict(rows=combined,mod=c['single_mod'],cosine=c['rope'][0],sine=c['rope'][1]),
                    lambda v:{'combined':block(hidden_states=v['rows'],encoder_hidden_states=None,temb_mod=v['mod'],image_rotary_emb=(v['cosine'],v['sine']),joint_attention_kwargs=None)},
                    {'combined':[1,1536,3072]},weights)
                combined=got['combined'];del got,block,weights
            target={**state,'image':combined[:,512:],'text':combined[:,:512],'phase':'readout','cursor':20}
            captured['target']=persist(target,original,root/'terminal-store',r['requestSHA256'],'readout',20,captured['source']['cutID'])
            captured['targetFields']={'image':retain(root/'arrays/target','image',target['image']),'text':retain(root/'arrays/target','text',target['text'])}
            names=['norm_out.linear.weight','proj_out.weight'];weights,receipt=load_selected(checkpoint,names,entries);loaded.extend(receipt)
            for x in receipt:
                rec=recipes[x['name']]
                if x['payloadSHA256']!=rec['payload_sha256'] or x['shape']!=rec['shape'] or x['dataOffsets']!=rec['data_offsets']:raise ValueError('Final-head recipe differs')
            with torch.device('meta'):
                norm=native.AdaLayerNormContinuous(3072,3072,elementwise_affine=False,eps=1e-6,bias=False)
                projection=torch.nn.Linear(3072,128,bias=False)
            norm.load_state_dict({'linear.weight':weights[names[0]]},strict=True,assign=True);projection.load_state_dict({'weight':weights[names[1]]},strict=True,assign=True)
            norm.eval().requires_grad_(False);projection.eval().requires_grad_(False)
            weights={names[0]:norm.linear.weight,names[1]:projection.weight}
            if any(v.device.type!='cuda' or v.dtype!=torch.bfloat16 for v in weights.values()):raise ValueError('Final head device/dtype differs')
            got=run('velocity',dict(image=target['image'],temb=c['temb']),lambda v:{'velocity':projection(norm(v['image'],v['temb']))},{'velocity':[1,1024,128]},weights)
            captured['targetFields']['velocity']=retain(root/'arrays/target','velocity',got['velocity'])
    finally:
        torch.nn.functional.linear=old_linear;native.dispatch_attention_fn=old_dispatch
    if original_state_descriptor!=identity(__import__('saturn_pub.values',fromlist=['describe']).describe(state)) or snapshot(torch)!=numerical:raise ValueError('Source/numerical closure differs')
    if not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise ValueError('RNG closure differs')
    if len(records)!=25 or len(attention)!=24 or len(operations)!=90:raise ValueError('Declared native prefix/target work census differs')
    if sum(x['nominalMACs'] for x in operations)!=r['plannedNativeAffineMACs'] or sum(x['nominalQKAndAVMACs'] for x in attention)!=r['plannedNativeAttentionMACs']:
        raise ValueError('Actual native capture matrix work differs')
    assets_unchanged(assets)
    save(out/'capture.json',captured);save(out/'capture-seal.json',{'requestSHA256':r['requestSHA256'],'captureSHA256':sha(out/'capture.json')})
    save(out/'execution.json',{'loadedParameters':loaded,'operations':operations,'attention':attention,'typedActs':records,'timings':timings,
        'nativeAffineMACs':sum(x['nominalMACs'] for x in operations),'nativeAttentionMACs':sum(x['nominalQKAndAVMACs'] for x in attention),
        'sourceInputParameterRNGClosure':True,'header':hr,'assets':assets,'numericalProfile':numerical,
        'wallSecondsAfterImports':time.perf_counter()-started,'CPUSecondsAfterImports':time.process_time()-cpu,
        'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),
        'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':s['deniedNumericalInputs'],**accesses,
            'outputStorePublicationMode':'Fresh local-only content-addressed links; no shared pool hydration or general root read exemption.'}})
    save(out/'report.json',{'schema':'tce-terminal-native-capture-v1','requestSHA256':r['requestSHA256'],'captureSealSHA256':sha(out/'capture-seal.json'),
        'nativeJointBlocks':4,'nativeSingleBlocks':20,'nativeVelocityHeads':1,'nativeAffines':90,'nativeAttentionCalls':24,
        'nativeAffineMACs':sum(x['nominalMACs'] for x in operations),'nativeAttentionMACs':sum(x['nominalQKAndAVMACs'] for x in attention),
        'scope':'Paid native suffix capture from supplied post-joint0 P state. Complete terminal single19 source/target cuts and velocity persisted. No reduced producer, encoder, learned clock, scheduler, VAE or image; not zero-neural execution.',
        'mechanics_status':'complete','instrument_status':'native-capture-only','terminal_status':'unassessed'})
    for n in ['request.json','inputs.json','parameter-artifacts.json','resource-basis.json','frozen-plan.md']:__import__('shutil').copyfile(root/n,out/n)
    print('TERMINAL_NATIVE_CAPTURE_COMPLETE',flush=True)


if __name__=='__main__':execute(sys.argv[1])
