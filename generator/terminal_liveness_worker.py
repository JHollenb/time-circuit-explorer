"""One target-isolated physically reduced terminal block and its own final head."""
import sys
import time
from pathlib import Path
from encoder_atoms import read,save,sha,identity


def execute(root):
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from mdb.titan import Frame,ActSpec,ActRegistry,ProvidedFingerprint
    from flux_clock_study import verify,source_checks,assets_unchanged
    from complete_masked_study import profile
    from flux_clock import header,snapshot
    from projected_complement import load_selected
    from paired_fields import payload,retain
    from terminal_capture_guard_v2 import quarantine
    from terminal_liveness import PARAMETERS,slices,compute,account,mechanics
    root=Path(root).resolve();r,s=read(root/'request.json'),read(root/'inputs.json');verify(root,r)
    started=time.perf_counter();cpu=time.process_time();out=root/'response-result';out.mkdir()
    save(out/'toy-mechanics.json',mechanics())
    source=s['sourceCut'];allowed=[str(Path(s['modelRoot'])/n) for n in s['imageAssets']]
    allowed += [source['cutFile']['path']]+[n for page in source['pages'].values() for n in page['files']]
    accesses=quarantine(root,allowed,s['deniedNumericalInputs'],root/'unused-output-store')
    numerical=profile(torch,4)
    if numerical!=s['numericalProfile']:raise ValueError('Original numeric policy differs')
    model,assets=source_checks(s,diffusers);checkpoint=model/'transformer/diffusion_pytorch_model.safetensors';entries,hr=header(checkpoint)
    cut=LocalStore(source['store']).load(source['cutID'],device='cuda')
    if cut.manifest()!=source['cut']['manifest'] or cut.fingerprint!=source['cutID']:raise ValueError('Complete source manifest differs')
    state=cut.payload;original_descriptor=identity(describe(state))
    if (state['step'],state['phase'],state['cursor'])!=(2,'single',19):raise ValueError('Terminal source clock differs')
    context=state['context'];rows=torch.cat([state['text'],state['image']],1)
    if list(rows.shape)!=[1,1536,3072] or list(rows.stride())!=[4718592,3072,1] or payload(rows)!=s['sourceCombinedPayloadSHA256']:
        raise ValueError('Native incoming combined residual ABI/bytes differ')
    original,loaded=load_selected(checkpoint,PARAMETERS,entries);recipes={x['source']['tensor']:x['source'] for x in s['originalParameterArtifacts']}
    for item in loaded:
        rec=recipes[item['name']]
        if item['payloadSHA256']!=rec['payload_sha256'] or item['shape']!=rec['shape'] or item['dataOffsets']!=rec['data_offsets']:raise ValueError('Original map recipe differs')
    pre='single_transformer_blocks.19.attn.'
    weights={**slices(original[pre+'to_qkv_mlp_proj.weight'],3072,9216),'qNorm':original[pre+'norm_q.weight'],'kNorm':original[pre+'norm_k.weight'],'out':original[pre+'to_out.weight']}
    physical=[{'name':n,'shape':list(v.shape),'stride':list(v.stride()),'payloadSHA256':payload(v),'bytes':v.numel()*v.element_size()} for n,v in weights.items()]
    with torch.device('meta'):
        norm=native.AdaLayerNormContinuous(3072,3072,elementwise_affine=False,eps=1e-6,bias=False);projection=torch.nn.Linear(3072,128,bias=False)
    norm.load_state_dict({'linear.weight':original['norm_out.linear.weight']},strict=True,assign=True);projection.load_state_dict({'weight':original['proj_out.weight']},strict=True,assign=True)
    norm.eval().requires_grad_(False);projection.eval().requires_grad_(False)
    head={'norm_out.linear.weight':norm.linear.weight,'proj_out.weight':projection.weight};original_payloads={n:payload(v) for n,v in original.items()}
    ops,att,acts,timings=[],[],[],[];stage='';active={};old_linear=torch.nn.functional.linear;old_dispatch=native.dispatch_attention_fn
    rng,rng_cuda=torch.get_rng_state(),torch.cuda.get_rng_state_all();torch.cuda.reset_peak_memory_stats()
    def linear(x,w,bias=None):
        name=active.get(w.data_ptr())
        if name is None or bias is not None:raise ValueError('Unaccounted terminal affine')
        ops.append({'stage':stage,'parameter':name,'inputShape':list(x.shape),'inputStride':list(x.stride()),'weightShape':list(w.shape),'weightStride':list(w.stride()),
                    'nominalMACs':x.numel()//x.shape[-1]*w.numel(),'dtype':str(x.dtype),'device':x.device.type})
        return old_linear(x,w,bias)
    def attention(q,k,v,**kwargs):
        att.append({'stage':stage,'QShape':list(q.shape),'KShape':list(k.shape),'VShape':list(v.shape),'QStride':list(q.stride()),'KStride':list(k.stride()),'VStride':list(v.stride()),
                    'nominalQKAndAVMACs':2*q.numel()*k.shape[1],'backend':kwargs.get('backend'),'maskIsNone':kwargs.get('attn_mask') is None})
        return old_dispatch(q,k,v,**kwargs)
    def run(name,inputs,operation,outputs):
        nonlocal stage,active
        stage=name;active={v.data_ptr():n for n,v in inputs.items()};registry=ActRegistry()
        before={n:payload(v) for n,v in inputs.items()}
        frame=Frame.from_values(inputs,kinds={n:'tensor' for n in inputs},abis={n:{'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype)} for n,v in inputs.items()},
            slot_fingerprints={n:ProvidedFingerprint.content_sha256(h) for n,h in before.items()},program_fingerprint=r['requestSHA256'],metadata={'phase':name})
        abis={n:{'shape':shape,'dtype':'torch.bfloat16'} for n,shape in outputs.items()};spec=ActSpec('tce.terminal_liveness_'+name,'v1',reads=tuple(inputs),writes=tuple(outputs),
            write_kinds={n:'tensor' for n in outputs},write_abis=abis,numerical_contract='Physically reduced native learned BF16 arithmetic; geometry drift measured after seal, no neural-free claim')
        got={}
        def act(values):
            torch.cuda.synchronize();tick=time.perf_counter();result=operation(values);torch.cuda.synchronize()
            timings.append({'stage':name,'synchronizedSecondsBeforeFingerprint':time.perf_counter()-tick});got.update(result)
            if {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in result.items()}!=abis:raise ValueError('Candidate output ABI differs')
            return result,{'outputPayloads':{n:payload(v) for n,v in result.items()}}
        registry.register(spec,name,act);after,ev=registry.run(spec.identity,name,frame)
        if before!={n:payload(v) for n,v in inputs.items()}:raise ValueError('Candidate source/map mutation')
        acts.append({'stage':name,'sourceFrame':frame.to_dict(),'outputFrame':after.to_dict(),'evidence':ev.to_dict()})
        return got
    try:
        torch.nn.functional.linear=linear
        with torch.inference_mode():
            got=run('image',{'rows':rows,'mod':context['single_mod'],'cosine':context['rope'][0],'sine':context['rope'][1],**weights},
                lambda x:{'ownImage':compute(x['rows'],x['mod'],x['cosine'],x['sine'],{n:x[n] for n in weights},torch.nn.functional.linear,attention,native)},{'ownImage':[1,1024,3072]})
            image=got['ownImage']
            if list(image.stride())!=s['nativeHeadInputABI']['stride']:raise ValueError('Consumed image-view ABI differs')
            got=run('velocity',{'image':image,'temb':context['temb'],**head},lambda x:{'ownVelocity':projection(norm(x['image'],x['temb']))},{'ownVelocity':[1,1024,128]})
            fields={'image':retain(root/'arrays/candidate','image',image),'velocity':retain(root/'arrays/candidate','velocity',got['ownVelocity'])}
    finally:torch.nn.functional.linear=old_linear
    if identity(describe(state))!=original_descriptor or original_payloads!={n:payload(v) for n,v in original.items()} or snapshot(torch)!=numerical:raise ValueError('Source/map/numeric closure differs')
    if not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):raise ValueError('RNG closure differs')
    cost=account(512,1024,3072,9216);affine=sum(x['nominalMACs'] for x in ops);attention_mac=sum(x['nominalQKAndAVMACs'] for x in att)
    if len(ops)!=5 or len(att)!=1 or affine!=cost['candidateAffine']+421527552 or attention_mac!=cost['candidateAttention'] or affine!=r['candidateAffineMACs'] or attention_mac!=r['candidateAttentionMACs']:
        raise ValueError('Physical reduced work ledger differs')
    assets_unchanged(assets)
    save(out/'candidate.json',{'fields':fields,'sourceCutID':cut.fingerprint,'imageOutputStride':list(image.stride()),'scope':'Own complete image state consumed by original head; no terminal text output produced.'})
    save(out/'execution.json',{'loadedParameters':loaded,'physicalMaps':physical,'weightRowIntervals':{'KV':[[3072,9216]],'QGU':[[0,3072],[9216,27648]]},
        'operations':ops,'attention':att,'typedActs':acts,'timings':timings,'affineMACs':affine,'attentionMACs':attention_mac,'totalMatrixMACs':affine+attention_mac,
        'originalParameterBytes':sum(x['bytes'] for x in loaded),'numericalProfile':numerical,'sourceInputParameterRNGClosure':True,'header':hr,'assets':assets,
        'wallSecondsAfterImports':time.perf_counter()-started,'CPUSecondsAfterImports':time.process_time()-cpu,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),
        'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':s['deniedNumericalInputs'],**accesses}})
    save(out/'candidate-seal.json',{'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')})
    print('TERMINAL_LIVENESS_CANDIDATE_SEALED',sha(out/'candidate-seal.json'),flush=True)


if __name__=='__main__':execute(sys.argv[1])
