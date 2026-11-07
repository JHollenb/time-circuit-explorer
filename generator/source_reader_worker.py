"""Source-state interface → own native Q/K → deferred V → own continuation."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha
from paired_fields import retain, payload
from projected_complement import load_selected
from propagated_context import P1, RETURN_PARAMETERS, NEXT_PARAMETERS, normalized, next_query
from image_dual import consume_head
from source_reader import derive, project_norm
from deferred_values import form

VALUE_PARAMETERS=(P1+'attn.to_v.weight',P1+'attn.add_v_proj.weight')
KEY_PARAMETERS=tuple(P1+'attn.'+n+'.weight' for n in ['to_k','add_k_proj','norm_k','norm_added_k'])
QUERY_PARAMETERS=tuple(P1+'attn.'+n+'.weight' for n in ['add_q_proj','norm_added_q'])
PARAMETERS=KEY_PARAMETERS+QUERY_PARAMETERS+VALUE_PARAMETERS+RETURN_PARAMETERS+NEXT_PARAMETERS


def execute(root):
    import torch
    import diffusers
    from mdb.titan import Frame, ActSpec, ActRegistry, ProvidedFingerprint
    from complete_masked_study import profile
    from flux_clock_study import verify, source_checks, assets_unchanged
    from flux_clock import header, snapshot
    from sparse_text_kv import quarantine
    from selected_source_v2 import load_pages
    import diffusers.models.transformers.transformer_flux2 as native
    root=Path(root).resolve();r,s=read(root/'request.json'),read(root/'inputs.json');verify(root,r)
    out=root/'response-result';out.mkdir()
    cuts={n:{**s['sourceCuts'][n],'pages':{k:v for k,v in s['sourceCuts'][n]['pages'].items()
          if k in (['image','text','context'] if n=='Q' else ['text','context'])}} for n in ['P','Q']}
    allowed=[str(Path(s['modelRoot'])/n) for n in s['imageAssets']]
    for cut in cuts.values():
        allowed += [cut['cutFile']['path']]+[p for page in cut['pages'].values() for p in page['files']]
    denied=s['deniedNumericalInputs']
    if set(denied)&set(allowed):raise ValueError('Allowed/denied numerical inputs overlap')
    accesses=quarantine(root,allowed,denied)
    started,cpu=time.perf_counter(),time.process_time();numerical=profile(torch,4)
    if numerical!=s['numericalProfile']:raise ValueError('Original finite consumer profile differs')
    model,assets=source_checks(s,diffusers);checkpoint=model/'transformer/diffusion_pytorch_model.safetensors'
    entries,header_receipt=header(checkpoint)
    if any(name.removesuffix('.weight')+'.bias' in entries for name in VALUE_PARAMETERS):
        raise ValueError('Source value path unexpectedly biased')
    weights,loaded=load_selected(checkpoint,PARAMETERS,entries)
    recipes={x['source']['tensor']:x['source'] for x in s['originalParameterArtifacts']}
    for x in loaded:
        rec=recipes[x['name']]
        if x['payloadSHA256']!=rec['payload_sha256'] or x['shape']!=rec['shape'] or x['dataOffsets']!=rec['data_offsets']:
            raise ValueError('Original parameter payload/header differs')
    before={n:(v.data_ptr(),v._version,payload(v)) for n,v in weights.items()}
    rng,rng_cuda=torch.get_rng_state(),torch.cuda.get_rng_state_all();torch.cuda.reset_peak_memory_stats()
    operations,timings,records=[],[],{};registry=ActRegistry();stage=''
    def mm(a,b,label):
        if a.dtype!=torch.float64 or b.dtype!=torch.float64 or a.device.type!='cpu' or b.device.type!='cpu':
            raise ValueError('Declared CPU FP64 contraction differs')
        result=a@b;operations.append({'stage':stage,'operation':label,'inputShape':list(a.shape),'rightShape':list(b.shape),
            'dtype':str(a.dtype),'device':'cpu','nominalMACs':a.numel()*b.shape[-1]});return result
    original_linear=torch.nn.functional.linear
    def linear(x,w,bias=None):
        matches=[n for n,v in weights.items() if v is w]
        if len(matches)!=1 or matches[0] in VALUE_PARAMETERS or bias is not None:
            raise ValueError('Undeclared original affine or full value projection')
        operations.append({'stage':stage,'operation':'original-affine','parameter':matches[0],
            'inputShape':list(x.shape),'weightShape':list(w.shape),'dtype':str(x.dtype),'device':x.device.type,
            'nominalMACs':x.numel()//x.shape[-1]*w.numel()});return original_linear(x,w,bias)
    abi=lambda shape,dtype='torch.float64':{'shape':shape,'dtype':dtype}
    def run(name,inputs,operation,abis,metadata):
        nonlocal stage
        stage=name;initial={n:payload(v) for n,v in inputs.items()}
        frame=Frame.from_values(inputs,kinds={n:'tensor' for n in inputs},
            abis={n:{'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype)} for n,v in inputs.items()},
            slot_fingerprints={n:ProvidedFingerprint.content_sha256(payload(v)) for n,v in inputs.items()},
            program_fingerprint=r['requestSHA256'],metadata=metadata)
        spec=ActSpec('tce.source_reader_'+name,'v1',reads=tuple(inputs),writes=tuple(abis),
            write_kinds={n:'tensor' for n in abis},write_abis=abis,
            numerical_contract='native-BF16-source-ModLN/Q/K/RMS/RoPE/CPU-FP64-QK-deferredV/native-finite-consumer')
        captures={}
        def act(v):
            torch.cuda.synchronize();tick=time.perf_counter();got=operation(v);torch.cuda.synchronize()
            timings.append({'stage':name,'arithmeticSecondsBeforeCapture':time.perf_counter()-tick})
            if any({'shape':list(got[n].shape),'dtype':str(got[n].dtype)}!=a for n,a in abis.items()):
                raise ValueError('Actual explicit output ABI differs')
            captures.update({n:retain(root/'arrays/actor'/name,n,x) for n,x in got.items()})
            return {n:got[n] for n in abis},{'capturedPayloads':{n:captures[n]['payloadSHA256'] for n in abis}}
        registry.register(spec,name,act);after,evidence=registry.run(spec.identity,name,frame)
        if initial!={n:payload(v) for n,v in inputs.items()}:raise ValueError('Typed input mutated')
        records[name]={'fields':captures,'sourceFrame':frame.to_dict(),'outputFrame':after.to_dict(),'evidence':evidence.to_dict()}
        return after
    try:
        torch.nn.functional.linear=linear
        with torch.inference_mode():
            state={n:load_pages(cut) for n,cut in cuts.items()}
            normal={}
            for name,role,slot,mod in [('image_modln','Q','image','img_mod'),('text_modln','Q','text','txt_mod'),('p_text_modln','P','text','txt_mod')]:
                rows=state[role][slot];normal[name]=run(name,{'rows':rows,'mod':state[role]['context'][mod]},
                    lambda v:{'x':normalized(v['rows'],v['mod'])},
                    {'x':abi(list(rows.shape),'torch.bfloat16')},{'sourceCut':cuts[role]['cutID'],'sourceSlot':slot,'modSlot':mod})
            source_inputs=run('source_inputs',dict(text=state['P']['text'],mod=state['P']['context']['txt_mod'],
                    cosine=state['P']['context']['rope'][0],sine=state['P']['context']['rope'][1]),
                lambda v:derive(v['text'],v['mod'],v['cosine'],v['sine'],r['support']),
                {'Ptext':abi([1,512,3072],'torch.bfloat16'),'R':abi([1,4,3072],'torch.bfloat16'),
                    'txtMod':abi([1,18432],'torch.bfloat16'),
                    **{n:abi([1536,128],'torch.float32') for n in ['fullRopeCos','fullRopeSin']},
                    **{n:abi([4,128],'torch.float32') for n in ['ropeCos','ropeSin']}},
                {'sourceCut':cuts['P']['cutID'],'support':r['support']})
            def splice(v):
                x=v['Qtext'].clone();x[:,r['support']]=v['Ptext'][:,r['support']];return {'x':x}
            text=run('splice',{'Qtext':normal['text_modln'].values['x'],'Ptext':normal['p_text_modln'].values['x']},splice,
                {'x':abi([1,512,3072],'torch.bfloat16')},{'QtextFrame':normal['text_modln'].frame_id,
                'PtextFrame':normal['p_text_modln'].frame_id,'support':r['support']})
            def query(v):
                x=normalized(v['R'],v['txtMod'])
                projected,normalized_q=project_norm(x,v[QUERY_PARAMETERS[0]],v[QUERY_PARAMETERS[1]],24)
                q=native.apply_rotary_emb(normalized_q,(v['ropeCos'],v['ropeSin']),sequence_dim=1)
                return dict(queryModLN=x,queryProjected=projected,queryNormalized=normalized_q,Q=q)
            query_inputs={n:source_inputs.values[n] for n in ['R','txtMod','ropeCos','ropeSin']}
            query_inputs.update({n:weights[n] for n in QUERY_PARAMETERS})
            q=run('query',query_inputs,query,{'queryModLN':abi([1,4,3072],'torch.bfloat16'),
                **{n:abi([1,4,24,128],'torch.bfloat16') for n in ['queryProjected','queryNormalized','Q']}},
                {'sourceInputsFrame':source_inputs.frame_id,'queryModLNGeometry':[1,4,3072]})
            def keys(v):
                ip,inn=project_norm(v['imageInputs'],v[KEY_PARAMETERS[0]],v[KEY_PARAMETERS[2]],24)
                tp,tn=project_norm(v['textInputs'],v[KEY_PARAMETERS[1]],v[KEY_PARAMETERS[3]],24)
                k=native.apply_rotary_emb(torch.cat((tn,inn),1),(v['cosine'],v['sine']),sequence_dim=1)
                return dict(imageProjected=ip,imageNormalized=inn,textProjected=tp,textNormalized=tn,
                            K=k,textK=k[:,:512],imageK=k[:,512:])
            key_inputs=dict(imageInputs=normal['image_modln'].values['x'],textInputs=text.values['x'],
                cosine=state['Q']['context']['rope'][0],sine=state['Q']['context']['rope'][1],**{n:weights[n] for n in KEY_PARAMETERS})
            k=run('keys',key_inputs,keys,{
                **{n:abi([1,1024,24,128],'torch.bfloat16') for n in ['imageProjected','imageNormalized','imageK']},
                **{n:abi([1,512,24,128],'torch.bfloat16') for n in ['textProjected','textNormalized','textK']},
                'K':abi([1,1536,24,128],'torch.bfloat16')},
                {'imageModLNFrame':normal['image_modln'].frame_id,'splicedTextFrame':text.frame_id,'sourceCut':cuts['Q']['cutID']})
            def allocate(v):
                logits=mm(v['Q'][0].cpu().double().permute(1,0,2),v['K'][0].cpu().double().permute(1,2,0),'live-QK')/128**.5
                peak=logits.max(-1,keepdim=True).values;w=torch.exp(logits-peak);partition=w.sum(-1,keepdim=True)
                probability=w/partition
                if not bool(torch.isfinite(probability).all() and (partition>0).all()):raise ValueError('Invalid live allocation')
                return dict(logits=logits,peak=peak,partition=partition,probability=probability)
            abis={n:abi([24,4,1536]) for n in ['logits','probability']}
            abis.update(peak=abi([24,4,1]),partition=abi([24,4,1]))
            allocation=run('allocate',dict(Q=q.values['Q'],K=k.values['K']),allocate,abis,
                {'queryFrame':q.frame_id,'keyFrame':k.frame_id})
            values=dict(probability=allocation.values['probability'],textInputs=text.values['x'],imageInputs=normal['image_modln'].values['x'],
                **{n:weights[n] for n in VALUE_PARAMETERS})
            def value(v):
                result=form(v['probability'],v['textInputs'][0].cpu().double(),v['imageInputs'][0].cpu().double(),
                    v[VALUE_PARAMETERS[1]].cpu().double(),v[VALUE_PARAMETERS[0]].cpu().double(),mm)
                result['H']=result['H_FP64'].permute(1,0,2).reshape(1,4,3072).float().bfloat16().to('cuda');return result
            abis={n:abi([24,4,3072]) for n in ['weightedText','weightedImage']}
            abis.update({n:abi([24,4,128]) for n in ['textContribution','imageContribution','H_FP64']})
            abis['H']=abi([1,4,3072],'torch.bfloat16')
            head=run('value',values,value,abis,{'allocationFrame':allocation.frame_id,'splicedTextFrame':text.frame_id,
                'imageModLNFrame':normal['image_modln'].frame_id,'admissionCast':'CPU FP64→FP32→BF16 RNE→CUDA'})
            source=run('source',dict(H=head.values['H'],R=source_inputs.values['R'],txtMod=source_inputs.values['txtMod'],**{n:weights[n] for n in RETURN_PARAMETERS}),
                lambda v:consume_head(v,v['H'],weights,native),{n:abi([1,4,3072],'torch.bfloat16') for n in ['A','U','M','T']},
                {'headOutputFrame':head.frame_id,'sourceInputsFrame':source_inputs.frame_id})
            run('next_query',dict(rows=source.values['T'],txtMod=source_inputs.values['txtMod'],ropeCos=source_inputs.values['ropeCos'],ropeSin=source_inputs.values['ropeSin'],
                    **{n:weights[n] for n in NEXT_PARAMETERS}),
                lambda v:next_query(v['rows'],v['txtMod'],(v['ropeCos'],v['ropeSin']),weights,native),
                {'nextQ':abi([1,4,24,128],'torch.bfloat16')},{'sourceOutputFrame':source.frame_id,'sourceInputsFrame':source_inputs.frame_id,'nextJoint':2})
    finally:
        torch.nn.functional.linear=original_linear
    if before!={n:(v.data_ptr(),v._version,payload(v)) for n,v in weights.items()} or snapshot(torch)!=numerical:
        raise ValueError('Parameter/numerical closure differs')
    if not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):
        raise ValueError('RNG closure differs')
    assets_unchanged(assets);total=sum(x['nominalMACs'] for x in operations)
    if total!=r['plannedMatrixMACs']['actor']:raise ValueError('Physical matrix work differs:'+str(total))
    save(out/'actor.json',records)
    save(out/'actor-seal.json',{'requestSHA256':r['requestSHA256'],'actorSHA256':sha(out/'actor.json')})
    save(out/'actor-execution.json',{'loadedParameters':loaded,'operations':operations,'nominalMatrixMACs':total,
        'parameterInputRNGClosure':True,'numericalProfile':numerical,'assets':assets,'header':header_receipt,
        'arithmeticTimings':timings,'wallSecondsAfterImports':time.perf_counter()-started,'CPUSecondsAfterImports':time.process_time()-cpu,
        'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),
        'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':denied,'successfulFileOpenCounts':accesses}})
    print('SOURCE_READER_ACTOR_SEALED',flush=True)


if __name__=='__main__':execute(sys.argv[1])
