"""Actual current-state weighted inputs, original V slices, own finite consumer."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha
from paired_fields import load, retain, payload
from projected_complement import fields, load_selected
from propagated_context import P1, RETURN_PARAMETERS, NEXT_PARAMETERS, normalized, next_query
from image_dual import image_numerators, consume_head
from deferred_values import form

VALUE_PARAMETERS=(P1+'attn.to_v.weight',P1+'attn.add_v_proj.weight')
PARAMETERS=VALUE_PARAMETERS+RETURN_PARAMETERS+NEXT_PARAMETERS


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
    allowed += [s['selected'][n]['path'] for n in s['selected'] if n!='V']+[s['textKeySlice']['path']]
    allowed += [x['path'] for x in s['reuse'].values()]
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
        spec=ActSpec('tce.deferred_value_'+name,'v1',reads=tuple(inputs),writes=tuple(abis),
            write_kinds={n:'tensor' for n in abis},write_abis=abis,
            numerical_contract='native-BF16-ModLN/same-retained-P-RMS-allocation/CPU-FP64-deferred-V/native-finite-consumer')
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
            def splice(v):
                x=v['Qtext'].clone();x[:,r['support']]=v['Ptext'][:,r['support']];return {'x':x}
            text=run('splice',{'Qtext':normal['text_modln'].values['x'],'Ptext':normal['p_text_modln'].values['x']},splice,
                {'x':abi([1,512,3072],'torch.bfloat16')},{'QtextFrame':normal['text_modln'].frame_id,
                'PtextFrame':normal['p_text_modln'].frame_id,'support':r['support']})
            selected={n:load(x,'cuda') for n,x in s['selected'].items() if n!='V'}
            values=dict(x=normal['image_modln'].values['x'],**{n:load(x,'cpu') for n,x in s['reuse'].items()},
                Q=selected['Q'],selectedK=selected['K'],textK=load(s['textKeySlice'],'cpu'))
            def allocate(v):
                q=v['Q'][0].cpu().double().permute(1,0,2);text=v['textK'].clone();text[:,r['support']]=v['selectedK'].cpu()
                numerator=image_numerators(v['x'][0].cpu().double(),v['duals'],v['groupIDs'],mm)
                image=numerator/(128**.5*v['pDenominator'].T[:,None])
                text_logits=mm(q,text[0].double().permute(1,2,0),'text-QK')/128**.5
                logits=torch.cat((text_logits,image),-1);peak=logits.max(-1,keepdim=True).values
                w=torch.exp(logits-peak);partition=w.sum(-1,keepdim=True);probability=w/partition
                if not bool(torch.isfinite(probability).all() and (partition>0).all()):raise ValueError('Invalid allocation')
                return dict(imageNumerator=numerator,textLogits=text_logits,logits=logits,peak=peak,partition=partition,
                    probability=probability,textK_hybrid=text)
            abis={n:abi([24,4,1536]) for n in ['logits','probability']}
            abis.update(imageNumerator=abi([24,4,1024]),textLogits=abi([24,4,512]),peak=abi([24,4,1]),partition=abi([24,4,1]),
                        textK_hybrid=abi([1,512,24,128],'torch.bfloat16'))
            allocation=run('allocate',values,allocate,abis,{'imageModLNFrame':normal['image_modln'].frame_id,
                'compilerSealSHA256':s['compilerSealSHA256'],'sameAllocationLaw':'cycle58 retained smooth P norm'})
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
            source=run('source',dict(H=head.values['H'],R=selected['R'],txtMod=selected['txtMod'],**{n:weights[n] for n in RETURN_PARAMETERS}),
                lambda v:consume_head(v,v['H'],weights,native),{n:abi([1,4,3072],'torch.bfloat16') for n in ['A','U','M','T']},
                {'headOutputFrame':head.frame_id})
            run('next_query',dict(rows=source.values['T'],txtMod=selected['txtMod'],ropeCos=selected['ropeCos'],ropeSin=selected['ropeSin'],
                    **{n:weights[n] for n in NEXT_PARAMETERS}),
                lambda v:next_query(v['rows'],v['txtMod'],(v['ropeCos'],v['ropeSin']),weights,native),
                {'nextQ':abi([1,4,24,128],'torch.bfloat16')},{'sourceOutputFrame':source.frame_id,'nextJoint':2})
    finally:
        torch.nn.functional.linear=original_linear
    if before!={n:(v.data_ptr(),v._version,payload(v)) for n,v in weights.items()} or snapshot(torch)!=numerical:
        raise ValueError('Parameter/numerical closure differs')
    if not torch.equal(rng,torch.get_rng_state()) or not all(torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):
        raise ValueError('RNG closure differs')
    assets_unchanged(assets);total=sum(x['nominalMACs'] for x in operations)
    if total!=r['plannedMatrixMACs']['actor']:raise ValueError('Physical matrix work differs:'+str(total))
    save(out/'actor.json',{**records,'compilerSealSHA256':s['compilerSealSHA256']})
    save(out/'actor-seal.json',{'requestSHA256':r['requestSHA256'],'actorSHA256':sha(out/'actor.json')})
    save(out/'actor-execution.json',{'loadedParameters':loaded,'operations':operations,'nominalMatrixMACs':total,
        'parameterInputRNGClosure':True,'numericalProfile':numerical,'assets':assets,'header':header_receipt,
        'arithmeticTimings':timings,'wallSecondsAfterImports':time.perf_counter()-started,'CPUSecondsAfterImports':time.process_time()-cpu,
        'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved(),
        'fileQuarantine':{'allowedFiles':sorted(set(allowed)),'deniedCanaries':denied,'successfulFileOpenCounts':accesses}})
    print('DEFERRED_VALUE_ACTOR_SEALED',flush=True)


if __name__=='__main__':execute(sys.argv[1])
