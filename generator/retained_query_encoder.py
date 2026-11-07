"""Native nonlinear encoder on a key-closed subset of original query positions."""
from pathlib import Path
import runpy
import time
from encoder_atoms import np,read,save,sha
from paired_fields import retain,payload
from padding_mlp import SourceWeights
from retained_queries import selection,visibility


def execute(root):
    import torch
    from torch.nn import functional as F
    from transformers import Qwen3Config
    from transformers.models.qwen3 import modeling_qwen3 as native
    from transformers.integrations import sdpa_attention
    from conditioning_demand import pack
    from complete_masked_study import profile
    from retained_query_study import verify,install_guard
    root=Path(root).resolve();request=read(root/'request.json');verify(root,request);source=read(root/'inputs.json');install_guard(root)
    started=time.perf_counter();numerical=profile(torch)
    if numerical!=source['encoderNumericalProfile'] or torch.__version__!=source['encoderVersions']['torch'] or __import__('transformers').__version__!=source['encoderVersions']['transformers']:
        raise ValueError('Encoder numerical/version contract differs')
    spec=source['lexicalSource'];model=Path(spec['modelRoot'])
    for p,h in [(native.__file__,spec['nativeSourceSHA256']),(sdpa_attention.__file__,source['sdpaSourceSHA256'])]:
        if sha(p)!=h:raise ValueError('Native implementation differs')
    for n,h in spec['assets'].items():
        if sha(model/n)!=h:raise ValueError('Original metadata differs')
    graph=runpy.run_path(str(root/'flux_functional_qwen_prefix.py'));history=runpy.run_path(str(root/'qwen_h18_state_history.py'))
    config=Qwen3Config.from_dict(read(model/'text_encoder/config.json'));config._attn_implementation='sdpa';contract=graph['_native_profile'](config,27)
    weights=SourceWeights(source);lexical=read(root/'lexical/prediction-seal.json');lex_sha=sha(root/'lexical/prediction-seal.json')
    if lexical['requestSHA256']!=request['requestSHA256'] or lexical['manifestSHA256']!=request['manifestSHA256']:raise ValueError('Lexical request differs')
    states={};contexts={};fields={};compact={};taps={};setup={};selections={};rows=[]
    out=root/'arrays/encoder'
    for prompt in request['prompts']:
        case=prompt['id'];record=next(x for x in lexical['rows'] if x['id']==case)
        if sha(record['path'])!=record['sha256']:raise ValueError('Lexical packet differs')
        with np.load(record['path'],allow_pickle=False) as p:v={k:p[k].copy() for k in p.files}
        selected,expansion=selection(v['mask'],request['paddingQueries']);index=torch.from_numpy(selected).to('cuda');expand=torch.from_numpy(expansion).to('cuda')
        mask=torch.from_numpy(v['mask']).to('cuda')[None,:];ids=torch.from_numpy(v['ids']).to('cuda')[None,:]
        if mask.shape!=(1,512):raise ValueError('Original query geometry differs')
        dispatch=graph['AttentionDispatch'].from_mapping({'backend':'sdpa','gqa_policy':'repeat_kv','source':spec['nativeSourceSHA256']})
        full=history['_context'](graph,ids,mask,contract,dispatch,{'init_device':'cpu','init_dtype':'float32','placement_dtype':'float32','source':spec['nativeSourceSHA256']},torch.bfloat16)
        if full[0] is None or full[0].dtype!=torch.bool or full[1]:raise ValueError('Explicit native padded visibility required')
        expected=visibility(full[0][0,0].cpu().numpy(),selected,v['mask']);small_mask=full[0].index_select(2,index).index_select(3,index)
        if not np.array_equal(small_mask[0,0].cpu().numpy(),expected):raise ValueError('Native mask slicing differs')
        positions=full[2].index_select(1,index);cos=full[3].index_select(1,index);sin=full[4].index_select(1,index)
        contexts[case]=(small_mask,False,positions,cos,sin);selections[case]=(selected,expansion,expand)
        h0=torch.from_numpy(v['e']).view(torch.bfloat16).to('cuda').view(1,512,2560);states[case]=h0.index_select(1,index)
        compact[case]={'H0':retain(out/case,'small-H0',states[case])};fields[case]={};taps[case]={}
        setup[case]={n:retain(out/case,n,value) for n,value in [('fullH0',h0),('activeMask',v['mask'].astype(bool)),('selected',selected),('expansion',expansion),
            ('fullVisibility',full[0]),('compactVisibility',small_mask),('fullPositions',full[2]),('compactPositions',positions),('fullCos',full[3]),('compactCos',cos),('fullSin',full[4]),('compactSin',sin)]}
        rows.append({'id':case,'activeRows':int(mask.sum()),'paddingQueries':len(selected)-int(mask.sum()),'retainedRows':len(selected),'omittedRows':512-len(selected),
            'selectedOriginalIndices':selected.tolist(),'expansionIndices':expansion.tolist(),'originalMaskProfile':full[5],'closedOriginalKeyReads':True})
    calls=[];handoffs=[];times=[];phase={};names={};old_linear=F.linear;old_sdpa=F.scaled_dot_product_attention
    def linear(x,weight,bias=None):
        n=names.get(weight.data_ptr());count=x.numel()//x.shape[-1]
        if n is None or count!=len(selections[phase['case']][0]):raise ValueError('Unexpected compact affine input')
        calls.append({**phase,'kind':'linear','parameter':n,'rows':count,'inputShape':list(x.shape),'weightShape':list(weight.shape),'nominalMACs':count*weight.numel()})
        return old_linear(x,weight,bias)
    def sdpa(q,k,v,*args,**kwargs):
        context=contexts[phase['case']];m=kwargs.get('attn_mask',args[0] if args else None)
        if m is not context[0] or kwargs.get('is_causal',False) or kwargs.get('enable_gqa',False) or kwargs.get('dropout_p',0.)!=0. or kwargs.get('scale')!=128**-.5:
            raise ValueError('Native compact attention contract differs')
        calls.append({**phase,'kind':'SDPA','shapes':[list(a.shape) for a in [q,k,v]],'actualMaskPayloadSHA256':payload(m),'isCausal':False,'scale':kwargs['scale']})
        return old_sdpa(q,k,v,*args,**kwargs)
    F.linear=linear;F.scaled_dot_product_attention=sdpa
    try:
        for layer in range(27):
            prefix='model.layers.'+str(layer)+'.';params={prefix+n:weights.parameter(prefix+n) for n in graph['QWEN_LAYER_PARAMETER_SUFFIXES']}
            names={p.data_ptr():n for n,p in params.items()};versions=[(n,p.data_ptr(),p._version) for n,p in params.items()]
            with torch.device('meta'):module=native.Qwen3DecoderLayer(config,layer)
            module.load_state_dict({n[len(prefix):]:p for n,p in params.items()},assign=True);module.eval().requires_grad_(False)
            with torch.inference_mode():
                for prompt in request['prompts']:
                    case=prompt['id'];phase.update(case=case,layer=layer);context=contexts[case];actual=payload(states[case]);previous=compact[case]['H'+str(layer)]
                    if actual!=previous['payloadSHA256']:raise ValueError('Wrong own compact state')
                    torch.cuda.synchronize();tick=time.perf_counter()
                    result=module(states[case],attention_mask=context[0],position_ids=context[2],past_key_values=None,use_cache=False,position_embeddings=(context[3],context[4]))
                    states[case]=result;item=retain(out/case,'small-H'+str(layer+1),result);compact[case]['H'+str(layer+1)]=item
                    handoffs.append({**phase,'actualInputPayloadSHA256':actual,'inputFieldSHA256':previous['sha256'],'outputPayloadSHA256':item['payloadSHA256']})
                    if layer+1 in [9,18,27]:
                        taps[case][layer+1]=result;fields[case]['H'+str(layer+1)]=retain(out/case,'expanded-H'+str(layer+1),result.index_select(1,selections[case][2]))
                    torch.cuda.synchronize();times.append({**phase,'wallSecondsWithCapture':time.perf_counter()-tick});print('RETAINED_QUERY_BLOCK',case,layer,flush=True)
            if versions!=[(n,p.data_ptr(),p._version) for n,p in params.items()]:raise ValueError('Parameters mutated')
            del params,module;names={};torch.cuda.empty_cache()
        for case in compact:
            small=pack(taps[case]);compact[case]['E']=retain(out/case,'small-E',small);fields[case]['E']=retain(out/case,'expanded-E',small.index_select(1,selections[case][2]))
    finally:F.linear=old_linear;F.scaled_dot_product_attention=old_sdpa
    weights.unchanged()
    if sum(x['kind']=='linear' for x in calls)!=378 or sum(x['kind']=='SDPA' for x in calls)!=54 or sha(root/'lexical/prediction-seal.json')!=lex_sha:raise ValueError('Primitive/lexical census differs')
    save(root/'encoder-candidate.json',{'schema':'tce-retained-query-encoder-v1','requestSHA256':request['requestSHA256'],'fields':fields,'compactFields':compact,'setupFields':setup,'selectionRows':rows,
        'lexicalSealSHA256':lex_sha,'handoffs':handoffs,'primitives':calls,'timings':times,'sourceParameterReads':weights.reads,'sourceAssetChecks':weights.checks,'numericalProfile':numerical,
        'workerSecondsAfterImports':time.perf_counter()-started,'torchVersion':torch.__version__,'transformersVersion':__import__('transformers').__version__,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),
        'scope':'Own compiled lexical H0→27 native compact blocks at original positions/mask/RoPE. All active+16padding queries; nearest expansion only at H9/H18/H27/E. Original nonlinear learned work paid on retained rows; no reference refresh.'})
