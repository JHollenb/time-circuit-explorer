"""Fresh raw strings -> lexical atoms -> own H1 -> mask-compacted H2."""
import argparse
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
from encoder_atoms import np, read, save, sha, identity, floats

HERE = Path(__file__).resolve().parent
PROMPTS = [
    {'id': 'short', 'prompt': 'An indigo teapot balancing on a brass compass beside a folded paper map.'},
    {'id': 'all-attended', 'prompt': ('An indigo teapot beside a brass compass and a folded paper map, detailed studio photograph. ' * 80)}]
PORTS = ('q', 'k', 'v', 'head', 'R', 'M', 'Hnext')


def field_metrics(a, b, retained=False):
    e=a-b;er=float(np.sqrt(np.mean(e*e)));br=float(np.sqrt(np.mean(b*b)))
    result={'rms':er,'referenceRms':br,'relativeRms':er/br if br else None,'maxAbs':float(np.max(np.abs(e))),'changedValues':int(np.count_nonzero(e)),'scalars':int(e.size)}
    if retained:
        aw=a.astype(np.float32).view(np.uint32)>>16;bw=b.astype(np.float32).view(np.uint32)>>16
        result.update(BF16WordMismatches=int(np.count_nonzero(aw!=bw)),signedZeroDefects=int(np.count_nonzero((a==0)&(b==0)&(aw!=bw))))
    return result


def verify(root, r):
    if identity({k: v for k, v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request changed')
    for name, digest in r['sourceHashes'].items():
        if sha(Path(root) / name) != digest: raise ValueError('Payload changed: ' + name)


def stage(root):
    from mrun.client.api import Api
    parent = HERE.parent / 'artifacts/encoder-atoms-full-v1'; j = read(parent / 'scheduler-job.json')
    live = Api().json('GET', '/api/jobs/' + j['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != j['payload_custody']:
        raise ValueError('Complete lexical parent differs')
    report = read(parent / 'result/report.json')
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); p = root / 'payload'; p.mkdir()
    for name in ('masked_kv.py', 'masked_kv_study.py', 'encoder_atoms.py', 'encoder-atoms-source.json'):
        shutil.copyfile(HERE / name, p / name)
    qualified = HERE.parent.parent / 'saturn-pub/packages/mdb/qualification'
    for name, digest in {'flux_functional_qwen_prefix.py': '763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a',
                         'qwen_h18_state_history.py': '17084d404f5ae00557e4ad8e94a8b84704787c47ebf9c875228cb452c4857036'}.items():
        if sha(qualified / name) != digest: raise ValueError('Qualified functional graph changed')
        shutil.copyfile(qualified / name, p / name)
    save(p / 'serve.json', {'store': report['compile']['store'], 'manifestSHA256': report['manifestSHA256'], 'prompts': PROMPTS, 'repeats': 1})
    r = {'schema': 'tce-masked-kv-own-state-request-v1', 'parentJobID': j['job_id'],
         'parentRequestSHA256': live['config']['requestSHA256'], 'manifestSHA256': report['manifestSHA256'],
         'scope': 'Two fresh raw strings, lexical table -> own H1 -> compact-KV H2. First-prompt conditional compiler reduction, not zero-neural encoder/images.',
         'protocol': 'All candidate H1/H2 seal before full-formation and zero-masked-KV controls or native two-block chain. Controls consume own candidate H1; native consumes own native H1. No target refresh.',
         'sourceHashes': {f.name: sha(f) for f in p.iterdir()}, 'device': 'cuda', 'backend': 'source-bound-torch-bf16', 'timeoutSeconds': 600}
    r['requestSHA256'] = identity(r); save(root / 'request.json', r); save(p / 'request.json', r)
    save(p / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [f.name for f in p.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': r['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root / 'payload', r)
    if (root / 'job.json').exists(): raise ValueError('Already submitted')
    cmd = command()[:-5] + ['python', '-u', 'masked_kv_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1']
    job = launch(cmd, experiment='tce-masked-kv-own-state-v1', task='analysis', payload=root / 'payload', pin='beast',
          needs={'cuda': True}, config={'schema': r['schema'], 'requestSHA256': r['requestSHA256'], 'device': 'cuda', 'backend': 'torch', 'seq_len': 512},
          timeout_s=600, detach=True, retry_on_kill=False, preflight=True, guarded=True,
          note='Rawprompt atoms->ownH1->compactKV H2;2prompts,12SDPA/78linears. ActiveKV rows only; all512queries/MLP rows retained. Full/zero controls and native2block chains after seal. Hostlocal atoms/checkpoint,22weights~404MB; arrays nearcompute. CUDA/CPU1,defaultplan,expected<5min,600s,no retries. No fullencoder/FLUX/VAE/history baseline or target refresh; distinct from neighbor block18 work.')
    result = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root / 'job.json', result); return result


def execute(root):
    import torch
    from torch.nn import functional as F
    from safetensors import safe_open
    from transformers import Qwen3Config
    from transformers.models.qwen3 import modeling_qwen3 as native
    from masked_kv import block
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root, r)
    start = time.perf_counter(); out = root / 'response-result'; out.mkdir(); raw = root / 'masked-kv-arrays'; raw.mkdir()
    # Separate lexical process enforces no neural imports/checkpoint/network use.
    subprocess.run([sys.executable, str(root / 'encoder_atoms.py'), str(root / 'serve.json'), str(root / 'lexical')], check=True)
    lexical_seal = read(root / 'lexical/prediction-seal.json')
    if lexical_seal['manifestSHA256'] != r['manifestSHA256'] or lexical_seal['coverageCount'] != 151936:
        raise ValueError('Complete lexical source differs')
    spec = read(root / 'encoder-atoms-source.json'); model = Path(spec['modelRoot'])
    if sha(native.__file__) != spec['nativeSourceSHA256']: raise ValueError('Native source differs')
    from transformers.integrations import sdpa_attention
    sdpa_source_sha = sha(sdpa_attention.__file__)
    if sdpa_source_sha != 'b72c5c96ea070d594f36e774cfc799160a0ecb157353502a61eb3c5ea1e35b46':
        raise ValueError('Native SDPA integration source differs')
    for name, digest in spec['assets'].items():
        if sha(model / name) != digest: raise ValueError('Checkpoint/tokenizer metadata differs')
    weight_path = model / spec['weightFile']['path']
    if weight_path.stat().st_size != spec['weightFile']['bytes'] or sha(weight_path) != spec['weightFile']['sha256']:
        raise ValueError('Original checkpoint shard differs')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    graph = runpy.run_path(str(root / 'flux_functional_qwen_prefix.py'))
    history = runpy.run_path(str(root / 'qwen_h18_state_history.py'))
    config = Qwen3Config.from_dict(read(model / 'text_encoder/config.json')); config._attn_implementation = 'sdpa'
    profile = graph['_native_profile'](config, 2); parameters = {}; parameter_receipts = []
    with safe_open(weight_path, framework='pt', device='cpu') as source:
        for i in range(2):
            for suffix in graph['QWEN_LAYER_PARAMETER_SUFFIXES']:
                name = f'model.layers.{i}.' + suffix; value = source.get_tensor(name)
                if value.dtype != torch.bfloat16: raise ValueError('Expected original BF16 weights')
                parameter_receipts.append({'name': name, 'shape': list(value.shape), 'bytes': value.numel()*2,
                    'payloadSHA256': __import__('hashlib').sha256(value.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()})
                parameters[name] = value.to('cuda')
    versions = [(name, t.data_ptr(), t._version) for name, t in parameters.items()]
    weight_names = {t.data_ptr(): name for name, t in parameters.items()}
    phase = {'case': None, 'lane': None, 'index': None}; primitives = []; module_entries = []; packets = {}; contexts = {}; inputs = {}; times = []
    old_linear, old_sdpa = F.linear, F.scaled_dot_product_attention
    def tensor(name, value):
        path = raw / name; path.parent.mkdir(parents=True, exist_ok=True)
        a = value.detach().contiguous().cpu().view(torch.uint16).numpy().copy()
        if not bool(torch.isfinite(value).all()): raise ValueError('Nonfinite retained field')
        np.save(path, a, allow_pickle=False)
        return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size, 'shape': list(value.shape), 'stride': list(value.stride()), 'dtype': str(value.dtype)}
    def capture(name, value):
        if name in PORTS:
            packets.setdefault((phase['case'], phase['lane'], phase['index']), {})[name] = tensor('/'.join(map(str,phase.values()))+'/'+name+'.npy', value)
    def linear(value, weight, bias=None):
        result = old_linear(value, weight, bias)
        primitives.append({**phase, 'primitive': 'linear', 'parameter': weight_names.get(weight.data_ptr()),
            'inputShape': list(value.shape), 'weightShape': list(weight.shape), 'rows': value.numel()//value.shape[-1],
            'nominalMACs': (value.numel()//value.shape[-1])*weight.numel()})
        return result
    def sdpa(q, k, v, *args, **kwargs):
        result = old_sdpa(q, k, v, *args, **kwargs)
        mask = kwargs.get('attn_mask', args[0] if args else None)
        primitives.append({**phase, 'primitive': 'SDPA', 'shapes': [list(x.shape) for x in (q,k,v)],
            'strides': [list(x.stride()) for x in (q,k,v)], 'maskShape': list(mask.shape) if mask is not None else None,
            'maskDtype': str(mask.dtype) if mask is not None else None, 'isCausal': kwargs.get('is_causal',False), 'enableGQA': kwargs.get('enable_gqa',False)})
        if phase['lane'] == 'native':
            capture('q',q);capture('k', k[:,::profile['num_key_value_groups']] if k.shape[1]==profile['num_attention_heads'] else k)
            capture('v', v[:,::profile['num_key_value_groups']] if v.shape[1]==profile['num_attention_heads'] else v)
            capture('head',result.transpose(1,2).contiguous())
        return result
    F.linear, F.scaled_dot_product_attention = linear, sdpa
    def tail(hidden, q, k, v, index, context, dispatch):
        prefix=f'model.layers.{index}.'
        p=lambda n:parameters[prefix+n+'.weight']
        for name,value in [('q',q),('k',k),('v',v)]:capture(name,value)
        head=graph['_attention'](q,k,v,context[0],is_causal=context[1],scale=profile['head_dim']**-.5,groups=profile['num_key_value_groups'],dispatch=dispatch);capture('head',head)
        residual=hidden+F.linear(head.reshape(1,512,-1),p('self_attn.o_proj'));capture('R',residual)
        post=graph['_rms_norm'](residual,p('post_attention_layernorm'),profile['rms_norm_eps'])
        product=F.silu(F.linear(post,p('mlp.gate_proj')))*F.linear(post,p('mlp.up_proj'))
        mlp=F.linear(product,p('mlp.down_proj'));capture('M',mlp);result=residual+mlp;capture('Hnext',result);return result
    def timed(lane,index,call):
        phase.update(lane=lane,index=index);torch.cuda.synchronize();tick=time.perf_counter();value=call();torch.cuda.synchronize()
        times.append({**phase,'wallSecondsIncludingPortWrites':time.perf_counter()-tick});return value
    def from_words(a):return torch.from_numpy(a.copy()).view(torch.bfloat16).to('cuda')
    try:
        with torch.inference_mode(), history['_no_modules']():
            for item in PROMPTS:
                name=item['id'];phase['case']=name;record=next(x for x in lexical_seal['rows']if x['id']==name and x['control']=='primary')
                if sha(record['path'])!=record['sha256']:raise ValueError('Lexical forecast changed')
                with np.load(record['path'],allow_pickle=False)as f:a={k:f[k]for k in f.files}
                mask=torch.from_numpy(a['mask']).to('cuda')[None,:];ids=torch.from_numpy(a['ids']).to('cuda')[None,:]
                dispatch=graph['AttentionDispatch'].from_mapping({'backend':'sdpa','gqa_policy':'repeat_kv' if not bool(mask.bool().all())else'enable_gqa','source':spec['nativeSourceSHA256']})
                context=history['_context'](graph,ids,mask,profile,dispatch,{'init_device':'cpu','init_dtype':'float32','placement_dtype':'float32','source':spec['nativeSourceSHA256']},torch.bfloat16)
                hidden=from_words(a['e']).view(1,512,-1)
                q=from_words(a['qpos']).view(1,512,32,128).transpose(1,2);k=from_words(a['kpos']).view(1,512,8,128).transpose(1,2);v=from_words(a['v']).view(1,512,8,128).transpose(1,2)
                h1=timed('candidate',0,lambda:tail(hidden,q,k,v,0,context,dispatch))
                h2=timed('candidate',1,lambda:block(graph,h1,parameters,profile,1,context,dispatch,mask,'compact',capture))
                inputs[name]={'E':hidden,'H1':h1,'mask':mask,'ids':ids};contexts[name]=(context,dispatch)
        candidate_records=[{'case':c,'lane':l,'index':i,'ports':ports}for(c,l,i),ports in packets.items()]
        save(out/'prediction-seal.json',{'schema':'tce-masked-kv-prediction-seal-v1','requestSHA256':r['requestSHA256'],
             'lexicalSealSHA256':sha(root/'lexical/prediction-seal.json'),'nativeModuleEntries':0,'nativeTargets':0,'rows':candidate_records})
        print('MASKED_KV_COHORT_SEALED',sha(out/'prediction-seal.json'),flush=True)
        # Diagnostic references begin only after the complete own-state forecast seal.
        with torch.inference_mode(), history['_no_modules']():
            for item in PROMPTS:
                name=item['id'];phase['case']=name;own=inputs[name];context,dispatch=contexts[name]
                timed('full',1,lambda:history['_block'](graph,own['H1'],parameters,profile,1,context,dispatch,lambda lane,key,value:capture(key,value),'full'))
                timed('zero',1,lambda:block(graph,own['H1'],parameters,profile,1,context,dispatch,own['mask'],'zero',capture))
        layers=[];handles=[]
        for index in range(2):
            with torch.device('meta'): layer=native.Qwen3DecoderLayer(config,index)
            layer.load_state_dict({name.split(f'model.layers.{index}.',1)[1]:t for name,t in parameters.items()if name.startswith(f'model.layers.{index}.')},assign=True)
            layer.eval();layers.append(layer)
            for name,module in layer.named_modules():handles.append(module.register_forward_pre_hook(lambda m,a,n=name:module_entries.append({**phase,'module':n})))
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a:capture('R',a[0])))
            handles.append(layer.mlp.down_proj.register_forward_hook(lambda m,a,value:capture('M',value)))
        with torch.inference_mode():
            for item in PROMPTS:
                name=item['id'];phase['case']=name;hidden=inputs[name]['E'];context,dispatch=contexts[name]
                for index,layer in enumerate(layers):
                    hidden=timed('native',index,lambda:layer(hidden,attention_mask=context[0],position_ids=context[2],past_key_values=None,use_cache=False,position_embeddings=(context[3],context[4])))
                    capture('Hnext',hidden)
        for handle in handles:handle.remove()
    finally:F.linear,F.scaled_dot_product_attention=old_linear,old_sdpa
    if versions!=[(name,t.data_ptr(),t._version)for name,t in parameters.items()]:raise ValueError('Source parameters changed')
    if sum(x['primitive']=='SDPA' for x in primitives)!=12 or sum(x['primitive']=='linear' for x in primitives)!=78:
        raise ValueError('Declared primitive instrument did not execute')
    for item in PROMPTS:
        name=item['id'];active=int(inputs[name]['mask'].sum())
        calls=[x for x in primitives if x['case']==name and x['lane']=='candidate' and x['index']==1 and x['primitive']=='linear']
        for call in calls:
            expected=active if any(s in call['parameter']for s in ('k_proj','v_proj')) else 512
            if call['rows']!=expected:raise ValueError('Actual projection rows differ from declared reduction')
    for row in candidate_records:
        for field in row['ports'].values():
            if sha(field['path'])!=field['sha256']:raise ValueError('Candidate changed after target release')
    def load(record):
        if sha(record['path'])!=record['sha256']:raise ValueError('Retained field changed')
        return floats(np.load(record['path'],allow_pickle=False)).astype(np.float64)
    accounts=[];observations=0
    with(out/'observations.jsonl').open('x')as stream:
        import json
        for item in PROMPTS:
            name=item['id'];mask=inputs[name]['mask'][0].cpu().numpy().astype(bool)
            for index,source_lane,target_lane in [(0,'candidate','native'),(1,'candidate','full'),(1,'candidate','zero'),(1,'candidate','native'),(1,'zero','full')]:
                channel=source_lane+'-minus-'+target_lane
                cand=packets[name,source_lane,index];ref=packets[name,target_lane,index]
                for port in PORTS:
                    a,b=load(cand[port]),load(ref[port])
                    if port in ('q','k','v'):a=a.transpose(0,2,1,3);b=b.transpose(0,2,1,3)
                    a=a.reshape(512,-1);b=b.reshape(512,-1)
                    for region,use in [('all',np.ones(512,dtype=bool)),('active',mask),('padding',~mask)]:
                        if use.any():accounts.append({'case':name,'index':index,'channel':channel,'kind':'endpoint','port':port,'region':region,**field_metrics(a[use],b[use],retained=True)})
                    if port=='Hnext':
                        baseline=inputs[name]['E'].float().cpu().numpy().astype(np.float64).reshape(512,-1) if index==0 else inputs[name]['H1'].float().cpu().numpy().astype(np.float64).reshape(512,-1)
                        reference_baseline=load(packets[name,'native',0]['Hnext']).reshape(512,-1) if index==1 and target_lane=='native'else baseline
                        update=a-baseline;reference_update=b-reference_baseline
                        for region,use in [('all',np.ones(512,dtype=bool)),('active',mask),('padding',~mask)]:
                            if use.any():accounts.append({'case':name,'index':index,'channel':channel,'kind':'block-update','port':port,'region':region,**field_metrics(update[use],reference_update[use])})
                        for i in range(512):
                            stream.write(json.dumps({'case':name,'index':index,'channel':channel,'position':i,'attended':bool(mask[i]),'endpoint':field_metrics(a[i],b[i],retained=True),'blockUpdate':field_metrics(update[i],reference_update[i])},allow_nan=False)+'\n');observations+=1
    all_packets=[{'case':c,'lane':l,'index':i,'ports':ports}for(c,l,i),ports in packets.items()]
    save(out/'retained-fields.json',{'rows':all_packets})
    save(out/'primitive-receipts.json',{'calls':primitives,'nativeModuleEntries':module_entries})
    save(out/'source.json',{'requestSHA256':r['requestSHA256'],'lexicalManifestSHA256':r['manifestSHA256'],'lexicalSeal':lexical_seal,'sourceSpec':spec,'parameters':parameter_receipts,
         'versions':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'numpy':np.__version__},'sdpaSourceSHA256':sdpa_source_sha,
         'contexts':[{'case':name,'activeRows':int(v['mask'].sum())}for name,v in inputs.items()]})
    save(out/'report.json',{'schema':'tce-masked-kv-own-state-result-v1','requestSHA256':r['requestSHA256'],'scope':r['scope'],'accounts':accounts,'observations':observations,'timings':times,
         'workerSeconds':time.perf_counter()-start,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),
         'work':{'fullEncoderCalls':0,'embeddingCalls':0,'FLUXCalls':0,'VAECalls':0,'nativeComponentBlocks':4,'lexicalOriginalProjections':0},
         'mechanics_status':'complete','instrument_status':'executed','terminal_status':'unassessed',
         'limitations':['Original attention/output/norm/MLP computation remains. One sample phase wall times include port capture, not a speedup comparison.','Zeroed masked K/V intentionally differ from unconsumed native addresses; score active operands separately.','Two fixed prompts and dependent query rows; not arbitrary encoder or image coverage.']})
    shutil.copyfile(root/'request.json',out/'request.json');print('MASKED_KV_DONE',flush=True)


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        result=collect(a.run)
    elif a.action=='stage':result=stage(a.run)
    else:result=globals()[a.action](a.run)
    if result is not None:print(json.dumps(result))
