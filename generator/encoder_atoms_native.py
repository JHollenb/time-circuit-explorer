"""Offline token-local compilation and separate post-seal native prefix reference.

Never imported by the guarded serving entry point. No full Qwen model is built.
"""
import argparse
from pathlib import Path
import resource
import shutil
import time

from encoder_atoms import np,read,save,sha,identity


def source(spec):
    import torch
    from safetensors import safe_open
    from transformers import Qwen2TokenizerFast,Qwen3Config
    from transformers.models.qwen3 import modeling_qwen3 as native
    model=Path(spec['modelRoot'])
    for name,digest in spec['assets'].items():
        if sha(model/name)!=digest:raise ValueError('Source asset differs: '+name)
    if sha(native.__file__)!=spec['nativeSourceSHA256']:raise ValueError('Installed Qwen source differs')
    pipeline=Path(native.__file__).parents[3]/'diffusers/pipelines/flux2/pipeline_flux2_klein.py'
    if sha(pipeline)!=spec['pipelineSourceSHA256']:raise ValueError('Installed pipeline source differs')
    path=model/spec['weightFile']['path']
    if path.stat().st_size!=spec['weightFile']['bytes'] or sha(path)!=spec['weightFile']['sha256']:raise ValueError('Checkpoint shard differs')
    config=Qwen3Config.from_pretrained(model/'text_encoder',local_files_only=True)
    if (config.vocab_size,config.hidden_size,config.num_attention_heads,config.num_key_value_heads,config.head_dim)!=(151936,2560,32,8,128):
        raise ValueError('Unexpected encoder geometry')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    with safe_open(path,framework='pt',device='cpu') as f:
        tensors={key:f.get_tensor(name).to('cuda') for key,name in spec['tensors'].items()}
    embed=torch.nn.Embedding.from_pretrained(tensors['e'],freeze=True)
    def norm(name,size):
        value=native.Qwen3RMSNorm(size,eps=config.rms_norm_eps).to('cuda',dtype=torch.bfloat16)
        value.weight=torch.nn.Parameter(tensors[name],requires_grad=False);return value
    norms={'input':norm('input',2560),'qn':norm('qn',128),'kn':norm('kn',128)}
    projections={}
    for key,width in [('q',4096),('k',1024),('v',1024)]:
        value=torch.nn.Linear(2560,width,bias=False,device='meta',dtype=torch.bfloat16)
        value.weight=torch.nn.Parameter(tensors[key],requires_grad=False);projections[key]=value
    rope=native.Qwen3RotaryEmbedding(config,device='cuda')
    tokenizer=Qwen2TokenizerFast.from_pretrained(model/'tokenizer',local_files_only=True)
    policy={'paddingSide':tokenizer.padding_side,'truncationSide':tokenizer.truncation_side,
            'padID':tokenizer.pad_token_id,'padToken':tokenizer.pad_token,'eosToken':tokenizer.eos_token,
            'addSpecialTokens':True,'addGenerationPrompt':True,'enableThinking':False}
    counts={'embedding':0,'rmsNorm':0,'linear':0,'rotaryTable':0,'rotaryApplication':0}
    def prefix(ids,positioned=False):
        with torch.inference_mode():
            e=embed(ids);counts['embedding']+=1;n=norms['input'](e);counts['rmsNorm']+=1
            q=norms['qn'](projections['q'](n).view(1,512,32,128));k=norms['kn'](projections['k'](n).view(1,512,8,128))
            v=projections['v'](n);counts['rmsNorm']+=2;counts['linear']+=3
            values={'e':e.reshape(512,-1),'q':q.reshape(512,-1),'k':k.reshape(512,-1),'v':v.reshape(512,-1)}
            if positioned:
                cos,sin=rope(e,torch.arange(512,device='cuda')[None,:]);counts['rotaryTable']+=1
                qp,kp=native.apply_rotary_pos_emb(q.transpose(1,2),k.transpose(1,2),cos,sin);counts['rotaryApplication']+=1
                values.update(qpos=qp.transpose(1,2).reshape(512,-1),kpos=kp.transpose(1,2).reshape(512,-1))
            return {key:value.contiguous().cpu().view(torch.uint16).numpy().copy() for key,value in values.items()}
    parameter_bytes=sum(value.numel()*value.element_size() for value in tensors.values())
    return torch,tokenizer,policy,rope,prefix,counts,parameter_bytes


def compile_atoms(spec_path,output):
    spec=read(spec_path);out=Path(output);out.mkdir(exist_ok=False)
    started=time.perf_counter();cpu=time.process_time();torch,tokenizer,policy,rope,prefix,counts,parameter_bytes=source(spec)
    setup_s=time.perf_counter()-started;torch.cuda.reset_peak_memory_stats()
    if spec['coverage']=='full':ids=np.arange(151936,dtype='<u4')
    else:
        selected=set(range(512))|set(range(151643,151936))
        for prompt in spec['smokePrompts']:
            rendered=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            selected.update(tokenizer(rendered)['input_ids'])
        ids=np.asarray(sorted(selected),dtype='<u4')
    np.save(out/'coverage.npy',ids,allow_pickle=False);row_hashes=np.empty((len(ids),32),dtype=np.uint8);shards=[]
    compile0=time.perf_counter()
    for start in range(0,len(ids),512):
        batch=ids[start:start+512];padded=np.pad(batch,(0,512-len(batch)),mode='edge').astype(np.int64)
        values=prefix(torch.from_numpy(padded).to('cuda')[None,:])
        from encoder_atoms import floats
        if any(not np.isfinite(floats(value)).all() for value in values.values()):raise ValueError('Nonfinite compiled atom')
        atoms=np.concatenate([values[key] for key in ['e','q','k','v']],axis=-1)[:len(batch)]
        for i,row in enumerate(atoms):row_hashes[start+i]=np.frombuffer(__import__('hashlib').sha256(row.tobytes()).digest(),dtype=np.uint8)
        path=out/('atoms-%04d.npy'%(start//512));np.save(path,atoms,allow_pickle=False)
        shards.append({'path':path.name,'bytes':path.stat().st_size,'shape':list(atoms.shape),'sha256':sha(path)})
        if start==0 or len(ids)<=start+512 or (start//512)%32==0:print('ENCODER_ATOMS_COMPILED',min(start+512,len(ids)),len(ids),flush=True)
    compile_s=time.perf_counter()-compile0;np.save(out/'row-hashes.npy',row_hashes,allow_pickle=False)
    with torch.inference_mode():
        cos,sin=rope(torch.empty((1,512,2560),device='cuda',dtype=torch.bfloat16),torch.arange(512,device='cuda')[None,:]);counts['rotaryTable']+=1
        position=np.stack([v[0].contiguous().cpu().view(torch.uint16).numpy() for v in [cos,sin]])
    np.save(out/'rope.npy',position,allow_pickle=False)
    for name in ['tokenizer.json','chat_template.jinja']:shutil.copyfile(Path(spec['modelRoot'])/'tokenizer'/name,out/name)
    def item(name):return {'path':name,'sha256':sha(out/name),'bytes':(out/name).stat().st_size}
    manifest={'schema':'tce-qwen3-token-atoms-v1','coverageMode':spec['coverage'],'vocabSize':151936,'coverageCount':len(ids),
              'hiddenSize':2560,'qWidth':4096,'kvWidth':1024,'headDim':128,'rowWidth':8704,'sequenceLength':512,'chunkRows':512,
              'coverage':item('coverage.npy'),'rowHashes':item('row-hashes.npy'),'rope':item('rope.npy'),
              'tokenizer':item('tokenizer.json'),'template':item('chat_template.jinja'),'tokenPolicy':policy,'shards':shards,
              'sourceSpecSHA256':sha(spec_path),'numericPolicy':'Native eager CUDA BF16 linears; each Qwen RMS FP32 norm -> BF16 -> BF16 gamma. Pre-RoPE token-local atoms. Position cos/sin cached from native FP32 RoPE at512 fixed positions.',
              'scope':'Learned token-local memoization; not contextual encoding, semantic composition or image generation.'}
    save(out/'manifest.json',manifest)
    receipt={'schema':'tce-token-atoms-compile-v1','sourceSpecSHA256':sha(spec_path),'manifestSHA256':sha(out/'manifest.json'),
             'coverageMode':spec['coverage'],'coverageCount':len(ids),'uncoveredCount':151936-len(ids),
             'atomPayloadBytes':int(len(ids)*8704*2),'projectionMACs':int(len(ids)*2560*6144),
             'physicalProjectionMACs':int(len(shards)*512*2560*6144),'physicalBatches':len(shards),'batchGeometry':[1,512,2560],
             'sourceParameterPayloadBytes':parameter_bytes,'sourceShardBytesHashed':spec['weightFile']['bytes'],
             'setupSeconds':setup_s,'atomCompileSeconds':compile_s,'wallSeconds':time.perf_counter()-started,'cpuSeconds':time.process_time()-cpu,
             'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'peakCudaAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCudaReservedBytes':torch.cuda.max_memory_reserved(),
             'nativePrimitiveCalls':counts,'fullEncoderForwards':0,'transformerDenoiserForwards':0,'vaeForwards':0,
             'versions':{'torch':torch.__version__,'numpy':np.__version__,'transformers':__import__('transformers').__version__},
             'source':spec,'store':str(out)}
    save(out/'compile-receipt.json',receipt)


def reference(spec_path,serve_path,output):
    spec=read(spec_path);serving=read(serve_path);out=Path(output);out.mkdir(exist_ok=False)
    started=time.perf_counter();torch,tokenizer,policy,rope,prefix,counts,parameter_bytes=source(spec);setup_s=time.perf_counter()-started
    rows=[];torch.cuda.reset_peak_memory_stats()
    for item in serving['prompts']:
        t=time.perf_counter();text=tokenizer.apply_chat_template([{'role':'user','content':item['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        encoded=tokenizer(text,return_tensors='pt',padding='max_length',truncation=True,max_length=512)
        values=prefix(encoded.input_ids.to('cuda'),positioned=True)
        values.update(ids=encoded.input_ids[0].numpy(),mask=encoded.attention_mask[0].numpy(),positions=np.arange(512,dtype=np.int64))
        path=out/(item['id']+'.npz');np.savez(path,**values)
        rows.append({'id':item['id'],'path':str(path),'sha256':sha(path),'renderedTextSHA256':__import__('hashlib').sha256(text.encode()).hexdigest(),'secondsIncludingSave':time.perf_counter()-t})
    save(out/'reference-receipt.json',{'rows':rows,'tokenPolicy':policy,'setupSeconds':setup_s,'wallSeconds':time.perf_counter()-started,
         'nativePrimitiveCalls':counts,'fullEncoderForwards':0,'highWaterRSSKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'peakCudaAllocatedBytes':torch.cuda.max_memory_allocated(),'scope':'Post-seal isolated first-layer prefix, fixed B1 T512, no contextual attention, later layers, denoising or image.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['compile','reference']);p.add_argument('spec');p.add_argument('output');p.add_argument('--serve');a=p.parse_args()
    compile_atoms(a.spec,a.output) if a.action=='compile' else reference(a.spec,a.serve,a.output)
