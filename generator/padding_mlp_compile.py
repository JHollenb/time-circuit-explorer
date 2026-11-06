"""Compile one lexical-pad native anchor and smooth full-input Jacobian."""
from pathlib import Path
import runpy
import time
from encoder_atoms import np,read,save,sha,AtomStore,floats
from paired_fields import retain,payload,metrics
from padding_mlp import SourceWeights,smooth_jacobian_torch


def execute(root):
    import torch
    from torch.nn import functional as F
    from tokenizers import Tokenizer
    from complete_masked_study import profile
    from padding_mlp_study import verify,install_guard
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json');install_guard(root)
    tick=time.perf_counter();cpu=time.process_time();numerical=profile(torch)
    if numerical!=s['encoderNumericalProfile'] or torch.__version__!=s['encoderVersions']['torch']:raise ValueError('Compiler numerical profile differs')
    model=Path(s['lexicalSource']['modelRoot'])
    for name,h in s['lexicalSource']['assets'].items():
        if sha(model/name)!=h:raise ValueError('Pinned model/tokenizer metadata differs')
    config=read(model/'text_encoder/config.json');token_config=read(model/'tokenizer/tokenizer_config.json');pad=token_config['pad_token'];pad=pad['content'] if isinstance(pad,dict) else pad
    tokenizer=Tokenizer.from_file(str(model/'tokenizer/tokenizer.json'));pad_id=tokenizer.token_to_id(pad)
    store=AtomStore(s['lexicalStore'],r['manifestSHA256']);policy=store.manifest['tokenPolicy']
    if pad_id is None or (pad,pad_id)!=(policy['padToken'],policy['padID']):raise ValueError('Pinned pad identity differs')
    weights=SourceWeights(s);embedding=weights.parameter('model.embed_tokens.weight',row=pad_id)
    atom=store.gather(np.array([pad_id]))[0,:2560]
    if payload(embedding)!=__import__('hashlib').sha256(atom.tobytes()).hexdigest():raise ValueError('Original pad embedding differs from compiled lexical source')
    names={'G':'mlp.gate_proj','U':'mlp.up_proj','D':'mlp.down_proj','gamma':'post_attention_layernorm'}
    ps={k:weights.parameter('model.layers.0.'+n+'.weight') for k,n in names.items()}
    if [list(ps[k].shape) for k in ['G','U','D','gamma']]!=[[9728,2560],[9728,2560],[2560,9728],[2560]]:raise ValueError('Source MLP geometry differs')
    graph=runpy.run_path(str(root/'flux_functional_qwen_prefix.py'));raw=embedding.reshape(1,1,2560).repeat(1,512,1)
    with torch.inference_mode():
        x=graph['_rms_norm'](raw,ps['gamma'],config['rms_norm_eps']);gn=F.linear(x,ps['G']);un=F.linear(x,ps['U']);bn=F.linear(F.silu(gn)*un,ps['D'])
        repeated={n:int(torch.count_nonzero(v!=v[:,0:1,:]).item()) for n,v in [('normalized',x),('gate',gn),('up',un),('write',bn)]}
        if any(repeated.values()):raise ValueError('Repeated source anchor is not row-independent')
        x0=x[0,0].clone();b=bn[0,0].clone();G,U,D=[ps[k].double() for k in ['G','U','D']];xd=x0.double()
        smooth,J,g,u=smooth_jacobian_torch(xd,G,U,D);emitted=J.float()
        out=root/'compiled';out.mkdir();fields={'x0':retain(out,'x0',x0),'b':retain(out,'b',b),'J':retain(out,'J',emitted),'smoothAnchor':retain(out,'smooth-anchor',smooth),
            'smoothGate':retain(out,'smooth-gate',g),'smoothUp':retain(out,'smooth-up',u),'jacobianEmissionError':retain(out,'jacobian-emission-error',emitted.double()-J)}
        offset=metrics(b.float().cpu().numpy(),smooth.cpu().numpy());cast=metrics(emitted.double().cpu().numpy(),J.cpu().numpy())
    weights.unchanged();m=store.manifest;atoms=[{'path':str(Path(s['lexicalStore'])/'manifest.json'),'sha256':r['manifestSHA256']}]
    for k in ['coverage','rowHashes']:atoms.append({'path':str(Path(s['lexicalStore'])/m[k]['path']),'sha256':m[k]['sha256']})
    atoms += [{'path':str(Path(s['lexicalStore'])/m['shards'][j]['path']),'sha256':m['shards'][j]['sha256']} for j in sorted(store.shards)]
    save(root/'compile-receipt.json',{'schema':'tce-padding-mlp-compile-v1','requestSHA256':r['requestSHA256'],'padToken':pad,'padID':pad_id,'fields':fields,'repeatedSourceRows':512,'repeatedRowMismatches':repeated,
        'nativeAnchorMinusSmooth':offset,'jacobianFP32Emission':cast,'sourceParameterReads':weights.reads,'sourceAssetChecks':weights.checks,'sourceAtomFiles':atoms,'numericalProfile':numerical,
        'work':{'nativeAnchorMLPLinears':3,'nativeAnchorMLPMACs':512*3*2560*9728,'smoothAnchorMACs':3*2560*9728,'jacobianProductsMACs':2*2560*2560*9728,'emittedJacobianPayloadBytes':2560*2560*4},
        'wallSecondsAfterImports':time.perf_counter()-tick,'cpuSecondsAfterImports':time.process_time()-cpu,
        'scope':'Source-only pad embedding and original weights. Native BF16 anchor at repeated512row profile; derivative of smooth FP64 widened-weight function at decoded native x0, emittedFP32. No target activation/fit/precision choice.'})
    print('PADDING_MLP_COMPILED',sha(root/'compile-receipt.json'),flush=True)
