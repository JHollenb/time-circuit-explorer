"""Tiny CUDA attention-bias contract check; synthetic operands, no model weights."""
import argparse
import json
from pathlib import Path
import shutil
import time
from encoder_atoms import np,read,save,sha,identity

HERE=Path(__file__).resolve().parent


def stage(root):
    root=Path(root).resolve();root.mkdir(exist_ok=False,parents=True);p=root/'payload';p.mkdir()
    for n in ['coarse_bias_mechanics.py','encoder_atoms.py']:shutil.copyfile(HERE/n,p/n)
    r={'schema':'tce-coarse-bias-kernel-request-v1','device':'cuda','backend':'diffusers-sdpa','ownedSources':['coarse_bias_mechanics.py','encoder_atoms.py'],
        'sourceHashes':{x.name:sha(x) for x in p.iterdir()},'attentionDispatchSHA256':'2acfeb88d200a6466c0b4c718e5699efd8dc2e61d8b7c96d9d7c938ffbb25bc8',
        'protocol':'Synthetic1x8x24x128BF16QKV,sourcefixedlog4bias. RecordFP32biasoutcome;requirefinitequerydtypebiasdispatch,compareindependentFP64softmaxasobservation. No weights/images/fits/learnedmaps.'}
    r['requestSHA256']=identity(r);save(root/'request.json',r);save(p/'request.json',r);save(p/'preflight.v1.json',{'schema_version':'mrun-preflight-config-v1','hashed_roots':[x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'requestSHA256':r['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root=Path(root).resolve();r=read(root/'request.json')
    if (root/'job.json').exists():raise ValueError('Already submitted')
    cmd=command()[:-5]+['python','-u','coarse_bias_mechanics.py','execute','.']
    job=launch(cmd,experiment='tce-coarse-bias-kernel-v1',task='analysis',payload=root/'payload',pin='beast',needs={'cuda':True},
        config={'requestSHA256':r['requestSHA256'],'specimen':'synthetic-attention-bias','backend':'diffusers','device':'cuda','task_family':'kernel-mechanics'},
        timeout_s=120,detach=True,retry_on_kill=False,preflight=True,guarded=True,
        note='Tiny synthetic8token24head128dim BF16attentionbias mechanics before correctedcoarseglobalrun. Exactpinneddispatch;FP32log4request->querydtypekernelbias,independentFP64softmaxcomparison diagnostic.0modelweights/learnedmaps/images/fits. Intentfirstautoresources,120scap,noretries.')
    v={'jobID':str(job),'requestSHA256':r['requestSHA256']};save(root/'job.json',v);return v


def execute(root):
    import sys
    root=Path(root).resolve();r=read(root/'request.json')
    if identity({k:x for k,x in r.items() if k!='requestSHA256'})!=r['requestSHA256']:raise ValueError('Request differs')
    for n,h in r['sourceHashes'].items():
        if sha(root/n)!=h:raise ValueError('Source differs')
    def deny(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and str(args[0]).endswith(('.safetensors','.pt','.pth')):raise PermissionError('No model weights in kernel mechanics')
        if event in ['socket.connect','socket.bind']:raise PermissionError('No network in kernel mechanics')
    sys.addaudithook(deny)
    import torch
    import diffusers.models.attention_dispatch as dispatch
    if sha(dispatch.__file__)!=r['attentionDispatchSHA256'] or torch.__version__!='2.13.0':raise ValueError('Pinned kernel source/version differs')
    start=time.perf_counter();x=torch.arange(8*24*128,device='cuda',dtype=torch.float32).reshape(1,8,24,128)
    q=((x%17-8)/8).to(torch.bfloat16);k=torch.roll(q,1,1);v=((x%13-6)/8).to(torch.bfloat16)
    requested=torch.cat([torch.zeros(4,device='cuda'),torch.full((4,),float(np.log(4)),device='cuda')]).reshape(1,1,1,8)
    prior={}
    try:
        y=dispatch.dispatch_attention_fn(q,k,v,attn_mask=requested)
        prior={'status':'accepted','shape':list(y.shape)}
    except RuntimeError as error:prior={'status':'rejected','error':str(error)}
    actual=requested.to(q.dtype);y=dispatch.dispatch_attention_fn(q,k,v,attn_mask=actual);torch.cuda.synchronize()
    if y.shape!=q.shape or y.dtype!=q.dtype or not torch.isfinite(y).all():raise ValueError('Invalid native bias result')
    a,b,c=[t.float().cpu().numpy().astype(np.float64).transpose(0,2,1,3) for t in [q,k,v]];bias=actual.float().cpu().numpy().astype(np.float64)
    scores=a@b.swapaxes(-1,-2)/np.sqrt(128)+bias;weights=np.exp(scores-np.max(scores,axis=-1,keepdims=True));weights/=weights.sum(axis=-1,keepdims=True)
    ref=(weights@c).transpose(0,2,1,3);observed=y.float().cpu().numpy().astype(np.float64);error=observed-ref
    out=root/'response-result';out.mkdir(exist_ok=False);shutil.copyfile(root/'request.json',out/'request.json')
    save(out/'report.json',{'schema':'tce-coarse-bias-kernel-result-v1','requestSHA256':r['requestSHA256'],'requestedFP32Outcome':prior,'consumedBiasDtype':str(actual.dtype),
        'requestedLog4':float(requested.flatten()[-1]),'consumedLog4':float(actual.flatten()[-1]),'queryShape':list(q.shape),'outputShape':list(y.shape),
        'independentFP64Comparison':{'rms':float(np.sqrt(np.mean(error*error))),'maxAbs':float(np.max(np.abs(error)))},'workerSecondsAfterImports':time.perf_counter()-start,
        'work':{'sourceWeightBytesRead':0,'learnedMapEvaluations':0,'newImages':0,'syntheticAttentionDispatches':2},'mechanics_status':'complete','terminal_status':'unassessed',
        'scope':'Tiny synthetic numericalcontract check, not realmodel quality or zero-neural generation. Rawlog4request and roundedBF16kerneloperand remain distinct.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','submit','execute','collect']);p.add_argument('run');a=p.parse_args()
    if a.action=='collect':
        from response_study import collect
        v=collect(a.run)
    else:v=globals()[a.action](a.run)
    if v is not None:print(json.dumps(v))
