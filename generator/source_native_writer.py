"""One closed-form source-only proposal and isolated FP64 writer serving."""
import argparse
from pathlib import Path
import time
from encoder_atoms import np,read,save,sha,words,floats,guard
from paired_fields import load,retain,metrics


def file_guard(spec_path,spec,mode):
    """Trusted-code file quarantine; imports may read runtime files, never labels."""
    import os
    import sys
    allowed={str(Path(spec_path).resolve())};outputs=[]
    if mode=='fit':
        allowed.add(str(Path(spec['packet']['path']).resolve()));outputs=[Path(spec['output']).resolve()]
    else:
        allowed.update(str(Path(spec[k]).resolve()) for k in ['coefficientSeal','fitReceipt'])
        info=read(spec['fitReceipt']);allowed.update(str(Path(x['path']).resolve()) for x in info['fields'].values())
        for item in spec['inputs']:
            allowed.update(str(Path(item[k]['path']).resolve()) for k in ['z','previous','H0'])
            outputs.append(Path(item['output']).resolve())
        allowed.add(str(Path(spec['receipt']).resolve()))
    roots=[Path(spec_path).resolve().parent,*[Path(p).resolve() for p in spec['protectedRoots']]]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if str(p) in allowed or any(p.is_relative_to(x) for x in outputs):return
        if p.suffix in {'.npy','.npz','.safetensors','.pt','.pth'} or (any(p.is_relative_to(x) for x in roots) and p.suffix not in {'.py','.pyc','.so'}):
            raise PermissionError('Undeclared fit/serve file unavailable')
    sys.addaudithook(audit)
    for canary in spec['deniedCanaries']:
        try:open(canary,'rb')
        except PermissionError:continue
        raise RuntimeError('Fit/serve file quarantine inactive')
    return {'mode':mode,'allowedReadFiles':sorted(allowed),'outputRoots':[str(p) for p in outputs],
        'deniedCanaries':spec['deniedCanaries'],'scope':'Python trusted-code audit hook, not an adversarial sandbox'}


def fit(raw,target,pairs,mask,ridge=.01):
    x0=np.asarray(raw,dtype=np.float64);y=np.asarray(target,dtype=np.float64)
    if x0.ndim!=3 or y.ndim!=3 or x0.shape[:2]!=y.shape[:2] or not np.isfinite(x0).all() or not np.isfinite(y).all():raise ValueError('Discovery field shape/finite contract differs')
    flat=x0.reshape(-1,x0.shape[-1]);mean=flat.mean(0);scale=flat.std(0);scale[scale==0]=1
    x=(flat-mean)/scale;yy=y.reshape(-1,y.shape[-1]);xm=x.mean(0);ym=yy.mean(0);xc=x-xm
    h=xc.T@xc/len(x)+ridge*np.eye(x.shape[1]);rhs=xc.T@(yy-ym)/len(x)
    delta_x=np.concatenate([((x0[b]-x0[a])/scale)[mask] for a,b in pairs])
    delta_y=np.concatenate([(y[b]-y[a])[mask] for a,b in pairs])
    h+=delta_x.T@delta_x/len(delta_x);rhs+=delta_x.T@delta_y/len(delta_x)
    weight=np.linalg.solve(h,rhs);bias=ym-xm@weight
    model={'mean':mean,'scale':scale,'weight':weight,'bias':bias}
    if any(not np.isfinite(v).all() for v in model.values()):raise ValueError('Nonfinite proposed coefficients')
    return model,{'discoveryFields':len(raw),'discoveryRows':len(x),'features':x.shape[-1],'outputs':yy.shape[-1],
        'pairedRows':len(delta_x),'ridge':ridge,'solver':'numpy.linalg.solve, FP64 centered normal equations; intercept unregularized',
        'gramNominalMACs':(len(x)+len(delta_x))*x.shape[-1]**2,
        'rhsNominalMACs':(len(x)+len(delta_x))*x.shape[-1]*yy.shape[-1],
        'biasNominalMACs':x.shape[-1]*yy.shape[-1],
        'solveEstimatedDenseLUFLOPs':2*x.shape[-1]**3/3+2*x.shape[-1]**2*yy.shape[-1],
        'costScope':'Nominal contraction sizes and approximate dense-LU operation count; excludes reductions/standardization/pivot bookkeeping and is not a hardware FLOP measurement.',
        'postFitTrainingWriterProducts':2,'postFitTrainingWriterNominalMACs':(len(x)+len(delta_x))*x.shape[-1]*yy.shape[-1],
        'meanTargetLoss':float(np.mean((x@weight+bias-yy)**2)),
        'meanPairedLoss':float(np.mean((delta_x@weight-delta_y)**2))}


def predict(model,z,previous,head,zero_read=False):
    z,previous,head=[np.asarray(a,dtype=np.float64) for a in [z,previous,head]]
    if z.shape!=previous.shape or z.shape[:-1]!=head.shape[:-1]:raise ValueError('Writer fields differ')
    if zero_read:head=np.zeros_like(head)
    x=np.concatenate([z,previous,head],-1).reshape(-1,len(model['mean']))
    correction=((x-model['mean'])/model['scale'])@model['weight']+model['bias']
    raw=previous+correction.reshape(previous.shape)
    bits=words(raw)
    if not np.isfinite(floats(bits)).all():raise ValueError('Nonfinite emitted velocity')
    return raw,bits


def fit_packet(spec_path):
    guard();spec=read(spec_path);quarantine=file_guard(spec_path,spec,'fit');packet=spec['packet']
    if sha(packet['path'])!=packet['sha256']:raise ValueError('Discovery packet differs')
    start=time.perf_counter();cpu=time.process_time()
    with np.load(packet['path'],allow_pickle=False) as data:
        if set(data.files)!={'features','targets','pairs','mask'}:raise ValueError('Discovery packet surface differs')
        model,info=fit(data['features'],data['targets'],data['pairs'].tolist(),data['mask'],.01)
    out=Path(spec['output']);out.mkdir(exist_ok=False)
    fields={n:retain(out,n,v) for n,v in model.items()}
    info.update(schema='tce-source-native-fit-v1',fields=fields,discoveryPacketSHA256=packet['sha256'],
        coefficientBytes=sum(v.nbytes for v in model.values()),wallSeconds=time.perf_counter()-start,cpuSeconds=time.process_time()-cpu,
        fileQuarantine=quarantine,roundingPolicy='FP64 raw writer -> FP32 -> BF16 round-to-nearest-even',frameworksLoaded=[],
        scope='Discovery packet only; no original model, target evaluation, feedback selection, epochs, gain tuning or neural frameworks.')
    save(out/'fit.json',info)


def serve_packet(spec_path):
    guard();spec=read(spec_path);quarantine=file_guard(spec_path,spec,'serve');start=time.perf_counter();cpu=time.process_time()
    sealed=read(spec['coefficientSeal']);info=read(spec['fitReceipt'])
    if sha(spec['fitReceipt'])!=sealed['fitReceiptSHA256'] or info['fields']!=sealed['fields']:raise ValueError('Coefficient seal differs')
    model={n:load(x) for n,x in info['fields'].items()};rows=[]
    for item in spec['inputs']:
        z=load(item['z']);previous=load(item['previous']);head=load(item['H0']);out=Path(item['output']);out.mkdir(exist_ok=False,parents=True)
        for zero in ([False,True] if item['population']=='main' else [False]):
            tick=time.perf_counter();raw,bits=predict(model,z,previous,head,zero);name='zeroRead' if zero else 'candidate'
            raw_field=retain(out,name+'Raw',raw);field=retain(out,name,bits);field['logicalDtype']='torch.bfloat16'
            rows.append({'id':item['id'],'population':item['population'],'method':name,'inputFields':{k:item[k] for k in ['z','previous','H0']},
                'rawVelocity':raw_field,'velocity':field,'writerNominalMACs':z.shape[1]*len(model['mean'])*model['weight'].shape[1],
                'wallSecondsIncludingRetention':time.perf_counter()-tick})
    save(spec['receipt'],{'schema':'tce-source-native-serving-v1','coefficientSealSHA256':sha(spec['coefficientSeal']),'rows':rows,
        'wallSeconds':time.perf_counter()-start,'cpuSeconds':time.process_time()-cpu,
        'fileQuarantine':quarantine,'roundingPolicy':'FP64 raw writer -> FP32 -> BF16 round-to-nearest-even','frameworksLoaded':[],
        'zeroReadPolicy':'Set raw H0=0 before discovery standardization; retain mean, scale, coefficients, bias and unit gain.',
        'scope':'Fresh optimizer-free FP64 affine writer at fixed unit gain; native H0 was produced from each own input and is charged outside this actor. No targets or feedback selection.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','serve']);p.add_argument('spec');a=p.parse_args()
    (fit_packet if a.action=='fit' else serve_packet)(a.spec)
