"""Source-bound FP64 response fitting/serving, with no neural imports in actors."""
import argparse
import os
from pathlib import Path
import sys
import time
from encoder_atoms import np,read,save,sha,guard
from paired_fields import load,retain,metrics


def mask(location):
    edited=np.zeros((32,32),bool)
    if location=='bottom-right':edited[-8:,-8:]=True
    elif location=='top-left':edited[:8,:8]=True
    else:raise ValueError('Unknown declared edit location')
    halo=edited.copy();halo[1:]|=edited[:-1];halo[:-1]|=edited[1:];halo[:,1:]|=edited[:,:-1];halo[:,:-1]|=edited[:,1:]
    return ~halo.ravel()


def features(pair,scale):
    selected=mask(pair['location']);a,b=pair['A'],pair['B']
    za,zb=load(a['z']),load(b['z']);na,nb=load(a['previous']),load(b['previous'])
    same=np.all(za==zb,axis=-1).ravel()&np.all(na==nb,axis=-1).ravel()
    applicable=selected&same
    h=(load(b['H0'])-load(a['H0']))[0]/scale
    if h.shape!=(1024,3072) or scale.shape!=(3072,) or not np.isfinite(h).all():raise ValueError('Head-response contract differs')
    return h[applicable],applicable,{'declaredQueries':int(selected.sum()),'applicableQueries':int(applicable.sum()),'inapplicableQueries':np.flatnonzero(selected&~same).tolist()}


def ridge(x,y,ridge=.01):
    x,y=np.asarray(x,np.float64),np.asarray(y,np.float64)
    if x.ndim!=2 or y.ndim!=2 or x.shape[0]!=y.shape[0] or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Discovery response geometry/finite contract differs')
    # Exact ridge identity: (X'X+NλI)^−1 X'Y = X'(XX'+NλI)^−1Y.
    n,d=x.shape;o=y.shape[1];gram=x@x.T;gram.flat[::n+1]+=n*ridge
    alpha=np.linalg.solve(gram,y);weight=x.T@alpha
    if not np.isfinite(weight).all():raise ValueError('Nonfinite response proposal')
    return weight,{'rows':n,'features':d,'outputs':o,'ridge':ridge,'intercept':False,'gain':1,
        'solver':'FP64 dual ridge, numpy.linalg.solve(XXT+N*ridge*I,Y), B=XT*alpha',
        'gramNominalMACs':n*n*d,'coefficientNominalMACs':d*n*o,'estimatedSolveFLOPs':2*n**3/3+2*n*n*o,
        'costScope':'Shape-derived contraction sizes/approximate dense-LU solve, not hardware counters; reductions and data movement separate.'}


def quarantine(spec_path,spec,allowed,outputs):
    allowed={str(Path(x).resolve()) for x in [spec_path,*allowed]};outputs=[Path(p).resolve() for p in outputs]
    roots=[Path(spec_path).resolve().parent,*[Path(p).resolve() for p in spec['protectedRoots']]]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if str(p) in allowed or any(p.is_relative_to(r) for r in outputs):return
        if p.suffix in {'.npy','.npz','.safetensors','.pt','.pth'} or (any(p.is_relative_to(r) for r in roots) and p.suffix not in {'.py','.pyc','.so'}):raise PermissionError('Undeclared response actor file')
    sys.addaudithook(audit)
    for canary in spec['deniedCanaries']:
        try:open(canary,'rb')
        except PermissionError:continue
        raise RuntimeError('Response target quarantine inactive')
    return {'allowedReadFiles':sorted(allowed),'outputRoots':[str(p) for p in outputs],'deniedCanaries':spec['deniedCanaries'],'scope':'Trusted-code Python audit hook, not adversarial sandbox.'}


def input_files(pairs):
    return [x['path'] for pair in pairs for side in ['A','B'] for x in pair[side].values()]


def fit(spec_path):
    guard();s=read(spec_path);out=Path(s['output']);out.mkdir(exist_ok=False)
    allowed=input_files(s['pairs'])+[s['scale']['path']]+[x['path'] for pair in s['pairs'] for x in pair['targets']]
    q=quarantine(spec_path,s,allowed,[out]);tick=time.perf_counter();cpu=time.process_time();scale=load(s['scale'])[256:];xs=[];ys=[];pairs=[]
    for pair in s['pairs']:
        x,selected,app=features(pair,scale);y=(load(pair['targets'][1])-load(pair['targets'][0]))[0,selected]
        if app['declaredQueries']!=944 or app['applicableQueries']!=944:raise ValueError('Discovery local-operand integrity differs')
        xs.append(x);ys.append(y);pairs.append({'id':pair['id'],'X':retain(out/pair['id'],'X',x),'Y':retain(out/pair['id'],'Y',y),'mask':retain(out/pair['id'],'mask',selected),**app})
    x,y=np.concatenate(xs),np.concatenate(ys);weight,info=ridge(x,y);prediction=x@weight
    fields={'weight':retain(out,'weight',weight),'baseline':retain(out,'baseline',np.zeros_like(weight))}
    info.update(schema='tce-source-response-fit-v1',fields=fields,discoveryPairs=pairs,sourceScale=s['scale'],sourceInputs=s['pairs'],
        meanSquaredError=float(np.mean((prediction-y)**2)),postFitNominalMACs=len(x)*x.shape[1]*y.shape[1],
        coefficientBytes=weight.nbytes,wallSeconds=time.perf_counter()-tick,cpuSeconds=time.process_time()-cpu,
        numpyVersion=np.__version__,pythonVersion=sys.version,frameworksLoaded=[],fileQuarantine=q)
    save(out/'fit.json',info)


def serve(spec_path):
    guard();s=read(spec_path);out=Path(s['output']);out.mkdir(exist_ok=False);seal=read(s['coefficientSeal']);fit_info=read(s['fitReceipt'])
    if sha(s['fitReceipt'])!=seal['fitReceiptSHA256'] or fit_info['fields']!=seal['fields']:raise ValueError('Response coefficient seal differs')
    allowed=input_files(s['pairs'])+[s['scale']['path'],s['oldWeight']['path'],s['coefficientSeal'],s['fitReceipt'],fit_info['fields']['weight']['path']]
    q=quarantine(spec_path,s,allowed,[out]);tick=time.perf_counter();cpu=time.process_time();scale=load(s['scale'])[256:];weight=load(fit_info['fields']['weight']);old=load(s['oldWeight'])[256:]
    rows=[];zeros={}
    for pair in s['pairs']:
        x,selected,app=features(pair,scale);fields={'X':retain(out/pair['id'],'X',x),'mask':retain(out/pair['id'],'mask',selected)}
        shape=(len(x),128)
        if shape not in zeros:zeros[shape]=retain(out,'zero-'+str(len(x)),np.zeros(shape))
        for method,w in [('response-only',weight),('original-writer',old),('zero',None)]:
            prediction=zeros[shape] if w is None else retain(out/pair['id'],method,x@w)
            rows.append({'id':pair['id'],'population':pair['population'],'location':pair['location'],'method':method,'fields':fields,'prediction':prediction,
                'nominalMACs':0 if w is None else len(x)*3072*128,**app})
    save(out/'serving.json',{'schema':'tce-source-response-serving-v1','rows':rows,'coefficientSealSHA256':sha(s['coefficientSeal']),'oldWeight':s['oldWeight'],'sourceScale':s['scale'],
        'wallSeconds':time.perf_counter()-tick,'cpuSeconds':time.process_time()-cpu,'numpyVersion':np.__version__,'frameworksLoaded':[],'fileQuarantine':q,
        'scope':'Raw FP64 response only, no intercept/requantization/scheduler/images. Zero arrays physically shared by geometry. No target reads.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','serve']);p.add_argument('spec');a=p.parse_args();globals()[a.action](a.spec)
