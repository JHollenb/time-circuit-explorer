"""One all-mode minimum-norm response map, isolated from held targets."""
import argparse
from pathlib import Path
import sys
import time
from encoder_atoms import np,read,save,sha,guard
from paired_fields import load,retain
from source_response import quarantine,mask


def minimum_norm(vt,s,c):
    vt,s,c=[np.asarray(v,np.float64) for v in [vt,s,c]]
    if vt.ndim!=2 or s.ndim!=1 or c.ndim!=2 or len(s)!=len(vt) or len(c)!=len(s) or not len(s):raise ValueError('Factor geometry differs')
    if not all(np.isfinite(v).all() for v in [vt,s,c]) or np.any(s<=0):raise ValueError('Unresolved/nonfinite mode; no silent truncation')
    scaled=c/s[:,None]
    if not np.isfinite(scaled).all():raise ValueError('Nonfinite inverse-mode response')
    weight=vt.T@scaled
    if not np.isfinite(weight).all():raise ValueError('Nonfinite minimum-norm coefficient')
    return weight,{'modes':len(s),'features':vt.shape[1],'outputs':c.shape[1],'coefficientNominalMACs':vt.shape[1]*len(s)*c.shape[1],
        'coefficientBytes':weight.nbytes,'coefficientEnergy':float(np.sum(weight*weight)),'modeCoefficientEnergy':float(np.sum(scaled*scaled)),
        'intercept':False,'gain':1,'regularization':0,'allModesUsed':True,'solver':'Retained FP64 Vt.T @ (C / sigma); no new SVD or rank selection'}


def form(spec_path):
    guard();s=read(spec_path);out=Path(s['output']);out.mkdir(exist_ok=False)
    allowed=[x['path'] for x in s['factors'].values()]+[s['scale']['path']]
    q=quarantine(spec_path,s,allowed,[out]);tick=time.perf_counter();cpu=time.process_time();scale=load(s['scale'])
    if scale.shape!=(3328,) or np.any(scale<=0):raise ValueError('Inherited scale differs')
    weight,info=minimum_norm(*[load(s['factors'][k]) for k in ['Vt','sigma','targetModes']])
    if [info['modes'],info['features'],info['outputs']]!=[1888,3072,128]:raise ValueError('Frozen factor dimensions differ')
    info.update(schema='tce-minimum-response-fit-v1',fields={'weight':retain(out,'weight',weight)},sourceFactors=s['factors'],sourceScale=s['scale'],
        wallSeconds=time.perf_counter()-tick,cpuSeconds=time.process_time()-cpu,numpyVersion=np.__version__,pythonVersion=sys.version,frameworksLoaded=[],fileQuarantine=q)
    save(out/'fit.json',info)


def serve(spec_path):
    guard();s=read(spec_path);out=Path(s['output']);out.mkdir(exist_ok=False);seal=read(s['coefficientSeal']);fit=read(s['fitReceipt'])
    if sha(s['fitReceipt'])!=seal['fitReceiptSHA256'] or fit['fields']!=seal['fields'] or fit['sourceScale']!=s['scale']:raise ValueError('Coefficient/source seal differs')
    allowed=[s['fitReceipt'],s['coefficientSeal'],s['scale']['path'],fit['fields']['weight']['path']]+[x['path'] for p in s['pairs'] for x in p['fields'].values()]
    q=quarantine(spec_path,s,allowed,[out]);tick=time.perf_counter();cpu=time.process_time();weight=load(fit['fields']['weight']);scale=load(s['scale'])
    if scale.shape!=(3328,) or np.any(scale<=0) or weight.shape!=(3072,128):raise ValueError('Serving geometry differs')
    rows=[]
    for pair in s['pairs']:
        x=load(pair['fields']['X']);selected=load(pair['fields']['mask']).astype(bool)
        if x.shape!=(944,3072) or not np.array_equal(selected,mask(pair['location'])):raise ValueError('Archived input/mask differs')
        rows.append({**pair,'method':'minimum-norm','prediction':retain(out/pair['id'],'minimum-norm',x@weight),'nominalMACs':944*3072*128})
    save(out/'serving.json',{'schema':'tce-minimum-response-serving-v1','rows':rows,'coefficientSealSHA256':sha(s['coefficientSeal']),'sourceScale':s['scale'],
        'wallSeconds':time.perf_counter()-tick,'cpuSeconds':time.process_time()-cpu,'numpyVersion':np.__version__,'pythonVersion':sys.version,'frameworksLoaded':[],
        'fileQuarantine':q,'scope':'Eight raw FP64 responses from archived normalized inputs. No targets, new H0, output quantization or images.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['form','serve']);p.add_argument('spec');a=p.parse_args();globals()[a.action](a.spec)
