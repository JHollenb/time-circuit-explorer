"""Saved-only boundary composition diagnostic. Not a generative rollout."""
import argparse
import json
from pathlib import Path
import sys
import time

from response_chart import np,Image,sha,encode,write

METHODS=['transported','additive','copy_order','copy_color','mirror_color','vertical']
def load(ref):
    path=Path(ref['path'])
    if sha(path)!=ref['sha256']:raise ValueError('Source bytes changed: '+str(path))
    if ref['kind']=='rgb':
        value=np.asarray(Image.open(path).convert('RGB')).astype(np.float64)
    else:
        value=np.load(path,allow_pickle=False)
        if ref['encoding']=='bf16-bits-v1':
            if value.dtype!=np.uint16:raise ValueError('Wrong BF16 archive')
            value=(value.astype(np.uint32)<<16).view(np.float32)
        value=value.astype(np.float64)
    if list(value.shape)!=ref['shape'] or not np.isfinite(value).all():raise ValueError('Source shape/finite contract differs')
    return value
def permutation(ids,axis):
    if ids.ndim!=3 or ids.shape[0]!=1 or ids.shape[-1]!=4:raise ValueError('Image IDs geometry differs')
    coordinates=[tuple(int(v) for v in row) for row in ids[0]]
    lookup={row:i for i,row in enumerate(coordinates)}
    if len(lookup)!=len(coordinates):raise ValueError('Image IDs are not unique')
    lo,hi=min(x[axis] for x in coordinates),max(x[axis] for x in coordinates)
    out=[]
    for row in coordinates:
        other=list(row);other[axis]=hi+lo-other[axis]
        if tuple(other) not in lookup:raise ValueError('Image grid not closed under supplied permutation')
        out.append(lookup[tuple(other)])
    out=np.asarray(out,np.int64)
    if not np.array_equal(out[out],np.arange(len(out))):raise ValueError('Grid permutation is not involutive')
    return out
def transform(value,space,h,v,vertical=False):
    if space=='text':return value
    if space=='rgb':return value[::-1] if vertical else value[:,::-1]
    if value.ndim!=3 or value.shape[1]!=len(h):raise ValueError('Spatial token geometry differs')
    return value[:,v if vertical else h,:]
def candidate(b,c,o,method,space,h,v):
    delta=c-b
    if method=='transported':return o+transform(delta,space,h,v)
    if method=='additive':return o+delta
    if method=='copy_order':return o.copy()
    if method=='copy_color':return c.copy()
    if method=='mirror_color':return transform(c,space,h,v).copy()
    if method=='vertical':return o+transform(delta,space,h,v,vertical=True)
    raise ValueError('Unknown method')
def stored(value,space):
    if space=='rgb':return encode(value).astype(np.float32)
    bits=np.asarray(value,np.float32).view(np.uint32)
    return ((bits+np.uint32(32767)+((bits>>16)&np.uint32(1)))&np.uint32(0xffff0000)).view(np.float32)
def array_sha(value):
    import hashlib
    return hashlib.sha256(np.asarray(value,dtype='<f4',order='C').tobytes()).hexdigest()
def prediction_guard(spec):
    root=Path(spec['sourceRoot']).resolve();allowed={Path(spec['ids']['path']).resolve()}
    allowed.update(Path(ref['path']).resolve() for row in spec['rows'] for ref in row['calibration'])
    opened=[]
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            path=Path(args[0]).resolve();opened.append(str(path))
            if path.name=='evaluation.json' or (path.is_relative_to(root) and path not in allowed):raise PermissionError('Prediction read outside calibration capability: '+str(path))
            if path.suffix in {'.safetensors','.pt','.pth','.bin'}:raise PermissionError('Weights prohibited')
        if event in {'socket.connect','socket.bind'}:raise PermissionError('Network prohibited')
    sys.addaudithook(audit);return opened
def predict(spec_path,out):
    started=time.perf_counter();spec=json.loads(Path(spec_path).read_text());opened=prediction_guard(spec)
    ids=load(spec['ids']);h,v=permutation(ids,2),permutation(ids,1);rows=[]
    for row in spec['rows']:
        b,c,o=[load(ref) for ref in row['calibration']]
        for method in METHODS:
            start=time.perf_counter();raw=candidate(b,c,o,method,row['space'],h,v);value=stored(raw,row['space'])
            item={'id':row['id'],'method':method,'arraySHA256':array_sha(value),'shape':list(value.shape),'seconds':time.perf_counter()-start}
            if row['space']=='rgb':
                path=Path(out)/(method+'.png');Image.fromarray(value.astype(np.uint8)).save(path);item.update(png=path.name,pngSHA256=sha(path))
            rows.append(item)
        print('PREDICTED '+row['id'],flush=True)
    seal={'schema':'tce-prompt-composition-seal-v1','calibrationSpecSHA256':sha(spec_path),'rows':rows,'opened':sorted(set(opened)),
          'wallSeconds':time.perf_counter()-started,'neuralEvaluations':0,'denseWeightMath':False,'scope':spec['scope']}
    write(Path(out)/'prediction-seal.json',seal)
def rms(a):return float(np.sqrt(np.mean(a*a)))
def ratio(a,b):return a/b if b else None
def cosine(a,b):
    den=rms(a)*rms(b);return float(np.mean(a*b))/den if den else None
def score(calibration,evaluation,out):
    spec=json.loads(Path(calibration).read_text());targets=json.loads(Path(evaluation).read_text());out=Path(out)
    seal=json.loads((out/'prediction-seal.json').read_text());before=sha(out/'prediction-seal.json')
    if seal['calibrationSpecSHA256']!=sha(calibration):raise ValueError('Calibration spec changed')
    ids=load(spec['ids']);h,v=permutation(ids,2),permutation(ids,1);rows=[]
    for row,target in zip(spec['rows'],targets['rows'],strict=True):
        if row['id']!=target['id']:raise ValueError('Target address differs')
        b,c,o=[load(ref) for ref in row['calibration']];t=load(target['target']);mixed=t-o-c+b
        correction=transform(c-b,row['space'],h,v)-(c-b);residual=mixed-correction
        decomposition={'mixedRms':rms(mixed),'transportCorrectionRms':rms(correction),'transportResidualRms':rms(residual),
            'correctionOverMixed':ratio(rms(correction),rms(mixed)),'residualOverMixed':ratio(rms(residual),rms(mixed)),
            'cosine':cosine(mixed,correction),'signedDot':float(np.mean(mixed*correction)),
            'identityMaxAbs':float(np.max(np.abs((mixed-correction)-(t-o-transform(c-b,row['space'],h,v)))))}
        for method in METHODS:
            forecast=next(x for x in seal['rows'] if x['id']==row['id'] and x['method']==method)
            raw=candidate(b,c,o,method,row['space'],h,v);value=stored(raw,row['space']);error=value.astype(np.float64)-t
            if array_sha(value)!=forecast['arraySHA256']:raise ValueError('Scored candidate differs from pretarget seal')
            if row['space']=='rgb' and sha(out/forecast['png'])!=forecast['pngSHA256']:raise ValueError('Saved PNG changed')
            rows.append({'id':row['id'],'space':row['space'],'method':method,'step':row.get('step'),'local':row.get('local'),'field':row['field'],
                'endpointRms':rms(error),'endpointRelativeRms':ratio(rms(error),rms(t)),'mae':float(np.mean(np.abs(error))),
                'nativeColorEffectRms':rms(t-o),'errorOverNativeColorEffect':ratio(rms(error),rms(t-o)),
                'colorResponseCosine':cosine(value-o,t-o),'rawMixedErrorRelative':ratio(rms(raw-(o+c-b)-mixed),rms(mixed)),
                'roundingOrClipRms':rms(value.astype(np.float64)-raw), 'decomposition':decomposition})
        print('SCORED '+row['id'],flush=True)
    if sha(out/'prediction-seal.json')!=before:raise ValueError('Prediction seal mutated')
    return {'rows':rows,'predictionSealSHA256':before,'nativeTargets':len(targets['rows']),'scope':spec['scope']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['predict']);p.add_argument('spec',type=Path);p.add_argument('out',type=Path);a=p.parse_args();predict(a.spec,a.out)
