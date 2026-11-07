"""Fixed-map source-span and inverse-mode geometry, without old forecasts."""
import argparse
import json
from pathlib import Path
import sys
import time
from encoder_atoms import np,read,save,guard
from paired_fields import load,retain
from source_response import quarantine,mask


def analyse(x,vt,s,c,filters,indices):
    x,vt,s,c,filters=[np.asarray(v,np.float64) for v in [x,vt,s,c,filters]];indices=np.asarray(indices)
    if x.ndim!=2 or vt.ndim!=2 or s.ndim!=1 or c.ndim!=2 or len(s)!=len(vt) or len(c)!=len(s) or x.shape[1]!=vt.shape[1] or filters.shape!=s.shape or indices.shape!=(len(x),):raise ValueError('Extension geometry differs')
    if np.any(s<=0) or not all(np.isfinite(v).all() for v in [x,vt,s,c,filters]):raise ValueError('Nonfinite/unresolved fixed factor')
    z=x@vt.T;a=z/s;parallel=z@vt;residual=x-parallel;prediction=a@c;d=a.T@prediction
    if not all(np.isfinite(v).all() for v in [z,a,parallel,residual,prediction,d]):raise ValueError('Nonfinite extension execution field')
    energy=lambda v,axis:np.sum(v*v,axis=axis)
    xe,pe,re=[energy(v,1) for v in [x,parallel,residual]];cross=2*np.sum(parallel*residual,axis=1);closure=xe-pe-re-cross;leverage=energy(a,1)
    ae=energy(a,0);ce=energy(c,1);diagonal=ae*ce;power=np.sum(c*d,axis=1);output=float(np.sum(prediction*prediction))
    queries=[{'selectedRowIndex':i,'imageQueryIndex':int(indices[i]),'inputEnergy':float(xe[i]),'parallelEnergy':float(pe[i]),'residualEnergy':float(re[i]),
        'twiceParallelResidual':float(cross[i]),'energyClosure':float(closure[i]),'leverage':float(leverage[i]),'predictionEnergy':float(np.sum(prediction[i]**2))} for i in range(len(x))]
    modes=[{'position':i,'singularValue':float(s[i]),'responseFilter':float(filters[i]),'inverseCoordinateEnergy':float(ae[i]),'sourceTargetModeEnergy':float(ce[i]),
        'diagonalPredictionEnergy':float(diagonal[i]),'signedPredictionPower':float(power[i]),'crossPowerContribution':float(power[i]-diagonal[i])} for i in range(len(s))]
    summary={'inputEnergy':float(xe.sum()),'parallelEnergy':float(pe.sum()),'residualEnergy':float(re.sum()),'twiceParallelResidual':float(cross.sum()),
        'energyClosure':float(closure.sum()),'maxQueryEnergyClosureAbs':float(np.max(np.abs(closure))),'residualEnergyFraction':float(re.sum()/xe.sum()) if xe.sum() else None,
        'leverageSum':float(leverage.sum()),'leverageMin':float(leverage.min()),'leverageMedian':float(np.median(leverage)),'leverageMax':float(leverage.max()),
        'predictionEnergy':output,'diagonalPredictionEnergy':float(diagonal.sum()),'signedModalPowerSum':float(power.sum()),'modalPowerClosure':float(power.sum()-output),
        'crossPredictionPower':float(output-diagonal.sum()),'negativeModalPowers':int(np.count_nonzero(power<0)),
        'nominalLeadingMACs':2*len(x)*x.shape[1]*len(s)+2*len(x)*len(s)*c.shape[1]}
    return {'inverseCoordinates':a,'reconstructedPrediction':prediction,'modalContraction':d},queries,modes,summary


def execute(spec_path):
    guard();s=read(spec_path);out=Path(s['output']);out.mkdir(exist_ok=False,parents=True);arrays=Path(s['arrays']);arrays.mkdir(exist_ok=False)
    allowed=[x['path'] for x in s['factors'].values()]+[x['path'] for p in s['pairs'] for x in p['fields'].values()]
    q=quarantine(spec_path,s,allowed,[out,arrays]);tick=time.perf_counter();cpu=time.process_time();vt,sigma,c,filters=[load(s['factors'][k]) for k in ['Vt','sigma','targetModes','responseFilter']]
    if vt.shape!=(1888,3072) or sigma.shape!=(1888,) or c.shape!=(1888,128):raise ValueError('Frozen factor dimensions differ')
    rows=[];retained={};nq=nm=0
    with (out/'queries.jsonl').open('x') as qs,(out/'modes.jsonl').open('x') as ms:
        for pair in s['pairs']:
            x=load(pair['fields']['X']);selected=load(pair['fields']['mask']).astype(bool)
            if x.shape!=(944,3072) or not np.array_equal(selected,mask(pair['location'])):raise ValueError('Archived input/query mask differs')
            fields,queries,modes,summary=analyse(x,vt,sigma,c,filters,np.flatnonzero(selected));label={k:pair[k] for k in ['id','population','location']}
            retained[pair['id']]={n:retain(arrays/pair['id'],n,v) for n,v in fields.items()};rows.append({**label,**summary})
            for row in queries:qs.write(json.dumps({**label,**row},allow_nan=False)+'\n');nq+=1
            for row in modes:ms.write(json.dumps({**label,**row},allow_nan=False)+'\n');nm+=1
    save(out/'geometry.json',{'schema':'tce-extension-geometry-v1','pairs':rows,'fields':retained,'queryRecords':nq,'modeRecords':nm,'frameworksLoaded':[],'fileQuarantine':q,
        'wallSecondsIncludingAccessRetention':time.perf_counter()-tick,'cpuSecondsIncludingAccessRetention':time.process_time()-cpu,'numpyVersion':np.__version__,'pythonVersion':sys.version,
        'sourceLeverageReference':{'value':1,'scope':'Exact full-row-rank source identity; no new source leverage measurement.'}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('spec');a=p.parse_args();execute(a.spec)
