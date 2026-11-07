"""One fixed-ridge source-spectrum diagnosis; no new serving program."""
from encoder_atoms import np


def analyse(x,y,old_prediction,ridge=.01):
    x,y,old_prediction=[np.asarray(v,np.float64) for v in [x,y,old_prediction]]
    if x.ndim!=2 or y.ndim!=2 or x.shape[0]!=len(y) or y.shape!=old_prediction.shape or not all(np.isfinite(v).all() for v in [x,y,old_prediction]):raise ValueError('Source geometry/finite contract differs')
    import time
    n,d=x.shape;o=y.shape[1];tick=time.perf_counter();u,s,vt=np.linalg.svd(x,full_matrices=False);svd_seconds=time.perf_counter()-tick
    k=len(s);tau=n*ridge;c=u.T@y;response_filter=s*s/(s*s+tau);residual_filter=tau/(s*s+tau)
    projection=u@c;prediction=(u*response_filter)@c;perp=y-projection;modal=projection-prediction;residual=y-prediction;old_error=prediction-old_prediction
    mode_energy=np.sum(c*c,axis=1);residual_energy=residual_filter**2*mode_energy
    ridge_coefficient_energy=s*s/(s*s+tau)**2*mode_energy
    x_error=(u*s)@vt-x
    rows=[]
    for i in range(k):
        if s[i]==0:amplification=None;status='undefined-zero-singular-value'
        else:
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):value=mode_energy[i]/(s[i]*s[i])
            amplification=float(value) if np.isfinite(value) else None;status='finite' if amplification is not None else 'nonfinite-floating-result'
        rows.append({'position':i,'singularValue':float(s[i]),'relativeSigma':float(s[i]/s[0]) if s[0] else None,
            'responseFilter':float(response_filter[i]),'residualFilter':float(residual_filter[i]),'targetEnergy':float(mode_energy[i]),
            'ridgeResidualEnergy':float(residual_energy[i]),'ridgeCoefficientEnergy':float(ridge_coefficient_energy[i]),
            'pseudoinverseCoefficientEnergy':amplification,'pseudoinverseStatus':status})
    energy=lambda a:float(np.sum(a*a));maxabs=lambda a:float(np.max(np.abs(a)))
    summary={'rows':n,'features':d,'outputs':o,'thinDirections':k,'ridge':ridge,'tau':tau,'svdWallSeconds':svd_seconds,
        'singularMaximum':float(s[0]),'singularMinimum':float(s[-1]),'singularSumSquares':float(s@s),
        'sourceInputEnergy':energy(x),'targetEnergy':energy(y),'outsideThinUEnergy':energy(perp),'modalResidualEnergy':energy(modal),
        'spectralResidualEnergy':float(residual_energy.sum()),'actualResidualEnergy':energy(residual),
        'perpModalTwiceCross':float(2*np.sum(perp*modal)),
        'energyClosure':energy(residual)-energy(perp)-energy(modal)-float(2*np.sum(perp*modal)),
        'spectralVsModalEnergyDifference':float(residual_energy.sum())-energy(modal),
        'originalPredictionDifferenceEnergy':energy(old_error),'originalPredictionDifferenceMaxAbs':maxabs(old_error),
        'sourceSVDReconstructionRMS':float(np.sqrt(np.mean(x_error*x_error))),'sourceSVDReconstructionMaxAbs':maxabs(x_error),
        'ridgeCoefficientEnergy':float(ridge_coefficient_energy.sum()),'undefinedPseudoinverseDirections':sum(r['pseudoinverseStatus']!='finite' for r in rows),
        'thinUIsSquare':u.shape[0]==u.shape[1],
        'rankPolicy':'No numerical rank chosen or predictor selected. Full mode/attenuation/target-energy records; a square thin U complement is not an information-sufficiency test.',
        'solver':'numpy.linalg.svd(full_matrices=False), FP64; one fixed ridge reconstruction',
        'contractionNominalMACs':{'targetProjection':k*n*o,'thinTargetReconstruction':n*k*o,'ridgePredictionReconstruction':n*k*o,'sourceSVDReconstruction':n*k*d},
        'SVDArithmeticScope':'SVD work measured by CPU/wall and geometry; not included in nominal contraction MACs or represented as a hardware FLOP count.'}
    arrays={'U':u,'sigma':s,'Vt':vt,'targetModes':c,'responseFilter':response_filter,'residualFilter':residual_filter,
        'modeTargetEnergy':mode_energy,'modeResidualEnergy':residual_energy,'modeRidgeCoefficientEnergy':ridge_coefficient_energy,
        'reconstructedPrediction':prediction,'outsideThinU':perp,'modalResidual':modal,'sourceResidual':residual,'originalPredictionDifference':old_error}
    if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite SVD/reconstruction execution field')
    return arrays,rows,summary
