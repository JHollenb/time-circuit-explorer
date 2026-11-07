"""Post-seal own-diagnostic scalars; no additional learned projection or future."""
from pathlib import Path
import json
import shutil
import sys
import time
from encoder_atoms import read,save,sha,np
from paired_fields import load,metrics
from gated_covariance import ROW_COLUMNS


def execute(root):
    from flux_clock_study import verify
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);out=root/'response-result';started=time.perf_counter()
    s=read(root/'inputs.json');c=read(out/'candidate.json');e=read(out/'execution.json');seal=read(out/'candidate-seal.json');seal_sha=sha(out/'candidate-seal.json')
    if seal!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')}:raise ValueError('Diagnostic not sealed before analysis')
    row=load(c['fields']['rowStatistics'])[0];grams=load(c['fields']['stratumGram']);panels={}
    for index,(name,part) in enumerate([('all',slice(None)),('text',slice(0,512)),('image',slice(512,None))]):
        values=row[part];g=grams[index];total=float(values[:,6].sum());oracle=float(values[:,7].sum());mean_r=float(values[:,8].sum());energy=lambda x:float(x)
        cosines={label:float(g[a,b]/np.sqrt(g[a,a]*g[b,b])) if g[a,a]>0 and g[b,b]>0 else None for label,a,b in [('meanCentered',0,1),('meanPolicy',0,2),('centeredPolicy',1,2)]}
        panels[name]={'rows':len(values),'gram':g.tolist(),'componentCosines':cosines,'totalGatedErrorEnergy':total,'gramReconstructedEnergy':float(g.sum()),'gramEnergyClosureResidual':float(g.sum()-total),
            'meanCovarianceWriteEnergy':energy(g[0,0]),'centeredWriteEnergy':energy(g[1,1]),'combinedPolicyReassociationEnergy':energy(g[2,2]),
            'twiceMeanCenteredInner':energy(2*g[0,1]),'twiceMeanPolicyInner':energy(2*g[0,2]),'twiceCenteredPolicyInner':energy(2*g[1,2]),
            'bestSuppliedConstantResidualEnergy':oracle,'bestSuppliedConstantRemovedEnergyFraction':(total-oracle)/total if total else None,
            'meanCovarianceCorrectionResidualEnergy':mean_r,'meanCovarianceCorrectionRemovedEnergyFraction':(total-mean_r)/total if total else None,
            'oracleResidualOverOriginalRMS':float(np.sqrt(oracle/total)) if total else None,'meanCovarianceResidualOverOriginalRMS':float(np.sqrt(mean_r/total)) if total else None,
            'meanCorrectionImprovedRows':int(np.sum(values[:,8]<values[:,6])),'meanCorrectionWorsenedRows':int(np.sum(values[:,8]>values[:,6])),
            'oracleCorrectionImprovedRows':int(np.sum(values[:,7]<values[:,6])),'oracleCorrectionWorsenedRows':int(np.sum(values[:,7]>values[:,6]))}
    with (out/'observations.jsonl').open('x') as stream:
        for index,values in enumerate(row):
            stream.write(json.dumps({'row':index,'stratum':'text' if index<512 else 'image',**{name:float(x) for name,x in zip(ROW_COLUMNS,values)},
                'meanCorrectionEnergyImprovement':float(values[6]-values[8]),'oracleCorrectionEnergyImprovement':float(values[6]-values[7])},sort_keys=True,allow_nan=False)+'\n')
    direct=load(c['fields']['covarianceDirect']);product_mean=load(c['fields']['centeredProductMean']);mu=load(c['fields']['upMean']);anchor=load(s['sourceFields']['anchorUp']);support=set(load(s['sourceFields']['selection_support']).astype(np.int64).tolist())
    covariance={'metrics':metrics(product_mean,direct),'scope':'Two finite FP64 reductions of population covariance on admittedA and newlypaidFP64U; no tolerancegate.'}
    with (out/'channel-observations.jsonl').open('x') as stream:
        for k,name in enumerate(['text','image']):
            for channel in range(9216):stream.write(json.dumps({'stratum':name,'channel':channel,'selected':channel in support,'directCovariance':float(direct[0,k,channel]),'meanCenteredProduct':float(product_mean[0,k,channel]),'covarianceResidual':float(product_mean[0,k,channel]-direct[0,k,channel]),'meanFP64Up':float(mu[0,k,channel]),'savedAnchor':float(anchor[0,k,channel]),'meanUpMinusSavedAnchor':float(mu[0,k,channel]-anchor[0,k,channel])},sort_keys=True,allow_nan=False)+'\n')
    diagnostics={}
    for name,reference in [('policyExpansionResidual','policyProduct'),('outputClosure','gatedTotal')]:
        a=load(c['fields'][name]);b=load(c['fields'][reference]);den=float(np.linalg.norm(b));diagnostics[name]={'rms':float(np.sqrt(np.mean(a*a))),'maxAbs':float(np.abs(a).max()),'relativeNorm':float(np.linalg.norm(a))/den if den else None,'reference':reference}
    product_policy={}
    for name in ['selectedAdmissionProduct','omittedAnchorAdmissionProduct','productRounding','policyProduct']:
        a=load(c['fields'][name]);product_policy[name]={'energy':float(np.sum(a*a)),'rms':float(np.sqrt(np.mean(a*a))),'signedMean':float(a.mean()),'min':float(a.min()),'max':float(a.max()),'scope':'Productspaceonly; not separately downprojected or a causalimageattribution.'}
    if sha(out/'candidate-seal.json')!=seal_sha or sha(out/'candidate.json')!=seal['candidateSHA256'] or sha(out/'execution.json')!=seal['executionSHA256']:raise ValueError('Diagnostic changed after release')
    save(out/'report.json',{'schema':'tce-own-state-gated-covariance-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal_sha,'fixedAddress':{'clock':3,'singleBlock':0},'panels':panels,'covarianceIdentity':covariance,'reductionDiagnostics':diagnostics,'productPolicyObservations':product_policy,
        'observations':1536,'channelObservations':18432,'totalMatrixMACs':e['totalMatrixMACs'],'newFP64FullUpReferences':1,'newDownContractions':2,'newImages':0,'newContinuations':0,'analysisSecondsAfterImports':time.perf_counter()-started,
        'mechanics_status':'complete','instrument_status':'one-source-qualified-own-state-representation-diagnostic','trend_status':'All signed energies/Gram terms and rowheterogeneity retained','terminal_status':'unassessed',
        'scope':'One fixedpaidownclock3single0. NewlypaidFP64U and2Downs reconstruct MLPwrites,notnativefiniteconcatblockerror. A/gate/currentS supplied,originalmaps/historicalgenerationpaid. Bestconstant oracle representationfloor only,no fittedproducer/correctionapplication/head/image/zero-neuralclaim. YQ containsunseparatedFP64distributivity error.'})
    for name in ['request.json','inputs.json','resource-basis.json','frozen-plan.md']:shutil.copyfile(root/name,out/name)
    print('GATED_COVARIANCE_ANALYZED',flush=True)


if __name__=='__main__':execute(sys.argv[1])
