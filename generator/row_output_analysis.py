"""Post-exit primary82/secondary73 and signed full/centered error accounts."""
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from encoder_atoms import read,save,sha
from paired_fields import load,retain
from gate_up_basis_analysis import statistics,panel,COLUMNS

PORTS=['preResidualUpdate','exit','exitIncrement']
ROW_COLUMNS=[name+'_'+col for name in ['vs82','vs73','old82vs73'] for col in COLUMNS]+['twiceOldDeltaInner','twiceCenteredOldDeltaInner','errorIdentityClosureEnergy','centeredIdentityClosureEnergy']


def analysis_abis(text=512,image=1024,rank=128):
    n=text+image
    return {'coordinateResidual':{'shape':[1,n,rank],'dtype':'torch.float64'},'basisGram':{'shape':[2,rank,rank],'dtype':'torch.float64'},
            'coordinateRowStatistics':{'shape':[n,8],'dtype':'float64'},'rowStatistics':{'shape':[n,len(PORTS),len(ROW_COLUMNS)],'dtype':'float64'}}


def accounts(candidate,old,target,text=512):
    def center(value):return np.concatenate([value[:,:text]-value[:,:text].mean(1,keepdims=True),value[:,text:]-value[:,text:].mean(1,keepdims=True)],1)
    delta=candidate-old;prior=old-target;total=candidate-target;closure=total-prior-delta;cp,cd,ct=map(center,[prior,delta,total]);cc=ct-cp-cd
    extras=np.stack([2*np.sum(prior[0]*delta[0],-1),2*np.sum(cp[0]*cd[0],-1),np.sum(closure[0]**2,-1),np.sum(cc[0]**2,-1)],-1)
    return np.concatenate([statistics(candidate,old,text),statistics(candidate,target,text),statistics(old,target,text),extras],-1)


def summarize(rows):
    result={name:panel(rows[:,8*i:8*i+8]) for i,name in enumerate(['vs82','vs73','old82vs73'])};totals=rows.sum(0)
    result['signedError']={'oldEnergy':float(totals[18]),'newCutDeltaEnergy':float(totals[2]),'totalEnergy':float(totals[10]),'twiceOldDeltaInner':float(totals[24]),
        'energyAccountRemainder':float(totals[10]-totals[18]-totals[2]-totals[24]),'centeredOldEnergy':float(totals[22]),'centeredDeltaEnergy':float(totals[6]),
        'centeredTotalEnergy':float(totals[14]),'twiceCenteredOldDeltaInner':float(totals[25]),'centeredEnergyAccountRemainder':float(totals[14]-totals[22]-totals[6]-totals[25]),
        'errorIdentityClosureEnergy':float(totals[26]),'centeredIdentityClosureEnergy':float(totals[27])}
    result['rowsNewTotalVsOldTotal']={'improve':int(np.sum(rows[:,10]<rows[:,18])),'worsen':int(np.sum(rows[:,10]>rows[:,18])),'equal':int(np.sum(rows[:,10]==rows[:,18]))}
    return result


def execute(root):
    import torch
    from flux_clock_study import verify
    from complete_masked_study import profile
    from row_output import work
    from row_output_ledger import Ledger
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);out=root/'response-result';s=read(root/'inputs.json');started=time.perf_counter()
    c=read(out/'candidate.json');seal=read(out/'candidate-seal.json');sealed=sha(out/'candidate-seal.json');exited=read(out/'producer-exit.json')
    if seal!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')} or exited['returncode']!=0 or exited['candidateSealSHA256']!=sealed or exited['requestSHA256']!=r['requestSHA256'] or exited['producerPID']==os.getpid():raise ValueError('Successful producer seal/exit absent')
    numeric=profile(torch,4)
    if numeric!=s['numericalProfile']:raise ValueError('Diagnostic profile differs')
    torch.cuda.reset_peak_memory_stats();ledger=Ledger(r['requestSHA256'],'post-exit-diagnostic');coords=load(s['sourceFields']['coordinates'],'cuda');u=load(c['fields']['rowBasis'],'cuda');rr=load(c['fields']['qrR'],'cuda');reconstructed=torch.empty_like(coords);grams=[]
    with torch.inference_mode():
        for stream,(lo,hi) in enumerate([(0,512),(512,1536)]):
            reconstructed[0,lo:hi]=ledger.matmul('coordinateReconstruction'+str(stream),u[0,lo:hi],rr[stream]);grams.append(ledger.matmul('basisGram'+str(stream),u[0,lo:hi].T,u[0,lo:hi]))
        gram=torch.stack(grams);residual=coords-reconstructed
    coordinate_stats=statistics(reconstructed.cpu().numpy(),coords.cpu().numpy());basis_diagnostics=[]
    for stream in range(2):
        error=gram[stream].cpu().numpy()-np.eye(128);diagonal=np.diag(rr[stream].cpu().numpy())
        basis_diagnostics.append({'stream':'text' if stream==0 else 'image','orthogonalityMaxAbs':float(np.max(np.abs(error))),'orthogonalityFrobenius':float(np.sqrt(np.sum(error*error))),
            'RDiagonal':diagonal.tolist(),'exactZeroRDiagonalCount':int(np.sum(diagonal==0)),'scope':'Finite QR diagnostics, not numerical rank or consumer sufficiency; all128 columns retained.'})
    fields={'coordinateResidual':retain(root/'arrays/analysis','coordinateResidual',residual),'basisGram':retain(root/'arrays/analysis','basisGram',gram)}
    del coords,u,rr,reconstructed,grams,gram,residual
    incoming=np.concatenate([load(s['sourceFields'][name]) for name in ['text','image']],1);rows=np.empty((1536,len(PORTS),len(ROW_COLUMNS)),dtype=np.float64)
    for index,name in enumerate(['preResidualUpdate','exit']):
        candidate=load(c['fields'][name]);old=load(s['comparisonFields']['82'][name]);target=load(s['comparisonFields']['73'][name]);rows[:,index]=accounts(candidate,old,target)
        if name=='exit':rows[:,-1]=accounts(candidate-incoming,old-incoming,target-incoming)
        del candidate,old,target
    fields.update(coordinateRowStatistics=retain(root/'arrays/analysis','coordinateRowStatistics',coordinate_stats),rowStatistics=retain(root/'arrays/analysis','rowStatistics',rows))
    if {name:{'shape':item['shape'],'dtype':item['logicalDtype']} for name,item in fields.items()}!=analysis_abis():raise ValueError('Diagnostic output ABI differs')
    panels={name:{port:summarize(rows[lo:hi,j]) for j,port in enumerate(PORTS)} for name,lo,hi in [('all',0,1536),('text',0,512),('image',512,1536)]}
    coordinate_panels={name:panel(coordinate_stats[lo:hi]) for name,lo,hi in [('all',0,1536),('text',0,512),('image',512,1536)]}
    with (out/'observations.jsonl').open('x') as stream:
        for index,row in enumerate(rows):stream.write(json.dumps({'row':index,'stratum':'text' if index<512 else 'image','ports':{port:dict(zip(ROW_COLUMNS,map(float,row[j]))) for j,port in enumerate(PORTS)},'coordinateReconstruction':dict(zip(COLUMNS,map(float,coordinate_stats[index])))},allow_nan=False)+'\n')
    costs=work()
    if len(ledger.products)!=4 or sum(x['nominalMACs'] for x in ledger.products)!=costs['diagnosticMACs'] or sha(out/'candidate-seal.json')!=sealed or sha(out/'candidate.json')!=seal['candidateSHA256'] or sha(out/'execution.json')!=seal['executionSHA256']:raise ValueError('Post-exit work/seal closure differs')
    save(out/'analysis-fields.json',{'fields':fields,'candidateSealSHA256':sealed,'ports':PORTS,'rowColumns':ROW_COLUMNS})
    save(out/'analysis-execution.json',{'products':ledger.products,'diagnosticMatrixMACs':costs['diagnosticMACs'],'numericalProfile':numeric,'analyzerPID':os.getpid(),'producerExitSHA256':sha(out/'producer-exit.json'),'candidateSealSHA256':sealed,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved()})
    save(out/'report.json',{'schema':'tce-current-row-output-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':sealed,'fixedAddress':{'clock':0,'singleBlock':0},'panels':panels,'coordinateReconstruction':coordinate_panels,'basisDiagnostics':basis_diagnostics,'work':costs,'observations':1536,'analysisSecondsAfterImports':time.perf_counter()-started,
        'newImages':0,'newContinuations':0,'newFits':0,'mechanics_status':'complete','instrument_status':'same-input-row-space-complete-write-test','trend_status':'Primary82/secondary73 and signed full/centered everyrow retained','terminal_status':'unassessed',
        'scope':'Retained82C/product and source73 moment-attention/read/gate/residual plus original output map paid. New12.684Boutput path, prospective composed32.011B not actualnewmeasurement. No native reference/continuation/image, runtime/generalization/zero-neural certification.'})
    for name in ['request.json','inputs.json','resource-basis.json','frozen-plan.md','implementation-notes.md']:shutil.copyfile(root/name,out/name)


if __name__=='__main__':execute(sys.argv[1])
