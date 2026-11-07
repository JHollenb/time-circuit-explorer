"""Post-exit source reconstruction, native local targets and every dependent row."""
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from encoder_atoms import read,save,sha
from paired_fields import load,retain

PORTS=['gateRaw','upRaw','activation','product','preResidualUpdate','exit','exitIncrement']
COLUMNS=['targetEnergy','candidateEnergy','errorEnergy','inner','centeredTargetEnergy','centeredCandidateEnergy','centeredErrorEnergy','centeredInner']


def analysis_abis(dim=3072,text=512,image=1024,rank=128):
    return {n:{'shape':shape,'dtype':'torch.float64'} for n,shape in [('inputResidual',[1,text+image,dim]),('basisGram',[2,rank,rank])]}|{
        'rowStatistics':{'shape':[text+image,len(PORTS),len(COLUMNS)],'dtype':'float64'},
        'inputRowStatistics':{'shape':[text+image,len(COLUMNS)],'dtype':'float64'}}


def statistics(candidate,target,text=512):
    a=np.asarray(candidate,dtype=np.float64)[0];b=np.asarray(target,dtype=np.float64)[0];error=a-b
    def centered(x):return np.concatenate([x[:text]-x[:text].mean(0,keepdims=True),x[text:]-x[text:].mean(0,keepdims=True)])
    ac,bc=centered(a),centered(b);ec=ac-bc
    energy=lambda x:np.sum(x*x,axis=-1)
    return np.stack([energy(b),energy(a),energy(error),np.sum(a*b,axis=-1),energy(bc),energy(ac),energy(ec),np.sum(ac*bc,axis=-1)],-1)


def panel(rows):
    totals=rows.sum(0);e=dict(zip(COLUMNS,map(float,totals)))
    ratio=lambda a,b:float(np.sqrt(a/b)) if b>0 else None
    return {'rows':len(rows),'energies':e,'errorOverTargetRMS':ratio(e['errorEnergy'],e['targetEnergy']),
        'centeredErrorOverCenteredTargetRMS':ratio(e['centeredErrorEnergy'],e['centeredTargetEnergy']),
        'cosine':e['inner']/float(np.sqrt(e['targetEnergy']*e['candidateEnergy'])) if e['targetEnergy']>0 and e['candidateEnergy']>0 else None,
        'rowsVsZero':{'improve':int(np.sum(rows[:,2]<rows[:,0])),'worsen':int(np.sum(rows[:,2]>rows[:,0])),'equal':int(np.sum(rows[:,2]==rows[:,0]))}}


def execute(root):
    import torch
    from gate_up_basis import COMPARISON_NAMES,work
    from gate_up_basis_ledger import Ledger
    from flux_clock_study import verify
    from complete_masked_study import profile
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);out=root/'response-result';started=time.perf_counter();s=read(root/'inputs.json')
    c=read(out/'candidate.json');seal=read(out/'candidate-seal.json');sealed=sha(out/'candidate-seal.json');exited=read(out/'producer-exit.json')
    if seal!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')} or exited['returncode']!=0 or exited['candidateSealSHA256']!=sealed or exited['requestSHA256']!=r['requestSHA256'] or exited['producerPID']==os.getpid():raise ValueError('Producer seal/exit absent')
    numeric=profile(torch,4)
    if numeric!=s['numericalProfile']:raise ValueError('Diagnostic numerical policy differs')
    ledger=Ledger(r['requestSHA256'],'post-exit-diagnostic');torch.cuda.reset_peak_memory_stats()
    z=load(s['sourceFields']['normalized'],'cuda').double();q=load(c['fields']['basis'],'cuda');coords=load(c['fields']['coordinates'],'cuda')
    reconstructed=torch.empty_like(z);grams=[]
    with torch.inference_mode():
        for stream,(lo,hi) in enumerate([(0,512),(512,1536)]):
            reconstructed[0,lo:hi]=ledger.matmul('inputReconstruction'+str(stream),coords[0,lo:hi],q[stream].T)
            grams.append(ledger.matmul('basisGram'+str(stream),q[stream].T,q[stream]))
        gram=torch.stack(grams)
    residual=z-reconstructed;input_stats=statistics(reconstructed.cpu().numpy(),z.cpu().numpy());basis_diag=[]
    rr=load(c['fields']['qrR'])
    for stream in range(2):
        diagonal=np.diag(rr[stream]);delta=gram[stream].cpu().numpy()-np.eye(128)
        basis_diag.append({'stream':'text' if stream==0 else 'image','orthogonalityMaxAbs':float(np.max(np.abs(delta))),
            'orthogonalityFrobenius':float(np.sqrt(np.sum(delta*delta))),'RDiagonal':diagonal.tolist(),'exactZeroRDiagonalCount':int(np.sum(diagonal==0)),
            'scope':'Unpivoted diagonal and orthogonality diagnostics; no numerical-rank certification, truncation or adaptive threshold.'})
    fields={'inputResidual':retain(root/'arrays/analysis','inputResidual',residual),'basisGram':retain(root/'arrays/analysis','basisGram',gram)}
    del z,q,coords,residual,gram,grams,reconstructed
    incoming=np.concatenate([load(s['sourceFields'][n]) for n in ['text','image']],1);rows=np.empty((1536,len(PORTS),len(COLUMNS)),dtype=np.float64)
    for index,name in enumerate(COMPARISON_NAMES):
        a=load(c['fields'][name]);b=load(s['comparisonFields'][name]);rows[:,index]=statistics(a,b)
        if name=='exit':rows[:,-1]=statistics(a-incoming,b-incoming)
        del a,b
    fields.update(rowStatistics=retain(root/'arrays/analysis','rowStatistics',rows),inputRowStatistics=retain(root/'arrays/analysis','inputRowStatistics',input_stats))
    if {n:{'shape':v['shape'],'dtype':v['logicalDtype']} for n,v in fields.items()}!=analysis_abis():raise ValueError('Analysis ABI differs')
    panels={};input_panels={}
    for name,lo,hi in [('all',0,1536),('text',0,512),('image',512,1536)]:
        panels[name]={port:panel(rows[lo:hi,j]) for j,port in enumerate(PORTS)};input_panels[name]=panel(input_stats[lo:hi])
    with (out/'observations.jsonl').open('x') as stream:
        for index in range(1536):
            stream.write(json.dumps({'row':index,'stratum':'text' if index<512 else 'image','ports':{port:dict(zip(COLUMNS,map(float,rows[index,j]))) for j,port in enumerate(PORTS)},
                'inputReconstruction':dict(zip(COLUMNS,map(float,input_stats[index])))},allow_nan=False)+'\n')
    if sha(out/'candidate-seal.json')!=sealed or sha(out/'candidate.json')!=seal['candidateSHA256'] or sha(out/'execution.json')!=seal['executionSHA256']:raise ValueError('Producer changed after release')
    costs=work();macs=sum(x['nominalMACs'] for x in ledger.products)
    if macs!=costs['diagnosticMACs'] or len(ledger.products)!=4:raise ValueError('Diagnostic work instrument differs')
    save(out/'analysis-fields.json',{'fields':fields,'candidateSealSHA256':sealed,'ports':PORTS,'rowColumns':COLUMNS})
    save(out/'analysis-execution.json',{'products':ledger.products,'diagnosticMatrixMACs':macs,'numericalProfile':numeric,'analyzerPID':os.getpid(),
        'producerExitSHA256':sha(out/'producer-exit.json'),'candidateSealSHA256':sealed,'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved()})
    save(out/'report.json',{'schema':'tce-current-state-gate-up-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':sealed,'fixedAddress':{'clock':0,'singleBlock':0},
        'panels':panels,'inputReconstruction':input_panels,'basisDiagnostics':basis_diag,'work':costs,'observations':1536,'analysisSecondsAfterImports':time.perf_counter()-started,
        'newImages':0,'newContinuations':0,'newFits':0,'mechanics_status':'complete','instrument_status':'one-paid-current-state-basis-local-consumer','trend_status':'Full and stream-centered operands/write/state/increment and every row retained','terminal_status':'unassessed',
        'scope':'Moment-attention source73 input and original learned GU/output maps remain paid. Complete current-input factorization, no archived output before producer exit. Local arithmetic comparison, no matched runtime, trajectory, prompt generalization or zero-neural image claim.'})
    for name in ['request.json','inputs.json','resource-basis.json','frozen-plan.md','implementation-notes.md']:shutil.copyfile(root/name,out/name)


if __name__=='__main__':execute(sys.argv[1])
