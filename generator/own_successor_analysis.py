"""Post-exit own-state D+W accounting and optional factorization observations."""
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from encoder_atoms import read,save,sha
from gate_up_basis_analysis import statistics,panel,COLUMNS
from paired_fields import load,retain

PORTS=['incoming','increment','outgoing','holdOutgoing']
EXTRAS=['twiceDWInner','twiceCenteredDWInner','identityClosureEnergy','centeredIdentityClosureEnergy']
ROW_COLUMNS=[p+'_'+c for p in PORTS for c in COLUMNS]+EXTRAS


def analysis_abis(text=512,image=1024,dim=3072,rank=128):
    n=text+image
    return {'inputResidual':{'shape':[1,n,dim],'dtype':'torch.float64'},
            'inputGram':{'shape':[2,rank,rank],'dtype':'torch.float64'},
            'inputRowStatistics':{'shape':[n,8],'dtype':'float64'},
            'sketchResidual':{'shape':[1,n,rank],'dtype':'torch.float64'},
            'basisGram':{'shape':[2,rank,rank],'dtype':'torch.float64'},
            'sketchRowStatistics':{'shape':[n,8],'dtype':'float64'},
            'rowStatistics':{'shape':[n,len(ROW_COLUMNS)],'dtype':'float64'}}


def accounts(candidate_in,candidate_out,parent_in,parent_out,text=512):
    ci,co,pi,po=candidate_in,candidate_out,parent_in,parent_out
    values=[statistics(a,b,text) for a,b in [(ci,pi),(co-ci,po-pi),(co,po),(ci,po)]]
    d=ci-pi;w=(co-ci)-(po-pi);out=co-po;closure=out-d-w
    def center(v):
        return np.concatenate([v[:,:text]-v[:,:text].mean(1,keepdims=True),v[:,text:]-v[:,text:].mean(1,keepdims=True)],1)
    cd,cw,ct=map(center,[d,w,out]);cc=ct-cd-cw
    values.append(np.stack([2*np.sum(d[0]*w[0],-1),2*np.sum(cd[0]*cw[0],-1),
                            np.sum(closure[0]**2,-1),np.sum(cc[0]**2,-1)],-1))
    return np.concatenate(values,-1)


def summarize(rows):
    result={p:panel(rows[:,8*i:8*i+8]) for i,p in enumerate(PORTS)};s=rows.sum(0)
    result['signedAccount']={'incomingDifferenceEnergy':float(s[2]),'incrementDifferenceEnergy':float(s[10]),
        'outgoingDifferenceEnergy':float(s[18]),'twiceDWInner':float(s[32]),
        'energyAccountRemainder':float(s[18]-s[2]-s[10]-s[32]),
        'centeredIncomingDifferenceEnergy':float(s[6]),'centeredIncrementDifferenceEnergy':float(s[14]),
        'centeredOutgoingDifferenceEnergy':float(s[22]),'twiceCenteredDWInner':float(s[33]),
        'centeredEnergyAccountRemainder':float(s[22]-s[6]-s[14]-s[33]),
        'identityClosureEnergy':float(s[34]),'centeredIdentityClosureEnergy':float(s[35])}
    a,b=rows[:,18],rows[:,26]
    result['outgoingRowsVsOwnHold']={'improve':int(np.sum(a<b)),'worsen':int(np.sum(a>b)),'equal':int(np.sum(a==b))}
    return result


def execute(root):
    import torch
    from own_successor import work
    from own_successor_ledger import Ledger
    from complete_masked_study import profile
    from flux_clock_study import verify
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json');out=root/'response-result';tick=time.perf_counter()
    c=read(out/'candidate.json');seal=sha(out/'candidate-seal.json');exited=read(out/'producer-exit.json')
    if read(out/'candidate-seal.json')!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')} or exited['returncode']!=0 or exited['candidateSealSHA256']!=seal or exited['producerPID']==os.getpid():raise ValueError('Observed producer release/exit differs')
    numeric=profile(torch,4)
    if numeric!=s['numericalProfile']:raise ValueError('Analysis numerical profile differs')
    torch.cuda.reset_peak_memory_stats();ledger=Ledger(r['requestSHA256'],'post-exit-diagnostic')
    q,coords,z=[load(c['fields'][n],'cuda') for n in ['gu_basis','gu_coordinates','normalized']]
    original=z.double();reconstructed=torch.empty_like(original);grams=[]
    with torch.inference_mode():
        for stream,(lo,hi) in enumerate([(0,512),(512,1536)]):
            reconstructed[0,lo:hi]=ledger.matmul('guInputReconstruction'+str(stream),coords[0,lo:hi],q[stream].T)
            grams.append(ledger.matmul('guBasisGram'+str(stream),q[stream].T,q[stream]))
    istats=statistics(reconstructed.cpu().numpy(),original.cpu().numpy())
    fields={'inputResidual':retain(root/'arrays/analysis','inputResidual',original-reconstructed),
            'inputGram':retain(root/'arrays/analysis','inputGram',torch.stack(grams)),
            'inputRowStatistics':retain(root/'arrays/analysis','inputRowStatistics',istats)}
    del q,coords,z,original,reconstructed,grams
    f,u,rr=[load(c['fields'][n],'cuda') for n in ['sketch','rowBasis','qrR']]
    reconstructed=torch.empty_like(f);grams=[]
    with torch.inference_mode():
        for stream,(lo,hi) in enumerate([(0,512),(512,1536)]):
            reconstructed[0,lo:hi]=ledger.matmul('outputSketchReconstruction'+str(stream),u[0,lo:hi],rr[stream])
            grams.append(ledger.matmul('outputBasisGram'+str(stream),u[0,lo:hi].T,u[0,lo:hi]))
    sstats=statistics(reconstructed.cpu().numpy(),f.cpu().numpy())
    fields.update(sketchResidual=retain(root/'arrays/analysis','sketchResidual',f-reconstructed),
                  basisGram=retain(root/'arrays/analysis','basisGram',torch.stack(grams)),
                  sketchRowStatistics=retain(root/'arrays/analysis','sketchRowStatistics',sstats))
    del f,u,rr,reconstructed,grams
    ci=np.concatenate([load(s['sourceFields'][n]) for n in ['text','image']],1)
    co=load(c['fields']['exit'])
    pi,po=[np.concatenate([load(s['comparisonFields'][port][n]) for n in ['text','image']],1) for port in ['incoming','outgoing']]
    rows=accounts(ci,co,pi,po);fields['rowStatistics']=retain(root/'arrays/analysis','rowStatistics',rows)
    panels={n:summarize(rows[lo:hi]) for n,lo,hi in [('all',0,1536),('text',0,512),('image',512,1536)]}
    diagnostics={n:{'inputReconstruction':panel(istats[lo:hi]),'outputSketchReconstruction':panel(sstats[lo:hi])} for n,lo,hi in [('all',0,1536),('text',0,512),('image',512,1536)]}
    with (out/'observations.jsonl').open('x') as stream:
        for i,row in enumerate(rows):stream.write(json.dumps({'row':i,'stratum':'text' if i<512 else 'image','statistics':dict(zip(ROW_COLUMNS,map(float,row))),
            'inputReconstruction':dict(zip(COLUMNS,map(float,istats[i]))),'outputSketchReconstruction':dict(zip(COLUMNS,map(float,sstats[i])))},allow_nan=False)+'\n')
    if {n:{'shape':v['shape'],'dtype':v['logicalDtype']} for n,v in fields.items()}!=analysis_abis() or len(ledger.products)!=8 or sum(x['nominalMACs'] for x in ledger.products)!=work()['diagnosticMACs'] or sha(out/'candidate-seal.json')!=seal:raise ValueError('Diagnostic ABI/work/release closure differs')
    save(out/'analysis-fields.json',{'fields':fields,'candidateSealSHA256':seal,'rowColumns':ROW_COLUMNS})
    save(out/'analysis-execution.json',{'products':ledger.products,'numericalProfile':numeric,'candidateSealSHA256':seal,'producerExitSHA256':sha(out/'producer-exit.json'),
        'analyzerPID':os.getpid(),'peakCUDAAllocatedBytes':torch.cuda.max_memory_allocated(),'peakCUDAReservedBytes':torch.cuda.max_memory_reserved()})
    save(out/'report.json',{'schema':'tce-own-successor-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal,'fixedAddress':{'clock':0,'singleBlock':1},
        'panels':panels,'factorizationDiagnostics':diagnostics,'work':work(),'observations':1536,'analysisSecondsAfterImports':time.perf_counter()-tick,
        'newImages':0,'newContinuations':1,'mechanics_status':'complete','instrument_status':'own-state-one-block-successor',
        'trend_status':'Full/centered D+W accounts, all dependent rows and own-hold comparisons retained.','terminal_status':'unassessed',
        'scope':'One physically sliced/reduced learned successor from actual86state against paid70 dense-affine moment-attention successor. Supplied clocks/maps paid; no independent native full-attention, error-source attribution, compact closure or arbitrarypromptzero-neural result.'})
    for name in ['request.json','inputs.json','resource-basis.json','frozen-plan.md','implementation-notes.md','intake.json']:
        shutil.copyfile(root/name,out/name)


if __name__=='__main__':execute(sys.argv[1])
