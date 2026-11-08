"""Five-corner post-exit accounting; CPU FP64 scalar/data work, no matrix products."""
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

PORTS=['matchedExit','ownExit','matchedIncrement','ownIncrement','responseVsCarry']
VECTORS=['D','E0','T','U','W','Eout']
CROSSES=[('E0','T'),('E0','U'),('D','W')]
CLOSURES=['outgoingFromE0T','incrementFromE0U','outgoingFromDW']
EXTRAS=[v+'_'+suffix for v in VECTORS for suffix in ['energy','centeredEnergy']]
EXTRAS += ['twice'+a+b+suffix for a,b in CROSSES for suffix in ['Inner','CenteredInner']]
EXTRAS += [v+'_'+suffix for v in CLOSURES for suffix in ['closureEnergy','centeredClosureEnergy']]
ROW_COLUMNS=[p+'_'+c for p in PORTS for c in COLUMNS]+EXTRAS


def analysis_abis(text=512,image=1024,dim=3072):
    return {n:{'shape':[1,text+image,dim],'dtype':'float64'} for n in ['E0','T','U']}|{
        'rowStatistics':{'shape':[text+image,len(ROW_COLUMNS)],'dtype':'float64'}}


def center(value,text):
    return np.concatenate([value[:,:text]-value[:,:text].mean(1,keepdims=True),
                           value[:,text:]-value[:,text:].mean(1,keepdims=True)],1)


def accounts(x,xhat,y,yhat,y0,text=512):
    x,xhat,y,yhat,y0=[np.asarray(v,dtype=np.float64) for v in [x,xhat,y,yhat,y0]]
    d=xhat-x;e0=y0-y;t=yhat-y0;u=t-d
    w=(yhat-xhat)-(y-x);out=yhat-y
    vectors=dict(zip(VECTORS,[d,e0,t,u,w,out]))
    centered={n:center(v,text) for n,v in vectors.items()}
    rows=[statistics(a,b,text) for a,b in [(y0,y),(yhat,y),(y0-x,y-x),(yhat-xhat,y-x),(t,d)]]
    extra=[]
    for name in VECTORS:
        for v in [vectors[name],centered[name]]:extra.append(np.sum(v[0]*v[0],-1))
    for a,b in CROSSES:
        for values in [vectors,centered]:extra.append(2*np.sum(values[a][0]*values[b][0],-1))
    for names in [('Eout','E0','T'),('W','E0','U'),('Eout','D','W')]:
        for values in [vectors,centered]:
            residual=values[names[0]]-values[names[1]]-values[names[2]]
            extra.append(np.sum(residual[0]**2,-1))
    return vectors,np.concatenate([*rows,np.stack(extra,-1)],-1)


def summarize(rows):
    value={p:panel(rows[:,i*8:i*8+8]) for i,p in enumerate(PORTS)}
    s=rows.sum(0);totals=dict(zip(ROW_COLUMNS,map(float,s)))
    energy={k:totals[k] for k in EXTRAS}
    energy.update(
        outgoingE0TAccountRemainder=totals['Eout_energy']-totals['E0_energy']-totals['T_energy']-totals['twiceE0TInner'],
        incrementE0UAccountRemainder=totals['W_energy']-totals['E0_energy']-totals['U_energy']-totals['twiceE0UInner'],
        outgoingDWAccountRemainder=totals['Eout_energy']-totals['D_energy']-totals['W_energy']-totals['twiceDWInner'],
        centeredOutgoingE0TAccountRemainder=totals['Eout_centeredEnergy']-totals['E0_centeredEnergy']-totals['T_centeredEnergy']-totals['twiceE0TCenteredInner'],
        centeredIncrementE0UAccountRemainder=totals['W_centeredEnergy']-totals['E0_centeredEnergy']-totals['U_centeredEnergy']-totals['twiceE0UCenteredInner'])
    ratio=lambda a,b:float(np.sqrt(a/b)) if b>0 else None
    value['signedAccount']=energy
    value['relativeNorms']={a+'Over'+b:ratio(totals[a+'_energy'],totals[b+'_energy']) for a,b in [('E0','T'),('E0','Eout'),('T','D'),('U','W')]}
    value['centeredRelativeNorms']={a+'Over'+b:ratio(totals[a+'_centeredEnergy'],totals[b+'_centeredEnergy']) for a,b in [('E0','T'),('E0','Eout'),('T','D'),('U','W')]}
    col=lambda k:rows[:,ROW_COLUMNS.index(k)]
    value['matchedVsOwn']={port:{'improve':int(np.sum(col('matched'+port+'_errorEnergy')<col('own'+port+'_errorEnergy'))),
                                'worsen':int(np.sum(col('matched'+port+'_errorEnergy')>col('own'+port+'_errorEnergy'))),
                                'equal':int(np.sum(col('matched'+port+'_errorEnergy')==col('own'+port+'_errorEnergy')))}
                           for port in ['Exit','Increment']}
    value['zeroDenominatorRows']={p:int(np.sum(col(p+'_targetEnergy')==0)) for p in PORTS}
    value['crossTermSigns']={a+b:{'negative':int(np.sum(col('twice'+a+b+'Inner')<0)),
                                  'positive':int(np.sum(col('twice'+a+b+'Inner')>0)),
                                  'zero':int(np.sum(col('twice'+a+b+'Inner')==0))} for a,b in CROSSES}
    return value


def execute(root):
    from matched_parent import work
    from flux_clock_study import verify
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json')
    out=root/'response-result';tick=time.perf_counter();c=read(out/'candidate.json')
    seal=sha(out/'candidate-seal.json');exited=read(out/'producer-exit.json')
    if read(out/'candidate-seal.json')!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')} or exited['returncode']!=0 or exited['requestSHA256']!=r['requestSHA256'] or exited['candidateSealSHA256']!=seal or exited['producerPID']==os.getpid():
        raise ValueError('Observed producer seal/exit differs')
    def streams(fields):return np.concatenate([load(fields[n]) for n in ['text','image']],1)
    x=streams(s['sourceFields']);xh=streams(s['comparisonFields']['ownIncoming'])
    y=streams(s['comparisonFields']['parentOutgoing']);yh=streams(s['comparisonFields']['ownOutgoing'])
    y0=load(c['fields']['exit'])
    vectors,rows=accounts(x,xh,y,yh,y0)
    fields={n:retain(root/'arrays/analysis',n,vectors[n]) for n in ['E0','T','U']}
    fields['rowStatistics']=retain(root/'arrays/analysis','rowStatistics',rows)
    if {n:{'shape':v['shape'],'dtype':v['logicalDtype']} for n,v in fields.items()}!=analysis_abis() or not np.isfinite(rows).all():
        raise ValueError('Finite five-corner ABI differs')
    panels={n:summarize(rows[lo:hi]) for n,lo,hi in [('all',0,1536),('text',0,512),('image',512,1536)]}
    with (out/'observations.jsonl').open('x') as f:
        for i,row in enumerate(rows):f.write(json.dumps({'row':i,'stratum':'text' if i<512 else 'image','statistics':dict(zip(ROW_COLUMNS,map(float,row)))},allow_nan=False)+'\n')
    if sha(out/'candidate-seal.json')!=seal:raise ValueError('Candidate changed after release')
    save(out/'analysis-fields.json',{'fields':fields,'candidateSealSHA256':seal,'rowColumns':ROW_COLUMNS})
    save(out/'analysis-execution.json',{'products':[],'diagnosticMatrixMACs':0,'device':'cpu','dtype':'float64','numpyVersion':np.__version__,
        'analyzerPID':os.getpid(),'candidateSealSHA256':seal,'producerExitSHA256':sha(out/'producer-exit.json'),
        'scope':'Post-exit NumPy FP64 differences,centering,elementwise products and reductions only. Scalar/data work paid;no learned map or matrix product.'})
    save(out/'report.json',{'schema':'tce-matched-parent-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal,
        'fixedAddress':{'clock':0,'singleBlock':1},'panels':panels,'work':work(),'observations':1536,
        'analysisSecondsAfterImports':time.perf_counter()-tick,'newImages':0,'newFits':0,'newContinuations':1,
        'mechanics_status':'complete','instrument_status':'same-reduced-program-two-inputs-five-corners',
        'trend_status':'Full/centered signed E0/T/U/W/D/Eout accounts and every dependent row retained.','terminal_status':'unassessed',
        'scope':'One frozen87 reduced program on parent70/73input vs existing own86input/87output. T is candidate response,U removes identity carry;not native propagation or compression-vs-slicing attribution. All learned maps/clocks/historical work paid;no image/compact closure/arbitrarypromptzero-neural result.'})
    for name in ['request.json','inputs.json','resource-basis.json','frozen-plan.md','implementation-notes.md','intake.json']:
        shutil.copyfile(root/name,out/name)


if __name__=='__main__':execute(sys.argv[1])
