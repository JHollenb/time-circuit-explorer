"""Post-exit six-corner native-comparator response and approximation-error accounts."""
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
from matched_parent_analysis import center

PORTS=['denseResponseVsCarry','approxResponseVsDense','ownApproxExit','parentApproxExit','denseIncrement','ownIncrement']
VECTORS=['D','E0','T','U','W','Eout','TN','E1','Q','V']
CROSSES=[('TN','E1'),('V','E1'),('V','Q'),('E0','Q')]
IDENTITIES=[('Eout','TN','E1'),('W','V','E1'),('U','V','Q'),('E1','E0','Q')]
EXTRAS=[v+'_'+suffix for v in VECTORS for suffix in ['energy','centeredEnergy']]
EXTRAS += ['twice'+a+b+suffix for a,b in CROSSES for suffix in ['Inner','CenteredInner']]
EXTRAS += [a+'From'+b+c+'_'+suffix for a,b,c in IDENTITIES for suffix in ['closureEnergy','centeredClosureEnergy']]
EXTRAS += ['dualQClosureEnergy','dualQCenteredClosureEnergy']
ROW_COLUMNS=[p+'_'+c for p in PORTS for c in COLUMNS]+EXTRAS


def analysis_abis(text=512,image=1024,dim=3072):
    return {n:{'shape':[1,text+image,dim],'dtype':'float64'} for n in ['TN','E1','Q','V']}|{
        'rowStatistics':{'shape':[text+image,len(ROW_COLUMNS)],'dtype':'float64'}}


def accounts(x,xh,y,yh,y0,y1,text=512):
    x,xh,y,yh,y0,y1=[np.asarray(a,dtype=np.float64) for a in [x,xh,y,yh,y0,y1]]
    d=xh-x;e0=y0-y;t=yh-y0;u=t-d;w=(yh-xh)-(y-x);out=yh-y
    tn=y1-y;e1=yh-y1;q=t-tn;v=tn-d;other_q=e1-e0
    values=dict(zip(VECTORS,[d,e0,t,u,w,out,tn,e1,q,v]));cs={n:center(a,text) for n,a in values.items()}
    rows=[statistics(a,b,text) for a,b in [(tn,d),(t,tn),(yh,y1),(y0,y),(y1-xh,y-x),(yh-xh,y-x)]]
    extra=[]
    for name in VECTORS:
        for a in [values[name],cs[name]]:extra.append(np.sum(a[0]*a[0],-1))
    for a,b in CROSSES:
        for group in [values,cs]:extra.append(2*np.sum(group[a][0]*group[b][0],-1))
    for a,b,c in IDENTITIES:
        for group in [values,cs]:
            residual=group[a]-group[b]-group[c];extra.append(np.sum(residual[0]**2,-1))
    extra += [np.sum((q-other_q)[0]**2,-1),np.sum((center(q,text)-center(other_q,text))[0]**2,-1)]
    return values,np.concatenate([*rows,np.stack(extra,-1)],-1)


def summarize(rows):
    value={p:panel(rows[:,i*8:i*8+8]) for i,p in enumerate(PORTS)}
    sums=dict(zip(ROW_COLUMNS,map(float,rows.sum(0))));account={n:sums[n] for n in EXTRAS}
    for a,b,c in IDENTITIES:
        for suffix,crosssuffix in [('energy','Inner'),('centeredEnergy','CenteredInner')]:
            account[a+'From'+b+c+'_'+suffix+'AccountRemainder']=sums[a+'_'+suffix]-sums[b+'_'+suffix]-sums[c+'_'+suffix]-sums['twice'+b+c+crosssuffix]
    ratio=lambda a,b:float(np.sqrt(a/b)) if b>0 else None
    pairs=[('TN','D'),('T','D'),('Q','TN'),('Q','T'),('Q','E0'),('E1','E0'),('V','W')]
    value['signedAccount']=account
    value['relativeNorms']={a+'Over'+b:ratio(sums[a+'_energy'],sums[b+'_energy']) for a,b in pairs}
    value['centeredRelativeNorms']={a+'Over'+b:ratio(sums[a+'_centeredEnergy'],sums[b+'_centeredEnergy']) for a,b in pairs}
    col=lambda k:rows[:,ROW_COLUMNS.index(k)]
    value['rowSigns']={'TNMoreThanD':int(np.sum(col('TN_energy')>col('D_energy'))),
                       'QMoreThanTN':int(np.sum(col('Q_energy')>col('TN_energy'))),
                       'E1MoreThanE0':int(np.sum(col('E1_energy')>col('E0_energy')))}
    value['zeroDenominatorRows']={p:int(np.sum(col(p+'_targetEnergy')==0)) for p in PORTS}
    return value


def execute(root):
    from dense_corner import work
    from flux_clock_study import verify
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json')
    out=root/'response-result';tick=time.perf_counter();c=read(out/'candidate.json')
    seal=sha(out/'candidate-seal.json');exited=read(out/'producer-exit.json')
    if read(out/'candidate-seal.json')!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')} or exited['returncode']!=0 or exited['requestSHA256']!=r['requestSHA256'] or exited['candidateSealSHA256']!=seal or exited['producerPID']==os.getpid():
        raise ValueError('Observed producer seal/exit differs')
    def streams(fields):return np.concatenate([load(fields[n]) for n in ['text','image']],1)
    xh=streams(s['sourceFields']);x=streams(s['comparisonFields']['parentIncoming'])
    y=streams(s['comparisonFields']['parentOutgoing']);yh=streams(s['comparisonFields']['ownOutgoing'])
    y0=streams(s['comparisonFields']['matchedOutgoing']);y1=load(c['fields']['exit'])
    values,rows=accounts(x,xh,y,yh,y0,y1)
    fields={n:retain(root/'arrays/analysis',n,values[n]) for n in ['TN','E1','Q','V']}
    fields['rowStatistics']=retain(root/'arrays/analysis','rowStatistics',rows)
    if {n:{'shape':v['shape'],'dtype':v['logicalDtype']} for n,v in fields.items()}!=analysis_abis() or not np.isfinite(rows).all():
        raise ValueError('Finite six-corner ABI differs')
    panels={n:summarize(rows[lo:hi]) for n,lo,hi in [('all',0,1536),('text',0,512),('image',512,1536)]}
    with (out/'observations.jsonl').open('x') as f:
        for i,row in enumerate(rows):f.write(json.dumps({'row':i,'stratum':'text' if i<512 else 'image','statistics':dict(zip(ROW_COLUMNS,map(float,row)))},allow_nan=False)+'\n')
    if sha(out/'candidate-seal.json')!=seal:raise ValueError('Candidate changed after release')
    save(out/'analysis-fields.json',{'fields':fields,'candidateSealSHA256':seal,'rowColumns':ROW_COLUMNS})
    save(out/'analysis-execution.json',{'products':[],'diagnosticMatrixMACs':0,'device':'cpu','dtype':'float64','numpyVersion':np.__version__,
        'analyzerPID':os.getpid(),'candidateSealSHA256':seal,'producerExitSHA256':sha(out/'producer-exit.json'),
        'scope':'Post-exit six-corner FP64 differences,centering,elementwise products/reductions only;scalar/data work paid,no learned map or matrix product.'})
    save(out/'report.json',{'schema':'tce-dense-corner-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal,
        'fixedAddress':{'clock':0,'singleBlock':1},'panels':panels,'work':work(),'observations':1536,
        'analysisSecondsAfterImports':time.perf_counter()-tick,'newImages':0,'newFits':0,'newContinuations':1,
        'mechanics_status':'complete','instrument_status':'six-corner-dense-moment-comparator-decision-audit',
        'trend_status':'Full/centered TN/E1/Q/V plus priorvectors,signedcrosses,independentdualQ and allrows retained.','terminal_status':'unassessed',
        'scope':'Original fullgeometry70dense-affine/momentattention function onactual86state. TN is declaredcomparatorresponse,not independent originalfullattention. Q is changingapproximationerror overonepair;not global/nativeattention generalization. All learned maps/state/clocks/history paid;no image/compactclosure/arbitrarypromptzero-neural result.'})
    for name in ['request.json','inputs.json','resource-basis.json','frozen-plan.md','implementation-notes.md','intake.json']:
        shutil.copyfile(root/name,out/name)


if __name__=='__main__':execute(sys.argv[1])
