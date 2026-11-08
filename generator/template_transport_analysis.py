"""Eight-corner post-exit accounts; parent finite-template difference remains evidence."""
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

PORTS=['fixedParentIncrement','fixedOwnIncrement','dynamicParentIncrement','dynamicOwnIncrement','fixedResponse','dynamicResponse']
VECTORS=['D','TN','E0','E1','T','Q','Ef0','Ef1','A0','A1','B','Tf','Qf']
IDENTITIES=[('T','Tf','B'),('Q','Qf','B'),('E0','Ef0','A0'),('E1','Ef1','A1')]
EXTRAS=[v+'_'+suffix for v in VECTORS for suffix in ['energy','centeredEnergy']]
EXTRAS += ['twice'+b+c+suffix for _,b,c in IDENTITIES for suffix in ['Inner','CenteredInner']]
EXTRAS += [a+'From'+b+c+'_'+suffix for a,b,c in IDENTITIES for suffix in ['closureEnergy','centeredClosureEnergy']]
EXTRAS += ['dualQfClosureEnergy','dualQfCenteredClosureEnergy']
ROW_COLUMNS=[p+'_'+c for p in PORTS for c in COLUMNS]+EXTRAS
RETAIN=['Ef0','Ef1','A0','A1','B','Tf','Qf']


def analysis_abis(text=512,image=1024,dim=3072):
    return {n:{'shape':[1,text+image,dim],'dtype':'float64'} for n in RETAIN}|{
        'rowStatistics':{'shape':[text+image,len(ROW_COLUMNS)],'dtype':'float64'}}


def accounts(x,xh,y,yh,y0,y1,z0,z1,text=512):
    x,xh,y,yh,y0,y1,z0,z1=[np.asarray(a,dtype=np.float64) for a in [x,xh,y,yh,y0,y1,z0,z1]]
    d=xh-x;tn=y1-y;e0=y0-y;e1=yh-y1;t=yh-y0;q=t-tn
    ef0=z0-y;ef1=z1-y1;a0=y0-z0;a1=yh-z1;b=a1-a0;tf=z1-z0;qf=tf-tn
    values=dict(zip(VECTORS,[d,tn,e0,e1,t,q,ef0,ef1,a0,a1,b,tf,qf]));cs={n:center(v,text) for n,v in values.items()}
    rows=[statistics(a,b,text) for a,b in [(z0-x,y-x),(z1-xh,y1-xh),(y0-x,y-x),(yh-xh,y1-xh),(tf,tn),(t,tn)]]
    extra=[]
    for name in VECTORS:
        for v in [values[name],cs[name]]:extra.append(np.sum(v[0]**2,-1))
    for _,a,b in IDENTITIES:
        for group in [values,cs]:extra.append(2*np.sum(group[a][0]*group[b][0],-1))
    for a,b,c in IDENTITIES:
        for group in [values,cs]:extra.append(np.sum((group[a]-group[b]-group[c])[0]**2,-1))
    other=ef1-ef0
    extra.extend([np.sum((qf-other)[0]**2,-1),np.sum((center(qf,text)-center(other,text))[0]**2,-1)])
    return values,np.concatenate([*rows,np.stack(extra,-1)],-1)


def summarize(rows):
    value={p:panel(rows[:,i*8:i*8+8]) for i,p in enumerate(PORTS)}
    sums=dict(zip(ROW_COLUMNS,map(float,rows.sum(0))));account={n:sums[n] for n in EXTRAS}
    for a,b,c in IDENTITIES:
        for suffix,cross in [('energy','Inner'),('centeredEnergy','CenteredInner')]:
            account[a+'From'+b+c+'_'+suffix+'AccountRemainder']=sums[a+'_'+suffix]-sums[b+'_'+suffix]-sums[c+'_'+suffix]-sums['twice'+b+c+cross]
    ratio=lambda a,b:float(np.sqrt(a/b)) if b>0 else None
    pairs=[('Ef1','E1'),('Ef0','E0'),('Qf','Q'),('Qf','TN'),('B','Q'),('A0','E0'),('A1','E1'),('Tf','D')]
    value['signedAccount']=account
    value['relativeNorms']={a+'Over'+b:ratio(sums[a+'_energy'],sums[b+'_energy']) for a,b in pairs}
    value['centeredRelativeNorms']={a+'Over'+b:ratio(sums[a+'_centeredEnergy'],sums[b+'_centeredEnergy']) for a,b in pairs}
    col=lambda key:rows[:,ROW_COLUMNS.index(key)]
    value['fixedVsDynamic']={}
    for name,fixed,dynamic in [('parent','Ef0','E0'),('own','Ef1','E1'),('response','Qf','Q')]:
        f,d=col(fixed+'_energy'),col(dynamic+'_energy')
        value['fixedVsDynamic'][name]={'improve':int(np.sum(f<d)),'equal':int(np.sum(f==d)),'worsen':int(np.sum(f>d))}
    value['nonzeroParentAssociationRows']=int(np.sum(col('A0_energy')>0))
    value['zeroDenominatorRows']={p:int(np.sum(col(p+'_targetEnergy')==0)) for p in PORTS}
    return value


def execute(root):
    from template_transport import work
    from flux_clock_study import verify
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);s=read(root/'inputs.json')
    out=root/'response-result';tick=time.perf_counter();c=read(out/'candidate.json')
    seal=sha(out/'candidate-seal.json');exited=read(out/'producer-exit.json')
    if read(out/'candidate-seal.json')!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')} or exited['returncode']!=0 or exited['requestSHA256']!=r['requestSHA256'] or exited['candidateSealSHA256']!=seal or exited['producerPID']==os.getpid():
        raise ValueError('Observed complete two-arm seal/exit differs')
    def streams(fields):return np.concatenate([load(fields[n]) for n in ['text','image']],1)
    xh=streams(s['sourceFields']);x=streams(s['comparisonFields']['parentIncoming'])
    y=streams(s['comparisonFields']['parentOutgoing']);yh=streams(s['comparisonFields']['ownOutgoing'])
    y0=streams(s['comparisonFields']['matchedOutgoing']);y1=load(s['comparisonFields']['denseOwnOutgoing'])
    z0=load(c['fields']['parent']['exit']);z1=load(c['fields']['own']['exit'])
    values,rows=accounts(x,xh,y,yh,y0,y1,z0,z1)
    fields={n:retain(root/'arrays/analysis',n,values[n]) for n in RETAIN}
    fields['rowStatistics']=retain(root/'arrays/analysis','rowStatistics',rows)
    if {n:{'shape':v['shape'],'dtype':v['logicalDtype']} for n,v in fields.items()}!=analysis_abis() or not np.isfinite(rows).all():
        raise ValueError('Finite eight-corner ABI differs')
    panels={n:summarize(rows[lo:hi]) for n,lo,hi in [('all',0,1536),('text',0,512),('image',512,1536)]}
    with (out/'observations.jsonl').open('x') as f:
        for i,row in enumerate(rows):f.write(json.dumps({'row':i,'stratum':'text' if i<512 else 'image','statistics':dict(zip(ROW_COLUMNS,map(float,row)))},allow_nan=False)+'\n')
    if sha(out/'candidate-seal.json')!=seal:raise ValueError('Candidate changed after release')
    save(out/'analysis-fields.json',{'fields':fields,'candidateSealSHA256':seal,'rowColumns':ROW_COLUMNS})
    save(out/'analysis-execution.json',{'products':[],'diagnosticMatrixMACs':0,'device':'cpu','dtype':'float64','numpyVersion':np.__version__,
        'analyzerPID':os.getpid(),'candidateSealSHA256':seal,'producerExitSHA256':sha(out/'producer-exit.json'),
        'scope':'Post-exit eight-corner differences/centering/elementwise products/reductions only; no learned map/matrix product. NonzeroA0 does not block collection.'})
    save(out/'report.json',{'schema':'tce-template-transport-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal,
        'fixedAddress':{'clock':0,'singleBlock':1},'panels':panels,'work':work(),'observations':1536,
        'analysisSecondsAfterImports':time.perf_counter()-tick,'newImages':0,'newFits':0,'newBlockEvaluations':2,
        'mechanics_status':'complete','instrument_status':'two-input-fixed-template-eight-corner-decision-audit',
        'trend_status':'Allrows/fullcentered/signed A0/A1/B/Ef0/Ef1/Tf/Qf retained,without quality gate.','terminal_status':'unassessed',
        'scope':'Explicit reloaded finite template onparent/own. Dense comparator retainsmomentattention;not independentfullattention. Jointimplementedtemplatepolicyeffect only,alllearnedmaps/state/clocks/templatehistory paid. No image/promptgeneralization/zero-neural claim.'})
    for name in ['request.json','inputs.json','resource-basis.json','frozen-plan.md','implementation-notes.md','intake.json']:
        shutil.copyfile(root/name,out/name)


if __name__=='__main__':execute(sys.argv[1])
