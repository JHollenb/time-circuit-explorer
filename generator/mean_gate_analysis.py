"""Post-release decomposition; subtraction-defined changing term is not a Down."""
from pathlib import Path
import json
import os
import shutil
import sys
import time
from encoder_atoms import np, read, save, sha
from paired_fields import load, retain, metrics

ROW_COLUMNS=['totalEnergy','linearEnergy','changingEnergy','policyEnergy','twiceLinearChanging',
    'twiceLinearPolicy','twiceChangingPolicy','linearTotalInner','fullResidualEnergy',
    'centeredTotalEnergy','centeredLinearEnergy','centeredChangingEnergy','centeredPolicyEnergy',
    'twiceCenteredLinearChanging','twiceCenteredLinearPolicy','twiceCenteredChangingPolicy',
    'centeredLinearTotalInner','centeredResidualEnergy','residualMeanEnergy','outputClosureEnergy']


def analysis_abis(dim=3072,mlp=9216,text=512,image=1024):
    shapes={'changingProduct':[1,text+image,mlp]}
    shapes.update({n:[1,text+image,dim] for n in ['changingOutput','gatedChanging','gatedChangingCentered','gatedPolicyCentered','fullResidual','centeredResidual','centeredTotal']})
    shapes.update({n:[1,2,dim] for n in ['gatedChangingMean','gatedPolicyMean','totalMean']})
    shapes.update({n:[3,3,3] for n in ['fullGram','meanGram','centeredGram']})
    shapes['rowStatistics']=[1,text+image,len(ROW_COLUMNS)]
    return {n:{'shape':v,'dtype':'float64'} for n,v in shapes.items()}


def decompose(produced,source,comparison,text=512):
    n=source['activation'].shape[1]
    def mean(x): return np.concatenate([x[:,:text].mean(1,keepdims=True),x[:,text:].mean(1,keepdims=True)],1)
    def expand(x): return np.concatenate([np.repeat(x[:,:1],text,axis=1),np.repeat(x[:,1:],n-text,axis=1)],1)
    def center(x): return x-expand(mean(x))
    gate=source['gate']; l=produced['gatedLinear']; b=gate*(comparison['centeredOutput']-produced['linearOutput']); q=gate*comparison['policyOutput']; x=gate*comparison['totalOutput']
    parts=[l,b,q]; means=[expand(mean(v)) for v in parts]; centered=[v-m for v,m in zip(parts,means)]
    cx=center(x); full=x-l; full_mean=expand(mean(full)); cr=full-full_mean; closure=x-(l+b+q)
    delta=source['upReference']-expand(source['upMean'])
    bp=(source['activation']-expand(produced['activationMean']))*delta
    bp[:,:,source['selection_support'].astype(np.int64)]=0
    product_closure=comparison['centeredProduct']-produced['linearProduct']-bp
    def gram(values):
        return np.array([[[float(np.sum(a[:,lo:hi]*b[:,lo:hi])) for b in values] for a in values] for lo,hi in [(0,n),(0,text),(text,n)]],dtype=np.float64)
    energy=lambda v:np.sum(v*v,axis=-1)
    inner=lambda a,b:np.sum(a*b,axis=-1)
    rows=np.stack([energy(x),energy(l),energy(b),energy(q),2*inner(l,b),2*inner(l,q),2*inner(b,q),inner(x,l),energy(full),
        energy(cx),energy(centered[0]),energy(centered[1]),energy(centered[2]),2*inner(centered[0],centered[1]),
        2*inner(centered[0],centered[2]),2*inner(centered[1],centered[2]),inner(cx,centered[0]),energy(cr),energy(full_mean),energy(closure)],-1)
    fields={'changingProduct':bp,'changingOutput':comparison['centeredOutput']-produced['linearOutput'],'gatedChanging':b,
        'gatedChangingCentered':centered[1],'gatedPolicyCentered':centered[2],'fullResidual':full,'centeredResidual':cr,'centeredTotal':cx,
        'gatedChangingMean':mean(b),'gatedPolicyMean':mean(q),'totalMean':mean(x),
        'fullGram':gram(parts),'meanGram':gram(means),'centeredGram':gram(centered),'rowStatistics':rows}
    diagnostics={'productExpansion':metrics(produced['linearProduct']+bp,comparison['centeredProduct']),
        'productClosureMaxAbs':float(np.max(np.abs(product_closure))),
        'outputClosure':metrics(l+b+q,x),'centeredResidualDistributivity':metrics(cr,cx-centered[0]),
        'producerGatedMeanReduction':metrics(produced['gatedLinearMean'],mean(l))}
    return fields,diagnostics


def ratio(a,b): return float(np.sqrt(a/b)) if b>0 else None


def panel(rows,grams):
    total=float(rows[:,0].sum()); full=float(rows[:,8].sum()); centered=float(rows[:,9].sum()); residual=float(rows[:,17].sum()); mean=float(rows[:,18].sum())
    gram_accounts={}
    for name,g in grams.items():
        gram_accounts[name]={'gram':g.tolist(),'energies':np.diag(g).tolist(),'twiceCross':[float(2*g[0,1]),float(2*g[0,2]),float(2*g[1,2])],
            'cosines':{label:float(g[a,b]/np.sqrt(g[a,a]*g[b,b])) if g[a,a]>0 and g[b,b]>0 else None for label,a,b in [('linearChanging',0,1),('linearPolicy',0,2),('changingPolicy',1,2)]}}
    return {'rows':len(rows),'totalEnergy':total,'fullResidualEnergy':full,'fullResidualOverTotalRMS':ratio(full,total),
        'fullRemovedEnergyFraction':(total-full)/total if total else None,
        'paidConstantOnlyResidualEnergy':centered,'paidConstantOnlyResidualOverTotalRMS':ratio(centered,total),
        'paidConstantOnlyScope':'Post-release optimal per-stream constant from supplied total discrepancy, not producer output. Same constant family as76; finite CPU reduction may differ from76 GPU mean.',
        'centeredTotalEnergy':centered,'centeredResidualEnergy':residual,'centeredResidualOverCenteredTotalRMS':ratio(residual,centered),
        'centeredResidualOverFullTotalRMS':ratio(residual,total),'centeredRemovedEnergyFraction':(centered-residual)/centered if centered else None,
        'residualMeanEnergy':mean,'residualMeanOverFullTotalRMS':ratio(mean,total),'residualPythagoreanDiscrepancy':full-mean-residual,
        'fullImprovedRows':int(np.sum(rows[:,8]<rows[:,0])),'fullWorsenedRows':int(np.sum(rows[:,8]>rows[:,0])),
        'centeredImprovedRows':int(np.sum(rows[:,17]<rows[:,9])),'centeredWorsenedRows':int(np.sum(rows[:,17]>rows[:,9])),
        'fullPositiveAlignmentRows':int(np.sum(rows[:,7]>0)),'fullNegativeAlignmentRows':int(np.sum(rows[:,7]<0)),
        'linearNormOverTotalNorm':ratio(float(rows[:,1].sum()),total),'gramAccounts':gram_accounts,
        'fullGramEnergyClosureResidual':float(grams['full'].sum()-total),'centeredGramEnergyClosureResidual':float(grams['centered'].sum()-centered)}


def execute(root):
    from flux_clock_study import verify
    from mean_gate import SOURCE_NAMES,COMPARISON_NAMES
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);out=root/'response-result';started=time.perf_counter()
    s=read(root/'inputs.json');c=read(out/'candidate.json');e=read(out/'execution.json');seal=read(out/'candidate-seal.json');sealed=sha(out/'candidate-seal.json')
    if seal!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')}: raise ValueError('Producer release absent')
    exited=read(out/'producer-exit.json')
    if exited['returncode']!=0 or exited['candidateSealSHA256']!=sealed or exited['requestSHA256']!=r['requestSHA256'] or exited['producerPID']==os.getpid(): raise ValueError('Producer has not exited before comparisons')
    produced={n:load(v) for n,v in c['fields'].items()}
    source={n:load(s['sourceFields'][n]) for n in SOURCE_NAMES if n!='weightD'}
    comparison={n:load(s['comparisonFields'][n]) for n in COMPARISON_NAMES}
    values,diagnostics=decompose(produced,source,comparison)
    if {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in values.items()}!=analysis_abis(): raise ValueError('Postrelease field ABI differs')
    fields={n:retain(root/'arrays/analysis',n,v) for n,v in values.items()}; rows=values['rowStatistics'][0]
    panels={name:panel(rows[part],{kind:values[kind+'Gram'][i] for kind in ['full','mean','centered']})
        for i,(name,part) in enumerate([('all',slice(None)),('text',slice(0,512)),('image',slice(512,None))])}
    with (out/'observations.jsonl').open('x') as stream:
        for index,row in enumerate(rows):
            entry={'row':index,'stratum':'text' if index<512 else 'image',**dict(zip(ROW_COLUMNS,map(float,row)))}
            entry.update(fullEnergyImprovement=float(row[0]-row[8]),centeredEnergyImprovement=float(row[9]-row[17]),
                fullCorrectionCosine=float(row[7]/np.sqrt(row[0]*row[1])) if row[0]>0 and row[1]>0 else None,
                centeredCorrectionCosine=float(row[16]/np.sqrt(row[9]*row[10])) if row[9]>0 and row[10]>0 else None)
            stream.write(json.dumps(entry,sort_keys=True,allow_nan=False)+'\n')
    if sha(out/'candidate-seal.json')!=sealed or sha(out/'candidate.json')!=seal['candidateSHA256'] or sha(out/'execution.json')!=seal['executionSHA256']: raise ValueError('Producer changed after release')
    save(out/'analysis-fields.json',{'fields':fields,'candidateSealSHA256':sealed,'scope':'Post-exit derived Bdelta/means/centered/Gram/residual arrays. No independent changing-product Down or correction application.'})
    save(out/'report.json',{'schema':'tce-mean-gate-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':sealed,
        'fixedAddress':{'clock':3,'singleBlock':0},'panels':panels,'diagnostics':diagnostics,'observations':1536,'totalMatrixMACs':e['totalMatrixMACs'],
        'analysisSecondsAfterImports':time.perf_counter()-started,'newImages':0,'newContinuations':0,'newFits':0,
        'mechanics_status':'complete','instrument_status':'one-paid-own-state-diagnostic','trend_status':'All full/centered signed energies and heterogeneous rows retained','terminal_status':'unassessed',
        'scope':'Primary residual is X-L, including unproduced mean. Centered score removes per-stream residual mean as a paid oracle diagnostic, not producer accuracy. Changing write is subtraction-defined YR-YL with reassociation. Original saved generation/Up/activation/gates/support/D remain paid; no native finite block error/image causality/economical compilation/generalization/zero-neural claim.'})
    for name in ['request.json','inputs.json','resource-basis.json','frozen-plan.md']: shutil.copyfile(root/name,out/name)
    print('MEAN_GATE_ANALYZED',flush=True)


if __name__=='__main__': execute(sys.argv[1])
