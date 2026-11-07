"""Post-exit odd/even/finite product account and four signed output terms."""
from pathlib import Path
import itertools
import json
import os
import shutil
import sys
import time
from encoder_atoms import np,read,save,sha
from paired_fields import load,retain,metrics

TERMS=['linear','quadratic','remaining','policy'];PAIRS=list(itertools.combinations(range(4),2))
ROW_COLUMNS=['totalEnergy','forecastEnergy','forecastTotalInner','fullResidualEnergy','baselineResidualEnergy','centeredTotalEnergy',
    'centeredForecastEnergy','centeredForecastTotalInner','centeredResidualEnergy','centeredBaselineResidualEnergy','residualMeanEnergy','quadraticBaselineInner','outputClosureEnergy']
ROW_COLUMNS += [n+'Energy' for n in TERMS]+['twice'+TERMS[a].title()+TERMS[b].title() for a,b in PAIRS]
ROW_COLUMNS += ['centered'+n.title()+'Energy' for n in TERMS]+['twiceCentered'+TERMS[a].title()+TERMS[b].title() for a,b in PAIRS]


def analysis_abis(dim=3072,mlp=9216,text=512,image=1024):
    shapes={n:[1,text+image,mlp] for n in ['evenProduct','finiteProduct','productClosure']}
    shapes.update({n:[1,text+image,dim] for n in ['remainingOutput','gatedRemaining','fullResidual','centeredResidual']})
    shapes.update({n:[1,2,dim] for n in ['gatedRemainingMean','gatedPolicyMean']})
    shapes.update({n:[3,4,4] for n in ['fullGram','meanGram','centeredGram']});shapes['productGram']=[3,3,3];shapes['rowStatistics']=[1,text+image,len(ROW_COLUMNS)]
    return {n:{'shape':v,'dtype':'float64'} for n,v in shapes.items()}


def decompose(produced,source,comparison,text=512):
    n=source['upReference'].shape[1]
    def mean(x):return np.concatenate([x[:,:text].mean(1,keepdims=True),x[:,text:].mean(1,keepdims=True)],1)
    def expand(x):return np.concatenate([np.repeat(x[:,:1],text,axis=1),np.repeat(x[:,1:],n-text,axis=1)],1)
    def center(x):return x-expand(mean(x))
    g=produced['gateReference'];h=.5*g*np.tanh(.5*g);astar=.5*g+h;finite=comparison['activation']-astar
    du=source['upReference']-expand(source['upMean']);hp=(h-expand(mean(h)))*du;fp=(finite-expand(mean(finite)))*du
    support=source['selection_support'].astype(np.int64);hp[:,:,support]=0;fp[:,:,support]=0
    product_sum=produced['quadraticProduct']+hp+fp;product_closure=comparison['changingProduct']-product_sum
    gate=source['gate'];l=gate*source['linearOutput'];t=produced['gatedQuadratic'];rem=comparison['centeredOutput']-source['linearOutput']-produced['quadraticOutput'];b=gate*rem;q=gate*comparison['policyOutput'];x=gate*comparison['totalOutput'];forecast=produced['forecast']
    parts=[l,t,b,q];means=[expand(mean(v)) for v in parts];centered=[v-m for v,m in zip(parts,means)]
    full=x-forecast;full_mean=expand(mean(full));cr=full-full_mean;cx=center(x);ck=center(forecast);baseline=x-l;cb=center(baseline);closure=x-(l+t+b+q)
    def gram(values):return np.array([[[float(np.sum(a[:,lo:hi]*b[:,lo:hi])) for b in values] for a in values] for lo,hi in [(0,n),(0,text),(text,n)]],dtype=np.float64)
    energy=lambda v:np.sum(v*v,axis=-1);inner=lambda a,b:np.sum(a*b,axis=-1)
    stats=[energy(x),energy(forecast),inner(x,forecast),energy(full),energy(baseline),energy(cx),energy(ck),inner(cx,ck),energy(cr),energy(cb),energy(full_mean),inner(baseline,t),energy(closure)]
    for values in [parts,centered]:stats += [energy(v) for v in values]+[2*inner(values[a],values[b]) for a,b in PAIRS]
    fields={'evenProduct':hp,'finiteProduct':fp,'productClosure':product_closure,'remainingOutput':rem,'gatedRemaining':b,
        'fullResidual':full,'centeredResidual':cr,'gatedRemainingMean':mean(b),'gatedPolicyMean':mean(q),
        'fullGram':gram(parts),'meanGram':gram(means),'centeredGram':gram(centered),
        'productGram':gram([produced['quadraticProduct'],hp,fp]),'rowStatistics':np.stack(stats,-1)}
    diagnostics={'productExpansion':metrics(product_sum,comparison['changingProduct']),'productClosureMaxAbs':float(np.max(np.abs(product_closure))),
        'outputClosure':metrics(l+t+b+q,x),'forecastRegrouping':metrics(forecast,l+t),'centeredResidualDistributivity':metrics(cr,cx-ck),
        'evaluatedActivationVsAdmitted':metrics(astar,comparison['activation']),
        'activationMeanReduction':metrics(mean(comparison['activation']),comparison['activationMean'])}
    return fields,diagnostics


def ratio(a,b):return float(np.sqrt(a/b)) if b>0 else None


def gram_account(g,names):
    pairs=list(itertools.combinations(range(len(names)),2))
    return {'terms':names,'gram':g.tolist(),'energies':np.diag(g).tolist(),'twiceCross':{names[a]+'_'+names[b]:float(2*g[a,b]) for a,b in pairs},
        'cosines':{names[a]+'_'+names[b]:float(g[a,b]/np.sqrt(g[a,a]*g[b,b])) if g[a,a]>0 and g[b,b]>0 else None for a,b in pairs}}


def panel(rows,grams,product):
    sums={n:float(rows[:,i].sum()) for i,n in enumerate(ROW_COLUMNS)};total=sums['totalEnergy'];res=sums['fullResidualEnergy'];ct=sums['centeredTotalEnergy'];cr=sums['centeredResidualEnergy'];base=sums['baselineResidualEnergy']
    return {'rows':len(rows),'energies':sums,'fullResidualOverTotalRMS':ratio(res,total),'fullResidualOverBaselineResidualRMS':ratio(res,base),
        'fullRemovedEnergyFraction':(total-res)/total if total else None,'additionalRemovedBaselineEnergyFraction':(base-res)/base if base else None,
        'baselineResidualOverTotalRMS':ratio(base,total),'centeredResidualOverCenteredTotalRMS':ratio(cr,ct),'centeredResidualOverFullTotalRMS':ratio(cr,total),
        'centeredBaselineResidualOverCenteredTotalRMS':ratio(sums['centeredBaselineResidualEnergy'],ct),
        'paidConstantOnlyResidualOverTotalRMS':ratio(ct,total),'residualMeanOverFullTotalRMS':ratio(sums['residualMeanEnergy'],total),
        'residualPythagoreanDiscrepancy':res-sums['residualMeanEnergy']-cr,
        'fullImprovedVsZero':int(np.sum(rows[:,3]<rows[:,0])),'fullWorsenedVsZero':int(np.sum(rows[:,3]>rows[:,0])),
        'fullImprovedVsLinear':int(np.sum(rows[:,3]<rows[:,4])),'fullWorsenedVsLinear':int(np.sum(rows[:,3]>rows[:,4])),
        'centeredImprovedVsLinear':int(np.sum(rows[:,8]<rows[:,9])),'centeredWorsenedVsLinear':int(np.sum(rows[:,8]>rows[:,9])),
        'gramAccounts':{name:gram_account(g,TERMS) for name,g in grams.items()},'productSpaceAccount':gram_account(product,['quadratic','even','finite']),
        'fullGramEnergyClosureResidual':float(grams['full'].sum()-total),'centeredGramEnergyClosureResidual':float(grams['centered'].sum()-ct)}


def execute(root):
    from flux_clock_study import verify
    from odd_gate import COMPARISON_NAMES
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);out=root/'response-result';started=time.perf_counter();s=read(root/'inputs.json');c=read(out/'candidate.json');e=read(out/'execution.json');seal=read(out/'candidate-seal.json');sealed=sha(out/'candidate-seal.json');exited=read(out/'producer-exit.json')
    if seal!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')} or exited['returncode']!=0 or exited['candidateSealSHA256']!=sealed or exited['requestSHA256']!=r['requestSHA256'] or exited['producerPID']==os.getpid():raise ValueError('Producer seal/exit absent')
    produced={n:load(v) for n,v in c['fields'].items() if n not in ['weightG','normalizedFP64']};source={n:load(s['sourceFields'][n]) for n in ['upReference','upMean','selection_support','gate','linearOutput']};comparison={n:load(s['comparisonFields'][n]) for n in COMPARISON_NAMES}
    values,diagnostics=decompose(produced,source,comparison)
    if {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in values.items()}!=analysis_abis():raise ValueError('Analysis field ABI differs')
    fields={n:retain(root/'arrays/analysis',n,v) for n,v in values.items()};rows=values['rowStatistics'][0]
    panels={name:panel(rows[part],{kind:values[kind+'Gram'][i] for kind in ['full','mean','centered']},values['productGram'][i]) for i,(name,part) in enumerate([('all',slice(None)),('text',slice(0,512)),('image',slice(512,None))])}
    with (out/'observations.jsonl').open('x') as stream:
        for index,row in enumerate(rows):
            item={'row':index,'stratum':'text' if index<512 else 'image',**dict(zip(ROW_COLUMNS,map(float,row)))}
            item.update(energyImprovementVsZero=float(row[0]-row[3]),energyImprovementVsLinear=float(row[4]-row[3]),centeredImprovementVsLinear=float(row[9]-row[8]),
                forecastCosine=float(row[2]/np.sqrt(row[0]*row[1])) if row[0]>0 and row[1]>0 else None)
            stream.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n')
    if sha(out/'candidate-seal.json')!=sealed or sha(out/'candidate.json')!=seal['candidateSHA256'] or sha(out/'execution.json')!=seal['executionSHA256']:raise ValueError('Producer changed after release')
    save(out/'analysis-fields.json',{'fields':fields,'candidateSealSHA256':sealed,'scope':'H/F productspace only; output remainder subtraction-defined with reassociation. No H/F Down contraction.'})
    save(out/'report.json',{'schema':'tce-odd-gate-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':sealed,'fixedAddress':{'clock':3,'singleBlock':0},
        'panels':panels,'diagnostics':diagnostics,'observations':1536,'totalMatrixMACs':e['totalMatrixMACs'],'analysisSecondsAfterImports':time.perf_counter()-started,
        'newImages':0,'newContinuations':0,'newFits':0,'mechanics_status':'complete','instrument_status':'one-paid-own-state-quadratic-diagnostic','trend_status':'Full/centered signed terms and all rows retained','terminal_status':'unassessed',
        'scope':'Paid originalG/Down plus inheritedUp/L/maps/history. G*notnativeG. H*=.5G*tanh(.5G*),A*=.5G*+H*,F=A-A* postrelease only; finite causes unseparated. Full correction scored first; centered/constant oracle diagnostic only. No nativefiniteblock/imagecausality/economicalcompiler/generalization/zero-neuralclaim.'})
    for name in ['request.json','inputs.json','resource-basis.json','frozen-plan.md']:shutil.copyfile(root/name,out/name)
    print('ODD_GATE_ANALYZED',flush=True)


if __name__=='__main__':execute(sys.argv[1])
