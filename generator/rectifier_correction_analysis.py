"""Post-exit full/centered forecast comparison and new-expression bookkeeping."""
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
from encoder_atoms import np,read,save,sha
from paired_fields import load,retain,metrics

TERMS=['linear','rectifier','remaining','policy'];PAIRS=list(itertools.combinations(range(4),2))
ROW_COLUMNS=['totalEnergy','centeredTotalEnergy','linearResidualEnergy','quadraticResidualEnergy','fullResidualEnergy',
    'centeredLinearResidualEnergy','centeredQuadraticResidualEnergy','centeredResidualEnergy','residualMeanEnergy','forecastEnergy','forecastTotalInner','outputClosureEnergy']
ROW_COLUMNS += [n+'Energy' for n in TERMS]+['twice'+TERMS[a].title()+TERMS[b].title() for a,b in PAIRS]
ROW_COLUMNS += ['centered'+n.title()+'Energy' for n in TERMS]+['twiceCentered'+TERMS[a].title()+TERMS[b].title() for a,b in PAIRS]


def analysis_abis(dim=3072,mlp=9216,text=512,image=1024):
    shapes={n:[1,text+image,mlp] for n in ['activationResidual','productResidual','productClosure']}
    shapes.update({n:[1,text+image,dim] for n in ['remainingOutput','gatedRemaining','fullResidual','centeredResidual']})
    shapes['gatedRemainingMean']=[1,2,dim];shapes.update({n:[3,4,4] for n in ['fullGram','meanGram','centeredGram']});shapes['rowStatistics']=[1,text+image,len(ROW_COLUMNS)]
    return {n:{'shape':v,'dtype':'float64'} for n,v in shapes.items()}


def decompose(produced,source,comparison,text=512):
    n=source['upReference'].shape[1]
    def mean(x):return np.concatenate([x[:,:text].mean(1,keepdims=True),x[:,text:].mean(1,keepdims=True)],1)
    def expand(x):return np.concatenate([np.repeat(x[:,:1],text,axis=1),np.repeat(x[:,1:],n-text,axis=1)],1)
    def center(x):return x-expand(mean(x))
    ar=comparison['activation']-produced['rectifier'];dm=comparison['activationMean']-produced['rectifierMean']
    expected=(ar-expand(dm))*(source['upReference']-expand(source['upMean']));expected[:,:,source['selection_support'].astype(np.int64)]=0
    product_residual=comparison['changingProduct']-produced['rectifierProduct'];pc=product_residual-expected
    gate=source['gate'];l=gate*source['linearOutput'];t=produced['gatedRectifier'];rem=(comparison['centeredOutput']-source['linearOutput'])-produced['rectifierOutput'];b=gate*rem;q=gate*comparison['policyOutput'];x=gate*comparison['totalOutput'];forecast=produced['forecast']
    parts=[l,t,b,q];means=[expand(mean(v)) for v in parts];centered=[v-m for v,m in zip(parts,means)]
    full=x-forecast;cr=center(full);cx=center(x);linear=x-l;old=x-comparison['quadraticForecast'];cl=center(linear);co=center(old);closure=x-(l+t+b+q)
    energy=lambda v:np.sum(v*v,axis=-1);inner=lambda a,b:np.sum(a*b,axis=-1)
    stats=[energy(x),energy(cx),energy(linear),energy(old),energy(full),energy(cl),energy(co),energy(cr),energy(expand(mean(full))),energy(forecast),inner(x,forecast),energy(closure)]
    for values in [parts,centered]:stats += [energy(v) for v in values]+[2*inner(values[a],values[b]) for a,b in PAIRS]
    def gram(values):return np.array([[[float(np.sum(a[:,lo:hi]*b[:,lo:hi])) for b in values] for a in values] for lo,hi in [(0,n),(0,text),(text,n)]],dtype=np.float64)
    fields={'activationResidual':ar,'productResidual':product_residual,'productClosure':pc,'remainingOutput':rem,'gatedRemaining':b,'fullResidual':full,'centeredResidual':cr,
        'gatedRemainingMean':mean(b),'fullGram':gram(parts),'meanGram':gram(means),'centeredGram':gram(centered),'rowStatistics':np.stack(stats,-1)}
    diagnostics={'productResidualExpression':metrics(expected,product_residual),'productClosureMaxAbs':float(np.max(np.abs(pc))),
        'activationResidualRange':{'min':float(np.min(ar)),'max':float(np.max(ar)),'maxAbs':float(np.max(np.abs(ar))),
            'scope':'Observed admittedA minus newrectifier; includes source/finite differences. IdealrealSiLU softening bound is not imposed on this field.'},
        'activationResidualMeanRegrouping':metrics(mean(ar),dm),'outputClosure':metrics(l+t+b+q,x),'forecastRegrouping':metrics(forecast,l+t),'centeredResidualDistributivity':metrics(cr,cx-center(forecast)),
        'newProductVsChanging':metrics(produced['rectifierProduct'],comparison['changingProduct'])}
    return fields,diagnostics


def panel(rows,grams):
    sums={n:float(rows[:,i].sum()) for i,n in enumerate(ROW_COLUMNS)};total=sums['totalEnergy'];ct=sums['centeredTotalEnergy'];full=sums['fullResidualEnergy'];cr=sums['centeredResidualEnergy']
    ratio=lambda a,b:float(np.sqrt(a/b)) if b>0 else None
    scores={'zero':1.0,'linear':ratio(sums['linearResidualEnergy'],total),'quadratic':ratio(sums['quadraticResidualEnergy'],total),'rectifier':ratio(full,total)}
    centered={'zero':1.0,'linear':ratio(sums['centeredLinearResidualEnergy'],ct),'quadratic':ratio(sums['centeredQuadraticResidualEnergy'],ct),'rectifier':ratio(cr,ct)}
    comparisons={}
    for name,col in [('zero','totalEnergy'),('linear','linearResidualEnergy'),('quadratic','quadraticResidualEnergy')]:
        a=rows[:,ROW_COLUMNS.index(col)];b=rows[:,ROW_COLUMNS.index('fullResidualEnergy')]
        comparisons[name]={'improve':int(np.sum(b<a)),'worsen':int(np.sum(b>a)),'equal':int(np.sum(b==a))}
    accounts={}
    for name,g in grams.items():
        accounts[name]={'terms':TERMS,'gram':g.tolist(),'twiceCross':{TERMS[a]+'_'+TERMS[b]:float(2*g[a,b]) for a,b in PAIRS},
            'cosines':{TERMS[a]+'_'+TERMS[b]:float(g[a,b]/np.sqrt(g[a,a]*g[b,b])) if g[a,a]>0 and g[b,b]>0 else None for a,b in PAIRS}}
    return {'rows':len(rows),'energies':sums,'fullResidualOverTotalRMS':scores,'centeredResidualOverCenteredTotalRMS':centered,'centeredResidualOverFullTotalRMS':ratio(cr,total),
        'fullRemovedEnergyFraction':(total-full)/total if total else None,'additionalRemovedLinearEnergyFraction':1-full/sums['linearResidualEnergy'] if sums['linearResidualEnergy'] else None,
        'additionalRemovedQuadraticEnergyFraction':1-full/sums['quadraticResidualEnergy'] if sums['quadraticResidualEnergy'] else None,'rowsVs':comparisons,'gramAccounts':accounts,
        'residualMeanOverFullTotalRMS':ratio(sums['residualMeanEnergy'],total),'residualPythagoreanDiscrepancy':full-sums['residualMeanEnergy']-cr,
        'fullGramClosureResidual':float(grams['full'].sum()-total),'centeredGramClosureResidual':float(grams['centered'].sum()-ct)}


def execute(root):
    from flux_clock_study import verify
    from rectifier_correction import COMPARISON_NAMES
    root=Path(root).resolve();r=read(root/'request.json');verify(root,r);out=root/'response-result';started=time.perf_counter();s=read(root/'inputs.json');c=read(out/'candidate.json');seal=read(out/'candidate-seal.json');sealed=sha(out/'candidate-seal.json');exited=read(out/'producer-exit.json')
    if seal!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')} or exited['returncode']!=0 or exited['candidateSealSHA256']!=sealed or exited['requestSHA256']!=r['requestSHA256'] or exited['producerPID']==os.getpid():raise ValueError('Producer seal/exit absent')
    produced={n:load(v) for n,v in c['fields'].items()};source={n:load(s['sourceFields'][n]) for n in ['upReference','upMean','selection_support','gate','linearOutput']};comparison={n:load(s['comparisonFields'][n]) for n in COMPARISON_NAMES}
    values,diagnostics=decompose(produced,source,comparison)
    if {n:{'shape':list(v.shape),'dtype':str(v.dtype)} for n,v in values.items()}!=analysis_abis():raise ValueError('Analysis ABI differs')
    fields={n:retain(root/'arrays/analysis',n,v) for n,v in values.items()};rows=values['rowStatistics'][0]
    panels={name:panel(rows[lo:hi],{kind:values[kind+'Gram'][i] for kind in ['full','mean','centered']}) for i,(name,lo,hi) in enumerate([('all',0,1536),('text',0,512),('image',512,1536)])}
    with (out/'observations.jsonl').open('x') as stream:
        for index,row in enumerate(rows):
            item={'row':index,'stratum':'text' if index<512 else 'image',**dict(zip(ROW_COLUMNS,map(float,row)))}
            item.update(energyImprovementVsZero=float(row[0]-row[4]),energyImprovementVsLinear=float(row[2]-row[4]),energyImprovementVsQuadratic=float(row[3]-row[4]))
            stream.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n')
    if sha(out/'candidate-seal.json')!=sealed or sha(out/'candidate.json')!=seal['candidateSHA256'] or sha(out/'execution.json')!=seal['executionSHA256']:raise ValueError('Producer changed after release')
    save(out/'analysis-fields.json',{'fields':fields,'candidateSealSHA256':sealed,'scope':'New expression residual from admittedA minus rectifier, no oldF/tailkernel/residualDown. Outputremainder subtraction-derived.'})
    save(out/'report.json',{'schema':'tce-rectifier-correction-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':sealed,'fixedAddress':{'clock':3,'singleBlock':0},'panels':panels,'diagnostics':diagnostics,'observations':1536,'totalMatrixMACs':43486543872,'rectifierElements':1536*9216,'analysisSecondsAfterImports':time.perf_counter()-started,
        'newImages':0,'newContinuations':0,'newFits':0,'mechanics_status':'complete','instrument_status':'one-paid-own-state-rectifier-diagnostic','trend_status':'Full/centered/mean/signed accounts and everyrow retained','terminal_status':'unassessed',
        'scope':'OriginalsavedGstar/Up/L/support/D/g/source/historypaid. One new rectifier plus learnedDown; no economical/zero-neuralservingclaim. New expression uses actualA-minusrectifier bookkeeping,not oldF. Full reconstructedmissingwrite notnativefiniteblockerror/imagecause. No tail/HDown/fit/meanrepair/continuation.'})
    for name in ['request.json','inputs.json','resource-basis.json','frozen-plan.md']:shutil.copyfile(root/name,out/name)


if __name__=='__main__':execute(sys.argv[1])
