"""Post-exit saved-array comparisons only; no new learned reference or contraction."""
from pathlib import Path
import json
import shutil
import sys
import time
from encoder_atoms import read,save,sha,np
from paired_fields import load,retain,metrics
from matched_single_analysis import rows

COMMON=['normalized','queryRaw','keyRaw','valueRaw','gateRaw','activation','attentionQuery','attentionKey','attentionValue',
        'imageAttention','attentionRead','preResidualUpdate','textOut','imageOut','exit','alpha','textRead','gate',
        *['moment_'+n for n in ['mean','covariance','logZI','logZT','quadratic']]]
SELECTION=['rowNormSquared','weightedGateEnergy','score','rankedIndices','support','omittedScoreSum','selectedScoreSum']


def specifications():
    result=[]
    for mode in ['dense','sparse']:
        pairs=[(n,n) for n in COMMON]+([('product','product'),('completedUp','upRaw')] if mode=='dense' else [('upRaw','upRaw')])
        pairs += [('selection_'+n,'selection_'+n) for n in SELECTION] if mode=='sparse' else []
        for candidate,reference in pairs:
            result.append({'name':mode+'_'+candidate,'candidate':candidate,'reference':mode+'_'+reference,'candidateSelected':False,'referenceSelected':False,
                           'requiresSameSupport':mode=='sparse' and candidate=='upRaw'})
    for name,candidate,reference,cs,rs in [('dense_selectedUp','upRaw','dense_upRaw',False,True),
        ('dense_selectedProduct','product','dense_product',True,True),('sparse_selectedProduct','product','sparse_product',True,False)]:
        result.append({'name':name,'candidate':candidate,'reference':reference,'candidateSelected':cs,'referenceSelected':rs,'requiresSameSupport':name=='sparse_selectedProduct'})
    return result


def execute(root):
    from flux_clock_study import verify
    root=Path(root).resolve();out=root/'response-result';r=read(root/'request.json');verify(root,r);started=time.perf_counter()
    c=read(out/'candidate.json');e=read(out/'execution.json');seal=read(out/'candidate-seal.json');seal_sha=sha(out/'candidate-seal.json')
    if seal!={'requestSHA256':r['requestSHA256'],'candidateSHA256':sha(out/'candidate.json'),'executionSHA256':sha(out/'execution.json')}:raise ValueError('Candidate not sealed before comparisons')
    t=read(root/'targets.json')['fields'];s=read(root/'inputs.json');support=load(c['fields']['selection_support']).astype(np.int64)
    support_matched=np.array_equal(support,load(t['sparse_selection_support']).astype(np.int64))
    score=load(c['fields']['selection_score']);order=np.lexsort((np.arange(9216),-score))
    if not np.array_equal(support,np.sort(order[:2304])) or not np.array_equal(load(c['fields']['selection_rankedIndices']).astype(np.int64),order):raise ValueError('Own stable support differs')
    comparisons={};observations=0;specs=specifications()
    with (out/'observations.jsonl').open('x') as stream:
        def compare(name,a,b,payload_exact=None):
            nonlocal observations
            error=a-b
            comparisons[name]={'metrics':metrics(a,b),'numericExact':bool(np.array_equal(a,b)),'payloadExact':payload_exact,
                               'signedMeanError':float(error.mean()),'minError':float(error.min()),'maxError':float(error.max())}
            ar,br=rows(a),rows(b)
            for index in range(len(ar)):
                stream.write(json.dumps({'comparison':name,'row':index,**metrics(ar[index],br[index]),'signedMeanError':float((ar[index]-br[index]).mean())},sort_keys=True,allow_nan=False)+'\n');observations+=1
        for spec in specs:
            if spec['requiresSameSupport'] and not support_matched:
                comparisons[spec['name']]={'status':'not-applicable','reason':'Own and historical sparse support/channel order differ; retained full-state/product comparisons remain applicable.',
                    'ownSupportPayloadSHA256':c['fields']['selection_support']['payloadSHA256'],'historicalSupportPayloadSHA256':t['sparse_selection_support']['payloadSHA256']}
                continue
            ai,bi=c['fields'][spec['candidate']],t[spec['reference']];a,b=load(ai),load(bi)
            if spec['candidateSelected']:a=a[...,support]
            if spec['referenceSelected']:b=b[...,support]
            compare(spec['name'],a,b,ai['payloadSHA256']==bi['payloadSHA256'] if not(spec['candidateSelected'] or spec['referenceSelected']) else None)
        # Diagnostic reference only after release: mean of admitted old native Up,
        # not a serving input or an exact finite identity with mean-Z projection.
        dense_up=load(t['dense_upRaw']);mean_up=np.concatenate([dense_up[:,:512].mean(1,keepdims=True),dense_up[:,512:].mean(1,keepdims=True)],1)
        derived=retain(root/'arrays/derived','denseUpStreamMeanFP64',mean_up)
        compare('dense_streamAnchor',load(c['fields']['anchorUp']),mean_up)
        del dense_up,mean_up
    incoming=np.concatenate([load(s['sourceFields']['text']),load(s['sourceFields']['image'])],1)
    dense,sparse,own=[load(x) for x in [t['dense_exit'],t['sparse_exit'],c['fields']['exit']]]
    prior=sparse-dense;error=own-dense;correction=own-sparse;native_write=dense-incoming
    differences={n:retain(root/'arrays/differences',n,v) for n,v in [('priorError',prior),('ownError',error),('ownCorrection',correction)]}
    panels={}
    def describe(a,b,w):
        pe=float(np.sum(a*a));oe=float(np.sum(b*b));ce=float(np.sum((b-a)**2));cross=float(2*np.sum(a*(b-a)));den=float(np.linalg.norm(w))
        return {'priorErrorEnergy':pe,'ownErrorEnergy':oe,'correctionEnergy':ce,'twicePriorCorrectionInnerProduct':cross,
            'energyImprovement':pe-oe,'priorErrorRelativeDenseWrite':float(np.linalg.norm(a))/den if den else None,
            'ownErrorRelativeDenseWrite':float(np.linalg.norm(b))/den if den else None,'ownOverPriorErrorRMS':float(np.linalg.norm(b))/float(np.linalg.norm(a)) if pe else None,
            'energyClosureResidual':oe-(pe+ce+cross),'correctionOpposedPriorErrorCosine':metrics(-(b-a),a)['cosine']}
    for name,part in [('all',slice(None)),('text',slice(0,512)),('image',slice(512,None))]:panels[name]=describe(prior[:,part],error[:,part],native_write[:,part])
    improved=0
    with (out/'write-observations.jsonl').open('x') as stream:
        for row in range(1536):
            item=describe(prior[0,row],error[0,row],native_write[0,row]);improved+=item['energyImprovement']>0
            stream.write(json.dumps({'row':row,'stratum':'text' if row<512 else 'image',**item},sort_keys=True,allow_nan=False)+'\n')
    if sha(out/'candidate-seal.json')!=seal_sha or sha(out/'candidate.json')!=seal['candidateSHA256'] or sha(out/'execution.json')!=seal['executionSHA256']:raise ValueError('Actor changed after release')
    if e['totalMatrixMACs']!=r['prospectiveMatrixMACs']:raise ValueError('Actual candidate work differs')
    save(out/'differences.json',differences);save(out/'derived-reference.json',{'field':derived,'source':t['dense_upRaw'],'policy':'FP64 mean of post-seal archived dense admitted Up, separately text/image; not a supplied anchor or native finite identity.'})
    save(out/'comparison-specifications.json',{'comparisons':specs,'extra':{'name':'dense_streamAnchor','candidate':'anchorUp','derived':'denseUpStreamMeanFP64'},'scope':'All historical arrays opened only after own actor seal and process exit.'})
    save(out/'report.json',{'schema':'tce-mean-up-result-v1','requestSHA256':r['requestSHA256'],'candidateSealSHA256':seal_sha,'comparisons':comparisons,'writePanels':panels,
        'observations':observations,'writeObservations':1536,'improvedExitRows':int(improved),'supportMatchedSparse':bool(support_matched),'totalMatrixMACs':e['totalMatrixMACs'],'savingVsDenseLocalMACs':200321531904-e['totalMatrixMACs'],
        'newLocalCandidates':1,'newDenseReferences':0,'newImages':0,'newPrefixCalls':0,'newHeadCalls':0,'newVAECalls':0,'analysisSecondsAfterImports':time.perf_counter()-started,
        'mechanics_status':'complete','instrument_status':'one-common-input-own-local-write','trend_status':'All signed per-row improvements/countertrends retained','terminal_status':'unassessed',
        'scope':'One source-defined current-stream mean-Up completion with ownquarter/allgates/fulloutput, saved73dense/sparse comparators afterexit. No fullcurrentUpbank/targetanchor/fit/sweep/newdense/head/image. Originalmaps/states/historypaid; strictarbitrarypromptzero-neuralopen.'})
    for name in ['request.json','inputs.json','targets.json','resource-basis.json','frozen-plan.md']:shutil.copyfile(root/name,out/name)
    print('MEAN_UP_ANALYZED',flush=True)


if __name__=='__main__':execute(sys.argv[1])
