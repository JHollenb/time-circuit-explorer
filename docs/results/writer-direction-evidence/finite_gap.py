"""Authenticated public-scalar differences; no tensors or new model computation."""
import hashlib
import json
from pathlib import Path
import sys
import time

def main(destination):
    tick=time.perf_counter();base=Path(__file__).resolve().parent.parent/'writer-direction';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    publication=base/'publication.json';assert sha(publication)=='e27390d0026b5d85f123bb28d6a18998e2535357b59391b14951b76b29023529'
    p=json.loads(publication.read_text());analysis=base/'analysis.json';assert sha(analysis)==p['files']['analysis.json']['sha256'];a=json.loads(analysis.read_text());rows=[]
    for objective in a['objectives']:
        s=next(x for x in a['summaries'] if x['candidate']==objective['candidate'] and x['group']=='all');slope=s['localModelSlopes'];parts=[]
        for term in ('Hendpoint','Hresponse','Zendpoint','Zresponse'):
            change=objective['finiteTermChanges'][term]/4;linear=slope[term+'.admitted']
            parts.append({'term':term,'actualContributionToMeanLossChange':change,'anchoredIdealLocalSlope':linear,'impliedFiniteMinusLinear':change-linear})
        rows.append({'candidate':objective['candidate'],'finiteLossChange':objective['finiteScoreChange'],'anchoredIdealLocalSlope':slope['saved.admitted'],
            'impliedFiniteMinusLinear':objective['finiteScoreChange']-slope['saved.admitted'],
            'writerToProposalSignedProductRatio':slope['saved.writer']/slope['saved.proposal'] if slope['saved.proposal'] else None,'components':parts})
    # A negative derivative can overshoot even an exact linear consumer's squared loss.
    parent=0.;step=3.;loss=lambda x:(x-1.)**2;linear=2*(parent-1.)*step;change=loss(parent+step)-loss(parent)
    assert (linear,change,change-linear)==(-6.,3.,9.)
    result={'schema':'tce-public-scalar-finite-gap-v1','sourceSHA256':sha(Path(__file__)),'publicationSHA256':sha(publication),'analysisSHA256':sha(analysis),'rows':rows,
        'suppliedLinearConsumerWitness':{'parent':parent,'step':step,'linear':linear,'finiteLossChange':change,'quadraticRemainder':change-linear},
        'scope':'Differences inferred from published scalar summaries. The finite-minus-linear gap is not an independently measured curvature term or a unique attribution to BF16/Jacobian error. No scientific vector,normalization,fit,prompt or image evaluated.'}
    result['secondsBeforeSerialization']=time.perf_counter()-tick
    with Path(destination).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'rows':len(rows),'sourceSHA256':result['sourceSHA256'],'secondsBeforeSerialization':result['secondsBeforeSerialization']}))

if __name__=='__main__':main(sys.argv[1])
