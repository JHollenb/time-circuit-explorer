"""Reconstruct ideal ray geometry from authenticated public scalar rows only."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(prediction_rms,target_rms,cosine,epsilon):
    p,t,c,e=prediction_rms,target_rms,cosine,epsilon
    if any(not math.isfinite(x) for x in (p,t,c,e)) or p<=0 or t<=0 or not -1<=c<=1 or e<0:raise ValueError('Invalid public geometry')
    alpha=max(0.,c*t/p);ray=math.sqrt(max(0.,1-max(0.,c)**2));x=p/math.sqrt(p*p+e);y=t/math.sqrt(t*t+e)
    radial=(x-y)**2;angular=2*x*y*(1-c)
    return {'radiusRatioOwnToNative':p/t,'baselineAngleDegrees':math.degrees(math.acos(c)),
            'bestNonnegativeScaleOwnTowardNative':alpha,'bestRayErrorRelativeToNative':ray,
            'normMatchedStateErrorRelativeToNative':math.sqrt(2*(1-c)),
            'idealNormalizedUnweightedRadialSquared':radial,'idealNormalizedUnweightedAngularSquared':angular,
            'idealNormalizedUnweightedErrorRMS':math.sqrt(math.fsum((radial,angular)))}


def mechanics():
    # Independent coordinate witnesses: p=(1,0), a=(3,4); then opposite and collinear rays.
    g=geometry(math.sqrt(.5),math.sqrt(12.5),3/5,0)
    assert math.isclose(g['bestNonnegativeScaleOwnTowardNative'],3.,rel_tol=1e-14)
    assert math.isclose(g['bestRayErrorRelativeToNative'],4/5,rel_tol=1e-14)
    assert math.isclose(g['normMatchedStateErrorRelativeToNative'],math.sqrt(20/25),rel_tol=1e-14)
    g=geometry(1.,1.,-1.,0);assert g['bestNonnegativeScaleOwnTowardNative']==0 and g['bestRayErrorRelativeToNative']==1
    g=geometry(1.,2.,1.,0);assert g['bestNonnegativeScaleOwnTowardNative']==2 and g['bestRayErrorRelativeToNative']==g['idealNormalizedUnweightedErrorRMS']==0
    return 3


def analyze(repo):
    start=time.perf_counter();repo=Path(repo);base=repo/'docs/results/contrast-writer';pub=json.loads((base/'publication.json').read_text());source=base/'observations.jsonl.gz'
    if sha(base/'publication.json')!='0a3a231312dc619a3266a7b15b6e35640b95616575765176c9fed10f041b5d54' or sha(source)!=pub['files']['observations.jsonl.gz']['sha256']:raise ValueError('Frozen public source differs')
    child=repo/'docs/results/finite-consumer-crossover';child_pub=json.loads((child/'publication.json').read_text())
    if sha(base/'programs.json')!=pub['files']['programs.json']['sha256'] or sha(child/'publication.json')!='81f4fb4078446d9ddeb510ac0115353aeeca701cd03f0b08d13786c64e8fc6ee' or sha(child/'request.json')!=child_pub['files']['request.json']['sha256']:raise ValueError('Program/crossover request publication differs')
    catalog=json.loads((base/'programs.json').read_text());request=json.loads((child/'request.json').read_text())
    if catalog['accepted']['programSHA256']!=catalog['candidate']['programSHA256'] or catalog['accepted']['programSHA256']!=request['acceptedProgramSHA256'] or request['rmsNormEpsilon']!=1e-6:raise ValueError('Accepted endpoint/epsilon identity differs')
    rows=[]
    with gzip.open(source,'rt') as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('case') in ('00','10') and row.get('method')=='candidate' and row.get('port')=='H1' and row.get('position')==9:rows.append(row)
    if len(rows)!=2 or {r['case'] for r in rows}!={'00','10'} or any(r['active'] is not True for r in rows):raise ValueError('Two unchanged active baseline rows required')
    results=[{'case':r['case'],'inputRow':r,'geometry':geometry(r['predictionRMS'],r['targetRMS'],r['cosine'],request['rmsNormEpsilon'])} for r in rows]
    return {'schema':'tce-public-scalar-baseline-geometry-v1','analysisSourceSHA256':sha(__file__),'publicTransportSHA256':sha(source),'publicParentPublicationSHA256':sha(base/'publication.json'),
            'acceptedProgramSHA256':request['acceptedProgramSHA256'],'epsilon':request['rmsNormEpsilon'],'suppliedMechanicsFixtures':mechanics(),'rows':results,'wallSecondsBeforeSerialization':time.perf_counter()-start,
            'formulas':{'bestScale':'alpha=max(0,c*T/P), minimizes ||alpha*p-a||/||a|| over alpha>=0','rayResidual':'sqrt(1-max(0,c)^2), relative to native target norm',
                        'normMatchedResidual':'sqrt(2*(1-c)), relative to native norm after matching radius','idealNormalization':'x=P/sqrt(P^2+epsilon),y=T/sqrt(T^2+epsilon); squared RMS=(x-y)^2+2*x*y*(1-c)'},
            'newLearnedMatrixMACs':0,'newOriginalNormalizationReconstructions':0,'scientificArraysRead':0,'newImages':0,
            'scope':'Public scalar algebra on two authenticated historical endpoints, not actual native finite/gamma normalization operands, neural serving, consumer-response attribution or new vector replay. Source norms/cosines remain producer measurements; minima are ideal ray formulas evaluated in binary64. Target-assisted scalar fits are diagnostic, not a serving initializer.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('repo');args=parser.parse_args();print(json.dumps(analyze(args.repo),indent=2,allow_nan=False))
