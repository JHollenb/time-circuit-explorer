"""Public-scalar inference only; three supplied coordinate witnesses, no arrays."""
import hashlib
import json
import math
from pathlib import Path
import sys

PUBLICATION='8bb04982247fdcaeab39b4d98565851d4988bde2991da23660570da8a89acce4'


def geometry(own,native,error):
    if any(not math.isfinite(x) or x<0 for x in (own,native,error)) or not own or not native:raise ValueError('Positive finite own/native energies required')
    inner=(own+native-error)/2;cosine=inner/math.sqrt(own*native);positive=max(0.,cosine)
    return {'inferredInnerProduct':inner,'inferredCosine':cosine,'normGain':math.sqrt(own/native),'bestNonnegativeOwnScale':max(0.,inner/own),
            'minimumRayErrorRelativeToNativeD':math.sqrt(max(0.,1-positive*positive))}


def run(root):
    for a,b,cosine,gain,scale,err in [(4.,1.,1.,2.,.5,0.),(1.,1.,0.,1.,0.,1.),(1.,1.,-1.,1.,0.,1.)]:
        distance=a+b-2*cosine*math.sqrt(a*b);g=geometry(a,b,distance)
        for key,expected in [('inferredCosine',cosine),('normGain',gain),('bestNonnegativeOwnScale',scale),('minimumRayErrorRelativeToNativeD',err)]:
            if g[key]!=expected:raise ValueError('Independent ray witness differs')
    base=Path(root)/'docs/results/coupled-normalization';pubpath=base/'publication.json';digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if digest(pubpath)!=PUBLICATION:raise ValueError('Publication identity differs')
    pub=json.loads(pubpath.read_text())
    for name in ('analysis.json','request.json'):
        if digest(base/name)!=pub['files'][name]['sha256']:raise ValueError('Public scalar member differs')
    a=json.loads((base/'analysis.json').read_text());own=a['edges']['own']['normSquared']['D'];native=a['edges']['native']['normSquared']['D'];error=a['factors']['E']['normSquared']['D']
    return {'schema':'tce-coupled-normalization-public-scalar-direction-v1','publicationSHA256':PUBLICATION,'sourceSHA256':digest(Path(__file__)),
            'primitiveEnergies':{'ownD':own,'nativeD':native,'differenceD':error},'geometry':geometry(own,native,error),'suppliedCoordinateWitnesses':3,
            'scope':'Inferred from three published FP64 energies,not a newly measured coordinate angle. Best ray uses the native target and cannot serve. Denominator feedback,endpoint direction,finite effects and full response remain separate;nativeD is a component,not the full normalized response.'}


if __name__=='__main__':print(json.dumps(run(sys.argv[1]),indent=2,allow_nan=False))
