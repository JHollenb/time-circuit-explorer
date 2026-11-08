"""Derive primary increment denominators and vector angles from public scalar energies."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def execute(source,destination):
    source=Path(source);r=json.loads(source.read_text());panels={}
    ratio=lambda a,b:math.sqrt(a/b) if b>0 else None
    def cosine(a,b,q):
        return max(-1.,min(1.,(a+b-q)/(2*math.sqrt(a*b)))) if a>0 and b>0 else None
    for name,p in r['panels'].items():
        a=p['signedAccount'];own=p['denseIncrement']['energies'];parent=p['ownIncrement']['energies']
        full=cosine(a['E0_energy'],a['E1_energy'],a['Q_energy'])
        centered=cosine(a['E0_centeredEnergy'],a['E1_centeredEnergy'],a['Q_centeredEnergy'])
        panels[name]={
            'E1OverActualDenseOwnIncrementRMS':ratio(a['E1_energy'],own['candidateEnergy']),
            'centeredE1OverActualDenseOwnIncrementRMS':ratio(a['E1_centeredEnergy'],own['centeredCandidateEnergy']),
            'E0OverDenseParentIncrementRMS':ratio(a['E0_energy'],parent['targetEnergy']),
            'centeredE0OverDenseParentIncrementRMS':ratio(a['E0_centeredEnergy'],parent['centeredTargetEnergy']),
            'errorVectorCosineE0E1':full,'centeredErrorVectorCosineE0E1':centered,
            'errorVectorAngleDegrees':math.degrees(math.acos(full)) if full is not None else None,
            'centeredErrorVectorAngleDegrees':math.degrees(math.acos(centered)) if centered is not None else None,
            'comparatorVsReducedResponseCosine':cosine(a['TN_energy'],a['T_energy'],a['Q_energy']),
            'centeredComparatorVsReducedResponseCosine':cosine(a['TN_centeredEnergy'],a['T_centeredEnergy'],a['Q_centeredEnergy'])}
    value={'schema':'tce-dense-corner-increment-normalized-v1','sourceReportSHA256':hashlib.sha256(source.read_bytes()).hexdigest(),
           'summarizerSourceSHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'panels':panels,
           'scope':'Published scalar energies only,not real-array replay. E1 denominator is actualdense increment atowninput,E0 denominator is actualdense parentincrement. Pair angles/gains are not global sensitivity or positive causal shares.'}
    with Path(destination).open('x') as f:json.dump(value,f,indent=2)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args()
    print(json.dumps(execute(a.source,a.destination),indent=2))
