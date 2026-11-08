"""Expand the endpoint/response tradeoff using published scalar energies only."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(source,destination):
    source=Path(source);r=json.loads(source.read_text());panels={}
    for name,panel in r['panels'].items():
        a=panel['signedAccount'];entries={}
        for label,suffix in [('full','energy'),('centered','centeredEnergy')]:
            energy=lambda n:a[n+'_'+suffix]
            parent=energy('Ef0')-energy('E0')
            response=energy('Qf')-energy('Q')
            dynamic_cross=energy('E1')-energy('E0')-energy('Q')
            fixed_cross=energy('Ef1')-energy('Ef0')-energy('Qf')
            cross=fixed_cross-dynamic_cross
            own=energy('Ef1')-energy('E1')
            entries[label]={'parentErrorEnergyChange':parent,'responseErrorEnergyChange':response,
                'dynamicTwiceParentErrorResponseInner':dynamic_cross,'fixedTwiceParentErrorResponseInner':fixed_cross,
                'signedCrossChange':cross,'ownErrorEnergyChange':own,'scalarExpansionRemainder':own-parent-response-cross}
        panels[name]=entries
    result={'schema':'tce-template-transport-tradeoff-v1','sourceReportSHA256':sha(source),'summarizerSourceSHA256':sha(__file__),
        'panels':panels,'scope':'Published scalar energies only. General expansion includes parent-error change; A0=0 observed here. Crosses inferred from squared-norm identities,not independent new vector products or positive causal shares.'}
    with Path(destination).open('x') as f:json.dump(result,f,indent=2)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args()
    print(json.dumps(execute(a.source,a.destination),indent=2))
