"""Three supplied exact/scalar witnesses; no model, arrays or fitted inputs."""
import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import struct
import time


def dot(a,b):
    return sum((x*y for x,y in zip(a,b)),Q(0))


def shared(metric,value):
    coefficient=dot(metric,value)/sum(metric)
    return [coefficient,coefficient]


def difference(a,b):
    return [x-y for x,y in zip(a,b)]


def admitted(value):
    bits=struct.unpack('<I',struct.pack('<f',value))[0]
    word=((bits+0x7fff+((bits>>16)&1))>>16)&65535
    if word&0x7f80==0x7f80:raise ValueError('Nonfinite supplied BF16 witness')
    return struct.unpack('<f',struct.pack('<I',word<<16))[0]


def investigate():
    g=[Q(2),Q(-1)];a=[Q(100),Q(1)];m=[Q(1),Q(1)]
    step=[-x/y for x,y in zip(g,a)];raw=shared(m,step);aligned=shared(a,step)
    values={'unconstrainedSlopeWithoutHalf':dot(g,step),'HMetricProjectedSlopeWithoutHalf':dot(g,raw),
            'AMetricProjectedSlopeWithoutHalf':dot(g,aligned),'negativeAMetricEnergy':-dot(aligned,[x*y for x,y in zip(a,aligned)])}
    assert values=={'unconstrainedSlopeWithoutHalf':Q(-26,25),'HMetricProjectedSlopeWithoutHalf':Q(49,100),
                    'AMetricProjectedSlopeWithoutHalf':Q(-1,101),'negativeAMetricEnergy':Q(-1,101)}
    assert all(x-y>=0 for x,y in zip(a,m))
    old=[Q(0),Q(1)];offset=difference(shared(m,old),old)
    assert offset==[Q(1,2),Q(-1,2)] and dot(g,offset)==Q(3,2)
    for dose in (Q(0),Q(1,2),Q(1)):
        actual=difference(shared(m,[x+dose*y for x,y in zip(old,step)]),old)
        assert actual==[x+dose*y for x,y in zip(offset,raw)]
    parent=admitted(admitted(256.+1.)+0.)
    states={name:admitted(admitted(256.+atom)+0.) for name,atom in [('zero',1.),('half',1.5),('full',2.)]}
    changes={k:v-parent for k,v in states.items()};secant=2*changes['half']-changes['full']
    assert parent==256. and states=={'zero':256.,'half':258.,'full':258.}
    assert changes=={'zero':0.,'half':2.,'full':2.} and secant==2. and secant!=changes['zero']
    return {'schema':'tce-projection-geometry-supplied-v1','cases':3,
        'metricMismatch':{'g':[str(x) for x in g],'A_diagonal':[str(x) for x in a],'M_diagonal':[str(x) for x in m],
            'sharedAtomDesignColumn':['1','1'],'step':[str(x) for x in step],'HMetricProjection':[str(x) for x in raw],
            'AMetricProjection':[str(x) for x in aligned],'exactProducts':{k:str(v) for k,v in values.items()}},
        'changedRoute':{'oldSlot':[str(x) for x in old],'newRouteZeroDoseOffset':[str(x) for x in offset],
            'zeroDoseLocalModelProductWithoutHalf':str(dot(g,offset)),'affineIdentityCheckedDoses':['0','1/2','1']},
        'finiteDose':{'H0':256.,'parentAtom':1.,'parentH':parent,'candidateH':states,'actualHChanges':changes,
            'twoHalfChangesMinusFull':secant,'actualZeroDoseChange':changes['zero']},
        'scope':'Exact supplied two-row/one-constant-atom projection and independent scalar BF16 witnesses. Not an observed model direction, native finite gradient, actual110 attribution or optimizer success.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('destination');args=parser.parse_args();tick=time.perf_counter();result=investigate()
    result['sourceSHA256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();result['secondsAfterStandardLibraryImports']=time.perf_counter()-tick
    with Path(args.destination).open('x') as output:json.dump(result,output,indent=2,allow_nan=False);output.write('\n')
    print(json.dumps({'cases':result['cases'],'sourceSHA256':result['sourceSHA256'],'secondsAfterStandardLibraryImports':result['secondsAfterStandardLibraryImports']}))


if __name__=='__main__':main()
