"""Exact arithmetic allocation of the published operator ledger; no tensor math."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import json
from encoder_atoms import read,save,sha


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/image-partition-moments';e=read(p/'execution.json');r=read(p/'report.json');groups=Counter()
    for x in e['linears']:
        if x['kind']!='denoise':continue
        name=x['parameter'];mac=x['nominalMACs']
        if '.ff.' in name or '.ff_context.' in name:groups['jointMLP']+=mac
        elif 'to_qkv_mlp_proj' in name:
            if x['weightShape']!=[27648,3072]:raise ValueError('Fused projection geometry differs')
            groups['singleGateUp']+=mac*Fraction(2,3)
        elif name.startswith('singlePhysical.') and name.endswith('.QGU'):
            if x['weightShape']!=[21504,3072]:raise ValueError('Terminal QGU geometry differs')
            groups['terminalGateUp']+=mac*Fraction(6,7)
        elif 'single_transformer_blocks' in name and '.to_out.' in name:
            if x['weightShape']!=[3072,12288]:raise ValueError('Output projection geometry differs')
            groups['singleDown']+=mac*Fraction(3,4)
    if any(Fraction(x).denominator!=1 for x in groups.values()):raise ValueError('Fractional MAC allocation')
    total=r['work']['modelMatrixAndConvolutionMACs'];mlp=int(sum(groups.values()))
    result={'schema':'tce-image-moments-affine-allocation-v1','executionSHA256':sha(p/'execution.json'),'accountingSourceSHA256':sha(__file__),
        'MLPByPhysicalPartitionMACs':{k:int(v) for k,v in groups.items()},'MLPMACs':mlp,'wholePathMACs':total,'MLPShare':mlp/total,'affineMACs':e['affineMACs'],'affineShare':e['affineMACs']/total,
        'scope':'Exact MAC arithmetic from executed dimensions. Fused input partitions 6D gate/up of9D, terminal6D of7D, output3D MLP of4D. Two source-native single matrices; no measured runtime/energy split or new tensor/model computation.'}
    save(root/'docs/results/image-partition-moments-evidence/affine-allocation.json',result);return result


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True))
