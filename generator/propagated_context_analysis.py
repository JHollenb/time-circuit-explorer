"""Saved-state context effects and signed write composition after both seals."""
import json
from pathlib import Path
import numpy as np
from encoder_atoms import read, sha
from paired_fields import load, metrics


def execute(root):
    root = Path(root)
    out = root / 'response-result'
    r = read(root / 'request.json')
    phases = {n: read(out / (n + '.json')) for n in ['banks', 'primary', 'control']}
    for n in phases:
        seal = read(out / (n + '-seal.json'))
        if seal[n + 'SHA256'] != sha(out / (n + '.json')) or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Complete phase seal/request join differs')
    if phases['control']['primarySealSHA256'] != sha(out / 'primary-seal.json'):
        raise ValueError('Primary-before-control release differs')
    names = ['H', 'A', 'D_attention', 'U', 'X', 'fused', 'product', 'M', 'D_mlp', 'T',
             'nextLayerNorm', 'nextX', 'nextQProjected', 'nextQNormalized', 'nextQ']
    values = {}
    for arm in ['primary', 'control']:
        records = {**phases[arm]['source']['fields'], **phases[arm]['nextQuery']['fields']}
        values[arm] = {n: load(records[n]) for n in names}
    observations = []
    effects = {}
    for n in names:
        a, b = values['primary'][n], values['control'][n]
        effects[n] = metrics(a, b)
        for row, address in enumerate(r['support']):
            observations.append({'field': n, 'row': address, 'panel': 'whole-row', 'difference': metrics(a[0, row], b[0, row])})
        if n in ['nextQProjected', 'nextQNormalized', 'nextQ']:
            for row, address in enumerate(r['support']):
                for head in range(a.shape[2]):
                    observations.append({'field': n, 'row': address, 'head': head, 'panel': 'query-head',
                                         'difference': metrics(a[0, row, head], b[0, row, head])})
    delta = {n: values['primary'][n] - values['control'][n] for n in names}
    rounding = delta['T'] - delta['D_attention'] - delta['D_mlp']
    terms = {'attentionWrite': delta['D_attention'], 'MLPWrite': delta['D_mlp'], 'residualCommitRounding': rounding}
    energy = float(np.mean(delta['T'] ** 2))
    composition = {'law': 'deltaT=deltaD_attention+deltaD_mlp+residualCommitRounding',
                   'targetRMS': float(np.sqrt(energy)), 'terms': {}, 'pairwiseTwiceMeanProducts': {}}
    for n, x in terms.items():
        composition['terms'][n] = {'rms': float(np.sqrt(np.mean(x*x))),
            'relativeToDeltaTRMS': float(np.sqrt(np.mean(x*x)/energy)) if energy else None,
            'meanDotWithDeltaT': float(np.mean(x*delta['T'])),
            'signedEnergyContribution': float(np.mean(x*delta['T'])/energy) if energy else None}
    keys = list(terms)
    for i, n in enumerate(keys):
        for m in keys[i+1:]:
            composition['pairwiseTwiceMeanProducts'][n + ':' + m] = float(2*np.mean(terms[n]*terms[m]))
    composition['identityMaxAbs'] = float(np.max(np.abs(delta['T'] - sum(terms.values()))))
    composition['scope'] = 'Exact signed accounting of captured BF16 writes and realized residual commits; not semantic causal percentages or an MLP surrogate.'
    with (out / 'observations.jsonl').open('x') as stream:
        for row in observations:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    executions = {n: read(out / (n + '-execution.json')) for n in phases}
    total = sum(x['nominalMatrixMACs'] for x in executions.values())
    if total != 58925776896:
        raise ValueError('Complete executed matrix sum differs')
    parameters = {x['name']: x['bytes'] for e in executions.values() for x in e['loadedParameters']}
    return {'effects': effects, 'signedWriteComposition': composition, 'observations': len(observations),
            'bankDifferences': phases['banks']['bankDifferences'], 'outsideSupportKVChanged': phases['banks']['outsideSupportKVChanged'],
            'totalNewMatrixMACs': total, 'distinctBF16ParameterBytes': sum(parameters.values()),
            'processBF16ParameterBytes': sum(x['bytes'] for e in executions.values() for x in e['loadedParameters']),
            'affineCalls': sum(len(x['linears']) for x in executions.values()), 'attentionCalls': sum(len(x['attentions']) for x in executions.values()),
            'contextEffectScope': 'Primary omits changed P complement, control retains it; all selected P operands are shared. Differences normalized to each full-P control endpoint. No separate full-width native target is run.'}
