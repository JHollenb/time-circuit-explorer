"""Post-seal fixed-sketch diagnostics; no new full-key projection or target repair."""
import json
from pathlib import Path
import numpy as np
from encoder_atoms import read, sha
from paired_fields import load, retain, metrics
from image_dual_analysis import NAMES


def execute(root):
    root = Path(root);out = root / 'response-result';r, s, targets = [read(root / n) for n in ['request.json', 'inputs.json', 'targets.json']]
    actor, compiler = read(out / 'actor.json'), read(out / 'compiler.json')
    for name in ['compiler', 'actor']:
        if read(out / (name + '-seal.json'))[name + 'SHA256'] != sha(out / (name + '.json')):
            raise ValueError('Complete phase seal differs')
    if actor['compilerSealSHA256'] != sha(out / 'compiler-seal.json'):
        raise ValueError('Compiler→actor release differs')
    ar = actor['read']['fields'];old = s['matchedControlFields']
    matched = {n: ar[n]['payloadSHA256'] == old[n]['payloadSHA256'] for n in ['imageNumerator', 'textLogits', 'textK_hybrid', 'V_hybrid']}
    matched['actualModLNInput'] = actor['modln']['fields']['x']['payloadSHA256'] == s['controlModLN']['payloadSHA256']
    candidate = {**actor['source']['fields'], **actor['next_query']['fields']}
    arrays = {n: {f: load(x[f]) for f in NAMES} for n, x in targets['trajectories'].items()}
    arrays['primary'] = {f: load(candidate[f]) for f in NAMES}
    scores, observations = {}, []
    for field in NAMES:
        pp, qq = arrays['PP'][field], arrays['QQ'][field];response = qq - pp
        scores[field] = {n + 'Response': metrics(arrays[n][field] - pp, response) for n in ['primary', 'retained', 'stale', 'diagonal']}
        scores[field].update(primaryEndpoint=metrics(arrays['primary'][field], qq), zeroResponse=metrics(np.zeros_like(response), response))
        for j, row in enumerate(r['support']):
            observations.append({'field': field, 'panel': 'whole-row', 'row': row,
                **{n + 'Response': metrics((arrays[n][field] - pp)[0, j], response[0, j]) for n in ['primary', 'retained', 'stale', 'diagonal']}})
        if field in ['nextQProjected', 'nextQNormalized', 'nextQ']:
            for j, row in enumerate(r['support']):
                for h in range(24):
                    observations.append({'field': field, 'panel': 'query-head', 'row': row, 'head': h,
                        **{n + 'Response': metrics((arrays[n][field] - pp)[0, j, h], response[0, j, h]) for n in ['primary', 'retained', 'stale', 'diagonal']}})
    qnorm, pnorm = [load(targets['norms'][n]) for n in ['Q', 'P']]
    actual_energy = qnorm * qnorm - 1e-6
    energy, norm = [load(actor['norm']['fields'][n]) for n in ['energy', 'norm']]
    difference = actual_energy - energy
    ratio = norm / qnorm
    difference_record = retain(root / 'arrays/diagnostics', 'savedQEnergyMinusSketchEnergy', difference)
    norm_rows = []
    for j in range(1024):
        for h in range(24):
            norm_rows.append({'field': 'key_norm', 'panel': 'image-key-head', 'row': j, 'head': h,
                'referenceEnergy': float(actual_energy[j, h]), 'sketchEnergy': float(energy[j, h]),
                'referenceMinusSketchEnergy': float(difference[j, h]),
                'normRatio': float(ratio[j, h]),
                'signedRelativeNormError': float(ratio[j, h] - 1)})
    observations += norm_rows
    logits = load(ar['logits'])[:, :, 512:]
    smooth_logits = load(targets['smoothImageLogits']);logz = load(ar['peak']) + np.log(load(ar['partition']))
    diagnostics = {'sketchNormToQ': metrics(norm, qnorm), 'retainedPNormToQ': metrics(pnorm, qnorm),
        'sketchNormResponse': metrics(norm - pnorm, qnorm - pnorm), 'imageLogitsToSmoothQ': metrics(logits, smooth_logits),
        'logPartitionToSmoothQ': metrics(logz, load(targets['smoothLogPartition'])),
        'referenceMinusSketchEnergy': {'rms': float(np.sqrt(np.mean(difference * difference))), 'signedMean': float(np.mean(difference)),
            'positiveCells': int(np.sum(difference > 0)), 'negativeCells': int(np.sum(difference < 0)),
            'zeroCells': int(np.sum(difference == 0)), 'min': float(np.min(difference)), 'max': float(np.max(difference))},
        'normRatio': {'min': float(np.min(ratio)), 'median': float(np.median(ratio)), 'max': float(np.max(ratio))},
        'energyToQ': metrics(energy, actual_energy),
        'pooledPredictedEnergyRatio': float(np.mean(energy) / np.mean(actual_energy)),
        'normOnlyContrastApplicability': {'status': 'matched-five-controls' if all(matched.values()) else 'not-applicable-control-mismatch',
            'scope': 'All actual input/numerator/text logits/K/V payloads must match for norm-only attribution. Signed energy error is sketch distortion, not an omitted-channel contribution. Adverse and mismatched results remain evidence.'}}
    with (out / 'observations.jsonl').open('x') as stream:
        for row in observations:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    executions = {n: read(out / (n + '-execution.json')) for n in ['compiler', 'actor']}
    return {'scores': scores, 'diagnostics': diagnostics, 'matchedControlPayloads': matched,
        'retainedEnergyDifference': difference_record, 'observations': len(observations),
        'workByPhase': {n: v['nominalMatrixMACs'] for n, v in executions.items()},
        'totalNewMatrixMACs': sum(v['nominalMatrixMACs'] for v in executions.values()),
        'nonmatrixOperationCounts': {n: v['elementwise'] for n, v in executions.items()},
        'distinctBF16ParameterBytes': sum(x['bytes'] for v in executions.values() for x in v['loadedParameters']),
        'processBF16ParameterBytes': sum(x['bytes'] for v in executions.values() for x in v['loadedParameters']),
        'coefficientNPYBytes': compiler['compile']['fields']['coefficients']['bytes'],
        'signNPYBytes': compiler['compile']['fields']['signs']['bytes'],
        'sketch': {'rank': 16, 'seed': 20261107, 'recipe': 'Independent head PCG64/SeedSequence([seed,h])/random_raw lowbit/Corder; no rerolls or target selection',
            'scope': 'One fixed sign sketch. Ideal unbiased energy expectation does not imply accurate or unbiased norms/consumers; possible blind directions retained.'},
        'primaryCandidate': 'Source-only fixed signed output image-key norm; same sealed query-side numerator and own finite consumer',
        'scope': 'One fixed source/query pair. No P/Qnorm input to actor, fresh key projection, fit, extra consumer arm, suffix or image. Native source states, dual table, text keys/values, original receiving maps and archived references remain paid.'}
