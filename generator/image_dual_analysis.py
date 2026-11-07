"""Post-seal response and explicit finite-policy telescope; no new consumer arm."""
import json
from pathlib import Path
import numpy as np
from encoder_atoms import read, sha
from paired_fields import load, retain, metrics

NAMES = ['H', 'A', 'D_attention', 'U', 'X', 'fused', 'product', 'M', 'D_mlp', 'T',
         'nextLayerNorm', 'nextX', 'nextQProjected', 'nextQNormalized', 'nextQ']


def effect(value, response):
    rms = float(np.sqrt(np.mean(value * value)))
    reference = float(np.sqrt(np.mean(response * response)))
    return {'rms': rms, 'relativeToActualResponse': rms / reference if reference else None,
            'signedMean': float(np.mean(value)), 'signedMeanProduct': float(np.mean(value * response)),
            'maxAbs': float(np.max(np.abs(value)))}


def execute(root):
    root = Path(root);out = root / 'response-result';request = read(root / 'request.json')
    for phase in ['compiler', 'actor', 'reference']:
        if read(out / (phase + '-seal.json'))[phase + 'SHA256'] != sha(out / (phase + '.json')):
            raise ValueError('Complete phase seal differs')
    actor, reference, compiler, targets = [read(x) for x in [out / 'actor.json', out / 'reference.json', out / 'compiler.json', root / 'targets.json']]
    if actor['compilerSealSHA256'] != sha(out / 'compiler-seal.json') or reference['actorSealSHA256'] != sha(out / 'actor-seal.json'):
        raise ValueError('Compiler→actor→post-seal reference order differs')
    candidate = {**actor['source']['fields'], **actor['next_query']['fields']}
    arrays = {role: {n: load(v[n]) for n in NAMES} for role, v in targets.items()}
    arrays['candidate'] = {n: load(candidate[n]) for n in NAMES}
    scores, observations = {}, []
    for field in NAMES:
        pp, qq, stale, primary = [arrays[n][field] for n in ['PP', 'QQ', 'stale', 'candidate']]
        actual = qq - pp
        scores[field] = {'primaryResponse': metrics(primary - pp, actual), 'staleResponse': metrics(stale - pp, actual),
                         'primaryEndpoint': metrics(primary, qq), 'zeroResponse': metrics(np.zeros_like(actual), actual)}
        for j, row in enumerate(request['support']):
            observations.append({'field': field, 'panel': 'whole-row', 'row': row,
                                 'primaryResponse': metrics((primary - pp)[0, j], actual[0, j]),
                                 'staleResponse': metrics((stale - pp)[0, j], actual[0, j])})
        if field in ['nextQProjected', 'nextQNormalized', 'nextQ']:
            for j, row in enumerate(request['support']):
                for h in range(24):
                    observations.append({'field': field, 'panel': 'query-head', 'row': row, 'head': h,
                        'primaryResponse': metrics((primary - pp)[0, j, h], actual[0, j, h]),
                        'staleResponse': metrics((stale - pp)[0, j, h], actual[0, j, h])})
    head = lambda value: value.reshape(1, 4, 24, 128)[0].transpose(1, 0, 2)
    raw = load(actor['read']['fields']['H_FP64']);admitted = head(arrays['candidate']['H'])
    smooth = load(reference['reference']['fields']['smoothH'])
    native_keys = load(reference['reference']['fields']['nativeKeysH'])
    target = head(arrays['QQ']['H']);response = target - head(arrays['PP']['H'])
    parts = {'admissionCast': admitted - raw, 'denominatorAndContraction': raw - smooth,
             'keyFinitePolicy': smooth - native_keys, 'attentionFinitePolicy': native_keys - target}
    retained = {n: retain(root / 'arrays/decomposition', n, v) for n, v in parts.items()}
    telescope = sum(parts.values())
    decomposition = {n: effect(v, response) for n, v in parts.items()}
    decomposition['total'] = effect(admitted - target, response)
    decomposition['telescopingRoundoffMaxAbs'] = float(np.max(np.abs(telescope - (admitted - target))))
    response_energy = float(np.mean(response * response))
    cross_energy = {a: {b: {'signedMeanProduct': float(np.mean(x * y)),
                           'relativeToResponseEnergy': float(np.mean(x * y)) / response_energy if response_energy else None}
                        for b, y in parts.items()} for a, x in parts.items()}
    for h in range(24):
        for j, row in enumerate(request['support']):
            observations.append({'field': 'H_policy', 'panel': 'query-head-policy', 'row': row, 'head': h,
                                 **{n: effect(v[h, j], response[h, j]) for n, v in parts.items()},
                                 'total': effect((admitted - target)[h, j], response[h, j])})
    rr, ar = reference['reference']['fields'], actor['read']['fields']
    numerator = metrics(load(ar['imageNumerator']), load(rr['directNumerator']))
    candidate_logits = load(ar['logits'])[:, :, 512:]
    smooth_logits, native_logits = load(rr['smoothImageLogits']), load(rr['nativeImageLogits'])
    log_partition = load(ar['peak']) + np.log(load(ar['partition']))
    diagnostics = {'numeratorFactorization': numerator,
        'retainedDenominatorAndContractionLogits': metrics(candidate_logits, smooth_logits),
        'combinedKeyFinitePolicyLogits': metrics(smooth_logits, native_logits),
        'candidateVsNativeKeyLogits': metrics(candidate_logits, native_logits),
        'denominatorChange': metrics(load(compiler['compile']['fields']['pDenominator']), load(rr['qDenominator'])),
        'candidateVsSmoothLogPartition': metrics(log_partition, load(rr['smoothLogPartition'])),
        'candidateVsNativeKeyLogPartition': metrics(log_partition, load(rr['nativeLogPartition']))}
    with (out / 'observations.jsonl').open('x') as stream:
        for item in observations:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + '\n')
    executions = {n: read(out / (n + '-execution.json')) for n in ['compiler', 'actor', 'reference']}
    return {'scores': scores, 'diagnostics': diagnostics, 'headErrorDecomposition': decomposition,
        'headErrorCrossEnergy': cross_energy,
        'retainedDecomposition': retained, 'observations': len(observations),
        'workByPhase': {n: v['nominalMatrixMACs'] for n, v in executions.items()},
        'totalNewMatrixMACs': sum(v['nominalMatrixMACs'] for v in executions.values()),
        'processBF16ParameterBytes': sum(x['bytes'] for v in executions.values() for x in v['loadedParameters']),
        'distinctBF16ParameterBytes': sum(x['bytes'] for x in {x['name']: x for v in executions.values() for x in v['loadedParameters']}.values()),
        'dualTableBytes': compiler['compile']['fields']['duals']['bytes'],
        'primaryCandidate': 'Query-specific image numerator with retained smooth P RMS denominator',
        'decompositionLaw': 'BF16admitted−archivedSDPA = admissionCast + denominatorAndContraction + keyFinitePolicy + attentionFinitePolicy',
        'scope': 'One paid native source pair/query family. Original maps, full supplied values/text keys, compiler and references paid; no extra MLP arm, full suffix, image, fit or arbitrary-prompt zero-neural completion.'}
