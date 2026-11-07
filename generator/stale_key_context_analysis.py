"""Actual context-response scoring and four-corner finite interactions."""
import json
from pathlib import Path
import numpy as np
from encoder_atoms import read, sha
from paired_fields import load, retain, metrics

NAMES = ['H', 'A', 'D_attention', 'U', 'X', 'fused', 'product', 'M', 'D_mlp', 'T',
         'nextLayerNorm', 'nextX', 'nextQProjected', 'nextQNormalized', 'nextQ']


def effect(x, reference):
    m = metrics(x, reference)
    return {'rms': m['candidateRms'], 'referenceRms': m['referenceRms'],
            'relativeToActualResponse': m['candidateRms']/m['referenceRms'] if m['referenceRms'] else None,
            'cosineWithActualResponse': m['cosine'], 'signedMeanProduct': float(np.mean(x*reference)),
            'maxAbs': float(np.max(np.abs(x)))}


def execute(root):
    root = Path(root);out = root / 'response-result';r = read(root / 'request.json')
    arms = {}
    for phase in ['primary', 'diagnostic']:
        seal = read(out / (phase + '-seal.json'));arms[phase] = read(out / (phase + '.json'))
        if seal[phase+'SHA256'] != sha(out / (phase + '.json')) or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Complete new phase seal differs')
    if arms['diagnostic']['primarySealSHA256'] != sha(out / 'primary-seal.json'):
        raise ValueError('Primary-before-diagnostic release differs')
    targets = read(root / 'targets.json')
    corners = {n: {field: load(x[field]) for field in NAMES} for n, x in targets['diagonals'].items()}
    for phase, name in [('primary', 'PQ'), ('diagnostic', 'QP')]:
        records = {**arms[phase]['source']['fields'], **arms[phase]['nextQuery']['fields']}
        corners[name] = {field: load(records[field]) for field in NAMES}
    scores, interactions, observations = {}, {}, []
    for field in NAMES:
        pp, qq, pq, qp = [corners[n][field] for n in ['PP', 'QQ', 'PQ', 'QP']]
        actual, candidate, diagnostic = qq-pp, pq-pp, qp-pp
        interaction = qq-qp-pq+pp
        interactions[field] = retain(root / 'arrays/interactions', field, interaction)
        scores[field] = {'primaryResponse': metrics(candidate, actual), 'diagnosticResponse': metrics(diagnostic, actual),
                         'primaryEndpoint': metrics(pq, qq), 'diagnosticEndpoint': metrics(qp, qq),
                         'zeroResponse': metrics(np.zeros_like(actual), actual), 'interaction': effect(interaction, actual)}
        def row(address, selection, panel, head=None):
            a, b, c, d = [x[selection] for x in [actual, candidate, diagnostic, interaction]]
            return {'field': field, 'row': address, 'panel': panel, **({'head': head} if head is not None else {}),
                    'primaryResponse': metrics(b, a), 'diagnosticResponse': metrics(c, a), 'interaction': effect(d, a)}
        for j, address in enumerate(r['support']):
            observations.append(row(address, (0, j), 'whole-row'))
        if field in ['nextQProjected', 'nextQNormalized', 'nextQ']:
            for j, address in enumerate(r['support']):
                for h in range(24):
                    observations.append(row(address, (0, j, h), 'query-head', h))
    with (out / 'observations.jsonl').open('x') as stream:
        for item in observations:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False)+'\n')
    executions = [read(out / (n + '-execution.json')) for n in ['primary', 'diagnostic']]
    total = sum(x['nominalMatrixMACs'] for x in executions)
    if total != 905969664:
        raise ValueError('Complete new physical operation sum differs')
    return {'scores': scores, 'interactions': interactions, 'observations': len(observations),
            'totalNewMatrixMACs': total, 'affineCalls': sum(len(x['linears']) for x in executions),
            'attentionCalls': sum(len(x['attentions']) for x in executions),
            'distinctBF16ParameterBytes': sum(x['bytes'] for x in executions[0]['loadedParameters']),
            'processBF16ParameterBytes': sum(x['bytes'] for e in executions for x in e['loadedParameters']),
            'primaryCandidate': 'F_PQ retained-P-complement-keys/new-Q-complement-values',
            'responseReference': 'F_QQ-F_PP from paid immutable parent continuations; selectedP operands fixed',
            'interactionLaw': 'F_QQ-F_QP-F_PQ+F_PP; later interactions also include receiving nonlinearities',
            'conditionalFutureQKeyFormationOpportunityMACs': 14495514624,
            'scope': 'Two new fixed-corner actors; retained banks, parent diagonals and original maps paid. Candidate scored on actual context response; no new source bank, full native reference, full suffix or image.'}
