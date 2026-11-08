"""Frozen-transcript inference contract; no source graph or new oracle access."""
import argparse
from fractions import Fraction as Q
import hashlib
import itertools
import json
from pathlib import Path
import time


PINS = {
    'docs/results/behavioral-realization/discovery-queries.json': '7bd02ebe4b3e419fd144bc66c6aa99d8d86b166c5c77e81615da38410fef0d5f',
    'docs/results/behavioral-realization/shallow-queries.json': 'a685f5d9a12d4f1a8396d86d8f4334c71601078c2033d25ebc687081d99922c1',
    'docs/results/behavioral-realization/request.json': '734ec850110072feee8df02908cd253459be938e1980bede7b45a589667f21fe',
    'docs/results/behavioral-realization/serving.json': '019247a9505d8763130e2af83b90c6e47cd81f146a0d68eec6bd82f8d9f8f2b9',
    'docs/results/behavioral-realization/prediction-seal.json': '55beff15cc7cda3f4edb67ac1ac29db00df5983cdf98e34a271d834f4716662b',
    'docs/results/behavioral-realization/realization.json': 'a67bef8dd993328e1774041f9658296d5491a381ead39fe5d2a6bce36c6ba82d',
    'docs/plans/behavioral-certificate-authority.md': '75e86349af0b9ff0c925473d5747f765353ac25cb5e91c599b54db3a1bcada89',
    'generator/worker.py': 'f398769ef587fbadabe4feba20af2f1b75387eae209be3403033556a312d6d23',
    'generator/encoder_atoms.py': '6889cd42e46b9881a6568b8378edef0488f651d6867bdb1dcf7b0a79266c6922',
    'generator/fresh_square_encoder.py': '5a1104c76dcb597edd9a31540b86be3a0657c2b17ebd70d1fe33c2053f583265',
    'docs/results/raw-prompt-encoder-atoms/manifest.json': '9a155bd1410c52909997bacaf1d1aac2f9e90be86dc269d5e7edfab2b5285432',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inverse(matrix):
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise ValueError('A nonempty square separating panel is required')
    a = [[Q(x) for x in row] + [Q(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for k in range(n):
        pivot = next((i for i in range(k, n) if a[i][k]), None)
        if pivot is None:
            raise ValueError('Separating panel is singular')
        a[k], a[pivot] = a[pivot], a[k]
        scale = a[k][k]
        a[k] = [x / scale for x in a[k]]
        for i in range(n):
            if i != k:
                scale = a[i][k]
                a[i] = [x - scale * y for x, y in zip(a[i], a[k])]
    return [row[n:] for row in a]


def row_times(row, matrix, work):
    if len(row) != len(matrix) or not matrix or any(len(x) != len(matrix[0]) for x in matrix):
        raise ValueError('Matrix geometry differs')
    work['rationalLinearMACs'] += len(row) * len(matrix[0])
    return [sum((Q(x) * Q(matrix[i][j]) for i, x in enumerate(row)), Q(0)) for j in range(len(matrix[0]))]


def add_observation(observations, tokens, output):
    key = tuple(tokens)
    if any(x not in ('a', 'b') for x in key) or len(output) != 1:
        raise ValueError('Outside scalar a/b transcript contract')
    value = Q(output[0])
    if key in observations and observations[key] != value:
        raise ValueError('Conflicting recorded answer')
    observations[key] = value


def reconstruct(observations, prefixes, suffixes, alphabet, work):
    get = lambda p, s=(): observations[tuple(p) + tuple(s)]
    basis = [[get(p, s) for s in suffixes] for p in prefixes]
    inv = inverse(basis)
    operators = {}
    for symbol in alphabet:
        shifted = [[get(tuple(p) + (symbol,), s) for s in suffixes] for p in prefixes]
        operators[symbol] = [row_times(row, shifted, work) for row in inv]
    outputs = [[get(p)] for p in prefixes]
    return {'basis': basis, 'initial': [get((), s) for s in suffixes],
            'operators': operators, 'readout': [row_times(row, outputs, work) for row in inv]}


def evaluate(model, word, work):
    state = model['initial']
    for token in word:
        state = row_times(state, model['operators'][token], work)
    return row_times(state, model['readout'], work)[0]


def exception(word, witness):
    """Supplied singleton-word recognizer; not a queried source or image renderer."""
    return int(tuple(word) == tuple(witness))


def analyze(repo, output):
    tick = time.perf_counter(); repo = Path(repo).resolve(); data = {}; inputs = {}
    for name, expected in PINS.items():
        path = repo / name; raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise ValueError('Input identity differs: ' + name)
        inputs[name] = {'sha256': actual, 'bytes': len(raw)}
        if name.endswith('.json'):
            data[name] = json.loads(raw)
    base = 'docs/results/behavioral-realization/'
    obs = {}
    for name in ('discovery-queries.json', 'shallow-queries.json'):
        for row in data[base + name]['rows']:
            add_observation(obs, row['tokens'], row['output'])
    request = data[base + 'request.json']; serving = data[base + 'serving.json']; seal = data[base + 'prediction-seal.json']
    if seal['servingSHA256'] != inputs[base + 'serving.json']['sha256'] or seal['requestSHA256'] != inputs[base + 'request.json']['sha256']:
        raise ValueError('Recorded serving/request seal differs')
    cases = {x['id']: x['tokens'] for x in request['cases']}; refusals = []
    for row in serving['rows']:
        tokens = cases[row['id']]
        if row['status'] == 'executed':
            add_observation(obs, tokens, row['output'])
        elif row['status'] == 'outside-declared-alphabet' and any(x not in ('a', 'b') for x in tokens):
            refusals.append(row['id'])
        else:
            raise ValueError('Unexpected serving status')
    frozen = data[base + 'realization.json']
    if frozen['rank'] != 3 or any(x['channel'] != 0 for x in frozen['futureTests']):
        raise ValueError('Wrong supplied scalar realization')
    work = {'rationalLinearMACs': 0}
    model = reconstruct(obs, frozen['prefixBasis'], [x['suffix'] for x in frozen['futureTests']], frozen['alphabet'], work)
    for key in model:
        if json.dumps(model[key], default=str, sort_keys=True) != json.dumps(frozen[key], sort_keys=True):
            raise ValueError('Reconstructed coefficient differs: ' + key)
    comparisons = [{'tokens': list(w), 'recorded': str(y), 'reconstructed': str(evaluate(model, w, work))}
                   for w, y in sorted(obs.items(), key=lambda x: (len(x[0]), x[0]))]
    if any(x['recorded'] != x['reconstructed'] for x in comparisons):
        raise ValueError('Reconstruction does not reproduce transcript')
    length = max(map(len, obs)) + 1; witness = ('a',) * length
    f_witness = evaluate(model, witness, work)
    if witness in obs or f_witness != 0:
        raise ValueError('Singleton counterexample contract differs')
    longest_short = max(len(x['tokens']) for x in data[base + 'discovery-queries.json']['rows'])
    missing5 = [list(w) for w in itertools.product(('a', 'b'), repeat=5) if w not in obs]
    result = {
        'schema': 'tce-behavioral-authority-v1', 'cycle': 93, 'inputPins': inputs,
        'sourceGraphOrCertificateRead': False, 'newSourceOracleCalls': 0,
        'newNeuralModelEvaluations': 0, 'scientificArrayReads': 0, 'newImages': 0,
        'observedDistinctWords': len(obs), 'maximumObservedWordLength': max(map(len, obs)),
        'maximumDiscoveryQueryLength': longest_short, 'recordedRefusals': refusals,
        'comparisons': comparisons, 'reconstructed': model,
        'conditionalRealization': {
            'sampledRankLowerBound': 3, 'globalRankUpperBoundPremise': 3,
            'premiseEstablishedByTranscript': False,
            'status': 'unique-global-realization-conditional-on-independent-rank-upper-bound',
            'proof': 'Global rank at most3 plus nonsingular3minor makes selected full rows a basis. Future-test coordinates separate rows; shifted basis rows, initial tests and current readouts fix every transition and output by induction.',
        },
        'unrestrictedAlternative': {
            'definition': 'g(w)=f_reconstructed(w)+1[w=a^33]', 'witness': list(witness),
            'fAtWitness': f_witness, 'gAtWitness': f_witness + exception(witness, witness),
            'agreesWithEveryRecordedWord': all(exception(w, witness) == 0 for w in obs),
            'singletonDifferenceExactHankelRank': length + 1,
            'rankProof': 'Rows a^i/columns a^j for0<=i,j<=33 form a34x34 anti-diagonal identity. A34state weighted chain realizes the singleton, so lower and upper bounds agree.',
            'alternativeRankLowerBound': length + 1 - 3,
            'alternativeRankUpperBound': length + 1 + 3,
            'scope': 'Different finite-state source. Does not contradict the prior complete supplied-graph certificate. Not another queried organism.',
        },
        'genericDimensionBound': {'twoRanks': [3, 3], 'sufficientExhaustiveMaximumLength': 5,
                                  'allBinaryWordsThroughLength5': 63, 'missingLength5Words': missing5,
                                  'scope': 'Sufficient comparison theorem, not a necessary test panel. Basis-based conditional reconstruction needs fewer words.'},
        'nativeInterface': {
            'rawWrapper': 'Nonempty stripped text; incoming length<=1000 Python characters; seed uint32.',
            'nativeEncoderCall': 'Native pipe.encode_prompt requests max_sequence_length=512.',
            'lexicalFrontend': 'Separately qualified151936-ID token-local store, Jinja2 chat template, right padding/truncation512, masks/positions/RoPE; not complete contextual encoding.',
            'globalEconomicalRankBound': 'not-established', 'qualifiedGlobalEquivalenceOracle': 'not-established',
            'sourceDerivedReducedCommutingProof': 'not-established',
            'finiteDomain': 'Effective bounded finite-vocabulary sequences are finite under fixed framing/controls. Token truncation alone does not bound raw strings; wrapper character cap separately restricts raw input.',
            'sourceDerivedContextAndPixelProgram': 'unresolved',
        },
        'work': work, 'sourceSHA256': digest(__file__),
        'elapsedSecondsBeforeReportWrite': time.perf_counter() - tick,
        'mechanics_status': 'complete', 'instrument_status': 'exact frozen-transcript conditional-realization analysis',
        'trend_status': 'An independent global complexity bound is substantive extra authority; sampled rank alone cannot supply it.',
        'terminal_status': 'arbitrary-prompt zero-neural image generation unestablished',
        'explorationAllowedWithoutGlobalProof': True,
        'scope': 'Supplied toy/transcript and native source interfaces only. Rational toy coefficient evaluation is paid analysis, not a new permissible image generator. Exact elimination, JSON, source/hash reads and Python overhead additional to counted linear MACs.',
    }
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, default=str); stream.write('\n')
    return {k: result[k] for k in ('observedDistinctWords', 'work', 'elapsedSecondsBeforeReportWrite', 'mechanics_status')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('output'); args = parser.parse_args()
    print(json.dumps(analyze(Path(__file__).resolve().parents[1], args.output)))
