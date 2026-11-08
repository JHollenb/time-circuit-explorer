"""Exact behavioral realization mechanics for a supplied finite machine.

Word queries discover a rational representation. A separate complete graph
certificate tests closure. This does not learn or evaluate a neural image model.
"""
from collections import deque
from copy import deepcopy
from fractions import Fraction as Q
import hashlib
from itertools import product
import json


ARITHMETIC = {'exactLinearMACs': 0}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def words(alphabet, depth):
    return [tuple(w) for n in range(depth + 1) for w in product(alphabet, repeat=n)]


def matrix(value):
    return [[Q(x) for x in row] for row in value]


def strings(value):
    return [[str(x) for x in row] for row in value]


def row_times(row, right, width):
    ARITHMETIC['exactLinearMACs'] += len(row) * width
    return [sum((row[i] * right[i][j] for i in range(len(row))), Q(0)) for j in range(width)]


def multiply(left, right, width):
    return [row_times(row, right, width) for row in left]


def unit(size):
    return [[Q(int(i == j)) for j in range(size)] for i in range(size)]


def pivots(value):
    reduced = matrix(value)
    if not reduced:
        return []
    result = []
    row = 0
    for column in range(len(reduced[0])):
        pivot = next((i for i in range(row, len(reduced)) if reduced[i][column]), None)
        if pivot is None:
            continue
        reduced[row], reduced[pivot] = reduced[pivot], reduced[row]
        scale = reduced[row][column]
        reduced[row] = [x / scale for x in reduced[row]]
        for i in range(len(reduced)):
            if i != row:
                scale = reduced[i][column]
                reduced[i] = [x - scale * y for x, y in zip(reduced[i], reduced[row])]
        result.append(column)
        row += 1
        if row == len(reduced):
            break
    return result


def inverse(value):
    size = len(value)
    augmented = [list(map(Q, row)) + eye for row, eye in zip(value, unit(size))]
    for column in range(size):
        pivot = next((i for i in range(column, size) if augmented[i][column]), None)
        if pivot is None:
            raise ValueError('Singular exact basis')
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [x / scale for x in augmented[column]]
        for i in range(size):
            if i != column:
                scale = augmented[i][column]
                augmented[i] = [x - scale * y for x, y in zip(augmented[i], augmented[column])]
    return [row[size:] for row in augmented]


class FiniteMachine:
    def __init__(self, data):
        self.data = deepcopy(data)
        self.alphabet = tuple(self.data['alphabet'])
        self.initial = self.data['initial']
        self.states = self.data['states']
        self.output_calls = 0
        self.symbol_steps = 0
        if not self.alphabet or len(set(self.alphabet)) != len(self.alphabet) or self.initial not in self.states:
            raise ValueError('Invalid finite source domain')
        widths = {len(row['output']) for row in self.states.values()}
        if len(widths) != 1 or not next(iter(widths)):
            raise ValueError('Uniform nonempty output required')
        self.width = next(iter(widths))
        for row in self.states.values():
            if set(row['next']) != set(self.alphabet) or any(s not in self.states for s in row['next'].values()):
                raise ValueError('Source graph must be total on the declared alphabet')
            if any(type(x) is not int or not 0 <= x <= 255 for x in row['output']):
                raise ValueError('Byte-valued source outputs required')

    def after(self, word, start=None):
        state = self.initial if start is None else start
        for symbol in word:
            if symbol not in self.alphabet:
                raise ValueError('Symbol outside declared finite language')
            state = self.states[state]['next'][symbol]
            self.symbol_steps += 1
        return state

    def output(self, word, start=None):
        self.output_calls += 1
        return tuple(self.states[self.after(word, start)]['output'])

    def reachable(self):
        paths = {self.initial: ()}
        queue = deque([self.initial])
        while queue:
            state = queue.popleft()
            for symbol in self.alphabet:
                target = self.states[state]['next'][symbol]
                if target not in paths:
                    paths[target] = paths[state] + (symbol,)
                    queue.append(target)
        return paths


class WordOracle:
    """Compiler-visible interface contains no source states or transition table."""
    def __init__(self, callback):
        self.callback = callback
        self.cache = {}
        self.requests = 0

    def __call__(self, word):
        word = tuple(word)
        self.requests += 1
        if word not in self.cache:
            self.cache[word] = tuple(Q(x) for x in self.callback(word))
        return self.cache[word]

    def accounting(self):
        return dict(requests=self.requests, physicalQueries=len(self.cache),
                    cacheHits=self.requests - len(self.cache),
                    physicalSourceTransitions=sum(map(len, self.cache)),
                    maximumPhysicalQueryLength=max(map(len, self.cache), default=0))


def compile_panel(oracle, alphabet, prefixes, tests):
    panel = [[oracle(tuple(p) + tuple(s))[j] for s, j in tests] for p in prefixes]
    selected_rows = pivots(list(map(list, zip(*panel))))
    basis_rows = [panel[i] for i in selected_rows]
    selected_columns = pivots(basis_rows)
    rank = len(selected_rows)
    prefix_basis = [tuple(prefixes[i]) for i in selected_rows]
    future_tests = [(tuple(tests[j][0]), tests[j][1]) for j in selected_columns]
    basis = [[oracle(p + s)[j] for s, j in future_tests] for p in prefix_basis]
    inv = inverse(basis)
    output_width = len(oracle(()))
    shifted = {a: [[oracle(p + (a,) + s)[j] for s, j in future_tests] for p in prefix_basis] for a in alphabet}
    operators = {a: strings(multiply(inv, shifted[a], rank)) for a in alphabet}
    outputs = [list(oracle(p)) for p in prefix_basis]
    return {'schema': 'tce-exact-behavioral-realization-v1', 'alphabet': list(alphabet),
            'rank': rank, 'outputWidth': output_width, 'panelShape': [len(prefixes), len(tests)],
            'prefixBasis': [list(p) for p in prefix_basis],
            'futureTests': [{'suffix': list(s), 'channel': j} for s, j in future_tests],
            'basis': strings(basis), 'coordinateTransform': strings(unit(rank)),
            'initial': [str(oracle(s)[j]) for s, j in future_tests],
            'operators': operators, 'readout': strings(multiply(inv, outputs, output_width)),
            'scope': 'Finite-panel exact-rank hypothesis; no global closure follows from compilation alone.'}


def evaluate(realization, word):
    state = [Q(x) for x in realization['initial']]
    rank = realization['rank']
    for symbol in word:
        state = row_times(state, matrix(realization['operators'][symbol]), rank)
    return row_times(state, matrix(realization['readout']), realization['outputWidth'])


def transformed(realization, change):
    result = deepcopy(realization)
    change = matrix(change)
    inv = inverse(change)
    rank = realization['rank']
    result['initial'] = [str(x) for x in row_times(list(map(Q, realization['initial'])), change, rank)]
    result['operators'] = {a: strings(multiply(multiply(inv, matrix(m), rank), change, rank))
                           for a, m in realization['operators'].items()}
    result['readout'] = strings(multiply(inv, matrix(realization['readout']), realization['outputWidth']))
    result['coordinateTransform'] = strings(multiply(matrix(realization['coordinateTransform']), change, rank))
    result['scope'] += ' One declared exact coordinate change; no gauge search.'
    return result


def distinguishing_suffix(machine, left, right):
    queue = deque([(left, right, ())])
    seen = set()
    while queue:
        a, b, suffix = queue.popleft()
        if (a, b) in seen:
            continue
        seen.add((a, b))
        if machine.states[a]['output'] != machine.states[b]['output']:
            return {'suffix': list(suffix), 'leftOutput': machine.states[a]['output'],
                    'rightOutput': machine.states[b]['output'], 'visitedPairs': len(seen)}
        for symbol in machine.alphabet:
            queue.append((machine.states[a]['next'][symbol], machine.states[b]['next'][symbol], suffix + (symbol,)))
    return None


def certify_finite(machine, realization):
    paths = machine.reachable()
    tests = realization['futureTests']
    rank = realization['rank']
    def code(state):
        raw = [Q(machine.output(t['suffix'], state)[t['channel']]) for t in tests]
        return row_times(raw, matrix(realization['coordinateTransform']), rank)
    codes = {s: code(s) for s in paths}
    transitions, outputs, collisions = [], [], []
    for state in paths:
        actual = row_times(codes[state], matrix(realization['readout']), machine.width)
        outputs.append({'state': state, 'predicted': list(map(str, actual)),
                        'target': machine.states[state]['output'],
                        'equal': actual == list(map(Q, machine.states[state]['output']))})
        for symbol in machine.alphabet:
            target = codes[machine.states[state]['next'][symbol]]
            actual = row_times(codes[state], matrix(realization['operators'][symbol]), rank)
            transitions.append({'state': state, 'symbol': symbol, 'predicted': list(map(str, actual)),
                                'target': list(map(str, target)), 'equal': actual == target})
    for index, left in enumerate(paths):
        for right in list(paths)[index + 1:]:
            if codes[left] == codes[right]:
                witness = distinguishing_suffix(machine, left, right)
                if witness is not None:
                    collisions.append({'states': [left, right], 'prefixes': [list(paths[left]), list(paths[right])],
                                       'sharedCode': list(map(str, codes[left])), **witness})
    initial_equal = list(map(Q, realization['initial'])) == codes[machine.initial]
    counterexamples = [row for row in transitions + outputs if not row['equal']]
    return {'schema': 'tce-complete-finite-closure-v1', 'sourceID': identity(machine.data),
            'realizationID': identity(realization), 'reachableStates': list(paths),
            'sourceCodes': {s: list(map(str, c)) for s, c in codes.items()},
            'initialEqual': initial_equal, 'transitions': transitions, 'outputs': outputs,
            'mergedFutureWitnesses': collisions, 'counterexamples': counterexamples,
            'closureStatus': 'proved-on-supplied-finite-domain' if initial_equal and not counterexamples else 'counterexample',
            'authority': 'Exact initial/output/all reachable state-symbol commuting equalities; induction over every finite word.',
            'scope': 'Complete only for the supplied finite source graph; no neural or image-model closure certificate.'}


def lower_copy_program(realization, certificate):
    if certificate['realizationID'] != identity(realization):
        raise ValueError('Closure certificate belongs to another realization')
    if certificate['closureStatus'] != 'proved-on-supplied-finite-domain':
        return {'status': 'declined-unclosed-realization'}
    rank = realization['rank']
    codes = [list(map(Q, row)) for row in certificate['sourceCodes'].values()]
    constants = {j: codes[0][j] for j in range(rank) if len({row[j] for row in codes}) == 1}
    live = [j for j in range(rank) if j not in constants]
    if any(x.denominator != 1 or not 0 <= x <= 255 for row in codes for x in row):
        return {'status': 'not-lowerable-by-byte-copy-constant-in-these-coordinates'}
    def operation(column):
        nonzero = [(i, value) for i, value in enumerate(column) if value]
        if not nonzero:
            return {'op': 'constant', 'value': 0}
        if len(nonzero) != 1 or nonzero[0][1] != 1:
            return None
        source = nonzero[0][0]
        return ({'op': 'constant', 'value': int(constants[source])} if source in constants
                else {'op': 'copy', 'register': live.index(source)})
    transitions = {}
    for symbol, value in realization['operators'].items():
        m = matrix(value)
        # Constant-coordinate invariance is explicitly checked before elimination.
        for j, constant in constants.items():
            if any(row_times(row, m, rank)[j] != constant for row in codes):
                raise ValueError('Claimed constant coordinate is not invariant')
        transitions[symbol] = [operation([m[i][j] for i in range(rank)]) for j in live]
    output = matrix(realization['readout'])
    readout = [operation([output[i][j] for i in range(rank)]) for j in range(realization['outputWidth'])]
    if any(op is None for rows in transitions.values() for op in rows) or any(op is None for op in readout):
        return {'status': 'not-lowerable-by-copy-constant-in-these-coordinates',
                'scope': 'Instruction-set/coordinate limitation; closure remains proved and another program may exist.'}
    return {'status': 'lowered', 'schema': 'tce-byte-copy-program-v1',
            'sourceID': certificate['sourceID'], 'realizationID': certificate['realizationID'],
            'alphabet': realization['alphabet'], 'persistentRegisters': len(live),
            'initial': [int(Q(realization['initial'][i])) for i in live],
            'eliminatedConstants': {str(j): int(value) for j, value in constants.items()},
            'transitions': transitions, 'readout': readout, 'updatePolicy': 'all reads from the complete old state',
            'scope': 'Supplied finite-machine mechanics only; scalar output, no text interpretation or image renderer.'}
