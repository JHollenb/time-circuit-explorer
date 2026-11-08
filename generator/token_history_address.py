"""Integer token-history address programs; no learned vector arithmetic.

This predicts our approximate reader's support port, not native model attention.
The module deliberately has no numerical-array, model or tokenizer dependency.
"""
from collections import Counter

HEADS = 32
WIDTH = 4
METHODS = ('primary', 'positionOnly', 'earliest', 'recent', 'headRotated', 'tokenNull')
REASONS = ('direct', 'unavailable', 'duplicate', 'exhausted')


def validate_packet(packet):
    if set(packet) != {'ids', 'mask', 'positions'}:
        raise ValueError('Only raw token IDs, masks and positions are admitted')
    n = len(packet['ids'])
    if any(len(packet[k]) != n for k in packet) or packet['positions'] != list(range(n)):
        raise ValueError('Sequential token clock required')
    if any(type(x) is not int or x < 0 for x in packet['ids']):
        raise ValueError('Nonnegative integer tokenizer IDs required')
    if any(type(x) is not int or x not in (0, 1) for x in packet['mask']):
        raise ValueError('Binary integer mask required')


def inventory(literals):
    generic = [(kind, j) for kind in ('first', 'recent') for j in range(WIDTH)]
    generic += [(kind, 0) for kind in ('same_first', 'same_last', 'same_previous', 'different_first', 'different_last')]
    return generic + [(kind, token) for token in sorted(set(literals)) for kind in ('literal_first', 'literal_last')]


class History:
    def __init__(self):
        self.positions = []
        self.tokens = []
        self.occurrences = {}
        self.steps = 0
        self.work = Counter()

    def admit(self, token, mask, position):
        if position != self.steps:
            raise ValueError('History must advance its own consecutive clock')
        if mask:
            self.positions.append(position)
            self.tokens.append(token)
            self.occurrences.setdefault(token, []).append(position)
            self.work['tokenAdmissions'] += 1
        self.steps += 1
        self.work['steps'] += 1

    def value(self, expr, query, position):
        kind, operand = expr
        self.work['selectorEvaluations'] += 1
        if kind in ('first', 'recent'):
            self.work['positionIndexReads'] += 1
            if len(self.positions) <= operand:
                return -1
            return self.positions[operand if kind == 'first' else -1-operand]
        if kind.startswith('same_') or kind.startswith('literal_'):
            self.work['occurrenceDictionaryLookups'] += 1
            hits = self.occurrences.get(query if kind.startswith('same_') else operand, ())
            if not hits:
                return -1
            if kind == 'same_previous':
                return hits[-2] if hits[-1] == position and len(hits) > 1 else (-1 if hits[-1] == position else hits[-1])
            return hits[0] if kind.endswith('first') else hits[-1]
        if kind in ('different_first', 'different_last'):
            pairs = zip(self.positions, self.tokens)
            if kind == 'different_last':
                pairs = zip(reversed(self.positions), reversed(self.tokens))
            for p, token in pairs:
                self.work['explicitTokenEqualityTests'] += 1
                if token != query:
                    return p
            return -1
        raise ValueError('Unknown selector')


def resolve(raw, prefix, used, work=None):
    if len(used) >= len(prefix):
        return -1, 3
    if raw != -1 and raw not in prefix:
        raise ValueError('Selector returned a noncausal/unadmitted address')
    reason = 1 if raw == -1 else (2 if raw in used else 0)
    if reason:
        if work is not None:
            work['fallbackEvents'] += 1
        for p in reversed(prefix):
            if work is not None:
                work['fallbackPositionProbes'] += 1
            if p not in used:
                return p, reason
        raise RuntimeError('Unreachable fallback exhaustion')
    return raw, reason


def fit(packets, labels):
    """Greedy fit on active discovery queries with prefix count > four."""
    if len(packets) != len(labels):
        raise ValueError('Discovery case counts differ')
    literals = sorted({x for packet in packets for x, active in zip(packet['ids'], packet['mask']) if active})
    expressions = inventory(literals)
    examples = []
    preparation = Counter()
    for packet, target in zip(packets, labels):
        validate_packet(packet)
        if len(target) != len(packet['ids']):
            raise ValueError('Discovery support length differs')
        state = History()
        for token, active, position, row in zip(packet['ids'], packet['mask'], packet['positions'], target):
            state.admit(token, active, position)
            if len(row) != HEADS or any(len(a) != WIDTH for a in row):
                raise ValueError('Discovery support geometry differs')
            if active and len(state.positions) > WIDTH:
                values = [state.value(expr, token, position) for expr in expressions]
                examples.append((values, state.positions.copy(), row))
        preparation.update(state.work)
    if not examples:
        raise ValueError('No nontrivial active discovery queries')
    work = Counter()

    def train(indices):
        rules = []
        objectives = []
        for head in range(HEADS):
            selected = [[] for _ in examples]
            rule = []
            for slot in range(WIDTH):
                best = None
                for index in indices:
                    matches = members = 0
                    for used, (values, prefix, target) in zip(selected, examples):
                        address, _ = resolve(values[index], prefix, used, work)
                        matches += address == target[head][slot]
                        members += address in target[head]
                        work['candidateResolvedRows'] += 1
                    objective = (matches, members)
                    if best is None or objective > best[0]:
                        best = objective, index
                objective, index = best
                rule.append(list(expressions[index]))
                objectives.append({'head': head, 'slot': slot, 'orderedMatches': objective[0], 'supportMembers': objective[1], 'examples': len(examples)})
                for used, (values, prefix, _) in zip(selected, examples):
                    address, _ = resolve(values[index], prefix, used)
                    used.append(address)
            rules.append(rule)
        return rules, objectives

    primary, primary_objectives = train(range(len(expressions)))
    positional, position_objectives = train(range(8))
    program = {'schema': 'tce-token-history-address-program-v1', 'heads': HEADS, 'width': WIDTH,
        'primary': primary, 'positionOnly': positional,
        'policy': 'greedy-ordered-membership-inventory-ties; recent-unused fallback; h+1 rotation; unique-negative-position token null'}
    receipt = {'discoveryActiveTokenIDs': literals, 'candidateInventory': [list(a) for a in expressions],
        'nontrivialDiscoveryQueriesPerHead': len(examples), 'preparationWork': dict(preparation), 'fitWork': dict(work),
        'objectives': {'primary': primary_objectives, 'positionOnly': position_objectives}}
    return program, receipt


def validate_program(program):
    if program.get('schema') != 'tce-token-history-address-program-v1' or program.get('heads') != HEADS or program.get('width') != WIDTH:
        raise ValueError('Program schema differs')
    generic = set(inventory([]))
    for family in ('primary', 'positionOnly'):
        rows = program[family]
        if len(rows) != HEADS or any(len(row) != WIDTH for row in rows):
            raise ValueError('Program geometry differs')
        for row in rows:
            for expr in row:
                if len(expr) != 2 or type(expr[1]) is not int:
                    raise ValueError('Malformed selector')
                key = tuple(expr)
                if key not in generic and not (family == 'primary' and expr[0] in ('literal_first', 'literal_last') and expr[1] >= 0):
                    raise ValueError('Program expression outside declared language')
                if family == 'positionOnly' and expr[0] not in ('first', 'recent'):
                    raise ValueError('Position-only program contains token operation')


def run_packet(packet, program, method):
    validate_packet(packet)
    validate_program(program)
    if method not in METHODS:
        raise ValueError('Unknown contrast')
    rules = program['positionOnly' if method == 'positionOnly' else 'primary']
    if method in ('earliest', 'recent'):
        rules = [[[('first' if method == 'earliest' else 'recent'), j] for j in range(WIDTH)] for _ in range(HEADS)]
    if method == 'headRotated':
        rules = rules[1:] + rules[:1]
    unique = list(dict.fromkeys(tuple(expr) for rule in rules for expr in rule))
    state = History()
    rows = []
    for input_token, active, position in zip(packet['ids'], packet['mask'], packet['positions']):
        token = -position-1 if method == 'tokenNull' else input_token
        state.admit(token, active, position)
        values = {expr: state.value(expr, token, position) for expr in unique}
        predictions = []
        raws = []
        reasons = []
        for rule in rules:
            chosen = []
            raw_row = []
            why_row = []
            for expr in rule:
                raw = values[tuple(expr)]
                address, why = resolve(raw, state.positions, chosen, state.work)
                raw_row.append(raw)
                why_row.append(why)
                if address != -1:
                    chosen.append(address)
                state.work['addressSlotsEmitted'] += 1
            predictions.append(chosen + [-1]*(WIDTH-len(chosen)))
            raws.append(raw_row)
            reasons.append(why_row)
            state.work['queryHeads'] += 1
        rows.append({'position': position, 'prefixCount': len(state.positions), 'addresses': predictions, 'rawSelectors': raws, 'reasons': reasons})
    return rows, {'work': dict(state.work), 'distinctUsedSelectors': len(unique), 'finalAdmittedTokens': len(state.positions),
        'finalDistinctTokenIDs': len(state.occurrences), 'tokenHistoryLogicalInt64Bytes': 8*(2*len(state.positions)+sum(len(v) for v in state.occurrences.values())),
        'memoryScope': 'Position/token/occurrence list payload only; dictionary keys, clock, program, Python objects and allocator overhead additional.'}


def metrics(predicted, support, weights):
    a = {x for x in predicted if x >= 0}
    b = {x for x in support if x >= 0}
    overlap = len(a & b)
    return {'ordered': int(predicted == support), 'set': int(a == b), 'top1': int(predicted[0] == support[0]),
        'overlap': overlap/len(b) if b else None,
        'oldTop4WeightCoverage': sum(w for p, w in zip(support, weights) if p >= 0 and p in a) if b else None}
