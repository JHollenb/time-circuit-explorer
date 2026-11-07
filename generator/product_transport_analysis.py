"""Post-seal native endpoints, signed response controls and complete query rows."""
import gc
import json
from pathlib import Path
from encoder_atoms import np, read, save, sha
from paired_fields import load, retain, metrics
from product_transport import STAGES


def execute(root):
    from source_response import mask
    root = Path(root)
    out = root / 'response-result'
    seal_sha = sha(out / 'primary-seal.json')
    seal = read(out / 'primary-seal.json')
    if seal['primarySHA256'] != sha(out / 'primary.json'):
        raise ValueError('Complete primary is not sealed')
    primary = read(out / 'primary.json')
    targets = read(root / 'targets.json')
    nonlocal_rows = mask('bottom-right')
    scopes = [('all', np.ones(1024, bool)), ('source-nonlocal', nonlocal_rows)]
    accounts = []
    operands = []
    fields = {}
    query_count = operand_count = 0
    with (out / 'queries.jsonl').open('x') as stream:
        for stage in STAGES:
            native = {cell: load(values[stage]) for cell, values in targets['cells'].items()}
            a, b = native['R1A'], native['R1B']
            target = b - a
            source = native['R0B'] - native['R0A']
            predicted = load(primary['fields'][stage])
            response = predicted - a
            error = response - target
            fields[stage] = {name: retain(root / 'arrays/responses' / stage, name, value) for name, value in [('nativeResponse', target), ('candidateResponse', response), ('candidateError', error), ('sourceTransferError', source - target)]}
            def account(sel):
                sl = lambda x: x[0, sel]
                arms = {'recipient-baseline-product': sl(response), 'raw-source-transfer': sl(source), 'zero-response': np.zeros_like(sl(target))}
                return {'endpoint': metrics(sl(predicted), sl(b)), 'response': {name: metrics(value, sl(target)) for name, value in arms.items()},
                    'signedErrorEnergy': {'candidateEnergy': float(np.sum(sl(response) ** 2)), 'targetEnergy': float(np.sum(sl(target) ** 2)), 'minusTwiceDot': float(-2 * np.sum(sl(response) * sl(target))), 'errorEnergy': float(np.sum(sl(error) ** 2))}}
            for scope, selected in scopes:
                accounts.append({'stage': stage, 'scope': scope, 'queries': int(selected.sum()), **account(selected)})
            for index in range(1024):
                stream.write(json.dumps({'stage': stage, 'imageQueryIndex': index, 'sourceNonlocal': bool(nonlocal_rows[index]), **account(index)}, allow_nan=False) + '\n')
                query_count += 1
            del native, a, b, target, source, predicted, response, error
            gc.collect()
    quant = load(primary['fields']['fused'])
    raw = load(primary['fields']['prequantFP64'])
    baseline = load(targets['fusedR1A'])
    actual = load(targets['fusedR1B'])
    with (out / 'operand-queries.jsonl').open('x') as stream:
        for name, start, stop in [('gate', 0, 9216), ('up', 9216, 18432)]:
            q, r, a, b = [x[..., start:stop] for x in [quant, raw, baseline, actual]]
            quant_response, raw_response, target_response, conversion = q - a, r - a, b - a, q - r
            def account(sel):
                sl = lambda x: x[0, sel]
                return {'nativeOperandResponse': metrics(sl(quant_response), sl(target_response)), 'prequantOperandResponse': metrics(sl(raw_response), sl(target_response)), 'quantizationDifferenceRms': float(np.sqrt(np.mean(sl(conversion) ** 2)))}
            for scope, selected in scopes:
                operands.append({'operand': name, 'scope': scope, 'queries': int(selected.sum()), **account(selected)})
            for index in range(1024):
                stream.write(json.dumps({'operand': name, 'imageQueryIndex': index, 'sourceNonlocal': bool(nonlocal_rows[index]), **account(index)}, allow_nan=False) + '\n')
                operand_count += 1
    residual = load(primary['fields']['residualCommit'])
    expected = load(targets['residualR1B'])
    if not np.array_equal(residual, expected):
        raise ValueError('Reconstructed B residual differs from frozen native prefix')
    if sha(out / 'primary-seal.json') != seal_sha or sha(out / 'primary.json') != seal['primarySHA256']:
        raise ValueError('Primary changed during scoring')
    save(out / 'response-fields.json', fields)
    return {'accounts': accounts, 'operandAccounts': operands, 'queryRecords': query_count, 'operandRecords': operand_count, 'nativeResidualPrefixExact': True, 'primarySealSHA256': seal_sha,
        'controlScope': 'Fieldwise retained native R0 write-response transfer and zero response, scored after seal. Controls are not newly executed coherent rollouts. Native R1A is the common response baseline. No effect-size gate.'}
