"""Saved-only stage responses, finite discrepancies and dependent query rows."""
import gc
import json
from pathlib import Path
from encoder_atoms import np, read, save, sha
from paired_fields import load, retain, metrics
from value_innovation import STAGES


def execute(root):
    from source_response import mask
    root = Path(root)
    out = root / 'response-result'
    seal = sha(out / 'primary-seal.json')
    a, reference, targets, request = [read(p) for p in [out / 'primary.json', out / 'reference.json', root / 'targets.json', root / 'request.json']]
    if sha(out / 'primary.json') != read(out / 'primary-seal.json')['primarySHA256'] or sha(out / 'reference.json') != read(out / 'reference-seal.json')['referenceSHA256']:
        raise ValueError('Candidate/native release is not sealed')
    indices = np.arange(1024)
    rows, columns = indices // 32, indices % 32
    near = np.zeros(1024, bool)
    for address in request['event']['imageRows']:
        near |= (np.abs(rows - address // 32) <= 2) & (np.abs(columns - address % 32) <= 2)
    nonlocal_mask = mask('bottom-right')
    scopes = [('all', np.ones(1024, bool)), ('source-nonlocal', nonlocal_mask), ('event-near', near), ('event-far', ~near)]
    accounts, operands, fields = [], [], {}
    count = operand_count = 0
    with (out / 'queries.jsonl').open('x') as stream:
        for stage in STAGES:
            native_a, native_b = [load(reference['fields'][key][stage]) for key in ['A', 'B']]
            baseline = load(targets['A'][stage])
            candidate = load(a['fields'][stage])
            target, predicted = native_b - native_a, candidate - baseline
            error = predicted - target
            fields[stage] = {n: retain(root / 'arrays/responses' / stage, n, x) for n, x in [('nativeResponse', target), ('candidateResponse', predicted), ('responseError', error)]}
            def account(sel):
                sl = lambda x: x[0, sel]
                return {'endpoint': metrics(sl(candidate), sl(native_b)), 'response': metrics(sl(predicted), sl(target)), 'zeroResponse': metrics(np.zeros_like(sl(target)), sl(target)),
                        'baselineDrift': metrics(sl(baseline), sl(native_a)), 'signedErrorEnergy': {'candidateEnergy': float(np.sum(sl(predicted) ** 2)), 'nativeEnergy': float(np.sum(sl(target) ** 2)),
                        'minusTwiceDot': float(-2 * np.sum(sl(predicted) * sl(target))), 'errorEnergy': float(np.sum(sl(error) ** 2))}}
            for scope, sel in scopes:
                accounts.append({'stage': stage, 'scope': scope, 'queries': int(sel.sum()), **account(sel)})
            for index in range(1024):
                stream.write(json.dumps({'stage': stage, 'imageQueryIndex': index, 'sourceNonlocal': bool(nonlocal_mask[index]), 'eventNear': bool(near[index]), **account(index)}, allow_nan=False) + '\n')
                count += 1
            del native_a, native_b, baseline, candidate, target, predicted, error
            gc.collect()
    with (out / 'operand-queries.jsonl').open('x') as stream:
        native_a, native_b = [load(reference['fields'][key]['fused']) for key in ['A', 'B']]
        baseline = load(targets['A']['fused'])
        quant, raw = load(a['fields']['fused']), load(a['fields']['prequantFP64'])
        rho = load(a['fields']['rho'])
        for operand, lo, hi in [('gate', 0, 9216), ('up', 9216, 18432)]:
            target = (native_b - native_a)[..., lo:hi]
            predicted, raw_response = (quant - baseline)[..., lo:hi], (raw - baseline)[..., lo:hi]
            def account(sel):
                sl = lambda x: x[0, sel]
                return {'response': metrics(sl(predicted), sl(target)), 'prequantResponse': metrics(sl(raw_response), sl(target)), 'rhoRms': float(np.sqrt(np.mean(sl(rho) ** 2)))}
            for scope, sel in scopes:
                operands.append({'operand': operand, 'scope': scope, 'queries': int(sel.sum()), **account(sel)})
            for index in range(1024):
                stream.write(json.dumps({'operand': operand, 'imageQueryIndex': index, 'sourceNonlocal': bool(nonlocal_mask[index]), 'eventNear': bool(near[index]), **account(index)}, allow_nan=False) + '\n')
                operand_count += 1
    if sha(out / 'primary-seal.json') != seal:
        raise ValueError('Candidate seal changed during saved scoring')
    save(out / 'response-fields.json', fields)
    return {'accounts': accounts, 'operandAccounts': operands, 'queryRecords': count, 'operandRecords': operand_count, 'primarySealSHA256': seal, 'referenceSealSHA256': sha(out / 'reference-seal.json'),
            'inheritedA': reference['inheritedA'], 'sham': reference['sham'], 'sourceNonlocalMaskScope': 'Inherited cycle48 bottom-right exclusion; unrelated to event placement.', 'eventNearQueries': int(near.sum()),
            'scope': 'One addressed intervention. Candidate response uses its archived native A baseline; native response uses newly qualified/reference A. Any baseline drift remains explicit. Queries are dependent; all signed and adverse outcomes retained.'}
