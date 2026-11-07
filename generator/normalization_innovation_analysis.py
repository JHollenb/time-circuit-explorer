"""Post-seal two-receiver response, finite-defect and inherited-control account."""
import gc
import json
from pathlib import Path
from encoder_atoms import np, read, save, sha
from paired_fields import load, retain, metrics
from normalization_innovation import STAGES


def execute(root):
    from source_response import mask
    root = Path(root)
    out = root / 'response-result'
    seal_sha = sha(out / 'primary-seal.json')
    seal = read(out / 'primary-seal.json')
    if sha(out / 'primary.json') != seal['primarySHA256']:
        raise ValueError('Two-recipient candidate is not sealed')
    primary, targets = read(out / 'primary.json'), read(root / 'targets.json')
    selected = mask('bottom-right')
    scopes = [('all', np.ones(1024, bool)), ('source-nonlocal', selected)]
    accounts, operands, prefixes, fields = [], [], [], {}
    query_count = operand_count = 0
    with (out / 'queries.jsonl').open('x') as stream:
        for receiver in ['R0', 'R1']:
            for stage in STAGES:
                a = load(targets['cells'][receiver + 'A'][stage])
                b = load(targets['cells'][receiver + 'B'][stage])
                target = b - a
                predicted = load(primary['cells'][receiver][stage])
                response = predicted - a
                error = response - target
                arms = {'shared-normalization-innovation': response}
                if receiver == 'R1':
                    arms['raw-source-transfer'] = load(targets['cells']['R0B'][stage]) - load(targets['cells']['R0A'][stage])
                    arms['prior-conditioned-product'] = load(targets['priorConditioned'][stage]) - a
                fields[receiver + '.' + stage] = {name: retain(root / 'arrays/responses' / receiver / stage, name, value) for name, value in [('nativeResponse', target), ('candidateResponse', response), ('candidateError', error)]}
                def account(sel):
                    sl = lambda x: x[0, sel]
                    comparisons = {name: metrics(sl(value), sl(target)) for name, value in arms.items()}
                    comparisons['zero-response'] = metrics(np.zeros_like(sl(target)), sl(target))
                    return {'endpoint': metrics(sl(predicted), sl(b)), 'response': comparisons, 'signedErrorEnergy': {'candidateEnergy': float(np.sum(sl(response) ** 2)), 'targetEnergy': float(np.sum(sl(target) ** 2)), 'minusTwiceDot': float(-2 * np.sum(sl(response) * sl(target))), 'errorEnergy': float(np.sum(sl(error) ** 2))}}
                for scope, sel in scopes:
                    accounts.append({'receiver': receiver, 'stage': stage, 'scope': scope, 'queries': int(sel.sum()), **account(sel)})
                for index in range(1024):
                    stream.write(json.dumps({'receiver': receiver, 'stage': stage, 'imageQueryIndex': index, 'sourceNonlocal': bool(selected[index]), **account(index)}, allow_nan=False) + '\n')
                    query_count += 1
                del a, b, target, predicted, response, error, arms
                gc.collect()
    with (out / 'operand-queries.jsonl').open('x') as stream:
        for receiver in ['R0', 'R1']:
            cell = primary['cells'][receiver]
            quant, raw, a, b = [load(x) for x in [cell['fused'], cell['prequantFP64'], targets['cells'][receiver + 'A']['fused'], targets['cells'][receiver + 'B']['fused']]]
            rho = load(cell['rho'])
            for operand, lo, hi in [('gate', 0, 9216), ('up', 9216, 18432)]:
                qresponse, rresponse, target, conversion = [(q - z)[..., lo:hi] for q, z in [(quant, a), (raw, a), (b, a), (quant, raw)]]
                def account(sel):
                    sl = lambda x: x[0, sel]
                    return {'nativeOperandResponse': metrics(sl(qresponse), sl(target)), 'prequantOperandResponse': metrics(sl(rresponse), sl(target)), 'quantizationDifferenceRms': float(np.sqrt(np.mean(sl(conversion) ** 2))), 'residualCommitRhoRms': float(np.sqrt(np.mean(sl(rho) ** 2)))}
                for scope, sel in scopes:
                    operands.append({'receiver': receiver, 'operand': operand, 'scope': scope, 'queries': int(sel.sum()), **account(sel)})
                for index in range(1024):
                    stream.write(json.dumps({'receiver': receiver, 'operand': operand, 'imageQueryIndex': index, 'sourceNonlocal': bool(selected[index]), **account(index)}, allow_nan=False) + '\n')
                    operand_count += 1
            for suffix in ['A', 'B']:
                actual = targets['residuals'][receiver + suffix]
                predicted = cell['residual' + suffix]
                if actual['payloadSHA256'] != predicted['payloadSHA256']:
                    raise ValueError('Reconstructed native residual payload differs')
                prefixes.append({'receiver': receiver, 'endpoint': suffix, 'payloadSHA256': predicted['payloadSHA256'], 'exact': True})
            del quant, raw, a, b, rho, qresponse, rresponse, target, conversion
            gc.collect()
    if sha(out / 'primary-seal.json') != seal_sha or sha(out / 'primary.json') != seal['primarySHA256']:
        raise ValueError('Primary changed during scoring')
    save(out / 'response-fields.json', fields)
    return {'accounts': accounts, 'operandAccounts': operands, 'prefixes': prefixes, 'queryRecords': query_count, 'operandRecords': operand_count, 'primarySealSHA256': seal_sha, 'controlScope': 'R0 uses zero response; its own native response is not presented as an independent transfer forecast. R1 source-transfer and cycle49 conditioned controls are reused after seal, with prior learned work paid. No new control projections, reference model or fit.'}
