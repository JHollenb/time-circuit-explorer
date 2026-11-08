"""Post-exit82primary/83and73secondary, signed full/centered every-row accounts."""
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from encoder_atoms import read, save, sha
from paired_fields import load, retain
from gate_up_basis_analysis import statistics, panel, COLUMNS

PORTS = ['preResidualUpdate', 'exit', 'exitIncrement']
COMPARISONS = ['vs82', 'vs83', 'vs73', 'old83vs82', 'old83vs73', 'old82vs73']
ACCOUNTS = ['primary82from83', 'total73from83', 'total73from82']
EXTRAS = ['twiceOldDeltaInner', 'twiceCenteredOldDeltaInner', 'errorIdentityClosureEnergy', 'centeredIdentityClosureEnergy']
ROW_COLUMNS = [name+'_'+col for name in COMPARISONS for col in COLUMNS]+[name+'_'+col for name in ACCOUNTS for col in EXTRAS]


def analysis_abis(text=512, image=1024, rank=128):
    n = text+image
    return {'sketchResidual': {'shape': [1, n, rank], 'dtype': 'torch.float64'}, 'basisGram': {'shape': [2, rank, rank], 'dtype': 'torch.float64'},
        'sketchRowStatistics': {'shape': [n, 8], 'dtype': 'float64'}, 'rowStatistics': {'shape': [n, len(PORTS), len(ROW_COLUMNS)], 'dtype': 'float64'}}


def accounts(candidate, old82, old83, target73, text=512):
    def center(value):
        return np.concatenate([value[:, :text]-value[:, :text].mean(1, keepdims=True), value[:, text:]-value[:, text:].mean(1, keepdims=True)], 1)
    values = [statistics(a, b, text) for a, b in [(candidate, old82), (candidate, old83), (candidate, target73), (old83, old82), (old83, target73), (old82, target73)]]
    for old, target in [(old83, old82), (old83, target73), (old82, target73)]:
        delta = candidate-old; prior = old-target; total = candidate-target; closure = total-prior-delta
        cp, cd, ct = map(center, [prior, delta, total]); cc = ct-cp-cd
        values.append(np.stack([2*np.sum(prior[0]*delta[0], -1), 2*np.sum(cp[0]*cd[0], -1), np.sum(closure[0]**2, -1), np.sum(cc[0]**2, -1)], -1))
    return np.concatenate(values, -1)


def summarize(rows):
    result = {name: panel(rows[:, 8*i:8*i+8]) for i, name in enumerate(COMPARISONS)}; totals = rows.sum(0)
    result['signedAccounts'] = {}
    for j, (name, prior, delta, total) in enumerate(zip(ACCOUNTS, [3, 4, 5], [1, 1, 0], [0, 2, 2])):
        offset = 48+4*j
        result['signedAccounts'][name] = {'oldEnergy': float(totals[8*prior+2]), 'deltaEnergy': float(totals[8*delta+2]), 'totalEnergy': float(totals[8*total+2]),
            'twiceOldDeltaInner': float(totals[offset]), 'energyAccountRemainder': float(totals[8*total+2]-totals[8*prior+2]-totals[8*delta+2]-totals[offset]),
            'centeredOldEnergy': float(totals[8*prior+6]), 'centeredDeltaEnergy': float(totals[8*delta+6]), 'centeredTotalEnergy': float(totals[8*total+6]),
            'twiceCenteredOldDeltaInner': float(totals[offset+1]), 'centeredEnergyAccountRemainder': float(totals[8*total+6]-totals[8*prior+6]-totals[8*delta+6]-totals[offset+1]),
            'identityClosureEnergy': float(totals[offset+2]), 'centeredIdentityClosureEnergy': float(totals[offset+3])}
    result['rowsNewVsOld'] = {}
    for name, new_index, old_index in [('primary82vs83', 0, 3), ('total73vs83', 2, 4), ('total73vs82', 2, 5)]:
        a, b = rows[:, 8*new_index+2], rows[:, 8*old_index+2]
        result['rowsNewVsOld'][name] = {'improve': int(np.sum(a < b)), 'worsen': int(np.sum(a > b)), 'equal': int(np.sum(a == b))}
    return result


def execute(root):
    import torch
    from flux_clock_study import verify
    from complete_masked_study import profile
    from operand_row_output import work
    from operand_row_output_ledger import Ledger
    root = Path(root).resolve(); r = read(root/'request.json'); verify(root, r); out = root/'response-result'; s = read(root/'inputs.json'); started = time.perf_counter()
    c = read(out/'candidate.json'); seal = read(out/'candidate-seal.json'); sealed = sha(out/'candidate-seal.json'); exited = read(out/'producer-exit.json')
    if seal != {'requestSHA256': r['requestSHA256'], 'candidateSHA256': sha(out/'candidate.json'), 'executionSHA256': sha(out/'execution.json')} or exited['returncode'] != 0 or exited['candidateSealSHA256'] != sealed or exited['requestSHA256'] != r['requestSHA256'] or exited['producerPID'] == os.getpid():
        raise ValueError('Successful sealed producer exit missing')
    numeric = profile(torch, 4)
    if numeric != s['numericalProfile']:
        raise ValueError('Diagnostic numerical policy differs')
    torch.cuda.reset_peak_memory_stats(); ledger = Ledger(r['requestSHA256'], 'post-exit-diagnostic')
    f, u, rr = [load(c['fields'][n], 'cuda') for n in ['sketch', 'rowBasis', 'qrR']]; reconstructed = torch.empty_like(f); grams = []
    with torch.inference_mode():
        for stream, (lo, hi) in enumerate([(0, 512), (512, 1536)]):
            reconstructed[0, lo:hi] = ledger.matmul('sketchReconstruction'+str(stream), u[0, lo:hi], rr[stream])
            grams.append(ledger.matmul('basisGram'+str(stream), u[0, lo:hi].T, u[0, lo:hi]))
        gram = torch.stack(grams); residual = f-reconstructed
    sketch_stats = statistics(reconstructed.cpu().numpy(), f.cpu().numpy()); qr_diagnostics = []
    for stream in range(2):
        error = gram[stream].cpu().numpy()-np.eye(128); diagonal = np.diag(rr[stream].cpu().numpy())
        qr_diagnostics.append({'stream': 'text' if stream == 0 else 'image', 'orthogonalityMaxAbs': float(np.max(np.abs(error))), 'orthogonalityFrobenius': float(np.sqrt(np.sum(error*error))),
            'RDiagonal': diagonal.tolist(), 'exactZeroRDiagonalCount': int(np.sum(diagonal == 0)), 'scope': 'Finite QR diagnostics, not rank/output-sufficiency certification; all128columns retained.'})
    fields = {'sketchResidual': retain(root/'arrays/analysis', 'sketchResidual', residual), 'basisGram': retain(root/'arrays/analysis', 'basisGram', gram)}
    del f, u, rr, reconstructed, grams, gram, residual
    incoming = np.concatenate([load(s['sourceFields'][n]) for n in ['text', 'image']], 1); rows = np.empty((1536, len(PORTS), len(ROW_COLUMNS)), dtype=np.float64)
    for index, name in enumerate(['preResidualUpdate', 'exit']):
        candidate = load(c['fields'][name]); old82, old83, target73 = [load(s['comparisonFields'][tag][name]) for tag in ['82', '83', '73']]
        rows[:, index] = accounts(candidate, old82, old83, target73)
        if name == 'exit':
            rows[:, -1] = accounts(candidate-incoming, old82-incoming, old83-incoming, target73-incoming)
        del candidate, old82, old83, target73
    fields.update(sketchRowStatistics=retain(root/'arrays/analysis', 'sketchRowStatistics', sketch_stats), rowStatistics=retain(root/'arrays/analysis', 'rowStatistics', rows))
    if {n: {'shape': a['shape'], 'dtype': a['logicalDtype']} for n, a in fields.items()} != analysis_abis():
        raise ValueError('Diagnostic output ABI differs')
    panels = {name: {port: summarize(rows[lo:hi, j]) for j, port in enumerate(PORTS)} for name, lo, hi in [('all', 0, 1536), ('text', 0, 512), ('image', 512, 1536)]}
    with (out/'observations.jsonl').open('x') as stream:
        for i, row in enumerate(rows):
            stream.write(json.dumps({'row': i, 'stratum': 'text' if i < 512 else 'image', 'ports': {port: dict(zip(ROW_COLUMNS, map(float, row[j]))) for j, port in enumerate(PORTS)}, 'sketchReconstruction': dict(zip(COLUMNS, map(float, sketch_stats[i])))}, allow_nan=False)+'\n')
    if len(ledger.products) != 4 or sum(a['nominalMACs'] for a in ledger.products) != work()['diagnosticMACs'] or sha(out/'candidate-seal.json') != sealed or sha(out/'candidate.json') != seal['candidateSHA256'] or sha(out/'execution.json') != seal['executionSHA256']:
        raise ValueError('Post-exit work/seal closure differs')
    save(out/'analysis-fields.json', {'fields': fields, 'candidateSealSHA256': sealed, 'ports': PORTS, 'rowColumns': ROW_COLUMNS})
    save(out/'analysis-execution.json', {'products': ledger.products, 'diagnosticMatrixMACs': work()['diagnosticMACs'], 'numericalProfile': numeric, 'analyzerPID': os.getpid(), 'producerExitSHA256': sha(out/'producer-exit.json'), 'candidateSealSHA256': sealed, 'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'peakCUDAReservedBytes': torch.cuda.max_memory_reserved()})
    save(out/'report.json', {'schema': 'tce-current-operand-row-output-result-v1', 'requestSHA256': r['requestSHA256'], 'candidateSealSHA256': sealed, 'fixedAddress': {'clock': 0, 'singleBlock': 0}, 'panels': panels,
        'sketchReconstruction': {name: panel(sketch_stats[lo:hi]) for name, lo, hi in [('all', 0, 1536), ('text', 0, 512), ('image', 512, 1536)]}, 'basisDiagnostics': qr_diagnostics, 'work': work(), 'observations': 1536, 'analysisSecondsAfterImports': time.perf_counter()-started,
        'newImages': 0, 'newContinuations': 0, 'newFits': 0, 'mechanics_status': 'complete', 'instrument_status': 'current-read-product-source-basis-test', 'trend_status': '82primary/83and73secondary, signed fullcentered everyrow retained', 'terminal_status': 'unassessed',
        'scope': 'Own82product, source73read/gate/residual and originaloutputmap paid. New15.099BcandidateMAC;34.427Bcomposedprospective only. No new native reference/continuation/image, runtime/generalization or arbitrary-prompt zero-neural certification.'})
    for name in ['request.json', 'inputs.json', 'resource-basis.json', 'frozen-plan.md', 'implementation-notes.md']:
        shutil.copyfile(root/name, out/name)


if __name__ == '__main__':
    execute(sys.argv[1])
