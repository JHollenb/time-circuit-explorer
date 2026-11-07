"""Post-hoc shared-error diagnosis on saved keys; no original learned projection."""
import argparse
import json
from pathlib import Path
import shutil
import time
from encoder_atoms import read, save, sha, identity

HERE = Path(__file__).resolve().parent
OWNED = ['paired_sketch_diagnostic.py', 'encoder_atoms.py', 'paired_fields.py', 'masked_kv_study.py']


def stage(root):
    from mrun.client.api import Api
    from publish_source_native import checked
    main, collection, job, request = checked(HERE.parent / 'artifacts/coherent-norm-v1')
    parent, pc, pj, pr = checked(HERE.parent / 'artifacts/image-dual-v1')
    api = Api()
    for c, j in [(collection, job), (pc, pj)]:
        live = api.json('GET', '/api/jobs/' + j['job_id'])
        if live['state'] != 'succeeded' or live['payload_custody'] != c['custody']:
            raise ValueError('Saved-input custody differs')
    p = main / 'result';a, c, report = [read(p / n) for n in ['actor.json', 'compiler.json', 'report.json']]
    if sha(p / 'report.json') != 'cc3b102e8f506ec216cc41fb06a8ea0fefa09da802ccb3ea0515004392d526de':
        raise ValueError('Frozen fixed-sketch report differs')
    oldc = read(parent / 'result/compiler.json')['compile']['fields'];oldq = read(parent / 'result/reference.json')['reference']['fields']
    source = {'rawP': oldc['rawP'], 'rawQ': oldq['rawQ'], 'pNorm': oldc['pDenominator'], 'qNorm': oldq['qDenominator'],
        'signs': c['compile']['fields']['signs'], 'actualQProjection': a['norm']['fields']['projection'],
        'parentJobID': job['job_id'], 'parentCustody': collection['custody'], 'referenceJobID': pj['job_id'], 'referenceCustody': pc['custody'],
        'parentReportSHA256': sha(p / 'report.json'), 'sameInput': report['matchedControlPayloads']['actualModLNInput']}
    root = Path(root).resolve();root.mkdir(parents=True, exist_ok=False);payload = root / 'payload';payload.mkdir()
    for n in OWNED:
        if n in request['ownedSources'] and sha(HERE / n) != request['sourceHashes'][n]:
            raise ValueError('Frozen saved-input decoder differs')
        shutil.copyfile(HERE / n, payload / n)
    save(payload / 'inputs.json', source)
    r = {'schema': 'tce-paired-sketch-diagnostic-request-v1', 'ownedSources': OWNED,
        'sourceHashes': {x.name: sha(x) for x in payload.iterdir()}, 'backend': 'numpy', 'device': 'cpu', 'dtype': 'float64',
        'plannedMatrixMACs': 100663296, 'newOriginalProjections': 0,
        'protocol': 'Post-hoc saved rawP/rawQ and executedS only. DirectSrawP/SrawQ, finite reassociation versus actual60Qprojection, signed error cancellation/covariance, rawkey directions, ideal shared variance. One paid-baseline anchoredenergy E_P+(Ehat_Q-Ehat_P); nonpositive radicands explicit, no clamp/fallback/consumer. No fitting/rank/seed/gain search or primary feedback.'}
    r['requestSHA256'] = identity(r);save(root / 'request.json', r);save(payload / 'request.json', r)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [x.name for x in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': r['requestSHA256']}


def submit(root):
    from response_study import command
    from mrun.client.submit import launch
    from flux_clock_study import verify
    root = Path(root).resolve();r = read(root / 'request.json');verify(root / 'payload', r)
    if (root / 'job.json').exists():
        raise ValueError('Already submitted; inspect same handle')
    job = launch(command()[:-5] + ['python', '-u', 'paired_sketch_diagnostic.py', 'execute', '.'],
        experiment='tce-paired-sketch-diagnostic-v1', task='analysis', payload=root / 'payload', pin='beast', needs={'cpu': True},
        config={k: r[k] for k in ['schema', 'requestSHA256', 'backend', 'device', 'dtype']},
        timeout_s=180, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='Saved-only shared-sketch P/Q error: two signed projections100.663MMAC, frozenr16/signs. No original learned weight/map/model/consumer/image. CPU1NumPy, target-assisted descriptive baseline anchoring and covariance/variance/directions; invalidradicands retained,no clamp/fallback/refit/sweep. Intentfirst admission,finite180s,noretry;arraysnearcompute.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']};save(root / 'job.json', value);return value


def execute(root):
    import importlib.abc
    import sys
    class NoFrameworks(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'torch', 'diffusers', 'transformers', 'jax', 'tensorflow'}:
                raise RuntimeError('Saved-only diagnostic cannot import model frameworks')
    if any(n in sys.modules for n in ['torch', 'diffusers', 'transformers', 'jax', 'tensorflow']):
        raise ValueError('Model framework already imported')
    sys.meta_path.insert(0, NoFrameworks())
    import numpy as np
    from paired_fields import load, retain, metrics
    root = Path(root).resolve();r, s = read(root / 'request.json'), read(root / 'inputs.json')
    if identity({k: v for k, v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request differs')
    for n, h in r['sourceHashes'].items():
        if sha(root / n) != h:
            raise ValueError('Frozen diagnostic source differs')
    started = time.perf_counter();p, q, pn, qn, signs, actual = [load(s[n]) for n in ['rawP', 'rawQ', 'pNorm', 'qNorm', 'signs', 'actualQProjection']]
    if p.shape != (1024, 24, 128) or q.shape != p.shape or signs.shape != (24, 16, 128) or actual.shape != (1024, 24, 16) or pn.shape != (1024, 24) or qn.shape != pn.shape:
        raise ValueError('Saved diagnostic geometry differs')
    if not bool(np.all((signs == 1) | (signs == -1))):
        raise ValueError('Frozen signs invalid')
    pp, pq = [np.matmul(z.transpose(1, 0, 2), signs.transpose(0, 2, 1)).transpose(1, 0, 2) for z in [p, q]]
    ep, eq = [np.mean(z*z, -1) for z in [p, q]]
    hp, hq = [np.mean(z*z, -1) / 128 for z in [pp, pq]]
    error_p, error_q = hp - ep, hq - eq;delta_error = error_q - error_p
    response, predicted_response = eq - ep, hq - hp
    anchored = ep + predicted_response;radicand = anchored + 1e-6;valid = radicand > 0
    indices = np.flatnonzero(valid);norm = np.sqrt(radicand.ravel()[indices])
    np_true, nq_true = np.sqrt(ep + 1e-6), np.sqrt(eq + 1e-6)
    dp = q-p;lp, lq = np.sum(p*p, -1), np.sum(q*q, -1);dot = np.sum(p*q, -1)
    nonzero = (lp > 0) & (lq > 0);ci = np.flatnonzero(nonzero)
    cos = dot.ravel()[ci] / np.sqrt(lp.ravel()[ci]*lq.ravel()[ci])
    displacement = np.sum(dp*dp, -1);midpoint = (p+q)/2
    midpoint_energy = np.sum(midpoint*midpoint, -1);midpoint_delta = np.sum(midpoint*dp, -1)
    frobenius = 2*(midpoint_energy*displacement + midpoint_delta*midpoint_delta)
    direct_frobenius = lq*lq + lp*lp - 2*dot*dot
    diagonal = 4*np.sum((midpoint*dp)**2, -1)
    ideal_variance = 2*(frobenius-diagonal)/(16*128**2)
    out = root / 'response-result';out.mkdir()
    arrays = {n: retain(root / 'arrays', n, v) for n, v in [
        ('projectedP', pp), ('projectedQDirect', pq), ('errorP', error_p), ('errorQ', error_q), ('errorDifference', delta_error),
        ('anchoredEnergy', anchored), ('radicand', radicand), ('validNormIndices', indices), ('validAnchoredNorm', norm),
        ('cosineIndices', ci), ('rawKeyCosine', cos), ('idealSharedVariance', ideal_variance),
        ('varianceCancellationResidual', frobenius-direct_frobenius)]}
    def error_moments(a, b):
        ma, mb = float(a.mean()), float(b.mean());va, vb = float(np.mean((a-ma)**2)), float(np.mean((b-mb)**2))
        cross = float(np.mean(a*b));cov = float(np.mean((a-ma)*(b-mb)));d = b-a
        mpa, mpb, md = [float(np.mean(v*v)) for v in [a, b, d]]
        return {'pErrorMean': ma, 'qErrorMean': mb, 'pErrorMSE': mpa, 'qErrorMSE': mpb,
            'uncenteredErrorCrossMoment': cross, 'centeredErrorCovariance': cov,
            'centeredErrorCorrelation': cov/(va*vb)**.5 if va*vb else None, 'differenceErrorMSE': md,
            'MSEIdentityResidual': md-(mpa+mpb-2*cross)}
    def summarize(mask=None):
        take = (lambda v: v if mask is None else v[:, mask]);v = take(valid);ix = np.flatnonzero(v)
        candidate_norm = np.sqrt(take(radicand).ravel()[ix]);tp, tq = take(np_true).ravel()[ix], take(nq_true).ravel()[ix]
        return {'energyResponse': metrics(take(predicted_response), take(response)),
            'anchoredEnergyEndpoint': metrics(take(anchored), take(eq)), 'errorMoments': error_moments(take(error_p), take(error_q)),
            'validNormCells': int(v.sum()), 'invalidNormCells': int(v.size-v.sum()),
            'anchoredNormResponseValidOnly': metrics(candidate_norm-tp, tq-tp) if ix.size else None,
            'anchoredNormEndpointValidOnly': metrics(candidate_norm, tq) if ix.size else None,
            'holdNormResponseValidOnly': metrics(np.zeros_like(tp), tq-tp) if ix.size else None,
            'idealSharedVarianceMean': float(take(ideal_variance).mean()),
            'negativeIdealVarianceCells': int(np.sum(take(ideal_variance) < 0)),
            'rawKeyDisplacement': metrics(p if mask is None else p[:, mask], q if mask is None else q[:, mask])}
    rows = [{'head': h, **summarize(h)} for h in range(24)]
    with (out / 'observations.jsonl').open('x') as stream:
        for j in range(1024):
            for h in range(24):
                vn = float(np.sqrt(radicand[j, h])) if valid[j, h] else None
                change = float(nq_true[j, h]-np_true[j, h])
                predicted_change = vn-float(np_true[j, h]) if vn is not None else None
                row = {'panel': 'post-hoc-paired-key-head', 'row': j, 'head': h,
                    'pEnergy': float(ep[j, h]), 'qEnergy': float(eq[j, h]),
                    'pSketchEnergy': float(hp[j, h]), 'qSketchEnergy': float(hq[j, h]),
                    'pError': float(error_p[j, h]), 'qError': float(error_q[j, h]), 'differenceError': float(delta_error[j, h]),
                    'actualEnergyResponse': float(response[j, h]), 'predictedEnergyResponse': float(predicted_response[j, h]),
                    'anchoredEnergy': float(anchored[j, h]), 'radicand': float(radicand[j, h]), 'normValid': bool(valid[j, h]),
                    'actualNormChange': change, 'anchoredNormChange': predicted_change,
                    'signedNormResponseError': predicted_change-change if predicted_change is not None else None,
                    'rawKeyCosine': float(dot[j, h]/np.sqrt(lp[j, h]*lq[j, h])) if nonzero[j, h] else None,
                    'relativeRawKeyDisplacement': float(np.sqrt(displacement[j, h]/lp[j, h])) if lp[j, h] > 0 else None,
                    'idealSharedEnergyResponseVariance': float(ideal_variance[j, h]),
                    'varianceCancellationResidual': float(frobenius[j, h]-direct_frobenius[j, h])}
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
    for n in ['request.json', 'inputs.json']:
        shutil.copyfile(root / n, out / n)
    save(out / 'report.json', {'schema': 'tce-paired-sketch-diagnostic-v1', 'requestSHA256': r['requestSHA256'],
        'pooled': summarize(), 'heads': rows, 'retainedFields': arrays,
        'pDenominatorReconstruction': metrics(np_true, pn), 'qDenominatorReconstruction': metrics(nq_true, qn),
        'directQProjectionToActualActor': metrics(pq, actual),
        'directQComparisonApplicability': 'same-input' if s['sameInput'] else 'not-applicable-input-mismatch',
        'directQSketchEnergyToActual': metrics(hq, np.mean(actual*actual, -1)/128),
        'rawKeyCosine': {'defined': int(ci.size), 'undefined': int(lp.size-ci.size), 'min': float(cos.min()) if ci.size else None,
            'median': float(np.median(cos)) if ci.size else None, 'max': float(cos.max()) if ci.size else None},
        'maxAbsVarianceCancellationResidual': float(np.abs(frobenius-direct_frobenius).max()),
        'zeroEnergyResponseCells': int(np.sum(response == 0)), 'newMatrixMACs': 100663296, 'newOriginalProjections': 0,
        'observations': 24576,
        'workerWallSeconds': time.perf_counter()-started, 'mechanics_status': 'complete', 'terminal_status': 'unassessed',
        'scope': 'Post-hoc saved native raw keys and fixed executed signs. Two unlearned signed projections paid. One hypothetical anchored norm requires paid P energy; valid subsets explicit, no clamp/fallback/consumer/model/refit/primary feedback. No inexpensive baseline formation or own-state recurrence claim.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']);p.add_argument('run');a = p.parse_args()
    if a.action == 'collect':
        from response_study import collect
        value = collect(a.run)
    else:
        value = globals()[a.action](a.run)
    if value is not None:
        print(json.dumps(value))
