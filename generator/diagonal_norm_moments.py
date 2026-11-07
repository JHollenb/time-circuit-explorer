"""Post-hoc saved-only spatial moments; paid diagonal, no original projections."""
import argparse
import json
from pathlib import Path
import shutil
import time
from encoder_atoms import read, save, sha, identity

HERE = Path(__file__).resolve().parent
OWNED = ['diagonal_norm_moments.py', 'encoder_atoms.py', 'paired_fields.py']


def stage(root):
    from mrun.client.api import Api
    from publish_source_native import checked
    parent, collection, job, previous = checked(HERE.parent / 'artifacts/diagonal-norm-v1')
    older, old_collection, old_job, old_request = checked(HERE.parent / 'artifacts/image-dual-v1')
    api = Api()
    for c, j in [(collection, job), (old_collection, old_job)]:
        live = api.json('GET', '/api/jobs/' + j['job_id'])
        if live['state'] != 'succeeded' or live['payload_custody'] != c['custody']:
            raise ValueError('Saved-only parent custody differs')
    p = parent / 'result';a, c, report = [read(p / n) for n in ['actor.json', 'compiler.json', 'report.json']]
    reference = read(older / 'result/reference.json')
    if sha(p / 'report.json') != 'b57f08719a7145091807bc3a59567a6045d22a66cb1e1058fd394e643e3f4275' or report['matchedControlPayloads']['actualModLNInput'] is not True:
        raise ValueError('Required same-input post-hoc diagnostic is inapplicable')
    s = {'x': a['modln']['fields']['x'], 'diagonal': c['compile']['fields']['diagonal'],
         'rawQ': reference['reference']['fields']['rawQ'], 'actorEnergy': a['norm']['fields']['energy'],
         'parentReportSHA256': sha(p / 'report.json'), 'parentJobID': job['job_id'], 'parentCustody': collection['custody'],
         'referenceJobID': old_job['job_id'], 'referenceCustody': old_collection['custody']}
    root = Path(root).resolve();root.mkdir(parents=True, exist_ok=False);payload = root / 'payload';payload.mkdir()
    for n in OWNED:
        if n in previous['ownedSources'] and sha(HERE / n) != previous['sourceHashes'][n]:
            raise ValueError('Frozen data decoder changed')
        shutil.copyfile(HERE / n, payload / n)
    save(payload / 'inputs.json', s)
    r = {'schema': 'tce-diagonal-norm-moments-request-v1', 'sourceHashes': {x.name: sha(x) for x in payload.iterdir()},
        'ownedSources': OWNED, 'backend': 'numpy', 'device': 'cpu', 'dtype': 'float64',
        'protocol': 'Post-hoc descriptive energy split from saved actualx,compiledD,oldrawQ; no original weight maps, fitting or new model/consumer. Paid compiledD contractions; mean/cross/centered spatial terms and target-assisted mean-direction capacity, all24heads.',
        'plannedMatrixMACs': 151142400, 'newOriginalMapProjections': 0, 'parentJobID': job['job_id']}
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
    cmd = command()[:-5] + ['python', '-u', 'diagonal_norm_moments.py', 'execute', '.']
    job = launch(cmd, experiment='tce-diagonal-norm-moments-v1', task='analysis', payload=root / 'payload', pin='beast',
        needs={'cpu': True}, config={k: r[k] for k in ['schema', 'requestSHA256', 'backend', 'device', 'dtype']},
        timeout_s=180, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='Posthoc saved norm moments: ownx/compiledD/oldrawQ/actorEnergy sameinput. CPU1NumPy, no frameworks/originalweights/newprojection/consumer. PaidDcontractions151.142MMAC; signedmean/cross/centered terms24heads/allrows, target-assistedmean-directioncapacity, no fit/feedback. Intentfirst admission, finite180s/noretry; arraysstaynearcompute.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']};save(root / 'job.json', value);return value


def execute(root):
    import importlib.abc
    import sys
    class NoFrameworks(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'torch', 'diffusers', 'transformers', 'jax', 'tensorflow'}:
                raise RuntimeError('Saved-only numerical diagnostic cannot import model frameworks')
    if any(n in sys.modules for n in ['torch', 'diffusers', 'transformers', 'jax', 'tensorflow']):
        raise ValueError('Saved-only process already imported a model framework')
    sys.meta_path.insert(0, NoFrameworks())
    import numpy as np
    from paired_fields import load, retain
    root = Path(root).resolve();r, s = read(root / 'request.json'), read(root / 'inputs.json')
    if identity({k: v for k, v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request differs')
    for n, h in r['sourceHashes'].items():
        if sha(root / n) != h:
            raise ValueError('Frozen saved-only source differs')
    started = time.perf_counter();x, d, z, actor_energy = [load(s[n]) for n in ['x', 'diagonal', 'rawQ', 'actorEnergy']]
    x = x[0]
    if x.shape != (1024, 3072) or d.shape != (24, 3072) or z.shape != (1024, 24, 128) or actor_energy.shape != (1024, 24):
        raise ValueError('Bound saved-only geometry differs')
    mx, mz = x.mean(0), z.mean(0)
    dx, dz = x - mx, z - mz
    vx, vz = np.mean(dx * dx, axis=0), np.mean(dz * dz, axis=0)
    true_mean, true_variation = np.mean(mz * mz, -1), np.mean(vz, -1)
    diag_mean, diag_variation = d @ (mx * mx), d @ vx
    mean_term, variation_term = true_mean - diag_mean, true_variation - diag_variation
    cross = 2 * (np.mean(dz * mz[None], axis=-1) - (dx * mx) @ d.T)
    centered_defect = np.mean(dz * dz, axis=-1) - (dx * dx) @ d.T
    direct_cells = np.mean(z * z, axis=-1) - actor_energy
    cell_residual = mean_term[None] + cross + centered_defect - direct_cells
    mean_length = np.sqrt(np.sum(mz * mz, axis=-1))
    direction = np.zeros_like(mz)
    for h in range(24):
        if mean_length[h] != 0:
            direction[h] = mz[h] / mean_length[h]
    projection = np.sum(z * direction[None], axis=-1)
    orthogonal_energy = np.mean((z - projection[..., None] * direction[None]) ** 2, axis=-1)
    direct = np.mean(z * z, axis=(0, 2)) - actor_energy.mean(0)
    residual = mean_term + variation_term - direct
    out = root / 'response-result';out.mkdir()
    arrays = {n: retain(root / 'arrays', n, v) for n, v in [('meanInput', mx), ('meanRawKey', mz), ('inputVariance', vx), ('rawKeyVariance', vz),
        ('rowCrossEnergy', cross), ('rowCenteredDefect', centered_defect), ('rowOrthogonalEnergy', orthogonal_energy), ('targetMeanDirection', direction)]}
    rows = [{'head': h, 'trueMeanEnergy': float(true_mean[h]), 'trueVariationEnergy': float(true_variation[h]),
        'diagonalMeanEnergy': float(diag_mean[h]), 'diagonalVariationEnergy': float(diag_variation[h]),
        'meanOmittedEnergy': float(mean_term[h]), 'variationOmittedEnergy': float(variation_term[h]),
        'totalOmittedEnergy': float(direct[h]), 'momentIdentityResidual': float(residual[h]),
        'rowCrossRMS': float(np.sqrt(np.mean(cross[:, h] ** 2))), 'rowCrossSignedMean': float(cross[:, h].mean()),
        'rowCenteredDefectRMS': float(np.sqrt(np.mean(centered_defect[:, h] ** 2))),
        'meanDirectionDefined': bool(mean_length[h] != 0),
        'targetMeanOrthogonalEnergy': float(orthogonal_energy[:, h].mean()),
        'maxAbsRowIdentityResidual': float(np.abs(cell_residual[:, h]).max())} for h in range(24)]
    for n in ['request.json', 'inputs.json']:
        shutil.copyfile(root / n, out / n)
    save(out / 'report.json', {'schema': 'tce-diagonal-norm-moments-v1', 'requestSHA256': r['requestSHA256'],
        'rows': rows, 'retainedMoments': arrays,
        'pooled': {'trueMeanEnergy': float(true_mean.mean()), 'trueVariationEnergy': float(true_variation.mean()),
            'diagonalMeanEnergy': float(diag_mean.mean()), 'diagonalVariationEnergy': float(diag_variation.mean()),
            'meanOmittedEnergy': float(mean_term.mean()), 'variationOmittedEnergy': float(variation_term.mean()),
            'totalOmittedEnergy': float(direct.mean()), 'maxAbsMomentIdentityResidual': float(np.abs(residual).max()),
            'rowCrossRMS': float(np.sqrt(np.mean(cross * cross))), 'rowCenteredDefectRMS': float(np.sqrt(np.mean(centered_defect * centered_defect))),
            'undefinedMeanDirections': int(np.sum(mean_length == 0)),
            'targetMeanOrthogonalEnergy': float(orthogonal_energy.mean()), 'maxAbsRowIdentityResidual': float(np.abs(cell_residual).max())},
        'newMatrixMACs': 151142400, 'rowDotMultiplications': 6291456, 'newOriginalMapProjections': 0, 'modelFrameworkImports': 0,
        'rowDotCountScope': 'Only the two explicit output-space contractions. Additional square/product/reduction/scaling and orthogonal-residual passes remain paid.',
        'workerWallSeconds': time.perf_counter() - started, 'mechanics_status': 'complete', 'terminal_status': 'unassessed',
        'scope': 'Post-hoc same-input saved-only descriptive moments. Mean terms locate shared spatial structure, not semantic roles or a serving law. For undefined zero mean directions use direction0 and full raw energy, explicitly flagged. Signed terms/roundoff retained; no model/refit/consumer execution or primary forecast feedback.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']);p.add_argument('run');a = p.parse_args()
    if a.action == 'collect':
        from response_study import collect
        value = collect(a.run)
    else:
        value = globals()[a.action](a.run)
    if value is not None:
        print(json.dumps(value))
