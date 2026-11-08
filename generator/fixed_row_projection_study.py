"""One guarded, saved-only CPU projection diagnostic; no neural runtime."""
import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from encoder_atoms import read, save, sha, identity
from fixed_row_projection import SOURCES, abis, work

HERE = Path(__file__).resolve().parent
OWNED = ['fixed_row_projection.py', 'fixed_row_projection_study.py', 'test_fixed_row_projection.py', 'encoder_atoms.py', 'paired_fields.py', 'terminal_capture_guard_v2.py']
PARENTS = {'83': 'current-row-output-v1', '82': 'current-state-gate-up-v1'}


def bindings(p83, p82):
    c83 = read(p83/'candidate.json')['fields']; c82 = read(p82/'candidate.json')['fields']
    s83 = read(p83/'inputs.json'); s82 = read(p82/'inputs.json')
    for name in ['gate', 'text', 'image']:
        if s83['sourceFields'][name] != s82['sourceFields'][name]:
            raise ValueError('Common gate/incoming source binding differs')
    result = {name: c83[name] for name in ['rowBasis', 'updateFP64']}
    for tag, candidate in [('83', c83), ('82', c82)]:
        for dest, name in [('update', 'preResidualUpdate'), ('gated', 'gatedWrite'), ('exit', 'exit')]:
            result[dest+tag] = candidate[name]
    result.update({name: s83['sourceFields'][name] for name in ['gate', 'text', 'image']})
    return result


def verify(root, r):
    if identity({k: v for k, v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']:
        raise ValueError('Request changed')
    for name, digest in r['sourceHashes'].items():
        if sha(Path(root)/name) != digest:
            raise ValueError('Frozen source/input differs: '+name)


def stage(root):
    from mrun.client.api import Api
    from publish_source_native import checked
    from publish_row_output import audit
    from projected_complement import fields
    api = Api(); audit(HERE.parent/'docs/results/current-row-output'); parents = {}; dirs = {}; witnesses = []
    for tag, folder in PARENTS.items():
        parent, col, job, previous = checked(HERE.parent/'artifacts'/folder)
        current = api.json('GET', '/api/jobs/'+job['job_id'])
        if current['state'] != 'succeeded' or current['payload_custody'] != col['custody']:
            raise ValueError('Parent current custody differs')
        dirs[tag] = parent/'result'; witnesses.append(read(parent/'retained-array-verification.json'))
        parents[tag] = {'jobID': job['job_id'], 'custody': col['custody'], 'root': job['result']['diagnostics']['cwd'],
            'records': {n: sha(dirs[tag]/n) for n in ['request.json', 'inputs.json', 'candidate.json', 'execution.json', 'candidate-seal.json', 'producer-exit.json', 'report.json']}}
        for name in OWNED:
            if name in previous['ownedSources'] and sha(HERE/name) != previous['sourceHashes'][name]:
                raise ValueError('Frozen reused helper differs: '+name)
    source = bindings(dirs['83'], dirs['82'])
    for item in source.values():
        if not any(v['files'].get(item['path'], {}).get('sha256') == item['sha256'] and v['npyPayloadWitnesses'].get(item['path'], {}).get('payloadSHA256') == item['payloadSHA256'] for v in witnesses):
            raise ValueError('Complete saved source byte witness absent')
    s83 = read(dirs['83']/'inputs.json'); s82 = read(dirs['82']/'inputs.json')
    lineage = {key: s83[key] for key in ['modelRoot', 'nativeSources', 'imageAssets', 'versions', 'numericalProfile']}
    if any(s82[key] != value for key, value in lineage.items()):
        raise ValueError('Historical model/profile identity differs')
    known = {item['path'] for tag in dirs for item in fields(read(dirs[tag]/'candidate.json'))}
    known |= {item['path'] for tag in dirs for item in fields(read(dirs[tag]/'analysis-fields.json'))}
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); payload = root/'payload'; payload.mkdir()
    for name in OWNED:
        shutil.copyfile(HERE/name, payload/name)
    for src, dest in [('docs/plans/fixed-row-projection-diagnostic.md', 'frozen-plan.md'), ('docs/plans/fixed-row-projection-implementation.md', 'implementation-notes.md')]:
        shutil.copyfile(HERE.parent/src, payload/dest)
    snapshots = ['docs/results/current-row-output-evidence/advisor.json', 'docs/results/research-review-cycle84/advisor-survey.json', 'docs/results/storage-audit-cycle84/publication.json']
    save(payload/'inputs.json', {'fields': source, 'parents': parents, 'historicalModelIdentity': lineage,
        'deniedNumericalInputs': sorted(known-{a['path'] for a in source.values()}), 'sourceRoots': sorted({a['root'] for a in parents.values()} | {str(Path(a['path']).parent.parent.parent) for a in source.values()}),
        'preStageReviewSnapshots': {name: sha(HERE.parent/name) for name in snapshots}})
    host = next(a for a in api.json('GET', '/api/hosts') if a['name'] == 'beast')
    logical_source = sum(math.prod(a['shape'])*8 for a in source.values())
    output = sum(math.prod(a['shape'])*8 for a in abis().values())
    save(payload/'resource-basis.json', {'sourceFiles': len(source), 'sourceFileBytes': sum(a['bytes'] for a in source.values()), 'promotedSourceBytes': logical_source, 'retainedOutputLogicalBytes': output,
        'extraTemporaryBudgetBytes': 256 << 20, 'workingArrayBudgetBytes': logical_source+output+(256 << 20),
        'liveHost': {k: host[k] for k in ['name', 'active_jobs', 'admission', 'telemetry', 'ram_total_mb']},
        'admission': 'Intent-first CPU analysis, no hand-reserved RAM/VRAM, four configured BLAS threads. Host pin required by authenticated retained arrays. No GPU/model acquisition. 180s cap, no retries; complete tensors stay remote.',
        'scope': 'Eleven promoted saved fields, fifteen full-width output/account arrays, two-stream compressed rows, scalar Grams/means and denominator arrays. Pairwise Gram temporaries reused; no stacked full-width term cube. Copies/hashes/NumPy/BLAS/import overhead remain additional.'})
    r = {'schema': 'tce-fixed-row-projection-request-v1', 'ownedSources': OWNED, 'sourceHashes': {a.name: sha(a) for a in payload.iterdir()},
        'device': 'cpu', 'backend': 'numpy', 'dtype': 'float64', 'timeoutSeconds': 180, 'plannedWork': work(), 'fixedAddress': {'clock': 0, 'singleBlock': 0},
        'protocol': 'One saved-U numerical projection of saved82 update. Four FP64 NumPy CPU contractions; no learned map/QR/fit/runtime/model/consumer/image. Retain a,b,c plus independently saved gate/commit remainders, explicit account closures and every full/centered signed Gram row. Denominators use saved exit-minus-common-incoming. All historical formation paid; b combined finite-policy/reassociation, not uniquely native rounding or pure attention/MLP escape.'}
    r['requestSHA256'] = identity(r); save(root/'request.json', r); save(payload/'request.json', r)
    save(payload/'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [a.name for a in payload.iterdir()]+['preflight.v1.json']})
    return {'root': str(root), 'requestSHA256': r['requestSHA256'], 'matrixMACs': work()['matrixMACs'], 'workingArrayBudgetBytes': logical_source+output+(256 << 20)}


def submit(root):
    from mrun.client.submit import launch
    from response_study import command
    root = Path(root).resolve(); r = read(root/'request.json'); verify(root/'payload', r)
    if (root/'job.json').exists():
        raise ValueError('Already submitted; inspect same handle')
    cmd = [a.replace('THREADS=1', 'THREADS=4') for a in command()[:-5]]+['python', '-u', 'fixed_row_projection_study.py', 'execute', '.']
    cmd.insert(1, 'PYTHONDONTWRITEBYTECODE=1')
    job = launch(cmd, experiment='tce-fixed-row-projection-v1', task='analysis', payload=root/'payload', pin='beast', needs={'cpu': True},
        config={k: r[k] for k in ['schema', 'requestSHA256', 'device', 'backend', 'dtype', 'plannedWork']}, cpu_threads=4,
        timeout_s=180, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='Saved83U/Y64/BF16/G/X and82Y/G/X, commongate/incoming;11authenticated arrays. FourFP64CPU project/reconstruct contractions1.208BMAC; pairwise scalar Gram/mean/denominator reductions separate, no model/learnedmap/QR/fit/consumer/image. Independently saved gate/commit remainders, fullcentered signed everyrow, explicit closures. CPU4,intentfirstRAM,noGPU,180scap,noretry,onelease; historical formation paid.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root/'job.json', value); return value


def execute(root):
    from encoder_atoms import guard
    guard()
    import numpy as np
    from paired_fields import load, retain, payload
    from terminal_capture_guard_v2 import quarantine
    from fixed_row_projection import compute, observations, summarize
    root = Path(root).resolve(); r = read(root/'request.json'); verify(root, r); s = read(root/'inputs.json'); start = time.perf_counter(); cpu = time.process_time()
    access = quarantine(root, [a['path'] for a in s['fields'].values()], s['deniedNumericalInputs'], root/'unused-store')
    f = {name: load(s['fields'][name]) for name in SOURCES}; before = {n: payload(a) for n, a in f.items()}; operations = []
    def mm(name, a, b):
        tick = time.perf_counter(); value = a@b; elapsed = time.perf_counter()-tick
        operations.append({'name': name, 'leftShape': list(a.shape), 'rightShape': list(b.shape), 'outputShape': list(value.shape), 'dtype': str(value.dtype), 'device': 'cpu', 'operator': 'numpy.matmul',
            'nominalMACs': value.size*a.shape[-1], 'secondsBeforeFingerprints': elapsed, 'leftPayloadSHA256': payload(a), 'rightPayloadSHA256': payload(b), 'outputPayloadSHA256': payload(value)})
        return value
    ready = time.perf_counter(); result = compute(f, mm); computed = time.perf_counter()
    if {n: {'shape': list(a.shape), 'dtype': str(a.dtype)} for n, a in result.items()} != abis() or any(not np.isfinite(a).all() for a in result.values()):
        raise ValueError('Finite complete output ABI differs')
    if len(operations) != 4 or sum(a['nominalMACs'] for a in operations) != work()['matrixMACs'] or work() != r['plannedWork']:
        raise ValueError('Four-product work instrument differs')
    out = root/'response-result'; out.mkdir(exist_ok=False); fields = {n: retain(root/'arrays', n, a) for n, a in result.items()}
    rows = list(observations(result))
    with (out/'observations.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    if before != {n: payload(a) for n, a in f.items()} or any(sha(a['path']) != a['sha256'] for a in s['fields'].values()):
        raise ValueError('Read-only source closure differs')
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    save(out/'fields.json', {'fields': fields, 'requestSHA256': r['requestSHA256']})
    save(out/'execution.json', {'products': operations, 'numericalProfile': {'numpy': np.__version__, 'BLASBuildConfiguration': config.getvalue(), 'environment': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']}, 'dtype': 'float64', 'device': 'cpu', 'scope': 'Recorded NumPy/build/thread configuration and operation calls, not retired-kernel or accumulator attestation.'},
        'sourceClosure': True, 'setupSeconds': ready-start, 'computeAndInternalFingerprintSeconds': computed-ready, 'CPUSeconds': time.process_time()-cpu, 'fileQuarantine': access,
        'blockedNeuralRuntimeRoots': sorted(__import__('encoder_atoms').NoNeural.roots), 'loadedNeuralRuntimeRoots': sorted(__import__('encoder_atoms').NoNeural.roots.intersection(sys.modules))})
    save(out/'report.json', {'schema': 'tce-fixed-row-projection-result-v1', 'requestSHA256': r['requestSHA256'], 'fixedAddress': r['fixedAddress'], 'work': work(),
        'panels': {name: summarize(rows[lo:hi]) for name, lo, hi in [('all', 0, 1536), ('text', 0, 512), ('image', 512, 1536)]}, 'observations': len(rows), 'workerWallSecondsAfterImports': time.perf_counter()-start,
        'mechanics_status': 'complete', 'instrument_status': 'saved-fixed-U-signed-finite-output-account', 'trend_status': 'All signed full/centered components and countertrends retained without quality gate', 'terminal_status': 'unassessed',
        'scope': 'Retrospective diagnosis of83-vs82 recorded updates/exits; historical original model/source formation remains paid. Fixed numerical projection not certified ideal projector; finitePolicy combined CPU FP64 projection/grouping versus recorded output policy. No serving correction, attention-vs-MLP attribution, continuation, image or arbitrary-prompt zero-neural claim.'})
    for name in ['inputs.json', 'request.json', 'resource-basis.json', 'frozen-plan.md', 'implementation-notes.md']:
        shutil.copyfile(root/name, out/name)
    print('FIXED_ROW_PROJECTION_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']); parser.add_argument('root'); args = parser.parse_args()
    if args.action == 'collect':
        from response_study import collect
        value = collect(args.root)
    else:
        value = globals()[args.action](args.root)
    if value is not None:
        print(json.dumps(value))
