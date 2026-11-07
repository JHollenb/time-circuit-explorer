"""One guarded saved-only CPU support census; no learned projection or consumer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from encoder_atoms import read, save, sha, identity

HERE = Path(__file__).resolve().parent
OWNED = ['rectifier_sign_support.py', 'rectifier_sign_support_study.py',
         'test_rectifier_sign_support.py', 'encoder_atoms.py', 'paired_fields.py',
         'terminal_capture_guard_v2.py']
PLAN = 'docs/plans/rectifier-sign-support.md'
REVIEWS = ['docs/results/rectifier-correction-evidence/advisor.json',
           'docs/results/research-review-cycle81/advisor-survey.json',
           'docs/results/storage-audit-cycle81/publication.json']


def bindings(parent):
    source = read(parent / 'inputs.json')
    fields = read(parent / 'candidate.json')['fields']
    return {'gateReference': source['sourceFields']['gateReference'],
            'rectifierMean': fields['rectifierMean'],
            'activationMean': source['comparisonFields']['activationMean'],
            'selection_support': source['sourceFields']['selection_support']}


def verify(root, request):
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']:
        raise ValueError('Request identity differs')
    for name, fingerprint in request['sourceHashes'].items():
        if sha(root / name) != fingerprint:
            raise ValueError('Frozen source/input differs: ' + name)


def stage(root):
    from mrun.client.api import Api
    from publish_source_native import checked
    from publish_rectifier_correction import audit
    api = Api()
    workspace = HERE.parent
    audit(workspace / 'docs/results/rectifier-correction')
    survey = read(workspace / REVIEWS[1])
    storage = read(workspace / REVIEWS[2])
    if not survey['newFullNewestPostSurvey'] or survey['cycle'] != 81 or storage['cycle'] != 81:
        raise ValueError('Due full newest-post/storage reviews incomplete')
    directory, collection, job, parent_request = checked(workspace / 'artifacts/rectifier-correction-v1')
    live = api.json('GET', '/api/jobs/' + job['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != collection['custody']:
        raise ValueError('Parent scheduler custody differs')
    parent = directory / 'result'
    fields = bindings(parent)
    witness = read(directory / 'retained-array-verification.json')
    for name in OWNED:
        if name in parent_request['ownedSources'] and sha(HERE / name) != parent_request['sourceHashes'][name]:
            raise ValueError('Reused executed dependency changed')
    for item in fields.values():
        if (witness['files'].get(item['path'], {}).get('sha256') != item['sha256']
                or witness['npyPayloadWitnesses'].get(item['path'], {}).get('payloadSHA256') != item['payloadSHA256']):
            raise ValueError('Full-byte source witness absent')
    prior = read(parent / 'inputs.json')
    all_fields = [*prior['sourceFields'].values(), *prior['comparisonFields'].values(),
                  *read(parent / 'candidate.json')['fields'].values(),
                  *read(parent / 'analysis-fields.json')['fields'].values()]
    allowed = {x['path'] for x in fields.values()}
    source = {'fields': fields,
              'parent': {'jobID': job['job_id'], 'custody': collection['custody'],
                         'root': job['result']['diagnostics']['cwd'],
                         'records': {n: sha(parent / n) for n in ['request.json', 'inputs.json', 'candidate.json',
                                     'execution.json', 'candidate-seal.json', 'analysis-fields.json', 'report.json']}},
              'sourceRoots': sorted({str(Path(x['path']).parents[2]) for x in fields.values()}),
              'deniedNumericalInputs': sorted({x['path'] for x in all_fields}-allowed),
              'preStageReviewSnapshots': {n: sha(workspace / n) for n in REVIEWS},
              'scope': 'Four paid source fields only. U/Upmeans/maps/old products/output errors never admitted.'}
    root = Path(root).resolve()
    root.mkdir(exist_ok=False, parents=True)
    payload = root / 'payload'
    payload.mkdir()
    for name in OWNED:
        shutil.copyfile(HERE / name, payload / name)
    shutil.copyfile(workspace / PLAN, payload / 'frozen-plan.md')
    save(payload / 'inputs.json', source)
    host = next(x for x in api.json('GET', '/api/hosts') if x['name'] == 'beast')
    save(payload / 'resource-basis.json', {
        'sourceFiles': 4, 'sourceFileBytes': sum(x['bytes'] for x in fields.values()),
        'newArrayLogicalBytes': 6124032, 'newArrayNPYBytesExpected': 6124800,
        'memoryPolicy': 'Read-only mmap gate113MB and two147KBmeans/support;16-row scalar-expression tiles,6.124MB retained bitsets/counts. No full coefficient/product/output or learned map matrix.',
        'liveHost': {k: host[k] for k in ['name', 'active_jobs', 'admission', 'telemetry', 'ram_total_mb']},
        'admission': 'Intent-first saved-only CPU analysis; no hand-declared RAM/VRAM. Hard host binding required by authenticated host-local arrays. Thread environments1, finite180s, one lease/no retries. Scalar observations and packed arrays under10MB.'})
    request = {'schema': 'tce-rectifier-sign-support-request-v1', 'ownedSources': OWNED,
               'sourceHashes': {p.name: sha(p) for p in payload.iterdir()},
               'device': 'cpu', 'backend': 'numpy', 'dtype': 'float64', 'timeoutSeconds': 180,
               'plannedLearnedMatrixMACs': 0, 'newCheckpointTensorLoads': 0,
               'newRectifierElements': 1536*9216, 'fixedAddress': {'clock': 3, 'singleBlock': 0},
               'fixedTileRows': 16, 'textRows': 512, 'imageRows': 1024,
               'protocol': 'Exact nonzero omitted coefficients for ReLU(Gstar),r-rbar,r+(Abar-rbar),fixed16-row stream tiles. Four paid inputs; no U/maps/outputerrors/threshold/fit/sorting/tile sweep. Packed complete supports plus row/channel/tile counts. Prospective Up/Down and dense stream fold counts only; no learned products/new state/image.'}
    request['requestSHA256'] = identity(request)
    save(root / 'request.json', request)
    save(payload / 'request.json', request)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1',
         'hashed_roots': [p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'root': str(root), 'requestSHA256': request['requestSHA256'],
            'sourceFiles': 4, 'sourceFileBytes': sum(x['bytes'] for x in fields.values())}


def submit(root):
    from mrun.client.submit import launch
    root = Path(root).resolve()
    request = read(root / 'request.json')
    verify(root / 'payload', request)
    if (root / 'job.json').exists():
        raise ValueError('Already submitted; inspect same authoritative handle')
    command = ['/usr/bin/env', 'PYTHONUNBUFFERED=1', 'PYTHONNOUSERSITE=1', 'PYTHONPATH=',
               'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
               'UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache', 'UV_LINK_MODE=symlink',
               'uv', 'run', '--no-project', '--offline', '--python', '3.14.6', '--with', 'numpy>=2.0',
               'python', '-u', 'rectifier_sign_support_study.py', 'execute', '.']
    job = launch(command, experiment='tce-rectifier-sign-support-v1', task='analysis',
                 payload=root / 'payload', pin='beast', needs={'cpu': True},
                 config={k: request[k] for k in ['schema', 'requestSHA256', 'device', 'backend', 'dtype', 'plannedLearnedMatrixMACs']},
                 timeout_s=180, detach=True, retry_on_kill=False, preflight=True, guarded=True,
                 note='Saved-only clock3single0 coefficient liveness:4 authenticated fields,3exact omitted supports,16-row fixed tiles. CPU1 env,intent-first,180scap,noretry. Zero learned products/U/maps/errors/thresholds/fit/state/image. Charge paid G/means/history and hypothetical dense c_s compile/apply;6.124MB complete bitsets/counts. Stop after census.')
    value = {'jobID': str(job), 'requestSHA256': request['requestSHA256']}
    save(root / 'job.json', value)
    return value


def execute(root):
    from encoder_atoms import guard, np
    guard()
    from paired_fields import retain
    from rectifier_sign_support import measure, abis, EXPRESSIONS, BITORDER
    from terminal_capture_guard_v2 import quarantine
    from test_rectifier_sign_support import mechanics
    root = Path(root).resolve()
    request = read(root / 'request.json')
    verify(root, request)
    source = read(root / 'inputs.json')
    started = time.perf_counter()
    cpu = time.process_time()
    output = root / 'response-result'
    output.mkdir(exist_ok=False)
    save(output / 'toy-mechanics.json', mechanics())
    allowed = [x['path'] for x in source['fields'].values()]
    access = quarantine(root, allowed, source['deniedNumericalInputs'], root / 'unused-store')
    expected = {'gateReference': ([1, 1536, 9216], 'float64'),
                'rectifierMean': ([1, 2, 9216], 'float64'), 'activationMean': ([1, 2, 9216], 'float64'),
                'selection_support': ([2304], 'int64')}
    fields = {}
    for name, item in source['fields'].items():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Saved source bytes differ')
        value = np.load(item['path'], mmap_mode='r', allow_pickle=False)
        if ((list(value.shape), str(value.dtype)) != expected[name]
                or list(value.shape) != item['shape'] or str(value.dtype) != item['storedDtype']
                or value.flags.writeable or hashlib.sha256(memoryview(value).cast('B')).hexdigest() != item['payloadSHA256']):
            raise ValueError('Read-only source ABI/payload differs')
        fields[name] = value
    setup = time.perf_counter()-started
    print('RECTIFIER_SIGN_SUPPORT_START', flush=True)
    counting = time.perf_counter()
    arrays, panels, fold = measure(fields['gateReference'][0], fields['rectifierMean'][0],
                                   fields['activationMean'][0], fields['selection_support'])
    counting = time.perf_counter()-counting
    retained = {name: retain(root / 'arrays', name, value) for name, value in arrays.items()}
    if {n: (x['shape'], x['storedDtype']) for n, x in retained.items()} != abis():
        raise ValueError('Complete support/count ABIs differ')
    for stream, lo, hi in [(0, 0, 512), (1, 512, 1536)]:
        if not np.array_equal(np.sum(arrays['rowCounts'][:, lo:hi], axis=1),
                              np.sum(arrays['channelCounts'][:, stream], axis=1)):
            raise ValueError('Row/channel support census disagrees')
    with (output / 'observations.jsonl').open('x') as stream:
        for row in range(1536):
            stream.write(json.dumps({'row': row, 'stratum': 'text' if row < 512 else 'image',
                         'activeOmitted': {name: int(arrays['rowCounts'][i, row]) for i, name in enumerate(EXPRESSIONS)}}, sort_keys=True)+'\n')
    with (output / 'tiles.jsonl').open('x') as stream:
        for tile in range(96):
            stream.write(json.dumps({'tile': tile, 'startRow': 16*tile, 'rows': 16,
                         'stratum': 'text' if tile < 32 else 'image',
                         'unionOmitted': {name: int(arrays['tileUnionCounts'][i, tile]) for i, name in enumerate(EXPRESSIONS)}}, sort_keys=True)+'\n')
    for item in source['fields'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Source mutation')
    for name in ['inputs.json', 'request.json', 'resource-basis.json', 'frozen-plan.md']:
        shutil.copyfile(root / name, output / name)
    save(output / 'fields.json', {'fields': retained, 'requestSHA256': request['requestSHA256'],
         'bitOrder': BITORDER, 'expressionAxis': list(EXPRESSIONS), 'streamAxis': ['text', 'image'],
         'tileOrder': '32text then64image;16consecutive rows, no cross-stream tile'})
    save(output / 'report.json', {'schema': 'tce-rectifier-sign-support-result-v1',
         'requestSHA256': request['requestSHA256'], 'fixedAddress': request['fixedAddress'],
         'panels': panels, 'denseStreamFold': fold, 'observations': 1536, 'tiles': 96,
         'fullWidthUpDownBaselineMACs': 86973087744, 'paidFreshGateFormationMACs': 43486543872,
         'physicalWork': {'newLearnedMatrixMACs': 0, 'newCheckpointTensorLoads': 0,
                          'newRectifierElements': 14155776, 'newRetainedArrays': 6,
                          'newNativeStateUpdates': 0, 'newImages': 0},
         'mechanics_status': 'complete', 'instrument_status': 'saved-source-coefficient-support-census',
         'trend_status': 'All coefficient and fixed-tile densities retained without quality threshold',
         'terminal_status': 'unassessed',
         'scope': 'Exact nonzero support of named FP64 coefficient expressions on one paid saved state. Coefficient demand is not actual product support, native finite output parity, speed or prompt generalization. Learned G/means/support/history remain paid; no U/maps/errors/output/state/image evaluation.'})
    save(output / 'execution.json', {'sourceManifestSHA256': identity(source['fields']),
         'setupSecondsAfterImports': setup, 'supportCountingSeconds': counting,
         'wallSecondsAfterImports': time.perf_counter()-started, 'CPUSecondsAfterImports': time.process_time()-cpu,
         'numpyVersion': np.__version__, 'threadEnvironments': {n: os.environ.get(n) for n in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
         'sourceClosure': True, 'sourceFiles': 4, 'sourceFileBytes': sum(x['bytes'] for x in source['fields'].values()),
         'newArrayLogicalBytes': sum(x.nbytes for x in arrays.values()),
         'newArrayFileBytes': sum(x['bytes'] for x in retained.values()),
         'fileQuarantine': {'allowedFiles': sorted(allowed), 'deniedCanaries': source['deniedNumericalInputs'], **access}})
    save(output / 'result-seal.json', {'requestSHA256': request['requestSHA256'],
         'files': {x.name: sha(x) for x in output.iterdir()}})
    print('RECTIFIER_SIGN_SUPPORT_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stage', 'submit', 'execute', 'collect'])
    parser.add_argument('root')
    args = parser.parse_args()
    if args.action == 'collect':
        from response_study import collect
        value = collect(args.root)
    else:
        value = globals()[args.action](args.root)
    print(json.dumps(value))
