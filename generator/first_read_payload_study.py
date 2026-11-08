"""One guarded saved-only first-reader allocation/value diagnostic."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
OWNED = ['first_read_payload.py', 'first_read_payload_study.py', 'test_first_read_payload.py', 'encoder_atoms.py']
PARENT_SEAL = '8f9b0eb30195036e3791ec11099373d9356a1049b3d08590b233b4c55adff8f5'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify(root, request):
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']:
        raise ValueError('Request identity')
    for name, digest in request['sourceHashes'].items():
        if sha(Path(root)/name) != digest:
            raise ValueError('Source/input differs: '+name)


def stage(root):
    from mrun.client.api import Api
    parent = HERE.parent/'docs/results/uncentered-reader'
    intake_path = HERE.parent/'artifacts/first-read-route-payload-intake97/source-headers.json'
    intake = read(intake_path); seal = read(parent/'reader-seal.json')
    if sha(parent/'reader-seal.json') != PARENT_SEAL or intake['readerSealSHA256'] != PARENT_SEAL or seal['policy'] != 'current-top4-uncentered-fp64-fp32-bf16-v1':
        raise ValueError('Historical reader binding')
    request_old = read(parent/'request.json')
    for name in ['encoder_atoms.py', 'uncentered_reader.py']:
        if sha(HERE/name) != request_old['sourceHashes'][name]:
            raise ValueError('Historical reader/cast source differs')
    expected = {(r['id'], 'lexical'): r for r in seal['lexicalRows']}
    expected.update({(r['id'], 'selected'): r for r in seal['packets']})
    if len(intake['intake']['files']) != 4 or {(x['id'], x['kind']) for x in intake['intake']['files']} != {('A', 'lexical'), ('B', 'lexical'), ('A', 'selected'), ('B', 'selected')}:
        raise ValueError('Complete four-source intake required')
    for item in intake['intake']['files']:
        old = expected[(item['id'], item['kind'])]
        if any(item[k] != old[k] for k in ['path', 'sha256']):
            raise ValueError('Source identity differs from historical seal')
    original = Api().json('GET', '/api/jobs/'+intake['originalJobID'])
    if original['state'] != 'succeeded' or original.get('payload_custody') != intake['originalPayloadCustody']:
        raise ValueError('Original succeeded job/custody differs')
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); payload = root/'payload'; payload.mkdir()
    for name in OWNED:
        shutil.copyfile(HERE/name, payload/name)
    shutil.copyfile(HERE/'uncentered_reader.py', payload/'original-reader.py')
    for name in ['reader-seal.json', 'request.json', 'provenance.json']:
        shutil.copyfile(parent/name, payload/('parent-'+name))
    shutil.copyfile(intake_path, payload/'inputs.json')
    for name, destination in [('first-read-route-payload.md', 'plan.md'), ('first-read-route-payload-implementation.md', 'implementation.md')]:
        shutil.copyfile(HERE.parent/'docs/plans'/name, payload/destination)
    r = {'schema': 'tce-first-read-payload-request-v1', 'sourceHashes': {p.name: sha(p) for p in payload.iterdir()},
         'device': 'cpu', 'backend': 'numpy2.5.4-source-FP64-per-query-dot/FP32-BF16-RNE', 'head': 0, 'kvGroup': 0,
         'paddingQueries': 477, 'headDim': 128, 'crossReadProducts': 488448, 'endpointReplayProducts': 488448,
         'scope': 'Paid retrospective selected-top4 first-read crossover on one35-key A/B pair. Captured weights and token-local values are supplied. No new prompt/image capability.'}
    r['requestSHA256'] = identity(r); save(root/'request.json', r); save(payload/'request.json', r)
    save(payload/'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'root': str(root), 'requestSHA256': r['requestSHA256'], 'retainedSourceBytes': intake['intake']['bytes']}


def submit(root):
    from mrun.client.submit import launch
    root = Path(root).resolve(); r = read(root/'request.json'); verify(root/'payload', r)
    if (root/'job.json').exists():
        raise ValueError('Already submitted; inspect same handle')
    command = ['/usr/bin/env', 'PYTHONUNBUFFERED=1', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONNOUSERSITE=1', 'PYTHONPATH=',
        'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache', 'UV_LINK_MODE=symlink',
        'uv', 'run', '--no-project', '--offline', '--python', '3.14.6', '--with', 'numpy==2.5.4',
        'python', '-u', 'first_read_payload_study.py', 'execute', '.']
    job = launch(command, experiment='tce-first-read-route-payload-v1', task='analysis', payload=root/'payload', pin='beast', needs={'cpu': True},
        config={k: r[k] for k in ['schema', 'requestSHA256', 'device', 'backend', 'head', 'crossReadProducts', 'endpointReplayProducts']},
        timeout_s=300, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='Saved firstreader head0/group0,one35-key A/Bedit@8,477dependentpaddingqueries. Two crossreads488448valueproducts,sealbeforehistoricalraw/roundedmemberreads;endpointreplay488448separate. Capturedorderedtop4weights/tokenlocalV supplied;diagnosticGram/rounding work paid. CPU/BLAS1,hostlocalsourcepin,300scap,noretry. No QK/softmax/checkpoint/MLP/suffix/VAE/image. Preserve finite mismatch/cancellation,notsemanticshares.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root/'job.json', value); return value


def execute(root):
    import importlib
    from encoder_atoms import np, guard, floats, words
    from first_read_payload import member, qualify, selected_read, effects, row_metrics, aggregate, endpoint_comparison, digest
    guard(); denied = []
    for name in ['torch', 'transformers', 'diffusers']:
        try:
            importlib.import_module(name)
        except ImportError:
            denied.append(name)
        else:
            raise RuntimeError('Neural import canary was not denied')
    try:
        with open('/mnt/big/llm-models/tce-first-read-canary.safetensors', 'rb'):
            pass
    except PermissionError:
        denied.append('checkpoint-open')
    else:
        raise RuntimeError('Checkpoint canary was not denied')
    root = Path(root).resolve(); r = read(root/'request.json'); verify(root, r)
    started = time.perf_counter(); cpu = time.process_time(); source = read(root/'inputs.json')
    if np.__version__ != '2.5.4':
        raise ValueError('Source-qualified NumPy version differs')
    out = root/'response-result'; out.mkdir(exist_ok=False); arrays = root/'arrays'; arrays.mkdir()
    items = {(x['id'], x['kind']): x for x in source['intake']['files']}
    for item in items.values():
        p = Path(item['path'])
        if p.is_symlink() or p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise ValueError('Retained source bytes changed')
    ledger = []; packets = {}
    for name in ['A', 'B']:
        lex, selected = items[name, 'lexical'], items[name, 'selected']
        packets[name] = {k: member(lex, k, slice(None), 'before-cross-seal', ledger) for k in ['ids', 'mask', 'positions']}
        packets[name]['v'] = member(lex, 'v', (slice(None), slice(0, 128)), 'before-cross-seal', ledger)
        for key, original, selection in [('support', 'supportPositions', (slice(35, None), 0, slice(None))),
                                          ('weights', 'selectedWeights', (slice(35, None), 0, slice(None))),
                                          ('counts', 'prefixCounts', slice(35, None))]:
            packets[name][key] = member(selected, original, selection, 'before-cross-seal', ledger)
    a, b = packets['A'], packets['B']; qualification = qualify(a, b); ready = time.perf_counter()
    crosses = {}; work = {}
    for label, reader, values in [('AB', a, b), ('BA', b, a)]:
        crosses[label], work[label] = selected_read(reader['weights'], reader['support'], values['bank'], values['admittedPositions'])
    np.savez(arrays/'crosses.npz', **crosses)
    save(out/'cross-seal.json', {'requestSHA256': r['requestSHA256'], 'arrayPath': str(arrays/'crosses.npz'),
         'sha256': sha(arrays/'crosses.npz'), 'bytes': (arrays/'crosses.npz').stat().st_size,
         'fields': {k: digest(v) for k, v in crosses.items()}, 'memberReadsBeforeSeal': list(ledger),
         'scope': 'Crosses sealed before numerical raw/rounded endpoint-member access. Whole source files already byte-hashed. Supplied weights/values are not predicted.'})
    replay = {}
    for label, packet in [('AA', a), ('BB', b)]:
        replay[label], work[label] = selected_read(packet['weights'], packet['support'], packet['bank'], packet['admittedPositions'])
    endpoints = {}; rounded = {}; comparison = {}
    for name, diagonal in [('A', 'AA'), ('B', 'BB')]:
        selected = items[name, 'selected']
        endpoints[diagonal] = member(selected, 'raw', (slice(35, None), slice(0, 128)), 'after-cross-seal', ledger)
        rounded[diagonal] = member(selected, 'rounded', (slice(35, None), slice(0, 128)), 'after-cross-seal', ledger)
        if not np.isfinite(endpoints[diagonal]).all() or not np.isfinite(floats(rounded[diagonal])).all():
            raise ValueError('Nonfinite retained endpoint')
        comparison[name] = endpoint_comparison(replay[diagonal], endpoints[diagonal], rounded[diagonal])
    p8 = {name: np.sum(packet['weights']*(packet['support'] == 8), axis=1) for name, packet in packets.items()}
    dv = b['bank'][8]-a['bank'][8]
    ideal_payload = .5*(p8['A']+p8['B'])[:, None]*dv
    ideal_interaction = (p8['B']-p8['A'])[:, None]*dv
    raw = {**endpoints, **crosses}; replay_corners = {**replay, **crosses}
    admitted = {**{k: floats(v).astype(np.float64) for k, v in rounded.items()}, **{k: floats(words(v)).astype(np.float64) for k, v in crosses.items()}}
    panels = {}; retained = {**{'raw-'+k: v for k, v in raw.items()}, **{'replay-'+k: v for k, v in replay.items()},
                             **{'bf16-'+k: words(v) for k, v in crosses.items()}, **{'retained-bf16-'+k: v for k, v in rounded.items()},
                             'ideal-payload': ideal_payload, 'ideal-interaction': ideal_interaction, 'value-delta': dv}
    row_panels = {}
    for name, corners in [('retainedRaw', raw), ('replayRaw', replay_corners), ('admittedBF16', admitted)]:
        metrics, vectors = row_metrics(effects(corners['AA'], corners['AB'], corners['BA'], corners['BB']), ideal_payload, ideal_interaction)
        panels[name] = aggregate(metrics); row_panels[name] = metrics
        retained.update({name+'-'+k: v for k, v in vectors.items()})
    records = []
    for i in range(477):
        sa, sb = a['support'][i].tolist(), b['support'][i].tolist()
        records.append({'queryPosition': i+35, 'supportA': sa, 'supportB': sb, 'weightsA': a['weights'][i].tolist(), 'weightsB': b['weights'][i].tolist(),
            'pA8': float(p8['A'][i]), 'pB8': float(p8['B'][i]), 'editedAddressInA': 8 in sa, 'editedAddressInB': 8 in sb,
            'supportSymmetricDifference': len(set(sa)^set(sb)), 'orderedSupportChanged': sa != sb,
            'admittedEndpointWordChanges': int(np.sum(rounded['AA'][i] != rounded['BB'][i])),
            'admittedPayloadAtAllocationAWordChanges': int(np.sum(words(crosses['AB'][i]) != rounded['AA'][i])),
            'admittedPayloadAtAllocationBWordChanges': int(np.sum(rounded['BB'][i] != words(crosses['BA'][i]))),
            **{name: values[i] for name, values in row_panels.items()}})
    with (out/'rows.jsonl.gz').open('xb') as target:
        with gzip.GzipFile(fileobj=target, mode='wb', filename='', mtime=0) as stream:
            for record in records:
                stream.write((json.dumps(record, sort_keys=True, allow_nan=False)+'\n').encode())
    np.savez(arrays/'effects.npz', **retained)
    array_manifest = {p.name: {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in arrays.iterdir()}
    for item in items.values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Source changed during assay')
    verify(root, r)
    profiles_match = all(v['rawBitDifferences'] == v['bf16WordDifferences'] == v['retainedRawAdmissionWordDifferences'] == 0 for v in comparison.values())
    save(out/'member-access.json', {'reads': ledger, 'forbiddenMembersOpened': [], 'scope': 'Explicit trusted-code member allowlist, not an adversarial sandbox; full original NPZ bytes hashed separately.'})
    save(out/'retained-arrays.json', {'files': array_manifest, 'totalBytes': sum(x['bytes'] for x in array_manifest.values()), 'scope': 'Retained near compute; no arrays transported to development host.'})
    save(out/'report.json', {'schema': 'tce-first-read-payload-result-v1', 'requestSHA256': r['requestSHA256'], 'qualification': qualification,
        'endpointReplay': comparison, 'panels': panels, 'readWork': work,
        'crossReadProducts': sum(work[k]['scalarValueProducts'] for k in ['AB', 'BA']),
        'endpointReplayProducts': sum(work[k]['scalarValueProducts'] for k in ['AA', 'BB']),
        'rowCounts': {'total': 477, 'addressInBoth': sum(x['editedAddressInA'] and x['editedAddressInB'] for x in records),
                      'addressOnlyA': sum(x['editedAddressInA'] and not x['editedAddressInB'] for x in records),
                      'addressOnlyB': sum(x['editedAddressInB'] and not x['editedAddressInA'] for x in records),
                      'addressInNeither': sum(not x['editedAddressInA'] and not x['editedAddressInB'] for x in records),
                      'orderedSupportChanged': sum(x['orderedSupportChanged'] for x in records)},
        'sourcePreparationSeconds': ready-started, 'workerWallSecondsBeforeReport': time.perf_counter()-started,
        'workerCPUSecondsBeforeReport': time.process_time()-cpu, 'highWaterRSSKiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'runtime': {'python': sys.version, 'numpy': np.__version__, 'mathThreads': 1, 'device': 'cpu', 'policy': 'FP64perquerydot->FP32->BF16RNE'},
        'guardCanaryDenials': denied,
        'newQKProducts': 0, 'newSoftmaxEvaluations': 0, 'newNeuralModelEncoderDenoiserVAEEvaluations': 0, 'newImages': 0,
        'costScope': 'Read products are exact counts, not total FLOPs. Qualification, diagnostics (norms/dots/factorial/rounding), file hashing, member materialization and serialization add paid CPU work. Historical lexical compilation, QK/softmax and all source acquisition remain paid.',
        'mechanics_status': 'complete', 'instrument_status': 'source-endpoint-bit-exact' if profiles_match else 'finite-replay-difference-retained',
        'trend_status': 'All dependent rows, signed effects and finite/admission remainders retained without quality gate',
        'terminal_status': 'arbitrary-prompt zero-neural generation unestablished', 'scope': r['scope']})
    for name in ['request.json', 'inputs.json', 'plan.md', 'implementation.md', 'parent-reader-seal.json', 'parent-request.json', 'parent-provenance.json']:
        shutil.copyfile(root/name, out/name)
    save(out/'result-files.json', {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in out.iterdir()})
    print('FIRST_READ_PAYLOAD_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']); parser.add_argument('root'); args = parser.parse_args()
    if args.action == 'collect':
        from response_study import collect
        value = collect(args.root)
    else:
        value = globals()[args.action](args.root)
    if value is not None:
        print(json.dumps(value))
