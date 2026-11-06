"""One finite, source-bound compiled writer -> actual-state QKV reader assay."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from response_study import read, save, sha, identity, collect

HERE = Path(__file__).resolve().parent
SCOPE = ('Supplied 32-row producer family at diffusion step2/joint1 -> joint2 image reader. '
         'State-fed pre-head-normalization, pre-RoPE Q/K/V affines only; fixed captured modulation. '
         'No attention, new prompt, full-block, denoising continuation or RGB claim.')
NATIVE_SOURCE = {'path': '/mnt/hdd8tb/mdb/uv-cache/archive-v0/B1KJlciU-SoDheiW/lib/python3.14/site-packages/diffusers/models/transformers/transformer_flux2.py',
                 'sha256': '0a3c56ed3ca28a416492b50953a433f20b8a89fd26c7b7a49b1b6b45d5236006'}


def stage(root, producer, context):
    from mrun.client.api import Api
    root, producer, context = [Path(p).resolve() for p in [root, producer, context]]
    prior, collection = read(producer / 'request.json'), read(producer / 'collection.json')
    if identity({k: v for k, v in prior.items() if k != 'requestSHA256'}) != prior['requestSHA256']:
        raise ValueError('Producer request changed')
    job = Api().json('GET', '/api/jobs/' + collection['jobID'])
    if job['state'] != 'succeeded' or job['config'].get('requestSHA256') != prior['requestSHA256']:
        raise ValueError('Producer job/request not authenticated')
    custody = job['payload_custody']
    if custody != collection['custody'] or not custody['required']:
        raise ValueError('Producer guarded custody differs')
    inventory = collection['inventory']['files']
    for name, item in inventory.items():
        path = producer / 'result' / name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Collected producer changed: ' + name)
    for name in ['compile.json', 'evaluation.json']:
        if sha(producer / 'payload' / name) != prior['sourceHashes'][name]:
            raise ValueError('Producer input manifest changed')
    old_spec = read(producer / 'payload/compile.json')
    remote = job['result']['diagnostics']['cwd'] + '/response-result'

    def pref(name):
        return {'path': remote + '/' + name, 'sha256': inventory[name]['sha256']}

    source = prior['sources'][1]
    if sha(context / 'files.json') != source['inventorySHA256']:
        raise ValueError('Context inventory no longer matches authenticated producer')
    files = read(context / 'files.json')

    def cref(name):
        if sha(context / name) != files[name]:
            raise ValueError('Context metadata/blob changed: ' + name)
        return {'path': source['remoteRoot'] + '/' + name, 'sha256': files[name]}

    cref('rows.json')
    row = next(r for r in read(context / 'rows.json') if r['id'] == 'control')
    cutname = 'store/cuts/' + row['writer']['exit'] + '.json'
    cutref = cref(cutname)
    cut = read(context / cutname)
    if cut['execution_point']['next_operation'] != 'joint_block.2' or cut['execution_point']['logical_step'] != 2:
        raise ValueError('Producer is not at the selected next reader')
    page_name = 'store/pages/' + cut['slots']['context'] + '.json'
    page_ref = cref(page_name)
    page = read(context / page_name)
    desc = next(i['value'] for i in page['value']['items'] if i['key']['value'] == 'img_mod')
    modulation = {**cref('store/blobs/' + page['tensor_blob'] + '.safetensors'), 'kind': 'safetensors',
                  'tensor': page['encoded']['items']['img_mod']['name'],
                  'shape': desc['shape'], 'tensorSHA256': desc['sha256']}
    kernel = read(HERE / 'kernel-manifest.json')
    for name in ['parameter-artifacts.json', 'retrieve_weights.py', 'bindings.json']:
        if sha(HERE / name) != kernel['files'][name]:
            raise ValueError('Original recipe source changed')
    parameters = read(HERE / 'parameter-artifacts.json')
    parameters['artifacts'] = [a for a in parameters['artifacts'] if a['source']['tensor'] in
                              [f'transformer_blocks.2.attn.to_{c}.weight' for c in ['q', 'k', 'v']]]
    if len(parameters['artifacts']) != 3:
        raise ValueError('Expected exactly three original QKV recipes')
    root.mkdir(parents=True, exist_ok=False)
    payload = root / 'payload'
    payload.mkdir()
    compile_spec = {'schema': 'tce-state-receiver-compile-spec-v1', 'coefficients': pref('coefficients.npy'),
                    'consumer': pref('consumer.npz'), 'modulation': modulation, 'parameters': parameters,
                    'checkpointRoot': old_spec['checkpointRoot'], 'checkpointFile': old_spec['checkpointFile'],
                    'sourceRoots': [*old_spec['sourceRoots'], remote], 'scope': SCOPE}
    seal = read(producer / 'result/prediction-seal.json')
    packets = [{**pref(item['path']), 'alpha': item['alpha']} for item in seal['rows']]
    if sorted(p['alpha'] for p in packets) != [0, .5, 1]:
        raise ValueError('Producer packet family differs')
    save(payload / 'compile.json', compile_spec)
    save(payload / 'packet-source.json', {'packets': packets, 'producerSeal': pref('prediction-seal.json')})
    shutil.copyfile(producer / 'payload/evaluation.json', payload / 'evaluation.json')
    for name in ['state_receiver.py', 'state_receiver_native.py', 'state_receiver_study.py',
                 'mlp_polynomial.py', 'response_chart.py', 'prompt_closure.py', 'response_study.py', 'retrieve_weights.py']:
        shutil.copyfile(HERE / name, payload / name)
    request = {'schema': 'tce-state-receiver-request-v1', 'scope': SCOPE,
               'producer': {'jobID': job['job_id'], 'requestSHA256': prior['requestSHA256'],
                            'collectionSHA256': sha(producer / 'collection.json'), 'custody': custody,
                            'predictionSeal': pref('prediction-seal.json'), 'sources': prior['sources']},
               'boundary': {'cut': cutref, 'contextPage': page_ref, 'executionPoint': cut['execution_point']},
               'selectedRows': old_spec['selectedRows'], 'nativeSource': NATIVE_SOURCE, 'eps': 1e-6,
               'timingRepeats': 31, 'nativeReceiverEvaluationsAfterSeal': 6,
               'servingReceivesDose': False, 'servingNativeRefresh': False,
               'sourceHashes': {p.name: sha(p) for p in payload.iterdir()}}
    request['requestSHA256'] = identity(request)
    save(root / 'request.json', request)
    save(payload / 'request.json', request)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1',
         'hashed_roots': [p.name for p in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root = Path(root).resolve()
    request = read(root / 'request.json')
    if (root / 'job.json').exists():
        raise ValueError('Already submitted; inspect existing job')
    cmd = command()
    cmd[1:1] = ['OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=']
    cmd[-5:] = ['python', '-u', 'state_receiver_study.py', 'execute', '.']
    job = launch(cmd, experiment='tce-state-receiver-v1', task='analysis', payload=root / 'payload',
                 pin='beast', needs={'cpu': True},
                 config={'schema': 'tce-state-receiver-v1', 'requestSHA256': request['requestSHA256'],
                         'backend': 'numpy-and-postseal-torch-cpu', 'device': 'cpu',
                         'onlineNeuralEvaluations': 0, 'offlineDenseCompilation': True,
                         'postSealNativeReceiverEvaluations': 6, 'physicalShape': [1, 32, 3072]},
                 timeout_s=300, detach=True, retry_on_kill=False, preflight=True, guarded=True,
                 note='Own-state reader:3 actual BF16 producer packets, no dose. CPU1 compile3QKV banks from host-local weights; fresh NumPy serving seals before6 CPU native LN+3Linear refs. Pre-RoPE only; residual/error decomposition. No encoder/VAE/attention/GPU. Finite300s,no retries.')
    result = {'jobID': str(job), 'requestSHA256': request['requestSHA256']}
    save(root / 'job.json', result)
    return result


def decompose(predicted, same_input, target, effect):
    from mlp_polynomial import np, rms, ratio
    error, reader, inherited = predicted - target, predicted - same_input, same_input - target
    return {'totalRms': rms(error), 'readerRms': rms(reader), 'inheritedRms': rms(inherited),
            'nativeEffectRms': rms(effect), 'totalOverEffect': ratio(rms(error), rms(effect)),
            'readerOverEffect': ratio(rms(reader), rms(effect)), 'inheritedOverEffect': ratio(rms(inherited), rms(effect)),
            'totalOverTarget': ratio(rms(error), rms(target)),
            'totalPower': float(np.mean(error * error)), 'readerPower': float(np.mean(reader * reader)),
            'inheritedPower': float(np.mean(inherited * inherited)),
            'crossPower': float(2 * np.mean(reader * inherited)),
            'readerInheritedCosine': ratio(float(np.mean(reader * inherited)), rms(reader) * rms(inherited)),
            'decompositionMaxAbs': float(np.max(np.abs(error - reader - inherited)))}


def score(output, request, packet_map, seal, native):
    from mlp_polynomial import np, rms, ratio

    def load(item, key):
        path = output / item[key]
        if sha(path) != item['sha256']:
            raise ValueError('Output changed after seal/reference')
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k].astype(np.float64) for k in data.files}

    primary, references, smooth = {}, {}, {}
    for item in seal['rows']:
        primary[item['id']] = load(item, 'output')
    for item in native['records']:
        references[item['id']] = load(item, 'path')
    for item in native['diagnostics']:
        smooth[item['id']] = load(item, 'path')
    by_alpha = {item['alpha']: item for item in packet_map}
    p0 = primary[by_alpha[0]['id']]
    c0 = references[by_alpha[0]['id']]
    t0 = references[by_alpha[0]['nativeID']]
    observations, aggregate, diagnostic = [], [], []
    for alpha, packet in by_alpha.items():
        p, c, t = primary[packet['id']], references[packet['id']], references[packet['nativeID']]
        d = smooth[packet['id']]
        for field in ['q', 'k', 'v']:
            effect = t[field] - t0[field]
            for view in ['rounded', 'raw']:
                key = field if view == 'rounded' else field + '_raw'
                for comparison in ['absolute', 'paired']:
                    values = (p[key], c[field], t[field]) if comparison == 'absolute' else (
                              p[key] - p0[key], c[field] - c0[field], effect)
                    meta = {'alpha': alpha, 'field': field, 'view': view, 'comparison': comparison}
                    aggregate.append({**meta, **decompose(*values, effect)})
                    for i, source_row in enumerate(request['selectedRows']):
                        observations.append({**meta, 'row': i, 'sourceImageRow': source_row,
                                             **decompose(*(v[:, i] for v in values), effect[:, i])})
            omitted = d[field + '_residual']
            identity_error = d[field + '_smooth'] - p[field + '_raw'] - omitted
            record = {'alpha': alpha, 'field': field, 'projectedResidualRms': rms(omitted),
                      'smoothIdentityMaxAbs': float(np.max(np.abs(identity_error))),
                      'smoothVsNativeRms': rms(d[field + '_smooth'] - c[field]),
                      'outputRoundingRms': rms(p[field] - p[field + '_raw']),
                      'zeroResponseErrorOverEffect': ratio(rms(effect), rms(effect)),
                      'perRowProjectedResidualRms': [rms(omitted[:, i]) for i in range(32)],
                      'perRowSmoothVsNativeRms': [rms(d[field + '_smooth'][:, i] - c[field][:, i]) for i in range(32)]}
            diagnostic.append(record)
    return aggregate, observations, diagnostic


def execute(root):
    from mlp_polynomial import np, checked_ref, consume
    root = Path(root).resolve()
    request = read(root / 'request.json')
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']:
        raise ValueError('Request changed')
    for name, digest in request['sourceHashes'].items():
        if sha(root / name) != digest:
            raise ValueError('Staged source changed: ' + name)
    output = root / 'response-result'
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    compile0 = time.perf_counter()
    subprocess.run([sys.executable, str(root / 'state_receiver.py'), 'compile', str(root / 'compile.json'), str(output)], check=True)
    compile_s = time.perf_counter() - compile0
    print('STATE_READER_COMPILED', flush=True)
    spec = read(root / 'compile.json')
    source = read(root / 'packet-source.json')
    if sha(source['producerSeal']['path']) != source['producerSeal']['sha256']:
        raise ValueError('Original producer seal changed')
    packets, packet_map = [], []
    for packet in source['packets']:
        if sha(packet['path']) != packet['sha256']:
            raise ValueError('Original producer output changed')
        opaque = identity({'request': request['requestSHA256'], 'source': packet['sha256']})[:16]
        path = output / ('state-' + opaque + '.npy')
        with np.load(packet['path'], allow_pickle=False) as data:
            np.save(path, data['state'], allow_pickle=False)
        packets.append({'id': opaque, 'path': str(path), 'sha256': sha(path), 'shape': [1, 32, 3072]})
        packet_map.append({'id': opaque, 'nativeID': opaque + '-native', 'alpha': packet['alpha'], 'producerArchiveSHA256': packet['sha256']})
    save(output / 'packet-map.json', packet_map)
    compiled = read(output / 'receiver-compile-receipt.json')
    serving = {'schema': 'tce-state-only-reader-input-v1',
               'basis': {'path': str(output / 'basis.npy'), 'sha256': compiled['basisSHA256']},
               'banks': {name: {'path': str(output / item['path']), 'sha256': item['sha256']} for name, item in compiled['banks'].items()},
               'packets': packets, 'eps': request['eps'], 'repeats': request['timingRepeats'],
               'scope': SCOPE, 'sourceRoots': [*spec['sourceRoots'], str(root)]}
    save(output / 'serve.json', serving)
    serve0 = time.perf_counter()
    subprocess.run([sys.executable, str(root / 'state_receiver.py'), 'serve', str(output / 'serve.json'), str(output)], check=True)
    serve_s = time.perf_counter() - serve0
    seal_path = output / 'receiver-seal.json'
    seal_hash, seal = sha(seal_path), read(seal_path)
    print('STATE_READER_SEALED ' + seal_hash, flush=True)
    # Neither the archived native MLP bundle nor native receiver execution occurs before this seal.
    evaluation = read(root / 'evaluation.json')
    native_spec = {key: spec[key] for key in ['parameters', 'checkpointRoot', 'checkpointFile', 'scope']}
    native_spec.update(seal={'path': str(seal_path), 'sha256': seal_hash}, source=request['nativeSource'],
                       eps=request['eps'], modulation={'path': str(output / 'modulation.npz'), 'sha256': sha(output / 'modulation.npz')})
    native_packets = []
    with np.load(spec['consumer']['path'], allow_pickle=False) as data:
        parent, gate = data['parent'], data['gate']
    labels = {0: 'base', .5: 'midpoint', 1: 'changed'}
    for packet, mapping in zip(packets, packet_map):
        prediction = next(r for r in seal['rows'] if r['id'] == packet['id'])
        native_packets.append({**packet, 'arm': 'predicted', 'prediction': {'path': prediction['output'], 'sha256': prediction['sha256']}})
        target_write = checked_ref(evaluation['native'][labels[mapping['alpha']]])
        _, target_state = consume(parent, gate, target_write)
        path = output / ('state-' + mapping['nativeID'] + '.npy')
        np.save(path, target_state, allow_pickle=False)
        native_packets.append({'id': mapping['nativeID'], 'path': str(path), 'sha256': sha(path),
                               'shape': [1, 32, 3072], 'arm': 'native-write'})
    native_spec['packets'] = native_packets
    save(output / 'native-spec.json', native_spec)
    native0 = time.perf_counter()
    subprocess.run([sys.executable, str(root / 'state_receiver_native.py'), str(output / 'native-spec.json'), str(output)], check=True)
    native_s = time.perf_counter() - native0
    native = read(output / 'native-receipt.json')
    aggregate, observations, diagnostics = score(output, request, packet_map, seal, native)
    with (output / 'observations.jsonl').open('x') as stream:
        for row in observations:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
    if sha(seal_path) != seal_hash:
        raise ValueError('Scoring mutated primary seal')
    save(output / 'report.json', {'schema': 'tce-state-receiver-report-v1', 'requestSHA256': request['requestSHA256'],
         'scope': SCOPE, 'compile': compiled, 'serving': seal, 'native': native,
         'predictionSealSHA256': seal_hash, 'compileFreshProcessSeconds': compile_s,
         'serveFreshProcessSeconds': serve_s, 'nativeFreshProcessSeconds': native_s,
         'workerWallSeconds': time.perf_counter() - started, 'aggregate': aggregate, 'diagnostics': diagnostics,
         'observations': {'path': 'observations.jsonl', 'sha256': sha(output / 'observations.jsonl'), 'rows': len(observations)},
         'mechanics_status': 'complete', 'instrument_status': 'bounded CPU native primitives; CUDA parity unmeasured',
         'terminal_status': 'unassessed', 'onlineNeuralEvaluations': 0, 'servingDoseInput': False,
         'offlineOriginalDenseWeightCompilation': True, 'postSealNativeReceiverEvaluations': len(native['records']),
         'targetStateScope': evaluation['scope']})
    print('STATE_READER_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stage', 'submit', 'execute', 'collect'])
    parser.add_argument('run', type=Path)
    parser.add_argument('--producer', type=Path)
    parser.add_argument('--context', type=Path)
    args = parser.parse_args()
    result = stage(args.run, args.producer, args.context) if args.action == 'stage' else globals()[args.action](args.run)
    if result is not None:
        print(json.dumps(result))
