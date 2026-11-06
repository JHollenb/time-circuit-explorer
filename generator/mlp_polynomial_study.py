"""Source-bound, finite mrun experiment for a coefficient-only MLP write."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

from response_study import read, save, sha, identity, command, collect

HERE = Path(__file__).resolve().parent
SCOPE = ('Retrospective 32-row input-line MLP compilation, native CPU BF16 reference. '
         'Offline original weights; coefficient-only serving and fixed-parent residual consumer. '
         'Supplied inputs/context, no arbitrary prompt or full-block/denoising rollout claim.')


def stage(root, mlp_source, context_source):
    from mrun.client.api import Api
    root = Path(root).resolve()
    mlp_source, context_source = Path(mlp_source).resolve(), Path(context_source).resolve()
    helper = mlp_source / 'audit-sources-v1/flux_family_custody.py'
    verify = runpy.run_path(str(helper))['verify']
    sources = []
    api = Api()

    def bind(source):
        collection = source / 'collection-01'
        custody = verify(collection)  # byte/log/archive/extraction integrity only
        job = api.json('GET', '/api/jobs/' + custody['job_id'])
        if job['state'] != 'succeeded' or job['config'].get('request_sha256') != custody['request_sha256']:
            raise ValueError('Native scheduler/request binding differs')
        report = read(collection / 'report.json')
        data = collection / 'extracted/binding-minimal-result'
        if sha(data / 'files.json') != report['files_manifest_sha256']:
            raise ValueError('Native inventory changed')
        inventory = read(data / 'files.json')
        remote = read(collection / 'recovery.json')['remote_root']
        sources.append({'custody': custody, 'payloadCustody': job['payload_custody'],
                        'reportSHA256': sha(collection / 'report.json'),
                        'inventorySHA256': sha(data / 'files.json'), 'remoteRoot': remote})

        def ref(name, kind='npy', **extras):
            path = data / name
            if sha(path) != inventory[name]:
                raise ValueError('Native operand/metadata changed: ' + name)
            return {'path': remote + '/' + name, 'sha256': inventory[name], 'kind': kind, **extras}
        return data, remote, ref

    data, mlp_remote, mref = bind(mlp_source)
    original, ctx_remote, cref = bind(context_source)
    plan = read(data / 'plan.json')
    indices = plan['selected_rows']
    if len(indices) != 32 or len(set(indices)) != 32:
        raise ValueError('Native selected rows differ')
    native = read(data / 'transaction/observations/native.json')['value']['items']['fields']['items']
    evaluation = {}
    for name in ['base', 'midpoint', 'changed']:
        desc = native[name]['items']['output']
        evaluation[name] = mref('transaction/' + desc['path'], 'bundle', offset=desc['offset'],
                               nbytes=desc['nbytes'], segmentSHA256=desc['sha256'], shape=desc['shape'])
    row = next(x for x in read(original / 'rows.json') if x['id'] == 'control')
    cutname = 'store/cuts/' + row['writer']['entry'] + '.json'
    cref(cutname)
    cut = read(original / cutname)
    page_name = 'store/pages/' + cut['slots']['image'] + '.json'
    cref(page_name)
    page = read(original / page_name)
    parent = cref('store/blobs/' + page['tensor_blob'] + '.safetensors', 'safetensors',
                  tensor=page['encoded']['name'], tensorSHA256=page['value']['sha256'], shape=page['value']['shape'])
    fields = row['writer']['fields']
    operands = {'base': mref('base.npy', shape=[1, 32, 3072]),
                'changed': mref('changed.npy', shape=[1, 32, 3072]), 'image': parent}
    for name, field in [('gateAttention', 'gate_attention'), ('gateMLP', 'gate_mlp'), ('attention', 'attention_native')]:
        source = fields[field]
        if cref(source['path'])['sha256'] != source['sha256']:
            raise ValueError('Native writer operand binding differs')
        operands[name] = cref(source['path'], shape=source['shape'])
    mref('parameter-artifacts.json')
    parameters = read(data / 'parameter-artifacts.json')
    if len(parameters['artifacts']) != 2:
        raise ValueError('Expected exactly two original MLP parameters')
    checkpoint_root = parameters['defaults']['checkpoint_root']
    checkpoint = checkpoint_root + '/' + parameters['artifacts'][0]['source']['file']
    root.mkdir(parents=True, exist_ok=False)
    payload = root / 'payload'
    payload.mkdir()
    save(payload / 'compile.json', {'schema': 'tce-mlp-line-compile-spec-v1', 'operands': operands,
         'selectedRows': indices, 'parameters': parameters, 'checkpointRoot': checkpoint_root,
         'checkpointFile': checkpoint, 'sourceRoots': [mlp_remote, ctx_remote, checkpoint_root], 'scope': SCOPE})
    save(payload / 'evaluation.json', {'native': evaluation, 'scope': 'Original isolated CPU BF16 MLP outputs; fixed U0/gM residual comparisons are reconstructed, not observed full-block midpoint exits.'})
    for name in ['mlp_polynomial.py', 'mlp_polynomial_study.py', 'response_chart.py', 'prompt_closure.py', 'response_study.py']:
        shutil.copyfile(HERE / name, payload / name)
    mref('retrieve_weights.py')
    shutil.copyfile(data / 'retrieve_weights.py', payload / 'retrieve_weights.py')
    request = {'schema': 'tce-compiled-mlp-write-request-v1', 'scope': SCOPE,
               'checkpoint': plan['checkpoint_sha256'], 'sourceJobID': 'job-f0cd6a040823',
               'sourceCustodyVerifierSHA256': sha(helper), 'sources': sources, 'alphas': [0, .5, 1],
               'methods': ['polynomial', 'endpoint_linear', 'copy_base'], 'timingRepeats': 31,
               'primaryServingUsesNativeOutputs': False, 'controlsSeparateProcess': True,
               'sourceHashes': {p.name: sha(p) for p in payload.iterdir()}}
    request['requestSHA256'] = identity(request)
    save(root / 'request.json', request)
    save(payload / 'request.json', request)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1',
         'hashed_roots': [p.name for p in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    root = Path(root).resolve()
    request = read(root / 'request.json')
    if (root / 'job.json').exists():
        raise ValueError('Already submitted; inspect existing job')
    cmd = [s.replace('response_study.py', 'mlp_polynomial_study.py') for s in command()]
    job = launch(cmd, experiment='tce-compiled-mlp-write-v1', task='analysis', payload=root / 'payload',
                 pin='beast', needs={'cpu': True},
                 config={'schema': 'tce-compiled-mlp-write-v1', 'requestSHA256': request['requestSHA256'],
                         'backend': 'numpy', 'device': 'cpu', 'plannedNeuralForwards': 0,
                         'offlineParameterCount': 84934656, 'onlineParameterMatmuls': 0},
                 timeout_s=300, detach=True, retry_on_kill=False, preflight=True, guarded=True,
                 note='Compile one32row MLP input line from2 original weight ranges; one CPU thread. Fresh serving:5 output coefficients, no matrices/NN/hidden channels.3dosesx3methods;31 timing repeats; residual consumer; archived truth only. Host-local pinned sources. Finite300s,no retries,noGPU.')
    result = {'jobID': str(job), 'requestSHA256': request['requestSHA256']}
    save(root / 'job.json', result)
    return result


def metrics(value, target, effect):
    from mlp_polynomial import np, rms, ratio
    error = value.astype(np.float64) - target
    return {'rms': rms(error), 'relativeRms': ratio(rms(error), rms(target)),
            'errorOverNativeChange': ratio(rms(error), rms(effect)),
            'maxAbs': float(np.max(np.abs(error))), 'meanError': float(error.mean())}


def execute(root):
    from mlp_polynomial import np, checked_ref, consume, rms, ratio
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
    compile_start = time.perf_counter()
    subprocess.run([sys.executable, str(root / 'mlp_polynomial.py'), 'compile', str(root / 'compile.json'), str(output)], check=True)
    compile_seconds = time.perf_counter() - compile_start
    print('MLP_COMPILED', flush=True)
    spec = read(root / 'compile.json')
    serving = {'mode': 'primary',
               'coefficients': {'path': str(output / 'coefficients.npy'), 'sha256': sha(output / 'coefficients.npy')},
               'consumer': {'path': str(output / 'consumer.npz'), 'sha256': sha(output / 'consumer.npz')},
               'sourceRoots': spec['sourceRoots'], 'alphas': request['alphas'], 'repeats': request['timingRepeats'], 'scope': SCOPE}
    save(output / 'serve.json', serving)
    serve_start = time.perf_counter()
    subprocess.run([sys.executable, str(root / 'mlp_polynomial.py'), 'serve', str(output / 'serve.json'), str(output)], check=True)
    serve_seconds = time.perf_counter() - serve_start
    seal_hash = sha(output / 'prediction-seal.json')
    seal = read(output / 'prediction-seal.json')
    print('MLP_SERVING_SEALED ' + seal_hash, flush=True)
    # The primary is sealed before any native bundle opens in this worker.
    # Hashing an endpoint's shared bundle also physically reads midpoint bytes;
    # both that read and later midpoint decoding occur after the primary seal.
    evaluation = read(root / 'evaluation.json')
    base = checked_ref(evaluation['native']['base'])
    changed = checked_ref(evaluation['native']['changed'])
    np.savez(output / 'controls.npz', base=base, changed=changed)
    control_spec = {'mode': 'controls', 'consumer': serving['consumer'],
                    'controls': {'path': str(output / 'controls.npz'), 'sha256': sha(output / 'controls.npz')},
                    'sourceRoots': spec['sourceRoots'], 'alphas': request['alphas'],
                    'repeats': request['timingRepeats'], 'scope': SCOPE}
    save(output / 'serve-controls.json', control_spec)
    control_start = time.perf_counter()
    subprocess.run([sys.executable, str(root / 'mlp_polynomial.py'), 'serve', str(output / 'serve-controls.json'), str(output)], check=True)
    control_seconds = time.perf_counter() - control_start
    control_hash = sha(output / 'control-seal.json')
    control_seal = read(output / 'control-seal.json')
    midpoint = checked_ref(evaluation['native']['midpoint'])
    targets = {0: base, .5: midpoint, 1: changed}
    with np.load(output / 'consumer.npz', allow_pickle=False) as ctx:
        parent, gate = ctx['parent'], ctx['gate']
    _, base_state = consume(parent, gate, base)
    rows, forecasts = [], {}
    all_forecasts = seal['rows'] + control_seal['rows']
    for item in all_forecasts:
        path = output / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('Sealed prediction changed')
        with np.load(path, allow_pickle=False) as fields:
            values = {k: fields[k] for k in ['raw', 'rounded', 'state']}
        forecasts[item['method'], item['alpha']] = values
        target = targets[item['alpha']]
        _, target_state = consume(parent, gate, target)
        for row in range(target.shape[1]):
            record = {'method': item['method'], 'alpha': item['alpha'], 'row': row,
                      'sourceImageRow': spec['selectedRows'][row], 'response': {}}
            for field in ['raw', 'rounded', 'state']:
                truth, effect = (target_state, target_state.astype(np.float64) - base_state) if field == 'state' else (target, target - base)
                record[field] = metrics(values[field][:, row], truth[:, row], effect[:, row])
                own_delta = values[field].astype(np.float64) - forecasts[item['method'], 0][field]
                record['response'][field] = metrics(own_delta[:, row], effect[:, row], effect[:, row])
            rows.append(record)
    aggregate = []
    for item in all_forecasts:
        value = forecasts[item['method'], item['alpha']]
        target = targets[item['alpha']]
        _, target_state = consume(parent, gate, target)
        state_effect = target_state.astype(np.float64) - base_state
        response = {}
        for field in ['raw', 'rounded', 'state']:
            effect = state_effect if field == 'state' else target - base
            own_delta = value[field].astype(np.float64) - forecasts[item['method'], 0][field]
            response[field] = metrics(own_delta, effect, effect)
        aggregate.append({'method': item['method'], 'alpha': item['alpha'], 'response': response,
                          'raw': metrics(value['raw'], target, target - base),
                          'rounded': metrics(value['rounded'], target, target - base),
                          'state': metrics(value['state'], target_state, state_effect)})
    native_curve = midpoint - (base + changed) / 2
    curvature = []
    for method in request['methods']:
        curve = forecasts[method, .5]['raw'] - (forecasts[method, 0]['raw'] + forecasts[method, 1]['raw']) / 2
        curvature.append({'method': method, 'nativeRms': rms(native_curve), 'predictedRms': rms(curve),
                          'residualRms': rms(curve - native_curve), 'residualOverNative': ratio(rms(curve - native_curve), rms(native_curve)),
                          'signedDot': float(np.mean(curve * native_curve))})
    if sha(output / 'prediction-seal.json') != seal_hash or sha(output / 'control-seal.json') != control_hash:
        raise ValueError('Prediction/control seal mutated')
    save(output / 'report.json', {'schema': 'tce-compiled-mlp-write-report-v1', 'requestSHA256': request['requestSHA256'],
         'compile': read(output / 'compile-receipt.json'), 'serving': seal, 'controls': control_seal,
         'predictionSealSHA256': seal_hash, 'controlSealSHA256': control_hash,
         'compileFreshProcessSeconds': compile_seconds, 'serveFreshProcessSeconds': serve_seconds,
         'controlsFreshProcessSeconds': control_seconds,
         'workerWallSeconds': time.perf_counter() - started, 'aggregate': aggregate, 'rows': rows, 'curvature': curvature,
         'mechanics_status': 'complete', 'terminal_status': 'unassessed', 'scope': SCOPE,
         'newNativeNeuralEvaluations': 0, 'offlineDenseLearnedWeightMath': True,
         'onlineOriginalDenseMatmuls': 0, 'onlineNeuralEvaluations': 0,
         'consumerReference': evaluation['scope']})
    print('MLP_POLYNOMIAL_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stage', 'submit', 'execute', 'collect'])
    parser.add_argument('run', type=Path)
    parser.add_argument('--mlp-source', type=Path)
    parser.add_argument('--context-source', type=Path)
    args = parser.parse_args()
    if args.action == 'stage':
        result = stage(args.run, args.mlp_source, args.context_source)
    else:
        result = globals()[args.action](args.run)
    if result is not None:
        print(json.dumps(result))
