"""Saved-array telescoping of the completed reader's signed computation defects."""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import time

from response_study import read, save, sha, identity, command, collect

HERE = Path(__file__).resolve().parent
SCOPE = ('Posthoc saved-array attribution of state-reader job; no new parameters, neural '
         'evaluations or target fitting. BF16 policy grouped across LN/modulation/projection. '
         'One supplied32row family, no arbitrary-prompt or image claim.')


def stage(root, parent):
    from mrun.client.api import Api
    root, parent = Path(root).resolve(), Path(parent).resolve()
    request, custody = read(parent / 'request.json'), read(parent / 'collection.json')
    job = Api().json('GET', '/api/jobs/' + custody['jobID'])
    if job['state'] != 'succeeded' or job['config']['requestSHA256'] != request['requestSHA256'] or job['payload_custody'] != custody['custody']:
        raise ValueError('Parent not authenticated')
    files = custody['inventory']['files']
    for name, item in files.items():
        path = parent / 'result' / name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Parent collection changed: ' + name)
    remote = job['result']['diagnostics']['cwd'] + '/response-result'

    def ref(name):
        return {'path': remote + '/' + name, 'sha256': files[name]['sha256']}

    report, serving = read(parent / 'result/report.json'), read(parent / 'result/serve.json')
    packets = []
    for item in read(parent / 'result/packet-map.json'):
        pid = item['id']
        prediction = next(p for p in report['serving']['rows'] if p['id'] == pid)
        diag = next(p for p in report['native']['diagnostics'] if p['id'] == pid)
        native = {p['id']: p for p in report['native']['records']}
        state = next(p for p in serving['packets'] if p['id'] == pid)
        packets.append({'alpha': item['alpha'], 'prediction': ref(prediction['output']),
                        'smooth': ref(diag['path']), 'sameInput': ref(native[pid]['path']),
                        'nativeWrite': ref(native[item['nativeID']]['path']), 'state': ref(Path(state['path']).name)})
    root.mkdir(parents=True, exist_ok=False)
    payload = root / 'payload'
    payload.mkdir()
    for name in ['state_reader_attribution.py', 'response_study.py', 'mlp_polynomial.py', 'response_chart.py', 'prompt_closure.py']:
        shutil.copyfile(HERE / name, payload / name)
    spec = {'schema': 'tce-state-reader-attribution-request-v1', 'packets': packets, 'sourceRoot': remote,
            'selectedRows': request['selectedRows'], 'scope': SCOPE,
            'parent': {'jobID': job['job_id'], 'requestSHA256': request['requestSHA256'],
                       'report': ref('report.json'), 'collectionSHA256': sha(parent / 'collection.json'),
                       'payloadCustody': job['payload_custody']},
            'sourceHashes': {p.name: sha(p) for p in payload.iterdir()}}
    spec['requestSHA256'] = identity(spec)
    save(root / 'request.json', spec)
    save(payload / 'request.json', spec)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1',
         'hashed_roots': [p.name for p in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': spec['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    root = Path(root).resolve()
    request = read(root / 'request.json')
    if (root / 'job.json').exists():
        raise ValueError('Already submitted')
    cmd = [v.replace('response_study.py', 'state_reader_attribution.py') for v in command()]
    job = launch(cmd, experiment='tce-state-reader-attribution-v1', task='analysis', payload=root / 'payload',
                 pin='beast', needs={'cpu': True},
                 config={'schema': request['schema'], 'requestSHA256': request['requestSHA256'],
                         'backend': 'numpy', 'device': 'cpu', 'newNeuralEvaluations': 0, 'originalWeightReads': 0},
                 timeout_s=120, detach=True, retry_on_kill=False, preflight=True, guarded=True,
                 note='Saved-array reader defect telescope O+S+N+H,3dosesxQKV,32rows; paired+absolute signed powers/cross terms, delta state residual. CPU1,no weights/no native calls/no fitting. Host-local immutable completed parent. Finite120s,no retries,noGPU.')
    result = {'jobID': str(job), 'requestSHA256': request['requestSHA256']}
    save(root / 'job.json', result)
    return result


def telescope(components, total, effect):
    from mlp_polynomial import np, rms, ratio
    powers = {k: float(np.mean(v * v)) for k, v in components.items()}
    pairs = {}
    for first, second in itertools.combinations(components, 2):
        a, b = components[first], components[second]
        dot = float(np.mean(a * b))
        pairs[first + ':' + second] = {'crossPower': 2 * dot, 'cosine': ratio(dot, rms(a) * rms(b))}
    total_power = float(np.mean(total * total))
    return {'totalRms': rms(total), 'nativeEffectRms': rms(effect),
            'totalOverEffect': ratio(rms(total), rms(effect)),
            'components': {k: {'rms': rms(v), 'overEffect': ratio(rms(v), rms(effect)), 'power': powers[k]} for k, v in components.items()},
            'pairs': pairs, 'totalPower': total_power,
            'identityMaxAbs': float(np.max(np.abs(total - sum(components.values())))),
            'powerClosureAbs': abs(total_power - sum(powers.values()) - sum(p['crossPower'] for p in pairs.values()))}


def execute(root):
    from mlp_polynomial import np, rms, ratio, quantize, audit_reads, peak_mib
    root = Path(root).resolve()
    request = read(root / 'request.json')
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']:
        raise ValueError('Request changed')
    for name, digest in request['sourceHashes'].items():
        if sha(root / name) != digest:
            raise ValueError('Analysis source changed')
    output = root / 'response-result'
    output.mkdir(exist_ok=False)
    allowed = [root / 'request.json', output / 'report.json', output / 'observations.jsonl']
    allowed += [ref['path'] for p in request['packets'] for ref in p.values() if isinstance(ref, dict)]
    opened = audit_reads(allowed, [request['sourceRoot'], str(root)], serving=True)
    started, cpu = time.perf_counter(), time.process_time()

    def load(ref):
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('Saved diagnostic operand changed')
        value = np.load(ref['path'], allow_pickle=False)
        if isinstance(value, np.ndarray):
            return value.astype(np.float64)
        with value as archive:
            return {k: archive[k].astype(np.float64) for k in archive.files}

    data = {p['alpha']: {k: load(v) for k, v in p.items() if isinstance(v, dict)} for p in request['packets']}
    baseline, aggregate, rows, residuals, counterfactuals = data[0], [], [], [], []
    for alpha, packet in data.items():
        p, s, c, t = [packet[k] for k in ['prediction', 'smooth', 'sameInput', 'nativeWrite']]
        state = packet['state'];state0 = baseline['state']
        delta_v = (state-state.mean(axis=-1,keepdims=True))-(state0-state0.mean(axis=-1,keepdims=True))
        delta_r = p['residual']-baseline['prediction']['residual']
        residuals.append({'alpha': alpha, 'deltaResidualRms': rms(delta_r), 'deltaCenteredStateRms': rms(delta_v),
                          'ratio': ratio(rms(delta_r), rms(delta_v)),
                          'perRowRatio': [ratio(rms(delta_r[:, i]), rms(delta_v[:, i])) for i in range(32)]})
        for field in ['q', 'k', 'v']:
            def terms(item):
                pred, smooth, same, native = [item[k] for k in ['prediction','smooth','sameInput','nativeWrite']]
                return {'outputRounding': pred[field]-pred[field+'_raw'],
                        'basisOmission': pred[field+'_raw']-smooth[field+'_smooth'],
                        'numericPolicy': smooth[field+'_smooth']-same[field],
                        'inheritedWriter': same[field]-native[field]}
            absolute, zero = terms(packet), terms(baseline)
            effect = t[field]-baseline['nativeWrite'][field]
            for comparison in ['absolute', 'paired']:
                comp = absolute if comparison == 'absolute' else {k:v-zero[k] for k,v in absolute.items()}
                total = p[field]-t[field]
                if comparison == 'paired':
                    total -= baseline['prediction'][field]-baseline['nativeWrite'][field]
                meta = {'alpha': alpha, 'field': field, 'comparison': comparison, 'decomposition': 'four-term-composite'}
                aggregate.append({**meta, **telescope(comp,total,effect)})
                for i, source_row in enumerate(request['selectedRows']):
                    rows.append({**meta, 'row': i, 'sourceImageRow': source_row,
                                 **telescope({k:v[:,i] for k,v in comp.items()},total[:,i],effect[:,i])})
            # Offline full-projection counterfactual through the SAME final quantizer.
            full = quantize(s[field+'_smooth']).astype(np.float64)
            full0 = quantize(baseline['smooth'][field+'_smooth']).astype(np.float64)
            a = {'basisThroughQuantizer':p[field]-full, 'remainingNumericPolicy':full-c[field]}
            z = {'basisThroughQuantizer':baseline['prediction'][field]-full0,
                 'remainingNumericPolicy':full0-baseline['sameInput'][field]}
            for comparison in ['absolute','paired']:
                comp = a if comparison=='absolute' else {k:v-z[k] for k,v in a.items()}
                total=p[field]-c[field]
                if comparison=='paired':
                    total-=baseline['prediction'][field]-baseline['sameInput'][field]
                meta={'alpha':alpha,'field':field,'comparison':comparison,'decomposition':'quantized-full-projection-reader'}
                counterfactuals.append({**meta,**telescope(comp,total,effect)})
                for i,source_row in enumerate(request['selectedRows']):
                    rows.append({**meta,'row':i,'sourceImageRow':source_row,
                                 **telescope({k:v[:,i] for k,v in comp.items()},total[:,i],effect[:,i])})
    with (output / 'observations.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    save(output / 'report.json', {'schema':'tce-state-reader-attribution-report-v1', 'requestSHA256':request['requestSHA256'],
         'scope':SCOPE, 'parent':request['parent'], 'aggregate':aggregate, 'roundedCounterfactual':counterfactuals,
         'counterfactualScope':'BF16(full smooth original projection), computed from saved diagnostic only; not a deployable reduced receiver. All ratios use native-write T own-baseline change.',
         'rows':len(rows), 'residuals':residuals,
         'observationSHA256':sha(output/'observations.jsonl'), 'wallSecondsAfterImports':time.perf_counter()-started,
         'cpuSecondsAfterImports':time.process_time()-cpu, 'peakRSSMiB':peak_mib(), 'opened':sorted(opened),
         'newNeuralEvaluations':0, 'newOriginalDenseMatmuls':0, 'originalWeightBytesRead':0,
         'mechanics_status':'complete', 'terminal_status':'unassessed'})
    print('STATE_READER_ATTRIBUTION_COMPLETE',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['stage','submit','execute','collect'])
    parser.add_argument('run',type=Path)
    parser.add_argument('--parent',type=Path)
    args=parser.parse_args()
    result=stage(args.run,args.parent) if args.action=='stage' else globals()[args.action](args.run)
    if result is not None:
        print(json.dumps(result))
