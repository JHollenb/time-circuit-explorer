"""One saved-only guarded CPU discovery/transfer of integer address programs."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OWNED = ['token_history_address.py', 'token_history_process.py', 'token_history_study.py', 'test_token_history_address.py']
PARENTS = [('uncentered-reader', 'discovery', '8f9b0eb30195036e3791ec11099373d9356a1049b3d08590b233b4c55adff8f5'),
    ('fresh-prompt-square', 'transfer', '0fcc4fec349612e840b6bcc3f3ab452c649d5db73a5289aed7fdc0256225158a')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)


def identity(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify(root, request):
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']:
        raise ValueError('Request identity differs')
    for name, digest in request['sourceHashes'].items():
        if sha(Path(root)/name) != digest:
            raise ValueError('Frozen payload differs: '+name)


def stage(root):
    from mrun.client.api import Api
    api = Api()
    cases = []
    parent_rows = []
    snapshots = {}
    for folder, split, digest in PARENTS:
        parent = HERE.parent/'docs/results'/folder
        if sha(parent/'reader-seal.json') != digest:
            raise ValueError('Historical reader seal changed')
        seal = read(parent/'reader-seal.json')
        provenance = read(parent/'provenance.json')
        current = api.json('GET', '/api/jobs/'+provenance['jobID'])
        if current['state'] != 'succeeded' or current['payload_custody'] != provenance['payloadCustody']:
            raise ValueError('Historical job/current custody differs')
        if seal['requestSHA256'] != provenance['requestSHA256'] or seal['policy'] != 'current-top4-uncentered-fp64-fp32-bf16-v1':
            raise ValueError('Historical reader contract differs')
        parents = {row['id']: row for row in seal['reader']['rows']}
        for row in seal['lexicalRows']:
            cases.append({'id': row['id'], 'split': split, 'prompt': row['prompt'], 'activeTokens': row['activeTokens'],
                'lexical': {k: row[k] for k in ['path', 'sha256', 'idsPayloadSHA256', 'maskPayloadSHA256']},
                'selected': parents[row['id']]})
        parent_rows.append({'folder': folder, 'jobID': provenance['jobID'], 'requestSHA256': seal['requestSHA256'],
            'readerSealSHA256': digest, 'payloadCustody': current['payload_custody']})
        for name in ['reader-seal.json', 'request.json', 'provenance.json']:
            snapshots[folder+'-'+name] = parent/name
    if [a['id'] for a in cases] != ['A', 'B', '00', '10', '01', '11']:
        raise ValueError('Frozen six-case split differs')
    files = [{k: item[k] for k in ['path', 'sha256']} for row in cases for item in [row['lexical'], row['selected']]]
    code = """import hashlib,json,stat
from pathlib import Path
items=json.loads(%r)
rows=[]
for item in items:
 p=Path(item['path']);s=p.lstat()
 if not stat.S_ISREG(s.st_mode):raise ValueError('Source is not a regular file')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 if h.hexdigest()!=item['sha256']:raise ValueError('Retained source SHA differs')
 rows.append({'path':str(p),'bytes':s.st_size,'sha256':h.hexdigest()})
print(json.dumps({'files':rows,'bytes':sum(r['bytes'] for r in rows),'scope':'Fresh full-byte authentication, no numerical-array interpretation.'}))
""" % json.dumps(files)
    intake = json.loads(subprocess.run(['ssh', '-T', 'beast', 'python3', '-'], input=code, text=True, capture_output=True, check=True, timeout=60).stdout)
    host = next(a for a in api.json('GET', '/api/hosts') if a['name'] == 'beast')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    payload = root/'payload'
    payload.mkdir()
    for name in OWNED:
        shutil.copyfile(HERE/name, payload/name)
    for name, source in snapshots.items():
        shutil.copyfile(source, payload/name)
    shutil.copyfile(HERE.parent/'docs/plans/token-history-address-program.md', payload/'frozen-plan.md')
    save(payload/'inputs.json', {'cases': cases, 'parents': parent_rows, 'intake': intake})
    save(payload/'resource-basis.json', {'sourceNPZFiles': len(files), 'sourceNPZBytes': intake['bytes'],
        'promotedMembers': ['ids', 'mask', 'positions', 'supportPositions', 'selectedWeights', 'prefixCounts'],
        'servingMembers': ['ids', 'mask', 'positions'], 'fitMembers': ['discovery IDs/mask/positions', 'discovery supportPositions'],
        'workingSet': 'One source member at a time; six small clean token packets, one selected support/weight bank during scoring, six address/trace packets per case. No hidden, Q/K/V, raw head vectors or model parameters loaded.',
        'liveHost': {k: host.get(k) for k in ['name', 'active_jobs', 'admission', 'telemetry', 'ram_total_mb']},
        'admission': 'Intent-first CPU, no hand reservation/thread/wall estimate. Hard pin required by retained host-local files. Numerical-library threads1, no CUDA/model acquisition, finite300s/no retries.'})
    r = {'schema': 'tce-token-history-address-request-v1', 'ownedSources': OWNED,
        'sourceHashes': {p.name: sha(p) for p in payload.iterdir()}, 'device': 'cpu', 'backend': 'stdlib-integer-program; numpy-saved-member-intake-and-scoring',
        'plannedNeuralEvaluations': 0, 'plannedLearnedMatrixMACs': 0, 'timeoutSeconds': 300,
        'scope': 'Retrospective transfer from two historical prompts to four historical prompts at our approximate top-four first-encoder support port. No native semantic dispatch, new consumer/image, general contextual encoder or arbitrary-prompt zero-neural claim.'}
    r['requestSHA256'] = identity(r)
    save(payload/'request.json', r)
    save(root/'request.json', r)
    save(payload/'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'root': str(root), 'requestSHA256': r['requestSHA256'], 'sourceFiles': len(files), 'freshFullSHABytes': intake['bytes']}


def submit(root):
    from mrun.client.submit import launch
    root = Path(root).resolve()
    r = read(root/'request.json')
    verify(root/'payload', r)
    if (root/'job.json').exists():
        raise ValueError('Already submitted; inspect the same handle')
    cmd = ['/usr/bin/env', 'PYTHONUNBUFFERED=1', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONNOUSERSITE=1', 'PYTHONPATH=',
        'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache', 'UV_LINK_MODE=symlink',
        'uv', 'run', '--no-project', '--offline', '--python', '3.14.6', '--with', 'numpy>=2.0',
        'python', '-u', 'token_history_study.py', 'execute', '.']
    job = launch(cmd, experiment='tce-token-history-address-v1', task='analysis', payload=root/'payload', pin='beast', needs={'cpu': True},
        config={k: r[k] for k in ['schema', 'requestSHA256', 'device', 'backend', 'plannedNeuralEvaluations', 'plannedLearnedMatrixMACs']},
        timeout_s=300, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='Saved-only token-history address program:2discovery/4retrospective-transfer prompts;32heads,512queries,4slots,6contrasts. Fresh integer compiler/serving children; targets sealed out until score. No QKV/hidden/weights/models/GPU/consumer/image. Host-local12NPZs fullSHA; intentfirstCPU,BLAS1,300scap,noretries. Preserve every support/trace/weighted-coverage row; no quality gate.')
    result = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}
    save(root/'job.json', result)
    return result


def child(root, action, spec, output):
    spec_path = root/(action+'-spec.json')
    sources = [root/name for name in ['token_history_address.py', 'token_history_process.py']]
    spec['readAllow'] = sorted(set(spec['readAllow']+[str(spec_path), *map(str, sources)]))
    save(spec_path, spec)
    tick = time.perf_counter()
    subprocess.run([sys.executable, '-I', '-B', str(root/'token_history_process.py'), action, str(spec_path), str(output)], check=True, cwd=root)
    return {'processWallSeconds': time.perf_counter()-tick, 'receipt': read(output/'receipt.json'), 'specSHA256': sha(spec_path)}


def execute(root):
    import numpy as np
    from token_history_address import validate_packet
    from token_history_process import write_rows
    root = Path(root).resolve()
    r = read(root/'request.json')
    verify(root, r)
    s = read(root/'inputs.json')
    start = time.perf_counter()
    cpu = time.process_time()
    clean = root/'clean'
    clean.mkdir()
    accesses = []

    def member(item, name, phase):
        with np.load(item['path'], allow_pickle=False) as packet:
            value = packet[name]
        accesses.append({'path': item['path'], 'member': name, 'phase': phase, 'shape': list(value.shape), 'dtype': str(value.dtype)})
        return value

    tokens = {}
    inputs = []
    for case in s['cases']:
        for item in [case['lexical'], case['selected']]:
            if sha(item['path']) != item['sha256']:
                raise ValueError('Saved source bytes differ')
        raw = {k: member(case['lexical'], k, 'raw-token-intake') for k in ['ids', 'mask', 'positions']}
        if any(a.shape != (512,) or a.dtype != np.dtype('int64') for a in raw.values()):
            raise ValueError('Qualified lexical integer geometry differs')
        for name in ['ids', 'mask']:
            if hashlib.sha256(raw[name].tobytes()).hexdigest() != case['lexical'][name+'PayloadSHA256']:
                raise ValueError('Saved lexical integer payload differs')
        packet = {k: a.tolist() for k, a in raw.items()}
        validate_packet(packet)
        if sum(packet['mask']) != case['activeTokens']:
            raise ValueError('Active token count differs')
        token_path = clean/(case['id']+'-tokens.json')
        save(token_path, packet)
        tokens[case['id']] = packet
        inputs.append({'id': case['id'], 'tokens': str(token_path)})
        if case['split'] == 'discovery':
            supports = member(case['selected'], 'supportPositions', 'discovery-label-intake')
            if supports.shape != (512, 32, 4) or supports.dtype != np.dtype('int64'):
                raise ValueError('Discovery support geometry differs')
            for position, row in enumerate(supports.tolist()):
                prefix = [p for p in range(position+1) if packet['mask'][p]]
                for addresses in row:
                    chosen = [p for p in addresses if p >= 0]
                    if len(chosen) != min(4, len(prefix)) or len(set(chosen)) != len(chosen) or any(p not in prefix for p in chosen):
                        raise ValueError('Discovery label violated causal support contract')
            save(clean/(case['id']+'-supports.json'), supports.tolist())
    intake_done = time.perf_counter()
    forbidden = [str(root/'request.json')]+[a[k]['path'] for a in s['cases'] for k in ['lexical', 'selected']]
    fit_cases = [{**a, 'supports': str(clean/(a['id']+'-supports.json'))} for a in inputs if a['id'] in ['A', 'B']]
    compile_result = child(root, 'compile', {'cases': fit_cases,
        'readAllow': [a[k] for a in fit_cases for k in ['tokens', 'supports']],
        'forbiddenPaths': forbidden+[a['tokens'] for a in inputs if a['id'] not in ['A', 'B']]}, root/'compiled')
    program = root/'compiled/program.json'
    program_hash = sha(program)
    serve_result = child(root, 'serve', {'cases': inputs, 'program': str(program), 'programSHA256': program_hash,
        'readAllow': [str(program)]+[a['tokens'] for a in inputs],
        'forbiddenPaths': forbidden+[a['supports'] for a in fit_cases]}, root/'served')
    out = root/'response-result'
    out.mkdir()
    sealed_files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in (root/'served').iterdir()}
    save(out/'prediction-seal.json', {'schema': 'tce-token-history-prediction-seal-v1', 'requestSHA256': r['requestSHA256'],
        'programSHA256': program_hash, 'serveReceiptSHA256': sha(root/'served/receipt.json'), 'files': sealed_files,
        'scope': 'All six methods/cases frozen before interpreting transfer supports or any selected weights. Full-byte input authentication is separate, performed before discovery.'})
    seal_hash = sha(out/'prediction-seal.json')
    print('TOKEN_HISTORY_PREDICTIONS_SEALED '+seal_hash, flush=True)
    scored = score(s['cases'], tokens, root/'served', out, member, read(root/'compiled/discovery.json'))
    if sha(program) != program_hash or sha(out/'prediction-seal.json') != seal_hash:
        raise ValueError('Program/seal changed during scoring')
    if sealed_files != {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in (root/'served').iterdir()}:
        raise ValueError('Prediction bytes changed during scoring')
    for case in s['cases']:
        for item in [case['lexical'], case['selected']]:
            if sha(item['path']) != item['sha256']:
                raise ValueError('Read-only historical source closure differs')
    save(out/'member-access.json', {'rows': accesses, 'transferLabelsAfterPredictionSeal': True, 'targetPort': 'our approximate first-encoder top-four supports'})
    save(out/'execution.json', {'compile': compile_result, 'serve': serve_result,
        'tokenAndDiscoveryIntakeSeconds': intake_done-start, 'workerCPUSecondsBeforeReport': time.process_time()-cpu,
        'workerWallSecondsBeforeReport': time.perf_counter()-start, 'numpy': np.__version__,
        'physicalWork': {'newNeuralEvaluations': 0, 'learnedMatrixMACs': 0, 'QKScoreProducts': 0, 'selectedValueProducts': 0, 'newImages': 0},
        'sourceClosure': True, 'predictionSealSHA256': seal_hash})
    save(out/'report.json', {'schema': 'tce-token-history-address-result-v1', 'requestSHA256': r['requestSHA256'], 'predictionSealSHA256': seal_hash,
        **scored, 'mechanics_status': 'complete', 'instrument_status': 'saved-support-transfer-with-fallback-traces',
        'trend_status': 'Every dependent row/control retained; no effect-size gate', 'terminal_status': 'unassessed', 'scope': r['scope']})
    for directory, prefix in [(root/'served', 'prediction-'), (root/'compiled', 'compile-'), (clean, 'input-')]:
        for path in directory.iterdir():
            shutil.copyfile(path, out/(prefix+path.name))
    for name in ['request.json', 'inputs.json', 'resource-basis.json', 'frozen-plan.md', 'compile-spec.json', 'serve-spec.json']:
        shutil.copyfile(root/name, out/name)
    save(out/'result-files.json', {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in out.iterdir()})
    print('TOKEN_HISTORY_ADDRESS_COMPLETE', flush=True)


def score(cases, tokens, served, out, member, discovery):
    import math
    from token_history_address import METHODS, REASONS, metrics
    from token_history_process import write_rows
    start = time.perf_counter()
    aggregate = defaultdict(lambda: {'rows': 0, 'ordered': 0, 'set': 0, 'top1': 0, 'overlap': 0.0, 'oldTop4WeightCoverage': 0.0,
        'missingDenominatorRows': 0, 'fallbackSlots': 0, 'exhaustedSlots': 0})
    compare = defaultdict(lambda: {'rows': 0, 'primaryDiffersRecent': 0, 'primaryDiffersTokenNull': 0, 'reasons': {a: 0 for a in REASONS}})
    known = set(discovery['discoveryActiveTokenIDs'])
    source_work = []
    total = 0

    def observations():
        nonlocal total
        for case in cases:
            packet = tokens[case['id']]
            support = member(case['selected'], 'supportPositions', 'post-seal-score')
            weights = member(case['selected'], 'selectedWeights', 'post-seal-score')
            counts = member(case['selected'], 'prefixCounts', 'post-seal-score')
            if support.shape != (512, 32, 4) or weights.shape != support.shape or counts.shape != (512,):
                raise ValueError('Reference support/weight/count geometry differs')
            if counts.tolist() != list(__import__('itertools').accumulate(packet['mask'])):
                raise ValueError('Native reader and program causal masks differ')
            predictions = {}
            for method in METHODS:
                with gzip.open(served/(case['id']+'-'+method+'.jsonl.gz'), 'rt') as stream:
                    predictions[method] = [json.loads(line) for line in stream]
                if len(predictions[method]) != 512:
                    raise ValueError('Prediction query count differs')
            refs = []
            for position in range(512):
                prefix = [p for p in range(position+1) if packet['mask'][p]]
                short = len(prefix) <= 4
                group = ('active' if packet['mask'][position] else 'padding')+('Short' if short else 'Nontrivial')
                groups = [group]
                if packet['mask'][position] and not short and packet['ids'][position] not in known:
                    groups.append('unseenQueryActiveNontrivial')
                reference = support[position].tolist()
                reference_weights = weights[position].tolist()
                refs.append({'position': position, 'supports': reference, 'oldTop4Weights': reference_weights})
                for head in range(32):
                    expected = reference[head]
                    ws = reference_weights[head]
                    valid = [p for p in expected if p >= 0]
                    if len(valid) != min(4, len(prefix)) or len(set(valid)) != len(valid) or any(p not in prefix for p in valid):
                        raise ValueError('Reference is not a causal four-support')
                    if any(not math.isfinite(w) or w < 0 for w in ws) or (prefix and not math.isclose(sum(ws), 1.0, abs_tol=1e-10)):
                        raise ValueError('Retained top-four weight normalization differs')
                    methods = {}
                    for method in METHODS:
                        item = predictions[method][position]
                        predicted = item['addresses'][head]
                        if item['position'] != position or item['prefixCount'] != len(prefix):
                            raise ValueError('Serving clock/prefix differs')
                        actual = [p for p in predicted if p >= 0]
                        if len(actual) != min(4, len(prefix)) or len(set(actual)) != len(actual) or any(p not in prefix for p in actual):
                            raise ValueError('Program support violated causal admission/uniqueness')
                        m = metrics(predicted, expected, ws)
                        reason = item['reasons'][head]
                        methods[method] = {'addresses': predicted, 'raw': item['rawSelectors'][head], 'reasons': reason, **m}
                        for g in groups:
                            for c, h in [('*', -1), (case['id'], -1), (case['id'], head)]:
                                a = aggregate[(case['split'], c, h, g, method)]
                                a['rows'] += 1
                                for name in ['ordered', 'set', 'top1', 'overlap', 'oldTop4WeightCoverage']:
                                    if m[name] is not None:
                                        a[name] += m[name]
                                a['missingDenominatorRows'] += m['overlap'] is None
                                a['fallbackSlots'] += sum(x in (1, 2) for x in reason)
                                a['exhaustedSlots'] += reason.count(3)
                    for g in groups:
                        a = compare[(case['split'], case['id'], g)]
                        a['rows'] += 1
                        a['primaryDiffersRecent'] += methods['primary']['addresses'] != methods['recent']['addresses']
                        a['primaryDiffersTokenNull'] += methods['primary']['addresses'] != methods['tokenNull']['addresses']
                        for why in methods['primary']['reasons']:
                            a['reasons'][REASONS[why]] += 1
                    total += 1
                    yield {'case': case['id'], 'split': case['split'], 'position': position, 'head': head, 'group': group,
                        'unseenQueryToken': packet['ids'][position] not in known, 'support': expected, 'oldTop4Weights': ws, 'methods': methods}
            write_rows(out/('reference-'+case['id']+'.jsonl.gz'), refs)
            source_work.append({'id': case['id'], 'historicalQKScoreProducts': int(sum(counts))*32*128,
                'historicalSelectedValueProducts': sum(min(int(x), 4) for x in counts)*32*128,
                'scope': 'Historical old-reader work targeted by this address-only replacement; not an executed parity/runtime saving. Lexical preparation, new selected scores/soft weights/values and suffix remain paid.'})

    write_rows(out/'observations.jsonl.gz', observations())
    rows = []
    for (split, case, head, group, method), values in sorted(aggregate.items()):
        n = values['rows']
        nonempty = n-values['missingDenominatorRows']
        rows.append({'split': split, 'case': case, 'head': head, 'group': group, 'method': method, **values,
            'rates': {k: values[k]/(nonempty if k in ['overlap', 'oldTop4WeightCoverage'] else n) if (nonempty if k in ['overlap', 'oldTop4WeightCoverage'] else n) else None
                for k in ['ordered', 'set', 'top1', 'overlap', 'oldTop4WeightCoverage']}})
    save(out/'panels.json', rows)
    comparisons = [{'split': k[0], 'case': k[1], 'group': k[2], **v} for k, v in sorted(compare.items())]
    save(out/'fallback-comparisons.json', comparisons)
    return {'panelsSHA256': sha(out/'panels.json'), 'observationsSHA256': sha(out/'observations.jsonl.gz'), 'dependentHeadQueryRows': total,
        'methodRows': total*len(METHODS), 'independentPromptCounts': {'discovery': 2, 'retrospectiveTransfer': 4},
        'pooledPanels': [a for a in rows if a['case'] == '*'], 'historicalPortWork': source_work,
        'scoringAndObservationSeconds': time.perf_counter()-start,
        'metricScope': 'Coverage is mass within old normalized top-four distribution, not full-attention mass, read accuracy, continuation or image quality. Token-null is synthetic; position-only has smaller search capacity. No iid inference from dependent head/token rows.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['stage', 'submit', 'execute', 'collect'])
    parser.add_argument('root')
    args = parser.parse_args()
    if args.action == 'collect':
        from response_study import collect
        result = collect(args.root)
    else:
        result = globals()[args.action](args.root)
    if result is not None:
        print(json.dumps(result))
