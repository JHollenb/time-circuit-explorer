"""Stage/execute one source-compiled integer ordering feasibility transaction."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from token_history_process import read, save, sha, write_rows

HERE = Path(__file__).resolve().parent
OWNED = ['source_order_program.py', 'source_order_compile.py', 'source_order_study.py', 'test_source_order.py',
    'encoder_atoms.py', 'token_history_address.py', 'token_history_process.py']
MANIFEST = '9a155bd1410c52909997bacaf1d1aac2f9e90be86dc269d5e7edfab2b5285432'


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify(root, request):
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']:
        raise ValueError('Frozen request differs')
    for name, digest in request['sourceHashes'].items():
        if sha(Path(root)/name) != digest:
            raise ValueError('Frozen source/input differs: '+name)


def stage(root):
    from mrun.client.api import Api
    api = Api()
    previous = HERE.parent/'docs/results/token-history-address'
    pub = read(previous/'publication.json')
    for name, item in pub['files'].items():
        if sha(previous/name) != item['sha256']:
            raise ValueError('Prior compact record changed')
    alphabet = read(previous/'compile-discovery.json')['discoveryActiveTokenIDs']
    if len(alphabet) != 30 or alphabet != sorted(set(alphabet)):
        raise ValueError('Frozen discovery alphabet differs')
    store = read(HERE.parent/'docs/results/uncentered-reader/inputs.json')['lexicalStore']
    manifest_code = "import json,hashlib;from pathlib import Path;p=Path(%r);b=p.read_bytes();print(json.dumps({'manifest':json.loads(b),'sha256':hashlib.sha256(b).hexdigest()}))" % (store+'/manifest.json')
    remote = json.loads(subprocess.run(['ssh', '-T', 'beast', 'python3', '-'], input=manifest_code, text=True, capture_output=True, check=True, timeout=30).stdout)
    m = remote['manifest']
    if remote['sha256'] != MANIFEST or (m['coverageCount'], m['vocabSize'], m['sequenceLength'], m['headDim']) != (151936, 151936, 512, 128):
        raise ValueError('Source manifest census differs')
    shards = sorted({token//m['chunkRows'] for token in alphabet+[151643]})
    source_files = [{'path': store+'/manifest.json', 'sha256': MANIFEST}]
    source_files += [{'path': store+'/'+m[k]['path'], 'sha256': m[k]['sha256']} for k in ['coverage', 'rowHashes', 'rope']]
    source_files += [{'path': store+'/'+m['shards'][i]['path'], 'sha256': m['shards'][i]['sha256']} for i in shards]
    historic = {a['id']: a for a in read(previous/'inputs.json')['cases'] if a['split'] == 'discovery'}
    source_files += [{k: a['lexical'][k] for k in ['path', 'sha256']} for a in historic.values()]
    code = """import json,hashlib,stat
from pathlib import Path
items=json.loads(%r);rows=[]
for item in items:
 p=Path(item['path']);s=p.lstat()
 if not stat.S_ISREG(s.st_mode):raise ValueError('Nonregular source')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 if h.hexdigest()!=item['sha256']:raise ValueError('Source hash differs')
 rows.append({'path':str(p),'sha256':h.hexdigest(),'bytes':s.st_size})
print(json.dumps({'files':rows,'bytes':sum(r['bytes'] for r in rows)}))
""" % json.dumps(source_files)
    intake = json.loads(subprocess.run(['ssh', '-T', 'beast', 'python3', '-'], input=code, text=True, capture_output=True, check=True, timeout=60).stdout)
    atom_job = api.json('GET', '/api/jobs/job-7dcbeae35deb')
    if atom_job['state'] != 'succeeded':
        raise ValueError('Historical atom job is not successfully terminal')
    host = next(a for a in api.json('GET', '/api/hosts') if a['name'] == 'beast')
    root = Path(root).resolve();root.mkdir(parents=True, exist_ok=False);payload = root/'payload';payload.mkdir()
    for name in OWNED:
        shutil.copyfile(HERE/name, payload/name)
    for source, destination in [('source-ordering-program-feasibility.md', 'frozen-plan.md'), ('source-ordering-implementation.md', 'implementation.md')]:
        shutil.copyfile(HERE.parent/'docs/plans'/source, payload/destination)
    for case in ['A', 'B', '00', '10', '01', '11']:
        shutil.copyfile(previous/('input-'+case+'-tokens.json'), payload/(case+'-tokens.json'))
    for case in ['A', 'B']:
        shutil.copyfile(previous/('reference-'+case+'.jsonl.gz'), payload/(case+'-old-reference.jsonl.gz'))
    save(payload/'atom-manifest.json', m)
    save(payload/'inputs.json', {'store': store, 'manifestSHA256': MANIFEST, 'alphabet': alphabet, 'intake': intake,
        'historicalDiscovery': historic, 'atomOriginalJob': {'jobID': atom_job['job_id'], 'state': atom_job['state'], 'payloadCustody': atom_job.get('payload_custody'),
            'scope': 'Original historical custody preserved; current full-byte authentication does not add missing historical execution attestation.'},
        'parentPublicationSHA256': sha(previous/'publication.json'), 'parentDiscoverySHA256': sha(previous/'compile-discovery.json')})
    save(payload/'resource-basis.json', {'rawOrdinalBytes': 7848960, 'ordinalCount': 3924480, 'compileCoordinateProducts': 502333440,
        'maximumProductBufferBytes': 128*128*8, 'maximumKeyOperandBytes': 30*512*128*8,
        'sourceFileBytesFreshHashed': intake['bytes'], 'sourceShardCount': len(shards),
        'liveHost': {k: host.get(k) for k in ['name', 'active_jobs', 'admission', 'telemetry', 'ram_total_mb']},
        'admission': 'Intent-firstCPU,host-local pin,BLAS1,600second cap/no retries. No original model acquisition/CUDA/encoder/denoiser/VAE. Offline source products and later reference products paid separately.'})
    r = {'schema': 'tce-source-order-request-v1', 'sourceHashes': {p.name: sha(p) for p in payload.iterdir()}, 'ownedSources': OWNED,
        'device': 'cpu', 'backend': 'numpy-elementwise-products/python-math-fsum; stdlib-integer-serving', 'length': 512, 'head': 0,
        'paddingQueryID': 151643, 'alphabetSize': 30, 'timeoutSeconds': 600,
        'compileWork': {'coordinateProducts': 502333440, 'fsumCalls': 3924480},
        'scope': 'Finite discovery-alphabet/padding-head ordering under new context-independent deterministic score contract. Serving no learned arithmetic; offline learned attention remains paid. No arbitrary-vocabulary/fullencoder/image claim.'}
    r['requestSHA256'] = identity(r);save(root/'request.json', r);save(payload/'request.json', r)
    save(payload/'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'root': str(root), 'requestSHA256': r['requestSHA256'], 'sourceFiles': len(intake['files']), 'freshSourceBytes': intake['bytes'], 'rawOrdinalBytes': 7848960}


def submit(root):
    from mrun.client.submit import launch
    root = Path(root).resolve();r = read(root/'request.json');verify(root/'payload', r)
    if (root/'job.json').exists():raise ValueError('Already submitted; inspect same handle')
    cmd = ['/usr/bin/env', 'PYTHONUNBUFFERED=1', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONNOUSERSITE=1', 'PYTHONPATH=', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        'UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache', 'UV_LINK_MODE=symlink', 'uv', 'run', '--no-project', '--offline', '--python', '3.14.6', '--with', 'numpy>=2.0',
        'python', '-u', 'source_order_study.py', 'execute', '.']
    job = launch(cmd, experiment='tce-source-order-program-v1', task='analysis', payload=root/'payload', pin='beast', needs={'cpu': True},
        config={k: r[k] for k in ['schema', 'requestSHA256', 'device', 'backend', 'length', 'head', 'compileWork']},
        timeout_s=600, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='Compile fixedpad151643/head0/discovery30IDs/all512absoluteclocks:3.924Mordinals,502.333MofflineFP64products+fsum;128rowbatches,oneclockbank. Integerfreshserving13histories/4controls,sealedbeforeindependentreferences/oldlabels. No QKV/scores in serving. Unknownactivehistoriesunsupported;clock+1lastinapplicable. IntentfirstCPU/BLAS1,600scap,noretries,noGPU/nativeencoder/denoiser/VAE/image. Report compile/storage/supportcoverage/costs separately.')
    v = {'jobID': str(job), 'requestSHA256': r['requestSHA256']};save(root/'job.json', v);return v


def synthetic(alphabet):
    cases = {}
    patterns = [('full-cycle', 511), ('singleton', 1), ('five-cycle', 5), ('thirtyfive-cycle', 35), ('edit-A', 35), ('edit-B', 35)]
    for name, count in patterns:
        ids = [alphabet[p % len(alphabet)] if p < count else 151643 for p in range(512)]
        if name in ['edit-A', 'edit-B']:ids[10] = alphabet[0 if name == 'edit-A' else 1]
        cases[name] = {'ids': ids, 'mask': [int(p < count) for p in range(512)], 'positions': list(range(512))}
    cases['holes'] = {'ids': [alphabet[(7*p) % len(alphabet)] if p % 4 == 0 else 151643 for p in range(512)],
        'mask': [int(p % 4 == 0) for p in range(512)], 'positions': list(range(512))}
    return cases


def execute(root):
    import numpy as np
    from encoder_atoms import guard
    from source_order_compile import source_operands, compile_order
    guard()
    root = Path(root).resolve();r = read(root/'request.json');verify(root, r);s = read(root/'inputs.json')
    start = time.perf_counter();cpu = time.process_time()
    for item in s['intake']['files']:
        if sha(item['path']) != item['sha256']:raise ValueError('Source bytes changed after stage')
    q, k, q_words, k_words, profile = source_operands(s['store'], s['manifestSHA256'], s['alphabet'], 151643, 512)
    alphabet_index = {token: i for i, token in enumerate(s['alphabet'])}
    qualification = []
    for case, item in s['historicalDiscovery'].items():
        packet = read(root/(case+'-tokens.json'))
        with np.load(item['lexical']['path'], allow_pickle=False) as f:
            legacy_q, legacy_k = f['qpos'], f['kpos']
        if legacy_q.shape != (512, 4096) or legacy_k.shape != (512, 1024) or legacy_q.dtype != np.uint16 or legacy_k.dtype != np.uint16:
            raise ValueError('Historical source word geometry differs')
        bad = 0;compared = 0
        for p, (token, active) in enumerate(zip(packet['ids'], packet['mask'])):
            expected = k_words[alphabet_index[token], p] if active else q_words[p]
            actual = legacy_k[p, :128] if active else legacy_q[p, :128]
            if not active and token != 151643:raise ValueError('Historical query type differs')
            bad += int(np.sum(expected != actual));compared += 128
        if bad:raise ValueError('Qualified source rotary operand words differ')
        qualification.append({'case': case, 'comparedWords': compared, 'mismatchedWords': bad, 'lexicalSHA256': item['lexical']['sha256']})
    del legacy_q, legacy_k
    operand_hashes = {'q': hashlib.sha256(q.tobytes()).hexdigest(), 'k': hashlib.sha256(k.tobytes()).hexdigest()}
    ready = time.perf_counter()
    metadata = compile_order(q, k, s['alphabet'], 151643, root/'compiled')
    compiled = time.perf_counter()
    packets = {case: read(root/(case+'-tokens.json')) for case in ['A', 'B', '00', '10', '01', '11']}
    packets.update(synthetic(s['alphabet']))
    clean = root/'clean';clean.mkdir();cases = []
    for name, packet in packets.items():
        path = clean/(name+'-tokens.json');save(path, packet)
        cases.append({'id': name, 'tokens': str(path), 'scope': 'historicalDiscovery' if name in ['A', 'B'] else 'historicalTransfer' if name in ['00', '10', '01', '11'] else 'synthetic'})
    spec = {'metadata': str(root/'compiled/program.json'), 'metadataSHA256': sha(root/'compiled/program.json'), 'bundle': str(root/'compiled/order.u16.gz'), 'cases': cases,
        'readAllow': [str(root/'compiled/program.json'), str(root/'compiled/order.u16.gz')]+[a['tokens'] for a in cases],
        'forbiddenPaths': [str(root/'request.json'), str(root/'A-old-reference.jsonl.gz'), str(root/'B-old-reference.jsonl.gz')]+[a['path'] for a in s['intake']['files']]}
    save(root/'serve-spec.json', spec);tick = time.perf_counter()
    subprocess.run([sys.executable, '-I', '-B', str(root/'source_order_program.py'), str(root/'serve-spec.json'), str(root/'served')], check=True, cwd=root)
    serve_wall = time.perf_counter()-tick
    out = root/'response-result';out.mkdir()
    sealed_files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in (root/'served').iterdir()}
    save(out/'prediction-seal.json', {'schema': 'tce-source-order-prediction-seal-v1', 'requestSHA256': r['requestSHA256'],
        'programSHA256': spec['metadataSHA256'], 'compressedSHA256': metadata['compressedSHA256'], 'files': sealed_files,
        'scope': 'Fresh serving exits before any per-history direct-score reference or historical support/weight interpretation.'})
    seal_sha = sha(out/'prediction-seal.json');print('SOURCE_ORDER_SERVING_SEALED '+seal_sha, flush=True)
    result = score(cases, packets, q, k, alphabet_index, root, out)
    if sealed_files != {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in (root/'served').iterdir()} or sha(root/'compiled/program.json') != spec['metadataSHA256']:
        raise ValueError('Sealed program/predictions changed during reference scoring')
    if operand_hashes != {'q': hashlib.sha256(q.tobytes()).hexdigest(), 'k': hashlib.sha256(k.tobytes()).hexdigest()}:
        raise ValueError('Source operands changed')
    for item in s['intake']['files']:
        if sha(item['path']) != item['sha256']:raise ValueError('Read-only source closure differs')
    save(out/'execution.json', {'profile': profile, 'qualification': qualification, 'operandHashes': operand_hashes,
        'sourceAndQualificationSeconds': ready-start, 'compilationWithSerializationSeconds': compiled-ready, 'serveProcessWallSeconds': serve_wall,
        'workerWallSecondsBeforeReport': time.perf_counter()-start, 'workerCPUSecondsBeforeReport': time.process_time()-cpu,
        'sourceClosure': True, 'predictionSealSHA256': seal_sha, 'originalModelEncoderDenoiserVAEEvaluations': 0, 'newImages': 0,
        'scope': 'Offline learned coordinate products/fsum remain paid. Zero learned score/vector arithmetic only in fresh integer serving.'})
    save(out/'report.json', {'schema': 'tce-source-order-result-v1', 'requestSHA256': r['requestSHA256'], 'program': metadata, **result,
        'mechanics_status': 'complete', 'instrument_status': 'source-order-restriction-under-new-score-contract', 'trend_status': 'All controls/unsupported rows retained without quality gate',
        'terminal_status': 'arbitrary-prompt zero-neural image generation unestablished', 'scope': r['scope']})
    for directory, prefix in [(root/'compiled', 'compile-'), (root/'served', 'prediction-'), (clean, 'input-')]:
        for p in directory.iterdir():shutil.copyfile(p, out/(prefix+p.name))
    for name in ['request.json', 'inputs.json', 'resource-basis.json', 'atom-manifest.json', 'frozen-plan.md', 'implementation.md', 'serve-spec.json', 'A-old-reference.jsonl.gz', 'B-old-reference.jsonl.gz']:
        shutil.copyfile(root/name, out/name)
    save(out/'result-files.json', {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in out.iterdir()})
    print('SOURCE_ORDER_COMPLETE', flush=True)


def score(cases, packets, q, k, alphabet_index, root, out):
    import numpy as np
    from source_order_compile import scores
    from source_order_program import METHODS
    from token_history_address import metrics
    counters = {'coordinateProducts': 0, 'fsumCalls': 0}
    panels = defaultdict(lambda: {'rows': 0, 'supported': 0, 'orderedMatches': 0, 'setMatches': 0, 'top1Matches': 0, 'overlapSum': 0., 'oldRows': 0, 'oldOrderedMatches': 0, 'oldWeightCoverageSum': 0.})
    statuses = Counter();reference_mismatches = 0;count = 0;records = [];references = {};started = time.perf_counter()
    for case in cases:
        name = case['id'];packet = packets[name];predictions = {}
        for method in METHODS:
            with gzip.open(root/'served'/(name+'-'+method+'.jsonl.gz'), 'rt') as f:predictions[method] = [json.loads(line) for line in f]
        old = None
        if name in ['A', 'B']:
            with gzip.open(root/(name+'-old-reference.jsonl.gz'), 'rt') as f:old = [json.loads(line) for line in f]
        history = [];case_refs = []
        for clock in range(512):
            if packet['mask'][clock]:history.append((packet['ids'][clock], clock))
            primary = predictions['primary'][clock]
            target = None
            if primary['status'] == 'supported':
                bank = np.asarray([k[alphabet_index[t], p] for t, p in history], dtype=np.float64).reshape(-1, 128)
                values = scores(q[clock], bank, counters)
                order = sorted(range(len(history)), key=lambda j: (-float(values[j]), history[j][1]))[:4]
                target = [history[j][1] for j in order]+[-1]*(4-len(order))
                case_refs.append({'clock': clock, 'prefixCount': len(history), 'addresses': target})
            group = 'active' if packet['mask'][clock] else 'paddingShort' if len(history) <= 4 else 'paddingNontrivial'
            methods = {}
            for method in METHODS:
                item = predictions[method][clock];status = item['status'];statuses[(case['scope'], name, method, status)] += 1
                m = None;old_metric = None
                if status == 'supported':
                    if target is None:raise ValueError('Control supported outside primary domain')
                    predicted = item['addresses'];a = {p for p in predicted if p >= 0};b = {p for p in target if p >= 0}
                    prefix = {p for _, p in history}
                    if len(a) != min(4, len(history)) or any(p not in prefix for p in a):raise ValueError('Causal output support differs')
                    m = {'ordered': int(predicted == target), 'set': int(a == b), 'top1': int(predicted[0] == target[0]), 'overlap': len(a & b)/len(b) if b else None}
                    if method == 'primary':reference_mismatches += not m['ordered']
                    if old is not None:old_metric = metrics(predicted, old[clock]['supports'][0], old[clock]['oldTop4Weights'][0])
                methods[method] = {**item, 'declaredOrder': m, 'oldReader': old_metric}
                for c in ['*', name]:
                    panel = panels[(case['scope'], c, group, method)];panel['rows'] += 1
                    if m:
                        panel['supported'] += 1;panel['orderedMatches'] += m['ordered'];panel['setMatches'] += m['set'];panel['top1Matches'] += m['top1'];panel['overlapSum'] += m['overlap'] or 0
                    if old_metric:
                        panel['oldRows'] += 1;panel['oldOrderedMatches'] += old_metric['ordered'];panel['oldWeightCoverageSum'] += old_metric['oldTop4WeightCoverage'] or 0
            records.append({'case': name, 'scope': case['scope'], 'clock': clock, 'group': group, 'mask': packet['mask'][clock], 'prefixCount': len(history),
                'declaredReference': target, 'oldSupport': old[clock]['supports'][0] if old is not None and target is not None else None,
                'oldWeights': old[clock]['oldTop4Weights'][0] if old is not None and target is not None else None, 'methods': methods})
            count += 1
        references[name] = case_refs
    write_rows(out/'observations.jsonl.gz', records)
    panel_rows = []
    for (scope, case, group, method), values in sorted(panels.items()):
        n = values['supported'];old_n = values['oldRows']
        panel_rows.append({'scope': scope, 'case': case, 'group': group, 'method': method, **values,
            'declaredOrderedRate': values['orderedMatches']/n if n else None, 'declaredSetRate': values['setMatches']/n if n else None,
            'declaredOverlapRate': values['overlapSum']/n if n else None, 'oldOrderedRate': values['oldOrderedMatches']/old_n if old_n else None,
            'oldTop4WeightCoverage': values['oldWeightCoverageSum']/old_n if old_n else None})
    save(out/'panels.json', panel_rows)
    save(out/'coverage.json', [{'scope': a, 'case': b, 'method': c, 'status': d, 'rows': n} for (a, b, c, d), n in sorted(statuses.items())])
    a, b = references['edit-A'], references['edit-B'];pairs = [{'clock': left['clock'], 'referenceChanged': left['addresses'] != right['addresses']} for left, right in zip(a, b)]
    save(out/'synthetic-edit-response.json', {'rows': pairs, 'changedReferenceRows': sum(r['referenceChanged'] for r in pairs),
        'scope': 'Synthetic key-substitution pair under declared source comparator; primary identity assessed separately. No semantic or image claim.'})
    return {'headQueryRows': count, 'methodRows': count*len(METHODS), 'primaryDeclaredOrderMismatches': reference_mismatches,
        'diagnosticReferenceWork': counters, 'scoringSeconds': time.perf_counter()-started, 'pooledPanels': [p for p in panel_rows if p['case'] == '*'],
        'syntheticEditChangedReferenceRows': sum(r['referenceChanged'] for r in pairs),
        'metricScope': 'New deterministic contract exactness and historical old-reader observations distinct; unknown admitted tokens unsupported, short prefixes separate. No independent prompt replication/native image or general-vocabulary claim.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']);p.add_argument('root');a = p.parse_args()
    if a.action == 'collect':
        from response_study import collect
        v = collect(a.root)
    else:v = globals()[a.action](a.root)
    if v is not None:print(json.dumps(v))
