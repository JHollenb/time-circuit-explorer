"""Independent compact evidence/byte publisher, with no local tensor hydration."""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from contextual_writer_program import read, save, sha


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tensor_fingerprint(payload, shape):
    return digest({'dtype': 'torch.bfloat16', 'shape': shape, 'bytes': payload})


def check_frame(frame):
    identity = {k: frame[k] for k in ('schema', 'slots', 'parent_frame_id', 'act', 'implementation', 'program_fingerprint', 'metadata')}
    if digest(identity) != frame['frame_id']: raise ValueError('Typed frame identity differs')


def serving_path(child, case, method, job):
    name = case+'-H0.bf16' if method == 'H0' else method+'-'+case+'.bf16'
    return str(Path(job['result']['diagnostics']['cwd'])/child/name)


def publish(root, mechanics, destination):
    from mrun.client.api import Api
    root, destination = Path(root).resolve(), Path(destination).resolve(); request = read(root/'request.json')
    if sha(__file__) != request['sourceHashes']['publish_contextual_writer.py']: raise ValueError('Bound publisher source differs')
    job = read(root/'scheduler-job.json'); collection = read(root/'collection.json'); job_id = job['job_id']
    live = Api().json('GET', '/api/jobs/'+job_id); custody = job['payload_custody']; phases = [custody[k] for k in ('declared', 'sealed', 'executed')]
    if live['state'] != 'succeeded' or live['result']['returncode'] != 0 or live['payload_custody'] != custody or custody['required'] is not True or not live.get('admission'):
        raise ValueError('Successful guarded custody missing')
    if any((p['sha256'], p['size_bytes']) != (phases[0]['sha256'], phases[0]['size_bytes']) for p in phases) or phases[-1]['host'] != 'beast': raise ValueError('Payload phases differ')
    if job['config']['requestSHA256'] != request['requestSHA256'] or live['config']['requestSHA256'] != request['requestSHA256'] or collection['custody'] != custody or collection['jobID'] != job_id: raise ValueError('Request/custody join differs')
    if live['result'] != job['result'] or collection['result'] != job['result']: raise ValueError('Current completed result identity differs')
    for n, h in request['sourceHashes'].items():
        if sha(root/'payload'/n) != h: raise ValueError('Executed source differs')
    actual = {str(p.relative_to(root/'result')): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in (root/'result').rglob('*') if p.is_file()}
    if actual != collection['inventory']['files']: raise ValueError('Complete collection differs')
    tests = read(mechanics)
    if tests['testsRun'] != 8 or tests['errors'] or tests['failures']: raise ValueError('Targeted mechanics incomplete')
    for n, h in tests['sourceHashes'].items():
        if sha(Path(__file__).parent/n) != h or request['sourceHashes'].get(n) != h: raise ValueError('Tested source differs')
    result = root/'result'; report = read(result/'report.json'); source = read(result/'inputs.json'); catalog = read(result/'programs.json')
    if read(result/'request.json') != request or sha(result/'inputs.json') != request['sourceHashes']['inputs.json'] or sha(result/'frozen-plan.md') != request['sourceHashes']['frozen-plan.md']: raise ValueError('Collected source authority differs')
    if report['requestSHA256'] != request['requestSHA256'] or report['consumerWork'] != {'units': 28, 'originalLinears': 84, 'nominalMACs': 225485783040, 'normalizations': 28, 'fullBlocks': 0, 'images': 0}: raise ValueError('Result work differs')
    if sha(result/'final-seal.json') != report['transferReleasedAfterSealSHA256'] or sha(result/'feedback.json') != report['feedbackSHA256']: raise ValueError('Target release or selection differs')
    feedback = read(result/'feedback.json'); selected = report['selected']
    if selected != feedback['selected'] or not feedback['exactRestore'] or feedback['acceptedProgramSHA256'] != catalog['accepted']['programSHA256'] or catalog[selected]['programSHA256'] != catalog['accepted']['programSHA256']: raise ValueError('Immutable restore join differs')
    for method in ('initial', 'half'):
        terms = [r for r in report['summaries'] if r['case'] in request['discovery'] and r['method'] == method and r['port'] in ('Q', 'K', 'V') and r['stratum'] in ('active', 'padding')]
        if len(terms) != 24: raise ValueError('Discovery feedback coverage differs')
        numerator = math.fsum(r['errorRMS']**2 for r in terms)/24; denominator = math.fsum(r['normalizerRMS']**2 for r in terms)/24
        score = numerator/denominator if denominator else None
        for key, value in [('score', score), ('errorMeanSquare', numerator), ('referenceIncrementMeanSquare', denominator)]:
            observed = feedback['scores'][method][key]
            if (value is None) != (observed is None) or value is not None and abs(observed-value) > 1e-10*(1+abs(value)): raise ValueError('Feedback score differs')
    expected_selection = 'half' if feedback['scores']['half']['score'] is not None and feedback['scores']['initial']['score'] is not None and feedback['scores']['half']['score'] < feedback['scores']['initial']['score'] else 'initial'
    if selected != expected_selection: raise ValueError('Discovery-only continuation policy differs')
    if set(catalog) != {'initial', 'half', 'position', 'accepted'}: raise ValueError('Program domain differs')
    children = {name: read(result/(name+'-serving.json')) for name in ('initializer', 'discovery', 'final')}
    child_rows = {}
    for name, child in children.items():
        expected_cases = request['discovery'] if name != 'final' else [r['id'] for r in request['cases']]
        expected_methods = [] if name == 'initializer' else ['initial', 'half', 'position'] if name == 'discovery' else ['accepted', 'position']
        if [r['id'] for r in child['rows']] != expected_cases: raise ValueError('Raw serving case domain differs')
        for row in child['rows']:
            key = row['id']; packet = row['packet']; child_rows[name, key] = row
            if packet['positions'] != list(range(512)) or len(packet['ids']) != 512 or len(packet['mask']) != 512 or any(type(t) is not int or not 0 <= t < 151936 for t in packet['ids']): raise ValueError('Owned packet ABI differs')
            if any(type(m) is not int or m not in (0, 1) for m in packet['mask']) or any(a < b for a, b in zip(packet['mask'], packet['mask'][1:])): raise ValueError('Owned mask ABI differs')
            if row['activeRows'] != sum(packet['mask']) or row['H0']['bytes'] != 2621440 or row['H0']['path'] != serving_path(name, key, 'H0', job): raise ValueError('Owned initializer field differs')
            if key in source['cases']:
                for label, expected in [('ids', source['cases'][key]['tokenIDsSHA256']), ('mask', source['cases'][key]['maskSHA256'])]:
                    payload = __import__('struct').pack('<512q', *packet[label])
                    if hashlib.sha256(payload).hexdigest() != expected: raise ValueError('Native raw-input packet binding differs')
            if [p['method'] for p in row['predictions']] != expected_methods: raise ValueError('Serving method domain differs')
            for pred in row['predictions']:
                if pred['bytes'] != 2621440 or pred['programSHA256'] != catalog[pred['method']]['programSHA256'] or pred['coordinateAdditions'] != 2611200 or len(pred['leaves']) != 510:
                    raise ValueError('Own complete-state/program binding differs')
                if pred['path'] != serving_path(name, key, pred['method'], job): raise ValueError('Owned output path differs')
    for key in request['discovery']:
        init = child_rows['initializer', key]; old = child_rows['discovery', key]; new = child_rows['final', key]
        if init['packet'] != old['packet'] or old['packet'] != new['packet'] or len({r['H0']['sha256'] for r in (init, old, new)}) != 1: raise ValueError('Fresh initializer reproduction differs')
        for method, previous in [('accepted', selected), ('position', 'position')]:
            if next(p for p in new['predictions'] if p['method'] == method)['sha256'] != next(p for p in old['predictions'] if p['method'] == previous)['sha256']: raise ValueError('Fresh selected-program reproduction differs')
    case_methods = {key: ['H0', 'initial', 'half', 'position'] for key in request['discovery']}
    case_methods.update({key: ['H0', 'accepted', 'position'] for key in request['transfer']})
    pair_methods = {('A', 'B'): ['H0', 'initial', 'half', 'position'], ('00', '10'): ['H0', 'initial', 'half', 'position'], ('01', '11'): ['H0', 'accepted', 'position']}
    expected_domain = {('case', key, method, port, pos) for key, methods in case_methods.items() for method in methods for port in ('H1', 'Q', 'K', 'V') for pos in range(512)}
    expected_domain.update({('pair', a, b, method, port, pos) for (a, b), methods in pair_methods.items() for method in methods for port in ('H1', 'Q', 'K', 'V') for pos in range(512)})
    seen = set()
    # Recompute every pooled state/consumer/response scalar from retained raw rows.
    groups = collections.defaultdict(list); rows = 0
    with gzip.open(result/'observations.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line); rows += 1
            pos = r['position']
            if type(pos) is not int: raise ValueError('Invalid observation position')
            if 'case' in r:
                identity = ('case', r['case'], r['method'], r['port'], pos)
                labels = ['all', 'active' if r['active'] else 'padding']; base = ('case', r['case'], r['method'], r['port'])
                packet = child_rows['discovery' if r['case'] in request['discovery'] else 'final', r['case']]['packet']
                if pos not in range(512) or r['active'] is not bool(packet['mask'][pos]): raise ValueError('Observation native mask differs')
            else:
                identity = ('pair', *r['pair'], r['method'], r['port'], pos)
                labels = ['all']+(['unchangedToken'] if r['sameToken'] else []); base = ('pair', *r['pair'], r['method'], r['port'])
                a, b = r['pair']; population = 'discovery' if a in request['discovery'] else 'final'
                same = child_rows[population, a]['packet']['ids'][pos] == child_rows[population, b]['packet']['ids'][pos] if pos in range(512) else None
                if r['sameToken'] is not same: raise ValueError('Observation unchanged-token binding differs')
            if identity not in expected_domain or identity in seen: raise ValueError('Observation identity/domain differs')
            seen.add(identity)
            for label in labels: groups[(*base, label)].append(r)
    verified = 0
    for summary, pair in [(r, False) for r in report['summaries']]+[(r, True) for r in report['pairs']]:
        key = ('pair', summary['A'], summary['B'], summary['method'], summary['port'], summary['selection']) if pair else ('case', summary['case'], summary['method'], summary['port'], summary['stratum'])
        values = groups.pop(key); n = len(values)
        if n != summary['rows']: raise ValueError('Observation coverage differs')
        pooled = {k: math.sqrt(math.fsum(v[k]**2 for v in values)/n) for k in ('errorRMS', 'normalizerRMS', 'predictionRMS', 'targetRMS')}
        pooled['relativeRMS'] = pooled['errorRMS']/pooled['normalizerRMS'] if pooled['normalizerRMS'] else None
        den = pooled['predictionRMS']*pooled['targetRMS']
        pooled['cosine'] = math.fsum((v['cosine'] or 0)*v['predictionRMS']*v['targetRMS'] for v in values)/n/den if den else None
        for k, expected in pooled.items():
            observed = summary[k]
            if (observed is None) != (expected is None) or expected is not None and (not math.isfinite(observed) or abs(observed-expected) > 1e-10*(1+abs(expected))): raise ValueError('Pooled scalar differs: '+str(key)+' '+k)
            verified += 1
    if groups or seen != expected_domain or rows != report['observationRows'] or rows != 67584: raise ValueError('Raw observation coverage differs')
    expected_files = {}; payload_specs = {}; cwd = job['result']['diagnostics']['cwd']
    if cwd != '/mnt/hdd8tb/mrun-work/'+job_id: raise ValueError('Unexpected remote root')
    for name, info in catalog.items():
        p = info['program']; encoded = json.dumps(p, indent=2, sort_keys=True, allow_nan=False).encode()
        if hashlib.sha256(encoded).hexdigest() != info['programSHA256'] or len(p['trees']) != 2 or p['budget'] != request['budget']: raise ValueError('Program identity/budget differs')
        folder = Path(info['directory']); expected_files[str(folder/'program.json')] = info['programSHA256']; expected_files[str(folder/'prefix.bf16')] = p['prefixSHA256']
        for tree, d in zip(p['trees'], p['dictionaries']):
            leaves = []
            def visit(node, depth):
                if 'atom' in node:
                    if not 0 <= node['atom'] < d['atoms']: raise ValueError('Dictionary address differs')
                    leaves.append(node['atom']); return
                if depth >= 4 or not 0 <= node['feature'] < len(p['featureNames']): raise ValueError('Branch grammar differs')
                visit(node['left'], depth+1); visit(node['right'], depth+1)
            visit(tree, 0)
            if len(leaves) > 16 or d['atoms'] > 16 or d['bytes'] != d['atoms']*2560*4: raise ValueError('Writer budget differs')
            expected_files[str(folder/d['path'])] = d['sha256']
    for child_name, seal_name in [('initializer', 'initializer-seal.json'), ('discovery', 'discovery-seal.json'), ('final', 'final-seal.json')]:
        child = read(result/(child_name+'-serving.json')); seal = read(result/seal_name)
        if sha(result/(child_name+'-serving.json')) != seal['receiptSHA256'] or child['loadedNumericalRuntimeRoots'] or child['newNeuralEvaluations'] != 0: raise ValueError('Serving seal/isolation differs')
        if child['fileCanaryDenials'] != source['forbiddenServingPaths'] or child['blockedImports'] != ['numpy', 'torch']: raise ValueError('Serving canary coverage differs')
        expected_outputs = [p for row in child['rows'] for p in [row['H0'], *row['predictions']]]
        if expected_outputs != seal['outputs']: raise ValueError('Complete output seal differs')
        for item in expected_outputs: expected_files[item['path']] = item['sha256']
    consumers = read(result/'consumer-fields.json'); typed = read(result/'typed-consumers.json')
    if len(consumers['rows']) != 28 or len(typed['rows']) != 28 or sum(r['evidence']['metrics']['linears'] for r in typed['rows']) != 84: raise ValueError('Physical typed consumer coverage differs')
    consumer_domain = {(key, method) for key, methods in case_methods.items() for method in ['native', *methods]}
    by_consumer = {(r['case'], r['method']): r for r in consumers['rows']}; by_typed = {(r['case'], r['method']): r for r in typed['rows']}
    if set(by_consumer) != consumer_domain or set(by_typed) != consumer_domain or len(by_consumer) != len(consumers['rows']) or len(by_typed) != len(typed['rows']): raise ValueError('Exact consumer domain differs')
    parent = typed['unchangedWeightParent']; check_frame(parent)
    names = {'normWeight': ('input_layernorm.weight', [2560]), 'qWeight': ('self_attn.q_proj.weight', [4096, 2560]), 'kWeight': ('self_attn.k_proj.weight', [1024, 2560]), 'vWeight': ('self_attn.v_proj.weight', [1024, 2560])}
    expected_params = {slot: next(p['payloadSHA256'] for p in source['parameters'] if p['name'] == 'model.layers.1.'+name) for slot, (name, shape) in names.items()}
    if typed['parameterPayloadBefore'] != expected_params or typed['parameterPayloadAfter'] != expected_params or set(parent['slots']) != set(names): raise ValueError('Actual parameter-byte immutability differs')
    if typed['spec']['reads'] != ['H1', *names] or typed['spec']['writes'] != ['Q', 'K', 'V']: raise ValueError('Declared native consumer surface differs')
    for slot, (name, shape) in names.items():
        if parent['slots'][slot]['fingerprint'] != tensor_fingerprint(expected_params[slot], shape): raise ValueError('Parameter tensor-slot payload binding differs')
    for identity, row in by_consumer.items():
        key, method = identity; t = by_typed[identity]; before, after, evidence = t['input'], t['output'], t['evidence']; check_frame(before); check_frame(after)
        child = child_rows['discovery' if key in request['discovery'] else 'final', key]
        expected_input = source['cases'][key]['H1'] if method == 'native' else child['H0'] if method == 'H0' else next(p for p in child['predictions'] if p['method'] == method)
        payload = expected_input.get('payloadSHA256', expected_input['sha256'])
        if row['inputField'] != expected_input or row['inputPayloadSHA256'] != payload: raise ValueError('Actual consumed state source differs')
        if set(before['slots']) != {*names, 'H1'} or set(after['slots']) != {*names, 'H1', 'Q', 'K', 'V'}: raise ValueError('Typed consumer slot domain differs')
        if before['parent_frame_id'] != parent['frame_id'] or after['parent_frame_id'] != before['frame_id'] or evidence['input_frame_id'] != before['frame_id'] or evidence['output_frame_id'] != after['frame_id'] or evidence['act'] != typed['spec']['identity'] or after['act'] != typed['spec']['identity']: raise ValueError('Typed consumer ancestry differs')
        if before['metadata']['case'] != key or before['metadata']['method'] != method or after['metadata'] != before['metadata']: raise ValueError('Typed consumer identity differs')
        if before['slots']['H1']['fingerprint'] != tensor_fingerprint(payload, [1, 512, 2560]) or after['slots']['H1'] != before['slots']['H1']: raise ValueError('Typed actual input digest differs')
        for slot in names:
            if before['slots'][slot] != parent['slots'][slot] or after['slots'][slot] != parent['slots'][slot]: raise ValueError('Typed native weight ancestry differs')
        calls = evidence['metrics']['calls']
        if [c['port'] for c in calls] != ['Q', 'K', 'V'] or evidence['metrics']['nominalMACs'] != 8053063680: raise ValueError('Actual affine geometry differs')
        if set(row['fields']) != {'Q', 'K', 'V'}: raise ValueError('Native consumer port domain differs')
        for port, width in [('Q', 4096), ('K', 1024), ('V', 1024)]:
            item = row['fields'][port]; call = next(c for c in calls if c['port'] == port)
            if item['shape'] != [1, 512, width] or call['inputShape'] != [1, 512, 2560] or call['weightShape'] != [width, 2560] or call['nominalMACs'] != 512*2560*width: raise ValueError('Typed projection geometry differs')
            if after['slots'][port]['fingerprint'] != tensor_fingerprint(item['payloadSHA256'], item['shape']): raise ValueError('Retained output/typed payload join differs')
            expected_files[item['path']] = item['sha256']; payload_specs[item['path']] = {'sha256': item['payloadSHA256'], 'shape': item['shape']}
    lexical = read(result/'lexical-manifest.json')
    if sha(result/'lexical-manifest.json') != report['lexicalCompilation']['manifestSHA256']: raise ValueError('Fullvocabulary lexical authority differs')
    expected_files[cwd+'/lexical/manifest.json'] = sha(result/'lexical-manifest.json')
    for name, item in lexical['files'].items(): expected_files[cwd+'/lexical/'+name] = item['sha256']
    for key in ('tokenizer', 'template'): expected_files[cwd+'/lexical/'+lexical[key]['path']] = lexical[key]['sha256']
    # Fresh complete byte hashes remain on Beast. This is not a scientific replay.
    code = '''import ast,hashlib,json,stat,struct
from pathlib import Path
expected=json.loads(%r);payloads=json.loads(%r);rows=[]
for name,digest in expected.items():
 p=Path(name);s=p.lstat()
 if not name.startswith(%r+'/') or not stat.S_ISREG(s.st_mode):raise ValueError('Unsafe retained path')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 if h.hexdigest()!=digest:raise ValueError('Fresh retained bytes differ')
 if name in payloads:
  with p.open('rb') as f:
   if f.read(8)!=b'\\x93NUMPY\\x01\\x00':raise ValueError('NPY format differs')
   size=struct.unpack('<H',f.read(2))[0];header=ast.literal_eval(f.read(size).decode('latin1'))
   if header!={'descr':'<u2','fortran_order':False,'shape':tuple(payloads[name]['shape'])}:raise ValueError('NPY word geometry differs')
   body=hashlib.sha256()
   for b in iter(lambda:f.read(8<<20),b''):body.update(b)
   if body.hexdigest()!=payloads[name]['sha256']:raise ValueError('Retained typed tensor payload differs')
 rows.append({'path':name,'bytes':s.st_size,'sha256':digest})
print(json.dumps(rows))
''' % (json.dumps(expected_files), json.dumps(payload_specs), cwd)
    remote = json.loads(subprocess.run(['ssh', '-T', 'beast', 'python3', '-'], input=code, text=True, capture_output=True, check=True, timeout=90).stdout)
    destination.mkdir(exist_ok=False)
    for p in result.iterdir(): shutil.copyfile(p, destination/p.name)
    for p, name in [(root/'collection.json', 'collection.json'), (root/'scheduler-job.json', 'scheduler-job.json'), (Path(mechanics), 'mechanics.json')]: shutil.copyfile(p, destination/name)
    publication = {'schema': 'tce-contextual-writer-publication-v1', 'jobID': job_id, 'requestSHA256': request['requestSHA256'], 'publisherSHA256': sha(__file__),
        'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in destination.iterdir()}, 'independentlyRebuiltPooledScalars': verified,
        'rawObservationRows': rows, 'freshRemoteFiles': remote, 'scope': 'Local metadata/byte/scalar QA only; scientific tensors remain near compute. No new numerical model or array evaluation.'}
    save(destination/'publication.json', publication); return publication


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); parser.add_argument('mechanics'); parser.add_argument('destination'); args = parser.parse_args()
    print(json.dumps(publish(args.root, args.mechanics, args.destination)))
