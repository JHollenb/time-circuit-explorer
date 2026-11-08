"""Source/custody/domain/scalar/typed-byte publisher; no local tensor hydration."""
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
from publish_contextual_writer import check_frame, tensor_fingerprint


def close(a, b):
    if (a is None) != (b is None) or b is not None and (not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > 1e-10*(1+abs(b))):
        raise ValueError('Rebuilt scalar differs')


def pooled(rows):
    if not rows: raise ValueError('Empty scalar population')
    n = len(rows); values = {k: math.sqrt(math.fsum(r[k]**2 for r in rows)/n) for k in ('errorRMS', 'normalizerRMS', 'predictionRMS', 'targetRMS')}
    values['relativeRMS'] = values['errorRMS']/values['normalizerRMS'] if values['normalizerRMS'] else None
    den = values['predictionRMS']*values['targetRMS']
    values['cosine'] = math.fsum((r['cosine'] or 0)*r['predictionRMS']*r['targetRMS'] for r in rows)/n/den if den else None
    return values


def publish(root, mechanics, destination):
    from mrun.client.api import Api
    from contrast_writer_study import verify, OWNED
    root, destination = Path(root).resolve(), Path(destination).resolve(); req = read(root/'request.json'); verify(root/'payload', req)
    for name in OWNED:
        if sha(Path(__file__).parent/name) != req['sourceHashes'][name]: raise ValueError('Current executed source differs')
    job = read(root/'scheduler-job.json'); live = Api().json('GET', '/api/jobs/'+job['job_id']); custody = job['payload_custody']
    if live['state'] != 'succeeded' or live['result']['returncode'] != 0 or live['payload_custody'] != custody or custody['required'] is not True or not live.get('admission'):
        raise ValueError('Current successful guarded custody absent')
    phases = [custody[k] for k in ('declared', 'sealed', 'executed')]
    if not phases[0]['sha256'] or any(any(p[k] != phases[0][k] for k in ('sha256', 'size_bytes')) for p in phases) or phases[2]['host'] != 'beast': raise ValueError('Executed payload differs')
    if job['config']['requestSHA256'] != req['requestSHA256'] or live['config']['requestSHA256'] != req['requestSHA256']: raise ValueError('Scheduler request differs')
    collection = read(root/'collection.json'); result = root/'result'
    if collection['jobID'] != job['job_id'] or collection['custody'] != custody: raise ValueError('Collection custody differs')
    if live['result'] != job['result'] or collection['result'] != job['result']: raise ValueError('Completed scheduler/collection result association differs')
    actual = {str(p.relative_to(result)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in result.rglob('*') if p.is_file()}
    if actual != collection['inventory']['files']: raise ValueError('Complete collection inventory differs')
    for name in req['retainedMetadata']:
        if name == 'request.json':
            if read(result/name) != req: raise ValueError('Collected request differs')
        elif sha(result/name) != req['sourceHashes'][name]: raise ValueError('Collected source metadata differs')
    tests = read(mechanics)
    if tests['testsRun'] != 8 or tests['errors'] or tests['failures']: raise ValueError('Targeted mechanics incomplete')
    for name, digest in tests['sourceHashes'].items():
        if req['sourceHashes'].get(name) != digest or sha(Path(__file__).parent/name) != digest: raise ValueError('Tested source differs')
    report = read(result/'report.json'); source = read(result/'parent-inputs.json'); reuse = read(result/'reuse.json')
    parent_pub = read(result/'parent-publication.json'); parent_cat = read(result/'parent-programs.json'); old_rows = read(result/'parent-consumer-fields.json')['rows']
    parent_live = Api().json('GET', '/api/jobs/'+reuse['jobID'])
    if parent_live['state'] != 'succeeded' or parent_live['payload_custody'] != reuse['custody'] or sha(result/'parent-publication.json') != reuse['publicationSHA256'] or parent_pub['jobID'] != reuse['jobID']:
        raise ValueError('Qualified parent lineage differs')
    for name in ('inputs.json', 'request.json', 'programs.json', 'consumer-fields.json', 'typed-consumers.json', 'lexical-manifest.json'):
        if sha(result/('parent-'+name)) != parent_pub['files'][name]['sha256']: raise ValueError('Parent compact authority differs')
    work = {'units': 6, 'originalLinears': 18, 'nominalMACs': 48318382080, 'normalizations': 6, 'fullBlocks': 0, 'images': 0}
    declared_work = {'consumerUnits': work['units'], **{k: v for k, v in work.items() if k not in ('units', 'normalizations')}}
    if report['requestSHA256'] != req['requestSHA256'] or report['consumerWork'] != work or req['physicalWork'] != declared_work: raise ValueError('Actual physical geometry differs')
    if report['numericalProfile'] != source['numericalProfile'] or report['fitting'] != read(result/'fit.json'): raise ValueError('Fit/profile join differs')
    if sha(result/'feedback.json') != report['feedbackSHA256'] or sha(result/'final-seal.json') != report['transferReleasedAfterSealSHA256']: raise ValueError('Selection/target release seal differs')
    programs = read(result/'programs.json'); feedback = read(result/'feedback.json'); selected = report['selected']
    if set(programs) != {'candidate', 'endpoint', 'position', 'accepted'} or selected not in ('candidate', 'endpoint') or feedback['selected'] != selected or not feedback['exactRestore']:
        raise ValueError('Program/selection domain differs')
    if programs['accepted']['programSHA256'] != programs[selected]['programSHA256'] or feedback['acceptedProgramSHA256'] != programs['accepted']['programSHA256']: raise ValueError('Restored immutable program differs')
    accepted = programs['accepted']['program']
    restored_expected = {'program.json': programs['accepted']['programSHA256'], 'prefix.bf16': accepted['prefixSHA256'], **{d['path']: d['sha256'] for d in accepted['dictionaries']}}
    if feedback['restoredFiles'] != restored_expected: raise ValueError('Complete immutable restore inventory differs')
    for name, prior in [('endpoint', 'accepted'), ('position', 'position')]:
        if programs[name] != parent_cat[prior]: raise ValueError('Frozen control changed')
    for name in ('candidate', 'accepted'):
        p = programs[name]['program']; prior = parent_cat['accepted']['program']
        if p['trees'][0] != prior['trees'][0] or p['dictionaries'][0] != prior['dictionaries'][0] or p['literals'] != prior['literals'] or p['prefixSHA256'] != prior['prefixSHA256']:
            raise ValueError('First slot/feature/prefix budget changed')
    cwd = job['result']['diagnostics']['cwd']; old_cwd = parent_live['result']['diagnostics']['cwd']
    if cwd != '/mnt/hdd8tb/mrun-work/'+job['job_id'] or reuse['lexicalDirectory'] != old_cwd+'/lexical': raise ValueError('Worker roots differ')
    expected_files = dict(reuse['servingAssets']); payloads = {}; children = {}; packets = {}
    forbidden = sorted(set(source['forbiddenServingPaths']) | {r['inputField']['path'] for r in old_rows} | {x['path'] for r in old_rows for x in r['fields'].values()})
    for phase in ('initializer', 'discovery', 'final'):
        child = read(result/(phase+'-serving.json')); seal = read(result/(phase+'-seal.json')); methods = [] if phase == 'initializer' else ['candidate', 'endpoint', 'position'] if phase == 'discovery' else ['candidate', 'endpoint', 'position', 'accepted']
        cases = req['discovery'] if phase != 'final' else [r['id'] for r in req['cases']]
        if [r['id'] for r in child['rows']] != cases or child['loadedNumericalRuntimeRoots'] or child['newNeuralEvaluations'] != 0 or child['blockedImports'] != ['numpy', 'torch'] or child['fileCanaryDenials'] != forbidden:
            raise ValueError('Fresh raw serving domain/isolation differs')
        if sha(result/(phase+'-serving.json')) != seal['receiptSHA256'] or seal['outputs'] != [p for row in child['rows'] for p in [row['H0'], *row['predictions']]]: raise ValueError('Complete serving seal differs')
        for row in child['rows']:
            key = row['id']; packet = row['packet']; children[phase, key] = row; packets[key] = packet
            if packet['positions'] != list(range(512)) or len(packet['ids']) != 512 or any(type(v) is not int or not 0 <= v < 151936 for v in packet['ids']) or len(packet['mask']) != 512:
                raise ValueError('Complete owned packet differs')
            if any(type(v) is not int or v not in (0, 1) for v in packet['mask']) or any(a < b for a, b in zip(packet['mask'], packet['mask'][1:])) or row['activeRows'] != sum(packet['mask']): raise ValueError('Native mask ABI differs')
            if key in source['cases']:
                for label, wanted in [('ids', source['cases'][key]['tokenIDsSHA256']), ('mask', source['cases'][key]['maskSHA256'])]:
                    if hashlib.sha256(__import__('struct').pack('<512q', *packet[label])).hexdigest() != wanted: raise ValueError('Owned native input binding differs')
            if [p['method'] for p in row['predictions']] != methods: raise ValueError('Exact fresh method domain differs')
            for item, method in [(row['H0'], 'H0')]+[(p, p['method']) for p in row['predictions']]:
                name = key+'-H0.bf16' if method == 'H0' else method+'-'+key+'.bf16'
                if item['path'] != cwd+'/'+phase+'/'+name or item['bytes'] != 2621440: raise ValueError('Owned numeric output path/size differs')
                expected_files[item['path']] = item['sha256']
                if method != 'H0' and (item['programSHA256'] != programs[method]['programSHA256'] or len(item['leaves']) != 510 or item['coordinateAdditions'] != 2611200): raise ValueError('Writer state/program binding differs')
    for key in req['discovery']:
        a, b, c = (children[phase, key] for phase in ('initializer', 'discovery', 'final'))
        if not (a['packet'] == b['packet'] == c['packet']) or len({r['H0']['sha256'] for r in (a, b, c)}) != 1: raise ValueError('Fresh initializer reproduction differs')
        for p in b['predictions']:
            if p['sha256'] != next(x['sha256'] for x in c['predictions'] if x['method'] == p['method']): raise ValueError('Fresh discovery reproduction differs')
    for row in read(result/'final-serving.json')['rows']:
        by = {p['method']: p for p in row['predictions']}
        if by['accepted']['sha256'] != by[selected]['sha256']: raise ValueError('Fresh accepted output differs')
    for name, info in programs.items():
        p = info['program']; encoded = json.dumps(p, indent=2, sort_keys=True, allow_nan=False).encode()
        if hashlib.sha256(encoded).hexdigest() != info['programSHA256'] or p['budget'] != req['budget'] or len(p['trees']) != 2: raise ValueError('Program identity/grammar differs')
        folder = Path(info['directory']); expected_files[str(folder/'program.json')] = info['programSHA256']; expected_files[str(folder/'prefix.bf16')] = p['prefixSHA256']
        for tree, dictionary in zip(p['trees'], p['dictionaries']):
            leaves = []
            def visit(node, depth):
                if 'atom' in node: leaves.append(node['atom']); return
                if depth >= 4 or not 0 <= node['feature'] < len(p['featureNames']): raise ValueError('Tree predicate differs')
                visit(node['left'], depth+1); visit(node['right'], depth+1)
            visit(tree, 0)
            if sorted(leaves) != list(range(dictionary['atoms'])) or len(leaves) > 16 or dictionary['bytes'] != 10240*len(leaves): raise ValueError('Leaf/dictionary budget differs')
            expected_files[str(folder/dictionary['path'])] = dictionary['sha256']
    reused = read(result/'reused-fields.json'); expected_reuse = {(k, m) for k in req['discovery']+req['transfer'] for m in ('native', 'H0', 'endpoint', 'position')}
    if len(reused['servingAssetsAuthenticated']) != len(reuse['servingAssets']) or {r['path']: r['sha256'] for r in reused['servingAssetsAuthenticated']} != reuse['servingAssets']: raise ValueError('Current serving asset authentication coverage differs')
    if len(reused['rows']) != 24 or {(r['case'], r['method']) for r in reused['rows']} != expected_reuse: raise ValueError('Reused field domain differs')
    for row in reused['rows']:
        key, method = row['case'], row['method']; previous = reuse['selected'] if method == 'endpoint' and key in req['discovery'] else 'accepted' if method == 'endpoint' else method
        old = next(r for r in old_rows if r['case'] == key and r['method'] == previous)
        own = children['discovery' if key in req['discovery'] else 'final', key]
        item = source['cases'][key]['H1'] if method == 'native' else own['H0'] if method == 'H0' else next(p for p in own['predictions'] if p['method'] == method)
        if row['historicalMethod'] != previous or row['fields'] != old['fields'] or row['inputPayloadSHA256'] != old['inputPayloadSHA256'] or row['inputPayloadSHA256'] != item.get('payloadSHA256', item['sha256']): raise ValueError('Historical exact input/field reuse differs')
        expected_files[old['inputField']['path']] = old['inputField']['sha256']
        for port, item in row['fields'].items(): expected_files[item['path']] = item['sha256']; payloads[item['path']] = {'sha256': item['payloadSHA256'], 'shape': item['shape']}
    consumers = read(result/'consumer-fields.json'); typed = read(result/'typed-consumers.json'); parent = typed['unchangedWeightParent']; check_frame(parent)
    selected_assets = [next(a for a in source['parameterAssets'] if a['path'] == name) for name in dict.fromkeys(p['file'] for p in source['parameters'])]
    if consumers['parameters'] != source['parameters'] or consumers['parameterAssets'] != selected_assets: raise ValueError('Native consumer source authority differs')
    if report['authenticatedWeightShardBytes'] != sum(a['bytes'] for a in selected_assets) or report['selectedParameterBytes'] != sum(p['bytes'] for p in source['parameters']): raise ValueError('Selected parameter cost accounting differs')
    if len(consumers['rows']) != 6 or len(typed['rows']) != 6 or {r['case'] for r in consumers['rows']} != set(req['discovery']+req['transfer']): raise ValueError('New typed consumer domain differs')
    names = {'normWeight': ('input_layernorm.weight', [2560]), 'qWeight': ('self_attn.q_proj.weight', [4096, 2560]), 'kWeight': ('self_attn.k_proj.weight', [1024, 2560]), 'vWeight': ('self_attn.v_proj.weight', [1024, 2560])}
    expected_params = {slot: next(x['payloadSHA256'] for x in source['parameters'] if x['name'] == 'model.layers.1.'+name) for slot, (name, _) in names.items()}
    if typed['parameterPayloadBefore'] != expected_params or typed['parameterPayloadAfter'] != expected_params or set(parent['slots']) != set(names) or typed['spec']['reads'] != ['H1', *names] or typed['spec']['writes'] != ['Q', 'K', 'V']: raise ValueError('Actual selected parameter binding differs')
    for slot, (_, shape) in names.items():
        if parent['slots'][slot]['fingerprint'] != tensor_fingerprint(expected_params[slot], shape): raise ValueError('Native tensor slot fingerprint differs')
    for row in consumers['rows']:
        key = row['case']; t = next(x for x in typed['rows'] if x['case'] == key); before, after, ev = t['input'], t['output'], t['evidence']; check_frame(before); check_frame(after)
        own = children['discovery' if key in req['discovery'] else 'final', key]; item = next(p for p in own['predictions'] if p['method'] == 'candidate')
        if row['method'] != 'candidate' or t['method'] != 'candidate' or row['inputField'] != item or row['inputPayloadSHA256'] != item['sha256']: raise ValueError('Actual candidate input identity differs')
        if before['parent_frame_id'] != parent['frame_id'] or after['parent_frame_id'] != before['frame_id'] or ev['input_frame_id'] != before['frame_id'] or ev['output_frame_id'] != after['frame_id'] or ev['act'] != typed['spec']['identity']: raise ValueError('Consumer ancestry differs')
        if before['metadata'] != {'case': key, 'method': 'candidate'} or after['metadata'] != before['metadata'] or after['act'] != typed['spec']['identity'] or any(f['program_fingerprint'] != req['requestSHA256'] for f in (parent, before, after)): raise ValueError('Typed case/program ancestry differs')
        if set(before['slots']) != {*names, 'H1'} or set(after['slots']) != {*names, 'H1', 'Q', 'K', 'V'} or before['slots']['H1']['fingerprint'] != tensor_fingerprint(item['sha256'], [1, 512, 2560]) or after['slots']['H1'] != before['slots']['H1']: raise ValueError('Actual H1 tensor binding differs')
        for slot in names:
            if before['slots'][slot] != parent['slots'][slot] or after['slots'][slot] != parent['slots'][slot]: raise ValueError('Parameter ancestry differs')
        if ev['metrics']['linears'] != 3 or ev['metrics']['nominalMACs'] != 8053063680 or [c['port'] for c in ev['metrics']['calls']] != ['Q', 'K', 'V']: raise ValueError('Actual native affine work differs')
        for port, width in [('Q', 4096), ('K', 1024), ('V', 1024)]:
            item = row['fields'][port]; call = next(c for c in ev['metrics']['calls'] if c['port'] == port)
            if item['shape'] != [1, 512, width] or call['inputShape'] != [1, 512, 2560] or call['weightShape'] != [width, 2560] or call['nominalMACs'] != 512*2560*width or after['slots'][port]['fingerprint'] != tensor_fingerprint(item['payloadSHA256'], item['shape']): raise ValueError('Retained consumer output/geometry differs')
            expected_files[item['path']] = item['sha256']; payloads[item['path']] = {'sha256': item['payloadSHA256'], 'shape': item['shape']}
    groups = collections.defaultdict(list); raw = {}; expected_domain = set()
    cases = req['discovery']+req['transfer']; methods = ('H0', 'candidate', 'endpoint', 'position'); ports = ('H1', 'Q', 'K', 'V')
    expected_domain.update(('case', k, m, p, i) for k in cases for m in methods for p in ports for i in range(512))
    expected_domain.update(('pair', a, b, m, p, i) for a, b in [('A', 'B'), ('00', '10'), ('01', '11')] for m in methods for p in ports for i in range(512))
    with gzip.open(result/'observations.jsonl.gz', 'rt') as stream:
        for line in stream:
            r = json.loads(line); i = r['position']
            if type(i) is not int or not 0 <= i < 512: raise ValueError('Invalid row position')
            if 'case' in r:
                key = ('case', r['case'], r['method'], r['port'], i); base = key[:-1]; mask = bool(packets[r['case']]['mask'][i])
                if r['active'] is not mask: raise ValueError('Observation active mask differs')
                labels = ['all', 'active' if mask else 'padding']+(['learned'] if i >= 2 else [])
            else:
                a, b = r['pair']; key = ('pair', a, b, r['method'], r['port'], i); base = key[:-1]
                same = packets[a]['ids'][i] == packets[b]['ids'][i]; mask = bool(packets[a]['mask'][i])
                if r['sameToken'] is not same or r['active'] is not mask: raise ValueError('Paired row binding differs')
                labels = ['all']+(['unchangedToken'] if same else [])+(['unchangedLearned', 'unchangedActive' if mask else 'unchangedPadding'] if same and i >= 2 else [])
            if key not in expected_domain or key in raw: raise ValueError('Duplicate/out-of-domain scalar row')
            raw[key] = r
            for label in labels: groups[(*base, label)].append(r)
    if set(raw) != expected_domain or len(raw) != report['observationRows'] or len(raw) != 73728: raise ValueError('Complete observation domain differs')
    count = 0
    for item in report['summaries']+report['pairs']:
        key = ('case', item['case'], item['method'], item['port'], item['stratum']) if 'case' in item else ('pair', item['A'], item['B'], item['method'], item['port'], item['selection'])
        rows = groups.pop(key)
        if len(rows) != item['rows']: raise ValueError('Pooled row coverage differs')
        for name, value in pooled(rows).items(): close(item[name], value); count += 1
    if groups: raise ValueError('Missing pooled scalar groups')
    for method in ('candidate', 'endpoint'):
        scores = feedback['scores'][method]; total = 0.0
        for port in ('Q', 'K', 'V'):
            for kind in ('endpoint', 'response'):
                values = []
                if kind == 'endpoint':
                    for case in req['discovery']:
                        for active in (True, False): values.append(pooled([raw['case', case, method, port, i] for i in range(2, 512) if bool(packets[case]['mask'][i]) is active]))
                else:
                    for a, b in [('A', 'B'), ('00', '10')]:
                        for active in (True, False): values.append(pooled([raw['pair', a, b, method, port, i] for i in range(2, 512) if packets[a]['ids'][i] == packets[b]['ids'][i] and bool(packets[a]['mask'][i]) is active]))
                e = math.fsum(v['errorRMS']**2 for v in values)/len(values); t = math.fsum(v['normalizerRMS']**2 for v in values)/len(values); score = e/t if t else e
                observed = scores['ports'][port][kind]
                for name, value in [('score', score), ('errorMeanSquare', e), ('referenceMeanSquare', t)]: close(observed[name], value)
                if observed['zeroReferenceUsesAbsolute'] is not (not bool(t)): raise ValueError('Zero energy feedback policy differs')
                total += score/3
        close(scores['score'], total)
    expected_selected = 'candidate' if feedback['scores']['candidate']['score'] < feedback['scores']['endpoint']['score'] else 'endpoint'
    if selected != expected_selected: raise ValueError('Discovery-only continuation choice differs')
    # Byte-only remote verification: model arrays never enter this process.
    allowed_roots = sorted({cwd, old_cwd, *(str(Path(source['cases'][k][n]['path']).parents[3]) for k in cases for n in ('H0', 'H1'))})
    code = '''import ast,hashlib,json,stat,struct
from pathlib import Path
expected=json.loads(%r);payloads=json.loads(%r);roots=json.loads(%r);rows=[]
for name,digest in expected.items():
 p=Path(name);s=p.lstat()
 if not stat.S_ISREG(s.st_mode) or not any(name.startswith(r+'/') for r in roots):raise ValueError('Unsafe retained file')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 if h.hexdigest()!=digest:raise ValueError('Fresh retained bytes differ')
 if name in payloads:
  with p.open('rb') as f:
   if f.read(8)!=b'\\x93NUMPY\\x01\\x00':raise ValueError('NPY format differs')
   n=struct.unpack('<H',f.read(2))[0];header=ast.literal_eval(f.read(n).decode('latin1'))
   if header!={'descr':'<u2','fortran_order':False,'shape':tuple(payloads[name]['shape'])}:raise ValueError('NPY ABI differs')
   body=hashlib.sha256()
   for block in iter(lambda:f.read(8<<20),b''):body.update(block)
   if body.hexdigest()!=payloads[name]['sha256']:raise ValueError('Typed raw payload differs')
 rows.append({'path':name,'bytes':s.st_size,'sha256':digest})
print(json.dumps(rows))
''' % (json.dumps(expected_files), json.dumps(payloads), json.dumps(allowed_roots))
    remote = json.loads(subprocess.run(['ssh', '-T', 'beast', 'python3', '-'], input=code, text=True, capture_output=True, check=True, timeout=90).stdout)
    destination.mkdir(exist_ok=False)
    for p in result.iterdir(): shutil.copyfile(p, destination/p.name)
    for p, name in [(root/'collection.json', 'collection.json'), (root/'scheduler-job.json', 'scheduler-job.json'), (Path(mechanics), 'mechanics.json')]: shutil.copyfile(p, destination/name)
    publication = {'schema': 'tce-contrast-writer-publication-v1', 'jobID': job['job_id'], 'requestSHA256': req['requestSHA256'], 'publisherSHA256': sha(__file__),
        'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in destination.iterdir()}, 'rawObservationRows': len(raw), 'independentlyRebuiltPooledScalars': count,
        'freshRemoteFiles': remote, 'scope': 'Source/custody/domain,complete endpoint/response pooled scalars,feedback error/reference score and typed-byte joins. Feedback increment cosine/RMS other than error/reference are retained,not independently reconstructed. No local scientific arrays or new numerical model replay.'}
    save(destination/'publication.json', publication); return publication


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('root'); parser.add_argument('mechanics'); parser.add_argument('destination'); args = parser.parse_args()
    print(json.dumps(publish(args.root, args.mechanics, args.destination)))
