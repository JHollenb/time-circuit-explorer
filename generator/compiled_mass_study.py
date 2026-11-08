"""One guarded offline compilation -> sealed stdlib serving -> diagnostics."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from token_history_process import read, save, sha, write_rows

HERE = Path(__file__).resolve().parent
OWNED = ['compiled_mass_program.py', 'compiled_mass_compile.py', 'compiled_mass_study.py', 'test_compiled_mass.py',
         'source_order_program.py', 'source_order_compile.py', 'encoder_atoms.py', 'token_history_address.py', 'token_history_process.py']
OLD = {'compile-program.json': ('rank-program.json', '3971cce66cd288193b3b600669dc11bbc26bbc8e600bad512b1edb8ec8da747c'),
       'compile-order.u16.gz': ('ranks.u16.gz', 'f3f64e04a3f326d460ea1d7237d219f72537c17db9dda31f4ef2b8240db3dcf7'),
       'compile-compiler.json': ('clock-ledger.json', '084a12ec5ff814d3ba470ee9b761bc29cafbf76ace84fae524935241c2069d6f'),
       'input-A-tokens.json': ('A-tokens.json', '7265b97bdaab561215d221f083fd2e1ff02c120da1196a4aeabac001bcd538ea'),
       'input-B-tokens.json': ('B-tokens.json', 'f29609947047c30b4e91d4fd254eaf305c665ae6e3785d42a3d94f29e1a03980')}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify(root, request):
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']:
        raise ValueError('Frozen request mismatch')
    for name, digest in request['sourceHashes'].items():
        if sha(Path(root)/name) != digest:
            raise ValueError('Frozen payload mismatch: '+name)


def source_custody(observed, retained):
    for custody in [observed, retained]:
        if not custody or custody.get('required') is not True:
            raise ValueError('Required historical source custody missing')
        phases = [custody.get(k) or {} for k in ['declared', 'sealed', 'executed']]
        if not phases[0].get('sha256') or not phases[0].get('size_bytes'):
            raise ValueError('Historical source declaration missing')
        if any(any(p.get(k) != phases[0][k] for k in ['sha256', 'size_bytes']) for p in phases) or phases[-1].get('host') != 'beast':
            raise ValueError('Historical source custody closure mismatch')
    if any(observed['declared'][k] != retained['declared'][k] for k in ['sha256', 'size_bytes']):
        raise ValueError('Fresh historical custody differs from retained original')
    return {'matchesRetainedOriginal': True, **retained['declared'], 'host': 'beast',
            'scope': 'Original declared/sealed/executed identity and host matched; no stronger historical claim added.'}


def synthetic(alphabet):
    ids = [alphabet[(7*p+3) % len(alphabet)] if p < 35 else 151643 for p in range(512)]
    packets = {name: {'ids': ids.copy(), 'mask': [int(p < 35) for p in range(512)], 'positions': list(range(512))}
               for name in ['C', 'D', 'unknown']}
    packets['D']['ids'][8] = alphabet[(alphabet.index(ids[8])+1) % len(alphabet)]
    if 151935 in alphabet or 151643 in alphabet or packets['D']['ids'][8] == ids[8]:
        raise ValueError('Controlled token contrast/domain mismatch')
    packets['unknown']['ids'][8] = 151935
    return packets


def stage(root):
    from mrun.client.api import Api
    api = Api()
    previous = HERE.parent/'docs/results/source-order-program'
    publication = read(previous/'publication.json')
    if sha(previous/'publication.json') != '83cf31777e4c598db0c55af48db40942ca05bb178133d29d4d25f9707dd8940d':
        raise ValueError('Original95 publication mismatch')
    for name in ['inputs.json', 'scheduler-job.json']:
        if sha(previous/name) != publication['files'][name]['sha256']:
            raise ValueError('Retained historical source witness mismatch')
    original = read(previous/'inputs.json')
    retained95 = read(previous/'scheduler-job.json')
    historical_custody = {'job-7dcbeae35deb': original['atomOriginalJob']['payloadCustody'],
                          'job-c22fba1e07cd': retained95['payload_custody']}
    source_files = [x for x in original['intake']['files'] if x['path'].startswith(original['store']+'/')]
    code = """import hashlib,json,stat
from pathlib import Path
rows=[]
for item in json.loads(%r):
 p=Path(item['path']);s=p.lstat()
 if not stat.S_ISREG(s.st_mode):raise ValueError('Nonregular source')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 if h.hexdigest()!=item['sha256'] or s.st_size!=item['bytes']:raise ValueError('Source identity mismatch')
 rows.append({'path':str(p),'bytes':s.st_size,'sha256':h.hexdigest()})
print(json.dumps({'files':rows,'bytes':sum(x['bytes'] for x in rows)}))
""" % json.dumps(source_files)
    intake = json.loads(subprocess.run(['ssh', '-T', 'beast', 'python3', '-'], input=code, capture_output=True, text=True, check=True, timeout=60).stdout)
    prior_jobs = []
    for job_id in ['job-7dcbeae35deb', 'job-c22fba1e07cd']:
        job = api.json('GET', '/api/jobs/'+job_id)
        if job['state'] != 'succeeded':
            raise ValueError('Historical source job not successfully terminal')
        prior_jobs.append({'jobID': job_id, 'state': job['state'], 'payloadCustody': job.get('payload_custody'),
                           'matchedOriginalCustody': source_custody(job.get('payload_custody'), historical_custody[job_id])})
    root = Path(root).resolve();root.mkdir(parents=True, exist_ok=False)
    payload = root/'payload';payload.mkdir()
    for name in OWNED:
        shutil.copyfile(HERE/name, payload/name)
    for source, (destination, digest) in OLD.items():
        if sha(previous/source) != digest:
            raise ValueError('Original95 program/ledger/input mismatch')
        shutil.copyfile(previous/source, payload/destination)
    old_weights = HERE.parent/'docs/results/first-read-route-payload/rows.jsonl.gz'
    if sha(old_weights) != '0b12bcbf1dd816a01a048199983f47817dc6035a7a172e2bdd654292fb54e7dc':
        raise ValueError('Historical97 public observations mismatch')
    shutil.copyfile(old_weights, payload/'old-weights.jsonl.gz')
    rank = read(payload/'rank-program.json')
    if (rank['length'], rank['head'], rank['paddingQueryID'], len(rank['alphabet']), rank['ordinalCount']) != (512, 0, 151643, 30, 3924480):
        raise ValueError('Fixed study domain mismatch')
    for name, packet in synthetic(rank['alphabet']).items():
        save(payload/(name+'-tokens.json'), packet)
    for source, destination in [('compiled-selected-mass.md', 'frozen-plan.md'), ('compiled-selected-mass-implementation.md', 'implementation.md')]:
        shutil.copyfile(HERE.parent/'docs/plans'/source, payload/destination)
    save(payload/'inputs.json', {'store': original['store'], 'manifestSHA256': original['manifestSHA256'],
         'alphabet': rank['alphabet'], 'intake': intake, 'historicalJobs': prior_jobs,
         'prior95PublicationSHA256': sha(previous/'publication.json'), 'prior95InputsSHA256': sha(previous/'inputs.json'),
         'retainedSourceWitnesses': {name: publication['files'][name]['sha256'] for name in ['inputs.json', 'scheduler-job.json']},
         'oldWeightScope': 'Public97 scalar weights/supports; numeric interpretation only after candidate seal.',
         'qualification': 'Complete recomputed clock scores match qualified95 hashes; original lexical NPZs not reopened.'})
    host = next(x for x in api.json('GET', '/api/hosts') if x['name'] == 'beast')
    save(payload/'resource-basis.json', {'rawScoreBytes': 31395840, 'rawRankBytes': 7848960,
         'compileCoordinateProducts': 502333440, 'referenceCoordinateProducts': 8547840,
         'sourceFileBytesFreshHashed': intake['bytes'], 'maximumProductBufferBytes': 128*128*8,
         'liveHost': {k: host.get(k) for k in ['name', 'active_jobs', 'admission', 'telemetry', 'ram_total_mb']},
         'policy': 'Intent-first CPU/host-local Beast/BLAS1/no retries/600s cap. Expected tens of seconds; no neural runtime or GPU.'})
    request = {'schema': 'tce-compiled-mass-request-v1', 'sourceHashes': {p.name: sha(p) for p in payload.iterdir()},
               'ownedSources': OWNED, 'cases': ['A', 'B', 'C', 'D', 'unknown'], 'device': 'cpu',
               'backend': 'numpy2.5.4 FP64 elementwise offline;stdlib score lookup/exp/fsum serving;Python scalar reference',
               'timeoutSeconds': 600, 'compileProducts': 502333440, 'referenceProducts': 8547840,
               'scope': 'One selected-weight producer only; scalar attention and offline learned compilation paid. No value/contextual/denoising/pixel or arbitrary-prompt zero-neural claim.'}
    request['requestSHA256'] = identity(request)
    save(root/'request.json', request);save(payload/'request.json', request)
    save(payload/'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'root': str(root), 'requestSHA256': request['requestSHA256'], 'freshSourceBytes': intake['bytes'], 'cases': request['cases']}


def submit(root):
    from mrun.client.submit import launch
    root = Path(root).resolve();request = read(root/'request.json');verify(root/'payload', request)
    if (root/'job.json').exists():
        raise ValueError('Already submitted; inspect the same handle')
    command = ['/usr/bin/env', 'PYTHONUNBUFFERED=1', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONNOUSERSITE=1', 'PYTHONPATH=',
               'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache',
               'UV_LINK_MODE=symlink', 'uv', 'run', '--no-project', '--offline', '--python', '3.14.6', '--with', 'numpy==2.5.4',
               'python', '-u', 'compiled_mass_study.py', 'execute', '.']
    job = launch(command, experiment='tce-compiled-selected-mass-v1', task='analysis', payload=root/'payload', pin='beast', needs={'cpu': True},
                 config={k: request[k] for k in ['schema', 'requestSHA256', 'device', 'backend', 'compileProducts', 'referenceProducts']},
                 timeout_s=600, detach=True, retry_on_kill=False, preflight=True, guarded=True,
                 note='Fixed95pad/head0/30IDs/L512:502.333Mofflineproducts/3.924Mfsums->31.4MBscorefile staysBeast;512oldscoreSHA checks. Freshstdlibownrank+selectedexp/fsum A/B/fixedC-D/unknown,sealedbefore8547840Pythonreferenceproducts/66780fsums+old97scalarreads. Scalarattentionpaid,noQKonline/noV/context/image. IntentfirstCPU/BLAS1,hostlocalsource,600scap,noretries/noGPU;measureauth/load/route/mass/serialize/RSS separately.')
    value = {'jobID': str(job), 'requestSHA256': request['requestSHA256']}
    save(root/'job.json', value)
    return value


def load_rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def diagnostics(root, packets, q, keys, alphabet):
    start = time.perf_counter()
    # This function is called only after the complete five-case seal is written.
    old = {r['queryPosition']: r for r in load_rows(root/'old-weights.jsonl.gz')}
    predictions = {name: load_rows(root/'served'/(name+'.jsonl.gz')) for name in packets}
    counters = {'coordinateProducts': 0, 'scoreFsumCalls': 0, 'normalizerExpCalls': 0, 'normalizerFsumCalls': 0}
    index = {token: i for i, token in enumerate(alphabet)}
    records, panels = [], []
    for name, packet in packets.items():
        statuses = Counter();panel = {'case': name, 'referenceRows': 0, 'supportDifferences': 0,
                                     'scoreWordDifferences': 0, 'weightWordDifferences': 0,
                                     'maximumAbsWeightDifference': 0., 'historicalRows': 0,
                                     'historicalSupportDifferences': 0, 'historicalMaximumAbsWeightDifference': 0.}
        history = [(j, keys[index[t], j].tolist()) for j, (t, active) in enumerate(zip(packet['ids'], packet['mask'])) if active] if name != 'unknown' else []
        for candidate in predictions[name]:
            clock = candidate['clock'];statuses[candidate['status']] += 1
            reference = historical = None
            if name != 'unknown' and not packet['mask'][clock]:
                query = q[clock].tolist()
                values = {}
                for position, key in history:
                    if position >= clock:
                        continue
                    values[position] = math.fsum(a*b for a, b in zip(query, key))
                    counters['coordinateProducts'] += len(query);counters['scoreFsumCalls'] += 1
                addresses = sorted(values, key=lambda j: (-values[j], j, packet['ids'][j]))[:4]
                selected = [values[j] for j in addresses]
                if selected:
                    maximum = max(selected)
                    masses = [math.exp(x-maximum) for x in selected]
                    denominator = math.fsum(masses)
                    weights = [x/denominator for x in masses]
                else:
                    weights = []
                counters['normalizerExpCalls'] += len(selected);counters['normalizerFsumCalls'] += int(bool(selected))
                reference = {'addresses': addresses+[-1]*(4-len(addresses)), 'selectedScores': selected, 'weights': weights}
                panel['referenceRows'] += 1
                same = candidate['addresses'] == reference['addresses']
                panel['supportDifferences'] += int(not same)
                if same:
                    panel['scoreWordDifferences'] += sum(a.hex() != b.hex() for a, b in zip(candidate['selectedScores'], selected))
                    panel['weightWordDifferences'] += sum(a.hex() != b.hex() for a, b in zip(candidate['weights'], weights))
                    differences = [a-b for a, b in zip(candidate['weights'], weights)]
                    reference['signedCandidateMinusReferenceWeights'] = differences
                    panel['maximumAbsWeightDifference'] = max([panel['maximumAbsWeightDifference']]+[abs(x) for x in differences])
                if name in ['A', 'B']:
                    target = old[clock]
                    old_addresses, old_weights = target['support'+name], target['weights'+name]
                    panel['historicalRows'] += 1;panel['historicalSupportDifferences'] += int(candidate['addresses'] != old_addresses)
                    new_by_position = dict(zip(candidate['addresses'], candidate['weights']))
                    old_by_position = dict(zip(old_addresses, old_weights))
                    union = sorted(set(new_by_position)|set(old_by_position))
                    difference = [new_by_position.get(j, 0.)-old_by_position.get(j, 0.) for j in union]
                    panel['historicalMaximumAbsWeightDifference'] = max([panel['historicalMaximumAbsWeightDifference']]+[abs(x) for x in difference])
                    historical = {'addresses': old_addresses, 'weights': old_weights, 'unionAddresses': union, 'signedCandidateMinusHistoricalWeights': difference}
            records.append({'case': name, 'clock': clock, 'candidate': candidate, 'sameProfileScalarReference': reference, 'historical97': historical})
        panel['statuses'] = dict(statuses);panels.append(panel)
    pairs = []
    for left, right in [('A', 'B'), ('C', 'D')]:
        rows = []
        for a, b in zip(predictions[left], predictions[right]):
            if a['status'] != 'supported' or b['status'] != 'supported':
                continue
            av, bv = dict(zip(a['addresses'], a['weights'])), dict(zip(b['addresses'], b['weights']))
            union = sorted(set(av)|set(bv))
            delta = [bv.get(j, 0.)-av.get(j, 0.) for j in union]
            rows.append({'clock': a['clock'], 'orderedSupportChanged': a['addresses'] != b['addresses'],
                         'unionAddresses': union, 'signedWeightChange': delta, 'weightL1Change': math.fsum(abs(x) for x in delta)})
        pairs.append({'pair': [left, right], 'rows': rows, 'changedOrderedSupports': sum(x['orderedSupportChanged'] for x in rows),
                      'changedWeightsWithSameOrderedSupport': sum(not x['orderedSupportChanged'] and x['weightL1Change'] > 0 for x in rows),
                      'zeroWeightChanges': sum(x['weightL1Change'] == 0 for x in rows)})
    if counters['coordinateProducts'] != 8547840 or counters['scoreFsumCalls'] != 66780:
        raise ValueError('Fixed diagnostic work geometry mismatch')
    write_rows(root/'response-result/observations.jsonl.gz', records)
    save(root/'response-result/panels.json', panels);save(root/'response-result/pairs.json', pairs)
    return {'panels': panels, 'diagnosticWork': counters, 'diagnosticSeconds': time.perf_counter()-start,
            'referenceScope': 'Independently coded Python scalar coordinate products on the same qualified operands/profile. Same libm policy, not an independent native reference. Old97 differences are observations.'}


def execute(root):
    start, cpu = time.perf_counter(), time.process_time()
    from encoder_atoms import guard
    guard()
    from compiled_mass_compile import compile_mass
    from source_order_compile import source_operands
    root = Path(root).resolve();request = read(root/'request.json');verify(root, request);source = read(root/'inputs.json')
    for item in source['intake']['files']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Source bytes changed since stage')
    q, keys, q_words, k_words, profile = source_operands(source['store'], source['manifestSHA256'], source['alphabet'], 151643, 512)
    del q_words, k_words
    operand_hashes = {'q': hashlib.sha256(q.tobytes()).hexdigest(), 'keys': hashlib.sha256(keys.tobytes()).hexdigest()}
    ready = time.perf_counter()
    metadata = compile_mass(q, keys, read(root/'rank-program.json'), read(root/'clock-ledger.json'), root/'rank-program.json', root/'compiled')
    compiled = time.perf_counter()
    packets = {name: read(root/(name+'-tokens.json')) for name in request['cases']}
    cases = [{'id': name, 'tokens': str(root/(name+'-tokens.json')), 'tokensSHA256': sha(root/(name+'-tokens.json'))} for name in packets]
    assets = {'metadata': str(root/'compiled/program.json'), 'rankMetadata': str(root/'rank-program.json'),
              'ranks': str(root/'ranks.u16.gz'), 'scores': str(root/'compiled/scores.f64')}
    spec = {**assets, 'metadataSHA256': sha(assets['metadata']), 'cases': cases,
            'readAllow': list(assets.values())+[c['tokens'] for c in cases],
            'forbiddenPaths': [str(root/'old-weights.jsonl.gz'), str(root/'clock-ledger.json'), str(root/'inputs.json'),
                               str(root/'compiled_mass_compile.py')]+[x['path'] for x in source['intake']['files']]}
    save(root/'serve-spec.json', spec)
    child_start = time.perf_counter()
    subprocess.run([sys.executable, '-I', '-B', str(root/'compiled_mass_program.py'), str(root/'serve-spec.json'), str(root/'served')], cwd=root, check=True)
    served = time.perf_counter()
    out = root/'response-result';out.mkdir()
    sealed = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in (root/'served').iterdir()}
    save(out/'prediction-seal.json', {'schema': 'tce-compiled-mass-prediction-seal-v1', 'requestSHA256': request['requestSHA256'],
         'metadataSHA256': spec['metadataSHA256'], 'scoresSHA256': metadata['scoreSHA256'], 'files': sealed,
         'scope': 'All five child outputs/receipt sealed and child exited before old weight interpretation or direct reference formation.'})
    print('COMPILED_MASS_SEALED '+sha(out/'prediction-seal.json'), flush=True)
    result = diagnostics(root, packets, q, keys, source['alphabet'])
    if sealed != {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in (root/'served').iterdir()} or sha(root/'compiled/scores.f64') != metadata['scoreSHA256']:
        raise ValueError('Predictions/program mutated during diagnostics')
    if operand_hashes != {'q': hashlib.sha256(q.tobytes()).hexdigest(), 'keys': hashlib.sha256(keys.tobytes()).hexdigest()}:
        raise ValueError('Source operands mutated')
    for item in source['intake']['files']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Read-only source closure differs')
    save(out/'retained-score-file.json', {'host': 'beast', 'path': str(root/'compiled/scores.f64'), 'bytes': metadata['scoreBytes'],
         'sha256': metadata['scoreSHA256'], 'clockLedgerSHA256': sha(root/'compiled/compiler.json'), 'scope': 'Complete compiled array retained near compute; not collected into public result.'})
    save(out/'execution.json', {'profile': profile, 'operandHashes': operand_hashes,
         'sourceAuthenticationAndFormationSeconds': ready-start, 'compilationWithSerializationSeconds': compiled-ready,
         'servePreparationSeconds': child_start-compiled, 'freshServeProcessWallSeconds': served-child_start,
         'workerWallSecondsBeforeReport': time.perf_counter()-start,
         'workerCPUSecondsBeforeReport': time.process_time()-cpu, 'sourceClosure': True,
         'originalModelEncoderDenoiserVAEEvaluations': 0, 'newImages': 0,
         'scope': 'Worker time starts at execute entry, includes import/request/source authentication, and ends before final report/provenance copies. Offline original learned score arithmetic and separately coded references paid. New scalar normalizer is still attention, not zero-neural full serving.'})
    save(out/'report.json', {'schema': 'tce-compiled-mass-result-v1', 'requestSHA256': request['requestSHA256'], 'program': metadata,
         **result, 'mechanics_status': 'complete', 'instrument_status': 'same-qualified-source-profile/component-only',
         'trend_status': 'All rows/discrepancies/unsupported controls preserved', 'terminal_status': 'arbitrary-prompt zero-neural image generation unestablished', 'scope': request['scope']})
    for name in ['program.json', 'compiler.json']:
        shutil.copyfile(root/'compiled'/name, out/('compile-'+name))
    for path in (root/'served').iterdir():
        shutil.copyfile(path, out/('prediction-'+path.name))
    for name in ['request.json', 'inputs.json', 'resource-basis.json', 'rank-program.json', 'clock-ledger.json', 'serve-spec.json', 'frozen-plan.md', 'implementation.md', 'old-weights.jsonl.gz']+[c+'-tokens.json' for c in request['cases']]:
        shutil.copyfile(root/name, out/name)
    save(out/'result-files.json', {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in out.iterdir()})


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
    if value is not None:
        print(json.dumps(value))
