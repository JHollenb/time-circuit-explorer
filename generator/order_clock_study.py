"""One guarded integer recurrence assay; no model or score evaluations."""
from array import array
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from token_history_process import read, save, sha

HERE = Path(__file__).resolve().parent
OWNED = ['order_clock_program.py', 'order_clock_study.py', 'test_order_clock.py', 'token_history_address.py', 'token_history_process.py']


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify(root, request):
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']: raise ValueError('Request identity')
    for name, digest in request['sourceHashes'].items():
        if sha(Path(root)/name) != digest: raise ValueError('Source/input differs '+name)


def stage(root):
    parent = HERE.parent/'docs/results/source-order-program'; publication = read(parent/'publication.json')
    for name, item in publication['files'].items():
        if sha(parent/name) != item['sha256']: raise ValueError('Parent publication differs')
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); payload = root/'payload'; payload.mkdir()
    for name in OWNED: shutil.copyfile(HERE/name, payload/name)
    names = ['compile-order.u16.gz', 'compile-program.json', 'compile-compiler.json', 'prediction-seal.json', 'report.json']
    cases = read(parent/'serve-spec.json')['cases']
    for case in cases:
        name = case['id']; names.append('input-'+name+'-tokens.json')
        names += ['prediction-'+name+'-'+m+'.jsonl.gz' for m in ['primary', 'clockPlus1', 'reverseRanks', 'recent']]
    for name in names: shutil.copyfile(parent/name, payload/name)
    shutil.copyfile(HERE.parent/'docs/plans/order-clock-recurrence.md', payload/'plan.md')
    save(payload/'basis.json', {'parentPublicationSHA256': sha(parent/'publication.json'), 'cases': [{k: c[k] for k in ['id', 'scope']} for c in cases],
                              'scope': 'Cycle95 exact finite compiled order, not an independent neural comparator. Original502333440compileproducts remain paid.'})
    request = {'schema': 'tce-order-clock-request-v1', 'ownedSources': OWNED, 'sourceHashes': {p.name: sha(p) for p in payload.iterdir()},
               'device': 'cpu', 'backend': 'stdlib-integer-copy-insert-v1', 'length': 512, 'alphabetSize': 30, 'head': 0,
               'scope': 'One fixed maximal-contiguous-COPY/INSERT grammar, same finite comparator domain. No format search/new neural score or image.'}
    request['requestSHA256'] = identity(request); save(root/'request.json', request); save(payload/'request.json', request)
    save(payload/'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [p.name for p in payload.iterdir()]+['preflight.v1.json']})
    return {'requestSHA256': request['requestSHA256'], 'payloadInputFiles': len(list(payload.iterdir()))}


def submit(root):
    from mrun.client.submit import launch
    root = Path(root).resolve(); request = read(root/'request.json'); verify(root/'payload', request)
    if (root/'job.json').exists(): raise ValueError('Already submitted')
    cmd = ['/usr/bin/env', 'PYTHONUNBUFFERED=1', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONNOUSERSITE=1', 'PYTHONPATH=',
           'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'python3', '-u', 'order_clock_study.py', 'execute', '.']
    job = launch(cmd, experiment='tce-order-clock-recurrence-v1', task='analysis', payload=root/'payload', prefer_host='beast', needs={'cpu': True},
                 config={k: request[k] for k in ['schema', 'requestSHA256', 'device', 'backend', 'length', 'alphabetSize', 'head']},
                 timeout_s=600, detach=True, retry_on_kill=False, preflight=True, guarded=True,
                 note='Exact finite order recurrence:fixed5B-COPY/3B-INSERT/2Bclockcounts;compile3924480savedordinals,no scores/modelcalls. Freshstdlib transports admitted ranks;full-order proof paidseparately.13packetsx4methods,sealedbefore512clock/26624oldrowcomparisons. Chargeallcopies/state/updatework/programbytes. IntentfirstCPU,softBeast,600scap,noretries;stopafteronegrammar,retainlarger/slowerresult.')
    save(root/'job.json', {'jobID': job, 'requestSHA256': request['requestSHA256']}); return job


def execute(root):
    from order_clock_program import compile_program, METHODS
    root = Path(root).resolve(); request = read(root/'request.json'); verify(root, request); started = time.perf_counter()
    out = root/'response-result'; out.mkdir(exist_ok=False); compiled = root/'compiled'; compiled.mkdir()
    source = read(root/'compile-program.json'); raw = gzip.decompress((root/'compile-order.u16.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest() != source['rawSHA256']: raise ValueError('Original raw rank identity')
    ranks = array('H'); ranks.frombytes(raw)
    if sys.byteorder != 'little': ranks.byteswap()
    code, compiler = compile_program(ranks, source['alphabet'], source['length']); compiled_at = time.perf_counter()
    with (compiled/'program.bin.gz').open('xb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb', filename='', mtime=0, compresslevel=9) as stream: stream.write(code)
    metadata = {'schema': 'tce-order-clock-v1', 'head': 0, 'paddingQueryID': source['paddingQueryID'], 'alphabet': source['alphabet'], 'length': source['length'],
                'rawBytes': len(code), 'rawSHA256': hashlib.sha256(code).hexdigest(), 'compressedBytes': (compiled/'program.bin.gz').stat().st_size,
                'compressedSHA256': sha(compiled/'program.bin.gz'), 'grammar': 'COPY0-u16start-u16length;INSERT1-u16token;u16opcountperclock;littleendian',
                'scope': 'Candidate encoding of95finite source-order program; not independent neural computation/semantic sufficiency.'}
    save(compiled/'metadata.json', metadata); compiler['serializationSeconds'] = time.perf_counter()-compiled_at; save(compiled/'compiler.json', compiler)
    forbidden = [str(root/n) for n in request['sourceHashes'] if n.startswith(('compile-', 'prediction-'))]
    cases = read(root/'basis.json')['cases']
    allowed = [str(compiled/n) for n in ['metadata.json', 'program.bin.gz']]+[str(root/('input-'+c['id']+'-tokens.json')) for c in cases]
    spec = {'metadata': allowed[0], 'metadataSHA256': sha(compiled/'metadata.json'), 'bytecode': allowed[1],
            'readAllow': allowed, 'forbiddenPaths': forbidden, 'cases': [{'id': c['id'], 'tokens': str(root/('input-'+c['id']+'-tokens.json'))} for c in cases]}
    save(root/'serve-spec.json', spec); begin = time.perf_counter()
    subprocess.run([sys.executable, '-I', '-B', str(root/'order_clock_program.py'), str(root/'serve-spec.json'), str(root/'served')], check=True)
    serve_seconds = time.perf_counter()-begin
    seal = {'requestSHA256': request['requestSHA256'], 'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in (root/'served').iterdir()},
            'scope': 'Fresh child exited; no old clock hashes/prediction rows interpreted before this seal.'}; save(out/'prediction-seal.json', seal)
    proof = read(root/'served/decoded-clocks.json'); old_ledger = read(root/'compile-compiler.json')['clocks']
    if len(proof['clocks']) != len(old_ledger) or len(proof['clocks']) != source['length']: raise ValueError('Complete clock census')
    clock_differences = [a['clock'] for a, b in zip(proof['clocks'], old_ledger) if a['rankSHA256'] != b['rankSHA256']]
    panels = []; total = mismatches = 0
    for case in cases:
        for method in METHODS:
            def stream(path):
                with gzip.open(path, 'rt') as f: return [json.loads(line) for line in f]
            actual = stream(root/'served'/(case['id']+'-'+method+'.jsonl.gz'))
            old = stream(root/('prediction-'+case['id']+'-'+method+'.jsonl.gz'))
            if len(actual) != 512 or len(old) != 512: raise ValueError('Complete serving packet')
            different = [i for i, (a, b) in enumerate(zip(actual, old)) if a != b]; total += len(actual); mismatches += len(different)
            panels.append({'case': case['id'], 'method': method, 'rows': len(actual), 'mismatchedClocks': different,
                           'supported': sum(r['status'] == 'supported' for r in actual)})
    verify(root, request)
    encoding_integrity_matches = not clock_differences and mismatches == 0 and proof['sparseRankProof']['mismatchedOrdinals'] == 0
    save(out/'report.json', {'schema': 'tce-order-clock-result-v1', 'requestSHA256': request['requestSHA256'], 'program': metadata,
         'compiler': compiler, 'clockHashMismatches': clock_differences, 'servingMethodRows': total, 'servingRowMismatches': mismatches,
         'panels': panels, 'freshChildWallSeconds': serve_seconds, 'workerWallSecondsBeforeReport': time.perf_counter()-started,
         'newLearnedScoreProducts': 0, 'newOriginalModelEncoderDenoiserVAEEvaluations': 0, 'newImages': 0,
         'mechanics_status': 'complete' if encoding_integrity_matches else 'evidence-mismatch',
         'instrument_status': 'exact-compiled-order-encoding' if encoding_integrity_matches else 'instrument-broken-encoding-mismatch',
         'trend_status': 'All storage/time/state measurements retained without quality gate',
         'terminal_status': 'arbitrary-prompt zero-neural image generation unestablished', 'scope': request['scope']})
    for directory, prefix in [(compiled, 'compile-'), (root/'served', 'prediction-')]:
        for p in directory.iterdir(): shutil.copyfile(p, out/(prefix+p.name))
    for name in ['request.json', 'basis.json', 'plan.md', 'serve-spec.json']: shutil.copyfile(root/name, out/name)
    save(out/'result-files.json', {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in out.iterdir()})
    print('ORDER_CLOCK_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']); p.add_argument('root'); args = p.parse_args()
    if args.action == 'collect':
        from response_study import collect
        value = collect(args.root)
    else: value = globals()[args.action](args.root)
    if value is not None: print(json.dumps(value))
