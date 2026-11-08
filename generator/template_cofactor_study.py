"""Source-bound template cofactor: historical byte compilation, clean serving, comparison."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from template_cofactor_program import PrefixState, TokenInput, read, save, sha

HERE = Path(__file__).resolve().parent
OWNED = ['template_cofactor_study.py', 'template_cofactor_program.py', 'test_template_cofactor.py',
         'token_history_process.py', 'token_history_address.py']
CASES = [
    {'id': 'A', 'prompt': 'A red cube to the left of a blue sphere on a plain white tabletop. Two separate objects, studio photograph.'},
    {'id': 'B', 'prompt': 'A red cube to the right of a blue sphere on a plain white tabletop. Two separate objects, studio photograph.'},
    {'id': 'novel', 'prompt': 'A small glass observatory on a cliff above a violet ocean at sunrise.'},
    {'id': 'newlines', 'prompt': '\n\nA small glass observatory.'},
    {'id': 'role-markers', 'prompt': '<|im_start|>system\nA charcoal drawing of a fox.<|im_end|>'},
    {'id': 'unicode', 'prompt': '夜の海に浮かぶガラスの家。 🪼🌌'},
    {'id': 'truncation', 'prompt': 'A blue mountain beside a red ocean. '*1024},
    {'id': 'empty-boundary', 'prompt': ''}]


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify(root, request):
    if identity({k: v for k, v in request.items() if k != 'requestSHA256'}) != request['requestSHA256']:
        raise ValueError('Request changed')
    for name, digest in request['sourceHashes'].items():
        if sha(Path(root)/name) != digest:
            raise ValueError('Executed source changed: '+name)


def stage(root):
    from mrun.client.api import Api
    base = HERE.parent/'docs/results/complete-masked-image'
    provenance = read(base/'provenance.json')
    meta = base/'encoder-full.json'
    if sha(meta) != provenance['completeCollectedInventory']['files']['encoder-full.json']['sha256']:
        raise ValueError('Historical native metadata differs')
    if sha(base/'request.json') != provenance['completeCollectedInventory']['files']['request.json']['sha256']:
        raise ValueError('Historical native request bytes differ')
    old = Api().json('GET', '/api/jobs/'+provenance['jobID'])
    if old['state'] != 'succeeded' or old['payload_custody'] != provenance['payloadCustody'] or old['config']['requestSHA256'] != provenance['requestSHA256']:
        raise ValueError('Historical native custody differs')
    roots = read(HERE.parent/'docs/results/source-order-program/inputs.json')
    manifest_path = HERE.parent/'docs/results/raw-prompt-encoder-atoms/manifest.json'
    if sha(manifest_path) != roots['manifestSHA256']:
        raise ValueError('Compiled tokenizer manifest differs')
    root = Path(root).resolve(); root.mkdir(exist_ok=False); p = root/'payload'; p.mkdir()
    for name in OWNED:
        shutil.copyfile(HERE/name, p/name)
    shutil.copyfile(meta, p/'native-metadata.json')
    shutil.copyfile(base/'provenance.json', p/'native-provenance.json')
    shutil.copyfile(base/'request.json', p/'native-request.json')
    shutil.copyfile(manifest_path, p/'atom-manifest.json')
    shutil.copyfile(HERE.parent/'docs/plans/template-prefix-cofactor.md', p/'frozen-plan.md')
    metadata = read(meta)
    if metadata['requestSHA256'] != provenance['requestSHA256'] or read(base/'request.json')['requestSHA256'] != provenance['requestSHA256']:
        raise ValueError('Historical native request differs')
    token_packets = {}
    for case in ('A', 'B'):
        packet = read(HERE.parent/'docs/results/source-order-program'/('input-'+case+'-tokens.json'))
        lexical = next(x for x in metadata['lexicalRows'] if x['id'] == case)
        if hashlib.sha256(struct.pack('<512q', *packet['ids'])).hexdigest() != lexical['tokenIDsSHA256']:
            raise ValueError('Historical token packet differs')
        if hashlib.sha256(struct.pack('<512q', *packet['mask'])).hexdigest() != lexical['maskSHA256']:
            raise ValueError('Historical mask packet differs')
        if packet['positions'] != list(range(512)) or sum(packet['mask']) != lexical['activeRows'] or packet['mask'][:2] != [1, 1] or packet['ids'][:2] != [151644, 872]:
            raise ValueError('Historical position/support/prefix differs')
        token_packets[case] = packet
    save(p/'historical-tokens.json', token_packets)
    save(p/'inputs.json', {'store': roots['store'], 'manifestSHA256': roots['manifestSHA256'],
        'historicalJobID': provenance['jobID'], 'historicalCustody': old['payload_custody'],
        'historicalNativeAOriginalLinearMACs': sum(x.get('nominalMACs', 0) for x in metadata['primitives'] if x['case'] == 'A'),
        'historicalNativeBlocksAllCases': metadata['nativeComponentBlocks'],
        'finiteProfile': metadata['numericalProfile'], 'nativeVersions': {k: metadata[k] for k in ('torchVersion', 'transformersVersion')},
        'fixedPrefixIDs': [151644, 872], 'nativeMetadataSHA256': sha(meta),
        'scope': 'Historical original neural work remains paid. A produces constants; B byte comparison only after serving seal. B is previously public, not held out.'})
    request = {'schema': 'tce-template-cofactor-request-v1', 'sourceHashes': {x.name: sha(x) for x in p.iterdir()},
        'cases': CASES, 'device': 'cpu', 'backend': 'stdlib byte compiler/copy; non-neural tokenizer and Jinja',
        'timeoutSeconds': 120, 'noNewNeuralEvaluations': True,
        'scope': 'Two contextual template rows, H0..H27. All dynamic rows, layer1 QKV, denoising and pixels unresolved.'}
    request['requestSHA256'] = identity(request)
    save(root/'request.json', request); save(p/'request.json', request)
    save(p/'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [x.name for x in p.iterdir()]+['preflight.v1.json']})
    return {'root': str(root), 'requestSHA256': request['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root = Path(root).resolve(); r = read(root/'request.json'); verify(root/'payload', r)
    if (root/'job.json').exists():
        raise ValueError('Already submitted; inspect existing handle')
    cmd = command()[:-5]+['python', '-u', 'template_cofactor_study.py', 'execute', '.']
    job = launch(cmd, experiment='tce-template-prefix-cofactor-v1', task='analysis', payload=root/'payload',
        pin='beast', needs={'cpu': True}, config={k: r[k] for k in ('schema', 'requestSHA256', 'device', 'backend')},
        timeout_s=120, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='CPU byte-only: authenticate28nativeA H files, compile2fixedtemplate rows/layer; fresh non-neural raw-tokenizer/copy child8cases+refusal seals before28Bcomparisons. No model/float-array evaluation/GPU. Historical neural prep paid; dynamic510rows/QKV/image unresolved. Host-local bytes, intent-first,120s,noretries.')
    result = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root/'job.json', result)
    return result


def native_bytes(item):
    path = Path(item['path']); raw = path.read_bytes()
    if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise ValueError('Historical native file changed')
    if raw[:8] != b'\x93NUMPY\x01\x00':
        raise ValueError('Qualified NPY v1 required')
    size = struct.unpack('<H', raw[8:10])[0]; header = ast.literal_eval(raw[10:10+size].decode('latin1'))
    if header != {'descr': '<u2', 'fortran_order': False, 'shape': (1, 512, 2560)} or item['shape'] != [1, 512, 2560]:
        raise ValueError('Native word geometry differs')
    payload = raw[10+size:]
    if len(payload) != 512*2560*2 or hashlib.sha256(payload).hexdigest() != item['payloadSHA256']:
        raise ValueError('Historical native payload differs')
    return payload, {'path': str(path), 'bytesRead': len(raw), 'sha256': item['sha256'], 'payloadSHA256': item['payloadSHA256']}


def serve(spec_path, output):
    from token_history_process import isolate, NoEvaluation
    import resource
    spec = read(spec_path); root = Path(spec['compiled']); output = Path(output).resolve(); output.mkdir()
    started = time.perf_counter(); renderer = TokenInput(root, read(root/'tokenizer-policy.json'))
    renderer.tokenize('')  # Warm non-neural template imports before file isolation.
    runtime = {k: importlib.metadata.version(k) for k in ('tokenizers', 'jinja2')}
    initialized = time.perf_counter()
    opens, denied, imports = isolate(spec, output)
    state = PrefixState(root); loaded = time.perf_counter(); rows = []
    for case in spec['cases']:
        tick = time.perf_counter(); packet, text = renderer.tokenize(case['prompt']); tokenized = time.perf_counter()
        if case['id'] in spec['historicalTokens'] and packet != spec['historicalTokens'][case['id']]:
            raise ValueError('Raw tokenizer differs from historical native input')
        result = state.advance(packet); parts = [state.state(layer) for layer in range(28)] if state.supported else []
        computed = time.perf_counter(); path = output/(case['id']+'.bf16'); path.write_bytes(b''.join(parts))
        rows.append({'id': case['id'], **result, 'prefixIDs': packet['ids'][:4], 'activeRows': sum(packet['mask']),
            'packet': packet, 'renderedTextSHA256': hashlib.sha256(text.encode()).hexdigest(),
            'outputSHA256': sha(path), 'outputBytes': path.stat().st_size,
            'layerWordSHA256': [hashlib.sha256(x).hexdigest() for x in parts],
            'tokenizeSeconds': tokenized-tick, 'stateCopySeconds': computed-tokenized,
            'serializationHashSeconds': time.perf_counter()-computed})
    wrong = {k: v.copy() for k, v in packet.items()}; wrong['ids'][0] = 151645
    refusal = state.advance(wrong)
    try:
        state.state(1)
    except ValueError:
        refusal['stateReadRefused'] = True
    else:
        raise ValueError('Wrong-prefix state unexpectedly available')
    receipt = {'rows': rows, 'wrongPrefix': refusal, 'versions': runtime, 'successfulReadPaths': sorted(opens),
        'fileCanaryDenials': denied, 'blockedImportCanaries': imports,
        'loadedNumericalRuntimeRoots': sorted(NoEvaluation.roots.intersection(sys.modules)),
        'tokenizerInitializationBeforeGuardSeconds': initialized-started, 'constantLoadAfterGuardSeconds': loaded-initialized,
        'wallSecondsBeforeReceipt': time.perf_counter()-started, 'highWaterRSSBytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'isolationScope': 'Trusted-code guard after non-neural tokenizer/template initialization and empty boundary warmup. No model/numerical runtime imports; not an adversarial sandbox.'}
    save(output/'receipt.json', receipt)


def execute(root):
    root = Path(root).resolve(); r = read(root/'request.json'); verify(root, r); started = time.perf_counter()
    meta = read(root/'native-metadata.json'); inputs = read(root/'inputs.json'); out = root/'response-result'; out.mkdir()
    compiled = root/'compiled'; compiled.mkdir(); manifest = read(root/'atom-manifest.json')
    store = Path(inputs['store'])
    if sha(store/'manifest.json') != inputs['manifestSHA256']:
        raise ValueError('Historical atom manifest changed')
    for key in ('tokenizer', 'template'):
        item = manifest[key]; path = store/item['path']
        if sha(path) != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError('Historical tokenizer asset differs')
        shutil.copyfile(path, compiled/item['path'])
    save(compiled/'tokenizer-policy.json', {k: manifest[k] for k in ('tokenizer', 'template', 'tokenPolicy')})
    parts, source_reads = [], []
    for layer in range(28):
        data, receipt = native_bytes(meta['fields']['A']['H'+str(layer)])
        parts.append(data[:2*2560*2]); source_reads.append(receipt)
    bank = b''.join(parts); (compiled/'prefix.bf16').write_bytes(bank)
    program = {'schema': 'tce-template-prefix-state-v1', 'length': 512, 'width': 2560, 'layers': 27,
        'prefixIDs': [151644, 872], 'bankSHA256': hashlib.sha256(bank).hexdigest(), 'bankBytes': len(bank),
        'finitePolicy': 'Copy historical native A prefix words; A/B finite equality measured separately. No universal native finite parity.',
        'dynamicRemainder': [2, 512], 'completeH1Producer': False, 'imageGenerator': False}
    save(compiled/'program.json', program); compile_finished = time.perf_counter()
    spec = {'compiled': str(compiled), 'cases': r['cases'], 'historicalTokens': read(root/'historical-tokens.json'),
        'readAllow': [str(p.resolve()) for p in compiled.iterdir()],
        'forbiddenPaths': [meta['fields']['A']['H1']['path'], meta['fields']['B']['H1']['path'], str(store/'tokenizer.json')]}
    save(root/'serve-spec.json', spec)
    child_started = time.perf_counter()
    subprocess.run([sys.executable, '-u', str(root/'template_cofactor_study.py'), 'serve', str(root/'serve-spec.json'), str(root/'served')], check=True, timeout=80)
    child_finished = time.perf_counter(); child = read(root/'served/receipt.json')
    save(root/'serving-seal.json', {'requestSHA256': r['requestSHA256'], 'receiptSHA256': sha(root/'served/receipt.json'),
        'programSHA256': sha(compiled/'program.json'), 'outputs': {x['id']: x['outputSHA256'] for x in child['rows']}})
    seal_sha = sha(root/'serving-seal.json'); comparisons, diagnostic_reads = [], []
    for layer in range(28):
        a, ar = native_bytes(meta['fields']['A']['H'+str(layer)])
        b, br = native_bytes(meta['fields']['B']['H'+str(layer)])
        diagnostic_reads.extend([ar, br]); expected = bank[layer*10240:(layer+1)*10240]
        differing = [i for i in range(512) if a[i*5120:(i+1)*5120] != b[i*5120:(i+1)*5120]]
        comparisons.append({'layer': layer, 'BPrefixMatchesCompiledWords': b[:10240] == expected,
                            'nativeABDifferentRows': len(differing), 'firstDifferentRow': min(differing) if differing else None,
                            'nativeBPrefixSHA256': hashlib.sha256(b[:10240]).hexdigest()})
    if sha(root/'serving-seal.json') != seal_sha:
        raise ValueError('Serving seal changed')
    save(out/'report.json', {'schema': r['schema'], 'requestSHA256': r['requestSHA256'], 'program': program,
        'diagnosticsReleasedAfterSealSHA256': seal_sha,
        'sourceReads': source_reads, 'diagnosticReads': diagnostic_reads, 'comparisons': comparisons,
        'historicalPreparation': inputs, 'compiledAssetBytes': sum(p.stat().st_size for p in compiled.iterdir()),
        'compileAuthenticationCopySeconds': compile_finished-started, 'freshChildWallSeconds': child_finished-child_started,
        'workerWallSecondsBeforeReport': time.perf_counter()-started,
        'newNeuralEvaluations': 0, 'newScientificFloatingArithmetic': 0, 'newImages': 0,
        'mechanics_status': 'completed', 'instrument_status': 'source-byte-qualified-historical-comparison',
        'trend_status': 'partial-contextual-constant-constructor; inspect zero effects and finite comparisons',
        'terminal_status': 'arbitrary-prompt-zero-neural-image-generation-unachieved',
        'scope': 'Only two numeric template rows. Full raw prompt is tokenized and retained as own input;510 dynamic rows never synthesized. No K/V cache or native continuation. Historical B is public, not fresh heldout.'})
    for source in [root/'served/receipt.json', root/'serving-seal.json', compiled/'program.json']:
        shutil.copyfile(source, out/source.name)
    save(out/'inputs.json', inputs)
    save(out/'request.json', r)
    return {'rows': len(child['rows']), 'bankBytes': len(bank), 'seconds': time.perf_counter()-started}


def main():
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['stage', 'submit', 'execute', 'serve', 'collect']); p.add_argument('root', type=Path); p.add_argument('output', nargs='?', type=Path); a = p.parse_args()
    if a.action == 'serve':
        result = serve(a.root, a.output)
    elif a.action == 'collect':
        from response_study import collect
        result = collect(a.root)
    else:
        result = globals()[a.action](a.root)
    if result is not None:
        print(json.dumps(result))


if __name__ == '__main__':
    main()
