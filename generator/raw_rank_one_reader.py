"""Fresh raw strings -> complete atoms -> frozen all-head causal moment reader."""
from pathlib import Path
import hashlib
import resource
import time
from encoder_atoms import np, Prefix, guard, read, save, sha, floats, words
from paired_fields import retain


def bank_field(root, name, raw, rounded):
    if raw.shape != (512, 4096) or rounded.shape != raw.shape or rounded.dtype != np.uint16:
        raise ValueError('Complete head bank geometry differs')
    if not np.isfinite(raw).all() or not np.isfinite(floats(rounded)).all() or not np.array_equal(words(raw), rounded):
        raise ValueError('Head bank finite/rounding contract differs')
    item = retain(root, name, rounded[None, :, :])
    # BF16 bits cross the no-framework boundary as uint16, just as lexical atoms do.
    item['logicalDtype'] = 'torch.bfloat16'
    return item


def execute(root):
    from raw_rank_one_study import verify
    from flux_clock import deny_targets
    root = Path(root).resolve(); request = read(root / 'request.json'); verify(root, request); source = read(root / 'inputs.json')
    guard(); deny_targets([root / 'targets.json', *source['forbiddenTargets']])
    from rank_one_tail import serve, POLICY
    if POLICY != source['readerPolicy']: raise ValueError('Frozen reader policy changed')
    start = time.perf_counter(); cpu = time.process_time(); prefix = Prefix(source['lexicalStore'], request['manifestSHA256'])
    if len(prefix.store.ids) != 151936: raise ValueError('Complete vocabulary required')
    lexical = root / 'lexical'; lexical.mkdir(); rows = []
    for prompt in request['prompts']:
        tick = time.perf_counter(); values, text = prefix.forward(prompt['prompt']); seconds = time.perf_counter() - tick
        path = lexical / (prompt['id'] + '.npz'); np.savez(path, **values)
        rows.append({'id': prompt['id'], 'prompt': prompt['prompt'], 'path': str(path), 'sha256': sha(path),
            'activeTokens': int(values['mask'].sum()), 'firstArithmeticSeconds': seconds,
            'renderedTextSHA256': hashlib.sha256(text.encode()).hexdigest(),
            'idsPayloadSHA256': hashlib.sha256(values['ids'].tobytes()).hexdigest(), 'maskPayloadSHA256': hashlib.sha256(values['mask'].tobytes()).hexdigest()})
    save(lexical / 'prediction-seal.json', {'manifestSHA256': request['manifestSHA256'], 'rows': rows, 'coverageCount': len(prefix.store.ids),
        'uniqueRowsRead': prefix.store.rowsRead, 'compiledRowPayloadBytesRead': prefix.store.rowBytesRead,
        'frameworkCalls': 0, 'originalDenseProjections': 0, 'scope': 'Fresh raw strings and complete learned atom memoization, not supplied prompt QKV.'})
    save(root / 'serve-source.json', {'asset': source['readerAsset'], 'basis': source['readerBasis'], 'geometry': source['readerGeometry'], 'ports': rows})
    tick = time.perf_counter(); serve(root / 'serve-source.json', root / 'reader'); reader_seconds = time.perf_counter() - tick
    seal = read(root / 'reader/prediction-seal.json'); fields = {}
    for item in seal['rows']:
        if sha(item['path']) != item['sha256']: raise ValueError('Reader output changed before handoff')
        with np.load(item['path'], allow_pickle=False) as packet:
            fields[item['id']] = bank_field(root / 'arrays/reader', item['id'] + '-head', packet['raw'], packet['rounded'])
    if set(fields) != {x['id'] for x in request['prompts']}: raise ValueError('Missing complete candidate bank')
    manifest = prefix.store.manifest
    atom_files = [{'path': str(Path(source['lexicalStore']) / 'manifest.json'), 'sha256': request['manifestSHA256']}]
    atom_files += [{'path': str(Path(source['lexicalStore']) / manifest[k]['path']), 'sha256': manifest[k]['sha256']}
        for k in ['tokenizer', 'template', 'coverage', 'rowHashes', 'rope']]
    atom_files += [{'path': str(Path(source['lexicalStore']) / manifest['shards'][j]['path']), 'sha256': manifest['shards'][j]['sha256']}
        for j in sorted(prefix.store.shards)]
    save(root / 'reader-seal.json', {'schema': 'tce-raw-rank-one-reader-seal-v1', 'requestSHA256': request['requestSHA256'],
        'lexicalSealSHA256': sha(lexical / 'prediction-seal.json'), 'readerPredictionSealSHA256': sha(root / 'reader/prediction-seal.json'),
        'fields': fields, 'lexicalRows': rows, 'sourceAtomFiles': atom_files, 'reader': seal, 'readerWallSeconds': reader_seconds,
        'workerSecondsAfterImports': time.perf_counter() - start, 'cpuSecondsAfterImports': time.process_time() - cpu,
        'highWaterRSSKiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'numpyVersion': np.__version__,
        'frameworkCalls': 0, 'originalDenseProjections': 0, 'sourceBasisRecompiled': False, 'exactHeadReferences': 0,
        'scope': 'Raw strings through complete lexical table and frozen32-head reader; full currentKV/moment/envelope/index and same-law fallbacks paid.'})
    print('RAW_RANK_ONE_HEADS_SEALED', sha(root / 'reader-seal.json'), flush=True)
