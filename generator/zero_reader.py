"""Raw lexical operands plus a positive-zero first attention bank; no reader."""
from pathlib import Path
import hashlib
import resource
import sys
import time
from encoder_atoms import np, Prefix, guard, read, save, sha
from paired_fields import retain


def zero_bank(root, name):
    item = retain(root, name, np.zeros((1, 512, 4096), dtype=np.uint16))
    item['logicalDtype'] = 'torch.bfloat16'
    return item


def execute(root):
    from zero_reader_study import verify
    from flux_clock import deny_targets
    root = Path(root).resolve(); request = read(root / 'request.json'); verify(root, request); source = read(root / 'inputs.json')
    guard(); deny_targets([root / 'targets.json', *source['forbiddenTargets']])
    start = time.perf_counter(); cpu = time.process_time(); prefix = Prefix(source['lexicalStore'], request['manifestSHA256'])
    if len(prefix.store.ids) != 151936: raise ValueError('Complete vocabulary required')
    lexical = root / 'lexical'; lexical.mkdir(); rows = []; fields = {}
    for prompt in request['prompts']:
        tick = time.perf_counter(); values, text = prefix.forward(prompt['prompt']); seconds = time.perf_counter() - tick
        path = lexical / (prompt['id'] + '.npz'); np.savez(path, **values)
        rows.append({'id': prompt['id'], 'prompt': prompt['prompt'], 'path': str(path), 'sha256': sha(path),
            'activeTokens': int(values['mask'].sum()), 'firstArithmeticSeconds': seconds,
            'renderedTextSHA256': hashlib.sha256(text.encode()).hexdigest(),
            'idsPayloadSHA256': hashlib.sha256(values['ids'].tobytes()).hexdigest(), 'maskPayloadSHA256': hashlib.sha256(values['mask'].tobytes()).hexdigest()})
        fields[prompt['id']] = zero_bank(root / 'arrays/zero-head', prompt['id'] + '-head')
    save(lexical / 'prediction-seal.json', {'manifestSHA256': request['manifestSHA256'], 'rows': rows, 'coverageCount': len(prefix.store.ids),
        'uniqueRowsRead': prefix.store.rowsRead, 'compiledRowPayloadBytesRead': prefix.store.rowBytesRead,
        'frameworkCalls': 0, 'originalDenseProjections': 0, 'scope': 'Same bundled lexical E/QKV lookup as rank-one parent; QKV are unused by zero first reader.'})
    manifest = prefix.store.manifest
    atom_files = [{'path': str(Path(source['lexicalStore']) / 'manifest.json'), 'sha256': request['manifestSHA256']}]
    atom_files += [{'path': str(Path(source['lexicalStore']) / manifest[k]['path']), 'sha256': manifest[k]['sha256']}
        for k in ['tokenizer', 'template', 'coverage', 'rowHashes', 'rope']]
    atom_files += [{'path': str(Path(source['lexicalStore']) / manifest['shards'][j]['path']), 'sha256': manifest['shards'][j]['sha256']}
        for j in sorted(prefix.store.shards)]
    if {'rank_one_tail', 'attention_bank', 'selector_objective', 'torch', 'transformers', 'diffusers'}.intersection(sys.modules):
        raise ValueError('Deleted reader or neural runtime was imported')
    save(root / 'reader-seal.json', {'schema': 'tce-zero-reader-seal-v1', 'requestSHA256': request['requestSHA256'],
        'lexicalSealSHA256': sha(lexical / 'prediction-seal.json'), 'fields': fields, 'lexicalRows': rows, 'sourceAtomFiles': atom_files,
        'workerSecondsAfterImports': time.perf_counter() - start, 'cpuSecondsAfterImports': time.process_time() - cpu,
        'highWaterRSSKiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'numpyVersion': np.__version__,
        'readerCalls': 0, 'readerScoreProducts': 0, 'readerMomentCells': 0, 'readerEnvelopes': 0, 'readerFallbacks': 0,
        'frameworkCalls': 0, 'originalDenseProjections': 0,
        'scope': 'First contextual read deleted. Positive-zero bank before original output projection; own lexical residual and subsequent learned model remain.'})
    print('ZERO_HEADS_AND_LEXICAL_SEALED', sha(root / 'reader-seal.json'), flush=True)
