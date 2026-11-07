"""Weight-only norm compiler and a fresh own-state actor without supplied norms."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha
from paired_fields import load, retain, payload
from projected_complement import load_selected
from propagated_context import P1, RETURN_PARAMETERS, NEXT_PARAMETERS, normalized, next_query
from image_dual import image_numerators, read_head, consume_head
from diagonal_key_norm import compile_diagonal, evaluate

WEIGHT = P1 + 'attn.to_k.weight'


def execute(phase, root):
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from mdb.titan import Frame, ActSpec, ActRegistry, ProvidedFingerprint
    from complete_masked_study import profile
    from flux_clock_study import verify, source_checks, assets_unchanged
    from flux_clock import header, snapshot
    from sparse_text_kv import quarantine
    from selected_source_v2 import load_pages
    root = Path(root).resolve();r, s = read(root / 'request.json'), read(root / 'inputs.json');verify(root, r)
    out = root / 'response-result';out.mkdir(exist_ok=phase != 'compiler')
    compiler = None
    if phase == 'actor':
        compiler = read(out / 'compiler.json');seal = read(out / 'compiler-seal.json')
        if seal['compilerSHA256'] != sha(out / 'compiler.json') or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Weight-only compiler must seal and exit first')
    allowed = [str(Path(s['modelRoot']) / n) for n in s['imageAssets']]
    denied = list(s['deniedNumericalInputs'])
    if phase == 'actor':
        cut = {**s['sourceCuts']['Q'], 'pages': {n: v for n, v in s['sourceCuts']['Q']['pages'].items() if n in ['image', 'context']}}
        allowed += [cut['cutFile']['path']] + [p for page in cut['pages'].values() for p in page['files']]
        allowed += [x['path'] for x in s['selected'].values()] + [s['textKeySlice']['path'], s['banks']['Q']['V']['path']]
        allowed += [s['reuse'][n]['path'] for n in ['duals', 'groupIDs']] + [compiler['compile']['fields']['diagonal']['path']]
    else:
        denied += list(s['compilerForbiddenSources'])
    if set(denied) & set(allowed):
        raise ValueError('Quarantine has conflicting declared access')
    accesses = quarantine(root, allowed, denied)
    started, cpu = time.perf_counter(), time.process_time();numerical = profile(torch, 4)
    if numerical != s['numericalProfile']:
        raise ValueError('Original finite profile differs')
    model, assets = source_checks(s, diffusers);checkpoint = model / 'transformer/diffusion_pytorch_model.safetensors'
    entries, head = header(checkpoint);names = (WEIGHT,) if phase == 'compiler' else RETURN_PARAMETERS + NEXT_PARAMETERS
    weights, loaded = load_selected(checkpoint, names, entries)
    recipes = {x['source']['tensor']: x['source'] for x in s['originalParameterArtifacts']}
    for x in loaded:
        old = recipes[x['name']]
        if x['payloadSHA256'] != old['payload_sha256'] or x['shape'] != old['shape'] or x['dataOffsets'] != old['data_offsets']:
            raise ValueError('Original parameter payload/header differs')
    before = {n: (v.data_ptr(), v._version, payload(v)) for n, v in weights.items()}
    rng, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all();torch.cuda.reset_peak_memory_stats()
    operations, elementwise, timings, records = [], [], [], {};stage = phase;registry = ActRegistry()
    def mm(a, b, label):
        if a.dtype != torch.float64 or b.dtype != torch.float64 or a.device.type != 'cpu' or b.device.type != 'cpu':
            raise ValueError('Declared CPU FP64 contraction differs')
        result = a @ b
        operations.append({'stage': stage, 'operation': label, 'inputShape': list(a.shape), 'rightShape': list(b.shape),
                           'dtype': str(a.dtype), 'device': 'cpu', 'nominalMACs': a.numel() * b.shape[-1]})
        return result
    old_linear = torch.nn.functional.linear
    def linear(x, w, bias=None):
        found = [n for n, v in weights.items() if v is w]
        if len(found) != 1 or bias is not None:
            raise ValueError('Undeclared original affine')
        operations.append({'stage': stage, 'operation': 'original-affine', 'parameter': found[0],
            'inputShape': list(x.shape), 'weightShape': list(w.shape), 'dtype': str(x.dtype), 'device': x.device.type,
            'nominalMACs': x.numel() // x.shape[-1] * w.numel()})
        return old_linear(x, w, bias)
    def run(name, inputs, operation, abis, metadata):
        nonlocal stage
        stage = name;initial = {n: payload(v) for n, v in inputs.items()}
        frame = Frame.from_values(inputs, kinds={n: 'tensor' for n in inputs},
            abis={n: {'shape': list(v.shape), 'stride': list(v.stride()), 'dtype': str(v.dtype)} for n, v in inputs.items()},
            slot_fingerprints={n: ProvidedFingerprint.content_sha256(payload(v)) for n, v in inputs.items()},
            program_fingerprint=r['requestSHA256'], metadata=metadata)
        spec = ActSpec('tce.diagonal_norm_' + name, 'v1', reads=tuple(inputs), writes=tuple(abis),
            write_kinds={n: 'tensor' for n in abis}, write_abis=abis,
            numerical_contract='weight-only-diagonal/own-native-BF16-ModLN/FP64-reader/native-finite-own-consumer')
        captures = {}
        def act(v):
            torch.cuda.synchronize();tick = time.perf_counter();got = operation(v);torch.cuda.synchronize()
            timings.append({'stage': name, 'arithmeticSecondsBeforeCapture': time.perf_counter() - tick})
            if any({'shape': list(got[n].shape), 'dtype': str(got[n].dtype)} != abi for n, abi in abis.items()):
                raise ValueError('Explicit output ABI differs')
            captures.update({n: retain(root / 'arrays' / phase / name, n, x) for n, x in got.items()})
            return {n: got[n] for n in abis}, {'capturedPayloads': {n: captures[n]['payloadSHA256'] for n in abis}}
        registry.register(spec, name, act);after, evidence = registry.run(spec.identity, name, frame)
        if initial != {n: payload(v) for n, v in inputs.items()}:
            raise ValueError('Typed input mutated')
        records[name] = {'fields': captures, 'sourceFrame': frame.to_dict(), 'outputFrame': after.to_dict(), 'evidence': evidence.to_dict()}
        return after
    abi = lambda shape, dtype='torch.float64': {'shape': shape, 'dtype': dtype}
    try:
        torch.nn.functional.linear = linear
        with torch.inference_mode():
            if phase == 'compiler':
                def compile(v):
                    w = v[WEIGHT].cpu().double()
                    elementwise.append({'stage': stage, 'weightSquares': w.numel(), 'sumAdditions': 24 * 3072 * 127,
                                        'meanScalings': 24 * 3072, 'input': 'Original W only; no source states/norms'})
                    return {'diagonal': compile_diagonal(w)}
                run('compile', {WEIGHT: weights[WEIGHT]}, compile, {'diagonal': abi([24, 3072])},
                    {'sourceOnly': True, 'normalizer': 'raw-key RMS before gamma/RoPE'})
            else:
                state = load_pages(cut);context = state['context']
                normal = run('modln', {'image': state['image'], 'imgMod': context['img_mod']},
                    lambda v: {'x': normalized(v['image'], v['imgMod'])}, {'x': abi([1, 1024, 3072], 'torch.bfloat16')},
                    {'sourceCut': cut['cutID']})
                def norm(v):
                    elementwise.append({'stage': stage, 'inputSquares': 1024 * 3072, 'epsilonAdditions': 1024 * 24,
                                        'squareRoots': 1024 * 24, 'epsilon': 1e-6})
                    return evaluate(v['x'][0].cpu().double(), v['diagonal'], mm)
                norms = run('norm', {'x': normal.values['x'], 'diagonal': load(compiler['compile']['fields']['diagonal'], 'cpu')},
                    norm, {n: abi([1024, 24]) for n in ['energy', 'norm']},
                    {'modLNOutputFrame': normal.frame_id, 'compilerSealSHA256': sha(out / 'compiler-seal.json')})
                selected = {n: load(x, 'cuda') for n, x in s['selected'].items()}
                values = dict(x=normal.values['x'], norm=norms.values['norm'],
                    **{n: load(s['reuse'][n], 'cpu') for n in ['duals', 'groupIDs']},
                    Q=selected['Q'], selectedK=selected['K'], selectedV=selected['V'],
                    textK=load(s['textKeySlice'], 'cpu'), bankV=load(s['banks']['Q']['V'], 'cpu'))
                def reader(v):
                    q = v['Q'][0].cpu().double().permute(1, 0, 2)
                    text, bank = v['textK'].clone(), v['bankV'].clone()
                    text[:, r['support']] = v['selectedK'].cpu();bank[:, r['support']] = v['selectedV'].cpu()
                    numerator = image_numerators(v['x'][0].cpu().double(), v['duals'], v['groupIDs'], mm)
                    image = numerator / (128 ** .5 * v['norm'].T[:, None])
                    text_logits = mm(q, text[0].double().permute(1, 2, 0), 'text-QK') / 128 ** .5
                    logits = torch.cat((text_logits, image), -1)
                    account = read_head(logits, bank[0].double().permute(1, 0, 2), mm)
                    h = account['H_FP64'].permute(1, 0, 2).reshape(1, 4, 3072).float().bfloat16().to('cuda')
                    return dict(imageNumerator=numerator, logits=logits, textLogits=text_logits, textK_hybrid=text,
                                V_hybrid=bank, H=h, **account)
                abis = {n: abi([24, 4, 1536]) for n in ['logits', 'probability']}
                abis.update(H=abi([1, 4, 3072], 'torch.bfloat16'), H_FP64=abi([24, 4, 128]), imageNumerator=abi([24, 4, 1024]),
                            textLogits=abi([24, 4, 512]), peak=abi([24, 4, 1]), partition=abi([24, 4, 1]))
                result = run('read', values, reader, abis, {'normOutputFrame': norms.frame_id, 'modLNOutputFrame': normal.frame_id,
                             'admissionCast': 'CPU FP64→FP32→BF16 RNE→CUDA'})
                values = dict(H=result.values['H'], R=selected['R'], txtMod=selected['txtMod'], **{n: weights[n] for n in RETURN_PARAMETERS})
                source = run('source', values, lambda v: consume_head(v, v['H'], weights, native),
                    {n: abi([1, 4, 3072], 'torch.bfloat16') for n in ['A', 'U', 'M', 'T']}, {'headOutputFrame': result.frame_id})
                values = dict(rows=source.values['T'], txtMod=selected['txtMod'], ropeCos=selected['ropeCos'], ropeSin=selected['ropeSin'],
                              **{n: weights[n] for n in NEXT_PARAMETERS})
                run('next_query', values, lambda v: next_query(v['rows'], v['txtMod'], (v['ropeCos'], v['ropeSin']), weights, native),
                    {'nextQ': abi([1, 4, 24, 128], 'torch.bfloat16')}, {'sourceOutputFrame': source.frame_id, 'nextJoint': 2})
    finally:
        torch.nn.functional.linear = old_linear
    if before != {n: (v.data_ptr(), v._version, payload(v)) for n, v in weights.items()} or snapshot(torch) != numerical or not torch.equal(rng, torch.get_rng_state()) or not all(torch.equal(a, b) for a, b in zip(rng_cuda, torch.cuda.get_rng_state_all())):
        raise ValueError('Parameter/input/profile/RNG closure differs')
    assets_unchanged(assets);total = sum(x['nominalMACs'] for x in operations)
    if total != (0 if phase == 'compiler' else 1421869056):
        raise ValueError('Complete matrix-work ledger differs')
    save(out / (phase + '.json'), {**records, 'compilerSealSHA256': sha(out / 'compiler-seal.json') if phase == 'actor' else None})
    save(out / (phase + '-seal.json'), {'requestSHA256': r['requestSHA256'], phase + 'SHA256': sha(out / (phase + '.json'))})
    save(out / (phase + '-execution.json'), {'loadedParameters': loaded, 'operations': operations, 'elementwise': elementwise,
        'nominalMatrixMACs': total, 'parameterInputRNGClosure': True, 'numericalProfile': numerical, 'assets': assets, 'header': head,
        'arithmeticTimings': timings, 'wallSecondsAfterImports': time.perf_counter() - started, 'CPUSecondsAfterImports': time.process_time() - cpu,
        'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'peakCUDAReservedBytes': torch.cuda.max_memory_reserved(),
        'fileQuarantine': {'allowedFiles': sorted(allowed), 'deniedCanaries': denied, 'successfulFileOpenCounts': accesses}})
    print('DIAGONAL_NORM_' + phase.upper() + '_SEALED', flush=True)


if __name__ == '__main__':
    execute(sys.argv[1], sys.argv[2])
