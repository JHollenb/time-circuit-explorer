"""Sealed compiler, target-isolated actor and post-seal numerical references."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha
from paired_fields import load, retain, payload
from projected_complement import fields, load_selected
from propagated_context import P1, RETURN_PARAMETERS, NEXT_PARAMETERS, normalized, next_query
from image_dual import image_groups, compile_duals, raw_projection, image_numerators, read_head, consume_head

KEY_PARAMETERS = (P1 + 'attn.to_k.weight', P1 + 'attn.norm_k.weight')


def execute(phase, root):
    import torch
    import diffusers
    from mdb.titan import Frame, ActSpec, ActRegistry, ProvidedFingerprint
    from complete_masked_study import profile
    from flux_clock_study import verify, source_checks, assets_unchanged
    from flux_clock import header, snapshot
    from sparse_text_kv import quarantine
    from selected_source_v2 import load_pages
    import diffusers.models.transformers.transformer_flux2 as native
    root = Path(root).resolve();r, s = read(root / 'request.json'), read(root / 'inputs.json');verify(root, r)
    out = root / 'response-result';out.mkdir(exist_ok=phase != 'compiler')
    compiler, actor = None, None
    if phase != 'compiler':
        compiler = read(out / 'compiler.json')
        if read(out / 'compiler-seal.json')['compilerSHA256'] != sha(out / 'compiler.json'):
            raise ValueError('Compiler release differs')
    if phase == 'reference':
        actor = read(out / 'actor.json')
        if read(out / 'actor-seal.json')['actorSHA256'] != sha(out / 'actor.json') or actor['compilerSealSHA256'] != sha(out / 'compiler-seal.json'):
            raise ValueError('Actor must seal and exit before reference')
    allowed = [str(Path(s['modelRoot']) / n) for n in s['imageAssets']]
    denied = list(s['deniedTargets'])
    if phase in ['compiler', 'actor']:
        role = 'P' if phase == 'compiler' else 'Q'
        cut = {**s['sourceCuts'][role], 'pages': {n: x for n, x in s['sourceCuts'][role]['pages'].items() if n in ['image', 'context']}}
        allowed += [cut['cutFile']['path']] + [p for page in cut['pages'].values() for p in page['files']]
        allowed += [s['selected']['Q']['path']]
        denied += [s['banks'][role]['K']['path'] for role in ['P', 'Q']]
        if phase == 'actor':
            allowed += [x['path'] for x in s['selected'].values()] + [s['textKeySlice']['path'], s['banks']['Q']['V']['path']]
            allowed += [compiler['compile']['fields'][n]['path'] for n in ['duals', 'pDenominator', 'groupIDs']]
        else:
            denied += [s['banks']['Q']['V']['path']]
    else:
        allowed += [x['path'] for x in fields(actor)] + [s['selected']['Q']['path'], s['banks']['Q']['K']['path']]
        allowed += [compiler['compile']['fields']['pDenominator']['path'], compiler['rope']['cosine']['path'], compiler['rope']['sine']['path']]
        denied = []
    accesses = quarantine(root, allowed, denied)
    started, cpu = time.perf_counter(), time.process_time();numerical = profile(torch, 4)
    if numerical != s['numericalProfile']:
        raise ValueError('Original finite consumer profile differs')
    model, assets = source_checks(s, diffusers);checkpoint = model / 'transformer/diffusion_pytorch_model.safetensors'
    entries, head = header(checkpoint)
    names = RETURN_PARAMETERS + NEXT_PARAMETERS if phase == 'actor' else KEY_PARAMETERS
    weights, loaded = load_selected(checkpoint, names, entries)
    recipes = {x['source']['tensor']: x['source'] for x in s['originalParameterArtifacts']}
    for x in loaded:
        recipe = recipes[x['name']]
        if x['payloadSHA256'] != recipe['payload_sha256'] or x['shape'] != recipe['shape'] or x['dataOffsets'] != recipe['data_offsets']:
            raise ValueError('Original parameter payload/header differs')
    before = {n: (x.data_ptr(), x._version, payload(x)) for n, x in weights.items()}
    rng, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all();torch.cuda.reset_peak_memory_stats()
    operations, timings, records = [], [], {};registry = ActRegistry();stage = phase
    def mm(a, b, label):
        if a.dtype != torch.float64 or b.dtype != torch.float64 or a.device.type != 'cpu' or b.device.type != 'cpu':
            raise ValueError('Declared CPU FP64 contraction differs')
        result = a @ b
        operations.append({'stage': stage, 'operation': label, 'inputShape': list(a.shape), 'rightShape': list(b.shape),
                           'dtype': str(a.dtype), 'device': 'cpu', 'nominalMACs': a.numel() * b.shape[-1]})
        return result
    original_linear = torch.nn.functional.linear
    def linear(x, w, bias=None):
        matches = [n for n, v in weights.items() if v is w]
        if len(matches) != 1 or bias is not None:
            raise ValueError('Undeclared receiving affine')
        operations.append({'stage': stage, 'operation': 'original-affine', 'parameter': matches[0],
                           'inputShape': list(x.shape), 'weightShape': list(w.shape), 'dtype': str(x.dtype),
                           'device': x.device.type, 'nominalMACs': x.numel() // x.shape[-1] * w.numel()})
        return original_linear(x, w, bias)
    def run(name, inputs, operation, abis, metadata):
        nonlocal stage
        stage = name;before_inputs = {n: payload(x) for n, x in inputs.items()}
        frame = Frame.from_values(inputs, kinds={n: 'tensor' for n in inputs},
            abis={n: {'shape': list(x.shape), 'stride': list(x.stride()), 'dtype': str(x.dtype)} for n, x in inputs.items()},
            slot_fingerprints={n: ProvidedFingerprint.content_sha256(payload(x)) for n, x in inputs.items()},
            program_fingerprint=r['requestSHA256'], metadata=metadata)
        spec = ActSpec('tce.image_dual_' + name, 'v1', reads=tuple(inputs), writes=tuple(abis),
                       write_kinds={n: 'tensor' for n in abis}, write_abis=abis,
                       numerical_contract='native-BF16-ModLN/CPU-FP64-dual-stale-P-RMS/native-finite-consumer')
        captures = {}
        def act(v):
            torch.cuda.synchronize();tick = time.perf_counter();got = operation(v);torch.cuda.synchronize()
            timings.append({'stage': name, 'arithmeticSecondsBeforeCapture': time.perf_counter() - tick})
            if any({'shape': list(got[n].shape), 'dtype': str(got[n].dtype)} != abi for n, abi in abis.items()):
                raise ValueError('Actual explicit output ABI differs')
            captures.update({n: retain(root / 'arrays' / phase / name, n, x) for n, x in got.items()})
            return {n: got[n] for n in abis}, {'capturedPayloads': {n: captures[n]['payloadSHA256'] for n in abis}}
        registry.register(spec, name, act);after, evidence = registry.run(spec.identity, name, frame)
        if before_inputs != {n: payload(x) for n, x in inputs.items()}:
            raise ValueError('Actual typed input mutated')
        records[name] = {'fields': captures, 'sourceFrame': frame.to_dict(), 'outputFrame': after.to_dict(), 'evidence': evidence.to_dict()}
        return after
    abi = lambda shape, dtype='torch.float64': {'shape': shape, 'dtype': dtype}
    try:
        torch.nn.functional.linear = linear
        with torch.inference_mode():
            if phase in ['compiler', 'actor']:
                state = load_pages(cut);context = state['context']
                normal = run('modln', {'image': state['image'], 'imgMod': context['img_mod']},
                    lambda v: {'x': normalized(v['image'], v['imgMod'])},
                    {'x': abi([1, 1024, 3072], 'torch.bfloat16')}, {'sourceCut': cut['cutID']})
            if phase == 'compiler':
                cos, sin = [x[512:].cpu().double() for x in context['rope']]
                ids, inventory = image_groups(cos, sin)
                text_factors = torch.cat([x[:512, 96:128].cpu().double() for x in context['rope']], -1)
                if len(torch.unique(text_factors, dim=0)) != 512:
                    raise ValueError('Declared distinct text L positions differ')
                rope = {n: retain(root / 'arrays/compiler/rope', n, x) for n, x in [('cosine', cos), ('sine', sin)]}
                values = dict(x=normal.values['x'], Q=load(s['selected']['Q'], 'cuda'), cosine=cos, sine=sin,
                              **{n: weights[n] for n in KEY_PARAMETERS})
                def compile(v):
                    w, gamma = [v[n].cpu().double() for n in KEY_PARAMETERS]
                    raw, denominator = raw_projection(v['x'][0].cpu().double(), w, 24, mm, 'P-denominator-projection')
                    table = compile_duals(v['Q'][0].cpu().double(), w, gamma, v['cosine'], v['sine'], mm)
                    return dict(rawP=raw, pDenominator=denominator, duals=table, groupIDs=ids)
                run('compile', values, compile, {'duals': abi([4, 24, 65, 3072]), 'pDenominator': abi([1024, 24]),
                    'rawP': abi([1024, 24, 128]), 'groupIDs': abi([1024, 2], 'torch.int64')},
                    {'modLNOutputFrame': normal.frame_id, 'parentBankSealSHA256': s['parentBankSealSHA256']})
                extra = {'groupInventory': inventory, 'textUniqueLPositions': 512, 'rope': rope}
            elif phase == 'actor':
                selected = {n: load(x, 'cuda') for n, x in s['selected'].items()}
                compiled = {n: load(compiler['compile']['fields'][n], 'cpu') for n in ['duals', 'pDenominator', 'groupIDs']}
                values = dict(x=normal.values['x'], **compiled, Q=selected['Q'], selectedK=selected['K'], selectedV=selected['V'],
                              textK=load(s['textKeySlice'], 'cpu'), bankV=load(s['banks']['Q']['V'], 'cpu'))
                def reader(v):
                    q = v['Q'][0].cpu().double().permute(1, 0, 2)
                    text, bank = v['textK'].clone(), v['bankV'].clone()
                    text[:, r['support']] = v['selectedK'].cpu();bank[:, r['support']] = v['selectedV'].cpu()
                    numerator = image_numerators(v['x'][0].cpu().double(), v['duals'], v['groupIDs'], mm)
                    image = numerator / (128 ** .5 * v['pDenominator'].T[:, None])
                    text_logits = mm(q, text[0].double().permute(1, 2, 0), 'text-QK') / 128 ** .5
                    logits = torch.cat((text_logits, image), -1)
                    account = read_head(logits, bank[0].double().permute(1, 0, 2), mm)
                    head = account['H_FP64'].permute(1, 0, 2).reshape(1, 4, 3072).float().bfloat16().to('cuda')
                    return dict(imageNumerator=numerator, logits=logits, textLogits=text_logits, textK_hybrid=text,
                                V_hybrid=bank, H=head, **account)
                score_abis = {n: abi([24, 4, 1536]) for n in ['logits', 'probability']}
                score_abis.update(H=abi([1, 4, 3072], 'torch.bfloat16'), H_FP64=abi([24, 4, 128]),
                    imageNumerator=abi([24, 4, 1024]), textLogits=abi([24, 4, 512]),
                    peak=abi([24, 4, 1]), partition=abi([24, 4, 1]))
                # Raw hybrid byte witnesses are captured but do not become continuation slots.
                score = run('read', values, reader, score_abis, {'compilerSealSHA256': sha(out / 'compiler-seal.json'),
                    'modLNOutputFrame': normal.frame_id, 'admissionCast': 'CPU FP64→FP32→BF16 RNE→CUDA'})
                values = dict(H=score.values['H'], R=selected['R'], txtMod=selected['txtMod'], **{n: weights[n] for n in RETURN_PARAMETERS})
                source = run('source', values, lambda v: consume_head(v, v['H'], weights, native),
                    {n: abi([1, 4, 3072], 'torch.bfloat16') for n in ['A', 'U', 'M', 'T']},
                    {'headOutputFrame': score.frame_id})
                values = dict(rows=source.values['T'], txtMod=selected['txtMod'], ropeCos=selected['ropeCos'], ropeSin=selected['ropeSin'],
                              **{n: weights[n] for n in NEXT_PARAMETERS})
                run('next_query', values, lambda v: next_query(v['rows'], v['txtMod'], (v['ropeCos'], v['ropeSin']), weights, native),
                    {'nextQ': abi([1, 4, 24, 128], 'torch.bfloat16')}, {'sourceOutputFrame': source.frame_id, 'nextJoint': 2})
                extra = {'compilerSealSHA256': sha(out / 'compiler-seal.json')}
            else:
                # The only Q projection consumes the actor's retained actual BF16 input.
                values = dict(x=load(actor['modln']['fields']['x'], 'cpu'), Q=load(s['selected']['Q'], 'cpu'),
                              nativeKeys=load(s['banks']['Q']['K'], 'cpu'),
                              textLogits=load(actor['read']['fields']['textLogits'], 'cpu'),
                              bankV=load(actor['read']['fields']['V_hybrid'], 'cpu'),
                              cosine=load(compiler['rope']['cosine'], 'cpu'), sine=load(compiler['rope']['sine'], 'cpu'),
                              **{n: weights[n] for n in KEY_PARAMETERS})
                def reference(v):
                    w, gamma = [v[n].cpu().double() for n in KEY_PARAMETERS]
                    raw, denominator = raw_projection(v['x'][0].double(), w, 24, mm, 'Q-reference-projection')
                    z = raw * gamma
                    rotated = torch.stack((-z[..., 1::2], z[..., 0::2]), -1).flatten(-2)
                    unnormalized = z * v['cosine'][:, None] + rotated * v['sine'][:, None]
                    q = v['Q'][0].double().permute(1, 0, 2)
                    numerator = mm(q, unnormalized.permute(1, 2, 0), 'direct-smooth-image-QK')
                    smooth = numerator / (128 ** .5 * denominator.T[:, None])
                    native_logits = mm(q, v['nativeKeys'][0, 512:].double().permute(1, 2, 0), 'native-image-QK') / 128 ** .5
                    values = v['bankV'][0].double().permute(1, 0, 2)
                    sh = read_head(torch.cat((v['textLogits'], smooth), -1), values, mm, 'smooth-reference-AV')
                    nh = read_head(torch.cat((v['textLogits'], native_logits), -1), values, mm, 'native-key-reference-AV')
                    return dict(rawQ=raw, qDenominator=denominator, directNumerator=numerator,
                                smoothImageLogits=smooth, nativeImageLogits=native_logits,
                                smoothH=sh['H_FP64'], nativeKeysH=nh['H_FP64'],
                                smoothLogPartition=sh['peak'] + sh['partition'].log(),
                                nativeLogPartition=nh['peak'] + nh['partition'].log())
                run('reference', values, reference, {'rawQ': abi([1024, 24, 128]), 'qDenominator': abi([1024, 24]),
                    **{n: abi([24, 4, 1024]) for n in ['directNumerator', 'smoothImageLogits', 'nativeImageLogits']},
                    **{n: abi([24, 4, 128]) for n in ['smoothH', 'nativeKeysH']},
                    **{n: abi([24, 4, 1]) for n in ['smoothLogPartition', 'nativeLogPartition']}},
                    {'actorSealSHA256': sha(out / 'actor-seal.json'), 'actorModLNOutputFrame': actor['modln']['outputFrame']['frame_id']})
                extra = {'actorSealSHA256': sha(out / 'actor-seal.json')}
    finally:
        torch.nn.functional.linear = original_linear
    if before != {n: (x.data_ptr(), x._version, payload(x)) for n, x in weights.items()} or snapshot(torch) != numerical:
        raise ValueError('Parameter/numerical-profile closure differs')
    if not torch.equal(rng, torch.get_rng_state()) or not all(torch.equal(a, b) for a, b in zip(rng_cuda, torch.cuda.get_rng_state_all())):
        raise ValueError('RNG closure differs')
    assets_unchanged(assets);total = sum(x['nominalMACs'] for x in operations)
    expected = {'compiler': 10286530560, 'actor': 1346371584, 'reference': 9726590976}[phase]
    if total != expected:
        raise ValueError('Physical matrix-work ledger differs: ' + str(total))
    save(out / (phase + '.json'), {**records, **extra})
    save(out / (phase + '-seal.json'), {'requestSHA256': r['requestSHA256'], phase + 'SHA256': sha(out / (phase + '.json'))})
    save(out / (phase + '-execution.json'), {'loadedParameters': loaded, 'operations': operations,
        'nominalMatrixMACs': total, 'parameterInputRNGClosure': True, 'numericalProfile': numerical,
        'assets': assets, 'header': head, 'arithmeticTimings': timings,
        'wallSecondsAfterImports': time.perf_counter() - started, 'CPUSecondsAfterImports': time.process_time() - cpu,
        'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'peakCUDAReservedBytes': torch.cuda.max_memory_reserved(),
        'fileQuarantine': {'allowedFiles': sorted(allowed), 'deniedCanaries': denied, 'successfulFileOpenCounts': accesses}})
    print('IMAGE_DUAL_' + phase.upper() + '_SEALED', flush=True)


if __name__ == '__main__':
    execute(sys.argv[1], sys.argv[2])
