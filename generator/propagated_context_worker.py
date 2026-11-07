"""Fresh full-bank producer, isolated complementary reads and own next queries."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha
from paired_fields import load, retain, payload
from projected_complement import fields, load_selected
from propagated_context import BANK_PARAMETERS, SELECTED_PARAMETERS, RETURN_PARAMETERS, NEXT_PARAMETERS, banks, selected, advance, next_query


def output_abis(name, values, writes, support):
    """Declare new output slots explicitly; Frame does not infer their ABI."""
    if name.startswith('bank_'):
        image, text = values['image'], values['text']
        width = values['transformer_blocks.1.attn.norm_k.weight'].numel()
        shape = [text.shape[0], text.shape[1] + image.shape[1], text.shape[2] // width, width]
        return {n: {'shape': shape, 'dtype': str(text.dtype)} for n in writes}
    if name == 'selected':
        b, _, width = values['text'].shape
        shape = [b, len(support), width]
        head_shape = [b, len(support), *values['bankK'].shape[2:]]
        result = {'R': {'shape': shape, 'dtype': str(values['text'].dtype)},
                  'txtMod': {'shape': list(values['txtMod'].shape), 'dtype': str(values['txtMod'].dtype)}}
        result.update({n: {'shape': head_shape, 'dtype': str(values['text'].dtype)} for n in ['Q', 'K', 'V']})
        result.update({n: {'shape': [len(support), values['full' + n[0].upper() + n[1:]].shape[-1]],
                           'dtype': str(values['full' + n[0].upper() + n[1:]].dtype)} for n in ['ropeCos', 'ropeSin']})
        return result
    if name == 'source':
        return {n: {'shape': list(values['R'].shape), 'dtype': str(values['R'].dtype)} for n in writes}
    if name == 'next_query':
        b, rows, width = values['rows'].shape
        d = values['transformer_blocks.2.attn.norm_added_q.weight'].numel()
        return {'nextQ': {'shape': [b, rows, width // d, d], 'dtype': str(values['rows'].dtype)}}
    raise ValueError('Unknown typed computation')


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
    root = Path(root).resolve()
    r, s = read(root / 'request.json'), read(root / 'inputs.json')
    verify(root, r)
    out = root / 'response-result'
    out.mkdir(exist_ok=phase != 'banks')
    shared = None
    if phase != 'banks':
        shared, seal = read(out / 'banks.json'), read(out / 'banks-seal.json')
        if seal['banksSHA256'] != sha(out / 'banks.json') or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Fresh bank/shared release differs')
        if not shared['outsideSupportKVChanged']:
            raise ValueError('Complement contrast inapplicable; driver must retain bank observations without serving arms')
    if phase == 'control':
        primary, seal = read(out / 'primary.json'), read(out / 'primary-seal.json')
        if seal['primarySHA256'] != sha(out / 'primary.json') or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Primary must seal and exit first')
    allowed = [str(Path(s['modelRoot']) / n) for n in s['imageAssets']]
    denied = list(s['deniedHistoricalArrays'])
    if phase == 'banks':
        for cut in s['sourceCuts'].values():
            allowed += [cut['cutFile']['path']] + [p for page in cut['pages'].values() for p in page['files']]
    else:
        role = 'Q' if phase == 'primary' else 'P'
        allowed += [x['path'] for x in shared['selected']['fields'].values()] + [x['path'] for x in shared['banks'][role]['fields'].values()]
        denied += [x['path'] for x in shared['banks']['P' if role == 'Q' else 'Q']['fields'].values()]
        if phase == 'control':
            denied += [x['path'] for x in fields(primary)]
    accesses = quarantine(root, allowed, denied)
    started, cpu = time.perf_counter(), time.process_time()
    numerical = profile(torch, 4)
    if numerical != s['numericalProfile']:
        raise ValueError('Pinned profile differs')
    model, assets = source_checks(s, diffusers)
    checkpoint = model / 'transformer/diffusion_pytorch_model.safetensors'
    entries, head = header(checkpoint)
    names = BANK_PARAMETERS + SELECTED_PARAMETERS if phase == 'banks' else RETURN_PARAMETERS + NEXT_PARAMETERS
    weights, loaded = load_selected(checkpoint, names, entries)
    original = {x['source']['tensor']: x['source'] for x in s['originalParameterArtifacts']}
    for x in loaded:
        expected = original[x['name']]
        if x['payloadSHA256'] != expected['payload_sha256'] or x['shape'] != expected['shape'] or x['dataOffsets'] != expected['data_offsets']:
            raise ValueError('Original parameter payload/header differs')
    before = {n: (v.data_ptr(), v._version, payload(v)) for n, v in weights.items()}
    rng, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    torch.cuda.reset_peak_memory_stats()
    linears, attentions, timings = [], [], []
    old_linear, old_attention = torch.nn.functional.linear, native.dispatch_attention_fn
    stage = phase
    def linear(x, w, bias=None):
        matches = [n for n, value in weights.items() if value is w]
        if len(matches) != 1 or bias is not None:
            raise ValueError('Undeclared learned affine or bias')
        linears.append({'stage': stage, 'parameter': matches[0], 'inputShape': list(x.shape), 'inputStride': list(x.stride()),
                        'weightShape': list(w.shape), 'dtype': str(x.dtype), 'device': x.device.type,
                        'nominalMACs': x.numel() // x.shape[-1] * w.numel()})
        return old_linear(x, w, bias)
    def attention(q, k, v, **kwargs):
        if tuple(q.shape) != (1, 4, 24, 128) or tuple(k.shape) != (1, 1536, 24, 128) or k.shape != v.shape:
            raise ValueError('Rectangular source geometry differs')
        value = old_attention(q, k, v, **kwargs)
        attentions.append({'queryShape': list(q.shape), 'keyShape': list(k.shape), 'keyStride': list(k.stride()),
                           'valueStride': list(v.stride()), 'nominalMACs': 37748736, 'outputPayloadSHA256': payload(value)})
        return value
    registry = ActRegistry()
    def run(name, values, operation, writes, metadata):
        nonlocal stage
        stage = name
        input_before = {n: payload(x) for n, x in values.items()}
        frame = Frame.from_values(values, kinds={n: 'tensor' for n in values},
            abis={n: {'shape': list(x.shape), 'stride': list(x.stride()), 'dtype': str(x.dtype)} for n, x in values.items()},
            slot_fingerprints={n: ProvidedFingerprint.content_sha256(payload(x)) for n, x in values.items()},
            program_fingerprint=r['requestSHA256'], metadata=metadata)
        declared = output_abis(name, values, writes, r['support'])
        spec = ActSpec('tce.propagated_context_' + name, 'v1', reads=tuple(values), writes=tuple(writes),
                       write_kinds={n: 'tensor' for n in writes}, write_abis=declared,
                       numerical_contract='pinned-original-maps-own-state-no-target-refresh')
        captures = {}
        def act(v):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            got = operation(v)
            torch.cuda.synchronize()
            timings.append({'stage': name, 'arithmeticSecondsBeforeCapture': time.perf_counter() - tick})
            if declared != {n: {'shape': list(got[n].shape), 'dtype': str(got[n].dtype)} for n in writes}:
                raise ValueError('Actual output geometry differs from declared typed ABI')
            captures.update({n: retain(root / 'arrays' / phase / name, n, x) for n, x in got.items()})
            return {n: got[n] for n in writes}, {'capturedPayloads': {n: captures[n]['payloadSHA256'] for n in writes}}
        registry.register(spec, name, act)
        after, evidence = registry.run(spec.identity, name, frame)
        if input_before != {n: payload(x) for n, x in values.items()}:
            raise ValueError('Actual typed input mutated')
        return after, {'fields': captures, 'sourceFrame': frame.to_dict(), 'outputFrame': after.to_dict(), 'evidence': evidence.to_dict()}
    try:
        torch.nn.functional.linear, native.dispatch_attention_fn = linear, attention
        with torch.inference_mode():
            if phase == 'banks':
                complete, records, states = {}, {}, {}
                for role in ['P', 'Q']:
                    state = load_pages(s['sourceCuts'][role])
                    states[role] = state
                    context = state['context']
                    values = dict(image=state['image'], text=state['text'], imgMod=context['img_mod'], txtMod=context['txt_mod'],
                                  ropeCos=context['rope'][0], ropeSin=context['rope'][1], **{n: weights[n] for n in BANK_PARAMETERS})
                    def form(v):
                        return banks(v['image'], v['text'], v['imgMod'], v['txtMod'], (v['ropeCos'], v['ropeSin']), weights, native)
                    complete[role], records[role] = run('bank_' + role, values, form, ['K', 'V'], {'sourceCut': s['sourceCuts'][role]['cutID']})
                p = states['P'];context = p['context']
                values = dict(text=p['text'], txtMod=context['txt_mod'], fullRopeCos=context['rope'][0], fullRopeSin=context['rope'][1],
                              bankK=complete['P'].values['K'], bankV=complete['P'].values['V'], **{n: weights[n] for n in SELECTED_PARAMETERS})
                def select(v):
                    return selected(v['text'], v['txtMod'], (v['fullRopeCos'], v['fullRopeSin']), {'K': v['bankK'], 'V': v['bankV']}, r['support'], weights, native)
                _, selected_record = run('selected', values, select, ['R', 'txtMod', 'Q', 'K', 'V', 'ropeCos', 'ropeSin'],
                                         {'sourceCut': s['sourceCuts']['P']['cutID'], 'sourceBankOutputFrame': complete['P'].frame_id, 'support': r['support']})
                diff = {}
                for n in ['K', 'V']:
                    changed = (complete['P'].values[n] != complete['Q'].values[n]).flatten(2).any(-1)[0]
                    diff[n] = {'changedRows': torch.nonzero(changed).flatten().cpu().tolist(),
                               'outsideSupportRows': [j for j in torch.nonzero(changed).flatten().cpu().tolist() if j not in r['support']]}
                result = {'banks': records, 'selected': selected_record, 'bankDifferences': diff,
                          'outsideSupportKVChanged': any(x['outsideSupportRows'] for x in diff.values())}
            else:
                values = {n: load(x, 'cuda') for n, x in shared['selected']['fields'].items()}
                values.update(bankK=load(shared['banks'][role]['fields']['K'], 'cuda'), bankV=load(shared['banks'][role]['fields']['V'], 'cuda'),
                              **{n: weights[n] for n in RETURN_PARAMETERS})
                def produce(v):
                    return advance(v, v['bankK'], v['bankV'], r['support'], weights, native)
                after, source_record = run('source', values, produce, ['A', 'U', 'M', 'T'],
                    {'complement': role, 'sharedSealSHA256': sha(out / 'banks-seal.json'), 'support': r['support']})
                query_values = dict(rows=after.values['T'], txtMod=values['txtMod'], ropeCos=values['ropeCos'], ropeSin=values['ropeSin'],
                                    **{n: weights[n] for n in NEXT_PARAMETERS})
                def query(v):
                    return next_query(v['rows'], v['txtMod'], (v['ropeCos'], v['ropeSin']), weights, native)
                _, query_record = run('next_query', query_values, query, ['nextQ'], {'sourceOutputFrame': after.frame_id, 'nextJoint': 2})
                result = {'complement': role, 'source': source_record, 'nextQuery': query_record,
                          'sharedSealSHA256': sha(out / 'banks-seal.json'),
                          'primarySealSHA256': sha(out / 'primary-seal.json') if phase == 'control' else None}
    finally:
        torch.nn.functional.linear, native.dispatch_attention_fn = old_linear, old_attention
    if before != {n: (v.data_ptr(), v._version, payload(v)) for n, v in weights.items()} or snapshot(torch) != numerical:
        raise ValueError('Parameter/profile closure differs')
    if not torch.equal(rng, torch.get_rng_state()) or not all(torch.equal(a, b) for a, b in zip(rng_cuda, torch.cuda.get_rng_state_all())):
        raise ValueError('RNG closure differs')
    assets_unchanged(assets)
    total = sum(x['nominalMACs'] for x in linears + attentions)
    if total != (58019807232 if phase == 'banks' else 452984832) or len(linears) != (9 if phase == 'banks' else 4) or len(attentions) != (0 if phase == 'banks' else 1):
        raise ValueError('Executed physical work differs')
    save(out / (phase + '.json'), result)
    save(out / (phase + '-seal.json'), {'requestSHA256': r['requestSHA256'], phase + 'SHA256': sha(out / (phase + '.json'))})
    save(out / (phase + '-execution.json'), {'loadedParameters': loaded, 'linears': linears, 'attentions': attentions,
         'nominalMatrixMACs': total, 'parameterInputRNGClosure': True, 'numericalProfile': numerical, 'assets': assets, 'header': head,
         'arithmeticTimings': timings, 'wallSecondsAfterImports': time.perf_counter() - started, 'CPUSecondsAfterImports': time.process_time() - cpu,
         'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'peakCUDAReservedBytes': torch.cuda.max_memory_reserved(),
         'fileQuarantine': {'allowedFiles': sorted(allowed), 'deniedCanaries': denied, 'successfulFileOpenCounts': accesses}})
    print('PROPAGATED_CONTEXT_' + phase.upper() + '_SEALED', sha(out / (phase + '-seal.json')), flush=True)


if __name__ == '__main__':
    execute(sys.argv[1], sys.argv[2])
