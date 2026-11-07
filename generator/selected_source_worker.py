"""Produce demanded text rows before admitting any archived writer returns."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha
from paired_fields import retain, payload
from selected_source import PARAMETERS, form, splice, load_pages


def execute(root):
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from mdb.titan import Frame, ActSpec, ActRegistry, ProvidedFingerprint
    from complete_masked_study import profile
    from flux_clock_study import verify, source_checks, assets_unchanged
    from flux_clock import header, snapshot
    from sparse_text_kv import quarantine, load_weights
    root = Path(root).resolve()
    r, s = read(root / 'request.json'), read(root / 'inputs.json')
    verify(root, r)
    out = root / 'response-result'
    out.mkdir(exist_ok=False)
    allowed = [p for rec in s['sourceCuts'].values() for item in rec['pages'].values() for p in item['files']]
    allowed += list(s['sourceMetadataFiles']) + [str(Path(s['modelRoot']) / n) for n in s['imageAssets']]
    denied = [str(root / 'targets.json'), s['parentPrimaryJSON'], s['parentPrimarySealJSON'], *s['deniedArrays']]
    accesses = quarantine(root, allowed, denied)
    started, cpu = time.perf_counter(), time.process_time()
    numerical = profile(torch, 4)
    if numerical != s['numericalProfile']:
        raise ValueError('Original profile differs')
    model, assets = source_checks(s, diffusers)
    checkpoint = model / 'transformer/diffusion_pytorch_model.safetensors'
    entries, header_record = header(checkpoint)
    weights, loaded = load_weights(checkpoint, PARAMETERS, entries)
    original = {x['source']['tensor']: x['source'] for x in s['originalParameterArtifacts']}
    for x in loaded:
        e = original[x['name']]
        if x['payloadSHA256'] != e['payload_sha256'] or x['shape'] != e['shape'] or x['dataOffsets'] != e['data_offsets']:
            raise ValueError('Original source parameter identity differs')
    if sum(x['bytes'] for x in loaded) != 283116288:
        raise ValueError('Source parameter geometry differs')
    before = {n: (v.data_ptr(), v._version, payload(v)) for n, v in weights.items()}
    rng, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    torch.cuda.reset_peak_memory_stats()
    linears, attentions, timings, results, tensors = [], [], [], {}, {}
    old_linear, old_attention = torch.nn.functional.linear, native.dispatch_attention_fn
    role = None
    def linear(x, w, bias=None):
        names = [n for n, v in weights.items() if v is w]
        if len(names) != 1 or bias is not None:
            raise ValueError('Undeclared source affine')
        linears.append({'role': role, 'parameter': names[0], 'inputShape': list(x.shape),
                        'inputStride': list(x.stride()), 'weightShape': list(w.shape),
                        'dtype': str(x.dtype), 'device': x.device.type,
                        'nominalMACs': x.numel() // x.shape[-1] * w.numel()})
        return old_linear(x, w, bias)
    def attention(q, k, v, **kwargs):
        if tuple(q.shape) != (1, 4, 24, 128) or tuple(k.shape) != (1, 1536, 24, 128) or k.shape != v.shape:
            raise ValueError('Selected-query/full-KV attention geometry differs')
        y = old_attention(q, k, v, **kwargs)
        attentions.append({'role': role, 'queryShape': list(q.shape), 'keyShape': list(k.shape),
                           'valueShape': list(v.shape), 'queryStride': list(q.stride()),
                           'outputPayloadSHA256': payload(y), 'nominalQKAVMACs': 37748736,
                           'dtype': str(q.dtype), 'device': q.device.type, 'requestedBackend': 'auto'})
        return y
    def frame(values, metadata):
        return Frame.from_values(values, kinds={n: 'tensor' for n in values},
            abis={n: {'shape': list(x.shape), 'dtype': str(x.dtype), 'stride': list(x.stride())} for n, x in values.items()},
            slot_fingerprints={n: ProvidedFingerprint.content_sha256(payload(x)) for n, x in values.items()},
            program_fingerprint=r['requestSHA256'], metadata=metadata)
    source_fields = ('R', 'A', 'U', 'M', 'T', 'gateA', 'gateM')
    registry, active = ActRegistry(), {}
    spec = ActSpec('tce.selected_source', 'v1', reads=tuple(['image', 'text', 'img_mod', 'txt_mod', 'cos', 'sin'] + list(PARAMETERS)),
                   writes=source_fields, write_kinds={n: 'tensor' for n in source_fields},
                   numerical_contract='all-current-KV-four-text-Q-original-rowwise-sequential-MLP-BF16')
    def operation(v):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        got = form(v['image'], v['text'], v['img_mod'], v['txt_mod'], (v['cos'], v['sin']), r['support'], weights, native)
        torch.cuda.synchronize()
        timings.append({'role': role, 'arithmeticSecondsBeforeCapture': time.perf_counter() - tick})
        tensors[role] = {n: got[n] for n in source_fields}
        active.update({n: retain(root / 'arrays/source' / role, n, value) for n, value in got.items()})
        return {n: got[n] for n in source_fields}, {'capturedPayloads': {n: active[n]['payloadSHA256'] for n in source_fields}}
    registry.register(spec, 'selected-text-backward-slice', operation)
    try:
        torch.nn.functional.linear, native.dispatch_attention_fn = linear, attention
        with torch.inference_mode():
            for role in ['P', 'Q']:
                state = load_pages(s['sourceCuts'][role])
                context = state['context']
                values = {n: state[n] for n in ['image', 'text']}
                values.update(img_mod=context['img_mod'], txt_mod=context['txt_mod'], cos=context['rope'][0], sin=context['rope'][1], **weights)
                input_before = {n: payload(x) for n, x in values.items()}
                active = {}
                parent = frame(values, {'role': role, 'sourceCut': s['sourceCuts'][role]['cutID'], 'support': r['support'], 'producerPoint': r['producerPoint']})
                after, evidence = registry.run(spec.identity, 'selected-text-backward-slice', parent)
                if input_before != {n: payload(x) for n, x in values.items()}:
                    raise ValueError('Source admitted input changed')
                results[role] = {'fields': active, 'sourceFrame': parent.to_dict(), 'outputFrame': after.to_dict(), 'evidence': evidence.to_dict()}
            # The original family uses common recipient Q modulation, not a new hybrid MLP.
            if any(payload(tensors['P'][n]) != payload(tensors['Q'][n]) for n in ['gateA', 'gateM']):
                raise ValueError('Declared common source gates differ')
            splice_values = {role + '.' + n: x for role, fields in tensors.items() for n, x in fields.items()}
            splice_spec = ActSpec('tce.selected_source_splice', 'v1', reads=tuple(splice_values), writes=('QQQ', 'PQP'),
                                 write_kinds={'QQQ': 'tensor', 'PQP': 'tensor'}, numerical_contract='recipient-Q-gates-P-MLP-from-own-P-attention')
            splice_fields = {}
            def assemble(v):
                own = splice({n: v['P.' + n] for n in source_fields}, {n: v['Q.' + n] for n in source_fields})
                splice_fields.update({n: retain(root / 'arrays/splice', n, x) for n, x in own.items()})
                return own, {'capturedPayloads': {n: x['payloadSHA256'] for n, x in splice_fields.items()}}
            registry.register(splice_spec, 'ordered-family-splice', assemble)
            parent = frame(splice_values, {'support': r['support'], 'sourceCuts': {n: x['cutID'] for n, x in s['sourceCuts'].items()}})
            after, evidence = registry.run(splice_spec.identity, 'ordered-family-splice', parent)
    finally:
        torch.nn.functional.linear, native.dispatch_attention_fn = old_linear, old_attention
    if before != {n: (v.data_ptr(), v._version, payload(v)) for n, v in weights.items()} or snapshot(torch) != numerical:
        raise ValueError('Source parameter/profile changed')
    if not torch.equal(rng, torch.get_rng_state()) or not all(torch.equal(a, b) for a, b in zip(rng_cuda, torch.cuda.get_rng_state_all())):
        raise ValueError('Source RNG changed')
    assets_unchanged(assets)
    affines = sum(x['nominalMACs'] for x in linears)
    attention_macs = sum(x['nominalQKAVMACs'] for x in attentions)
    if len(linears) != 16 or affines != 58812530688 or len(attentions) != 2 or attention_macs != 75497472:
        raise ValueError('Physical source execution ledger differs')
    save(out / 'producer.json', {'sources': results, 'splice': {'fields': splice_fields, 'sourceFrame': parent.to_dict(),
         'outputFrame': after.to_dict(), 'evidence': evidence.to_dict()}, 'commonGatePayloadsExact': True})
    save(out / 'producer-seal.json', {'requestSHA256': r['requestSHA256'], 'producerSHA256': sha(out / 'producer.json')})
    save(out / 'producer-execution.json', {'linears': linears, 'attentions': attentions, 'affineMatrixMACs': affines,
         'QKAVMatrixMACs': attention_macs, 'nominalMatrixMACs': affines + attention_macs, 'loadedParameters': loaded,
         'numericalProfile': numerical, 'header': header_record, 'assets': assets, 'parameterInputRNGClosure': True,
         'arithmeticTimings': timings, 'wallSecondsAfterImports': time.perf_counter() - started,
         'CPUSecondsAfterImports': time.process_time() - cpu, 'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(),
         'peakCUDAReservedBytes': torch.cuda.max_memory_reserved(), 'fileQuarantine': {'allowedFiles': sorted(allowed),
         'deniedCanaries': denied, 'successfulFileOpenCounts': accesses, 'scope': 'No historical writer return, finished row or cached source KV admitted.'}})
    print('SELECTED_SOURCE_PRODUCER_SEALED', sha(out / 'producer-seal.json'), flush=True)


if __name__ == '__main__':
    execute(sys.argv[1])
