"""Shared selected operands; isolated Q-complement primary then P control."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha
from paired_fields import load, retain, payload
from source_complement import SHARED_PARAMETERS, RETURN_PARAMETERS, selected, source, recipient
from projected_complement import ACTOR_PARAMETERS, WO, fields, load_selected


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
    out.mkdir(exist_ok=phase != 'shared')
    shared = None
    if phase != 'shared':
        shared = read(out / 'shared.json')
        seal = read(out / 'shared-seal.json')
        if seal['sharedSHA256'] != sha(out / 'shared.json') or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Shared selected-operand release differs')
    if phase == 'control':
        primary, seal = read(out / 'primary.json'), read(out / 'primary-seal.json')
        if seal['primarySHA256'] != sha(out / 'primary.json') or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Primary must seal and exit before P complement is admitted')
    assets_allowed = [str(Path(s['modelRoot']) / n) for n in s['imageAssets']]
    denied = [str(root / 'targets.json'), *s['deniedTargets']]
    if phase in ['shared', 'primary']:
        denied += [x['path'] for x in s['banks']['P'].values()]
    if phase == 'shared':
        allowed = [s['pCut']['cutFile']['path']] + [p for page in s['pCut']['pages'].values() for p in page['files']] + assets_allowed
    else:
        bank_role = 'Q' if phase == 'primary' else 'P'
        allowed = [x['path'] for x in shared['fields'].values()] + [x['path'] for x in s['banks'][bank_role].values()]
        allowed += [s['qAttention']['path']] + [x['path'] for x in s['readerInputs'].values()] + assets_allowed
        if phase == 'control':
            denied += [x['path'] for x in fields(primary)]
    accesses = quarantine(root, allowed, denied)
    started, cpu = time.perf_counter(), time.process_time()
    numerical = profile(torch, 4)
    if numerical != s['numericalProfile']:
        raise ValueError('Pinned numerical profile differs')
    model, assets = source_checks(s, diffusers)
    checkpoint = model / 'transformer/diffusion_pytorch_model.safetensors'
    entries, head = header(checkpoint)
    names = SHARED_PARAMETERS if phase == 'shared' else RETURN_PARAMETERS + ACTOR_PARAMETERS
    weights, loaded = load_selected(checkpoint, names, entries)
    original = {x['source']['tensor']: x['source'] for x in s['originalParameterArtifacts']}
    for x in loaded:
        e = original[x['name']]
        if x['payloadSHA256'] != e['payload_sha256'] or x['shape'] != e['shape'] or x['dataOffsets'] != e['data_offsets']:
            raise ValueError('Original selected parameter identity differs')
    wo = None if phase == 'shared' else weights[WO].double()
    before = {n: (v.data_ptr(), v._version, payload(v)) for n, v in weights.items()}
    wo_before = None if wo is None else (wo.data_ptr(), wo._version, payload(wo))
    rng, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    torch.cuda.reset_peak_memory_stats()
    linears, attentions, timings, active = [], [], [], {}
    old_linear, old_attention = torch.nn.functional.linear, native.dispatch_attention_fn
    stage = phase
    def linear(x, w, bias=None):
        extra = {}
        if wo is not None and w.untyped_storage().data_ptr() == wo.untyped_storage().data_ptr():
            name = WO
            extra = {'head': w.storage_offset() // 128, 'physicalColumnSlice': True}
            if tuple(w.shape) != (3072, 128):
                raise ValueError('Reader must use physical Wo head blocks')
        else:
            matches = [n for n, value in weights.items() if value is w]
            if len(matches) != 1:
                raise ValueError('Undeclared learned affine')
            name = matches[0]
        if bias is not None:
            raise ValueError('Unexpected affine bias')
        linears.append({'stage': stage, 'parameter': name, 'inputShape': list(x.shape), 'inputStride': list(x.stride()),
                        'weightShape': list(w.shape), 'dtype': str(x.dtype), 'device': x.device.type,
                        'nominalMACs': x.numel() // x.shape[-1] * w.numel(), **extra})
        return old_linear(x, w, bias)
    def attention(q, k, v, **kwargs):
        if tuple(q.shape) != (1, 4, 24, 128) or tuple(k.shape) != (1, 1536, 24, 128) or k.shape != v.shape:
            raise ValueError('Source attention geometry differs')
        result = old_attention(q, k, v, **kwargs)
        attentions.append({'queryShape': list(q.shape), 'keyShape': list(k.shape), 'keyStride': list(k.stride()),
                           'valueShape': list(v.shape), 'valueStride': list(v.stride()), 'nominalMACs': 37748736,
                           'outputPayloadSHA256': payload(result), 'requestedBackend': 'auto'})
        return result
    def frame(values, metadata):
        return Frame.from_values(values, kinds={n: 'tensor' for n in values},
            abis={n: {'shape': list(x.shape), 'dtype': str(x.dtype), 'stride': list(x.stride())} for n, x in values.items()},
            slot_fingerprints={n: ProvidedFingerprint.content_sha256(payload(x)) for n, x in values.items()},
            program_fingerprint=r['requestSHA256'], metadata=metadata)
    def keep(group, values):
        return {n: retain(root / 'arrays' / phase / group, n, x) for n, x in values.items()}
    registry, result = ActRegistry(), {}
    try:
        torch.nn.functional.linear, native.dispatch_attention_fn = linear, attention
        with torch.inference_mode():
            if phase == 'shared':
                state = load_pages(s['pCut'])
                context = state['context']
                values = dict(text=state['text'], mod=context['txt_mod'], cos=context['rope'][0], sin=context['rope'][1], **weights)
                input_before = {n: payload(x) for n, x in values.items()}
                spec = ActSpec('tce.source_complement_selected', 'v1', reads=tuple(values), writes=('R', 'txtMod', 'Q', 'K', 'V'),
                               write_kinds={n: 'tensor' for n in ['R', 'txtMod', 'Q', 'K', 'V']},
                               numerical_contract='authenticated-P-rows-original-selected-QKV-absolute-RoPE')
                def operation(v):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    got = selected(v['text'], v['mod'], (v['cos'], v['sin']), r['support'], weights, native)
                    torch.cuda.synchronize()
                    timings.append({'stage': 'selected-operands', 'arithmeticSecondsBeforeCapture': time.perf_counter() - tick})
                    parts = got['txtMod'][:, None, :].chunk(6, -1)
                    if payload(parts[2]) != s['qGatePayloads']['gateA'] or payload(parts[5]) != s['qGatePayloads']['gateM']:
                        raise ValueError('Recipient Q gate identity differs')
                    active.update(keep('operands', got))
                    return got, {'capturedPayloads': {n: x['payloadSHA256'] for n, x in active.items()}}
                registry.register(spec, 'selected-P-operands', operation)
                parent = frame(values, {'sourceCut': s['pCut']['cutID'], 'support': r['support']})
                after, evidence = registry.run(spec.identity, 'selected-P-operands', parent)
                result = {'fields': active, 'sourceFrame': parent.to_dict(), 'outputFrame': after.to_dict(), 'evidence': evidence.to_dict()}
                if input_before != {n: payload(x) for n, x in values.items()}:
                    raise ValueError('Selected source input changed')
            else:
                values = {n: load(x, 'cuda') for n, x in shared['fields'].items()}
                values.update(bankK=load(s['banks'][bank_role]['K'], 'cuda'), bankV=load(s['banks'][bank_role]['V'], 'cuda'),
                              qAttention=load(s['qAttention'], 'cuda'), **{n: weights[n] for n in RETURN_PARAMETERS})
                input_before = {n: payload(x) for n, x in values.items()}
                spec = ActSpec('tce.source_complement_return', 'v1', reads=tuple(values), writes=('A', 'U', 'M', 'PQP'),
                               write_kinds={n: 'tensor' for n in ['A', 'U', 'M', 'PQP']},
                               numerical_contract='fixed-selected-operands-one-complement-own-sequential-P-MLP-Q-direct-return')
                def operation(v):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    got = source({n: v[n] for n in shared['fields']}, v['bankK'], v['bankV'], v['qAttention'], r['support'], weights, native)
                    torch.cuda.synchronize()
                    timings.append({'stage': 'source-return', 'arithmeticSecondsBeforeCapture': time.perf_counter() - tick})
                    active.update(keep('source', got))
                    return {n: got[n] for n in ['A', 'U', 'M', 'PQP']}, {'capturedPayloads': {n: active[n]['payloadSHA256'] for n in ['A', 'U', 'M', 'PQP']}}
                registry.register(spec, 'one-complement-own-source', operation)
                parent = frame(values, {'complement': bank_role, 'support': r['support'], 'sharedSealSHA256': sha(out / 'shared-seal.json')})
                after, evidence = registry.run(spec.identity, 'one-complement-own-source', parent)
                source_result = {'fields': active, 'sourceFrame': parent.to_dict(), 'outputFrame': after.to_dict(), 'evidence': evidence.to_dict()}
                if input_before != {n: payload(x) for n, x in values.items()}:
                    raise ValueError('Shared selected operands or retained context changed')
                stage, active = 'recipient', {}
                reader = {n: load(x, 'cpu' if n in ['projectedN', 'm', 'z'] else 'cuda') for n, x in s['readerInputs'].items()}
                reader.update(textRows=after.values['PQP'], Kweight=weights['transformer_blocks.1.attn.add_k_proj.weight'],
                              Vweight=weights['transformer_blocks.1.attn.add_v_proj.weight'],
                              KnormWeight=weights['transformer_blocks.1.attn.norm_added_k.weight'], WoFP64=wo)
                reader_before = {n: payload(x) for n, x in reader.items()}
                spec2 = ActSpec('tce.source_complement_recipient', 'v1', reads=tuple(reader), writes=('R_attention',),
                                write_kinds={'R_attention': 'tensor'}, numerical_contract='unchanged-projected-positive-read-own-PQP-source-rows')
                def consume(v):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    got = recipient(v, v['textRows'], weights, v['WoFP64'], native)
                    torch.cuda.synchronize()
                    timings.append({'stage': 'recipient', 'arithmeticSecondsBeforeCapture': time.perf_counter() - tick})
                    active.update(keep('recipient', got))
                    return {'R_attention': got['R_attention']}, {'capturedPayloadSHA256': active['R_attention']['payloadSHA256']}
                registry.register(spec2, 'projected-recipient', consume)
                parent2 = frame(reader, {'complement': bank_role, 'sourceOutputFrame': after.frame_id})
                after2, evidence2 = registry.run(spec2.identity, 'projected-recipient', parent2)
                if reader_before != {n: payload(x) for n, x in reader.items()}:
                    raise ValueError('Recipient inputs changed')
                result = {'complement': bank_role, 'source': source_result,
                          'recipient': {'fields': active, 'sourceFrame': parent2.to_dict(), 'outputFrame': after2.to_dict(), 'evidence': evidence2.to_dict()},
                          'sharedSealSHA256': sha(out / 'shared-seal.json'),
                          'primarySealSHA256': sha(out / 'primary-seal.json') if phase == 'control' else None}
    finally:
        torch.nn.functional.linear, native.dispatch_attention_fn = old_linear, old_attention
    if before != {n: (v.data_ptr(), v._version, payload(v)) for n, v in weights.items()} or snapshot(torch) != numerical:
        raise ValueError('Parameter/profile closure differs')
    if wo is not None and wo_before != (wo.data_ptr(), wo._version, payload(wo)):
        raise ValueError('Reader FP64 map changed')
    if not torch.equal(rng, torch.get_rng_state()) or not all(torch.equal(a, b) for a, b in zip(rng_cuda, torch.cuda.get_rng_state_all())):
        raise ValueError('RNG closure differs')
    assets_unchanged(assets)
    contractions = [] if phase == 'shared' else [{'kind': 'recipient-QK', 'nominalMACs': 12582912}, {'kind': 'recipient-projected-AV', 'nominalMACs': 301989888}]
    total = sum(x['nominalMACs'] for x in linears + attentions + contractions)
    if total != (113246208 if phase == 'shared' else 843055104) or len(linears) != (3 if phase == 'shared' else 29):
        raise ValueError('Physical contraction ledger differs')
    save(out / (phase + '.json'), result)
    save(out / (phase + '-seal.json'), {'requestSHA256': r['requestSHA256'], phase + 'SHA256': sha(out / (phase + '.json'))})
    save(out / (phase + '-execution.json'), {'loadedParameters': loaded, 'linears': linears, 'attentions': attentions, 'contractions': contractions,
         'nominalMatrixMACs': total, 'parameterInputRNGClosure': True, 'numericalProfile': numerical, 'header': head, 'assets': assets,
         'WoFP64': None if wo is None else {'bytes': wo.numel() * 8, 'payloadSHA256': wo_before[2]}, 'arithmeticTimings': timings,
         'wallSecondsAfterImports': time.perf_counter() - started, 'CPUSecondsAfterImports': time.process_time() - cpu,
         'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'peakCUDAReservedBytes': torch.cuda.max_memory_reserved(),
         'fileQuarantine': {'allowedFiles': sorted(allowed), 'deniedCanaries': denied, 'successfulFileOpenCounts': accesses,
                            'scope': 'Exact numeric read allowlist. Primary cannot read P complement; control cannot read primary arrays; archived targets denied throughout.'}})
    print('SOURCE_COMPLEMENT_' + phase.upper() + '_SEALED', sha(out / (phase + '-seal.json')), flush=True)


if __name__ == '__main__':
    execute(sys.argv[1], sys.argv[2])
