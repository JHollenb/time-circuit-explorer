"""Two fixed K/V corners; each own joint1 state feeds its own next query."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha
from paired_fields import load, retain, payload
from projected_complement import fields, load_selected
from propagated_context import RETURN_PARAMETERS, NEXT_PARAMETERS, advance, next_query
from propagated_context_worker import output_abis

CORNERS = {'primary': ('P', 'Q'), 'diagnostic': ('Q', 'P')}


def execute(phase, root):
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from mdb.titan import Frame, ActSpec, ActRegistry, ProvidedFingerprint
    from complete_masked_study import profile
    from flux_clock_study import verify, source_checks, assets_unchanged
    from flux_clock import header, snapshot
    from sparse_text_kv import quarantine
    root = Path(root).resolve();r, s = read(root / 'request.json'), read(root / 'inputs.json');verify(root, r)
    out = root / 'response-result';out.mkdir(exist_ok=phase != 'primary')
    keys, values_role = CORNERS[phase]
    if r['corners'] != {n: list(x) for n, x in CORNERS.items()}:
        raise ValueError('Prespecified candidate/diagnostic changed')
    if phase == 'diagnostic':
        primary, seal = read(out / 'primary.json'), read(out / 'primary-seal.json')
        if seal['primarySHA256'] != sha(out / 'primary.json') or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Primary must seal and exit before diagnostic admission')
    allowed = [x['path'] for x in s['selected'].values()] + [s['banks'][keys]['K']['path'], s['banks'][values_role]['V']['path']]
    allowed += [str(Path(s['modelRoot']) / n) for n in s['imageAssets']]
    denied = [str(root / 'targets.json'), *s['deniedTargets'], s['banks']['Q' if keys == 'P' else 'P']['K']['path'],
              s['banks']['Q' if values_role == 'P' else 'P']['V']['path']]
    if phase == 'diagnostic':
        denied += [x['path'] for x in fields(primary)]
    accesses = quarantine(root, allowed, denied)
    started, cpu = time.perf_counter(), time.process_time();numerical = profile(torch, 4)
    if numerical != s['numericalProfile']:
        raise ValueError('Pinned numerical profile differs')
    model, assets = source_checks(s, diffusers);checkpoint = model / 'transformer/diffusion_pytorch_model.safetensors'
    entries, head = header(checkpoint);weights, loaded = load_selected(checkpoint, RETURN_PARAMETERS + NEXT_PARAMETERS, entries)
    original = {x['source']['tensor']: x['source'] for x in s['originalParameterArtifacts']}
    for x in loaded:
        old = original[x['name']]
        if x['payloadSHA256'] != old['payload_sha256'] or x['shape'] != old['shape'] or x['dataOffsets'] != old['data_offsets']:
            raise ValueError('Original map identity/header differs')
    before = {n: (v.data_ptr(), v._version, payload(v)) for n, v in weights.items()}
    rng, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all();torch.cuda.reset_peak_memory_stats()
    linears, attentions, timings = [], [], [];old_linear, old_attention = torch.nn.functional.linear, native.dispatch_attention_fn;stage = 'source'
    def linear(x, w, bias=None):
        matches = [n for n, value in weights.items() if value is w]
        if len(matches) != 1 or bias is not None:
            raise ValueError('Undeclared learned affine')
        linears.append({'stage': stage, 'parameter': matches[0], 'inputShape': list(x.shape), 'inputStride': list(x.stride()),
                        'weightShape': list(w.shape), 'dtype': str(x.dtype), 'device': x.device.type,
                        'nominalMACs': x.numel() // x.shape[-1] * w.numel()})
        return old_linear(x, w, bias)
    def attention(q, k, v, **kwargs):
        if tuple(q.shape) != (1, 4, 24, 128) or tuple(k.shape) != (1, 1536, 24, 128) or k.shape != v.shape:
            raise ValueError('Matched rectangular geometry differs')
        y = old_attention(q, k, v, **kwargs)
        attentions.append({'queryShape': list(q.shape), 'keyShape': list(k.shape), 'keyStride': list(k.stride()),
                           'valueStride': list(v.stride()), 'nominalMACs': 37748736, 'outputPayloadSHA256': payload(y)})
        return y
    registry = ActRegistry()
    def run(name, inputs, operation, writes, metadata):
        nonlocal stage
        stage = name;input_before = {n: payload(x) for n, x in inputs.items()}
        parent = Frame.from_values(inputs, kinds={n: 'tensor' for n in inputs},
            abis={n: {'shape': list(x.shape), 'stride': list(x.stride()), 'dtype': str(x.dtype)} for n, x in inputs.items()},
            slot_fingerprints={n: ProvidedFingerprint.content_sha256(payload(x)) for n, x in inputs.items()},
            program_fingerprint=r['requestSHA256'], metadata=metadata)
        declared = output_abis(name, inputs, writes, r['support'])
        spec = ActSpec('tce.stale_key_context_' + name, 'v1', reads=tuple(inputs), writes=tuple(writes),
                       write_kinds={n: 'tensor' for n in writes}, write_abis=declared,
                       numerical_contract='unchanged-native-finite-ownMLP-ownquery-fixed-KV-corner')
        captures = {}
        def act(v):
            torch.cuda.synchronize();tick = time.perf_counter();got = operation(v);torch.cuda.synchronize()
            timings.append({'stage': name, 'arithmeticSecondsBeforeCapture': time.perf_counter() - tick})
            if declared != {n: {'shape': list(got[n].shape), 'dtype': str(got[n].dtype)} for n in writes}:
                raise ValueError('Actual output ABI differs')
            captures.update({n: retain(root / 'arrays' / phase / name, n, x) for n, x in got.items()})
            return {n: got[n] for n in writes}, {'capturedPayloads': {n: captures[n]['payloadSHA256'] for n in writes}}
        registry.register(spec, name, act);after, evidence = registry.run(spec.identity, name, parent)
        if input_before != {n: payload(x) for n, x in inputs.items()}:
            raise ValueError('Actual typed input mutated')
        return after, {'fields': captures, 'sourceFrame': parent.to_dict(), 'outputFrame': after.to_dict(), 'evidence': evidence.to_dict()}
    try:
        torch.nn.functional.linear, native.dispatch_attention_fn = linear, attention
        with torch.inference_mode():
            values = {n: load(x, 'cuda') for n, x in s['selected'].items()}
            values.update(bankK=load(s['banks'][keys]['K'], 'cuda'), bankV=load(s['banks'][values_role]['V'], 'cuda'),
                          **{n: weights[n] for n in RETURN_PARAMETERS})
            after, source_record = run('source', values, lambda v: advance(v, v['bankK'], v['bankV'], r['support'], weights, native),
                ['A', 'U', 'M', 'T'], {'corner': keys + values_role, 'keysFrom': keys, 'valuesFrom': values_role, 'parentBankSealSHA256': s['parentBankSealSHA256']})
            inputs = dict(rows=after.values['T'], txtMod=values['txtMod'], ropeCos=values['ropeCos'], ropeSin=values['ropeSin'], **{n: weights[n] for n in NEXT_PARAMETERS})
            _, query_record = run('next_query', inputs, lambda v: next_query(v['rows'], v['txtMod'], (v['ropeCos'], v['ropeSin']), weights, native),
                ['nextQ'], {'sourceOutputFrame': after.frame_id, 'corner': keys + values_role, 'nextJoint': 2})
    finally:
        torch.nn.functional.linear, native.dispatch_attention_fn = old_linear, old_attention
    if before != {n: (v.data_ptr(), v._version, payload(v)) for n, v in weights.items()} or snapshot(torch) != numerical or not torch.equal(rng, torch.get_rng_state()) or not all(torch.equal(a, b) for a, b in zip(rng_cuda, torch.cuda.get_rng_state_all())):
        raise ValueError('Parameter/profile/input/RNG closure differs')
    assets_unchanged(assets);total = sum(x['nominalMACs'] for x in linears + attentions)
    if total != 452984832 or len(linears) != 4 or len(attentions) != 1:
        raise ValueError('Physical operation ledger differs')
    result = {'corner': keys + values_role, 'source': source_record, 'nextQuery': query_record,
              'primarySealSHA256': sha(out / 'primary-seal.json') if phase == 'diagnostic' else None,
              'parentBankSealSHA256': s['parentBankSealSHA256']}
    save(out / (phase + '.json'), result);save(out / (phase + '-seal.json'), {'requestSHA256': r['requestSHA256'], phase + 'SHA256': sha(out / (phase + '.json'))})
    save(out / (phase + '-execution.json'), {'loadedParameters': loaded, 'linears': linears, 'attentions': attentions,
        'nominalMatrixMACs': total, 'parameterInputRNGClosure': True, 'numericalProfile': numerical, 'header': head, 'assets': assets,
        'arithmeticTimings': timings, 'wallSecondsAfterImports': time.perf_counter() - started, 'CPUSecondsAfterImports': time.process_time() - cpu,
        'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'peakCUDAReservedBytes': torch.cuda.max_memory_reserved(),
        'fileQuarantine': {'allowedFiles': sorted(allowed), 'deniedCanaries': denied, 'successfulFileOpenCounts': accesses}})
    print('STALE_KEY_CONTEXT_' + phase.upper() + '_SEALED', flush=True)


if __name__ == '__main__':
    execute(sys.argv[1], sys.argv[2])
