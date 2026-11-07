"""Fresh Q/K/V, factored value event and sealed own local B candidate."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha
from paired_fields import load, retain, payload
from value_innovation import acquire, event, compile_event, IN, DOWN, WO


def weight_record(name, value, entries):
    return {'name': name, 'shape': list(value.shape), 'bytes': value.numel() * value.element_size(), 'payloadSHA256': payload(value), 'dataOffsets': entries[name]['data_offsets']}


def execute(root):
    from flux_clock_study import verify, source_checks, assets_unchanged
    from product_transport import install_guard, path
    root = Path(root).resolve()
    request, s = read(root / 'request.json'), read(root / 'inputs.json')
    verify(root, request)
    out = root / 'response-result'
    out.mkdir(exist_ok=False)
    allowed = [s[n]['path'] for n in ['z', 'imgIDs', 'modImage', 'modText', 'carrierA', 'fusedA', 'onePlusScale', 'beta']]
    allowed += [n for key in ['conditioning', 'txtIDs'] for n in s[key]['files']]
    allowed += [str(Path(s['modelRoot']) / n) for n in s['imageAssets']]
    denied = [str(root / 'targets.json'), *s['deniedCanaries']]
    accesses = install_guard(root, allowed, s['protectedRoots'], denied)
    for name in denied:
        try:
            open(name, 'rb')
        except PermissionError:
            continue
        raise ValueError('Primary target quarantine inactive')
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from source_native_reader import load as load_reader
    from native_reader_corpus import load as historical
    from complete_masked_study import profile
    from flux_clock import header, selected_load, snapshot
    from normalization_innovation import make_selected, standard_deviation, predict
    from mdb.titan import Frame, ActSpec, ActRegistry, ProvidedFingerprint
    from safetensors import safe_open
    started, cpu = time.perf_counter(), time.process_time()
    numerical = profile(torch, 4)
    if numerical != s['numericalProfile']:
        raise ValueError('Qualified profile differs')
    model_root, assets = source_checks(s, diffusers)
    weights_file = model_root / 'transformer/diffusion_pytorch_model.safetensors'
    config = read(model_root / 'transformer/config.json')
    weights, norms, pos, loaded_prefix = load_reader(weights_file, config)
    entries, head = header(weights_file)
    with torch.device('meta'):
        model = make_selected(torch)
    loaded_mlp = selected_load(model, weights_file, sorted([IN, DOWN]), entries, 'cpu')
    model.transformer_blocks[0].ff.linear_out.to('cuda')
    model.eval().requires_grad_(False)
    with safe_open(str(weights_file), framework='pt', device='cpu') as stream:
        wo = stream.get_tensor(WO)
    loaded = loaded_prefix + loaded_mlp + [weight_record(WO, wo, entries)]
    expected = {x['name']: x for x in s['selectedParameters']}
    if {x['name']: x for x in loaded} != expected or sum(x['bytes'] for x in loaded) != 331088640:
        raise ValueError('Distinct selected parameter admission differs')
    ff = model.transformer_blocks[0].ff
    all_parameters = {**weights, IN: ff.linear_in.weight, DOWN: ff.linear_out.weight, WO: wo}
    def census():
        return {n: (x.data_ptr(), x._version, str(x.dtype), x.device.type, payload(x)) for n, x in all_parameters.items()}
    parameter_before = census()
    values = {'z': historical(s['z'], 'cuda'), 'conditioning': historical(s['conditioning'], 'cuda'), 'imgIDs': load(s['imgIDs'], 'cuda'), 'txtIDs': historical(s['txtIDs'], 'cuda'),
              'modImage': load(s['modImage'], 'cuda'), 'modText': load(s['modText'], 'cuda'), 'carrierA': load(s['carrierA'], 'cuda'),
              'fusedA': load(s['fusedA'], 'cpu'), 'onePlusScale': load(s['onePlusScale'], 'cpu'), 'beta': load(s['beta'], 'cpu')}
    input_before = {n: (x.data_ptr(), x._version, payload(x)) for n, x in values.items()}
    win64, wo64 = ff.linear_in.weight.to(torch.float64), wo.to(torch.float64)
    wide_before = [(x.data_ptr(), x._version) for x in [win64, wo64]]
    primary = {'acquisition': {}, 'fields': {}, 'frames': []}
    active = primary['acquisition']
    phase = 'acquisition'
    linears, contractions = [], []
    names = {x.data_ptr(): n for n, x in weights.items()}
    old_linear, old_attention = torch.nn.functional.linear, native.dispatch_attention_fn
    def linear(x, w, bias=None):
        name = names.get(w.data_ptr())
        extra = {}
        if w is win64:
            name = IN
        elif w is ff.linear_out.weight:
            name = DOWN
        elif w.untyped_storage().data_ptr() == wo64.untyped_storage().data_ptr():
            name = WO
            extra = {'headIndex': w.storage_offset() // 128, 'columns': [w.storage_offset(), w.storage_offset() + 128], 'physicalHeadBlock': True}
        if name is None or bias is not None:
            raise ValueError('Unadmitted candidate learned affine')
        linears.append({'phase': phase, 'parameter': name, 'inputShape': list(x.shape), 'inputStride': list(x.stride()), 'weightShape': list(w.shape), 'dtype': str(w.dtype), 'device': w.device.type,
                        'nominalMACs': x.numel() // x.shape[-1] * w.numel(), **extra})
        return old_linear(x, w, bias)
    def no_attention(*args, **kwargs):
        raise ValueError('Primary attention evaluation is outside contract')
    def capture(name, value):
        if name in active:
            raise ValueError('Repeated primary field')
        active[name] = {**retain(root / 'arrays' / phase, name, value), 'nativeStride': list(value.stride()), 'nativeStorageOffset': value.storage_offset()}
    nonlinear = native.Flux2SwiGLU()
    hook = nonlinear.gate_fn.register_forward_hook(lambda m, args, y: capture('activation', y))
    registry = ActRegistry()
    spec = ActSpec('tce.addressed_value_innovation', 'v1', reads=('Q', 'K', 'V', 'embedding', 'carrierA', 'fusedA', 'beta', 'modulation', 'onePlusScale'), writes=('imageExit',),
                   write_kinds={'imageExit': 'tensor'}, write_abis={'imageExit': {'shape': [1, 1024, 3072], 'dtype': 'torch.bfloat16'}},
                   numerical_contract='fresh-native-qkv-full-key-fp64-value-swap-factors-own-bf16-state-native-down')
    def operation(v):
        gamma, directions, selected = event(*[v[n].to('cpu', torch.float64) for n in ['Q', 'K', 'V']], *request['event']['keyIndices'])
        contractions.append({'kind': 'full-key-QK', 'batch': 24, 'inputShapes': [[24, 1024, 128], [24, 128, 1536]], 'dtype': 'torch.float64', 'device': 'cpu', 'nominalMACs': 4831838208, 'softmaxScalars': 37748736})
        parts = native.Flux2Modulation.split(v['modulation'], 2)
        b, t, delta, j = compile_event(wo64, win64, parts[0][2].to('cpu', torch.float64), v['onePlusScale'].to(torch.float64), gamma, directions)
        contractions.extend([{'kind': 'assemble-carrier', 'inputShapes': [[1024, 24], [24, 3072]], 'nominalMACs': 75497472},
                             {'kind': 'assemble-J', 'inputShapes': [[1024, 24], [24, 18432]], 'nominalMACs': 452984832}])
        raw_carrier = v['carrierA'].to('cpu', torch.float64) + delta
        carrier = raw_carrier.to(torch.float32).to(torch.bfloat16).to('cuda')
        ua, ub = v['embedding'] + v['carrierA'], v['embedding'] + carrier
        sa, sb = standard_deviation(ua), standard_deviation(ub)
        raw, fused, alpha = predict(v['fusedA'], j, v['beta'], sa, sb)
        fields = [('gamma', gamma), ('valueDirections', directions), ('selectedProbabilities', selected), ('carrierDirections', b), ('fusedDirections', t), ('carrierInnovation', delta), ('J', j),
                  ('rawCarrierFP64', raw_carrier), ('gatedCarrier', carrier), ('residualA', ua), ('residualB', ub), ('rho', ub.to('cpu', torch.float64) - ua.to('cpu', torch.float64) - delta),
                  ('sA', sa), ('sB', sb), ('alpha', alpha), ('prequantFP64', raw), ('fused', fused), ('quantizationDifference', fused.to(torch.float64) - raw)]
        for name, value in fields:
            capture(name, value)
        y = path(fused.to('cuda'), v['embedding'], carrier, parts[1][2], nonlinear, ff.linear_out, capture)
        return {'imageExit': y}, {'capturedImageExitPayloadSHA256': payload(y), 'rankByConstruction': 24, 'fullKeyCount': 1536, 'nativeAttentionCalls': 0}
    registry.register(spec, 'finite-fixed-QK-value-event', operation)
    rng_cpu, rng_cuda = torch.get_rng_state().clone(), [x.clone() for x in torch.cuda.get_rng_state_all()]
    torch.cuda.reset_peak_memory_stats()
    setup = time.perf_counter() - started
    try:
        torch.nn.functional.linear, native.dispatch_attention_fn = linear, no_attention
        with torch.inference_mode():
            image, q, k, v = acquire(weights, values['z'], values['conditioning'], values['imgIDs'], values['txtIDs'], values['modImage'], values['modText'], norms, pos, native, 24)
            if payload(image) != s['expectedEmbeddingPayloadSHA256']:
                raise ValueError('Fresh acquisition embedding is a different specimen')
            for name, x in [('embedding', image), ('Q', q), ('K', k), ('V', v)]:
                capture(name, x)
            primary['eventImageIDs'] = values['imgIDs'][0, request['event']['imageRows']].cpu().tolist()
            phase, active = 'candidate', primary['fields']
            frame_values = {'Q': q, 'K': k, 'V': v, 'embedding': image, 'carrierA': values['carrierA'], 'fusedA': values['fusedA'], 'beta': values['beta'], 'modulation': values['modImage'], 'onePlusScale': values['onePlusScale']}
            frame = Frame.from_values(frame_values, kinds={n: 'tensor' for n in frame_values}, abis={n: {'shape': list(x.shape), 'dtype': str(x.dtype), 'stride': list(x.stride())} for n, x in frame_values.items()},
                                      slot_fingerprints={n: ProvidedFingerprint.content_sha256(payload(x)) for n, x in frame_values.items()}, program_fingerprint=request['requestSHA256'], metadata={'event': request['event'], 'selectedParameters': loaded})
            result, evidence = registry.run(spec.identity, 'finite-fixed-QK-value-event', frame)
            if payload(result.values['imageExit']) != active['imageExit']['payloadSHA256']:
                raise ValueError('Typed output differs from captured own state')
            primary['frames'].append({'sourceFrame': frame.to_dict(), 'outputFrame': result.to_dict(), 'evidence': evidence.to_dict()})
    finally:
        torch.nn.functional.linear, native.dispatch_attention_fn = old_linear, old_attention
        hook.remove()
    torch.cuda.synchronize()
    if census() != parameter_before or input_before != {n: (x.data_ptr(), x._version, payload(x)) for n, x in values.items()} or wide_before != [(x.data_ptr(), x._version) for x in [win64, wo64]] or snapshot(torch) != numerical:
        raise ValueError('Candidate parameter/input/profile closure differs')
    if not torch.equal(rng_cpu, torch.get_rng_state()) or not all(torch.equal(a, b) for a, b in zip(rng_cuda, torch.cuda.get_rng_state_all())):
        raise ValueError('Candidate RNG changed')
    total = sum(x['nominalMACs'] for x in linears + contractions)
    if len(linears) != 33 or total != 86856695808 or len(contractions) != 3:
        raise ValueError('Physical candidate work differs')
    assets_unchanged(assets)
    save(out / 'primary.json', primary)
    save(out / 'primary-seal.json', {'schema': 'tce-value-innovation-seal-v1', 'requestSHA256': request['requestSHA256'], 'primarySHA256': sha(out / 'primary.json')})
    save(out / 'execution.json', {'schema': 'tce-value-innovation-execution-v1', 'loadedParameters': loaded, 'linears': linears, 'contractions': contractions, 'nominalCandidateMACs': total,
                                 'parameterInputRNGClosure': True, 'numericalProfile': numerical, 'assets': assets, 'header': head, 'setupSecondsAfterImports': setup,
                                 'workerWallSecondsAfterImports': time.perf_counter() - started, 'workerCPUSecondsAfterImports': time.process_time() - cpu,
                                 'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'peakCUDAReservedBytes': torch.cuda.max_memory_reserved(),
                                 'fileQuarantine': {'allowedFiles': sorted(allowed), 'deniedCanaries': denied, 'successfulFileOpenCounts': accesses, 'scope': 'Trusted-code guard, no new native B fields or old downstream output controls before seal/child exit.'}})
    print('VALUE_INNOVATION_PRIMARY_SEALED', sha(out / 'primary-seal.json'), flush=True)


if __name__ == '__main__':
    execute(sys.argv[1])
