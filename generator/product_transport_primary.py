"""Isolated single-down candidate; recipient B targets remain unavailable."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha
from paired_fields import load, retain, payload
from product_transport import DOWN, transport, make_down, path, install_guard


def execute(root):
    from flux_clock_study import verify, source_checks, assets_unchanged
    root = Path(root).resolve()
    request = read(root / 'request.json')
    verify(root, request)
    s = read(root / 'inputs.json')
    out = root / 'response-result'
    out.mkdir(exist_ok=False)
    allowed = [x['path'] for x in s['fused'].values()] + [s[k]['path'] for k in ['embedding', 'carrier', 'modulation']]
    allowed += [str(Path(s['modelRoot']) / name) for name in s['imageAssets']]
    denied = s['deniedCanaries'] + [str(root / 'targets.json')]
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
    from complete_masked_study import profile
    from flux_clock import header, selected_load, snapshot
    from mdb.titan import Frame, ActSpec, ActRegistry, ProvidedFingerprint
    start = time.perf_counter()
    cpu = time.process_time()
    numerical = profile(torch, 4)
    if numerical != s['numericalProfile']:
        raise ValueError('Qualified numerical profile differs')
    model_root, assets = source_checks(s, diffusers)
    weights = model_root / 'transformer/diffusion_pytorch_model.safetensors'
    entries, head = header(weights)
    if len(entries) != 169 or entries[DOWN]['shape'] != [3072, 9216]:
        raise ValueError('Original down header differs')
    torch.cuda.reset_peak_memory_stats()
    with torch.device('meta'):
        model = make_down(torch)
    loaded = selected_load(model, weights, [DOWN], entries, 'cuda')
    model.eval().requires_grad_(False)
    if loaded != s['selectedParameters'] or set(dict(model.named_parameters())) != {DOWN}:
        raise ValueError('Single selected parameter differs')
    down = model.transformer_blocks[0].ff.linear_out
    before_parameter = (down.weight.data_ptr(), down.weight._version, payload(down.weight))
    values = {'fused' + k: load(v, 'cpu') for k, v in s['fused'].items()}
    values.update(embedding=load(s['embedding'], 'cuda'), carrier=load(s['carrier'], 'cuda'))
    mod = load(s['modulation'], 'cpu')
    gate = mod[..., 15360:18432].unsqueeze(1).to('cuda')
    values['gate'] = gate
    before = {k: (v.data_ptr(), v._version, payload(v)) for k, v in values.items()}
    rng_cpu = torch.get_rng_state().clone()
    rng_cuda = [v.clone() for v in torch.cuda.get_rng_state_all()]
    native_path = native.Flux2SwiGLU()
    primary = {'fields': {}, 'clockGateParent': s['modulation'], 'clockGateLastAxisSlice': [15360, 18432]}
    linears = []
    def capture(name, value):
        if name in primary['fields']:
            raise ValueError('Repeated candidate boundary')
        primary['fields'][name] = retain(root / 'arrays/candidate', name, value)
    hook = native_path.gate_fn.register_forward_hook(lambda module, args, result: capture('activation', result))
    old_linear = torch.nn.functional.linear
    old_attention = native.dispatch_attention_fn
    def linear(x, weight, bias=None):
        if weight.data_ptr() != down.weight.data_ptr() or bias is not None:
            raise ValueError('Unadmitted learned affine')
        linears.append({'parameter': DOWN, 'inputShape': list(x.shape), 'inputStride': list(x.stride()), 'weightShape': list(weight.shape), 'nominalMACs': x.numel() // x.shape[-1] * weight.numel()})
        return old_linear(x, weight, bias)
    def deny_attention(*args, **kwargs):
        raise ValueError('Attention outside candidate contract')
    act = ActSpec('tce.recipient_product_transport', 'v1', reads=tuple(values), writes=('imageExit',), write_kinds={'imageExit': 'tensor'}, write_abis={'imageExit': {'shape': [1, 1024, 3072], 'dtype': 'torch.bfloat16'}}, numerical_contract='cpu-fp64-fp32-bf16-transport-native-cuda-swiglu-one-down')
    registry = ActRegistry()
    def operation(v):
        raw, fused = transport(v['fusedR0A'], v['fusedR0B'], v['fusedR1A'])
        capture('prequantFP64', raw)
        capture('fused', fused)
        capture('quantizationDifference', fused.to(torch.float64) - raw)
        capture('clockGate', v['gate'])
        fused = fused.to('cuda')
        result = path(fused, v['embedding'], v['carrier'], v['gate'], native_path, down, capture)
        return {'imageExit': result}, {'capturedImageExitPayloadSHA256': payload(result), 'affineCalls': len(linears)}
    registry.register(act, 'recipient-baseline-native-product', operation)
    frame = Frame.from_values(values, kinds={k: 'tensor' for k in values}, abis={k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'device': v.device.type} for k, v in values.items()}, slot_fingerprints={k: ProvidedFingerprint.content_sha256(payload(v)) for k, v in values.items()}, program_fingerprint=request['requestSHA256'], metadata={'sourceJobID': s['parentJobID'], 'clockGateSlice': [15360, 18432]})
    setup = time.perf_counter() - start
    try:
        torch.nn.functional.linear = linear
        native.dispatch_attention_fn = deny_attention
        torch.cuda.synchronize()
        tick = time.perf_counter()
        with torch.inference_mode():
            result, evidence = registry.run(act.identity, 'recipient-baseline-native-product', frame)
        torch.cuda.synchronize()
        path_wall = time.perf_counter() - tick
    finally:
        torch.nn.functional.linear = old_linear
        native.dispatch_attention_fn = old_attention
        hook.remove()
    if before != {k: (v.data_ptr(), v._version, payload(v)) for k, v in values.items()} or before_parameter != (down.weight.data_ptr(), down.weight._version, payload(down.weight)) or snapshot(torch) != numerical:
        raise ValueError('Original parameter/input/profile closure changed')
    if not torch.equal(rng_cpu, torch.get_rng_state()) or not all(torch.equal(a, b) for a, b in zip(rng_cuda, torch.cuda.get_rng_state_all())):
        raise ValueError('Candidate RNG closure changed')
    if len(linears) != 1 or linears[0]['nominalMACs'] != 28991029248 or payload(result.values['imageExit']) != primary['fields']['imageExit']['payloadSHA256']:
        raise ValueError('Single-down candidate work/output closure differs')
    assets_unchanged(assets)
    primary.update(sourceFrame=frame.to_dict(), outputFrame=result.to_dict(), evidence=evidence.to_dict())
    save(out / 'primary.json', primary)
    save(out / 'primary-seal.json', {'schema': 'tce-product-transport-seal-v1', 'requestSHA256': request['requestSHA256'], 'primarySHA256': sha(out / 'primary.json')})
    save(out / 'execution.json', {'schema': 'tce-product-transport-execution-v1', 'loadedParameters': loaded, 'header': head, 'assets': assets, 'numericalProfile': numerical, 'parameterInputRNGClosure': True, 'linears': linears, 'setupSecondsAfterImports': setup, 'pathWallSecondsInstrumented': path_wall, 'workerWallSecondsAfterImports': time.perf_counter() - start, 'workerCPUSecondsAfterImports': time.process_time() - cpu, 'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'peakCUDAReservedBytes': torch.cuda.max_memory_reserved(), 'fileQuarantine': {'allowedFiles': sorted(allowed), 'successfulFileOpenCounts': accesses, 'deniedCanaries': denied, 'scope': 'Trusted-code audit hook. No target arrays or scoring metadata admitted to candidate after source verification; separate process exits before scoring.'}})
    print('PRODUCT_TRANSPORT_SEALED', sha(out / 'primary-seal.json'), flush=True)


if __name__ == '__main__':
    execute(sys.argv[1])
