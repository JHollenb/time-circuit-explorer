"""Two B predictions, no supplied B fused endpoint or native LayerNorm."""
from pathlib import Path
import sys
import time
from encoder_atoms import read, save, sha
from paired_fields import load, retain, payload
from normalization_innovation import IN, DOWN, make_selected, shared, standard_deviation, predict
from product_transport import install_guard, path


def execute(root):
    from flux_clock_study import verify, source_checks, assets_unchanged
    root = Path(root).resolve()
    request = read(root / 'request.json')
    verify(root, request)
    s = read(root / 'inputs.json')
    out = root / 'response-result'
    out.mkdir(exist_ok=False)
    allowed = [x['path'] for group in [s['fusedA'], s['embeddings'], s['carriers']] for x in group.values()]
    allowed += [s[k]['path'] for k in ['onePlusScale', 'modulation']]
    allowed += [str(Path(s['modelRoot']) / n) for n in s['imageAssets']]
    denied = [*s['deniedCanaries'], str(root / 'targets.json')]
    accesses = install_guard(root, allowed, s['protectedRoots'], denied)
    for name in denied:
        try:
            open(name, 'rb')
        except PermissionError:
            continue
        raise ValueError('B endpoint quarantine inactive')
    import torch
    import diffusers
    import diffusers.models.transformers.transformer_flux2 as native
    from complete_masked_study import profile
    from flux_clock import header, selected_load, snapshot
    from mdb.titan import Frame, ActSpec, ActRegistry, ProvidedFingerprint
    started = time.perf_counter()
    cpu = time.process_time()
    numerical = profile(torch, 4)
    if numerical != s['numericalProfile']:
        raise ValueError('Qualified native profile differs')
    model_root, assets = source_checks(s, diffusers)
    weights = model_root / 'transformer/diffusion_pytorch_model.safetensors'
    entries, head = header(weights)
    if len(entries) != 169 or entries[IN]['shape'] != [18432, 3072] or entries[DOWN]['shape'] != [3072, 9216]:
        raise ValueError('Selected two-matrix header differs')
    torch.cuda.reset_peak_memory_stats()
    with torch.device('meta'):
        model = make_selected(torch)
    loaded = selected_load(model, weights, sorted([IN, DOWN]), entries, 'cpu')
    if loaded != s['selectedParameters'] or sum(x['bytes'] for x in loaded) != 169869312:
        raise ValueError('Selected two-matrix payload differs')
    model.transformer_blocks[0].ff.linear_out.to('cuda')
    model.eval().requires_grad_(False)
    ff = model.transformer_blocks[0].ff
    def census():
        return {n: (v.data_ptr(), v._version, str(v.dtype), v.device.type, payload(v)) for n, v in model.named_parameters()}
    parameter_before = census()
    values = {'fusedA.' + r: load(x, 'cpu') for r, x in s['fusedA'].items()}
    values.update({'embedding.' + r: load(x, 'cuda') for r, x in s['embeddings'].items()})
    values.update({'carrier.' + h: load(x, 'cuda') for h, x in s['carriers'].items()})
    mod = load(s['modulation'], 'cpu')
    values.update(scale=load(s['onePlusScale'], 'cpu'), shift=mod[..., 9216:12288].unsqueeze(1), gate=mod[..., 15360:18432].unsqueeze(1).to('cuda'))
    before = {k: (v.data_ptr(), v._version, payload(v)) for k, v in values.items()}
    rng_cpu = torch.get_rng_state().clone()
    rng_cuda = [v.clone() for v in torch.cuda.get_rng_state_all()]
    primary = {'shared': {}, 'cells': {}, 'frames': []}
    active = primary['shared']
    phase = {'role': 'shared-innovation'}
    linears = []
    timings = []
    def capture(name, value):
        if name in active:
            raise ValueError('Repeated candidate capture')
        active[name] = retain(root / 'arrays' / phase.get('receiver', 'shared'), name, value)
    weight = ff.linear_in.weight.to(torch.float64)
    wide_version = weight._version
    old_linear = torch.nn.functional.linear
    old_attention = native.dispatch_attention_fn
    def linear(x, w, bias=None):
        name = IN if w is weight else DOWN if w is ff.linear_out.weight else None
        if name is None or bias is not None:
            raise ValueError('Unadmitted learned affine')
        linears.append({**phase, 'parameter': name, 'dtype': str(w.dtype), 'device': w.device.type, 'inputShape': list(x.shape), 'weightShape': list(w.shape), 'nominalMACs': x.numel() // x.shape[-1] * w.numel()})
        return old_linear(x, w, bias)
    def no_attention(*args, **kwargs):
        raise ValueError('Attention outside shared-innovation contract')
    native_path = native.Flux2SwiGLU()
    hook = native_path.gate_fn.register_forward_hook(lambda m, args, y: capture('activation', y))
    registry = ActRegistry()
    act = ActSpec('tce.normalization_innovation_exit', 'v1', reads=('fusedA', 'embedding', 'carrierA', 'carrierB', 'gate', 'J', 'beta'), writes=('imageExit',), write_kinds={'imageExit': 'tensor'}, write_abis={'imageExit': {'shape': [1, 1024, 3072], 'dtype': 'torch.bfloat16'}}, numerical_contract='cpu-fp64-ln-affine-secant-fp32-bf16-native-cuda-product-down')
    def operation(v):
        ua, ub = v['embedding'] + v['carrierA'], v['embedding'] + v['carrierB']
        sa, sb = standard_deviation(ua), standard_deviation(ub)
        raw, fused, alpha = predict(v['fusedA'], v['J'], v['beta'], sa, sb)
        rho = (ub.to('cpu', torch.float64) - ua.to('cpu', torch.float64)) - delta
        for name, value in [('residualA', ua), ('residualB', ub), ('sA', sa), ('sB', sb), ('alpha', alpha), ('rho', rho), ('prequantFP64', raw), ('fused', fused), ('quantizationDifference', fused.to(torch.float64) - raw)]:
            capture(name, value)
        y = path(fused.to('cuda'), v['embedding'], v['carrierB'], v['gate'], native_path, ff.linear_out, capture)
        if payload(ub) != payload(v['embedding'] + v['carrierB']):
            raise ValueError('Reconstructed B residual is unstable')
        return {'imageExit': y}, {'capturedImageExitPayloadSHA256': payload(y), 'sharedJPayloadSHA256': payload(v['J'])}
    registry.register(act, 'shared-ln-affine-innovation', operation)
    setup = time.perf_counter() - started
    try:
        torch.nn.functional.linear = linear
        native.dispatch_attention_fn = no_attention
        with torch.inference_mode():
            tick = time.perf_counter()
            delta, centered, j, beta = shared(weight, values['scale'], values['shift'], values['carrier.A'].cpu(), values['carrier.B'].cpu())
            for name, value in [('carrierDelta', delta), ('centeredDelta', centered), ('J', j), ('beta', beta), ('scale', values['scale']), ('shift', values['shift']), ('gate', values['gate'])]:
                capture(name, value)
            timings.append({'role': 'shared-innovation', 'wallSecondsInstrumented': time.perf_counter() - tick})
            if weight._version != wide_version:
                raise ValueError('FP64 matrix working copy mutated')
            for r in ['R0', 'R1']:
                phase.clear()
                phase.update(role='recipient-exit', receiver=r)
                active = {}
                v = {'fusedA': values['fusedA.' + r], 'embedding': values['embedding.' + r], 'carrierA': values['carrier.A'], 'carrierB': values['carrier.B'], 'gate': values['gate'], 'J': j, 'beta': beta}
                frame = Frame.from_values(v, kinds={k: 'tensor' for k in v}, abis={k: {'shape': list(x.shape), 'dtype': str(x.dtype), 'device': x.device.type} for k, x in v.items()}, slot_fingerprints={k: ProvidedFingerprint.content_sha256(payload(x)) for k, x in v.items()}, program_fingerprint=request['requestSHA256'], metadata={'receiver': r, 'sharedJ': primary['shared']['J'], 'clockSource': s['modulation'], 'parentJobID': s['parentJobID']})
                torch.cuda.synchronize()
                tick = time.perf_counter()
                result, evidence = registry.run(act.identity, 'shared-ln-affine-innovation', frame)
                torch.cuda.synchronize()
                timings.append({**phase, 'wallSecondsInstrumented': time.perf_counter() - tick})
                if payload(result.values['imageExit']) != active['imageExit']['payloadSHA256']:
                    raise ValueError('Typed candidate output differs')
                primary['cells'][r] = active
                primary['frames'].append({'receiver': r, 'sourceFrame': frame.to_dict(), 'outputFrame': result.to_dict(), 'evidence': evidence.to_dict()})
                print('NORMALIZATION_INNOVATION_CELL', r, flush=True)
    finally:
        torch.nn.functional.linear = old_linear
        native.dispatch_attention_fn = old_attention
        hook.remove()
    if census() != parameter_before or before != {k: (v.data_ptr(), v._version, payload(v)) for k, v in values.items()} or snapshot(torch) != numerical:
        raise ValueError('Original parameter/input/profile closure changed')
    if not torch.equal(rng_cpu, torch.get_rng_state()) or not all(torch.equal(a, b) for a, b in zip(rng_cuda, torch.cuda.get_rng_state_all())):
        raise ValueError('Candidate RNG closure changed')
    if len(linears) != 4 or sum(x['nominalMACs'] for x in linears) != 116020740096:
        raise ValueError('Four-affine physical census differs')
    assets_unchanged(assets)
    save(out / 'primary.json', primary)
    save(out / 'primary-seal.json', {'schema': 'tce-normalization-innovation-seal-v1', 'requestSHA256': request['requestSHA256'], 'primarySHA256': sha(out / 'primary.json')})
    save(out / 'execution.json', {'schema': 'tce-normalization-innovation-execution-v1', 'loadedParameters': loaded, 'header': head, 'assets': assets, 'numericalProfile': numerical, 'parameterInputRNGClosure': True, 'linears': linears, 'timings': timings, 'setupSecondsAfterImports': setup, 'workerWallSecondsAfterImports': time.perf_counter() - started, 'workerCPUSecondsAfterImports': time.process_time() - cpu, 'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'peakCUDAReservedBytes': torch.cuda.max_memory_reserved(), 'fileQuarantine': {'allowedFiles': sorted(allowed), 'successfulFileOpenCounts': accesses, 'deniedCanaries': denied, 'scope': 'Trusted-code guard; both B fused and all target/control outputs denied until complete primary seal/child exit.'}})
    print('NORMALIZATION_INNOVATION_SEALED', sha(out / 'primary-seal.json'), flush=True)


if __name__ == '__main__':
    execute(sys.argv[1])
