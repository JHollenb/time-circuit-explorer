"""Original learned clock compilation, isolated from fresh serving."""
from pathlib import Path
import time
from encoder_atoms import np, read, save, sha
from paired_fields import payload, retain
from flux_clock import header, partition, selected_load, clock_path, deny_targets, CLOCK_WEIGHTS, PORTS


def execute(root):
    import torch
    import diffusers
    from diffusers import Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
    from complete_masked_study import profile
    from flux_clock_study import verify, source_checks, assets_unchanged
    root = Path(root).resolve(); request = read(root / 'request.json'); verify(root, request); source = read(root / 'inputs.json')
    deny_targets([root / 'targets.json', *source['forbiddenTargets']]); numerical = profile(torch, 4)
    if numerical != source['numericalProfile']: raise ValueError('Compiler profile differs from sealed reference')
    started = time.perf_counter(); cpu = time.process_time(); modelroot, assets = source_checks(source, diffusers)
    config = read(modelroot / 'transformer/config.json')
    if config.get('guidance_embeds') is not False: raise ValueError('Unexpected guidance model')
    path = modelroot / 'transformer/diffusion_pytorch_model.safetensors'; entries, h = header(path); removed, kept = partition(entries)
    with torch.device('meta'): container = Flux2Transformer2DModel.from_config(config)
    # Retain only original clock modules, then load only the six eliminated tensors.
    modules = torch.nn.Module()
    for name in ['time_guidance_embed', 'double_stream_modulation_img', 'double_stream_modulation_txt', 'single_stream_modulation']:
        modules.add_module(name, getattr(container, name))
    out = torch.nn.Module(); out.add_module('linear', container.norm_out.linear); modules.add_module('norm_out', out)
    del container
    reads = selected_load(modules, path, removed, entries, 'cuda'); modules.eval().requires_grad_(False)
    versions = [(n, v.data_ptr(), v._version) for n, v in modules.named_parameters()]
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(read(modelroot / 'scheduler/scheduler_config.json'))
    sigmas = None if getattr(scheduler.config, 'use_flow_sigmas', False) else np.linspace(1.0, .25, 4)
    ts, count = retrieve_timesteps(scheduler, 4, 'cuda', sigmas=sigmas, mu=compute_empirical_mu(image_seq_len=1024, num_steps=4))
    expected = source['schedule']
    if count != 4 or payload(ts) != expected['timestepsPayloadSHA256'] or payload(scheduler.sigmas) != expected['sigmasPayloadSHA256']:
        raise ValueError('Original deployed schedule differs')
    rows = []; linears = []; ptrs = {v.data_ptr(): n for n, v in modules.named_parameters()}; old = torch.nn.functional.linear
    def linear(x, weight, bias=None):
        if weight.data_ptr() not in ptrs or bias is not None: raise ValueError('Unexpected offline clock affine')
        linears.append({'clock': len(rows), 'parameter': ptrs[weight.data_ptr()], 'inputShape': list(x.shape), 'weightShape': list(weight.shape),
            'nominalMACs': x.numel() // x.shape[-1] * weight.numel()})
        return old(x, weight, bias)
    torch.nn.functional.linear = linear; torch.cuda.synchronize(); compile_tick = time.perf_counter()
    try:
        with torch.inference_mode():
            for i, t in enumerate(ts):
                clocks = clock_path(t, torch); z = modules.time_guidance_embed(clocks['internal'], None)
                fields = {'temb': z, 'double_img': modules.double_stream_modulation_img(z), 'double_txt': modules.double_stream_modulation_txt(z),
                    'single': modules.single_stream_modulation(z), 'output_affine': modules.norm_out.linear(torch.nn.functional.silu(z).to(torch.bfloat16))}
                rows.append({'clock': i, 'clockOperands': {k: retain(root / 'clock-arrays' / str(i), k, v) for k, v in clocks.items()},
                    'fields': {k: retain(root / 'clock-arrays' / str(i), k, v) for k, v in fields.items()}})
    finally: torch.nn.functional.linear = old
    torch.cuda.synchronize(); arithmetic_capture_seconds = time.perf_counter() - compile_tick
    if len(linears) != 24 or {r['parameter'] for r in linears} != set(CLOCK_WEIGHTS): raise ValueError('Clock compiler affine count differs')
    if versions != [(n, v.data_ptr(), v._version) for n, v in modules.named_parameters()]: raise ValueError('Compiler parameters mutated')
    assets_unchanged(assets)
    table = {'schema': 'tce-flux-clock-table-v1', 'requestSHA256': request['requestSHA256'], 'rows': rows,
        'contract': {'checkpointSHA256': source['imageAssets']['transformer/diffusion_pytorch_model.safetensors'], 'numericalProfile': numerical,
            'dtype': 'torch.bfloat16', 'latentShape': [1, 1024, 128], 'EShape': [1, 512, 7680], 'schedule': expected,
            'guidance': None, 'referenceTokens': 0, 'KVMode': None},
        'removedParameterNames': removed, 'removedParameterBytes': sum(entries[n]['data_offsets'][1] - entries[n]['data_offsets'][0] for n in removed),
        'remainingParameterNames': kept, 'remainingParameterBytes': sum(entries[n]['data_offsets'][1] - entries[n]['data_offsets'][0] for n in kept),
        'tableOutputPayloadBytes': sum(np.prod(r['fields'][p]['shape']).item() * 2 for r in rows for p in PORTS),
        'header': h, 'offlineParameterReads': reads, 'offlineLinears': linears, 'offlineNominalLinearMACs': sum(x['nominalMACs'] for x in linears),
        'compileWallSecondsWithCapture': arithmetic_capture_seconds, 'workerSecondsAfterImports': time.perf_counter() - started,
        'cpuSecondsAfterImports': time.process_time() - cpu, 'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(),
        'fullFileIdentityChecks': assets, 'scope': 'Paid original learned compilation of finite clock operands, not raw-prompt/image generation.'}
    save(root / 'clock-table.json', table); print('CLOCK_TABLE_SEALED', sha(root / 'clock-table.json'), flush=True)
