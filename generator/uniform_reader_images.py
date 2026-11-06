"""Own uniform-reader E through original live images and inherited exact clocks."""
from pathlib import Path
import time
from encoder_atoms import np, read, save, sha
from paired_fields import payload, retain, load, array
from flux_clock import header, partition, selected_load, install_tables, check_mode, snapshot, deny_targets, CLOCK_PREFIXES


def execute(root):
    import torch
    import diffusers
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2 import pipeline_flux2_klein
    from complete_masked_study import profile
    from flux_clock_study import source_checks, assets_unchanged
    from uniform_reader_study import verify, inherited_clock
    root = Path(root).resolve(); request = read(root / 'request.json'); verify(root, request); source = read(root / 'inputs.json')
    deny_targets([root / 'targets.json', *source['forbiddenTargets']]); numerical = profile(torch, 4)
    tick = time.perf_counter(); cpu = time.process_time(); model, assets = source_checks(source, diffusers)
    table_path = root / 'clock-table.json'; table = inherited_clock(root, source); table_sha = sha(table_path)
    encoder = read(root / 'encoder-candidate.json')
    if encoder['requestSHA256'] != request['requestSHA256']: raise ValueError('Candidate E belongs to another request')
    e_fields = {case: encoder['fields'][case]['E'] for case in ['A', 'B']}
    if table['contract']['numericalProfile'] != numerical or numerical != source['numericalProfile'] or table['contract']['schedule'] != source['schedule']:
        raise ValueError('Compiled table request/profile/schedule differs')
    if table['contract']['checkpointSHA256'] != source['imageAssets']['transformer/diffusion_pytorch_model.safetensors']:
        raise ValueError('Compiled checkpoint binding differs')
    config = read(model / 'transformer/config.json')
    if config.get('guidance_embeds') is not False: raise ValueError('Unexpected guidance model')
    entries, h = header(model / 'transformer/diffusion_pytorch_model.safetensors'); removed, kept = partition(entries)
    if h != table['header'] or removed != table['removedParameterNames'] or kept != table['remainingParameterNames']: raise ValueError('Compiled parameter partition differs')
    with torch.device('meta'): transformer = Flux2Transformer2DModel.from_config(config)
    phase = {}; clock_calls = []
    install_tables(transformer, table, 'cuda', phase, clock_calls)
    reads = selected_load(transformer, model / 'transformer/diffusion_pytorch_model.safetensors', kept, entries, 'cuda')
    if set(removed) & {x['name'] for x in reads} or any(n.startswith(CLOCK_PREFIXES) for n, _ in transformer.named_parameters()):
        raise ValueError('Eliminated learned tensors loaded in fresh serving')
    if transformer.is_cache_enabled or getattr(transformer, '_hf_hook', None) is not None: raise ValueError('Unexpected model caching/offload hooks')
    def processors(): return sorted({type(m.processor).__name__ for m in transformer.modules() if hasattr(m, 'processor')})
    if processors() != source['attentionProcessors']: raise ValueError('Selected constructor attention processors differ from reference')
    vae = AutoencoderKLFlux2.from_pretrained(str(model / 'vae'), torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True, device_map={'': 'cuda'})
    transformer.eval().requires_grad_(False); vae.eval().requires_grad_(False)
    scheduler_config = read(model / 'scheduler/scheduler_config.json')
    if scheduler_config.get('stochastic_sampling', False): raise ValueError('Deterministic schedule required')
    pipe = Flux2KleinPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler_config), vae=vae, text_encoder=None,
        tokenizer=None, transformer=transformer, is_distilled=read(model / 'model_index.json')['is_distilled'])
    if not pipe.config.is_distilled: raise ValueError('Original distilled path required')
    pipe.set_progress_bar_config(disable=True)
    def census():
        return {prefix + '.' + n: [v.data_ptr(), v._version, list(v.shape), str(v.dtype)] for prefix, module in [('transformer', transformer), ('vae', vae)]
            for n, v in [*module.named_parameters(), *module.named_buffers()]}
    before = census(); constant_hashes = {n: payload(v) for n, v in transformer.named_buffers()}
    fields = {}; actual = []; schedules = {}; timings = []; images = {}; counts = {}; expected = {}; starts = {}; linears = []
    raw = root / 'arrays/candidate'; raw_noise = load(source['noise']['raw'], 'cuda'); noise = source['noise']
    embeddings = {case: load(item, 'cuda') for case, item in e_fields.items()}
    old_randn = pipeline_flux2_klein.randn_tensor; old_prepare = pipe.prepare_latents; old_decode = vae.decode; old_linear = torch.nn.functional.linear
    weight_names = {v.data_ptr(): prefix + '.' + n for prefix, module in [('transformer', transformer), ('vae', vae)] for n, v in module.named_parameters()}
    def no_draw(*args, **kwargs): raise ValueError('Noise redraw prohibited')
    pipeline_flux2_klein.randn_tensor = no_draw
    def capture(name, value):
        item = retain(raw / phase['case'], name, value); fields[phase['case']][name] = item; return item
    def prepare(*args, **kwargs):
        packed, ids = old_prepare(*args, **kwargs)
        if payload(packed) != noise['packed']['payloadSHA256'] or payload(ids) != noise['ids']['payloadSHA256']: raise ValueError('Inherited noise/IDs differ')
        actual.append({**phase, 'kind': 'prepared-noise', 'payloadSHA256': payload(packed)}); return packed, ids
    pipe.prepare_latents = prepare
    def pre(module, args, kwargs):
        check_mode(kwargs, snapshot(torch), table['contract'])
        if module.is_cache_enabled or getattr(module, '_hf_hook', None) is not None: raise ValueError('Unexpected model cache/offload')
        case = phase['case']; x = kwargs['hidden_states']; e = kwargs['encoder_hidden_states']; step = phase['step']
        if payload(x) != expected[case] or payload(e) != e_fields[case]['payloadSHA256']: raise ValueError('Wrong own latent/E')
        clock = table['rows'][step]['clockOperands']['transformer_argument']
        if payload(kwargs['timestep']) != clock['payloadSHA256'] or list(kwargs['timestep'].shape) != clock['shape'] or str(kwargs['timestep'].dtype) != clock['logicalDtype']:
            raise ValueError('Actual pipeline clock differs')
        schedule = {'timesteps': array(pipe.scheduler.timesteps).tolist(), 'sigmas': array(pipe.scheduler.sigmas).tolist(),
            'timestepsPayloadSHA256': payload(pipe.scheduler.timesteps), 'sigmasPayloadSHA256': payload(pipe.scheduler.sigmas)}
        if schedule != source['schedule']: raise ValueError('Actual scheduler differs')
        schedules[case] = schedule; counts[case]['denoiser'] += 1
        actual.append({**phase, 'kind': 'denoiser-input', 'latentPayloadSHA256': payload(x), 'EPayloadSHA256': payload(e), 'clockArgumentPayloadSHA256': payload(kwargs['timestep'])})
        torch.cuda.synchronize(); starts[case] = time.perf_counter()
    def post(module, args, value):
        torch.cuda.synchronize(); timings.append({**phase, 'kind': 'denoiser', 'wallSecondsWithCapture': time.perf_counter() - starts[phase['case']]})
        capture('velocity' + str(phase['step']), value[0])
    def context_pre(module, args):
        if payload(args[0]) != e_fields[phase['case']]['payloadSHA256']: raise ValueError('Wrong actual context operand')
        counts[phase['case']]['context'] += 1
    def context_post(module, args, value): capture('C' + str(phase['step']), value)
    def block(kind):
        def hook(module, args): counts[phase['case']][kind] += 1
        return hook
    def linear(x, weight, bias=None):
        name = weight_names.get(weight.data_ptr())
        if name is None or any(name == 'transformer.' + n for n in removed): raise ValueError('Unexpected serving affine tensor')
        linears.append({**phase, 'parameter': name, 'inputShape': list(x.shape), 'weightShape': list(weight.shape),
            'nominalMACs': x.numel() // x.shape[-1] * weight.numel()})
        return old_linear(x, weight, bias)
    handles = [transformer.register_forward_pre_hook(pre, with_kwargs=True), transformer.register_forward_hook(post),
        transformer.context_embedder.register_forward_pre_hook(context_pre), transformer.context_embedder.register_forward_hook(context_post)]
    for m in transformer.transformer_blocks: handles.append(m.register_forward_pre_hook(block('joint')))
    for m in transformer.single_transformer_blocks: handles.append(m.register_forward_pre_hook(block('single')))
    def decode(z, *args, **kwargs):
        counts[phase['case']]['VAE'] += 1; capture('vaeInput', z); torch.cuda.synchronize(); start = time.perf_counter()
        value = old_decode(z, *args, **kwargs); torch.cuda.synchronize()
        timings.append({**phase, 'kind': 'VAE', 'wallSecondsExcludingCapture': time.perf_counter() - start}); capture('decodedRaw', value[0]); return value
    vae.decode = decode
    def callback(pipeline, step, timestep, data):
        item = capture('latent' + str(step + 1), data['latents']); expected[phase['case']] = item['payloadSHA256']; phase['step'] = step + 1; return {}
    setup_seconds = time.perf_counter() - tick; torch.nn.functional.linear = linear
    try:
        with torch.inference_mode():
            for case in e_fields:
                phase.update(case=case, step=0); fields[case] = {}; expected[case] = noise['packed']['payloadSHA256']
                counts[case] = {'denoiser': 0, 'context': 0, 'joint': 0, 'single': 0, 'VAE': 0}
                pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config); torch.cuda.synchronize(); start = time.perf_counter()
                result = pipe(prompt=None, prompt_embeds=embeddings[case], height=512, width=512, num_inference_steps=4, guidance_scale=1.0,
                    latents=raw_noise.clone(), output_type='np', callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents']).images
                torch.cuda.synchronize(); timings.append({**phase, 'kind': 'completeImage', 'wallSecondsWithCapture': time.perf_counter() - start})
                if result.shape != (1, 512, 512, 3) or phase['step'] != 4: raise ValueError('Image geometry/steps differ')
                capture('rgbFloat', result); png = pipe.image_processor.numpy_to_pil(result)[0]; path = root / ('candidate-' + case + '.png')
                png.save(path); capture('pngRGB', np.asarray(png)); images[case] = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
                print('UNIFORM_READER_IMAGE_COMPLETE', case, counts[case], flush=True)
    finally:
        pipeline_flux2_klein.randn_tensor = old_randn; pipe.prepare_latents = old_prepare; vae.decode = old_decode; torch.nn.functional.linear = old_linear
        for hook in handles: hook.remove()
    if census() != before or constant_hashes != {n: payload(v) for n, v in transformer.named_buffers()} or payload(raw_noise) != noise['raw']['payloadSHA256']:
        raise ValueError('Serving parameters/constants/noise mutated')
    if sha(table_path) != table_sha or len(clock_calls) != 40: raise ValueError('Table identity or gather instrument differs')
    if processors() != source['attentionProcessors']: raise ValueError('Native attention processors changed')
    for case in counts:
        if counts[case] != {'denoiser': 4, 'context': 4, 'joint': 20, 'single': 80, 'VAE': 1}: raise ValueError('Native work instrument differs')
    assets_unchanged(assets)
    save(root / 'images-candidate.json', {'schema': 'tce-uniform-reader-images-v1', 'requestSHA256': request['requestSHA256'], 'tableSHA256': table_sha,
        'encoderSHA256': sha(root / 'encoder-candidate.json'), 'actualEFields': e_fields,
        'fields': fields, 'images': images, 'actual': actual, 'counts': counts, 'schedules': schedules, 'timings': timings, 'clockCalls': clock_calls,
        'loadedParameters': reads, 'removedParameters': removed, 'remainingParameterBytesLoaded': sum(x['bytes'] for x in reads),
        'linears': linears, 'setupSecondsAfterImports': setup_seconds, 'workerSecondsAfterImports': time.perf_counter() - tick,
        'cpuSecondsAfterImports': time.process_time() - cpu, 'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'fullFileIdentityChecks': assets,
        'parameterAndBufferClosureUnchanged': True, 'constantBufferPayloads': constant_hashes, 'numericalProfile': numerical,
        'newNoiseDraws': 0, 'newClockCompilations': 0, 'inheritedClockRequestSHA256': table['requestSHA256'], 'torchVersion': torch.__version__, 'diffusersVersion': diffusers.__version__,
        'attentionProcessors': sorted({type(m.processor).__name__ for m in transformer.modules() if hasattr(m, 'processor')})})
