"""Two actual E fields through original native images, with shared sealed noise."""
from pathlib import Path
import time
from encoder_atoms import np, read, save, sha
from paired_fields import payload, retain, load, array


def execute(root, lane):
    import torch
    import diffusers
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2 import pipeline_flux2_klein
    from complete_masked_study import verify, profile
    root = Path(root).resolve(); request = read(root / 'request.json'); verify(root, request); source = read(root / 'source.json')
    numerical = profile(torch, threads=4); tick = time.perf_counter(); cpu = time.process_time(); model = Path(source['lexicalSource']['modelRoot'])
    if lane not in ('candidate', 'full'): raise ValueError('Unknown image lane')
    encoder = read(root / ('encoder-' + lane + '.json'))
    if encoder['requestSHA256'] != request['requestSHA256']: raise ValueError('Encoder request differs')
    if lane == 'full' and not (root / 'candidate-seal.json').is_file(): raise ValueError('Candidates must seal first')
    assets = []
    for name, digest in source['nativeSources'].items():
        if sha(Path(diffusers.__file__).parent / name) != digest: raise ValueError('Native image implementation changed')
    for name, digest in source['imageAssets'].items():
        p = model / name; s = p.stat(); signature = [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
        if sha(p) != digest: raise ValueError('Original image asset changed')
        assets.append({'path': str(p), 'sha256': digest, 'signature': signature})
    transformer = Flux2Transformer2DModel.from_pretrained(str(model / 'transformer'), torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True, device_map={'': 'cuda'})
    vae = AutoencoderKLFlux2.from_pretrained(str(model / 'vae'), torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True, device_map={'': 'cuda'})
    transformer.eval().requires_grad_(False); vae.eval().requires_grad_(False)
    scheduler_config = read(model / 'scheduler/scheduler_config.json')
    if scheduler_config.get('stochastic_sampling', False): raise ValueError('Deterministic schedule required')
    pipe = Flux2KleinPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler_config), vae=vae, text_encoder=None,
        tokenizer=None, transformer=transformer, is_distilled=read(model / 'model_index.json')['is_distilled'])
    if not pipe.config.is_distilled: raise ValueError('Expected original distilled path')
    pipe.set_progress_bar_config(disable=True)
    def census():
        return {prefix + '.' + n: [v.data_ptr(), v._version, list(v.shape), str(v.dtype)] for prefix, module in [('transformer', transformer), ('vae', vae)]
            for n, v in [*module.named_parameters(), *module.named_buffers()]}
    before = census(); raw = root / 'arrays' / lane; fields = {}; actual = []; schedules = {}; timings = []; images = {}; counts = {}; phase = {}; starts = {}
    raw_noise = load(source['noise']['raw'], 'cuda'); noise = source['noise']; expected = {}; embeddings = {}
    if tuple(raw_noise.shape) != (1, 128, 32, 32): raise ValueError('Original raw noise geometry differs')
    for item in request['prompts']:
        case = item['id']; fields[case] = {}; counts[case] = {'denoiser': 0, 'context': 0, 'joint': 0, 'single': 0, 'VAE': 0}
        embeddings[case] = load(encoder['fields'][case]['E'], 'cuda')
        if tuple(embeddings[case].shape) != (1, 512, 7680): raise ValueError('Actual E geometry differs')
    old_randn = pipeline_flux2_klein.randn_tensor; old_prepare = pipe.prepare_latents; old_decode = vae.decode
    def no_draw(*args, **kwargs): raise ValueError('Noise redraw prohibited')
    pipeline_flux2_klein.randn_tensor = no_draw
    def capture(name, value):
        item = retain(raw / phase['case'], name, value); fields[phase['case']][name] = item; return item
    def prepare(*args, **kwargs):
        packed, ids = old_prepare(*args, **kwargs)
        if payload(packed) != noise['packed']['payloadSHA256'] or payload(ids) != noise['ids']['payloadSHA256']: raise ValueError('Initial noise/IDs differ')
        actual.append({**phase, 'kind': 'prepared-noise', 'payloadSHA256': payload(packed)}); return packed, ids
    pipe.prepare_latents = prepare
    def pre(module, args, kwargs):
        case = phase['case']; x = kwargs['hidden_states']; e = kwargs['encoder_hidden_states']
        if tuple(x.shape) != (1, 1024, 128) or payload(x) != expected[case] or payload(e) != encoder['fields'][case]['E']['payloadSHA256']:
            raise ValueError('Denoiser read wrong own state/E')
        actual.append({**phase, 'kind': 'denoiser-input', 'latentPayloadSHA256': payload(x), 'EPayloadSHA256': payload(e)})
        schedule = {'timesteps': array(pipe.scheduler.timesteps).tolist(), 'sigmas': array(pipe.scheduler.sigmas).tolist(),
            'timestepsPayloadSHA256': payload(pipe.scheduler.timesteps), 'sigmasPayloadSHA256': payload(pipe.scheduler.sigmas)}
        if case in schedules and schedules[case] != schedule: raise ValueError('Schedule changed during image')
        schedules[case] = schedule; counts[case]['denoiser'] += 1; torch.cuda.synchronize(); starts[case] = time.perf_counter()
    def post(module, args, value):
        torch.cuda.synchronize(); timings.append({**phase, 'kind': 'denoiser', 'wallSecondsWithContextCapture': time.perf_counter() - starts[phase['case']]})
        capture('velocity' + str(phase['step']), value[0])
    def context_pre(module, args):
        if payload(args[0]) != encoder['fields'][phase['case']]['E']['payloadSHA256']: raise ValueError('Context projection read another E')
        actual.append({**phase, 'kind': 'context-input', 'payloadSHA256': payload(args[0])}); counts[phase['case']]['context'] += 1
    def context_post(module, args, value): capture('C' + str(phase['step']), value)
    def block(kind):
        def hook(module, args): counts[phase['case']][kind] += 1
        return hook
    handles = [transformer.register_forward_pre_hook(pre, with_kwargs=True), transformer.register_forward_hook(post),
        transformer.context_embedder.register_forward_pre_hook(context_pre), transformer.context_embedder.register_forward_hook(context_post)]
    for m in transformer.transformer_blocks: handles.append(m.register_forward_pre_hook(block('joint')))
    for m in transformer.single_transformer_blocks: handles.append(m.register_forward_pre_hook(block('single')))
    def decode(z, *args, **kwargs):
        counts[phase['case']]['VAE'] += 1; capture('vaeInput', z); torch.cuda.synchronize(); started = time.perf_counter()
        value = old_decode(z, *args, **kwargs); torch.cuda.synchronize()
        timings.append({**phase, 'kind': 'VAE', 'wallSecondsExcludingCapture': time.perf_counter() - started}); capture('decodedRaw', value[0]); return value
    vae.decode = decode
    def callback(pipeline, step, timestep, data):
        item = capture('latent' + str(step + 1), data['latents']); expected[phase['case']] = item['payloadSHA256']; phase['step'] = step + 1; return {}
    setup_seconds = time.perf_counter() - tick
    try:
        with torch.inference_mode():
            for prompt in request['prompts']:
                case = prompt['id']; phase.update(case=case, step=0); expected[case] = noise['packed']['payloadSHA256']
                pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config); torch.cuda.synchronize(); started = time.perf_counter()
                result = pipe(prompt=None, prompt_embeds=embeddings[case], height=512, width=512, num_inference_steps=4, guidance_scale=1.0,
                    latents=raw_noise.clone(), output_type='np', callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents']).images
                torch.cuda.synchronize(); timings.append({**phase, 'kind': 'completeImage', 'wallSecondsWithCapture': time.perf_counter() - started})
                if result.shape != (1, 512, 512, 3) or phase['step'] != 4: raise ValueError('Image/step geometry differs')
                capture('rgbFloat', result); png = pipe.image_processor.numpy_to_pil(result)[0]
                path = root / (lane + '-' + case + '.png'); png.save(path); capture('pngRGB', np.asarray(png))
                images[case] = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
                print('NATIVE_IMAGE_COMPLETE', lane, case, counts[case], flush=True)
    finally:
        pipeline_flux2_klein.randn_tensor = old_randn; pipe.prepare_latents = old_prepare; vae.decode = old_decode
        for h in handles: h.remove()
    if census() != before or payload(raw_noise) != noise['raw']['payloadSHA256']: raise ValueError('Shared model/noise mutated')
    for case in counts:
        if counts[case] != {'denoiser': 4, 'context': 4, 'joint': 20, 'single': 80, 'VAE': 1} or schedules[case] != schedules[request['prompts'][0]['id']]: raise ValueError('Native image work/schedule differs')
    for item in assets:
        s = Path(item['path']).stat()
        if [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns] != item['signature']: raise ValueError('Image asset mutated')
    save(root / ('images-' + lane + '.json'), {'lane': lane, 'requestSHA256': request['requestSHA256'], 'encoderSHA256': sha(root / ('encoder-' + lane + '.json')),
        'fields': fields, 'images': images, 'actual': actual, 'counts': counts, 'schedules': schedules, 'timings': timings,
        'setupSecondsAfterImports': setup_seconds, 'workerSecondsAfterImports': time.perf_counter() - tick, 'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(),
        'assetChecks': assets, 'parameterAndBufferClosureUnchanged': True, 'newNoiseDraws': 0, 'torchVersion': torch.__version__, 'diffusersVersion': diffusers.__version__,
        'numericalProfile': numerical, 'cpuSecondsAfterImports': time.process_time() - cpu,
        'attentionProcessors': sorted({type(m.processor).__name__ for m in transformer.modules() if hasattr(m, 'processor')})})
