"""Only the finite native scheduler and original VAE consume each forecast."""
from pathlib import Path
import time
from encoder_atoms import read, save, sha
from paired_fields import load, payload, retain


def scheduler(source, torch, Scheduler):
    model = Path(source['modelRoot'])
    cfg = read(model / 'scheduler/scheduler_config.json')
    if cfg.get('stochastic_sampling', False):
        raise ValueError('Deterministic scheduler required')
    result = Scheduler.from_config(cfg)
    result.sigmas = torch.tensor(source['schedule']['sigmas'], dtype=torch.float32, device='cuda')
    result.timesteps = torch.tensor(source['schedule']['timesteps'], dtype=torch.float32, device='cuda')
    if payload(result.sigmas) != source['schedule']['sigmasPayloadSHA256'] or payload(result.timesteps) != source['schedule']['timestepsPayloadSHA256']:
        raise ValueError('Finite schedule differs')
    result.set_begin_index(3)
    return result


def checked_assets(source, diffusers):
    import torch
    if {'torch': torch.__version__, 'diffusers': diffusers.__version__} != source['versions']:
        raise ValueError('Native versions differ')
    for n, h in source['nativeSources'].items():
        if sha(Path(diffusers.__file__).parent / n) != h:
            raise ValueError('Native source differs: '+n)
    assets = []
    for n, h in source['decoderAssets'].items():
        p = Path(source['modelRoot']) / n; st = p.stat()
        if sha(p) != h:
            raise ValueError('Native decoding asset differs')
        assets.append({'path': str(p), 'sha256': h, 'bytesHashed': st.st_size,
            'signature': [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]})
    return assets


def execute(root):
    from last_velocity import install_guard
    from flux_clock_study import verify, assets_unchanged
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root, r)
    s = read(root / 'inputs.json'); f = read(root / 'forecast-seal.json'); fsha = sha(root / 'forecast-seal.json')
    if f['requestSHA256'] != r['requestSHA256'] or f['prefix'] != s['prefix']:
        raise ValueError('Forecast parent differs')
    install_guard(root)
    import torch
    import diffusers
    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler
    from complete_masked_study import profile
    tick = time.perf_counter(); cpu = time.process_time()
    numerical = profile(torch, 4)
    if numerical != s['numericalProfile']:
        raise ValueError('Native numerical profile differs')
    assets = checked_assets(s, diffusers)
    vae = AutoencoderKLFlux2.from_pretrained(str(Path(s['modelRoot']) / 'vae'), torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True, device_map={'': 'cuda'})
    vae.eval().requires_grad_(False)
    pipe = Flux2KleinPipeline(scheduler=None, vae=vae, transformer=None, text_encoder=None, tokenizer=None, is_distilled=True)
    if pipe.transformer is not None or pipe.text_encoder is not None:
        raise ValueError('A forbidden model was constructed')
    ids = load(s['ids'], 'cuda'); fields = {}; images = {}; actual = []; timings = []; ops = []; phase = {}; hooks = []
    def census():
        return {n: [p.data_ptr(), p._version, list(p.shape), str(p.dtype)] for n,p in [*vae.named_parameters(),*vae.named_buffers()]}
    before = census()
    def op(name):
        def hook(module, args, out):
            y = out
            ops.append({**phase, 'module': name, 'kind': type(module).__name__, 'inputShape': list(args[0].shape),
                'outputShape': list(y.shape), 'nominalMACs': y.numel() * (module.weight.numel() // module.weight.shape[0])})
        return hook
    for name, module in vae.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(op(name)))
    setup = time.perf_counter()-tick
    try:
        with torch.inference_mode():
            for key, fs in f['fields'].items():
                policy, case = key.split('-'); phase.update(policy=policy, case=case)
                z3 = load(s['prefix'][case]['latent3'], 'cuda'); v = load(fs['velocity3'], 'cuda')
                if z3.dtype != torch.bfloat16 or v.dtype != torch.bfloat16 or list(z3.shape) != list(v.shape):
                    raise ValueError('Finite commit operands differ')
                sch = scheduler(s, torch, FlowMatchEulerDiscreteScheduler); torch.cuda.synchronize(); start = time.perf_counter()
                z4 = sch.step(v, sch.timesteps[3], z3, return_dict=False)[0]
                torch.cuda.synchronize(); commit_seconds = time.perf_counter()-start
                fields[key] = {'latent4': retain(root / 'arrays/images' / key, 'latent4', z4)}
                actual.append({**phase, 'Z3PayloadSHA256': payload(z3), 'velocityPayloadSHA256': payload(v),
                    'Z4PayloadSHA256': payload(z4), 'sigmaDevice': str(sch.sigmas.device), 'sigmaDtype': str(sch.sigmas.dtype),
                    'finalStepIndex': sch.step_index})
                z = pipe._unpack_latents_with_ids(z4, ids, 32, 32)
                mean = vae.bn.running_mean.view(1,-1,1,1).to(z.device,z.dtype)
                std = torch.sqrt(vae.bn.running_var.view(1,-1,1,1)+vae.config.batch_norm_eps).to(z.device,z.dtype)
                z = pipe._unpatchify_latents(z*std+mean)
                fields[key]['vaeInput'] = retain(root / 'arrays/images' / key, 'vaeInput', z)
                torch.cuda.synchronize(); start = time.perf_counter()
                decoded = vae.decode(z, return_dict=False)[0]
                torch.cuda.synchronize(); vae_seconds = time.perf_counter()-start
                fields[key]['decodedRaw'] = retain(root / 'arrays/images' / key, 'decodedRaw', decoded)
                rgb = pipe.image_processor.postprocess(decoded, output_type='np')
                png = pipe.image_processor.numpy_to_pil(rgb)[0]; path = root / (key+'.png'); png.save(path)
                fields[key]['rgbFloat'] = retain(root / 'arrays/images' / key, 'rgbFloat', rgb)
                import numpy as np
                fields[key]['pngRGB'] = retain(root / 'arrays/images' / key, 'pngRGB', np.asarray(png))
                if rgb.shape != (1,512,512,3) or sch.step_index != 4:
                    raise ValueError('Consumer shape/step differs')
                images[key] = {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
                timings.append({**phase, 'schedulerSeconds': commit_seconds, 'VAESeconds': vae_seconds})
                print('LAST_VELOCITY_IMAGE', key, timings[-1], flush=True)
    finally:
        for h in hooks: h.remove()
    if census() != before or sha(root / 'forecast-seal.json') != fsha:
        raise ValueError('Native state or forecasts mutated')
    assets_unchanged(assets)
    save(root / 'images.json', {'schema': 'tce-last-velocity-images-v1', 'requestSHA256': r['requestSHA256'],
        'forecastSealSHA256': fsha, 'fields': fields, 'images': images, 'actual': actual, 'timings': timings,
        'operations': ops, 'assets': assets, 'numericalProfile': numerical, 'setupSecondsAfterImports': setup,
        'workerSecondsAfterImports': time.perf_counter()-tick, 'cpuSecondsAfterImports': time.process_time()-cpu,
        'newEncoderEvaluations': 0, 'newDenoiserEvaluations': 0, 'nativeVAEDecodes': len(images),
        'VAEParameterBytes': sum(p.numel()*p.element_size() for p in vae.parameters()),
        'scope': 'Four original VAE decodes on forecast-driven finite scheduler outputs; paid three-pass prefixes supplied.'})
