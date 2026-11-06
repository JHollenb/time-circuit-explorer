"""One same-noise, three-conditioning native image-consumer assay."""
import argparse
from pathlib import Path
import shutil
import time
from encoder_atoms import np, read, save, sha, identity, floats

HERE = Path(__file__).resolve().parent
SOURCE_CACHE = Path('/Users/jakeholl/.cache/mdb/qualification/flux-initial-noise-corrected-20261105/lib/python3.12/site-packages/diffusers')
NATIVE_FILES = ['pipelines/flux2/pipeline_flux2_klein.py', 'models/transformers/transformer_flux2.py',
    'models/autoencoders/autoencoder_kl_flux2.py', 'schedulers/scheduling_flow_match_euler_discrete.py',
    'image_processor.py', 'models/attention_dispatch.py', 'models/attention_processor.py']


def verify(root, r):
    if identity({k: v for k, v in r.items() if k != 'requestSHA256'}) != r['requestSHA256']: raise ValueError('Request changed')
    for n, h in r['sourceHashes'].items():
        if sha(Path(root) / n) != h: raise ValueError('Staged source changed: ' + n)


def stage(root):
    from mrun.client.api import Api
    from mrun.client.submit import _plans_for_hosts
    from publish_context_statistics import checked_collection
    parent = HERE.parent / 'artifacts/conditioning-demand-v2'; c, j, old = checked_collection(parent, current_workers=False)
    api = Api(); live = api.json('GET', '/api/jobs/' + j['job_id'])
    if live['state'] != 'succeeded' or live['payload_custody'] != c['custody']: raise ValueError('Parent job differs')
    history = read(parent / 'result/retained-fields.json')['histories']; previous = read(parent / 'result/source.json')
    manifest = read(HERE / 'kernel-manifest.json')
    if sha(HERE / 'checkpoint-files.json') != manifest['files']['checkpoint-files.json']: raise ValueError('Original asset identity changed')
    checkpoint = read(HERE / 'checkpoint-files.json')
    preview = _plans_for_hosts('flux2-klein-4b-distilled', 'forward', api, seq_lens=[1536], backend='diffusers', dtype='bfloat16', device='cuda')
    measured = api.json('GET', '/api/jobs/job-4350531b1dcd')
    if measured['state'] != 'succeeded' or measured['result']['peak_vram_mb'] != 12936.0: raise ValueError('Historical native profile differs')
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False); payload = root / 'payload'; payload.mkdir()
    for n in ['image_consequence.py', 'image_consequence_study.py', 'conditioning_demand.py', 'masked_kv_study.py', 'encoder_atoms.py']:
        shutil.copyfile(HERE / n, payload / n)
    source = {'parentJobID': j['job_id'], 'parentRequestSHA256': old['requestSHA256'],
        'parentConsumerSealSHA256': sha(parent / 'result/consumer-seal.json'), 'parentReportSHA256': sha(parent / 'result/report.json'),
        'parentByteVerificationSHA256': sha(parent / 'retained-array-verification.json'), 'parentRoot': j['result']['diagnostics']['cwd'],
        'E': {b: history[b]['E'] for b in ('full', 'old', 'new')}, 'priorC': {b: history[b]['C'] for b in ('full', 'old', 'new')},
        'modelRoot': previous['sourceSpec']['modelRoot'],
        'assets': {n: h for n, h in checkpoint.items() if n == 'model_index.json' or n.startswith(('transformer/', 'vae/', 'scheduler/'))},
        'nativeSources': {n: sha(SOURCE_CACHE / n) for n in NATIVE_FILES}}
    save(payload / 'source.json', source)
    basis = {'schema': 'tce-native-image-resource-basis-v1', 'intentFirstPreview': preview,
        'historicalNativeJobID': measured['job_id'], 'historicalReservation': measured['reservation'],
        'historicalResult': {k: measured['result'][k] for k in ('elapsed_s', 'peak_rss_mb', 'peak_vram_mb')},
        'historicalQueueNote': measured['meta']['queue_note'],
        'exception': 'Generic diffusion plan assumesB64/missing-config fallback and estimates40.2GBVRAM for actualB1. Reuse native same-model/BF16/512 B1 measured14336MBRAM/13000MBVRAM envelope. No encoder, fewer native branches/trajectory captures; actual peaks retained. Guard ceiling14300MB fits16376MBcard minus819MBmargin. This is not a speed claim.'}
    save(payload / 'resource-basis.json', basis)
    r = {'schema': 'tce-conditioning-image-request-v1', 'parentJobID': j['job_id'], 'parentRequestSHA256': old['requestSHA256'],
        'scope': 'Three actual sealed E fields through same-noise original native FLUX/VAE. Paid consumer work, not zero-neural image generation.',
        'protocol': 'One seed314159/noise seal; three serial B1/512px branches, fresh identical scheduler perbranch, four denoising steps, no encoder, extra compaction or seed selection. Seal all outputs before scoring.',
        'sourceHashes': {f.name: sha(f) for f in payload.iterdir()}, 'model': 'flux2-klein-4b-distilled', 'backend': 'diffusers', 'device': 'cuda', 'dtype': 'bfloat16',
        'batch_size': 1, 'seq_len': 1536, 'height': 512, 'width': 512, 'seed': 314159, 'steps': 4, 'timeoutSeconds': 600,
        'resourceException': {'ramMB': 14336, 'vramMB': 13000, 'cpuThreads': 4, 'diskGB': 1, 'basisSHA256': sha(payload / 'resource-basis.json')}}
    r['requestSHA256'] = identity(r); save(root / 'request.json', r); save(payload / 'request.json', r)
    save(payload / 'preflight.v1.json', {'schema_version': 'mrun-preflight-config-v1', 'hashed_roots': [f.name for f in payload.iterdir()] + ['preflight.v1.json']})
    return {'run': str(root), 'requestSHA256': r['requestSHA256']}


def submit(root):
    from mrun.client.submit import launch
    from run import command
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root / 'payload', r)
    if (root / 'job.json').exists(): raise ValueError('Already submitted')
    cmd = command()[:-5] + ['python', '-u', 'image_consequence_study.py', 'execute', '.']
    cmd[cmd.index('uv'):cmd.index('uv')] = ['OMP_NUM_THREADS=4', 'OPENBLAS_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'CUBLAS_WORKSPACE_CONFIG=:4096:8']
    job = launch(cmd, experiment='tce-conditioning-image-v1', model=r['model'], task='forward', payload=root / 'payload', pin='beast',
        needs={'cuda': True}, config={k: r[k] for k in ('schema', 'requestSHA256', 'model', 'backend', 'device', 'dtype', 'batch_size', 'seq_len', 'height', 'width')},
        ram_mb=14336, vram_mb=13000, cpu_threads=4, disk_gb=1, est_wall_s=180, timeout_s=600, detach=True, retry_on_kill=False, preflight=True, guarded=True,
        note='3sealedE branches,same seed314159/noise,512px B1 nativeKlein4B4steps. Expected12denoisers/12contextprojections/3VAE; no encoder/extra approximation. ActualE/ownlatent/schedule joins; full latent/floatpixels/PNG nearcompute. Native measured4350531b1dcd14336RAM/13000VRAM exception vs badB64 genericplan. CPU4,expected<5min/600s,no retries; paid consumer,not zero-neural.')
    value = {'jobID': str(job), 'requestSHA256': r['requestSHA256']}; save(root / 'job.json', value); return value


def execute(root):
    import hashlib
    import json
    import os
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8': raise ValueError('Pinned deterministic cuBLAS workspace required before Torch import')
    import torch
    import diffusers
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2 import pipeline_flux2_klein
    from masked_kv_study import field_metrics
    from conditioning_demand import alignment
    from image_consequence import png_account
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root, r); source = read(root / 'source.json')
    tick = time.perf_counter(); cpu = time.process_time(); out = root / 'response-result'; out.mkdir(); raw = root / 'image-arrays'; raw.mkdir()
    model = Path(source['modelRoot']); source_checks = []; setup_tick = time.perf_counter()
    for n, h in source['nativeSources'].items():
        if sha(Path(diffusers.__file__).parent / n) != h: raise ValueError('Native implementation changed: ' + n)
    for n, h in source['assets'].items():
        if sha(model / n) != h: raise ValueError('Original image-consumer asset changed: ' + n)
        st = (model / n).stat(); source_checks.append({'path': n, 'sha256': h, 'bytes': st.st_size, 'signature': [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]})
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    transformer = Flux2Transformer2DModel.from_pretrained(str(model / 'transformer'), torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True, device_map={'': 'cuda'})
    vae = AutoencoderKLFlux2.from_pretrained(str(model / 'vae'), torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True, device_map={'': 'cuda'})
    transformer.eval().requires_grad_(False); vae.eval().requires_grad_(False)
    scheduler_config = read(model / 'scheduler/scheduler_config.json')
    if scheduler_config.get('stochastic_sampling', False): raise ValueError('Fixed-noise assay requires the bound deterministic scheduler')
    pipe = Flux2KleinPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler_config), vae=vae, text_encoder=None, tokenizer=None,
        transformer=transformer, is_distilled=read(model / 'model_index.json')['is_distilled'])
    pipe.set_progress_bar_config(disable=True)
    if not pipe.config.is_distilled or pipe.text_encoder is not None or pipe.tokenizer is not None: raise ValueError('Unexpected native image configuration')
    def census():
        return {prefix + '.' + name: [v.data_ptr(), v._version, list(v.shape), str(v.dtype)] for prefix, m in [('transformer', transformer), ('vae', vae)]
            for name, v in [*m.named_parameters(), *m.named_buffers()]}
    before = census(); fields = {}; embeddings = {}; actual = []; times = []; steps = []; image_records = {}
    def words(value):
        a = value.detach().contiguous().cpu()
        return a.view(torch.uint16).numpy().copy() if value.dtype == torch.bfloat16 else a.numpy().copy()
    def payload(value): return hashlib.sha256(words(value).tobytes()).hexdigest()
    def retain(branch, name, value):
        a = words(value) if isinstance(value, torch.Tensor) else np.asarray(value)
        logical = str(value.dtype)
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()): raise ValueError('Nonfinite image-consumer field')
        if a.dtype.kind == 'f' and not np.isfinite(a).all(): raise ValueError('Nonfinite retained field')
        path = raw / branch / (name + '.npy'); path.parent.mkdir(parents=True, exist_ok=True); np.save(path, a, allow_pickle=False)
        item = {'path': str(path), 'sha256': sha(path), 'payloadSHA256': hashlib.sha256(a.tobytes()).hexdigest(), 'bytes': path.stat().st_size,
            'shape': list(a.shape), 'storedDtype': str(a.dtype), 'logicalDtype': logical}
        fields.setdefault(branch, {})[name] = item; return item
    for branch, item in source['E'].items():
        path = Path(item['path'])
        if not path.is_relative_to(source['parentRoot']) or sha(path) != item['sha256']: raise ValueError('Sealed E changed')
        a = np.load(path, allow_pickle=False)
        if a.shape != (1, 512, 7680) or a.dtype != np.uint16: raise ValueError('Sealed E geometry differs')
        embeddings[branch] = torch.from_numpy(a.copy()).view(torch.bfloat16).to('cuda')
        if payload(embeddings[branch]) != item['payloadSHA256']: raise ValueError('E transfer changed payload')
    phase = {'branch': None, 'step': 0}; expected_latent = {}; noise_calls = []; native_counts = {b: {'denoiser': 0, 'context': 0, 'joint': 0, 'single': 0, 'VAE': 0} for b in embeddings}
    old_randn = pipeline_flux2_klein.randn_tensor
    def randn(*args, **kwargs):
        value = old_randn(*args, **kwargs); noise_calls.append({'shape': list(value.shape), 'payloadSHA256': payload(value)}); return value
    pipeline_flux2_klein.randn_tensor = randn
    old_prepare = pipe.prepare_latents
    def prepare(*args, **kwargs):
        value, ids = old_prepare(*args, **kwargs)
        if phase['branch'] is not None:
            if payload(value) != noise_seal['packed']['payloadSHA256'] or payload(ids) != noise_seal['ids']['payloadSHA256']:
                raise ValueError('Branch initial noise/IDs differ')
            actual.append({**phase, 'kind': 'prepared-noise', 'payloadSHA256': payload(value)})
        return value, ids
    pipe.prepare_latents = prepare
    with torch.inference_mode():
        packed_noise, noise_ids = pipe.prepare_latents(1, transformer.config.in_channels // 4, 512, 512, torch.bfloat16, torch.device('cuda'), torch.Generator(device='cpu').manual_seed(r['seed']))
        raw_noise = pipe._unpack_latents_with_ids(packed_noise, noise_ids, 32, 32)
        if payload(pipe._pack_latents(raw_noise)) != payload(packed_noise): raise ValueError('Native noise unpack/repack does not close')
        noise_seal = {'schema': 'tce-image-noise-seal-v1', 'requestSHA256': r['requestSHA256'], 'seed': r['seed'],
            'packed': retain('shared', 'packedNoise', packed_noise), 'raw': retain('shared', 'rawNoise', raw_noise), 'ids': retain('shared', 'imageIDs', noise_ids)}
    save(out / 'noise-seal.json', noise_seal); setup_seconds = time.perf_counter() - setup_tick
    print('IMAGE_NOISE_SEALED', sha(out / 'noise-seal.json'), flush=True)
    handles = []; model_starts = {}; schedule = {}; c_reads = []; decode_starts = []
    def model_pre(module, args, kwargs):
        branch = phase['branch']; step = phase['step']; state = kwargs['hidden_states']; e = kwargs['encoder_hidden_states']
        if state.shape != (1, 1024, 128) or payload(state) != expected_latent[branch] or payload(e) != source['E'][branch]['payloadSHA256']:
            raise ValueError('Native denoiser read wrong branch state/E')
        ts = words(pipe.scheduler.timesteps); sig = words(pipe.scheduler.sigmas)
        schedule.setdefault(branch, {'timesteps': ts.tolist(), 'sigmas': sig.tolist(), 'timestepsPayloadSHA256': hashlib.sha256(ts.tobytes()).hexdigest(), 'sigmasPayloadSHA256': hashlib.sha256(sig.tobytes()).hexdigest()})
        actual.append({**phase, 'kind': 'denoiser-input', 'latentPayloadSHA256': payload(state), 'EPayloadSHA256': payload(e)})
        native_counts[branch]['denoiser'] += 1; torch.cuda.synchronize(); model_starts[branch] = time.perf_counter()
    def model_post(module, args, value):
        branch = phase['branch']; torch.cuda.synchronize()
        times.append({**phase, 'kind': 'denoiser', 'wallSecondsWithContextHooks': time.perf_counter() - model_starts[branch]})
        retain(branch, 'velocity' + str(phase['step']), value[0])
    def context_pre(module, args):
        branch = phase['branch']; digest = payload(args[0])
        if digest != source['E'][branch]['payloadSHA256']: raise ValueError('Native context projection read wrong E')
        native_counts[branch]['context'] += 1; actual.append({**phase, 'kind': 'context-input', 'payloadSHA256': digest})
    def context_post(module, args, value):
        branch = phase['branch']; digest = payload(value); name = 'context' + str(phase['step'])
        item = retain(branch, name, value)
        c_reads.append({**phase, 'field': item, 'priorPayloadExact': digest == source['priorC'][branch]['payloadSHA256']})
    handles += [transformer.register_forward_pre_hook(model_pre, with_kwargs=True), transformer.register_forward_hook(model_post),
        transformer.context_embedder.register_forward_pre_hook(context_pre), transformer.context_embedder.register_forward_hook(context_post)]
    def block_hook(kind):
        def hook(module, args): native_counts[phase['branch']][kind] += 1
        return hook
    for m in transformer.transformer_blocks: handles.append(m.register_forward_pre_hook(block_hook('joint')))
    for m in transformer.single_transformer_blocks: handles.append(m.register_forward_pre_hook(block_hook('single')))
    old_decode = vae.decode
    def decode(z, *args, **kwargs):
        branch = phase['branch']; native_counts[branch]['VAE'] += 1; retain(branch, 'vaeInput', z)
        torch.cuda.synchronize(); started = time.perf_counter(); value = old_decode(z, *args, **kwargs); torch.cuda.synchronize()
        times.append({'branch': branch, 'kind': 'VAE', 'wallSecondsExcludingCapture': time.perf_counter() - started})
        retain(branch, 'decodedRaw', value[0]); return value
    vae.decode = decode
    def callback(pipeline, step, timestep, data):
        branch = phase['branch']; item = retain(branch, 'latent' + str(step + 1), data['latents'])
        expected_latent[branch] = item['payloadSHA256']; steps.append({'branch': branch, 'step': step + 1, 'latent': item})
        phase['step'] = step + 1; return {}
    try:
        with torch.inference_mode():
            for branch in ('full', 'old', 'new'):
                phase.update(branch=branch, step=0); expected_latent[branch] = noise_seal['packed']['payloadSHA256']
                pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
                torch.cuda.synchronize(); started = time.perf_counter()
                image = pipe(prompt=None, prompt_embeds=embeddings[branch], height=512, width=512, num_inference_steps=4, guidance_scale=1.0,
                    latents=raw_noise.clone(), output_type='np', callback_on_step_end=callback, callback_on_step_end_tensor_inputs=['latents']).images
                torch.cuda.synchronize(); times.append({'branch': branch, 'kind': 'completeImage', 'wallSecondsWithCapture': time.perf_counter() - started})
                if image.shape != (1, 512, 512, 3) or phase['step'] != 4: raise ValueError('Native image/step geometry differs')
                retain(branch, 'rgbFloat', image)
                png = pipe.image_processor.numpy_to_pil(image)[0]; path = out / (branch + '.png'); png.save(path)
                retain(branch, 'pngRGB', np.asarray(png))
                image_records[branch] = {'path': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size, 'size': list(png.size)}
                print('IMAGE_BRANCH_COMPLETE', branch, native_counts[branch], flush=True)
    finally:
        pipeline_flux2_klein.randn_tensor = old_randn; pipe.prepare_latents = old_prepare; vae.decode = old_decode
        for handle in handles: handle.remove()
    if census() != before: raise ValueError('Native learned parameter/buffer closure changed')
    if len(noise_calls) != 1: raise ValueError('Initial noise redrawn')
    for branch, counts in native_counts.items():
        if counts != {'denoiser': 4, 'context': 4, 'joint': 20, 'single': 80, 'VAE': 1}: raise ValueError('Declared native consumer census differs')
        if schedule[branch] != schedule['full']: raise ValueError('Branch schedule differs')
    if payload(raw_noise) != noise_seal['raw']['payloadSHA256'] or payload(packed_noise) != noise_seal['packed']['payloadSHA256']: raise ValueError('Shared noise mutated')
    for entry in source_checks:
        s = (model / entry['path']).stat()
        if [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns] != entry['signature']: raise ValueError('Native asset changed during consumption')
    save(out / 'consumer-seal.json', {'schema': 'tce-native-image-consumer-seal-v1', 'requestSHA256': r['requestSHA256'],
        'noiseSealSHA256': sha(out / 'noise-seal.json'), 'fields': fields, 'images': image_records,
        'scope': 'Complete paid native image consumers seal before comparisons. No zero-neural generation or fresh-prompt generalization claim.'})
    print('IMAGE_CONSUMERS_SEALED', sha(out / 'consumer-seal.json'), flush=True)
    def load(item):
        if sha(item['path']) != item['sha256']: raise ValueError('Retained consumer field changed')
        a = np.load(item['path'], allow_pickle=False)
        return (floats(a) if item['logicalDtype'] == 'torch.bfloat16' else a).astype(np.float64)
    accounts = []; differences = {}; observations = 0
    ports = ['latent4', 'vaeInput', 'decodedRaw', 'rgbFloat', 'pngRGB']
    with (out / 'observations.jsonl').open('x') as stream:
        for port in ports:
            base, old, new = [load(fields[b][port]) for b in ('full', 'old', 'new')]
            is_bf16 = fields['full'][port]['logicalDtype'] == 'torch.bfloat16'
            accounts.append({'port': port, 'old': {**field_metrics(old, base, retained=is_bf16), 'mae': float(np.mean(np.abs(old - base)))},
                'new': {**field_metrics(new, base, retained=is_bf16), 'mae': float(np.mean(np.abs(new - base)))}, 'alignment': alignment(old, new, base)})
            for branch, value in [('old', old), ('new', new)]: differences[branch + '.' + port] = retain('differences', branch + '.' + port, value - base)
            if port in ('rgbFloat', 'pngRGB'):
                if base.ndim == 4: base, old, new = base[0], old[0], new[0]
                for row in range(512):
                    stream.write(json.dumps({'port': port, 'row': row, 'old': field_metrics(old[row], base[row]), 'new': field_metrics(new[row], base[row]), 'alignment': alignment(old[row], new[row], base[row])}, allow_nan=False) + '\n'); observations += 1
    png_metrics = png_account(*[np.load(fields[b]['pngRGB']['path'], allow_pickle=False) for b in ('full', 'old', 'new')])
    c_reproduction = []
    for branch in ('full', 'old', 'new'):
        prior = source['priorC'][branch]
        if sha(prior['path']) != prior['sha256']: raise ValueError('Prior context field changed')
        prior_value = floats(np.load(prior['path'], allow_pickle=False)).astype(np.float64)
        for item in c_reads:
            if item['branch'] == branch: c_reproduction.append({'branch': branch, 'step': item['step'], **field_metrics(load(item['field']), prior_value, retained=True)})
    for ports_ in fields.values():
        for item in ports_.values():
            if sha(item['path']) != item['sha256']: raise ValueError('Consumer field mutated after seal')
    save(out / 'retained-fields.json', {'fields': fields, 'differences': differences})
    save(out / 'consumer-receipts.json', {'actual': actual, 'nativeCounts': native_counts, 'noiseCalls': noise_calls, 'schedules': schedule, 'steps': steps,
        'contextReads': c_reads, 'parameterAndExistingBufferClosureUnchanged': True, 'parameterAndBufferEntries': len(before)})
    save(out / 'source.json', {**source, 'assetChecks': source_checks, 'versions': {'torch': torch.__version__, 'diffusers': diffusers.__version__, 'numpy': np.__version__},
        'cublasWorkspaceConfig': os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'numericalProfile': {'deterministicAlgorithms': torch.are_deterministic_algorithms_enabled(), 'threads': torch.get_num_threads(),
            'cudaMatmulTF32': torch.backends.cuda.matmul.allow_tf32, 'float32MatmulPrecision': torch.get_float32_matmul_precision(),
            'bf16ReducedPrecisionReduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'cudnnTF32': torch.backends.cudnn.allow_tf32, 'cudnnBenchmark': torch.backends.cudnn.benchmark, 'cudnnDeterministic': torch.backends.cudnn.deterministic},
        'attentionProcessors': sorted({type(m.processor).__name__ for m in transformer.modules() if hasattr(m, 'processor')})})
    save(out / 'report.json', {'schema': 'tce-native-image-consequence-v1', 'requestSHA256': r['requestSHA256'], 'scope': r['scope'], 'accounts': accounts,
        'png': png_metrics, 'contextReproduction': c_reproduction, 'observations': observations, 'timings': times,
        'setupSecondsIncludingAssetHashesAndNoise': setup_seconds, 'workerSecondsAfterFrameworkImports': time.perf_counter() - tick, 'cpuSecondsAfterFrameworkImports': time.process_time() - cpu,
        'peakCUDAAllocatedBytes': torch.cuda.max_memory_allocated(), 'work': {'nativeDenoiserCalls': 12, 'nativeContextProjections': 12, 'nativeJointBlocks': 60, 'nativeSingleBlocks': 240,
            'nativeVAEDecodes': 3, 'textEncoderCalls': 0, 'initialNoiseDraws': 1, 'newImages': 3},
        'mechanics_status': 'complete', 'instrument_status': 'executed', 'terminal_status': 'unassessed',
        'limitations': ['Paid original native FLUX/VAE consequence of three already-sealed E fields; not zero-neural image generation.',
            'One context/seed and dependent pixels; reference similarity does not certify prompt faithfulness or semantic quality.',
            'Denoiser timings include context hook checks; VAE capture is outside decode compute clock; complete-image clock includes callback/context/decoder captures but excludes final RGB-array/PNG save. No matched speedup.',
            'Original asset hashes/setup and historical prefix/atom compilation remain paid; no encoder reference rerun.']})
    shutil.copyfile(root / 'request.json', out / 'request.json'); shutil.copyfile(root / 'resource-basis.json', out / 'resource-basis.json'); print('IMAGE_CONSEQUENCE_DONE', flush=True)


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['stage', 'submit', 'execute', 'collect']); p.add_argument('run'); a = p.parse_args()
    if a.action == 'collect':
        from response_study import collect
        result = collect(a.run)
    else: result = globals()[a.action](a.run)
    if result is not None: print(json.dumps(result))
