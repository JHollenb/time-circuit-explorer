"""Fixed temporal law; no neural imports, fitting or target values."""
from pathlib import Path
import os
import sys
import time
import numpy as np
from encoder_atoms import read, save, words
from paired_fields import load, retain


def coefficient(sigmas):
    s = np.asarray(sigmas, dtype=np.float64)
    if s.shape != (5,) or not np.isfinite(s).all() or not np.all(np.diff(s) < 0):
        raise ValueError('Expected five strictly decreasing finite sigma values')
    return float((s[3] - s[2]) / (s[2] - s[1]))


def forecast(v1, v2, sigmas, policy):
    x, y = [np.asarray(v, dtype=np.float32) for v in (v1, v2)]
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid preceding velocities')
    a = np.float32(coefficient(sigmas))
    if policy == 'hold':
        raw = y.copy()
    elif policy == 'secant':
        delta = np.subtract(y, x, dtype=np.float32)
        scaled = np.multiply(delta, a, dtype=np.float32)
        raw = np.add(y, scaled, dtype=np.float32)
    else:
        raise ValueError('Unknown frozen policy')
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite forecast')
    return raw, words(raw)


def install_guard(root):
    root = Path(root).resolve(); s = read(root / 'inputs.json')
    forbidden = {str(Path(p).resolve()) for p in [root / 'targets.json', *s['forbiddenTargets']]}
    model = Path(s['modelRoot']).resolve()
    allowed = {str((model / p).resolve()) for p in s['decoderAssets']}
    allowed_prefix = {str(Path(x['path']).resolve()) for fs in s['prefix'].values() for x in fs.values()}
    allowed_prefix.add(str(Path(s['ids']['path']).resolve()))
    parent = Path(s['parentRoot']).resolve()
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        p = Path(os.fsdecode(args[0])).resolve(); name = str(p)
        if name in forbidden or (p.is_relative_to(parent) and name not in allowed_prefix):
            raise PermissionError('Target unavailable before complete candidate seal')
        if (p.is_relative_to(model) or p.suffix in ('.safetensors', '.pt', '.pth')) and name not in allowed:
            raise PermissionError('Only the declared native VAE assets are available')
    sys.addaudithook(audit)
    for p in [root / 'targets.json', model / 'transformer/diffusion_pytorch_model.safetensors']:
        try:
            open(p, 'rb')
        except PermissionError:
            continue
        raise RuntimeError('Target/model exclusion guard inactive')


def execute(root):
    from encoder_atoms import guard
    from flux_clock_study import verify
    root = Path(root).resolve(); r = read(root / 'request.json'); verify(root, r)
    s = read(root / 'inputs.json'); install_guard(root); guard()
    tick = time.perf_counter(); cpu = time.process_time(); fields = {}; timings = []
    for case, fs in s['prefix'].items():
        # Authenticate the actual common parent as well as the two predictor operands.
        z = load(fs['latent3']); v1 = load(fs['velocity1']); v2 = load(fs['velocity2'])
        if z.shape != v1.shape or z.shape != v2.shape:
            raise ValueError('Prefix shapes differ')
        for policy in r['policies']:
            start = time.perf_counter(); raw, emitted = forecast(v1, v2, s['schedule']['sigmas'], policy)
            arithmetic = time.perf_counter() - start; key = policy + '-' + case
            fields[key] = {'forecastFP32': retain(root / 'arrays/forecast' / key, 'forecastFP32', raw),
                           'velocity3': retain(root / 'arrays/forecast' / key, 'velocity3', emitted)}
            fields[key]['velocity3']['logicalDtype'] = 'torch.bfloat16'
            timings.append({'case': case, 'policy': policy, 'arithmeticSeconds': arithmetic})
    save(root / 'forecast-seal.json', {'schema': 'tce-last-velocity-forecast-seal-v1',
        'requestSHA256': r['requestSHA256'], 'prefix': s['prefix'], 'coefficientFP64': coefficient(s['schedule']['sigmas']),
        'coefficientFP32': float(np.float32(coefficient(s['schedule']['sigmas']))),
        'fields': fields, 'timings': timings, 'workerSecondsAfterImports': time.perf_counter()-tick,
        'cpuSecondsAfterImports': time.process_time()-cpu, 'neuralEvaluations': 0,
        'scope': 'Known own-prefix fields; no original weights, target values or neural framework.'})
