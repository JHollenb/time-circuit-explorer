"""Complete retained fields and paired-response accounts; no model imports."""
import hashlib
from pathlib import Path
from encoder_atoms import np, sha, floats


def payload(value):
    return hashlib.sha256(array(value).tobytes()).hexdigest()


def array(value):
    if hasattr(value, 'detach'):
        import torch
        a = value.detach().contiguous().cpu()
        return a.view(torch.uint16).numpy().copy() if value.dtype == torch.bfloat16 else a.numpy().copy()
    return np.asarray(value)


def retain(root, name, value):
    a = array(value); logical = str(value.dtype)
    decoded = floats(a) if logical == 'torch.bfloat16' else a
    if not np.isfinite(decoded).all(): raise ValueError('Nonfinite retained field')
    p = Path(root) / (name + '.npy'); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists(): raise ValueError('Retained field already exists')
    np.save(p, a, allow_pickle=False)
    return {'path': str(p.resolve()), 'sha256': sha(p), 'payloadSHA256': hashlib.sha256(a.tobytes()).hexdigest(),
        'bytes': p.stat().st_size, 'shape': list(a.shape), 'storedDtype': str(a.dtype), 'logicalDtype': logical}


def load(item, device=None):
    if sha(item['path']) != item['sha256']: raise ValueError('Retained field changed')
    a = np.load(item['path'], allow_pickle=False)
    if list(a.shape) != item['shape'] or hashlib.sha256(a.tobytes()).hexdigest() != item['payloadSHA256']:
        raise ValueError('Retained field geometry/payload changed')
    if device is not None:
        import torch
        value = torch.from_numpy(a.copy())
        if item['logicalDtype'] == 'torch.bfloat16': value = value.view(torch.bfloat16)
        return value.to(device)
    return (floats(a) if item['logicalDtype'] == 'torch.bfloat16' else a).astype(np.float64)


def metrics(candidate, reference):
    from masked_kv_study import field_metrics
    a = np.asarray(candidate, dtype=np.float64); b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape: raise ValueError('Paired geometry differs')
    return {**field_metrics(a, b), 'mae': float(np.mean(np.abs(a - b))),
        'candidateRms': float(np.sqrt(np.mean(a * a))),
        'cosine': float(np.sum(a*b) / np.sqrt(np.sum(a*a)*np.sum(b*b))) if np.any(a) and np.any(b) else None}


def paired(candidate_a, candidate_b, reference_a, reference_b):
    # Cast before subtraction: native PNG inputs may be unsigned bytes.
    ca, cb, ra, rb = [np.asarray(x, dtype=np.float64) for x in (candidate_a, candidate_b, reference_a, reference_b)]
    return metrics(cb - ca, rb - ra)
