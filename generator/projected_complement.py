"""Compose a fixed-query positive attention cache with physical Wo head blocks."""
from sparse_text_kv import PREFIX, ACTOR_PARAMETERS

WO = PREFIX + 'to_out.0.weight'


def project_cache(n, wo, linear):
    import torch
    h, queries, width = n.shape
    if wo.shape[1] != h * width or wo.dtype != torch.float64 or n.dtype != torch.float64 or wo.device.type != 'cpu' or n.device.type != 'cpu':
        raise ValueError('Projected complement geometry/dtype differs')
    result = torch.empty((h, queries, wo.shape[0]), dtype=n.dtype, device=n.device)
    for head in range(h):
        result[head] = linear(n[head], wo[:, head * width:(head + 1) * width])
    return result


def terms(q, k, m, z):
    import torch
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** .5
    peak = torch.maximum(m, scores.max(-1, keepdim=True).values)
    scale, w = torch.exp(m - peak), torch.exp(scores - peak)
    denominator = scale * z + w.sum(-1, keepdim=True)
    if not bool(torch.isfinite(denominator).all() and (denominator > 0).all()):
        raise ValueError('Invalid positive projected-reader partition')
    return {'scores': scores, 'peak': peak, 'cacheScale': scale, 'selectedWeights': w, 'denominator': denominator}


def projected_read(q, k, v, m, z, nt, wo, linear):
    """Literal per-head AV, cache scale/add, division, then ordered head sum."""
    import torch
    account = terms(q, k, m, z)
    h, selected, width = v.shape
    if q.shape[0] != h or nt.shape != (h, q.shape[1], wo.shape[0]) or wo.shape[1] != h * width:
        raise ValueError('Projected read geometry differs')
    projected = torch.empty((h, selected, wo.shape[0]), dtype=torch.float64, device='cpu')
    result = None
    for head in range(h):
        projected[head] = linear(v[head], wo[:, head * width:(head + 1) * width])
        a = account['selectedWeights'][head] @ projected[head]
        t = (account['cacheScale'][head] * nt[head] + a) / account['denominator'][head]
        result = t if result is None else result + t
    if not bool(torch.isfinite(result).all()):
        raise ValueError('Nonfinite projected-reader output')
    return {**account, 'projectedValues': projected, 'projectionFP64': result}


def load_selected(path, names, entries):
    import torch
    from safetensors import safe_open
    from paired_fields import payload
    weights, receipts = {}, []
    with safe_open(str(path), framework='pt', device='cpu') as archive:
        for name in names:
            value = archive.get_tensor(name)
            entry = entries[name]
            if value.dtype != torch.bfloat16 or list(value.shape) != entry['shape']:
                raise ValueError('Selected original parameter profile differs')
            receipts.append({'name': name, 'shape': entry['shape'], 'bytes': value.numel() * 2,
                             'payloadSHA256': payload(value), 'dataOffsets': entry['data_offsets']})
            weights[name] = value if name == WO else value.to('cuda')
    return weights, receipts


def fields(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            yield value
        else:
            for child in value.values():
                yield from fields(child)
    elif isinstance(value, list):
        for child in value:
            yield from fields(child)
