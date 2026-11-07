"""Source-addressed row reconstruction and positive softmax cache updates.

The cache is conditional on fixed image queries and a fixed complement bank.
No learned model is imported at module import time.
"""
PREFIX = 'transformer_blocks.1.attn.'
MAPS = ('to_q', 'to_k', 'to_v', 'add_k_proj', 'add_v_proj', 'to_out.0')
NORMS = ('norm_q', 'norm_k', 'norm_added_k')
PARAMETERS = tuple(PREFIX + n + '.weight' for n in MAPS + NORMS)
ACTOR_PARAMETERS = tuple(PREFIX + n + '.weight' for n in ('add_k_proj', 'add_v_proj', 'to_out.0', 'norm_added_k'))


def resolve(context, tokens):
    """Original changed spans plus first two unchanged valid content rows."""
    a, b = tokens['base'], tokens['source']
    if not len(a['ids']) == len(b['ids']) == 512:
        raise ValueError('Original padded token geometry differs')
    changed = [i for i, (x, y) in enumerate(zip(a['ids'], b['ids'])) if x != y]
    spans = sorted({i for t in [a, b] for s in t['attribute_spans'] for i in s['rows']})
    if changed != spans or len(changed) != 2:
        raise ValueError('Changed-ID/span instrument differs')
    intervals = []
    for role, t in [('base', a), ('source', b)]:
        prompt = context['prompts'][role]
        origins = {s['start'] - prompt.index(s['attribute']) for s in t['attribute_spans']}
        if len(origins) != 1:
            raise ValueError('Ambiguous raw prompt origin')
        start = origins.pop()
        intervals.append((start, start + len(prompt)))
    candidates = [i for i in range(max(changed) + 1, 512)
                  if a['ids'][i] == b['ids'][i] and a['mask'][i] and b['mask'][i]
                  and all(lo <= t['offsets'][i][0] < t['offsets'][i][1] <= hi
                          for t, (lo, hi) in zip([a, b], intervals))
                  and not (a['pieces'][i].startswith('<|') and a['pieces'][i].endswith('|>'))]
    if len(candidates) < 2:
        raise ValueError('Unchanged content instrument incomplete')
    return {'changed': changed, 'unchanged': candidates[:2], 'support': sorted(changed + candidates[:2]),
            'candidates': candidates, 'input': 'original IDs, masks, attribute spans, offsets and pieces only'}


def rows(r, a, m, ga, gm):
    """Preserve separate BF16 multiply/add boundaries in the family writer."""
    return (r + ga * a) + gm * m


def complement(q, k, v, support):
    import torch
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** .5
    m = scores.max(-1, keepdim=True).values
    weights = torch.exp(scores - m)
    weights[..., support] = 0
    z, n = weights.sum(-1, keepdim=True), weights @ v
    if not bool(torch.isfinite(n).all() and torch.isfinite(z).all() and (z > 0).all()):
        raise ValueError('Invalid positive complement')
    return m, z, n


def update(q, k, v, m, z, n):
    import torch
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** .5
    peak = torch.maximum(m, scores.max(-1, keepdim=True).values)
    scale, weights = torch.exp(m - peak), torch.exp(scores - peak)
    denominator = scale * z + weights.sum(-1, keepdim=True)
    numerator = scale * n + weights @ v
    h = numerator / denominator
    if not bool(torch.isfinite(h).all() and torch.isfinite(denominator).all() and (denominator > 0).all()):
        raise ValueError('Invalid sparse positive update')
    return {'scores': scores, 'peak': peak, 'cacheScale': scale, 'selectedWeights': weights,
            'denominator': denominator, 'numerator': numerator, 'H_FP64': h}


def load_weights(path, selected, entries):
    import torch
    from safetensors import safe_open
    from paired_fields import payload
    weights, receipts = {}, []
    with safe_open(str(path), framework='pt', device='cpu') as archive:
        for name in selected:
            value = archive.get_tensor(name)
            e = entries[name]
            if value.dtype != torch.bfloat16 or list(value.shape) != e['shape']:
                raise ValueError('Original selected parameter differs')
            receipts.append({'name': name, 'shape': e['shape'], 'bytes': value.numel() * 2,
                             'payloadSHA256': payload(value), 'dataOffsets': e['data_offsets']})
            weights[name] = value.to('cuda')
    return weights, receipts


def normalize(x, modulation):
    import torch.nn.functional as F
    shift, scale, gate = modulation[:, None, :].chunk(6, -1)[:3]
    return (1 + scale) * F.layer_norm(x, (x.shape[-1],), eps=1e-6) + shift, gate


def project(weights, name, x, log):
    import torch.nn.functional as F
    w = weights[PREFIX + name + '.weight']
    y = F.linear(x, w)
    log.append({'parameter': PREFIX + name + '.weight', 'inputShape': list(x.shape), 'inputStride': list(x.stride()),
                'weightShape': list(w.shape), 'dtype': str(x.dtype), 'device': x.device.type,
                'nominalMACs': x.numel() // x.shape[-1] * w.numel()})
    return y


def headnorm(weights, name, x):
    import torch
    # Native torch.nn.RMSNorm is the original selected attention normalizer.
    return torch.nn.functional.rms_norm(x, (128,), weights[PREFIX + name + '.weight'], eps=1e-6)


def heads(x):
    return x[0].permute(1, 0, 2).to('cpu').double()


def quarantine(root, allowed, denied):
    """Trusted-code file guard, including other phases' arrays inside root."""
    import os
    import sys
    from pathlib import Path
    root = Path(root).resolve()
    allowed = {str(Path(p).resolve()) for p in allowed}
    denied = {str(Path(p).resolve()) for p in denied}
    accesses, created = {}, set()
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        key = str(p)
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY | os.O_RDWR))
        if key in denied:
            raise PermissionError('Sealed comparison unavailable to serving')
        if writing and p.is_relative_to(root):
            created.add(key)
        elif key in allowed:
            accesses[key] = accesses.get(key, 0) + 1
        elif p.suffix in {'.npy', '.npz', '.safetensors', '.pt', '.pth'} and key not in created:
            raise PermissionError('Undeclared numerical serving input')
    sys.addaudithook(audit)
    for p in denied:
        try:
            open(p, 'rb')
        except PermissionError:
            continue
        raise ValueError('Target denial inactive')
    return accesses
