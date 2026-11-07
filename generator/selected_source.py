"""Backward slice of joint0 to four demanded text rows; original learned maps."""
PREFIX = 'transformer_blocks.0.'
AFFINES = tuple(PREFIX + n + '.weight' for n in (
    'attn.to_k', 'attn.to_v', 'attn.add_k_proj', 'attn.add_v_proj',
    'attn.add_q_proj', 'attn.to_add_out', 'ff_context.linear_in', 'ff_context.linear_out'))
NORMS = tuple(PREFIX + 'attn.' + n + '.weight' for n in ('norm_k', 'norm_added_k', 'norm_added_q'))
PARAMETERS = AFFINES + NORMS


def form(image, text, img_mod, txt_mod, rope, support, weights, native):
    import torch
    import torch.nn.functional as F
    width = text.shape[-1]
    heads = weights[PREFIX + 'attn.norm_k.weight'].numel()
    head_count = width // heads
    im, tm = [mod[:, None, :].chunk(6, -1) for mod in (img_mod, txt_mod)]
    xi = (1 + im[1]) * F.layer_norm(image, (width,), eps=1e-6) + im[0]
    xt = (1 + tm[1]) * F.layer_norm(text, (width,), eps=1e-6) + tm[0]
    def linear(name, x):
        return F.linear(x, weights[PREFIX + name + '.weight'])
    def norm(name, x):
        return F.rms_norm(x.unflatten(-1, (head_count, heads)), (heads,),
                          weights[PREFIX + 'attn.' + name + '.weight'], eps=1e-6)
    ki = norm('norm_k', linear('attn.to_k', xi))
    vi = linear('attn.to_v', xi).unflatten(-1, (head_count, heads))
    kt = norm('norm_added_k', linear('attn.add_k_proj', xt))
    vt = linear('attn.add_v_proj', xt).unflatten(-1, (head_count, heads))
    q = norm('norm_added_q', linear('attn.add_q_proj', xt[:, support]))
    q = native.apply_rotary_emb(q, tuple(x[support] for x in rope), sequence_dim=1)
    k = native.apply_rotary_emb(torch.cat([kt, ki], 1), rope, sequence_dim=1)
    v = torch.cat([vt, vi], 1)
    h = native.dispatch_attention_fn(q, k, v, attn_mask=None, backend=None, parallel_config=None).flatten(2, 3).to(q.dtype)
    a = linear('attn.to_add_out', h)
    selected = text[:, support]
    u = selected + tm[2] * a
    x = F.layer_norm(u, (width,), eps=1e-6) * (1 + tm[4]) + tm[3]
    fused = linear('ff_context.linear_in', x)
    product = native.Flux2SwiGLU()(fused)
    m = linear('ff_context.linear_out', product)
    t = u + tm[5] * m
    return dict(R=selected, Q=q, K=k, V=v, H=h, A=a, U=u, X=x, fused=fused,
                product=product, M=m, T=t, gateA=tm[2], gateM=tm[5])


def splice(p, q):
    """P's MLP has already consumed its own P attention; recipient gates are Q."""
    return {'QQQ': q['T'], 'PQP': (p['R'] + q['gateA'] * q['A']) + q['gateM'] * p['M']}


def load_pages(record, device='cuda'):
    import torch
    from saturn_pub.store import LocalStore
    from encoder_atoms import sha
    values = {}
    for slot, item in record['pages'].items():
        for path, digest in item['files'].items():
            if sha(path) != digest:
                raise ValueError('Source cut/page/blob bytes changed')
        value, metadata = LocalStore(item['store'])._read_page(item['pageID'])
        if metadata != item['metadata']:
            raise ValueError('Source slot metadata changed')
        def restore(x):
            if isinstance(x, torch.Tensor):
                y = torch.empty_strided(tuple(x.shape), tuple(x.stride()), dtype=x.dtype, device=device)
                return y.copy_(x.to(device))
            if isinstance(x, dict):
                return {k: restore(v) for k, v in x.items()}
            if isinstance(x, tuple):
                return tuple(restore(v) for v in x)
            return x
        values[slot] = restore(value)
    return values
