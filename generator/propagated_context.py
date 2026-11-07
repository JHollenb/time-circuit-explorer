"""Joint1 complementary reads → own sequential MLP → own joint2 query."""
P1, P2 = 'transformer_blocks.1.', 'transformer_blocks.2.'
BANK_PARAMETERS = tuple(P1 + n + '.weight' for n in
    ['attn.to_k', 'attn.to_v', 'attn.add_k_proj', 'attn.add_v_proj', 'attn.norm_k', 'attn.norm_added_k'])
SELECTED_PARAMETERS = tuple(P1 + 'attn.' + n + '.weight' for n in ['add_q_proj', 'norm_added_q'])
RETURN_PARAMETERS = tuple(P1 + n + '.weight' for n in ['attn.to_add_out', 'ff_context.linear_in', 'ff_context.linear_out'])
NEXT_PARAMETERS = tuple(P2 + 'attn.' + n + '.weight' for n in ['add_q_proj', 'norm_added_q'])


def normalized(rows, mod):
    import torch.nn.functional as F
    m = mod[:, None, :].chunk(6, -1)
    return (1 + m[1]) * F.layer_norm(rows, (rows.shape[-1],), eps=1e-6) + m[0]


def project(x, prefix, name, norm, weights, native, rope):
    import torch.nn.functional as F
    head_width = weights[prefix + 'attn.' + (norm or 'norm_added_k') + '.weight'].numel()
    y = F.linear(x, weights[prefix + 'attn.' + name + '.weight']).unflatten(-1, (x.shape[-1] // head_width, head_width))
    if norm:
        y = F.rms_norm(y, (head_width,), weights[prefix + 'attn.' + norm + '.weight'], eps=1e-6)
        if rope is not None:
            y = native.apply_rotary_emb(y, rope, sequence_dim=1)
    return y


def banks(image, text, im, tm, rope, weights, native):
    import torch
    xi, xt = normalized(image, im), normalized(text, tm)
    # Text before image, preserving the source's full-query projection geometry.
    k = torch.cat([project(xt, P1, 'add_k_proj', 'norm_added_k', weights, native, None),
                   project(xi, P1, 'to_k', 'norm_k', weights, native, None)], 1)
    k = native.apply_rotary_emb(k, rope, sequence_dim=1)
    v = torch.cat([project(xt, P1, 'add_v_proj', None, weights, native, rope),
                   project(xi, P1, 'to_v', None, weights, native, rope)], 1)
    return {'K': k, 'V': v}


def selected(text, mod, rope, bank, support, weights, native):
    r = text[:, support]
    factors = tuple(x[support] for x in rope)
    q = project(normalized(r, mod), P1, 'add_q_proj', 'norm_added_q', weights, native, factors)
    return {'R': r, 'txtMod': mod, 'Q': q, 'K': bank['K'][:, support], 'V': bank['V'][:, support],
            'ropeCos': factors[0], 'ropeSin': factors[1]}


def advance(values, bank_k, bank_v, support, weights, native):
    import torch.nn.functional as F
    k, v = bank_k.contiguous().clone(), bank_v.contiguous().clone()
    k[:, support], v[:, support] = values['K'], values['V']
    q = values['Q']
    h = native.dispatch_attention_fn(q, k, v, attn_mask=None, backend=None, parallel_config=None).flatten(2, 3).to(q.dtype)
    a = F.linear(h, weights[P1 + 'attn.to_add_out.weight'])
    m = values['txtMod'][:, None, :].chunk(6, -1)
    da = m[2] * a
    u = values['R'] + da
    x = F.layer_norm(u, (u.shape[-1],), eps=1e-6) * (1 + m[4]) + m[3]
    fused = F.linear(x, weights[P1 + 'ff_context.linear_in.weight'])
    product = native.Flux2SwiGLU()(fused)
    returned = F.linear(product, weights[P1 + 'ff_context.linear_out.weight'])
    dm = m[5] * returned
    return dict(K_hybrid=k, V_hybrid=v, H=h, A=a, D_attention=da, U=u, X=x,
                fused=fused, product=product, M=returned, D_mlp=dm, T=u + dm)


def next_query(rows, mod, rope, weights, native):
    import torch.nn.functional as F
    layernorm = F.layer_norm(rows, (rows.shape[-1],), eps=1e-6)
    m = mod[:, None, :].chunk(6, -1)
    x = (1 + m[1]) * layernorm + m[0]
    head_width = weights[P2 + 'attn.norm_added_q.weight'].numel()
    q0 = F.linear(x, weights[P2 + 'attn.add_q_proj.weight']).unflatten(-1, (rows.shape[-1] // head_width, head_width))
    qn = F.rms_norm(q0, (head_width,), weights[P2 + 'attn.norm_added_q.weight'], eps=1e-6)
    return {'nextLayerNorm': layernorm, 'nextX': x, 'nextQProjected': q0, 'nextQNormalized': qn,
            'nextQ': native.apply_rotary_emb(qn, rope, sequence_dim=1)}
