"""Selected source operands with an explicitly supplied complementary context."""
P0 = 'transformer_blocks.0.'
SHARED_PARAMETERS = tuple(P0 + 'attn.' + n + '.weight' for n in
                          ('add_q_proj', 'add_k_proj', 'add_v_proj', 'norm_added_q', 'norm_added_k'))
RETURN_PARAMETERS = tuple(P0 + n + '.weight' for n in
                          ('attn.to_add_out', 'ff_context.linear_in', 'ff_context.linear_out'))


def selected(text, mod, rope, support, weights, native):
    import torch.nn.functional as F
    r = text[:, support]
    m = mod[:, None, :].chunk(6, -1)
    x = (1 + m[1]) * F.layer_norm(r, (r.shape[-1],), eps=1e-6) + m[0]
    width = weights[P0 + 'attn.norm_added_k.weight'].numel()
    heads = r.shape[-1] // width
    values = {}
    for name, norm, field in [('add_q_proj', 'norm_added_q', 'Q'), ('add_k_proj', 'norm_added_k', 'K'), ('add_v_proj', None, 'V')]:
        y = F.linear(x, weights[P0 + 'attn.' + name + '.weight']).unflatten(-1, (heads, width))
        if norm is not None:
            y = F.rms_norm(y, (width,), weights[P0 + 'attn.' + norm + '.weight'], eps=1e-6)
            y = native.apply_rotary_emb(y, tuple(v[support] for v in rope), sequence_dim=1)
        values[field] = y
    return {'R': r, 'txtMod': mod, **values}


def source(values, bank_k, bank_v, q_attention, support, weights, native):
    import torch.nn.functional as F
    k, v = bank_k.contiguous().clone(), bank_v.contiguous().clone()
    k[:, support], v[:, support] = values['K'], values['V']
    q = values['Q']
    h = native.dispatch_attention_fn(q, k, v, attn_mask=None, backend=None, parallel_config=None).flatten(2, 3).to(q.dtype)
    a = F.linear(h, weights[P0 + 'attn.to_add_out.weight'])
    m = values['txtMod'][:, None, :].chunk(6, -1)
    u = values['R'] + m[2] * a
    x = F.layer_norm(u, (u.shape[-1],), eps=1e-6) * (1 + m[4]) + m[3]
    fused = F.linear(x, weights[P0 + 'ff_context.linear_in.weight'])
    product = native.Flux2SwiGLU()(fused)
    returned = F.linear(product, weights[P0 + 'ff_context.linear_out.weight'])
    own = u + m[5] * returned
    pqp = (values['R'] + m[2] * q_attention) + m[5] * returned
    return dict(K_hybrid=k, V_hybrid=v, H=h, A=a, U=u, X=x, fused=fused,
                product=product, M=returned, T=own, PQP=pqp)


def recipient(values, text_rows, weights, wo, native):
    from sparse_text_kv import normalize, project, headnorm, heads
    from projected_complement import projected_read
    import torch.nn.functional as F
    x, _ = normalize(text_rows, values['txtMod'])
    k = headnorm(weights, 'norm_added_k', project(weights, 'add_k_proj', x, []).unflatten(-1, (24, 128)))
    k = native.apply_rotary_emb(k, (values['ropeCos'], values['ropeSin']), sequence_dim=1)
    v = project(weights, 'add_v_proj', x, []).unflatten(-1, (24, 128))
    account = projected_read(heads(values['Q']), heads(k), heads(v), values['m'], values['z'], values['projectedN'], wo, F.linear)
    attention = account['projectionFP64'].reshape(1, 1024, 3072).float().bfloat16().to('cuda')
    carrier = values['gate'] * attention
    return {'K': k, 'V': v, **account, 'attentionReturn': attention, 'carrier': carrier, 'R_attention': values['image'] + carrier}
