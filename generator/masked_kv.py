"""Functional block with input-mask-defined K/V formation; queries stay complete."""
def block(graph, hidden, parameters, profile, index, context, dispatch, key_mask, mode, observe):
    import torch
    from torch.nn import functional as F
    if mode not in ('compact', 'zero'): raise ValueError('Unknown formation mode')
    mask, causal, _, cos, sin, _ = context
    b, length, width = hidden.shape; heads = profile['num_attention_heads']
    kv = profile['num_key_value_heads']; dim = profile['head_dim']; inter = profile['intermediate_size']
    if key_mask.shape != (b, length) or b != 1: raise ValueError('Complete key mask required')
    active = torch.nonzero(key_mask[0].bool(), as_tuple=False).flatten()
    if not len(active): raise ValueError('At least one attended key is required')
    if len(active) != length:
        if mask is None or mask.dtype != torch.bool or bool(mask[..., ~key_mask[0].bool()].any()):
            raise ValueError('Actual attention support does not exclude every omitted key')
    root = f'model.layers.{index}.'
    def p(name): return parameters[root + name + '.weight']
    def emit(name, value):
        observe(name, value); return value
    norm = graph['_rms_norm'](hidden, p('input_layernorm'), profile['rms_norm_eps'])
    q = F.linear(norm, p('self_attn.q_proj')).view(b, length, heads, dim)
    q = graph['_rms_norm'](q, p('self_attn.q_norm'), profile['rms_norm_eps']).transpose(1, 2)
    q = emit('q', q * cos.unsqueeze(1) + graph['_rotate_half'](q) * sin.unsqueeze(1))
    source = norm.index_select(1, active) if mode == 'compact' else norm
    count = source.shape[1]
    k = F.linear(source, p('self_attn.k_proj')).view(b, count, kv, dim)
    k = graph['_rms_norm'](k, p('self_attn.k_norm'), profile['rms_norm_eps']).transpose(1, 2)
    v = F.linear(source, p('self_attn.v_proj')).view(b, count, kv, dim).transpose(1, 2)
    c = cos.index_select(1, active) if mode == 'compact' else cos
    s = sin.index_select(1, active) if mode == 'compact' else sin
    k = k * c.unsqueeze(1) + graph['_rotate_half'](k) * s.unsqueeze(1)
    # Row-major B,T,H,D storage gives the source's original noncontiguous B,H,T,D stride.
    def scatter(value):
        result = torch.zeros((b, length, kv, dim), dtype=hidden.dtype, device=hidden.device)
        chosen = value if mode == 'compact' else value.index_select(2, active)
        result.index_copy_(1, active, chosen.transpose(1, 2))
        return result.transpose(1, 2)
    k, v = emit('k', scatter(k)), emit('v', scatter(v))
    head = emit('head', graph['_attention'](q, k, v, mask, is_causal=causal,
                scale=dim ** -.5, groups=profile['num_key_value_groups'], dispatch=dispatch))
    residual = emit('R', hidden + F.linear(head.reshape(b, length, -1), p('self_attn.o_proj')))
    post = graph['_rms_norm'](residual, p('post_attention_layernorm'), profile['rms_norm_eps'])
    product = F.silu(F.linear(post, p('mlp.gate_proj'))) * F.linear(post, p('mlp.up_proj'))
    mlp = emit('M', F.linear(product, p('mlp.down_proj')))
    return emit('Hnext', residual + mlp)
