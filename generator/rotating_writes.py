"""Full readable state with source-fixed, rotating residual writes.

Torch and the native consumer are supplied lazily. This is reduced original
learned computation, not a zero-neural executor.
"""
import numpy as np


def image_rows(block_index, side=32):
    if not isinstance(block_index, int) or block_index < 0 or side < 2 or side % 2:
        raise ValueError('Expected a nonnegative block and an even image side')
    h, w = np.indices((side, side))
    return np.flatnonzero((2 * (h % 2) + w % 2).reshape(-1) == block_index % 4)


def joint(block, image, text, mod_image, mod_text, rope, selected_image, native):
    import torch
    ai, fi = native.Flux2Modulation.split(mod_image, 2)
    at, ft = native.Flux2Modulation.split(mod_text, 2)
    xi = (1 + ai[1]) * block.norm1(image) + ai[0]
    xt = (1 + at[1]) * block.norm1_context(text) + at[0]
    attn = block.attn
    qi = attn.norm_q(attn.to_q(xi.index_select(1, selected_image)).unflatten(-1, (attn.heads, -1)))
    ki = attn.norm_k(attn.to_k(xi).unflatten(-1, (attn.heads, -1)))
    vi = attn.to_v(xi).unflatten(-1, (attn.heads, -1))
    qt = attn.norm_added_q(attn.add_q_proj(xt).unflatten(-1, (attn.heads, -1)))
    kt = attn.norm_added_k(attn.add_k_proj(xt).unflatten(-1, (attn.heads, -1)))
    vt = attn.add_v_proj(xt).unflatten(-1, (attn.heads, -1))
    indices = torch.cat([torch.arange(text.shape[1], device=image.device), text.shape[1] + selected_image])
    q = torch.cat([qt, qi], dim=1)
    k = torch.cat([kt, ki], dim=1)
    v = torch.cat([vt, vi], dim=1)
    if rope is not None:
        q = native.apply_rotary_emb(q, tuple(x.index_select(0, indices) for x in rope), sequence_dim=1)
        k = native.apply_rotary_emb(k, rope, sequence_dim=1)
    y = native.dispatch_attention_fn(q, k, v, attn_mask=None,
        backend=getattr(attn.processor, '_attention_backend', None),
        parallel_config=getattr(attn.processor, '_parallel_config', None)).flatten(2, 3).to(q.dtype)
    yt, yi = y.split([text.shape[1], selected_image.numel()], dim=1)
    yi = attn.to_out[1](attn.to_out[0](yi))
    yt = attn.to_add_out(yt)
    # Joint MLPs read their own newly attention-updated residuals.
    zi = image.index_select(1, selected_image) + ai[2] * yi
    zi = zi + fi[2] * block.ff(block.norm2(zi) * (1 + fi[1]) + fi[0])
    zt = text + at[2] * yt
    zt = zt + ft[2] * block.ff_context(block.norm2_context(zt) * (1 + ft[1]) + ft[0])
    result = image.clone()
    result.index_copy_(1, selected_image, zi)
    if result.dtype == torch.float16:
        zt = zt.clip(-65504, 65504)
    return zt, result


def single(block, state, modulation, rope, selected, native):
    import torch
    import torch.nn.functional as F
    shift, scale, gate = native.Flux2Modulation.split(modulation, 1)[0]
    x = (1 + scale) * block.norm(state) + shift
    xs = x.index_select(1, selected)
    attn = block.attn
    d = attn.inner_dim
    weight = attn.to_qkv_mlp_proj.weight
    bias = attn.to_qkv_mlp_proj.bias
    def affine(value, start, stop):
        return F.linear(value, weight[start:stop], None if bias is None else bias[start:stop])
    # Slice output weights before evaluation. Never form unused Q/MLP outputs.
    q = affine(xs, 0, d).unflatten(-1, (attn.heads, -1))
    k, v = affine(x, d, 3 * d).chunk(2, dim=-1)
    k = k.unflatten(-1, (attn.heads, -1))
    v = v.unflatten(-1, (attn.heads, -1))
    mlp = affine(xs, 3 * d, weight.shape[0])
    q, k = attn.norm_q(q), attn.norm_k(k)
    if rope is not None:
        q = native.apply_rotary_emb(q, tuple(a.index_select(0, selected) for a in rope), sequence_dim=1)
        k = native.apply_rotary_emb(k, rope, sequence_dim=1)
    y = native.dispatch_attention_fn(q, k, v, attn_mask=None,
        backend=getattr(attn.processor, '_attention_backend', None),
        parallel_config=getattr(attn.processor, '_parallel_config', None)).flatten(2, 3).to(q.dtype)
    # Both parallel branches consume the same incoming normalized state.
    update = attn.to_out(torch.cat([y, attn.mlp_act_fn(mlp)], dim=-1))
    z = state.index_select(1, selected) + gate * update
    if state.dtype == torch.float16:
        z = z.clip(-65504, 65504)
    result = state.clone()
    result.index_copy_(1, selected, z)
    return result
