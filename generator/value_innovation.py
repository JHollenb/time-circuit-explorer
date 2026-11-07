"""A finite fixed-Q/K value event, with no fitted response directions."""
from source_native_reader import AFFINES, PARAMETERS, NORMS
from normalization_innovation import IN, DOWN

WO = 'transformer_blocks.0.attn.to_out.0.weight'
STAGES = ('gatedCarrier', 'residualCommit', 'fused', 'product', 'mlpWrite', 'gatedMLP', 'imageExit')


def acquire(weights, z, e, img_ids, txt_ids, mod_image, mod_text, normalizers, pos_embed, native, heads):
    """Qualified source_reader.form prefix, stopping before its SDPA call."""
    import torch
    import torch.nn.functional as F
    if mod_image.ndim != 2 or mod_image.shape[-1] != 6 * weights['x_embedder.weight'].shape[0] or mod_text.shape != mod_image.shape:
        raise ValueError('Two-set modulation geometry differs')
    image = F.linear(z, weights['x_embedder.weight'])
    text = F.linear(e, weights['context_embedder.weight'])
    mi = mod_image[:, None, :].chunk(6, dim=-1)
    mt = mod_text[:, None, :].chunk(6, dim=-1)
    xi = (1 + mi[1]) * F.layer_norm(image, (image.shape[-1],), eps=1e-6) + mi[0]
    xt = (1 + mt[1]) * F.layer_norm(text, (text.shape[-1],), eps=1e-6) + mt[0]
    prefix = 'transformer_blocks.0.attn.'
    q = normalizers['norm_q'](F.linear(xi, weights[prefix + 'to_q.weight']).unflatten(-1, (heads, -1)))
    ki = normalizers['norm_k'](F.linear(xi, weights[prefix + 'to_k.weight']).unflatten(-1, (heads, -1)))
    vi = F.linear(xi, weights[prefix + 'to_v.weight']).unflatten(-1, (heads, -1))
    kt = normalizers['norm_added_k'](F.linear(xt, weights[prefix + 'add_k_proj.weight']).unflatten(-1, (heads, -1)))
    vt = F.linear(xt, weights[prefix + 'add_v_proj.weight']).unflatten(-1, (heads, -1))
    image_rope = pos_embed(img_ids[0] if img_ids.ndim == 3 else img_ids)
    text_rope = pos_embed(txt_ids[0] if txt_ids.ndim == 3 else txt_ids)
    full_rope = tuple(torch.cat([t, i], 0) for t, i in zip(text_rope, image_rope))
    q = native.apply_rotary_emb(q, image_rope, sequence_dim=1)
    k = native.apply_rotary_emb(torch.cat([kt, ki], 1), full_rope, sequence_dim=1)
    return image, q, k, torch.cat([vt, vi], 1)


def event(q, k, v, j, address_k):
    import torch
    if q.ndim != 4 or q.shape[0] != 1 or k.shape != v.shape or q.shape[2:] != k.shape[2:]:
        raise ValueError('Value-event Q/K/V geometry differs')
    if j == address_k or min(j, address_k) < 0 or max(j, address_k) >= k.shape[1]:
        raise ValueError('Distinct admitted key addresses required')
    if any(x.device.type != 'cpu' or x.dtype != torch.float64 or not torch.isfinite(x).all() for x in (q, k, v)):
        raise ValueError('Value event requires finite CPU FP64 operands')
    scores = torch.matmul(q[0].transpose(0, 1), k[0].permute(1, 2, 0)) / q.shape[-1] ** .5
    probabilities = scores.softmax(-1)
    selected = probabilities[..., [j, address_k]].transpose(0, 1).contiguous()
    gamma = selected[..., 0] - selected[..., 1]
    directions = v[0, address_k] - v[0, j]
    return gamma, directions, selected


def compile_event(wo, win, attention_gate, mlp_scale, gamma, directions):
    import torch
    import torch.nn.functional as F
    heads, head_width = directions.shape
    if wo.ndim != 2 or wo.shape[1] != heads * head_width or win.shape[1] != wo.shape[0] or gamma.shape[1] != heads:
        raise ValueError('Physical head-block geometry differs')
    if any(x.device.type != 'cpu' or x.dtype != torch.float64 or not torch.isfinite(x).all() for x in (wo, win, attention_gate, mlp_scale, gamma, directions)):
        raise ValueError('Factor compilation requires finite CPU FP64 operands')
    b = torch.stack([F.linear(directions[h], wo[:, h * head_width:(h + 1) * head_width]) for h in range(heads)])
    b = b * attention_gate.reshape(1, -1)
    centered = b - b.mean(-1, keepdim=True)
    t = F.linear(centered * mlp_scale.reshape(1, -1), win)
    return b, t, (gamma @ b).unsqueeze(0), (gamma @ t).unsqueeze(0)
