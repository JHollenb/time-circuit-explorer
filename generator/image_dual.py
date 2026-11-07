"""Query-specific image-score factorization; original learned maps remain paid."""


def transpose_rope(q, cosine, sine):
    """Transpose the actual stored two-channel map, without assuming orthogonality."""
    import torch
    a = q[..., 0::2] * cosine[..., 0::2] + q[..., 1::2] * sine[..., 1::2]
    b = q[..., 1::2] * cosine[..., 1::2] - q[..., 0::2] * sine[..., 0::2]
    return torch.stack((a, b), -1).flatten(-2)


def image_groups(cosine, sine, side=32, axis_width=32):
    """Authenticate the declared grid against stored factor bytes before grouping."""
    import torch
    if cosine.shape != sine.shape or tuple(cosine.shape) != (side * side, 4 * axis_width):
        raise ValueError('Declared four-axis image geometry differs')
    indices = torch.arange(side * side, dtype=torch.int64)
    ids = torch.stack((indices // side, indices % side), -1)
    for values in (cosine, sine):
        for axis in range(4):
            block = values[:, axis * axis_width:(axis + 1) * axis_width]
            expected = block[0].expand_as(block) if axis in [0, 3] else block[
                (ids[:, 0] * side) if axis == 1 else ids[:, 1]]
            if not torch.equal(block, expected):
                raise ValueError('Stored position factors violate declared group membership')
    # Uniqueness is checked jointly across cosine/sine, not from rounded cosines alone.
    for axis in [1, 2]:
        rows = torch.arange(side) * side if axis == 1 else torch.arange(side)
        block = torch.cat((cosine[rows, axis * axis_width:(axis + 1) * axis_width],
                           sine[rows, axis * axis_width:(axis + 1) * axis_width]), -1)
        if len(torch.unique(block, dim=0)) != side:
            raise ValueError('Declared varying position groups are not distinct')
    return ids, {'side': side, 'axisWidth': axis_width, 'constantAxes': [0, 3],
                 'varyingAxes': [1, 2], 'axisOrder': ['T', 'H', 'W', 'L'], 'groupsPerVaryingAxis': side,
                 'storedCoefficientGroups': 1 + 2 * side,
                 'membership': 'Every stored cosine/sine row byte-exact to its grid representative'}


def compile_duals(q, weight, gamma, cosine, sine, mm, side=32):
    """All operands are CPU FP64; outputs [query, head, 1+2*side, input]."""
    import torch
    queries, heads, width = q.shape
    axis = width // 4
    if width % 8 or weight.shape[0] != heads * width or gamma.shape != (width,):
        raise ValueError('Dual coefficient geometry differs')
    table = torch.empty((queries, heads, 1 + 2 * side, weight.shape[1]), dtype=torch.float64)
    constant_indices = torch.cat((torch.arange(axis), torch.arange(3 * axis, 4 * axis)))
    for h in range(heads):
        w = weight[h * width:(h + 1) * width]
        constant = transpose_rope(q[:, h, constant_indices], cosine[0, constant_indices], sine[0, constant_indices])
        table[:, h, 0] = mm(constant * gamma[constant_indices], w[constant_indices], 'dual-constant')
        for a, rows, start in [(1, torch.arange(side) * side, 1), (2, torch.arange(side), 1 + side)]:
            sl = slice(a * axis, (a + 1) * axis)
            coefficients = transpose_rope(q[None, :, h, sl], cosine[rows, sl][:, None], sine[rows, sl][:, None])
            coefficients = coefficients * gamma[sl]
            result = mm(coefficients.reshape(side * queries, axis), w[sl], 'dual-axis-' + str(a))
            table[:, h, start:start + side] = result.reshape(side, queries, -1).permute(1, 0, 2)
    return table


def raw_projection(x, weight, heads, mm, label):
    raw = mm(x, weight.T, label).reshape(x.shape[0], heads, -1)
    denominator = (raw.square().mean(-1) + 1e-6).sqrt()
    return raw, denominator


def image_numerators(x, duals, group_ids, mm):
    """Three grouped dot terms, never a query×head×row×input tensor."""
    import torch
    queries, heads, groups, width = duals.shape
    side = (groups - 1) // 2
    if groups != 1 + 2 * side or tuple(group_ids.shape) != (len(x), 2) or x.shape[1] != width:
        raise ValueError('Actor grouped input geometry differs')
    result = mm(duals[:, :, 0].reshape(queries * heads, width), x.T, 'image-constant').reshape(queries, heads, len(x))
    for axis, start in [(0, 1), (1, 1 + side)]:
        for group in range(side):
            rows = torch.nonzero(group_ids[:, axis] == group).flatten()
            part = mm(duals[:, :, start + group].reshape(queries * heads, width),
                      x[rows].T, 'image-axis-' + str(axis))
            result[:, :, rows] += part.reshape(queries, heads, len(rows))
    return result.permute(1, 0, 2).contiguous()


def smooth_rotated_keys(raw, gamma, cosine, sine):
    """Smooth FP64 projection/RMS/RoPE law, without native BF16 boundaries."""
    import torch
    denominator = (raw.square().mean(-1) + 1e-6).sqrt()
    z = raw * gamma / denominator[..., None]
    rotated = torch.stack((-z[..., 1::2], z[..., 0::2]), -1).flatten(-2)
    return z * cosine[:, None] + rotated * sine[:, None], denominator


def read_head(logits, values, mm, label='full-AV'):
    import torch
    peak = logits.max(-1, keepdim=True).values
    weights = torch.exp(logits - peak)
    partition = weights.sum(-1, keepdim=True)
    probability = weights / partition
    if not bool(torch.isfinite(partition).all() and (partition > 0).all()):
        raise ValueError('Invalid stable attention partition')
    return {'peak': peak, 'partition': partition, 'probability': probability,
            'H_FP64': mm(probability, values, label)}


def consume_head(values, head, weights, native):
    """Unchanged finite receiving law from propagated_context.advance, after H."""
    import torch.nn.functional as F
    from propagated_context import P1
    a = F.linear(head, weights[P1 + 'attn.to_add_out.weight'])
    m = values['txtMod'][:, None, :].chunk(6, -1)
    da = m[2] * a
    u = values['R'] + da
    x = F.layer_norm(u, (u.shape[-1],), eps=1e-6) * (1 + m[4]) + m[3]
    fused = F.linear(x, weights[P1 + 'ff_context.linear_in.weight'])
    product = native.Flux2SwiGLU()(fused)
    returned = F.linear(product, weights[P1 + 'ff_context.linear_out.weight'])
    dm = m[5] * returned
    return dict(H=head, A=a, D_attention=da, U=u, X=x, fused=fused,
                product=product, M=returned, D_mlp=dm, T=u + dm)
