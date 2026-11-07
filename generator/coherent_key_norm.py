"""One fixed signed output sketch; no target-dependent rank, seed or gain."""
SEED = 20261107
RANK = 16


def fixed_signs(heads=24, rank=RANK, head_width=128):
    import numpy as np
    rows = []
    for h in range(heads):
        bits = np.random.PCG64(np.random.SeedSequence([SEED, h])).random_raw(rank * head_width) & 1
        rows.append(np.where(bits, 1, -1).astype(np.int8).reshape(rank, head_width))
    return np.stack(rows)


def compile_sketch(weight, signs, mm, heads=24, head_width=128):
    import torch
    if weight.ndim != 2 or weight.shape[0] != heads * head_width or weight.dtype != torch.float64 or weight.device.type != 'cpu':
        raise ValueError('Original CPU FP64 weight geometry differs')
    if signs.ndim != 3 or signs.shape[0] != heads or signs.shape[2] != head_width or signs.dtype != torch.int8 or signs.device.type != 'cpu' or not bool(((signs == 1) | (signs == -1)).all()):
        raise ValueError('Pinned signed output geometry differs')
    return mm(signs.double(), weight.reshape(heads, head_width, -1), 'signed-output-compile')


def evaluate(x, coefficients, mm, head_width=128):
    import torch
    if x.ndim != 2 or coefficients.ndim != 3 or x.shape[1] != coefficients.shape[2] or x.dtype != torch.float64 or coefficients.dtype != torch.float64 or x.device.type != 'cpu' or coefficients.device.type != 'cpu':
        raise ValueError('Own-input coherent norm geometry/profile differs')
    heads, rank, width = coefficients.shape
    projection = mm(x, coefficients.reshape(heads * rank, width).T, 'sketched-key-projection').reshape(x.shape[0], heads, rank)
    energy = projection.square().sum(-1) / (rank * head_width)
    norm = (energy + 1e-6).sqrt()
    if not bool(torch.isfinite(norm).all() and (norm > 0).all()):
        raise ValueError('Invalid positive normalization')
    return {'projection': projection, 'energy': energy, 'norm': norm}
