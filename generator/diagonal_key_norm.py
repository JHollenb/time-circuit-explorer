"""A source-defined positive quadratic; discarded cross terms remain explicit."""


def compile_diagonal(weight, heads=24, head_width=128):
    import torch
    if weight.ndim != 2 or weight.shape[0] != heads * head_width or weight.dtype != torch.float64 or weight.device.type != 'cpu':
        raise ValueError('Source-only CPU FP64 weight geometry differs')
    return weight.reshape(heads, head_width, -1).square().mean(1)


def evaluate(x, diagonal, mm):
    import torch
    if x.ndim != 2 or diagonal.ndim != 2 or x.shape[1] != diagonal.shape[1] or x.dtype != torch.float64 or diagonal.dtype != torch.float64 or x.device.type != 'cpu' or diagonal.device.type != 'cpu':
        raise ValueError('Own-input positive quadratic geometry/profile differs')
    energy = mm(x.square(), diagonal.T, 'diagonal-key-energy')
    norm = (energy + 1e-6).sqrt()
    if not bool(torch.isfinite(norm).all() and (norm > 0).all()):
        raise ValueError('Invalid positive normalization')
    return {'energy': energy, 'norm': norm}
