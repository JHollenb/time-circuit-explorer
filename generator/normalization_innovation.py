"""Shared real-arithmetic LayerNorm-affine secant, finite commits explicit."""
IN = 'transformer_blocks.0.ff.linear_in.weight'
DOWN = 'transformer_blocks.0.ff.linear_out.weight'
STAGES = ('fused', 'product', 'mlpWrite', 'gatedMLP', 'imageExit')


def make_selected(torch, rows=3072, width=9216):
    model = torch.nn.Module()
    block = torch.nn.Module()
    block.ff = torch.nn.Module()
    block.ff.linear_in = torch.nn.Linear(rows, 2 * width, bias=False)
    block.ff.linear_out = torch.nn.Linear(width, rows, bias=False)
    model.transformer_blocks = torch.nn.ModuleList([block])
    return model


def shared(weight, scale, shift, carrier_a, carrier_b):
    import torch
    if weight.dtype != torch.float64 or weight.device.type != 'cpu':
        raise ValueError('Shared projection requires CPU FP64 original matrix')
    delta = carrier_b.to(torch.float64) - carrier_a.to(torch.float64)
    centered = delta - delta.mean(-1, keepdim=True)
    j = torch.nn.functional.linear(scale.to(torch.float64) * centered, weight)
    beta = torch.nn.functional.linear(shift.to(torch.float64), weight)
    return delta, centered, j, beta


def standard_deviation(residual, epsilon=1e-6):
    import torch
    value = residual.to('cpu', torch.float64)
    centered = value - value.mean(-1, keepdim=True)
    return (centered.square().mean(-1, keepdim=True) + epsilon).sqrt()


def predict(fused_a, j, beta, sa, sb):
    import torch
    alpha = sa / sb
    raw = alpha * (fused_a.to(torch.float64) - beta) + beta + j / sb
    if not torch.isfinite(raw).all():
        raise ValueError('Nonfinite normalization-innovation candidate')
    return raw, raw.to(torch.float32).to(torch.bfloat16), alpha
