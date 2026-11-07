"""One fixed recipient-baseline transport and source-ordered native commit."""
import os
from pathlib import Path
import sys

DOWN = 'transformer_blocks.0.ff.linear_out.weight'
STAGES = ('product', 'mlpWrite', 'gatedMLP', 'imageExit')


def transport(source_a, source_b, recipient_a):
    import torch
    fields = (source_a, source_b, recipient_a)
    if any(x.shape != source_a.shape or x.dtype != torch.bfloat16 or x.device.type != 'cpu' for x in fields) or source_a.shape[-1] % 2:
        raise ValueError('Transport requires matching even-width CPU BF16 operands')
    raw = recipient_a.to(torch.float64) + (source_b.to(torch.float64) - source_a.to(torch.float64))
    if not torch.isfinite(raw).all():
        raise ValueError('Nonfinite transported operand')
    return raw, raw.to(torch.float32).to(torch.bfloat16)


def make_down(torch, rows=3072, width=9216):
    model = torch.nn.Module()
    block = torch.nn.Module()
    block.ff = torch.nn.Module()
    block.ff.linear_out = torch.nn.Linear(width, rows, bias=False)
    model.transformer_blocks = torch.nn.ModuleList([block])
    return model


def path(fused, embedding, carrier, gate, nonlinear, down, capture):
    product = nonlinear(fused)
    capture('product', product)
    write = down(product)
    capture('mlpWrite', write)
    gated = gate * write
    capture('gatedMLP', gated)
    residual = embedding + carrier
    capture('residualCommit', residual)
    result = residual + gated
    capture('imageExit', result)
    return result


def install_guard(root, allowed, protected, denied):
    root = Path(root).resolve()
    allowed = {str(Path(p).resolve()) for p in allowed}
    protected = [Path(p).resolve() for p in protected]
    denied = {str(Path(p).resolve()) for p in denied}
    accesses = {}
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        key = str(p)
        if key in denied:
            raise PermissionError('Scoring target unavailable to primary')
        if key in allowed:
            accesses[key] = accesses.get(key, 0) + 1
        elif not p.is_relative_to(root) and (p.suffix in {'.npy', '.npz', '.safetensors', '.pt', '.pth'} or any(p.is_relative_to(x) for x in protected)):
            raise PermissionError('Undeclared primary numerical input')
    sys.addaudithook(audit)
    return accesses
