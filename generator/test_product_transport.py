"""Independent numerical boundary, conditional response and quarantine checks."""
import json
from pathlib import Path
import subprocess
import sys
import math
import pytest
import torch
from product_transport import transport, path, make_down, DOWN


def test_explicit_fp32_intermediate_retains_its_rounding_boundary():
    a = torch.tensor([[-2.0 ** -30, 0.]], dtype=torch.bfloat16)
    b = torch.tensor([[2.0 ** -8, 0.]], dtype=torch.bfloat16)
    c = torch.tensor([[1., 0.]], dtype=torch.bfloat16)
    raw, quant = transport(a, b, c)
    assert raw[0, 0].item() == 1 + 2.0 ** -8 + 2.0 ** -30
    assert raw[0, 0].item() > raw.float()[0, 0].item() == 1 + 2.0 ** -8
    assert quant[0, 0].item() == 1.
    with pytest.raises(ValueError):
        transport(a.float(), b, c)
    with pytest.raises(ValueError):
        transport(a, b[:, :1], c)


def test_native_nonlinearity_uses_recipient_baseline_before_one_down():
    a = torch.tensor([[[-1., 1.]]], dtype=torch.bfloat16)
    b = torch.tensor([[[-.5, 1.25]]], dtype=torch.bfloat16)
    c = torch.tensor([[[2., 3.]]], dtype=torch.bfloat16)
    raw, candidate = transport(a, b, c)
    assert candidate.tolist() == [[[2.5, 3.25]]]
    class Nonlinear(torch.nn.Module):
        def forward(self, x):
            g, u = x.chunk(2, -1)
            return torch.nn.functional.silu(g) * u
    model = make_down(torch, 1, 1).to(dtype=torch.bfloat16)
    assert set(dict(model.named_parameters())) == {DOWN}
    with torch.no_grad():
        model.transformer_blocks[0].ff.linear_out.weight.fill_(2.)
    e = torch.tensor([[[1.]]], dtype=torch.bfloat16)
    carrier = torch.tensor([[[.25]]], dtype=torch.bfloat16)
    gate = torch.tensor([[[.5]]], dtype=torch.bfloat16)
    seen = {}
    with torch.inference_mode():
        y = path(candidate, e, carrier, gate, Nonlinear(), model.transformer_blocks[0].ff.linear_out, lambda n, v: seen.setdefault(n, v.clone()))
    bf = lambda x: torch.tensor(x, dtype=torch.bfloat16).item()
    product = bf(bf(2.5 / (1 + math.exp(-2.5))) * 3.25)
    expected = bf(bf(1. + .25) + bf(.5 * bf(2. * product)))
    assert y.item() == expected
    source_change = (Nonlinear()(b) - Nonlinear()(a)).item()
    recipient_change = (Nonlinear()(candidate) - Nonlinear()(c)).item()
    assert abs(source_change - recipient_change) > .5
    assert list(seen) == ['product', 'mlpWrite', 'gatedMLP', 'residualCommit', 'imageExit']


def test_target_and_undeclared_array_quarantine_in_fresh_process(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    allowed = tmp_path / 'allowed.npy'
    allowed.write_bytes(b'supplied')
    target = tmp_path / 'target.npy'
    target.write_bytes(b'not admitted')
    scoring = root / 'targets.json'
    scoring.write_text('{}')
    script = f'''from product_transport import install_guard
import json
access=install_guard({str(root)!r},[{str(allowed)!r}],[{str(tmp_path)!r}],[{str(scoring)!r}])
assert open({str(allowed)!r},'rb').read()==b'supplied'
for name in [{str(target)!r},{str(scoring)!r}]:
 try:open(name,'rb')
 except PermissionError:pass
 else:raise AssertionError('Target admitted')
print(json.dumps(access))
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).parent, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == {str(allowed.resolve()): 1}


def test_nonfinite_source_increment_is_rejected_before_native_path():
    zero = torch.zeros((1, 2), dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        transport(zero, torch.full_like(zero, float('inf')), zero)
