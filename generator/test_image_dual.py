"""Small independent identities, including nonorthogonal stored RoPE factors."""
import torch
from image_dual import compile_duals, image_groups, image_numerators, transpose_rope, consume_head


def test_transpose_uses_actual_unequal_stored_factors():
    q = torch.tensor([2., 3., 5., 7.], dtype=torch.float64)
    c = torch.tensor([.8, .7, .4, .3], dtype=torch.float64)
    s = torch.tensor([.2, .1, .9, .6], dtype=torch.float64)
    matrix = torch.tensor([[.8, -.2, 0, 0], [.1, .7, 0, 0],
                           [0, 0, .4, -.9], [0, 0, .6, .3]], dtype=torch.float64)
    torch.testing.assert_close(transpose_rope(q, c, s), matrix.T @ q)


def test_grouped_duals_match_independent_explicit_rotary_matrices():
    torch.manual_seed(58)
    q, weight, gamma, x = [torch.randn(shape, dtype=torch.float64) for shape in [(3, 2, 8), (16, 6), (8,), (4, 6)]]
    c, s = torch.randn((4, 8), dtype=torch.float64), torch.randn((4, 8), dtype=torch.float64)
    for v in [c, s]:
        v[:, :2] = v[0, :2].clone();v[:, 6:8] = v[0, 6:8].clone()
        v[1, 2:4] = v[0, 2:4];v[3, 2:4] = v[2, 2:4]
        v[2, 4:6] = v[0, 4:6];v[3, 4:6] = v[1, 4:6]
    ids, _ = image_groups(c, s, side=2, axis_width=2)
    duals = compile_duals(q, weight, gamma, c, s, lambda a, b, label: a @ b, side=2)
    actual = image_numerators(x, duals, ids, lambda a, b, label: a @ b)
    expected = torch.empty((2, 3, 4), dtype=torch.float64)
    for h in range(2):
        for j in range(4):
            rotation = torch.zeros((8, 8), dtype=torch.float64)
            for k in range(0, 8, 2):
                rotation[k:k+2, k:k+2] = torch.tensor([[c[j, k], -s[j, k]], [s[j, k+1], c[j, k+1]]], dtype=torch.float64)
            key = rotation @ (gamma * (weight[h*8:(h+1)*8] @ x[j]))
            expected[h, :, j] = q[:, h] @ key
    torch.testing.assert_close(actual, expected)
    broken = c.clone();broken[1, 0] += .1
    import pytest
    with pytest.raises(ValueError, match='membership'):
        image_groups(broken, s, side=2, axis_width=2)


def test_receiving_law_matches_frozen_native_consumer_after_same_head():
    from propagated_context import advance, P1
    class Native:
        @staticmethod
        def dispatch_attention_fn(q, k, v, **kw):
            return torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4) / 17
        @staticmethod
        def Flux2SwiGLU():
            def swiglu(x):
                gate, up = x.chunk(2, -1)
                return torch.nn.functional.silu(gate) * up
            return swiglu
    torch.manual_seed(59)
    weights = {P1 + 'attn.to_add_out.weight': torch.randn(8, 8),
               P1 + 'ff_context.linear_in.weight': torch.randn(24, 8),
               P1 + 'ff_context.linear_out.weight': torch.randn(8, 12)}
    values = dict(R=torch.randn(1, 2, 8), txtMod=torch.randn(1, 48), Q=torch.randn(1, 2, 2, 4),
                  K=torch.randn(1, 2, 2, 4), V=torch.randn(1, 2, 2, 4))
    bank = torch.randn(1, 5, 2, 4)
    old = advance(values, bank, bank, [1, 3], weights, Native)
    new = consume_head(values, old['H'], weights, Native)
    for name, tensor in new.items():
        assert torch.equal(tensor, old[name]), name


def test_byte_only_text_slice_matches_independent_array_prefix(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys
    import numpy as np
    from image_dual_intake import SLICE
    original = (np.arange(1536 * 24 * 128, dtype=np.uint32) % 65521).astype(np.uint16).reshape(1, 1536, 24, 128)
    path = tmp_path / 'full.npy';np.save(path, original, allow_pickle=False)
    digest = lambda data: hashlib.sha256(data).hexdigest()
    source = dict(path=str(path), shape=list(original.shape), logicalDtype='torch.bfloat16',
                  sha256=digest(path.read_bytes()), payloadSHA256=digest(original.tobytes()))
    destination = tmp_path / 'text.npy'
    result = subprocess.run([sys.executable, '-c', SLICE],
        input=json.dumps({'source': source, 'destination': str(destination)}),
        text=True, capture_output=True, check=True)
    receipt = json.loads(result.stdout)
    np.testing.assert_array_equal(np.load(destination, allow_pickle=False), original[:, :512])
    assert receipt['payloadSHA256'] == digest(original[:, :512].tobytes())
    assert receipt['selectedPayloadBytes'] == 512 * 24 * 128 * 2
