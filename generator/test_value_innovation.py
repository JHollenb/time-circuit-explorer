import types
import numpy as np
import pytest
import torch
from value_innovation import acquire, event, compile_event, AFFINES


def test_finite_swap_matches_independent_full_attention():
    rng = np.random.default_rng(72)
    q = rng.normal(size=(1, 7, 3, 4))
    k = rng.normal(size=(1, 11, 3, 4))
    v = rng.normal(size=k.shape)
    scores = np.einsum('bihd,bjhd->bhij', q, k) / 2
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    swapped = v.copy()
    swapped[:, [2, 9]] = swapped[:, [9, 2]]
    expected = np.einsum('bhij,bjhd->bihd', p, swapped) - np.einsum('bhij,bjhd->bihd', p, v)
    gamma, directions, selected = event(*[torch.from_numpy(x) for x in (q, k, v)], 2, 9)
    actual = gamma[:, :, None] * directions[None, :, :]
    np.testing.assert_allclose(actual.numpy(), expected[0], rtol=1e-12, atol=1e-14)
    assert selected.shape == (7, 3, 2)
    assert np.any(selected.sum(-1).numpy() < .9)


def test_physical_head_blocks_match_full_linear_action_and_secant():
    rng = np.random.default_rng(61)
    wo, win = [torch.from_numpy(rng.normal(size=s)) for s in [(8, 12), (10, 8)]]
    gate, scale = [torch.from_numpy(rng.normal(size=8)) for _ in range(2)]
    gamma = torch.from_numpy(rng.normal(size=(5, 3)))
    directions = torch.from_numpy(rng.normal(size=(3, 4)))
    b, t, delta, j = compile_event(wo, win, gate, scale, gamma, directions)
    full_head = (gamma[:, :, None] * directions[None, :, :]).reshape(5, 12).numpy()
    expected_delta = (full_head @ wo.numpy().T) * gate.numpy()
    expected_j = ((expected_delta - expected_delta.mean(-1, keepdims=True)) * scale.numpy()) @ win.numpy().T
    np.testing.assert_allclose(delta[0], expected_delta, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(j[0], expected_j, rtol=1e-12, atol=1e-13)
    assert b.shape == (3, 8) and t.shape == (3, 10)
    a = rng.normal(size=(5, 8)); shift = rng.normal(size=8)
    normalize = lambda x: (x-x.mean(-1, keepdims=True))/np.sqrt(x.var(-1, keepdims=True)+1e-6)
    fa = (normalize(a)*scale.numpy()+shift) @ win.numpy().T
    sb = np.sqrt((a+expected_delta).var(-1, keepdims=True)+1e-6)
    sa = np.sqrt(a.var(-1, keepdims=True)+1e-6)
    beta = shift @ win.numpy().T
    prediction = sa/sb*(fa-beta)+beta+expected_j/sb
    np.testing.assert_allclose(prediction, (normalize(a+expected_delta)*scale.numpy()+shift)@win.numpy().T, rtol=1e-11, atol=1e-12)


def test_acquisition_stops_before_attention_and_forms_seven_affines(monkeypatch):
    generator = torch.Generator().manual_seed(51)
    sizes = [(6, 2), (6, 3)] + [(6, 6)] * 5
    weights = {n: torch.randn(s, generator=generator) for n, s in zip(AFFINES, sizes)}
    calls = []
    original = torch.nn.functional.linear
    def linear(x, w, bias=None):
        calls.append(tuple(w.shape))
        return original(x, w, bias)
    monkeypatch.setattr(torch.nn.functional, 'linear', linear)
    native = types.SimpleNamespace(apply_rotary_emb=lambda x, *a, **kw: x)
    values = acquire(weights, torch.ones(1, 5, 2), torch.ones(1, 4, 3), torch.zeros(1, 5, 4), torch.zeros(1, 4, 4), torch.zeros(1, 36), torch.zeros(1, 36), {n: lambda x: x for n in ['norm_q', 'norm_k', 'norm_added_k']}, lambda ids: (ids, ids), native, 2)
    assert len(calls) == 7
    assert [tuple(x.shape) for x in values] == [(1, 5, 6), (1, 5, 2, 3), (1, 9, 2, 3), (1, 9, 2, 3)]


def test_invalid_addresses_and_nonfinite_inputs_are_refused():
    x = torch.ones(1, 3, 2, 4, dtype=torch.float64)
    with pytest.raises(ValueError):
        event(x, x, x, 1, 1)
    with pytest.raises(ValueError):
        event(x, x, x, 0, 3)
    with pytest.raises(ValueError):
        event(x.float(), x, x, 0, 1)
    x[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError):
        event(x, x, x, 0, 1)


def test_scheduler_inventory_adapter_preserves_native_and_wrapped_rows():
    from value_innovation_study import host_rows
    rows = [{'name': 'beast', 'ram_total_mb': 66512}]
    assert host_rows(rows) == rows
    assert host_rows({'hosts': rows}) == rows
    with pytest.raises(ValueError):
        host_rows({'hosts': 'missing'})


def test_manual_output_weight_receipt_matches_selected_loader_header_contract():
    import hashlib
    from value_innovation_primary_v2 import weight_record
    value = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    entries = {'Wo': {'data_offsets': [123, 135]}}
    expected = {'name': 'Wo', 'shape': [2, 3], 'bytes': 12, 'dataOffsets': [123, 135],
                'payloadSHA256': hashlib.sha256(value.view(torch.uint16).numpy().tobytes()).hexdigest()}
    assert weight_record('Wo', value, entries) == expected
