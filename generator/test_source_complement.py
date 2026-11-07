"""Tiny matched-complement and sequential-source arithmetic fixtures."""
import torch
from test_selected_source import fixture
from selected_source_v2 import form
from source_complement import selected, source


def test_selected_operands_and_full_complement_reproduce_all_selected_outputs():
    weights, native, (image, text), (im, tm) = fixture()
    support = [0, 2]
    rope = (torch.ones(8, 4), torch.zeros(8, 4))
    complete = form(image, text, im, tm, rope, support, weights, native)
    fresh = selected(text, tm, rope, support, weights, native)
    actual = source(fresh, complete['K'], complete['V'], complete['A'], support, weights, native)
    for name in ['A', 'U', 'X', 'M', 'T']:
        torch.testing.assert_close(actual[name], complete[name], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual['PQP'], complete['T'], rtol=1e-12, atol=1e-12)


def test_complement_substitution_keeps_identical_selected_operands_and_uses_own_mlp():
    weights, native, (image, text), (im, tm) = fixture()
    support = [0, 2]
    rope = (torch.ones(8, 4), torch.zeros(8, 4))
    p = form(image, text, im, tm, rope, support, weights, native)
    shift = torch.linspace(-.4, .3, 8, dtype=torch.float64)
    q = form(image + shift, text - shift, im, tm, rope, support, weights, native)
    fresh = selected(text, tm, rope, support, weights, native)
    before = {n: x.clone() for n, x in fresh.items()}
    primary = source(fresh, q['K'], q['V'], q['A'], support, weights, native)
    control = source(fresh, p['K'], p['V'], q['A'], support, weights, native)
    for name in ['K', 'V']:
        for arm in [primary, control]:
            torch.testing.assert_close(arm[name + '_hybrid'][:, support], fresh[name], atol=0, rtol=0)
    for name, value in fresh.items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert not torch.allclose(primary['M'], control['M'])
    mod = tm[:, None, :].chunk(6, -1)
    for arm in [primary, control]:
        torch.testing.assert_close(arm['PQP'], (fresh['R'] + mod[2] * q['A']) + mod[5] * arm['M'])
