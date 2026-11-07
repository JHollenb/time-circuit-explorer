"""Tiny independent full-query oracle and changed-context own-query fixture."""
import torch
import torch.nn.functional as F
from test_selected_source import fixture
from selected_source_v2 import form
from propagated_context import P1, P2, banks, selected, advance, next_query


def setup():
    old, native, states, mods = fixture()
    weights = {n.replace('transformer_blocks.0.', P1): x for n, x in old.items()}
    weights[P2 + 'attn.add_q_proj.weight'] = old['transformer_blocks.0.attn.add_q_proj.weight'].flip(0).clone()
    weights[P2 + 'attn.norm_added_q.weight'] = old['transformer_blocks.0.attn.norm_added_q.weight'].clone()
    return old, weights, native, states, mods, (torch.ones(8, 4), torch.zeros(8, 4))


def test_full_complement_matches_independent_full_source_and_next_query():
    old, w, native, (image, text), (im, tm), rope = setup()
    support = [0, 2]
    bank = banks(image, text, im, tm, rope, w, native)
    values = selected(text, tm, rope, bank, support, w, native)
    got = advance(values, bank['K'], bank['V'], support, w, native)
    ref = form(image, text, im, tm, rope, support, old, native)
    for n in ['K', 'V']:
        torch.testing.assert_close(bank[n], ref[n], atol=1e-12, rtol=1e-12)
    for n in ['A', 'U', 'X', 'M', 'T']:
        torch.testing.assert_close(got[n], ref[n], atol=1e-12, rtol=1e-12)
    actual = next_query(got['T'], tm, tuple(x[support] for x in rope), w, native)
    mod = tm[:, None].chunk(6, -1)
    x = (1 + mod[1]) * F.layer_norm(ref['T'], (8,), eps=1e-6) + mod[0]
    q = F.linear(x, w[P2 + 'attn.add_q_proj.weight']).unflatten(-1, (2, 4))
    q = F.rms_norm(q, (4,), w[P2 + 'attn.norm_added_q.weight'], eps=1e-6)
    torch.testing.assert_close(actual['nextQ'], q, atol=1e-12, rtol=1e-12)


def test_changed_complement_advances_own_mlp_and_next_query_without_input_mutation():
    _, w, native, (image, text), (im, tm), rope = setup()
    support = [0, 2]
    p = banks(image, text, im, tm, rope, w, native)
    direction = torch.linspace(-.4, .3, 8, dtype=torch.float64)
    q = banks(image + direction, text - direction, im, tm, rope, w, native)
    values = selected(text, tm, rope, p, support, w, native)
    before = {n: x.clone() for n, x in values.items()}
    arms = [advance(values, bank['K'], bank['V'], support, w, native) for bank in [q, p]]
    queries = [next_query(a['T'], tm, tuple(x[support] for x in rope), w, native) for a in arms]
    assert not torch.allclose(arms[0]['M'], arms[1]['M'])
    assert not torch.allclose(queries[0]['nextQ'], queries[1]['nextQ'])
    for a in arms:
        for n in ['K', 'V']:
            torch.testing.assert_close(a[n + '_hybrid'][:, support], values[n], atol=0, rtol=0)
        torch.testing.assert_close(a['T'], a['U'] + a['D_mlp'], atol=0, rtol=0)
    for n, x in before.items():
        torch.testing.assert_close(values[n], x, atol=0, rtol=0)


def test_selected_queries_keep_absolute_rope_positions_across_both_layers():
    old, w, native, (image, text), (im, tm), _ = setup()
    support = [0, 2]
    rope = (torch.arange(32, dtype=torch.float64).reshape(8, 4) / 29 + .2,
            torch.arange(32, dtype=torch.float64).reshape(8, 4) / 37)
    native.apply_rotary_emb = lambda x, factors, **kw: x * factors[0][None, :, None] + x.flip(-1) * factors[1][None, :, None]
    bank = banks(image, text, im, tm, rope, w, native)
    actual = selected(text, tm, rope, bank, support, w, native)
    ref = form(image, text, im, tm, rope, support, old, native)
    torch.testing.assert_close(bank['K'], ref['K'], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual['Q'], ref['Q'], atol=1e-12, rtol=1e-12)
    rows = text[:, support]
    result = next_query(rows, tm, tuple(x[support] for x in rope), w, native)
    unrotated = result['nextQNormalized']
    expected = unrotated * rope[0][support][None, :, None] + unrotated.flip(-1) * rope[1][support][None, :, None]
    torch.testing.assert_close(result['nextQ'], expected, atol=0, rtol=0)


def test_actual_selected_act_does_not_inherit_full_rope_output_geometry():
    from mdb.titan import Frame, ActSpec, ActRegistry
    from propagated_context_worker import output_abis
    _, w, native, (image, text), (im, tm), rope = setup()
    bank = banks(image, text, im, tm, rope, w, native)
    values = {'text': text, 'txtMod': tm, 'fullRopeCos': rope[0], 'fullRopeSin': rope[1], 'bankK': bank['K'], 'bankV': bank['V']}
    before = Frame.from_values(values, kinds={n: 'tensor' for n in values},
        abis={n: {'shape': list(x.shape), 'stride': list(x.stride()), 'dtype': str(x.dtype)} for n, x in values.items()})
    names = ('R', 'txtMod', 'Q', 'K', 'V', 'ropeCos', 'ropeSin')
    spec = ActSpec('test.propagated_selected', 'v1', reads=tuple(values), writes=names, write_kinds={n: 'tensor' for n in names},
                   write_abis=output_abis('selected', values, names, [0, 2]))
    registry = ActRegistry()
    def operation(v):
        return selected(v['text'], v['txtMod'], (v['fullRopeCos'], v['fullRopeSin']),
                        {'K': v['bankK'], 'V': v['bankV']}, [0, 2], w, native), {}
    registry.register(spec, 'selected', operation)
    after, _ = registry.run(spec.identity, 'selected', before)
    for name in ['ropeCos', 'ropeSin']:
        assert list(after.values[name].shape) == [2, 4]
        assert after.to_dict()['slots'][name]['abi']['shape'] == [2, 4]
