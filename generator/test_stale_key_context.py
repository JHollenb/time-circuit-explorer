"""Independent fixed-allocation payload law and four-corner interaction."""
import torch
from test_propagated_context import setup
from propagated_context import banks, selected, advance, next_query
from stale_key_context_worker import CORNERS


def test_crossed_corners_obey_fixed_query_attention_interaction():
    _, w, native, (image, text), (im, tm), rope = setup()
    support = [0, 2];p = banks(image, text, im, tm, rope, w, native)
    direction = torch.linspace(-.5, .4, 8, dtype=torch.float64)
    q = banks(image + direction, text - direction, im, tm, rope, w, native)
    shared = selected(text, tm, rope, p, support, w, native);bank = {'P': p, 'Q': q}
    arms = {a+b: advance(shared, bank[a]['K'], bank[b]['V'], support, w, native) for a in 'PQ' for b in 'PQ'}
    query = shared['Q'].transpose(1, 2)
    probabilities = {a: torch.softmax(query @ arms[a+'P']['K_hybrid'].transpose(1, 2).transpose(-1, -2) / 2, -1) for a in 'PQ'}
    delta_v = (arms['PQ']['V_hybrid'] - arms['PP']['V_hybrid']).transpose(1, 2)
    expected = ((probabilities['Q']-probabilities['P']) @ delta_v).transpose(1, 2).flatten(2, 3)
    actual = arms['QQ']['H'] - arms['QP']['H'] - arms['PQ']['H'] + arms['PP']['H']
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    value_effect = (probabilities['P'] @ delta_v).transpose(1, 2).flatten(2, 3)
    torch.testing.assert_close(arms['PQ']['H']-arms['PP']['H'], value_effect, atol=1e-12, rtol=1e-12)


def test_prespecified_stale_key_candidate_advances_own_state():
    assert CORNERS == {'primary': ('P', 'Q'), 'diagnostic': ('Q', 'P')}
    _, w, native, (image, text), (im, tm), rope = setup()
    support = [0, 2];p = banks(image, text, im, tm, rope, w, native)
    direction = torch.linspace(-.5, .4, 8, dtype=torch.float64)
    q = banks(image + direction, text - direction, im, tm, rope, w, native)
    shared = selected(text, tm, rope, p, support, w, native)
    got = advance(shared, p['K'], q['V'], support, w, native)
    full = advance(shared, q['K'], q['V'], support, w, native)
    own = next_query(got['T'], tm, tuple(x[support] for x in rope), w, native)
    target = next_query(full['T'], tm, tuple(x[support] for x in rope), w, native)
    assert not torch.allclose(own['nextQ'], target['nextQ'])
    for n in ['K', 'V']:
        torch.testing.assert_close(got[n+'_hybrid'][:, support], shared[n], atol=0, rtol=0)
