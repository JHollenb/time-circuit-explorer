"""Independent scalar sums expose both the retained diagonal and omitted terms."""
import math
import torch
from diagonal_key_norm import compile_diagonal, evaluate


def test_source_diagonal_and_omitted_cross_energy_identity():
    weight = torch.tensor([[1., 2., -1.], [3., -2., 4.], [2., 0., 1.], [1., -3., 2.]], dtype=torch.float64)
    x = torch.tensor([[1., 2., 3.], [-2., 1., .5]], dtype=torch.float64)
    d = compile_diagonal(weight, heads=2, head_width=2)
    expected = torch.tensor([[sum(float(weight[h*2+c, a])**2 for c in range(2))/2 for a in range(3)] for h in range(2)], dtype=torch.float64)
    assert torch.equal(d, expected)
    calls = []
    def mm(a, b, label):
        calls.append((tuple(a.shape), tuple(b.shape), label))
        return a @ b
    got = evaluate(x, d, mm)
    for j in range(2):
        for h in range(2):
            diagonal = sum(float(d[h, a]) * float(x[j, a])**2 for a in range(3))
            exact = sum(sum(float(weight[2*h+c, a] * x[j, a]) for a in range(3))**2 for c in range(2))/2
            cross = sum(float(x[j, a] * x[j, b]) * sum(float(weight[2*h+c, a] * weight[2*h+c, b]) for c in range(2))/2
                        for a in range(3) for b in range(3) if a != b)
            assert math.isclose(float(got['energy'][j, h]), diagonal)
            assert math.isclose(float(got['norm'][j, h]), math.sqrt(diagonal + 1e-6))
            assert math.isclose(exact - diagonal, cross)
    assert calls == [((2, 3), (3, 2), 'diagonal-key-energy')]


def test_zero_input_has_declared_positive_epsilon_norm():
    d = compile_diagonal(torch.eye(4, dtype=torch.float64), heads=2, head_width=2)
    result = evaluate(torch.zeros((3, 4), dtype=torch.float64), d, lambda a, b, label: a @ b)
    assert torch.equal(result['energy'], torch.zeros((3, 2), dtype=torch.float64))
    torch.testing.assert_close(result['norm'], torch.full((3, 2), .001, dtype=torch.float64))
