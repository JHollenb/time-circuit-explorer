import itertools
import math
import numpy as np
import torch
from coherent_key_norm import fixed_signs, compile_sketch, evaluate


def test_fixed_recipe_and_independent_scalar_projection():
    signs = fixed_signs(heads=2, rank=2, head_width=4)
    assert signs.tolist() == [[[1, 1, 1, 1], [-1, -1, 1, 1]], [[1, -1, 1, 1], [-1, 1, 1, 1]]]
    w = torch.tensor([[1., 2., -1.], [-2., .5, 3.], [1., -1., 2.], [.5, 4., -2.],
                      [2., -1., 3.], [1., 2., .5], [-3., 1., 2.], [4., -2., 1.]], dtype=torch.float64)
    x = torch.tensor([[1., -2., .5], [3., .25, -1.]], dtype=torch.float64)
    mm = lambda a, b, label: a @ b
    b = compile_sketch(w, torch.from_numpy(signs), mm, heads=2, head_width=4)
    expected_b = [[[sum(int(signs[h, k, c]) * w[h * 4 + c, a].item() for c in range(4)) for a in range(3)] for k in range(2)] for h in range(2)]
    assert torch.equal(b, torch.tensor(expected_b, dtype=torch.float64))
    got = evaluate(x, b, mm, head_width=4)
    for j, h, k in itertools.product(range(2), repeat=3):
        expected = sum(x[j, a].item() * expected_b[h][k][a] for a in range(3))
        assert got['projection'][j, h, k].item() == expected
    for j, h in itertools.product(range(2), repeat=2):
        expected = sum(got['projection'][j, h, k].item() ** 2 for k in range(2)) / 8
        assert got['energy'][j, h].item() == expected
        assert got['norm'][j, h].item() == math.sqrt(expected + 1e-6)


def test_exact_sign_ensemble_energy_variance_and_zero_input():
    z = np.array([1., -2., .5]);energies = np.array([(np.array(s) @ z) ** 2 / 3 for s in itertools.product([-1, 1], repeat=3)])
    assert np.isclose(energies.mean(), np.sum(z*z) / 3)
    assert np.isclose(energies.var(), 2 * (np.sum(z*z) ** 2 - np.sum(z**4)) / 9)
    result = evaluate(torch.zeros((2, 3), dtype=torch.float64), torch.ones((2, 2, 3), dtype=torch.float64), lambda a, b, label: a @ b, head_width=4)
    assert torch.equal(result['energy'], torch.zeros((2, 2), dtype=torch.float64))
    assert torch.equal(result['norm'], torch.full((2, 2), .001, dtype=torch.float64))
