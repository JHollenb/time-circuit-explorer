"""Independent smooth identity, finite defect and selected-operation checks."""
import numpy as np
import pytest
import torch
from normalization_innovation import shared, standard_deviation, predict, make_selected, IN, DOWN


def smooth(x, w, d, b, epsilon=1e-6):
    x, w, d, b = [np.asarray(v, np.float64) for v in [x, w, d, b]]
    c = x - x.mean(-1, keepdims=True)
    return ((c / np.sqrt(np.mean(c * c, axis=-1, keepdims=True) + epsilon)) * d + b) @ w.T


def test_shared_secant_closes_two_independent_receiver_baselines():
    w = torch.tensor([[1., -2., 3.], [2., 1., -.5]], dtype=torch.float64)
    d = torch.tensor([[[1.5, .5, -.25]]], dtype=torch.float64)
    b = torch.tensor([[[.25, -.5, 1.]]], dtype=torch.float64)
    aa = torch.tensor([[[.25, -.5, .125], [1., -.25, .5]]], dtype=torch.float64)
    ab = aa + torch.tensor([[[.125, -.25, .5], [-.5, .125, .25]]], dtype=torch.float64)
    delta, centered, j, beta = shared(w, d, b, aa, ab)
    for embedding in [torch.tensor([[[1., 2., 4.], [2., -1., 3.]]], dtype=torch.float64), torch.tensor([[[-3., 1., 2.], [4., 5., -2.]]], dtype=torch.float64)]:
        ua, ub = embedding + aa, embedding + ab
        fa = torch.from_numpy(smooth(ua.numpy(), w.numpy(), d.numpy(), b.numpy()))
        raw, _, _ = predict(fa, j, beta, standard_deviation(ua), standard_deviation(ub))
        np.testing.assert_allclose(raw.numpy(), smooth(ub.numpy(), w.numpy(), d.numpy(), b.numpy()), rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(centered.mean(-1).numpy(), 0, atol=1e-16)


def test_finite_commit_defect_has_the_declared_sign_and_baseline_terms():
    bf = torch.bfloat16
    embedding = torch.tensor([[[8., 1., -2.]]], dtype=bf)
    aa = torch.tensor([[[.03125, .125, .25]]], dtype=bf)
    ab = aa + torch.tensor([[[.03125, .015625, -.03125]]], dtype=bf)
    ua, ub = embedding + aa, embedding + ab
    w = torch.tensor([[1., -2., 3.], [2., 1., -.5]], dtype=torch.float64)
    d = torch.tensor([[[1.5, .5, -.25]]], dtype=bf)
    b = torch.tensor([[[.25, -.5, 1.]]], dtype=bf)
    delta, centered, j, beta = shared(w, d, b, aa, ab)
    ta = torch.from_numpy(smooth(ua.double().numpy(), w.numpy(), d.double().numpy(), b.double().numpy()))
    tb = torch.from_numpy(smooth(ub.double().numpy(), w.numpy(), d.double().numpy(), b.double().numpy()))
    fa, fb = ta.to(bf), tb.to(bf)
    raw, quant, alpha = predict(fa, j, beta, standard_deviation(ua), standard_deviation(ub))
    rho = (ub.double() - ua.double()) - delta
    crho = rho - rho.mean(-1, keepdim=True)
    defect = alpha * (fa.double() - ta) - (fb.double() - tb) - torch.nn.functional.linear(d.double() * crho, w) / standard_deviation(ub)
    torch.testing.assert_close(raw - fb.double(), defect, rtol=1e-12, atol=1e-14)
    assert torch.any(rho != 0) and torch.any(raw != fb.double())
    assert torch.equal(quant, raw.float().to(bf))


def test_only_two_selected_parameters_and_two_shared_contractions(monkeypatch):
    model = make_selected(torch, rows=3, width=2)
    assert {n: list(v.shape) for n, v in model.named_parameters()} == {IN: [4, 3], DOWN: [3, 2]}
    original = torch.nn.functional.linear
    calls = []
    def counted(x, w, bias=None):
        calls.append((tuple(x.shape), tuple(w.shape), str(w.dtype)))
        return original(x, w, bias)
    monkeypatch.setattr(torch.nn.functional, 'linear', counted)
    shared(torch.ones((4, 3), dtype=torch.float64), torch.ones((1, 1, 3)), torch.zeros((1, 1, 3)), torch.zeros((1, 2, 3)), torch.ones((1, 2, 3)))
    assert calls == [((1, 2, 3), (4, 3), 'torch.float64'), ((1, 1, 3), (4, 3), 'torch.float64')]


def test_invalid_precision_and_nonfinite_prediction_are_refused():
    with pytest.raises(ValueError):
        shared(torch.ones((2, 3)), torch.ones(3), torch.zeros(3), torch.zeros(3), torch.ones(3))
    with pytest.raises(ValueError):
        predict(torch.zeros((1, 2)), torch.full((1, 2), float('inf')), torch.zeros((1, 2)), torch.ones((1, 1)), torch.ones((1, 1)))
