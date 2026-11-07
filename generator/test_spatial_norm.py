import numpy as np
from spatial_norm import geometry, reconstruct, score_diagnostics


def test_physical_permutation_bilinear_and_clamped_edge():
    # Independent explicit scalar polynomial, with reversed storage order.
    ids = np.asarray([[[7, h, w, 0] for h in range(32) for w in range(32)]], dtype=np.int64)[:, ::-1].copy()
    knots, strata = geometry(ids)
    field = np.asarray([[2+h+3*w+h*w/4] for _, h, w, _ in ids[0]], dtype=np.float64)
    got, square, actual_knots, actual_strata = reconstruct(ids, np.sqrt(field[knots]))
    wanted = np.asarray([[2+min(h,30)+3*min(w,30)+min(h,30)*min(w,30)/4] for _, h, w, _ in ids[0]])
    np.testing.assert_allclose(square, wanted, rtol=1e-15, atol=1e-12)
    np.testing.assert_allclose(got*got, wanted, rtol=1e-15, atol=1e-12)
    assert np.array_equal(knots, actual_knots) and np.array_equal(strata, actual_strata)
    assert [int(np.sum(strata == a)) for a in range(3)] == [256, 705, 63]


def test_constant_smooth_epsilon_is_not_added_twice():
    ids = np.asarray([[[0, h, w, 0] for h in range(32) for w in range(32)]], dtype=np.int64)
    got, _, _, _ = reconstruct(ids, np.full((256, 2), .001))
    np.testing.assert_array_equal(got, np.full((1024, 2), .001))


def test_weighted_centering_and_kl_against_explicit_distribution():
    reference = np.log(np.asarray([[.1, .2, .3, .4]]))
    candidate = reference + np.asarray([[.4, -.3, 0, .2]])
    d = score_diagnostics(reference, candidate)
    p = [.1, .2, .3, .4];delta = [.4, -.3, 0, .2]
    mean = sum(a*b for a, b in zip(p, delta))
    z = sum(a*np.exp(b) for a, b in zip(p, delta))
    expected_kl = sum(a*np.log(a/(a*np.exp(b)/z)) for a, b in zip(p, delta))
    np.testing.assert_allclose(d['mean'], [mean], atol=1e-15)
    np.testing.assert_allclose(d['variance'], [sum(a*(b-mean)**2 for a,b in zip(p,delta))], atol=1e-15)
    np.testing.assert_allclose(d['kl'], [expected_kl], atol=1e-15)
    shifted = score_diagnostics(reference, reference+2)
    np.testing.assert_allclose(shifted['variance'], [0], atol=1e-30)
    np.testing.assert_allclose(shifted['kl'], [0], atol=1e-15)
