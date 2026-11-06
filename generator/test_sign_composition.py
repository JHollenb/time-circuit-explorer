import unittest
import numpy as np
from sign_composition import support_closed, decompose, account


class SignCompositionTests(unittest.TestCase):
    def test_actual_support_includes_empty_and_mixed_rows(self):
        s = np.array([-1, -1, 1, 1])
        mask = np.tri(4, dtype=bool); mask[0] = False
        np.testing.assert_array_equal(support_closed(s, mask), [False, True, False, False])
        with self.assertRaises(ValueError): support_closed([0, 1], np.eye(2, dtype=bool))

    def test_even_write_leaves_sign_and_scaling_families(self):
        p = {'R': np.array([[2., 0.]]), 'M': np.array([[0., 1.]]),
             'Hnext': np.array([[2., 1.]])}
        q = {'R': -p['R'], 'M': p['M'], 'Hnext': np.array([[-2., 1.]])}
        a = decompose(p, q, [-1]); np.testing.assert_array_equal(a['defect'], [[0., 2.]])
        np.testing.assert_array_equal(a['residual'], [[0., 0.]])
        np.testing.assert_array_equal(a['closure'], [[0., 0.]])
        np.testing.assert_allclose(a['bestScale'], [-.6])
        np.testing.assert_allclose(np.sum(a['orthogonal'] * p['Hnext'], axis=1), 0, atol=1e-15)
        self.assertGreater(account(a, q['Hnext'], [True])['bestScaleOrthogonalRms'], 0)

    def test_finite_addition_remainder_is_required(self):
        p = {'R': np.array([[4., 8.]]), 'M': np.array([[.125, .25]]),
             'Hnext': np.array([[4., 8.]])}
        q = {'R': -p['R'], 'M': p['M'], 'Hnext': np.array([[-4., -8.]])}
        a = decompose(p, q, [-1])
        self.assertTrue(np.any(a['additionRemainder'] != 0))
        np.testing.assert_array_equal(a['closure'], 0)
        self.assertFalse(np.array_equal(a['defect'], a['mlp']))

    def test_zero_baseline_and_zero_denominator_remain_explicit(self):
        p = {k: np.zeros((1, 2)) for k in ('R', 'M', 'Hnext')}
        q = {**p, 'Hnext': np.ones((1, 2))}
        a = decompose(p, q, [1])
        self.assertEqual(account(a, q['Hnext'], [True])['undefinedScaleRows'], 1)
        self.assertIsNone(account(a, q['Hnext'], [False]))
        self.assertIsNone(account(decompose(p, p, [1]), p['Hnext'], [True])['defectOverEndpointRms'])
