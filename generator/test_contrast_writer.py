"""Tiny supplied organisms and independent quadratic witnesses; no model data."""
import math
import unittest
import numpy as np
from contrast_writer_fit import quadratic, system, loss, split_statistics, candidates, fit_contrast_tree
from contextual_writer_program import BUDGET, fit_tree, finite_add_array, route, add_words, decode_words
from publish_contrast_writer import pooled, close


class ContrastWriterTests(unittest.TestCase):
    def setUp(self):
        self.r = np.asarray([[1., 3.], [4., -2.], [7., 1.], [-1., 2.]])
        self.edges = np.asarray([[0, 1], [2, 3]]); self.w = np.asarray([.1, .2, .3, .4]); self.v = np.asarray([.7, .2])

    def test_graph_solve_matches_independent_whitened_least_squares(self):
        q = quadratic(self.r, self.w, self.edges, self.v); labels = np.asarray([0, 1, 0, 1]); a = np.eye(2)[labels]
        p = np.asarray([[-1., 1., 0., 0.], [0., 0., -1., 1.]])
        design = np.concatenate([np.sqrt(self.w)[:, None]*a, np.sqrt(self.v)[:, None]*(p@a)])
        target = np.concatenate([np.sqrt(self.w)[:, None]*self.r, np.sqrt(self.v)[:, None]*(p@self.r)])
        expected = np.linalg.lstsq(design, target, rcond=None)[0]
        matrix, _, actual = system(q, labels, 2)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(matrix).min(), 0)

    def test_schur_gain_matches_direct_new_partition_including_internal_edges(self):
        q = quadratic(self.r, self.w, self.edges, self.v); labels = np.zeros(4, int)
        _, chol, constants = system(q, labels, 1); masks = np.asarray([[0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 0]], bool)
        diagonal = np.asarray([self.w[t].sum()+self.v[t[self.edges[:, 0]] != t[self.edges[:, 1]]].sum() for t in masks])
        gain, fallback, schur = split_statistics(q, labels, constants, chol, np.arange(4), masks, masks@q['qr'], diagonal)
        self.assertEqual(len(fallback), 0); self.assertTrue((schur > 0).all())
        for i, mask in enumerate(masks):
            trial = mask.astype(int); _, _, fitted = system(q, trial, 2)
            self.assertAlmostEqual(gain[i], loss(q, labels, constants)-loss(q, trial, fitted), places=11)
        self.assertGreater(diagonal[0], self.w[masks[0]].sum())

    def test_candidate_duplicate_partitions_keep_first_feature(self):
        x = np.repeat(np.arange(8)[:, None], 2, axis=1); q = quadratic(np.arange(8.)[:, None], np.ones(8)/8, [[0, 4]], [.1])
        budget = {**BUDGET, 'minLeafRows': 2, 'thresholdsPerFeature': 2}
        value = candidates(x, q, np.arange(8), budget)
        self.assertEqual(value['enumerated'], 4); self.assertEqual(len(value['meta']), 2)
        self.assertTrue(all(column == 0 for column, threshold in value['meta']))

    def test_paired_objective_preserves_context_instead_of_common_position(self):
        x = np.asarray([[0, 0], [0, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [1, 1]], float)
        y = np.asarray([10, 10, 20, 20, 11, 11, 21, 21], float)[:, None]; w = np.ones(8)/8
        budget = {**BUDGET, 'depth': 1, 'maxLeavesPerSlot': 2, 'minLeafRows': 2}
        endpoint_tree, _, _ = fit_tree(x, y, w, [0, 1], budget)
        tree, atoms, _, diagnostic = fit_contrast_tree(x, y, w, [[i, i+4] for i in range(4)], np.ones(4)/4, float(np.mean(y*y)), 1., budget)
        self.assertEqual(endpoint_tree['feature'], 0); self.assertEqual(tree['feature'], 1)
        pred = atoms[[route(tree, row)[0] for row in x]]
        np.testing.assert_array_equal(pred[4:]-pred[:4], np.ones((4, 1)))
        self.assertEqual(len(diagnostic['splits']), 1); self.assertIn('lossAfter', diagnostic['splits'][0])

    def test_constant_features_leave_irreducible_pair_response_visible(self):
        x = np.zeros((4, 1)); w = np.ones(4)/4
        tree, atoms, _, diagnostic = fit_contrast_tree(x, self.r, w, self.edges, self.v, 1., 1., {**BUDGET, 'minLeafRows': 1})
        self.assertEqual(tree, {'atom': 0}); self.assertEqual(len(atoms), 1)
        self.assertGreater(diagnostic['relaxedLoss'], 0); self.assertEqual(diagnostic['splits'], [])

    def test_zero_energy_absolute_policy_and_invalid_energy(self):
        args = (np.arange(4.)[:, None], self.r, np.ones(4)/4, self.edges, self.v)
        _, atoms, _, diagnostic = fit_contrast_tree(*args, 0., 0., {**BUDGET, 'minLeafRows': 1, 'depth': 1})
        self.assertTrue(np.isfinite(atoms).all()); self.assertTrue(math.isfinite(diagnostic['relaxedLoss']))
        for bad in (float('nan'), float('inf'), -1.):
            with self.assertRaises(ValueError): fit_contrast_tree(*args, bad, 1.)

    def test_admitted_atoms_use_existing_stdlib_finite_addition(self):
        a = np.asarray([[1., -3., .125]], np.float64); b = np.asarray([[.0031, .007, -.0004]], '<f4')
        expected = finite_add_array(a, b)[0]
        actual = np.asarray(decode_words(add_words(a[0].tolist(), b[0].tolist())), '<f4')
        np.testing.assert_array_equal(actual, expected)

    def test_pooled_scalar_denominators_and_undefined_zero(self):
        rows = [{'errorRMS': 1., 'normalizerRMS': math.sqrt(.5), 'predictionRMS': math.sqrt(.5), 'targetRMS': math.sqrt(.5), 'cosine': 0.},
                {'errorRMS': math.sqrt(2), 'normalizerRMS': math.sqrt(2), 'predictionRMS': 0., 'targetRMS': math.sqrt(2), 'cosine': None}]
        result = pooled(rows); self.assertAlmostEqual(result['relativeRMS'], math.sqrt(1.2)); self.assertEqual(result['cosine'], 0.)
        zero = pooled([dict.fromkeys(('errorRMS', 'normalizerRMS', 'predictionRMS', 'targetRMS'), 0.) | {'cosine': None}])
        self.assertIsNone(zero['relativeRMS']); self.assertIsNone(zero['cosine'])
        with self.assertRaises(ValueError): close(float('nan'), 0.)
        with self.assertRaises(ValueError): close(0., float('nan'))


if __name__ == '__main__': unittest.main()
