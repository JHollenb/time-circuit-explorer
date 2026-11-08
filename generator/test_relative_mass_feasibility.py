"""Tiny exact layout/ambiguity/rounding mechanics; no real model inputs."""
from fractions import Fraction
import unittest
from relative_mass_feasibility import census, score_index, rotary_witness, weight_witness


class FeasibilityMechanics(unittest.TestCase):
    def test_complete_small_layout_and_legal_diagonal_count(self):
        indices = [score_index(2, 4, u, t, j) for t in range(1, 4) for u in range(2) for j in range(t)]
        self.assertEqual(indices, list(range(12)))
        values = census(2, 4, 3, 2, 8)
        self.assertEqual(values['currentDomainScores'], 12)
        self.assertEqual(values['additionalLegalActiveDiagonalScoresPerHead'], 12)
        self.assertEqual(values['allQueryIDsWithLegalDiagonalFP64BytesAllHeads'], 1056)
        for args in [(2, 4, 2, 1, 0), (2, 4, 0, 1, 1), (2, 4, 0, 4, 0), (2, 4, 0, 0, 0)]:
            with self.assertRaises(ValueError): score_index(*args)

    def test_same_rank_distinct_weights(self):
        witness = weight_witness()
        a = list(map(Fraction, witness['normalizedA'])); b = list(map(Fraction, witness['normalizedB']))
        self.assertEqual(sum(a), 1); self.assertEqual(sum(b), 1)
        self.assertTrue(all(x > y for x, y in zip(a, a[1:])))
        self.assertTrue(all(x > y for x, y in zip(b, b[1:])))
        self.assertNotEqual(a, b)

    def test_exact_rounded_translation_counterexample(self):
        witness = rotary_witness()
        self.assertEqual(Fraction(witness['firstScore']), Fraction(237, 256))
        self.assertEqual(Fraction(witness['translatedScore']), Fraction(60635, 65536))
        self.assertEqual(Fraction(witness['signedDifference']), -Fraction(37, 65536))


if __name__ == '__main__': unittest.main()
