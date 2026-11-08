import itertools
from fractions import Fraction as Q
import unittest
from behavioral_authority import add_observation, exception, inverse, reconstruct, evaluate


class AuthorityTests(unittest.TestCase):
    def observations(self):
        words = [w for n in range(5) for w in itertools.product(('a', 'b'), repeat=n)]
        words += [tuple('baaba'), tuple('babba')]
        return {w: Q(int(len(w) >= 2 and w[-2] == 'b')) for w in words}

    def test_exact_reconstruction_and_noncommuting_order(self):
        self.assertEqual(inverse([[2, 0], [0, 3]]), [[Q(1, 2), Q(0)], [Q(0), Q(1, 3)]])
        work = {'rationalLinearMACs': 0}
        model = reconstruct(self.observations(), [(), ('b',), ('b', 'a')], [(), ('a',), ('b', 'a')], ['a', 'b'], work)
        self.assertEqual(model['initial'], [0, 0, 1])
        self.assertEqual(model['operators']['b'], [[0, 0, 0], [1, 0, 0], [0, 1, 1]])
        self.assertEqual(evaluate(model, tuple('ab'), work), 0)
        self.assertEqual(evaluate(model, tuple('ba'), work), 1)

    def test_missing_and_conflicting_evidence_refused(self):
        obs = self.observations()
        with self.assertRaises(ValueError): add_observation(obs, ['b', 'a'], [0])
        with self.assertRaises(ValueError): add_observation(obs, ['?'], [0])
        del obs[tuple('babba')]
        with self.assertRaises(KeyError):
            reconstruct(obs, [(), ('b',), ('b', 'a')], [(), ('a',), ('b', 'a')], ['a', 'b'], {'rationalLinearMACs': 0})
        with self.assertRaises(ValueError): inverse([[1, 1], [1, 1]])

    def test_unseen_singleton_and_rank_minor(self):
        witness = ('a',) * 33
        self.assertTrue(all(exception(w, witness) == 0 for w in self.observations()))
        self.assertEqual(exception(witness, witness), 1)
        self.assertEqual(exception(witness + ('a',), witness), 0)
        # Every row/column of the singleton Hankel minor has one1 at the anti-diagonal.
        minor = [[exception(('a',) * (i + j), witness) for j in range(34)] for i in range(34)]
        self.assertEqual([sum(row) for row in minor], [1] * 34)
        self.assertEqual([row[::-1] for row in minor], [[int(i == j) for j in range(34)] for i in range(34)])


if __name__ == '__main__':
    unittest.main()
