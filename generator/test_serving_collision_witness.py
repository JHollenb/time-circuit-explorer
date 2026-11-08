"""Independent tiny mechanics for a scalar bound and continuation witnesses."""
import math
import unittest

from serving_collision_witness import collision_classes, closure_witnesses, comparison_status, known_wiring, pair_bounds, word_bag


class ServingCollisionTests(unittest.TestCase):
    def test_binding_is_lost_but_word_order_is_available(self):
        prompts = [{'id': 'a', 'prompt': 'red cube blue sphere'},
                   {'id': 'b', 'prompt': 'blue cube red sphere'},
                   {'id': 'c', 'prompt': 'red sphere'}]
        self.assertEqual(collision_classes(prompts, word_bag), [['a', 'b']])
        self.assertEqual(collision_classes(prompts, lambda p: tuple(p.split())), [])

    def test_parallelogram_floor_attained_at_midpoint(self):
        left, right = (1, -3), (7, 5)
        distance2 = sum((x - y) ** 2 for x, y in zip(left, right)) / 2
        bound = pair_bounds(math.sqrt(distance2))
        midpoint = tuple((x + y) / 2 for x, y in zip(left, right))
        errors = [sum((z - y) ** 2 for z, y in zip(midpoint, target)) / 2 for target in (left, right)]
        self.assertAlmostEqual(sum(errors) / 2, bound['minimumMeanEndpointSquaredRMS'])
        self.assertAlmostEqual(math.sqrt(max(errors)), bound['minimumWorstEndpointRMS'])
        # Endpoint and off-axis choices both preserve the lower bound.
        for prediction in [left, (0, 0), (9, -10)]:
            errors = [sum((z - y) ** 2 for z, y in zip(prediction, target)) / 2 for target in (left, right)]
            self.assertGreaterEqual(sum(errors) / 2 + 1e-12, distance2 / 4)

    def test_current_readout_does_not_establish_future_closure(self):
        toy = known_wiring()
        self.assertEqual(toy['transitionsPerRepresentation'], 8)
        self.assertEqual(len(toy['visibleOnlyWitnesses']), 2)
        self.assertEqual({row['control'] for row in toy['visibleOnlyWitnesses']}, {'reveal'})
        self.assertEqual(toy['refinedObservedWitnesses'], [])

    def test_controls_and_clock_are_part_of_the_condition(self):
        rows = [dict(clock=0, control='a', state=[0], code=[0], nextCode=[0]),
                dict(clock=0, control='b', state=[1], code=[0], nextCode=[1]),
                dict(clock=1, control='a', state=[2], code=[0], nextCode=[1])]
        self.assertEqual(closure_witnesses(rows), [])
        with self.assertRaises(ValueError):
            pair_bounds(float('nan'))

    def test_missing_and_incomparable_are_not_scientific_nulls(self):
        self.assertEqual(comparison_status(True, True, 2), 'counterexample')
        self.assertEqual(comparison_status(True, False, 2), 'incomparable-controls')
        self.assertEqual(comparison_status(True, True, None), 'missing-evidence')
        self.assertEqual(comparison_status(False, True, 2), 'no-counterexample-in-inspected-records')


if __name__ == '__main__':
    unittest.main()
