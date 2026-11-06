import unittest
import numpy as np
import torch
from conditioning_demand import pack, alignment, parameter_from_words


class ConditioningDemand(unittest.TestCase):
    def test_parameter_mutation_is_observable_inside_inference_caller(self):
        words = torch.tensor([1., 2.], dtype=torch.bfloat16).view(torch.uint16).numpy()
        with torch.inference_mode():
            value = parameter_from_words(words, (1, 2), 'cpu')
            version = value._version
            value.add_(1.)
            self.assertGreater(value._version, version)
        self.assertEqual(words.tolist(), [16256, 16384])

    def test_pipeline_layout_keeps_each_rows_three_taps(self):
        states = {i: torch.arange(12).reshape(1, 4, 3) + i * 100 for i in (9, 18, 27)}
        result = pack(states)
        self.assertEqual(tuple(result.shape), (1, 4, 9))
        self.assertEqual(result[0, 2].tolist(), [906, 907, 908, 1806, 1807, 1808, 2706, 2707, 2708])

    def test_signed_energy_account_keeps_direction_reversal(self):
        baseline = np.array([10., -4.]); old = baseline + [3., 4.]; new = baseline + [-1., -2.]
        r = alignment(old, new, baseline)
        self.assertEqual(r['oldErrorEnergy'], 25.)
        self.assertEqual(r['newErrorEnergy'], 5.)
        self.assertEqual(r['oldNewErrorDot'], -11.)
        self.assertEqual(r['energyIdentityRemainder'], 0.)
        self.assertLess(r['oldNewErrorCosine'], 0.)
