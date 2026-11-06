import unittest
import torch
from kv_accumulation import projection_policy


class AccumulationScope(unittest.TestCase):
    def test_only_alternative_kv_changes_and_exception_restores(self):
        flags = torch.backends.cuda.matmul
        before = flags.allow_bf16_reduced_precision_reduction
        try:
            flags.allow_bf16_reduced_precision_reduction = True
            for policy, is_kv in [('original', True), ('alternative', False)]:
                with projection_policy(policy, is_kv):
                    self.assertTrue(flags.allow_bf16_reduced_precision_reduction)
            with self.assertRaisesRegex(RuntimeError, 'projection failed'):
                with projection_policy('alternative', True):
                    self.assertFalse(flags.allow_bf16_reduced_precision_reduction)
                    raise RuntimeError('projection failed')
            self.assertTrue(flags.allow_bf16_reduced_precision_reduction)
            flags.allow_bf16_reduced_precision_reduction = False
            with projection_policy('original', True):
                self.assertFalse(flags.allow_bf16_reduced_precision_reduction)
            self.assertFalse(flags.allow_bf16_reduced_precision_reduction)
        finally:
            flags.allow_bf16_reduced_precision_reduction = before

    def test_unknown_policy_refuses_before_computation(self):
        with self.assertRaises(ValueError):
            with projection_policy('guess', True):
                self.fail('Invalid policy reached consumer')
