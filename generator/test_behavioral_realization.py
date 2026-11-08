"""Independent exact mechanics, not native-image or semantic quality tests."""
from copy import deepcopy
from fractions import Fraction as Q
import unittest

from behavioral_program import run_word, validate
from behavioral_realization import FiniteMachine, WordOracle, certify_finite, compile_panel, evaluate, inverse, lower_copy_program, multiply, transformed, unit, words
from behavioral_realization_study import toy_source


def candidate(machine, shallow=False):
    oracle = WordOracle(machine.output)
    tests = [((), 0)] if shallow else [(suffix, 0) for suffix in words(machine.alphabet, 2)]
    return compile_panel(oracle, machine.alphabet, words(machine.alphabet, 2), tests)


class BehavioralRealizationTests(unittest.TestCase):
    def test_exact_inverse_with_fractional_entries(self):
        basis = [[Q(2), Q(1)], [Q(1), Q(1)]]
        self.assertEqual(inverse(basis), [[Q(1), Q(-1)], [Q(-1), Q(2)]])
        self.assertEqual(multiply(basis, inverse(basis), 2), unit(2))
        self.assertEqual(inverse([[2, 1], [0, 1]]), [[Q(1, 2), Q(-1, 2)], [Q(0), Q(1)]])
        with self.assertRaises(ValueError):
            inverse([[1, 1], [1, 1]])

    def test_discovered_basis_and_independent_penultimate_bit(self):
        machine = FiniteMachine(toy_source())
        result = candidate(machine)
        self.assertEqual(result['rank'], 3)
        self.assertEqual(result['basis'], [['0', '0', '1'], ['0', '1', '1'], ['1', '0', '1']])
        proof = certify_finite(machine, result)
        self.assertEqual(proof['closureStatus'], 'proved-on-supplied-finite-domain')
        self.assertEqual(len(proof['transitions']), 8)
        for word in words(machine.alphabet, 4):
            expected = int(word[-2] == 'b') if len(word) >= 2 else 0
            self.assertEqual(evaluate(result, word), [Q(expected)])

    def test_shallow_panel_retains_a_shortest_future_counterexample(self):
        machine = FiniteMachine(toy_source())
        result = candidate(machine, shallow=True)
        proof = certify_finite(machine, result)
        self.assertEqual(result['rank'], 1)
        self.assertEqual(proof['closureStatus'], 'counterexample')
        witness = next(x for x in proof['mergedFutureWitnesses'] if x['states'] == ['00', '01'])
        self.assertEqual(witness['suffix'], ['a'])
        failures = [w for w in words(machine.alphabet, 3) if tuple(evaluate(result, w)) != machine.output(w)]
        self.assertEqual(failures[0], ('b', 'a'))
        self.assertEqual(lower_copy_program(result, proof)['status'], 'declined-unclosed-realization')

    def test_lowered_program_and_gauge_limit_are_different_from_closure(self):
        machine = FiniteMachine(toy_source())
        result = candidate(machine)
        program = lower_copy_program(result, certify_finite(machine, result))
        validate(program)
        self.assertEqual(program['persistentRegisters'], 2)
        for word in words(machine.alphabet, 4):
            self.assertEqual(run_word(program, word)['output'], list(machine.output(word)))
        changed = transformed(result, [[1, 0, 0], [1, 1, 0], [0, 0, 1]])
        proof = certify_finite(machine, changed)
        self.assertEqual(proof['closureStatus'], 'proved-on-supplied-finite-domain')
        self.assertEqual(lower_copy_program(changed, proof)['status'], 'not-lowerable-by-copy-constant-in-these-coordinates')

    def test_simultaneous_old_state_reads_and_domain_refusal(self):
        machine = FiniteMachine(toy_source())
        result = candidate(machine)
        program = lower_copy_program(result, certify_finite(machine, result))
        program['initial'] = [0, 1]
        program['transitions']['a'] = [{'op': 'copy', 'register': 1}, {'op': 'copy', 'register': 0}]
        self.assertEqual(run_word(program, ['a'])['finalRegisters'], [1, 0])
        self.assertEqual(run_word(program, ['?'])['status'], 'outside-declared-alphabet')
        invalid = deepcopy(program)
        invalid['readout'] = [{'op': 'multiply', 'coefficient': 1, 'register': 0}]
        with self.assertRaises(ValueError):
            validate(invalid)

    def test_zero_rank_and_mismatched_certificate(self):
        machine = FiniteMachine({'alphabet': ['a'], 'initial': 's', 'states': {'s': {'output': [0], 'next': {'a': 's'}}}})
        result = candidate(machine)
        proof = certify_finite(machine, result)
        program = lower_copy_program(result, proof)
        self.assertEqual(result['rank'], 0)
        self.assertEqual(run_word(program, ['a', 'a'])['output'], [0])
        changed = deepcopy(result)
        changed['scope'] += ' changed'
        with self.assertRaises(ValueError):
            lower_copy_program(changed, proof)


if __name__ == '__main__':
    unittest.main()
