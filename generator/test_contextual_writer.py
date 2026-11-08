"""Tiny supplied-data checks; no model, scientific corpus or real fitting run."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from contextual_writer_program import (add_words, decode_words, features, fit_tree,
    finite_add_array, restore_program, route, validate)


def packet(ids, mask=None):
    return {'ids': ids, 'mask': mask or [1]*len(ids), 'positions': list(range(len(ids)))}


class Mechanics(unittest.TestCase):
    def test_ordered_finite_addition(self):
        first = decode_words(add_words([1.0], [.00390625]))
        second = decode_words(add_words(first, [.00390625]))
        combined = decode_words(add_words([1.0], [.0078125]))
        self.assertEqual(first, [1.0]); self.assertEqual(second, [1.0]); self.assertEqual(combined, [1.0078125])
        np.testing.assert_array_equal(finite_add_array(np.array([1.0]), np.array([.00390625])), first)

    def test_order_registers_break_bag_collision(self):
        a = features(packet([3, 4, 0], [1, 1, 0]), [[1.], [2.], [0.]], [3, 4], [0])
        b = features(packet([4, 3, 0], [1, 1, 0]), [[2.], [1.], [0.]], [3, 4], [0])
        self.assertEqual(a[-1][3], b[-1][3]); self.assertEqual(a[-1][6], b[-1][6])
        self.assertNotEqual(a[-1][4:6], b[-1][4:6]); self.assertEqual(a[-1][-1], b[-1][-1])

    def test_causal_features_do_not_see_future(self):
        a = features(packet([1, 2, 3]), [[1.], [2.], [3.]], [1, 2, 3, 9], [0])
        b = features(packet([1, 2, 9]), [[1.], [2.], [99.]], [1, 2, 3, 9], [0])
        self.assertEqual(a[:2], b[:2]); self.assertNotEqual(a[-1], b[-1])

    def test_numpy_and_stdlib_features_share_binary64_registers(self):
        lexical = np.array([[1e8], [1.], [-1e8]], np.float32); p = packet([1, 2, 3])
        offline = features(p, lexical, [1, 2, 3], [0]); online = features(p, lexical.tolist(), [1, 2, 3], [0])
        self.assertEqual(offline, online); self.assertEqual(offline[-1][-1], 1/3)
        tree = {'feature': len(offline[-1])-1, 'threshold': .1, 'left': {'atom': 0}, 'right': {'atom': 1}}
        self.assertEqual(route(tree, offline[-1]), route(tree, online[-1]))

    def test_unknown_literal_retains_lexical_state(self):
        p = packet([17, 0], [1, 0]); validate(p, length=2, vocab=32)
        x = features(p, [[42.], [0.]], [3], [0])
        self.assertEqual(x[0][-2:], [42., 42.]); self.assertEqual(x[0][3:6], [0, -1, -1])
        with self.assertRaises(ValueError): validate(packet([32, 0]), length=2, vocab=32)

    def test_weighted_tree_has_shared_constant_leaves(self):
        x = np.array([[0.], [0.], [1.], [1.]])
        y = np.array([[1., -1.], [3., -3.], [8., -8.], [8., -8.]])
        tree, atoms, trace = fit_tree(x, y, np.array([3., 1., 1., 1.]), [0], {'depth': 1, 'minLeafRows': 1, 'thresholdsPerFeature': 16})
        self.assertEqual(len(atoms), 2); self.assertEqual(len(trace), 2)
        np.testing.assert_array_equal(atoms[route(tree, [0.])[0]], [1.5, -1.5])
        np.testing.assert_array_equal(atoms[route(tree, [1.])[0]], [8., -8.])

    def test_exact_restore_preserves_both_branches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root/'initial'; source.mkdir(); (source/'program.json').write_text('{"atom":1}\n')
            proposal = root/'half'; proposal.mkdir(); (proposal/'program.json').write_text('{"atom":0.5}\n')
            expected = restore_program(source, root/'accepted')
            self.assertEqual((root/'accepted/program.json').read_bytes(), (source/'program.json').read_bytes())
            self.assertNotEqual((root/'accepted/program.json').read_bytes(), (proposal/'program.json').read_bytes()); self.assertEqual(set(expected), {'program.json'})

    def test_fresh_process_target_and_runtime_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'allowed').write_text('owned'); (root/'target').write_text('oracle'); (root/'out').mkdir()
            spec = {'readAllow': [str(root/'allowed')], 'forbiddenPaths': [str(root/'target')]}
            script = "import json,sys;sys.path.insert(0,sys.argv[1]);from token_history_process import isolate;spec=json.loads(sys.argv[2]);opens,denied,imports=isolate(spec,sys.argv[3]);assert open(spec['readAllow'][0]).read()=='owned';print(json.dumps({'denied':denied,'imports':imports}))"
            result = subprocess.run([sys.executable, '-c', script, str(Path(__file__).parent), json.dumps(spec), str(root/'out')], check=True, capture_output=True, text=True)
            value = json.loads(result.stdout); self.assertEqual(value['denied'], [str(root/'target')]); self.assertEqual(value['imports'], ['numpy', 'torch'])


if __name__ == '__main__': unittest.main()
