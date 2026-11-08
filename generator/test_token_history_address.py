"""Small synthetic mechanics only; no saved model arrays."""
import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from token_history_address import HEADS, History, fit, inventory, metrics, resolve, run_packet, validate_packet


def program(rule):
    return {'schema': 'tce-token-history-address-program-v1', 'heads': HEADS, 'width': 4,
        'primary': [copy.deepcopy(rule) for _ in range(HEADS)], 'positionOnly': [[['recent', j] for j in range(4)] for _ in range(HEADS)]}


class Mechanics(unittest.TestCase):
    def test_mask_clock_and_occurrences(self):
        s = History()
        for p, (token, mask) in enumerate([(5, 1), (8, 1), (5, 1), (5, 0)]):
            s.admit(token, mask, p)
        self.assertEqual(s.positions, [0, 1, 2])
        self.assertEqual(s.value(('same_previous', 0), 5, 2), 0)
        self.assertEqual(s.value(('same_previous', 0), 5, 3), 2)
        self.assertEqual(s.value(('different_first', 0), 5, 3), 1)
        with self.assertRaises(ValueError):
            s.admit(1, 1, 5)

    def test_fallback_duplicates_and_causal_refusal(self):
        self.assertEqual(resolve(-1, [0, 2, 4], []), (4, 1))
        self.assertEqual(resolve(4, [0, 2, 4], [4]), (2, 2))
        self.assertEqual(resolve(0, [0], [0]), (-1, 3))
        with self.assertRaises(ValueError):
            resolve(3, [0, 2, 4], [])

    def test_no_padding_admission_and_short_prefix(self):
        packet = {'ids': [8, 8, 3, 8], 'mask': [0, 1, 1, 0], 'positions': [0, 1, 2, 3]}
        p = program([['same_last', 0]]*4)
        rows, receipt = run_packet(packet, p, 'primary')
        self.assertEqual(rows[0]['addresses'][0], [-1]*4)
        self.assertEqual(rows[-1]['addresses'][0], [1, 2, -1, -1])
        self.assertEqual(rows[-1]['reasons'][0], [0, 2, 3, 3])
        self.assertEqual(receipt['finalAdmittedTokens'], 2)

    def test_fitter_scores_resolved_rule_and_freezes_literals(self):
        packet = {'ids': [7, 8, 9, 10, 11, 12], 'mask': [1]*6, 'positions': list(range(6))}
        target = [[[max(0, p-j) if p-j >= 0 else -1 for j in range(4)] for _ in range(HEADS)] for p in range(6)]
        p, receipt = fit([packet], [target])
        rows, _ = run_packet(packet, p, 'primary')
        self.assertEqual([a['addresses'][0] for a in rows[4:]], [target[4][0], target[5][0]])
        self.assertEqual(receipt['nontrivialDiscoveryQueriesPerHead'], 2)
        self.assertEqual(p['primary'][0][:2], [['recent', 0]]*2)
        self.assertEqual(receipt['candidateInventory'], [list(a) for a in inventory(range(7, 13))])

    def test_fixed_controls_and_sentinel_domain(self):
        packet = {'ids': [5, 8, 5, 9, 5], 'mask': [1]*5, 'positions': list(range(5))}
        p = program([['literal_first', 5], ['same_previous', 0], ['recent', 0], ['first', 0]])
        p['primary'][1] = [['first', j] for j in range(4)]
        normal, _ = run_packet(packet, p, 'primary')
        rotated, _ = run_packet(packet, p, 'headRotated')
        null, _ = run_packet(packet, p, 'tokenNull')
        recent, _ = run_packet(packet, p, 'recent')
        self.assertEqual(rotated[-1]['addresses'][0], normal[-1]['addresses'][1])
        self.assertEqual(null[-1]['addresses'][0], [4, 3, 2, 0])
        self.assertEqual(recent[-1]['addresses'][0], [4, 3, 2, 1])
        self.assertNotEqual(normal[-1]['addresses'][0], null[-1]['addresses'][0])

    def test_metrics_ignore_minus_one_and_name_mass(self):
        m = metrics([2, 0, -1, -1], [0, 2, -1, -1], [.75, .25, 0, 0])
        self.assertEqual(m, {'ordered': 0, 'set': 1, 'top1': 0, 'overlap': 1, 'oldTop4WeightCoverage': 1})
        self.assertEqual(metrics([2, 1, 4, 6], [0, 2, 3, 4], [.4, .3, .2, .1])['oldTop4WeightCoverage'], .4)

    def test_forbidden_features_and_noncausal_packet(self):
        packet = {'ids': [0], 'mask': [1], 'positions': [0]}
        with self.assertRaises(ValueError):
            validate_packet({**packet, 'q': []})
        with self.assertRaises(ValueError):
            validate_packet({**packet, 'positions': [1]})
        with self.assertRaises(ValueError):
            validate_packet({**packet, 'ids': [-1]})

    def test_fresh_compile_and_serve_exclude_labels_and_runtimes(self):
        source = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet = {'ids': [1, 2, 3, 4, 5, 6], 'mask': [1]*6, 'positions': list(range(6))}
            labels = [[[p-j if p-j >= 0 else -1 for j in range(4)] for _ in range(HEADS)] for p in range(6)]
            for name, value in [('tokens.json', packet), ('labels.json', labels), ('forbidden.json', {'hidden': 'canary'})]:
                (root/name).write_text(json.dumps(value))
            compile_spec = {'cases': [{'tokens': str(root/'tokens.json'), 'supports': str(root/'labels.json')}],
                'readAllow': [str(root/'tokens.json'), str(root/'labels.json')], 'forbiddenPaths': [str(root/'forbidden.json')]}
            (root/'compile.json').write_text(json.dumps(compile_spec))
            def invoke(action, spec, output):
                result = subprocess.run([sys.executable, '-I', '-B', str(source/'token_history_process.py'), action, str(spec), str(output)], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads((output/'receipt.json').read_text())
            compiled = invoke('compile', root/'compile.json', root/'compiled')
            program_path = root/'compiled/program.json'
            serve_spec = {'program': str(program_path), 'programSHA256': compiled['programSHA256'],
                'cases': [{'id': 'synthetic', 'tokens': str(root/'tokens.json')}],
                'readAllow': [str(program_path), str(root/'tokens.json')],
                'forbiddenPaths': [str(root/'labels.json'), str(root/'forbidden.json')]}
            (root/'serve.json').write_text(json.dumps(serve_spec))
            served = invoke('serve', root/'serve.json', root/'served')
            self.assertEqual(len(served['fileCanaryDenials']), 2)
            self.assertEqual(served['blockedImportCanaries'], ['numpy', 'torch'])
            self.assertEqual(served['loadedNumericalRuntimeRoots'], [])
            self.assertNotIn(str(root/'labels.json'), served['successfulReadPaths'])
            self.assertEqual(len(served['rows']), 6)
            with gzip.open(root/'served/synthetic-primary.jsonl.gz', 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(rows[-1]['addresses'][0], [5, 4, 3, 2])


if __name__ == '__main__':
    unittest.main()
