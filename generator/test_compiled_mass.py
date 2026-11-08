"""Tiny supplied mechanics; no source-model operand arrays or checkpoints."""
from array import array
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from compiled_mass_compile import compile_mass
from compiled_mass_program import MassProgram, normalize
from compiled_mass_study import source_custody
from source_order_compile import compile_order


def fixture(root):
    q = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 2.], [2., -1.]], dtype=np.float64)
    k = np.array([[[float(t+j), float(t-j)] for j in range(5)] for t in [1, 3]], dtype=np.float64)
    ranks = root/'rank'
    metadata = compile_order(q, k, [10, 20], 99, ranks)
    ledger = json.loads((ranks/'compiler.json').read_text())
    compile_mass(q, k, metadata, ledger, ranks/'program.json', root/'mass')
    program = MassProgram.load(root/'mass/program.json', ranks/'program.json', ranks/'order.u16.gz', root/'mass/scores.f64')
    return program, q, k


class Mechanics(unittest.TestCase):
    def test_historical_custody_must_match_retained_original(self):
        phase = {'sha256': 'a'*64, 'size_bytes': 12}
        custody = {'required': True, 'declared': dict(phase), 'sealed': dict(phase), 'executed': dict(phase, host='beast')}
        self.assertTrue(source_custody(custody, custody)['matchesRetainedOriginal'])
        wrong = dict(custody, executed=dict(phase, host='elsewhere'))
        with self.assertRaises(ValueError): source_custody(wrong, custody)
        wrong = dict(custody, sealed=dict(phase, sha256='b'*64))
        with self.assertRaises(ValueError): source_custody(wrong, custody)

    def test_empty_shift_invariance_and_underflow(self):
        self.assertEqual(normalize([]), [])
        self.assertEqual(normalize([2.]), [1.])
        self.assertEqual(normalize([2., 1.]), normalize([12., 11.]))
        self.assertEqual(normalize([0., -10000.]), [1., 0.])
        self.assertEqual(normalize([4., 4.]), [.5, .5])
        with self.assertRaises(ValueError): normalize([float('nan')])

    def test_own_support_short_prefix_and_scalar_reference(self):
        with tempfile.TemporaryDirectory() as d:
            program, q, k = fixture(Path(d))
            packet = {'ids': [10, 99, 20, 99, 99], 'mask': [1, 0, 1, 0, 0], 'positions': list(range(5))}
            rows, work = program.serve(packet)
            self.assertEqual(rows[1]['weights'], [1.])
            self.assertEqual(rows[1]['paddedWeights'], [1., 0., 0., 0.])
            for i in [3, 4]:
                scores = {j: math.fsum(float(a)*float(b) for a, b in zip(q[i], k[t, j])) for j, t in [(0, 0), (2, 1)]}
                order = sorted(scores, key=lambda j: (-scores[j], j))
                self.assertEqual(rows[i]['addresses'], order+[-1, -1])
                self.assertEqual(rows[i]['weights'], normalize([scores[j] for j in order]))
            self.assertEqual(work['massWork']['scoreLookups'], 5)
            self.assertEqual(work['massWork']['expCalls'], 5)

    def test_empty_padding_and_unknown_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            program, _, _ = fixture(Path(d))
            rows, _ = program.serve({'ids': [99, 123, 10, 99], 'mask': [0, 1, 1, 0], 'positions': [0, 1, 2, 3]})
            self.assertEqual(rows[0]['weights'], [])
            self.assertEqual(rows[0]['paddedWeights'], [0.]*4)
            self.assertEqual(rows[-1]['status'], 'unknown-admitted-token')
            self.assertIsNone(rows[-1]['weights'])
            self.assertEqual(rows[-1]['unknownPositions'], [1])

    def test_domain_and_score_corruption_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);program, _, _ = fixture(root)
            metadata = dict(program.metadata, paddingQueryID=98)
            with self.assertRaises(ValueError): MassProgram(program.order, metadata, program.values)
            with (root/'mass/scores.f64').open('ab') as f: f.write(b'x')
            with self.assertRaises(ValueError):
                MassProgram.load(root/'mass/program.json', root/'rank/program.json', root/'rank/order.u16.gz', root/'mass/scores.f64')

    def test_fresh_child_runtime_and_target_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);fixture(root)
            packet = root/'tokens.json';packet.write_text(json.dumps({'ids': [10, 20, 99], 'mask': [1, 1, 0], 'positions': [0, 1, 2]}))
            target = root/'old-weights.json';target.write_text('{}')
            paths = {'metadata': root/'mass/program.json', 'rankMetadata': root/'rank/program.json',
                     'ranks': root/'rank/order.u16.gz', 'scores': root/'mass/scores.f64'}
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            spec = {**{k: str(v) for k, v in paths.items()}, 'metadataSHA256': digest(paths['metadata']),
                    'cases': [{'id': 'toy', 'tokens': str(packet), 'tokensSHA256': digest(packet)}],
                    'readAllow': [str(p) for p in paths.values()]+[str(packet)], 'forbiddenPaths': [str(target)]}
            (root/'spec.json').write_text(json.dumps(spec))
            child = Path(__file__).resolve().parent/'compiled_mass_program.py'
            result = subprocess.run([sys.executable, '-I', '-B', str(child), str(root/'spec.json'), str(root/'served')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((root/'served/receipt.json').read_text())
            self.assertEqual(receipt['fileCanaryDenials'], [str(target)])
            self.assertEqual(receipt['blockedImportCanaries'], ['numpy', 'torch'])
            self.assertEqual(receipt['loadedNumericalRuntimeRoots'], [])
            self.assertTrue(receipt['scalarAttentionComputed'])
            self.assertEqual(receipt['onlineQKCoordinateProducts'], 0)


if __name__ == '__main__': unittest.main()
