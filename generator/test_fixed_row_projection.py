"""Tiny independent fixed-projector, finite-boundary and signed Gram references."""
import unittest


def mechanics():
    import math
    import numpy as np
    from encoder_atoms import bf
    from fixed_row_projection import compute, abis, work, observations, summarize, UPDATE_TERMS, EXIT_TERMS
    n, text, width, rank = 6, 2, 3, 2
    u = np.array([[1, 0], [0, 1], [.5, .5], [.5, -.5], [.5, .5], [.5, -.5]], dtype=np.float64)[None]
    y82 = bf(np.arange(n*width).reshape(1, n, width)/8-1).astype(np.float64)
    y64 = (np.arange(n*width).reshape(1, n, width)/11-.7).astype(np.float64)
    y83 = bf(y64).astype(np.float64); gate = bf([[[.3, -.7, .2]]]).astype(np.float64)
    incoming = bf(np.arange(n*width).reshape(1, n, width)/3+2).astype(np.float64)
    g82 = bf(gate*y82).astype(np.float64); g83 = bf(gate*y83).astype(np.float64)
    f = {'rowBasis': u, 'updateFP64': y64, 'update83': y83, 'gated83': g83, 'exit83': bf(incoming+g83).astype(np.float64),
        'update82': y82, 'gated82': g82, 'exit82': bf(incoming+g82).astype(np.float64), 'gate': gate,
        'text': incoming[:, :text], 'image': incoming[:, text:]}
    before = {name: a.copy() for name, a in f.items()}; calls = []
    def mm(name, a, b):
        calls.append((name, a.shape, b.shape)); return a@b
    result = compute(f, mm, text)
    def scalar_mm(a, b):
        return np.array([[sum(float(a[i, k])*float(b[k, j]) for k in range(a.shape[1])) for j in range(b.shape[1])] for i in range(a.shape[0])])
    expected = np.empty_like(y82)
    for lo, hi in [(0, text), (text, n)]:
        k = scalar_mm(u[0, lo:hi].T, y82[0, lo:hi]); expected[0, lo:hi] = scalar_mm(u[0, lo:hi], k)
    np.testing.assert_array_equal(result['projectedOutput'], expected)
    for name, value in [('projectionResidual', y82-expected), ('finitePolicy', expected-y64), ('admission', y64-y83),
        ('gateRemainder', (g82-g83)-gate*(y82-y83)), ('commitRemainder', (f['exit82']-f['exit83'])-(g82-g83))]:
        np.testing.assert_array_equal(result[name], value)
    for prefix, names in [('update', UPDATE_TERMS), ('exit', EXIT_TERMS)]:
        for lo, hi in [(0, text), (text, n)]:
            means = [[math.fsum(float(result[name][0, row, k]) for row in range(lo, hi))/(hi-lo) for k in range(width)] for name in names]
            for row in range(lo, hi):
                for i, a in enumerate(names):
                    for j, b in enumerate(names):
                        full = math.fsum(float(result[a][0, row, k])*float(result[b][0, row, k]) for k in range(width))
                        centered = math.fsum((float(result[a][0, row, k])-means[i][k])*(float(result[b][0, row, k])-means[j][k]) for k in range(width))
                        np.testing.assert_allclose(result[prefix+'GramRows'][row, i, j], full, rtol=2e-14, atol=2e-14)
                        np.testing.assert_allclose(result[prefix+'CenteredGramRows'][row, i, j], centered, rtol=2e-14, atol=2e-14)
    for name, value in f.items():
        np.testing.assert_array_equal(value, before[name])
    assert {name: {'shape': list(a.shape), 'dtype': str(a.dtype)} for name, a in result.items()} == abis(text, n-text, width, rank)
    assert [a[0] for a in calls] == ['project0', 'reconstruct0', 'project1', 'reconstruct1']
    assert sum(np.prod(a)*b[-1] for _, a, b in calls) == work(text, n-text, width, rank)['matrixMACs']
    rows = list(observations(result, text)); panel = summarize(rows)
    np.testing.assert_allclose(panel['exit']['full']['directDifferenceEnergy'], np.sum((f['exit82']-f['exit83'])**2), rtol=2e-14)
    np.testing.assert_allclose(panel['exit']['full']['energyAccountRemainder'], 0, rtol=0, atol=2e-13)
    np.testing.assert_allclose(panel['exit']['centered']['energyAccountRemainder'], 0, rtol=0, atol=2e-13)
    # A deficient saved basis is a diagnostic specimen, not a reason to abort.
    deficient = {**f, 'rowBasis': np.zeros_like(u)}
    zero = compute(deficient, lambda name, a, b: a@b, text)
    np.testing.assert_array_equal(zero['projectedOutput'], np.zeros_like(y82))
    np.testing.assert_array_equal(zero['projectionResidual'], y82)
    assert work()['matrixMACs'] == 1207959552 and work()['newLearnedMapApplications'] == 0
    assert np.any(result['gateRemainder']) and np.any(result['commitRemainder']) and np.any(result['admission'])
    return {'schema': 'tce-fixed-row-projection-toy-v1', 'sourceABIs': len(f), 'outputABIs': len(result), 'projectorContractions': len(calls),
        'independentScalarProjectionExact': True, 'independentFullCenteredGramReferences': True, 'independentSavedGateCommitRemainders': True,
        'sourceUnchanged': True, 'deficientBasisRetained': True, 'nativePretrainedEvaluations': 0,
        'scope': 'Tiny synthetic saved-number specimen only. Scalar contractions, signed Grams, finite BF16-like saved boundaries and explicit closures; no real model arrays or native accumulator attestation.'}


class FixedRowProjectionTests(unittest.TestCase):
    def test_independent_fixed_projection_and_boundary_accounts(self):
        self.assertTrue(mechanics()['independentScalarProjectionExact'])

    def test_fresh_no_neural_runtime_guard(self):
        import subprocess
        import sys
        from pathlib import Path
        code = "from encoder_atoms import guard; guard(); from test_fixed_row_projection import mechanics; mechanics(); import importlib;\ntry: importlib.import_module('torch')\nexcept ImportError: pass\nelse: raise AssertionError('Neural runtime admitted')\ntry: open('/mnt/big/llm-models/forbidden.safetensors','rb')\nexcept PermissionError: pass\nelse: raise AssertionError('Checkpoint admitted')\nprint('GUARDED_TINY_OK')"
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('GUARDED_TINY_OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
