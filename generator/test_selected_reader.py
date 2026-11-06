import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from encoder_atoms import words, identity, save
from selected_reader import stream, selected_read, logsum
from selected_reader_study import OWNED, execute


class SelectedReaderMechanics(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(31); n, d, heads, rank = 12, 4, 2, 3
        kv = {'k': words(rng.normal(size=(n, d))), 'v': words(rng.normal(size=(n, d))),
            'mask': np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1]), 'positions': np.arange(n)}
        packets = [{**kv, 'q': words(rng.normal(size=(n, d)))} for _ in range(heads)]
        asset = {'landmarks': rng.normal(size=(heads, rank, d)), 'metricRoot': np.eye(d), 'keyCenter': rng.normal(size=d)}
        return packets, asset

    def test_exact_parent_selected_arithmetic_support_routes_and_cached_operands(self):
        from rank_one_tail import stream as parent_stream
        p, a = self.fixture(); candidate, state = stream(p, a)
        parent, old = parent_stream(p, a, np.array([1., 0, 0, 0]), np.array([0., 1, 0, 0]))
        for k in ['supportPositions', 'nearestIndex']: np.testing.assert_array_equal(candidate[k], parent[k])
        for k in ['keys', 'values', 'scores', 'positions']: np.testing.assert_array_equal(state[k], old[k])
        self.assertFalse({'moment', 'absoluteMoment', 'U', 'absoluteU', 'maximum', 'maxAbsScore', 'z'} & state.keys())
        for i in range(len(p[0]['mask'])):
            count = int(candidate['prefixCounts'][i])
            for h in range(2):
                support = parent['supportPositions'][i, h]; support = support[support >= 0]
                if not len(support):
                    self.assertFalse(candidate['rounded'][i].any()); continue
                chosen = np.array([np.flatnonzero(old['positions'][:count] == pos)[0] for pos in support])
                route = parent['nearestIndex'][i, h]
                from encoder_atoms import floats
                delta = floats(p[h]['q'][i]).astype(float)/4**.25 - a['landmarks'][h, route]
                s, U, V = old['scores'][h, route, :count], old['keys'][:count], old['values'][:count]
                logits = s[chosen] + U[chosen] @ delta; expected = np.exp(logits-logsum(logits)) @ V[chosen]
                np.testing.assert_array_equal(candidate['raw'][i, h*4:(h+1)*4], expected)
                if count <= 4: np.testing.assert_array_equal(candidate['rounded'][i, h*4:(h+1)*4], parent['rounded'][i, h*4:(h+1)*4])

    def test_finite_selected_expression_keeps_anchor_delta_order(self):
        U = np.array([[1.], [1.]]); V = np.eye(2); s = np.array([1e16, 1e16+2.]); delta = np.array([-1e16])
        result, logits, weights = selected_read(s, U, V, np.array([0, 1]), delta)
        np.testing.assert_array_equal(logits, [0., 2.]); np.testing.assert_array_equal(result, weights)
        self.assertGreater(weights[1], .8)

    def test_empty_mask_future_changes_and_causal_tie_order(self):
        p, a = self.fixture(); baseline, _ = stream(p, a)
        changed = [{k: v.copy() for k, v in packet.items()} for packet in p]
        for packet in changed: packet['v'][0] = words([100.]*4); packet['q'][-1] = words([-50.]*4)
        other, _ = stream(changed, a)
        np.testing.assert_array_equal(baseline['rounded'][:-1], other['rounded'][:-1])
        for packet in changed: packet['mask'][:] = 0
        empty, _ = stream(changed, a); self.assertFalse(empty['rounded'].any())
        from selected_reader import select
        np.testing.assert_array_equal(select(np.array([1., 1., 1.]), np.array([7, 2, 5]), 2), [1, 2])

    def test_split_npz_continuation_closes_without_parent_mutation(self):
        p, a = self.fixture(); full, _ = stream(p, a); cut = 6
        first, parent = stream([{k: v[:cut] for k, v in x.items()} for x in p], a); before = parent['scores'].copy()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.npz'; np.savez(path, **parent)
            with np.load(path, allow_pickle=False) as f: restored = {k: f[k].copy() for k in f.files}
            second, final = stream([{k: v[cut:] for k, v in x.items()} for x in p], a, restored)
        for k in ['raw', 'rounded', 'supportPositions', 'nearestIndex']:
            np.testing.assert_array_equal(np.concatenate([first[k], second[k]]), full[k])
        np.testing.assert_array_equal(parent['scores'], before); self.assertEqual(parent['steps'], cut); self.assertEqual(final['steps'], 12)

    def test_foreign_geometry_nonfinite_and_wrong_clock_rejected(self):
        p, a = self.fixture(); _, parent = stream([{k: v[:6] for k, v in x.items()} for x in p], a)
        with self.assertRaises(ValueError): stream(p, a, parent)
        other = {**a, 'keyCenter': a['keyCenter']+1}
        with self.assertRaises(ValueError): stream([{k: v[6:] for k, v in x.items()} for x in p], other, parent)
        p[0]['q'][0, 0] = 0x7f80
        with self.assertRaises(ValueError): stream(p, a)

    def test_isolated_payload_has_no_old_readers_or_neural_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in OWNED: shutil.copyfile(Path(__file__).resolve().parent/name, Path(tmp)/name)
            code = 'import sys;sys.path.insert(0,".");from encoder_atoms import guard;guard();import selected_reader,selected_reader_study;assert not {"rank_one_tail","attention_bank","torch","diffusers","transformers"}.intersection(sys.modules)'
            subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp, check=True, capture_output=True, timeout=20)

    def test_failed_child_cannot_seal_score_or_read_parent_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); r = {'sourceHashes': {}}; r['requestSHA256'] = identity(r); save(root/'request.json', r)
            with patch('selected_reader_study.subprocess.run', side_effect=RuntimeError('partial')), patch('selected_reader_study.seal') as seal, patch('selected_reader_study.score') as score, patch('selected_reader_study.parent_diagnostic') as diagnostic:
                with self.assertRaises(RuntimeError): execute(root)
                seal.assert_not_called(); score.assert_not_called(); diagnostic.assert_not_called()


if __name__ == '__main__': unittest.main()
