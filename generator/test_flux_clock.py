import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from encoder_atoms import save, identity
from paired_fields import retain, payload
from flux_clock import CLOCK_WEIGHTS, PORTS, partition, selected_load, install_tables, snapshot, check_mode, clock_path, deny_targets


class ClockMechanics(unittest.TestCase):
    def fixture(self, tmp):
        import torch
        root = Path(tmp)
        def toy():
            m = torch.nn.Module(); m.time_guidance_embed = torch.nn.Module(); m.time_guidance_embed.timestep_embedder = torch.nn.Module()
            for key in ['linear_1', 'linear_2']: m.time_guidance_embed.timestep_embedder.add_module(key, torch.nn.Linear(4, 4, bias=False))
            for key in ['double_stream_modulation_img', 'double_stream_modulation_txt', 'single_stream_modulation']:
                v = torch.nn.Module(); v.linear = torch.nn.Linear(4, 4, bias=False); m.add_module(key, v)
            m.norm_out = torch.nn.Module(); m.norm_out.norm = torch.nn.LayerNorm(4, elementwise_affine=False)
            m.norm_out.linear = torch.nn.Linear(4, 8, bias=False); m.x_embedder = torch.nn.Linear(4, 4, bias=False)
            return m
        clock = torch.tensor([960.], dtype=torch.bfloat16); z = torch.tensor([[1., 2., 3., 4.]], dtype=torch.bfloat16)
        vals = {'temb': z, 'double_img': torch.zeros(1, 24, dtype=torch.bfloat16), 'double_txt': torch.zeros(1, 24, dtype=torch.bfloat16),
            'single': torch.zeros(1, 12, dtype=torch.bfloat16), 'output_affine': torch.tensor([[.25]*4 + [.5]*4], dtype=torch.bfloat16)}
        row = {'clockOperands': {'internal': retain(root, 'clock', clock)}, 'fields': {n: retain(root, n, v) for n, v in vals.items()}}
        table = {'rows': [row], 'contract': {'numericalProfile': snapshot(torch), 'dtype': 'torch.bfloat16', 'latentShape': [1, 2, 4], 'EShape': [1, 3, 4]}}
        return toy, table, clock, z

    def test_table_rejects_wrong_bits_dtype_guidance_and_corruption(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            toy, table, clock, z = self.fixture(tmp); model = toy(); calls = []; install_tables(model, table, 'cpu', {}, calls)
            self.assertEqual(payload(model.time_guidance_embed(clock, None)), payload(z))
            wrong = clock.clone(); wrong.view(torch.int16)[0] ^= 1
            for x, g in [(wrong, None), (clock.float(), None), (clock, torch.ones(1))]:
                with self.assertRaises(ValueError): model.time_guidance_embed(x, g)
            Path(table['rows'][0]['fields']['single']['path']).write_bytes(b'bad clock table')
            with self.assertRaises(ValueError): install_tables(toy(), table, 'cpu', {}, [])

    def test_selected_loader_never_materializes_removed_tensors(self):
        import torch
        from safetensors.torch import save_file
        from flux_clock import header
        with tempfile.TemporaryDirectory() as tmp:
            toy, table, _, _ = self.fixture(tmp); dense = toy().to(torch.bfloat16)
            path = Path(tmp) / 'weights.safetensors'; save_file(dense.state_dict(), str(path))
            entries, _ = header(path); removed, kept = partition(entries)
            self.assertEqual(set(removed), set(CLOCK_WEIGHTS)); self.assertEqual(kept, ['x_embedder.weight'])
            with torch.device('meta'): candidate = toy()
            install_tables(candidate, table, 'cpu', {}, [])
            with self.assertRaises(ValueError): selected_load(candidate, path, removed + kept, entries, 'cpu')
            reads = selected_load(candidate, path, kept, entries, 'cpu')
            self.assertEqual([x['name'] for x in reads], kept)
            self.assertEqual(list(dict(candidate.named_parameters())), kept)
            self.assertTrue(torch.equal(candidate.x_embedder.weight, dense.x_embedder.weight))
            self.assertEqual(payload(candidate.time_guidance_embed.row0), table['rows'][0]['fields']['temb']['payloadSHA256'])

    def test_live_normalization_changes_with_state_under_same_clock(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            toy, table, clock, z = self.fixture(tmp); m = toy(); calls = []; install_tables(m, table, 'cpu', {}, calls)
            x = torch.tensor([[[1., 4., -2., 3.], [3., 1., 0., -1.]]], dtype=torch.bfloat16)
            y = torch.flip(x, [-1]); first = m.norm_out(x, z); second = m.norm_out(y, z)
            self.assertFalse(torch.equal(first, second))
            torch.testing.assert_close(first, torch.nn.functional.layer_norm(x, [4]) * 1.25 + .5, rtol=0, atol=0)
            self.assertEqual([c['port'] for c in calls], ['output_affine', 'output_affine'])

    def test_mode_contract_does_not_bind_scene_values(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            _, table, _, _ = self.fixture(tmp); contract = table['contract']; numerical = snapshot(torch)
            kw = {'hidden_states': torch.zeros(1, 2, 4, dtype=torch.bfloat16), 'encoder_hidden_states': torch.zeros(1, 3, 4, dtype=torch.bfloat16), 'guidance': None}
            check_mode(kw, numerical, contract)
            check_mode({**kw, 'hidden_states': torch.ones_like(kw['hidden_states'])}, numerical, contract)
            for change in [{'guidance': torch.zeros(1)}, {'kv_cache_mode': 'cached'}, {'kv_cache': object()}, {'num_ref_tokens': 1},
                {'hidden_states': kw['hidden_states'].float()}, {'joint_attention_kwargs': {'scale': .5}}]:
                with self.assertRaises(ValueError): check_mode({**kw, **change}, numerical, contract)
            with self.assertRaises(ValueError): check_mode(kw, {**numerical, 'threads': numerical['threads'] + 1}, contract)

    def test_actual_clock_cast_precedes_embedding(self):
        import torch
        t = torch.tensor(958.0853881835938, dtype=torch.float32); trace = clock_path(t, torch)
        self.assertEqual(trace['pipeline_cast'].item(), 960.)
        self.assertEqual(trace['transformer_argument'].dtype, torch.bfloat16)
        self.assertEqual(trace['internal'].item(), 960.)
        self.assertNotEqual(trace['internal'].item(), t.item())

    def test_target_values_denied_until_separate_scoring_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            denied = Path(tmp) / 'target.npy'; allowed = Path(tmp) / 'E.npy'; denied.write_bytes(b'target'); allowed.write_bytes(b'input')
            deny_targets([denied])
            with self.assertRaises(PermissionError): denied.read_bytes()
            self.assertEqual(allowed.read_bytes(), b'input')

    def test_failed_candidate_cannot_release_scoring(self):
        from flux_clock_study import execute
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); request = {'sourceHashes': {}}; request['requestSHA256'] = identity(request); save(root / 'request.json', request)
            def failure(cmd, **kw):
                if cmd[-2] == 'images': raise RuntimeError('mechanics failure')
            with patch('flux_clock_study.subprocess.run', side_effect=failure), patch('flux_clock_study.score') as score:
                with self.assertRaises(RuntimeError): execute(root)
                score.assert_not_called()
            self.assertFalse((root / 'candidate-seal.json').exists())


if __name__ == '__main__': unittest.main()
