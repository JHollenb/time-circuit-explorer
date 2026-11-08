"""Supplied byte fixtures only; no pretrained parameters or tokenizer execution."""
import hashlib
import tempfile
from pathlib import Path
import unittest
from template_cofactor_program import PrefixState, save


class TemplateCofactorMechanics(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
        self.bank = b''.join(bytes([i])*10240 for i in range(28))
        (self.root/'prefix.bf16').write_bytes(self.bank)
        save(self.root/'program.json', {'length': 512, 'width': 2560, 'layers': 27, 'prefixIDs': [151644, 872],
            'bankSHA256': hashlib.sha256(self.bank).hexdigest(), 'bankBytes': len(self.bank)})
        self.packet = {'ids': [151644, 872, 198]+[151643]*509, 'mask': [1]*3+[0]*509, 'positions': list(range(512))}

    def test_layer_order_and_explicit_partial_output(self):
        state = PrefixState(self.root); result = state.advance(self.packet)
        self.assertEqual(result['unresolvedPositions'], [2, 512]); self.assertFalse(result['completeH1Producer'])
        self.assertEqual(state.state(1), bytes([1])*10240); self.assertEqual(state.state(27), bytes([27])*10240)

    def test_owns_input_and_reports_later_changes(self):
        state = PrefixState(self.root); state.advance(self.packet); self.packet['ids'][2] = 151935
        self.assertEqual(state.packet['ids'][2], 198)
        self.assertEqual(state.advance(self.packet)['changedInputPositions'], [2])
        self.assertEqual(state.state(1), bytes([1])*10240)

    def test_wrong_prefix_cannot_read_old_state(self):
        state = PrefixState(self.root); state.advance(self.packet); self.packet['ids'][0] = 151645
        self.assertEqual(state.advance(self.packet)['status'], 'unsupported-prefix')
        with self.assertRaises(ValueError): state.state(1)

    def test_mask_and_positions_are_not_ignored(self):
        state = PrefixState(self.root); self.packet['positions'][0] = 1
        with self.assertRaises(ValueError): state.advance(self.packet)
        self.packet['positions'] = list(range(512)); self.packet['mask'][6] = 1
        with self.assertRaises(ValueError): state.advance(self.packet)

    def test_corrupt_compiled_words_refused(self):
        (self.root/'prefix.bf16').write_bytes(b'broken')
        with self.assertRaises(ValueError): PrefixState(self.root)


if __name__ == '__main__':
    unittest.main()
