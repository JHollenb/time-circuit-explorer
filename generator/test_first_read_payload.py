"""Tiny supplied mechanics; no historical arrays or learned model execution."""
import struct
import unittest
import numpy as np
from encoder_atoms import words, floats
from first_read_payload import selected_read, effects, row_metrics, aggregate, qualify


class FirstReadMechanics(unittest.TestCase):
    def fixture(self):
        a = np.zeros((4, 128), dtype=np.float64); a[:, :4] = np.eye(4)
        b = a.copy(); b[2, :4] += [2, -4, 6, -8]
        sa = np.array([[90, 30, 10, 50]], dtype=np.int64)
        sb = np.array([[10, 50, 90, 30]], dtype=np.int64)
        wa = np.array([[.125, .25, .125, .5]], dtype=np.float64)
        wb = np.array([[.25, .5, .125, .125]], dtype=np.float64)
        return a, b, sa, sb, wa, wb, np.array([10, 30, 50, 90])

    def corners(self):
        a, b, sa, sb, wa, wb, positions = self.fixture()
        return [selected_read(w, s, v, positions)[0] for w, s, v in [(wa, sa, a), (wa, sa, b), (wb, sb, a), (wb, sb, b)]]

    def test_addresses_resolve_through_map_and_preserve_weight_order(self):
        a, _, sa, _, wa, _, positions = self.fixture()
        actual, work = selected_read(wa, sa, a, positions)
        np.testing.assert_array_equal(actual[0, :4], [.125, .25, .5, .125])
        self.assertEqual(work['scalarValueProducts'], 512)

    def test_factorial_and_single_address_identity(self):
        aa, ab, ba, bb = self.corners(); e = effects(aa, ab, ba, bb)
        a, b, *_ = self.fixture(); dv = b[2]-a[2]
        np.testing.assert_array_equal(e['payload'][0], .5*(.5+.5)*dv)
        np.testing.assert_array_equal(e['interaction'][0], 0*dv)
        np.testing.assert_array_equal(e['delta'], e['payload']+e['allocation'])

    def test_changed_mass_interaction_is_signed(self):
        a, b, sa, sb, wa, wb, positions = self.fixture(); wb = wb.copy(); wb[0] = [.25, .25, .375, .125]
        aa, ab, ba, bb = [selected_read(w, s, v, positions)[0] for w, s, v in [(wa, sa, a), (wa, sa, b), (wb, sb, a), (wb, sb, b)]]
        np.testing.assert_array_equal(effects(aa, ab, ba, bb)['interaction'][0], -.25*(b[2]-a[2]))

    def test_does_not_renormalize_supplied_weights(self):
        a, _, sa, _, wa, _, positions = self.fixture()
        np.testing.assert_array_equal(selected_read(2*wa, sa, a, positions)[0], 2*selected_read(wa, sa, a, positions)[0])

    def test_invalid_addresses_shape_and_negative_weights(self):
        a, _, sa, _, wa, _, positions = self.fixture()
        bad = sa.copy(); bad[0, 1] = 999
        for w, s, v in [(wa, bad, a), (-wa, sa, a), (wa[:, :3], sa, a), (wa, sa, a[:, :4])]:
            with self.assertRaises(ValueError): selected_read(w, s, v, positions)

    def test_bf16_ties_and_fp32_admission(self):
        values = np.array([1+2**-8, 1+3*2**-8, -1-2**-8, 1+2**-8+2**-30], dtype=np.float64)
        oracle = []
        for value in values:
            bits = struct.unpack('<I', struct.pack('<f', float(value)))[0]
            oracle.append((bits+0x7fff+((bits >> 16) & 1)) >> 16)
        self.assertEqual(words(values).tolist(), oracle)
        self.assertEqual(oracle[0], oracle[3])

    def test_signed_gram_cancellation_and_zero_denominator(self):
        p = np.ones((2, 128)); allocation = -p; delta = np.zeros_like(p)
        rows, _ = row_metrics({'payload': p, 'allocation': allocation, 'interaction': delta, 'delta': delta}, p, delta)
        self.assertLess(rows[0]['payloadDotAllocation'], 0)
        self.assertIsNone(rows[0]['payloadNormOverDelta'])
        self.assertEqual(aggregate(rows)['deltaSquaredMinusGram'], 0)

    def test_admission_can_change_factorial_effect(self):
        aa = np.ones((1, 128)); ab = aa+2**-10; ba = aa.copy(); bb = ab.copy()
        self.assertGreater(np.linalg.norm(effects(aa, ab, ba, bb)['payload']), 0)
        admitted = [floats(words(x)).astype(np.float64) for x in [aa, ab, ba, bb]]
        np.testing.assert_array_equal(effects(*admitted)['payload'], np.zeros((1, 128)))

    def test_qualification_allows_zero_payload_but_not_other_addresses(self):
        def packet(token):
            ids = np.zeros(512, dtype=np.int64); ids[8] = token
            return {'ids': ids, 'positions': np.arange(512), 'mask': (np.arange(512) < 35).astype(np.int64),
                'v': np.zeros((512, 128), dtype=np.uint16), 'support': np.tile(np.arange(4), (477, 1)),
                'weights': np.full((477, 4), .25), 'counts': np.full(477, 35)}
        a, b = packet(2115), packet(1290)
        self.assertEqual(qualify(a, b)['changedValuePositions'], [])
        b['v'][9, 0] = words(np.array([1.]))[0]
        with self.assertRaises(ValueError): qualify(a, b)


if __name__ == '__main__':
    unittest.main()
