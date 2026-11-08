"""Independent tiny current-operand sketch/write/signed/typed contracts."""
import unittest


def mechanics():
    import numpy as np
    import torch
    from operand_row_output import compute, abis, work
    from operand_row_output_analysis import accounts, summarize, ROW_COLUMNS
    prefixes = [[1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1], [1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1]]
    def scalar_mm(a, b):
        return np.array([[sum(float(a[i, k])*float(b[k, j]) for k in range(a.shape[1])) for j in range(b.shape[1])] for i in range(a.shape[0])])
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(85); n, text, dim, mlp, rank = 7, 3, 4, 8, 2
        source = {'product': torch.randn(1, n, mlp).bfloat16(), 'attentionRead': torch.randn(1, n, dim).bfloat16(), 'gate': torch.randn(1, 1, dim).bfloat16(),
            'text': torch.randn(1, text, dim).bfloat16(), 'image': torch.randn(1, n-text, dim).bfloat16()}; weight = torch.randn(dim, dim+mlp).bfloat16()
        calls = []; joins = []; sketches = []; qr_calls = []; rng = torch.get_rng_state(); nrng = np.random.get_state()
        def mm(name, a, b):
            assert a.dtype == b.dtype == torch.float64; calls.append((name, list(a.shape), list(b.shape))); return a@b
        def qr(name, value):
            qr_calls.append(name); return torch.linalg.qr(value, mode='reduced')
        got, trace = compute(source, weight, mm, qr, lambda value: joins.append(value.clone()), lambda stream, raw, omega: sketches.append((stream, raw.copy(), omega.clone())), text, rank)
        assert torch.equal(rng, torch.get_rng_state())
        after = np.random.get_state(); assert nrng[0] == after[0] and np.array_equal(nrng[1], after[1]) and nrng[2:] == after[2:]
        joined = torch.cat([source['attentionRead'], source['product']], -1)
        assert len(joins) == 1 and torch.equal(joins[0], joined)
        expected = []; maximum = 0.
        for stream, (lo, hi) in enumerate([(0, text), (text, n)]):
            raw = np.random.PCG64(np.random.SeedSequence([20261107, 85, stream])).random_raw((dim+mlp, rank))
            omega = np.array([1. if int(word) & 1 else -1. for word in raw.flat]).reshape(dim+mlp, rank)
            np.testing.assert_array_equal(omega.flat[:16], prefixes[stream]); np.testing.assert_array_equal(raw, sketches[stream][1])
            np.testing.assert_array_equal(omega, got['sketchSigns'][stream].numpy()); np.testing.assert_array_equal(omega, sketches[stream][2].numpy())
            j = joined[0, lo:hi].double().numpy(); f = scalar_mm(j, omega); u = got['rowBasis'][0, lo:hi].numpy(); rr = got['qrR'][stream].numpy()
            np.testing.assert_allclose(f, got['sketch'][0, lo:hi].numpy(), rtol=3e-14, atol=3e-14)
            np.testing.assert_allclose(scalar_mm(u, rr), f, rtol=3e-14, atol=3e-14)
            orth = []
            for column in f.T:
                value = column.copy()
                for previous in orth: value -= sum(value*previous)*previous
                orth.append(value/np.sqrt(sum(value*value)))
            q = np.stack(orth, 1); np.testing.assert_allclose(u@u.T, q@q.T, rtol=5e-14, atol=5e-14); assert np.all(np.diag(rr) >= 0)
            k = scalar_mm(u.T, j); b = scalar_mm(k, weight.double().numpy().T); y = scalar_mm(u, b); expected.append(y)
            maximum = max(maximum, float(np.max(np.abs(y-got['updateFP64'][0, lo:hi].numpy()))))
            np.testing.assert_allclose(y, got['updateFP64'][0, lo:hi].numpy(), rtol=3e-13, atol=3e-13)
        update = torch.from_numpy(np.concatenate(expected)[None]).float().bfloat16(); gated = source['gate']*update; exited = torch.cat([source['text'], source['image']], 1)+gated
        for name, value in [('preResidualUpdate', update), ('gatedWrite', gated), ('exit', exited)]: assert torch.equal(value, got[name]), name
        assert {name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in got.items()} == abis(dim, mlp, text, n-text, rank)
        assert [a[0] for a in calls] == [name+str(s) for s in range(2) for name in ['sketch', 'projectInputs', 'compressedOut', 'reconstructUpdate']] and len(qr_calls) == 2
        assert sum(np.prod(a)*b[-1] for _, a, b in calls) == work(dim, mlp, text, n-text, rank)['candidateMACs']
        old82, old83, target73 = [torch.randn(1, n, dim, dtype=torch.float64).numpy() for _ in range(3)]
        candidate = got['updateFP64'].numpy(); rows = accounts(candidate, old82, old83, target73, text); result = summarize(rows)
        assert rows.shape == (n, len(ROW_COLUMNS))
        for index, (old, target) in enumerate([(old83, old82), (old83, target73), (old82, target73)]):
            for i in range(n):
                cross = 2*sum(float(old[0, i, k]-target[0, i, k])*float(candidate[0, i, k]-old[0, i, k]) for k in range(dim))
                np.testing.assert_allclose(rows[i, 48+4*index], cross, rtol=3e-14, atol=3e-14)
        for value in result['signedAccounts'].values():
            np.testing.assert_allclose(value['energyAccountRemainder'], 0, rtol=0, atol=2e-12); np.testing.assert_allclose(value['centeredEnergyAccountRemainder'], 0, rtol=0, atol=2e-12)
        costs = work(); assert costs['candidateMACs'] == 15099494400 and costs['diagnosticMACs'] == 50331648 and costs['assayMACs'] == 15149826048 and costs['prospectiveComposedProducerMACs'] == 34426847232
        assert trace['admittedUpdate']['stride'] == [n*dim, dim, 1] and trace['admittedUpdate']['storageOffset'] == 0
        return {'schema': 'tce-current-operand-row-output-toy-v1', 'candidateProducts': 8, 'QRCalls': 2, 'producerABIs': len(got), 'maxIndependentUpdateAbsError': maximum,
            'nativeGateResidualExact': True, 'globalTorchAndNumPyRNGUnchanged': True, 'actualJoinBeforePromotion': True, 'fixedPCG64LowBitPrefixes': prefixes, 'nativePretrainedModelEvaluations': 0,
            'scope': 'Tiny scalar sketch/output contractions, independent Gram-Schmidt span, fixed raw-PCG prefixes, actual final admissions/gate/add/layout and signed primary/secondary/centered accounts. No pretrained or saved scientific arrays.'}


class OperandRowOutputTests(unittest.TestCase):
    def test_independent_current_operand_projection_and_accounts(self):
        self.assertTrue(mechanics()['nativeGateResidualExact'])

    def test_actual_typed_transaction_and_deficient_sketch(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import torch
        from operand_row_output import compute, abis
        from operand_row_output_ledger import Ledger
        from paired_fields import retain
        from publish_terminal_trajectory import check_act
        with torch.inference_mode(), tempfile.TemporaryDirectory() as directory, patch('torch.cuda.synchronize', lambda: None):
            source = {'product': torch.zeros(1, 8, 8, dtype=torch.bfloat16), 'attentionRead': torch.zeros(1, 8, 4, dtype=torch.bfloat16), 'gate': torch.ones(1, 1, 4, dtype=torch.bfloat16), 'text': torch.ones(1, 4, 4, dtype=torch.bfloat16), 'image': torch.ones(1, 4, 4, dtype=torch.bfloat16)}
            weight = torch.ones(4, 12, dtype=torch.bfloat16); ledger = Ledger('toy-request', 'toy-manifest')
            def operation(values):
                return compute({n: values[n] for n in source}, values['originalOut'], lambda name, a, b: a@b, lambda name, f: torch.linalg.qr(f, mode='reduced'), lambda joined: None, lambda stream, raw, omega: None, 4, 2)[0]
            got = ledger.run({**source, 'originalOut': weight}, operation, abis(4, 8, 4, 4, 2)); descriptors = {n: retain(Path(directory), n, a) for n, a in got.items()}
            check_act(ledger.records[0], descriptors, 'toy-request', 'toy-manifest')
            self.assertEqual(len(ledger.records[0]['sourceFrame']['slots']), 6); self.assertEqual(len(got), 13)
            self.assertTrue(bool((got['qrR'] == 0).all())); self.assertTrue(bool((got['columnSigns'] == 1).all())); self.assertTrue(bool((got['exit'] == 1).all()))


if __name__ == '__main__':
    unittest.main()
