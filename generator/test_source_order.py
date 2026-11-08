"""Tiny supplied-array mechanics; no pretrained model/legacy scientific arrays."""
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
from source_order_compile import compile_order, scores
from source_order_program import OrderProgram, offset


def fixture(out):
    alphabet = [10, 20, 30]
    q = np.asarray([[1., 0.] if i % 2 == 0 else [0., 1.] for i in range(7)])
    k = np.asarray([[[float(t+j), float(30-t-j)] for j in range(7)] for t in [1, 2, 3]])
    meta = compile_order(q, k, alphabet, 99, out)
    return q, k, meta


class Mechanics(unittest.TestCase):
    def test_batched_products_and_fsum_match_scalar_words(self):
        q = np.asarray([1e30, 1., -1e30, 1e-20], dtype=np.float64)
        k = np.asarray([[1., 1., 1., 2.], [.25, 3., .25, 4.], [2., -1., 2., 0.]], dtype=np.float64)
        expected = np.asarray([math.fsum(float(a)*float(b) for a, b in zip(q, row)) for row in k])
        for batch in [1, 2, 128]:
            work = {'coordinateProducts': 0, 'fsumCalls': 0}
            result = scores(q, k, work, batch)
            self.assertEqual(result.tobytes(), expected.tobytes())
            self.assertEqual(work, {'coordinateProducts': 12, 'fsumCalls': 3})
        products = np.multiply(k, q, dtype=np.float64)
        scalar_products = np.asarray([[float(a)*float(b) for a, b in zip(q, row)] for row in k])
        self.assertEqual(products.tobytes(), scalar_products.tobytes())

    def test_offsets_cover_exact_triangular_layout(self):
        found = [offset(i, t, j, 3) for i in range(7) for t in range(3) for j in range(i)]
        self.assertEqual(found, list(range(63)))
        with self.assertRaises(ValueError):
            offset(2, 0, 2, 3)

    def test_integer_order_matches_independent_subset_scores(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'program'
            q, k, meta = fixture(p)
            program = OrderProgram.load(p/'program.json', p/'order.u16.gz')
            packet = {'ids': [10, 20, 30, 99, 99, 99, 99], 'mask': [1, 1, 1, 0, 0, 0, 0], 'positions': list(range(7))}
            rows, _ = program.serve(packet)
            for clock in range(3, 7):
                values = [math.fsum(float(a)*float(b) for a, b in zip(q[clock], k[j, j])) for j in range(3)]
                expected = sorted(range(3), key=lambda j: (-values[j], j))+[-1]
                self.assertEqual(rows[clock]['addresses'], expected)
            self.assertNotEqual(rows[3]['addresses'], rows[4]['addresses'])
            with gzip.open(p/'order.u16.gz', 'rb') as f:
                self.assertEqual(len(f.read()), meta['rawBytes'])

    def test_ties_use_positions_before_token_ids(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'program'
            compile_order(np.zeros((4, 2)), np.zeros((2, 4, 2)), [10, 20], 99, p)
            program = OrderProgram.load(p/'program.json', p/'order.u16.gz')
            rows, _ = program.serve({'ids': [20, 10, 99], 'mask': [1, 1, 0], 'positions': [0, 1, 2]})
            self.assertEqual(rows[-1]['addresses'], [0, 1, -1, -1])

    def test_coverage_empty_prefix_and_sticky_unknown(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'program';fixture(p)
            program = OrderProgram.load(p/'program.json', p/'order.u16.gz')
            rows, _ = program.serve({'ids': [99, 123, 10, 99, 123, 99], 'mask': [0, 0, 1, 0, 1, 0], 'positions': list(range(6))})
            self.assertEqual(rows[0]['addresses'], [-1]*4)
            self.assertEqual(rows[1]['status'], 'wrong-query-token')
            self.assertEqual(rows[2]['status'], 'active-query-outside-padding-contract')
            self.assertEqual(rows[3]['addresses'], [2, -1, -1, -1])
            self.assertEqual(rows[-1]['status'], 'unknown-admitted-token')
            self.assertEqual(rows[-1]['unknownPositions'], [4])

    def test_controls_and_last_clock_inapplicable(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'program';fixture(p)
            program=OrderProgram.load(p/'program.json',p/'order.u16.gz')
            packet={'ids':[10,20,30,99,99,99,99],'mask':[1,1,1,0,0,0,0],'positions':list(range(7))}
            primary,_=program.serve(packet);wrong,_=program.serve(packet,'clockPlus1');reverse,_=program.serve(packet,'reverseRanks')
            self.assertEqual(wrong[3]['addresses'],primary[4]['addresses'])
            self.assertEqual(wrong[-1]['status'],'control-clock-outside-domain')
            self.assertEqual(reverse[3]['addresses'][:3],list(reversed(primary[3]['addresses'][:3])))

    def test_serialization_reproducible_and_corruption_refused(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root/'one');fixture(root/'two')
            a=(root/'one/order.u16.gz').read_bytes();b=(root/'two/order.u16.gz').read_bytes();self.assertEqual(a,b)
            (root/'bad.gz').write_bytes(a+b'x')
            with self.assertRaises(ValueError):OrderProgram.load(root/'one/program.json',root/'bad.gz')

    def test_fresh_serving_file_and_import_exclusions(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixture(root/'program')
            packet={'ids':[10,20,99,99],'mask':[1,1,0,0],'positions':list(range(4))}
            (root/'tokens.json').write_text(json.dumps(packet));(root/'scores.json').write_text('{}')
            metadata=root/'program/program.json';bundle=root/'program/order.u16.gz'
            spec={'metadata':str(metadata),'metadataSHA256':hashlib.sha256(metadata.read_bytes()).hexdigest(),'bundle':str(bundle),
                'cases':[{'id':'toy','tokens':str(root/'tokens.json')}],'readAllow':[str(metadata),str(bundle),str(root/'tokens.json')],
                'forbiddenPaths':[str(root/'scores.json')]}
            (root/'spec.json').write_text(json.dumps(spec))
            process=Path(__file__).resolve().parent/'source_order_program.py'
            r=subprocess.run([sys.executable,'-I','-B',str(process),str(root/'spec.json'),str(root/'served')],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            receipt=json.loads((root/'served/receipt.json').read_text())
            self.assertEqual(receipt['fileCanaryDenials'],[str(root/'scores.json')]);self.assertEqual(receipt['blockedImportCanaries'],['numpy','torch'])
            self.assertEqual(len(receipt['rows']),4)


if __name__=='__main__':unittest.main()
