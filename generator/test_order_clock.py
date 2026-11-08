"""Tiny supplied-integer mechanics; never read scientific/model operands."""
from array import array
import gzip
import hashlib
import json
from pathlib import Path
from itertools import product
import struct
import subprocess
import sys
import tempfile
import unittest
from order_clock_program import OrderClock, RankTransitions, RankHistory, compile_program, little_bytes


def fixture():
    alphabet = [11, 13]; length = 4
    orders = [[], [0, 4], [4, 0, 5, 1], [4, 0, 5, 1, 2, 6]]
    ranks = array('H')
    for clock, order in enumerate(orders):
        inverse = {v: p for p, v in enumerate(order)}
        ranks.extend(inverse[t*length+j] for t in range(2) for j in range(clock))
    code, report = compile_program(ranks, alphabet, length)
    metadata = {'schema': 'tce-order-clock-v1', 'head': 0, 'paddingQueryID': 151643, 'alphabet': alphabet, 'length': length}
    return metadata, code, report, ranks, orders


def packet(ids, mask):
    return {'ids': ids, 'mask': mask, 'positions': list(range(len(ids)))}


class Mechanics(unittest.TestCase):
    def test_sparse_rank_transport_all_tiny_legal_histories(self):
        meta, code, _, _, _ = fixture(); transitions = RankTransitions(meta, code)
        for choices in product([None, 11, 13], repeat=4):
            p = packet([151643 if x is None else x for x in choices], [int(x is not None) for x in choices])
            for method in ['primary', 'clockPlus1', 'reverseRanks', 'recent']:
                a, _ = RankHistory(transitions).serve(p, method); b, _ = OrderClock(meta, code).serve(p, method)
                self.assertEqual(a, b)
        self.assertEqual(len(transitions.offsets), 5)
        self.assertEqual(transitions.payload()['insertRanksBytes'], 16)

    def test_every_rank_reconstructed_from_empty(self):
        meta, code, _, ranks, orders = fixture(); m = OrderClock(meta, code); actual = array('H')
        for clock in range(meta['length']):
            if clock: m.advance()
            self.assertEqual(list(m.order), orders[clock])
            actual.extend(m.inverse[t*4+j] for t in range(2) for j in range(clock))
        self.assertEqual(actual, ranks); self.assertEqual(m.cursor, len(code))
        self.assertEqual(m.work['expandedIdentities'], len(ranks))
        self.assertEqual(m.work['inverseRankWrites'], len(ranks))

    def test_fixed_grammar_byte_account_and_maximal_run(self):
        _, code, report, _, _ = fixture(); work = report['work']
        self.assertEqual(work['insertOperations'], 6)
        self.assertEqual(len(code), 5*work['copyOperations']+3*6+2*3)
        self.assertEqual(report['clocks'][-1]['copies'], 1)
        self.assertEqual(report['clocks'][-1]['copiedIdentities'], 4)

    def test_sort_restriction_and_controls(self):
        meta, code, _, _, _ = fixture(); p = packet([11, 13, 151643, 151643], [1, 1, 0, 0])
        primary, _ = OrderClock(meta, code).serve(p, 'primary')
        self.assertEqual(primary[2]['addresses'], [0, 1, -1, -1])
        for method in ['clockPlus1', 'reverseRanks', 'recent']:
            rows, _ = OrderClock(meta, code).serve(p, method)
            if method == 'clockPlus1': self.assertEqual(rows[-1]['status'], 'control-clock-outside-domain')
            elif method == 'recent': self.assertEqual(rows[2]['addresses'], [1, 0, -1, -1])
            else: self.assertEqual(rows[2]['addresses'], [1, 0, -1, -1])

    def test_unknown_sticky_masked_unknown_and_empty(self):
        meta, code, _, _, _ = fixture()
        rows, _ = OrderClock(meta, code).serve(packet([151643, 77, 151643, 151643], [0, 1, 0, 0]), 'primary')
        self.assertEqual(rows[0]['addresses'], [-1]*4)
        self.assertEqual(rows[-1]['status'], 'unknown-admitted-token')
        rows, _ = OrderClock(meta, code).serve(packet([77, 151643], [0, 0]), 'primary')
        self.assertEqual(rows[-1]['status'], 'supported')

    def test_invalid_bytecode_and_source_permutation_refused(self):
        meta, code, _, ranks, _ = fixture()
        for bad in [b'', b'\x01\x00\x09', struct.pack('<HBHH', 1, 0, 0, 1), struct.pack('<HBH', 1, 1, 9)]:
            with self.assertRaises(ValueError): OrderClock(meta, bad).advance()
        m = OrderClock(meta, code+b'\0')
        with self.assertRaises(ValueError):
            for _ in range(3): m.advance()
        ranks[0] = ranks[1]
        with self.assertRaises(ValueError): compile_program(ranks, [11, 13], 4)

    def test_little_endian_and_deterministic_compression(self):
        self.assertEqual(little_bytes([0, 256, 65535]), b'\0\0\0\1\xff\xff')
        _, code, _, _, _ = fixture()
        self.assertEqual(gzip.compress(code, compresslevel=9, mtime=0), gzip.compress(code, compresslevel=9, mtime=0))

    def test_fresh_file_import_exclusions_and_replay(self):
        meta, code, _, _, _ = fixture()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); binary = root/'program.gz'
            with binary.open('wb') as f:
                with gzip.GzipFile(fileobj=f, mode='wb', filename='', mtime=0) as stream: stream.write(code)
            meta.update({'rawBytes': len(code), 'rawSHA256': hashlib.sha256(code).hexdigest(),
                         'compressedSHA256': hashlib.sha256(binary.read_bytes()).hexdigest()})
            metadata = root/'meta.json'; metadata.write_text(json.dumps(meta)); tokens = root/'tokens.json'
            tokens.write_text(json.dumps(packet([11, 13, 151643, 151643], [1, 1, 0, 0])))
            forbidden = root/'rank-table.bin'; forbidden.write_bytes(b'cannot read')
            spec = root/'spec.json'; out = root/'out'
            spec.write_text(json.dumps({'metadata': str(metadata), 'metadataSHA256': hashlib.sha256(metadata.read_bytes()).hexdigest(),
                'bytecode': str(binary), 'readAllow': [str(metadata), str(binary), str(tokens)], 'forbiddenPaths': [str(forbidden)],
                'cases': [{'id': 'tiny', 'tokens': str(tokens)}]}))
            result = subprocess.run([sys.executable, '-I', '-B', str(Path(__file__).with_name('order_clock_program.py')), str(spec), str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((out/'receipt.json').read_text())
            self.assertEqual(receipt['blockedImportCanaries'], ['numpy', 'torch'])
            self.assertEqual(receipt['fileCanaryDenials'], [str(forbidden)])
            self.assertEqual(len(json.loads((out/'decoded-clocks.json').read_text())['clocks']), 4)


if __name__ == '__main__': unittest.main()
