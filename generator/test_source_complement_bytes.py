"""Independent hostile fixture for whole-byte selected/complement witnesses."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from publish_source_complement import BYTES


def specimen(tmp_path, corrupt=False):
    root = tmp_path.resolve()
    support = [1, 4]
    bank = np.arange(24, dtype=np.uint16).reshape(1, 6, 2, 2)
    selected = (100 + np.arange(8, dtype=np.uint16)).reshape(1, 2, 2, 2)
    hybrid = bank.copy()
    hybrid[:, support] = selected
    if corrupt:
        hybrid[0, 2, 0, 0] += 1
    items = {}
    for name, a in [('selected', selected), ('bank', bank), ('hybrid', hybrid)]:
        p = root / (name + '.npy')
        np.save(p, a, allow_pickle=False)
        items[name] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                       'shape': list(a.shape), 'payloadSHA256': hashlib.sha256(a.tobytes()).hexdigest()}
    return {'allowedRoots': [str(root)], 'items': list(items.values()),
            'spliceChecks': [{'arm': 'fixture', 'field': 'K', 'support': support, **items}]}


def test_byte_splice_proves_selected_and_complement_rows(tmp_path):
    result = subprocess.run([sys.executable, '-c', BYTES], input=json.dumps(specimen(tmp_path)), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    witness = json.loads(result.stdout)['bankSpliceWitnesses'][0]
    assert witness['selectedRowsByteExact'] and witness['outsideSupportByteExact']


def test_valid_rehashed_file_still_rejects_a_changed_complement_row(tmp_path):
    result = subprocess.run([sys.executable, '-c', BYTES], input=json.dumps(specimen(tmp_path, corrupt=True)), text=True, capture_output=True)
    assert result.returncode != 0
    assert 'Hybrid complement differs outside declared support' in result.stderr
