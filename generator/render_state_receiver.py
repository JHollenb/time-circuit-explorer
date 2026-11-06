"""Portable state-only reader. NumPy only; no dose, checkpoint or native refresh."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from response_study import read, save, sha


def render(bundle, state, output):
    bundle, state, output = [Path(p).resolve() for p in [bundle, state, output]]
    files = read(bundle / 'files.json')
    needed = ['basis.npy', 'bank-q.npz', 'bank-k.npz', 'bank-v.npz', 'receiver-compile-receipt.json']
    for name in needed:
        if sha(bundle / name) != files[name]:
            raise ValueError('Published reader bundle changed: ' + name)
    receipt = read(bundle / 'receiver-compile-receipt.json')
    output.mkdir(parents=True, exist_ok=False)
    spec = {'schema': 'tce-state-only-reader-input-v1',
            'basis': {'path': str(bundle / 'basis.npy'), 'sha256': receipt['basisSHA256']},
            'banks': {name: {'path': str(bundle / item['path']), 'sha256': item['sha256']} for name, item in receipt['banks'].items()},
            'packets': [{'id': 'input', 'path': str(state), 'sha256': sha(state), 'shape': [1, 32, 3072]}],
            'eps': 1e-6, 'repeats': 1, 'scope': receipt['scope'],
            'sourceRoots': [str(bundle), str(state.parent), str(output)]}
    path = output / 'serve.json'
    save(path, spec)
    subprocess.run([sys.executable, str(Path(__file__).with_name('state_receiver.py')), 'serve', str(path), str(output)], check=True)
    seal = read(output / 'receiver-seal.json')
    return {'output': str(output / seal['rows'][0]['output']), 'sha256': seal['rows'][0]['sha256'],
            'scope': receipt['scope'], 'residualOverCenteredState': seal['rows'][0]['residualOverCenteredState']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.bundle, args.state, args.output)))
