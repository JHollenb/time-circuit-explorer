"""Post-seal CPU references. Intentionally independent of guarded serving imports."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_array(ref):
    if sha(ref['path']) != ref['sha256']:
        raise ValueError('Reference operand changed')
    array = np.load(ref['path'], allow_pickle=False)
    if list(array.shape) != ref['shape'] or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError('Reference state contract differs')
    if np.any(array.view(np.uint32) & np.uint32(65535)):
        raise ValueError('Reference requires exact widened BF16 state')
    return array


def execute(spec_path, output):
    started, cpu = time.perf_counter(), time.process_time()
    spec, output = read(spec_path), Path(output)
    if sha(spec['seal']['path']) != spec['seal']['sha256']:
        raise ValueError('Primary must be sealed before native evaluation')
    if sha(spec['source']['path']) != spec['source']['sha256']:
        raise ValueError('Inspected native implementation source changed')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    if torch.__version__.split('+')[0] != '2.13.0':
        raise ValueError('Unqualified native CPU version')
    if sha(spec['modulation']['path']) != spec['modulation']['sha256']:
        raise ValueError('Fixed modulation changed')
    with np.load(spec['modulation']['path'], allow_pickle=False) as data:
        factor, shift = data['factor'], data['shift']
    factor_t = torch.from_numpy(factor.copy()).to(torch.bfloat16)
    shift_t = torch.from_numpy(shift.copy()).to(torch.bfloat16)
    layer_norm = torch.nn.LayerNorm(3072, elementwise_affine=False, eps=spec['eps'], device='cpu')
    layers, weights = {}, {}
    import retrieve_weights as retrieval
    manifest = retrieval.validate_manifest(spec['parameters'])
    for entry, stream, size in retrieval._source_entries(manifest, spec['checkpointRoot']):
        raw = b''.join(retrieval._artifact_chunks(entry, stream, size))
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Native parameter differs')
        words = np.load(io.BytesIO(raw), allow_pickle=False)
        value = (words.astype(np.uint32) << 16).view(np.float32)
        name = entry['source']['tensor'].split('.')[-2].removeprefix('to_')
        if name in layers or name not in {'q', 'k', 'v'} or value.shape != (3072, 3072):
            raise ValueError('Native receiver geometry differs')
        layer = torch.nn.Linear(3072, 3072, bias=False, device='meta', dtype=torch.bfloat16)
        layer.weight = torch.nn.Parameter(torch.from_numpy(value.copy()).to(torch.bfloat16), requires_grad=False)
        layers[name], weights[name] = layer, value.astype(np.float64)
        del raw, words, value
    if set(layers) != {'q', 'k', 'v'}:
        raise ValueError('Native projections incomplete')
    records, diagnostics = [], []
    with torch.inference_mode():
        for packet in spec['packets']:
            array = checked_array(packet)
            state = torch.from_numpy(array.copy()).to(torch.bfloat16)
            wall0, cpu0 = time.perf_counter(), time.process_time()
            normalized = layer_norm(state)
            # Exact inspected native expression, with its two BF16 arithmetic boundaries.
            modulated = factor_t * normalized + shift_t
            values = {name: layer(modulated).float().numpy() for name, layer in layers.items()}
            wall_s, cpu_s = time.perf_counter() - wall0, time.process_time() - cpu0
            if not all(np.isfinite(v).all() for v in values.values()):
                raise ValueError('Native receiver nonfinite')
            path = output / ('native-' + packet['id'] + '.npz')
            np.savez(path, **values)
            records.append({'id': packet['id'], 'inputSHA256': packet['sha256'], 'path': path.name,
                            'sha256': sha(path), 'wallSeconds': wall_s, 'cpuSeconds': cpu_s})
            if packet['arm'] == 'predicted':
                primary_path = output / packet['prediction']['path']
                if sha(primary_path) != packet['prediction']['sha256']:
                    raise ValueError('Primary changed before smooth diagnostic')
                with np.load(primary_path, allow_pickle=False) as data:
                    residual, scale = data['residual'], data['scale']
                centered = array.astype(np.float64) - array.mean(axis=-1, keepdims=True, dtype=np.float64)
                smooth_input = factor * centered / scale + shift
                diagnostic = {}
                for name, weight in weights.items():
                    diagnostic[name + '_smooth'] = smooth_input @ weight.T
                    diagnostic[name + '_residual'] = (factor * residual / scale) @ weight.T
                path = output / ('smooth-' + packet['id'] + '.npz')
                np.savez(path, **diagnostic)
                diagnostics.append({'id': packet['id'], 'path': path.name, 'sha256': sha(path)})
    if sha(spec['seal']['path']) != spec['seal']['sha256']:
        raise ValueError('Native evaluation mutated primary seal')
    receipt = {'schema': 'tce-state-receiver-native-v1', 'specSHA256': sha(spec_path),
               'records': records, 'diagnostics': diagnostics, 'wallSecondsAfterImports': time.perf_counter() - started,
               'cpuSecondsAfterImports': time.process_time() - cpu,
               'peakRSSMiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               'profile': {'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version,
                           'device': 'cpu', 'dtype': 'bfloat16', 'threads': torch.get_num_threads(),
                           'interopThreads': torch.get_num_interop_threads(), 'deterministic': True,
                           'mkldnnEnabled': torch.backends.mkldnn.enabled,
                           'cpuCapability': torch.backends.cpu.get_cpu_capability(),
                           'nativeCUDAParity': 'unmeasured', 'source': spec['source']},
               'work': {'receiverEvaluations': len(records), 'nativeLayerNormCalls': len(records),
                        'nativeLinearCalls': 3 * len(records), 'fullBlockCalls': 0, 'encoderCalls': 0,
                        'vaeCalls': 0, 'cudaCalls': 0, 'originalParameterBytes': 3 * 3072 * 3072 * 2,
                        'fullCheckpointBytesHashed': Path(spec['checkpointFile']).stat().st_size,
                        'smoothDiagnosticOriginalDenseMatmuls': 6 * len(diagnostics)},
               'scope': spec['scope']}
    with (output / 'native-receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
    print('NATIVE_RECEIVER_REFERENCES_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    execute(args.spec, args.output)
