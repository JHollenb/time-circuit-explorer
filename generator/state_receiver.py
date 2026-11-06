"""A state-only reduced reader of the compiled producer's emitted rows."""
import argparse
import hashlib
import io
from pathlib import Path
import time

from mlp_polynomial import np, read, sha, write, audit_reads, checked_ref, quantize, rms, ratio, peak_mib


def state_basis(coefficients, parent, gate):
    directions = coefficients.astype(np.float64) * gate
    directions[0] += parent
    directions -= directions.mean(axis=-1, keepdims=True)
    columns = directions[:, 0].transpose(1, 2, 0)  # row, channel, direction
    norms = np.linalg.norm(columns, axis=1)
    scaled = columns / np.where(norms == 0, 1, norms)[:, None, :]
    left, values, _ = np.linalg.svd(scaled, full_matrices=False)
    cutoffs = max(columns.shape[1:]) * np.finfo(np.float64).eps * values[:, :1]
    keep = values > cutoffs
    basis = (left * keep[:, None, :]).transpose(0, 2, 1)
    return basis, {'columnNorms': norms.tolist(), 'scaledSingularValues': values.tolist(),
                   'cutoffs': cutoffs[:, 0].tolist(), 'ranks': keep.sum(axis=1).tolist(),
                   'construction': 'Centered smooth producer state coefficient directions; column normalization then SVD. No native outputs or receiver targets.'}


def read_state(packet):
    if set(packet) != {'id', 'path', 'sha256', 'shape'}:
        raise ValueError('Receiver packet admits only identity and state bytes')
    if sha(packet['path']) != packet['sha256']:
        raise ValueError('Emitted state file changed')
    state = np.load(packet['path'], allow_pickle=False)
    if list(state.shape) != packet['shape'] or state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError('Emitted state contract differs')
    if np.any(state.view(np.uint32) & np.uint32(65535)):
        raise ValueError('Expected exact widened BF16 emitted state')
    return state


def receive(state, basis, banks, offsets, eps):
    """No dose or native-state input; all coordinates come from actual state."""
    value = state.astype(np.float64)
    centered = value - value.mean(axis=-1, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + eps)
    coordinates = np.einsum('brd,rkd->brk', centered, basis)
    reconstruction = np.einsum('brk,rkd->brd', coordinates, basis)
    residual = centered - reconstruction
    outputs = {}
    for name in ['q', 'k', 'v']:
        raw = np.einsum('brk,rko->bro', coordinates, banks[name]) / scale + offsets[name]
        outputs[name + '_raw'] = raw
        outputs[name] = quantize(raw)
    return {**outputs, 'residual': residual, 'scale': scale, 'coordinates': coordinates}


def compile_receiver(spec_path, output):
    started, cpu = time.perf_counter(), time.process_time()
    spec = read(spec_path)
    allowed = [spec_path, spec['coefficients']['path'], spec['consumer']['path'],
               spec['modulation']['path'], spec['checkpointFile']]
    opened = audit_reads(allowed, spec['sourceRoots'])
    for name in ['coefficients', 'consumer']:
        if sha(spec[name]['path']) != spec[name]['sha256']:
            raise ValueError('Producer bundle changed')
    coefficients = np.load(spec['coefficients']['path'], allow_pickle=False)
    with np.load(spec['consumer']['path'], allow_pickle=False) as context:
        parent, gate = context['parent'].astype(np.float64), context['gate'].astype(np.float64)
    if coefficients.shape != (5, *parent.shape):
        raise ValueError('Producer shape differs')
    basis, diagnostic = state_basis(coefficients, parent, gate)
    modulation = checked_ref(spec['modulation']).reshape(1, 1, -1)
    shift, scale, _, _, _, _ = np.split(modulation, 6, axis=-1)
    factor = quantize(1 + scale).astype(np.float64)
    output = Path(output)
    np.save(output / 'basis.npy', basis, allow_pickle=False)
    np.savez(output / 'modulation.npz', factor=factor, shift=shift)
    import retrieve_weights as retrieval
    manifest = retrieval.validate_manifest(spec['parameters'])
    banks = {}
    for entry, stream, size in retrieval._source_entries(manifest, spec['checkpointRoot']):
        raw = b''.join(retrieval._artifact_chunks(entry, stream, size))
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Original projection parameter differs')
        words = np.load(io.BytesIO(raw), allow_pickle=False)
        weight = (words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        name = entry['source']['tensor'].split('.')[-2].removeprefix('to_')
        if name not in {'q', 'k', 'v'} or name in banks or weight.shape != (parent.shape[-1], parent.shape[-1]):
            raise ValueError('Unexpected receiver parameter')
        modulated_basis = basis * factor[0, 0]
        bank = (modulated_basis.reshape(-1, basis.shape[-1]) @ weight.T).reshape(*basis.shape[:2], -1)
        offset = shift @ weight.T
        path = output / ('bank-' + name + '.npz')
        np.savez(path, bank=bank, offset=offset)
        banks[name] = {'path': path.name, 'sha256': sha(path), 'payloadBytes': bank.nbytes + offset.nbytes,
                       'parameterBytes': size, 'parameter': entry['source']['tensor']}
        del raw, words, weight, bank
    if set(banks) != {'q', 'k', 'v'} or not np.isfinite(basis).all():
        raise ValueError('Receiver bank incomplete')
    write(output / 'receiver-compile-receipt.json', {'schema': 'tce-state-receiver-compile-v1',
          'compileSpecSHA256': sha(spec_path), 'basisSHA256': sha(output / 'basis.npy'),
          'basisPayloadBytes': basis.nbytes, 'banks': banks, 'basis': diagnostic,
          'modulationSHA256': sha(output / 'modulation.npz'),
          'wallSeconds': time.perf_counter() - started, 'cpuSeconds': time.process_time() - cpu,
          'peakRSSMiB': peak_mib(), 'opened': sorted(opened), 'offlineDenseLearnedWeightMath': True,
          'fullCheckpointBytesHashed': Path(spec['checkpointFile']).stat().st_size,
          'scope': spec['scope']})


def serve(spec_path, output):
    started, cpu = time.perf_counter(), time.process_time()
    spec, output = read(spec_path), Path(output)
    if set(spec) != {'schema', 'basis', 'banks', 'packets', 'eps', 'repeats', 'scope', 'sourceRoots'}:
        raise ValueError('Unexpected receiver input field')
    paths = [spec_path, spec['basis']['path'], output / 'receiver-seal.json']
    paths += [p['path'] for p in spec['packets']] + [b['path'] for b in spec['banks'].values()]
    paths += [output / ('prediction-' + p['id'] + '.npz') for p in spec['packets']]
    opened = audit_reads(paths, spec['sourceRoots'], serving=True)
    if sha(spec['basis']['path']) != spec['basis']['sha256']:
        raise ValueError('Basis changed')
    basis = np.load(spec['basis']['path'], allow_pickle=False)
    banks, offsets = {}, {}
    for name, ref in spec['banks'].items():
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('Compiled reader bank changed')
        with np.load(ref['path'], allow_pickle=False) as data:
            banks[name], offsets[name] = data['bank'], data['offset']
    if not 0 < spec['eps'] < 1 or set(banks) != {'q', 'k', 'v'}:
        raise ValueError('Receiver profile differs')
    rows = []
    for packet in spec['packets']:
        state = read_state(packet)
        if basis.shape[0] != state.shape[1] or basis.shape[2] != state.shape[2]:
            raise ValueError('Basis/state geometry differs')
        receive(state, basis, banks, offsets, spec['eps'])
        samples, cpu_samples = [], []
        for _ in range(spec['repeats']):
            wall0, cpu0 = time.perf_counter(), time.process_time()
            values = receive(state, basis, banks, offsets, spec['eps'])
            samples.append(time.perf_counter() - wall0)
            cpu_samples.append(time.process_time() - cpu0)
        if not all(np.isfinite(v).all() for v in values.values()):
            raise ValueError('Predicted reader not finite')
        path = output / ('prediction-' + packet['id'] + '.npz')
        np.savez(path, **values)
        residual = values['residual']
        centered = state.astype(np.float64) - state.mean(axis=-1, keepdims=True, dtype=np.float64)
        rows.append({'id': packet['id'], 'inputSHA256': packet['sha256'], 'output': path.name, 'sha256': sha(path),
                     'arithmeticSeconds': samples, 'medianArithmeticSeconds': float(np.median(samples)),
                     'medianArithmeticCPUSeconds': float(np.median(cpu_samples)),
                     'residualRms': rms(residual), 'residualOverCenteredState': ratio(rms(residual), rms(centered)),
                     'perRowResidualRms': [rms(residual[:, i]) for i in range(state.shape[1])]})
    write(output / 'receiver-seal.json', {'schema': 'tce-state-receiver-seal-v1', 'specSHA256': sha(spec_path),
          'rows': rows, 'opened': sorted(opened), 'wallSecondsAfterImports': time.perf_counter() - started,
          'cpuSecondsAfterImports': time.process_time() - cpu, 'peakRSSMiB': peak_mib(),
          'warmupsPerPacket': 1, 'repeatsPerPacket': spec['repeats'], 'doseInput': False,
          'originalDenseProjections': 0, 'reducedBankContractionsPerEvaluation': 3,
          'neuralFrameworkCalls': 0, 'originalCheckpointBytesRead': 0,
          'scope': spec['scope']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['compile', 'serve'])
    parser.add_argument('spec', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    (compile_receiver if args.action == 'compile' else serve)(args.spec, args.output)
