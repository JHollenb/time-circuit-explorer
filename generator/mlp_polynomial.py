"""Compile a supplied SwiGLU input line; serve only output coefficients."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import resource
import statistics
import sys
import time

from response_chart import np, sha, write
from prompt_closure import stored, rms, ratio


def read(path):
    return json.loads(Path(path).read_text())


def quantize(value):
    return stored(value, 'tokens')


def peak_mib():
    divisor = 1024 ** 2 if sys.platform == 'darwin' else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def checked_ref(ref):
    path = Path(ref['path'])
    if sha(path) != ref['sha256']:
        raise ValueError('Source file changed: ' + str(path))
    if ref['kind'] == 'bundle':
        with path.open('rb') as stream:
            stream.seek(ref['offset'])
            raw = stream.read(ref['nbytes'])
        if len(raw) != ref['nbytes'] or hashlib.sha256(raw).hexdigest() != ref['segmentSHA256']:
            raise ValueError('Native array segment changed')
        value = np.load(io.BytesIO(raw), allow_pickle=False)
    elif ref['kind'] == 'safetensors':
        with path.open('rb') as stream:
            length = int.from_bytes(stream.read(8), 'little')
            if not 2 <= length <= 16 << 20:
                raise ValueError('Invalid tensor header')
            header = json.loads(stream.read(length))[ref['tensor']]
            start, end = header['data_offsets']
            stream.seek(8 + length + start)
            raw = stream.read(end - start)
        if header['shape'] != ref['shape'] or header['dtype'] != 'BF16':
            raise ValueError('Native parent tensor geometry differs')
        if hashlib.sha256(raw).hexdigest() != ref['tensorSHA256']:
            raise ValueError('Parent tensor payload differs')
        value = (np.frombuffer(raw, '<u2').astype(np.uint32) << 16).view(np.float32).reshape(ref['shape'])
    else:
        value = np.load(path, allow_pickle=False)
    if list(value.shape) != ref['shape'] or not np.isfinite(value).all():
        raise ValueError('Source shape/finite contract differs')
    return value.astype(np.float64)


def audit_reads(allowed, roots, *, serving=False):
    allowed = {Path(p).resolve() for p in allowed}
    roots = [Path(p).resolve() for p in roots]
    opened = set()

    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = Path(args[0]).resolve()
            opened.add(str(path))
            if path.name in {'evaluation.json', 'compile.json'} or any(path.is_relative_to(r) for r in roots):
                if path not in allowed:
                    raise PermissionError('Read outside declared operands: ' + str(path))
            if serving and path.suffix in {'.safetensors', '.pt', '.pth', '.bin', '.npyb'}:
                raise PermissionError('Original parameters/native archives prohibited online')
        if event in {'socket.connect', 'socket.bind'}:
            raise PermissionError('Network prohibited')

    sys.addaudithook(audit)
    return opened


def hidden_coefficients(a0, a1, v0, v1):
    """Retain the gate/value product, rather than truncating its total degree."""
    q = np.empty_like(a0, dtype=np.float64)
    positive = a0 >= 0
    q[positive] = 1 / (1 + np.exp(-a0[positive]))
    e = np.exp(a0[~positive])
    q[~positive] = e / (1 + e)
    sig = [q, q * (1 - q) * a1, .5 * q * (1 - q) * (1 - 2 * q) * a1 * a1]
    silu = [a0 * sig[0], a0 * sig[1] + a1 * sig[0],
            a0 * sig[2] + a1 * sig[1], a1 * sig[2]]
    return np.stack([silu[0] * v0, silu[1] * v0 + silu[0] * v1,
                     silu[2] * v0 + silu[1] * v1, silu[3] * v0 + silu[2] * v1,
                     silu[3] * v1])


def horner(coefficients, alpha):
    value = coefficients[-1].copy()
    for coefficient in coefficients[-2::-1]:
        value = value * alpha + coefficient
    return value


def consume(parent, gate, value):
    rounded = quantize(value)
    return rounded, quantize(parent + quantize(gate * rounded))


def compile_line(spec_path, output):
    started, cpu = time.perf_counter(), time.process_time()
    spec = read(spec_path)
    allowed = [r['path'] for r in spec['operands'].values()] + [spec['checkpointFile'], spec_path]
    opened = audit_reads(allowed, spec['sourceRoots'])
    operands = {name: checked_ref(ref) for name, ref in spec['operands'].items()}
    indices = spec['selectedRows']
    parent = quantize(operands['image'][:, indices] + quantize(operands['gateAttention'] * operands['attention'][:, indices]))
    gate = operands['gateMLP'].astype(np.float32)
    x0, x1 = operands['base'], operands['changed']
    if x0.shape != x1.shape or x0.shape != parent.shape or gate.shape != (1, 1, x0.shape[-1]):
        raise ValueError('Input-line/consumer geometry differs')
    input_rounding = []
    for alpha in [0, .5, 1]:
        line = x0 + alpha * (x1 - x0)
        displacement = quantize(line).astype(np.float64) - line
        input_rounding.append({'alpha': alpha, 'rms': rms(displacement),
                               'maxAbs': float(np.max(np.abs(displacement))),
                               'changedScalars': int(np.count_nonzero(displacement)),
                               'perRowRms': [rms(displacement[:, i]) for i in range(line.shape[1])]})
    # The pinned stdlib retriever hashes the complete checkpoint and validates each
    # safetensors name, dtype, shape, byte range, payload and reconstructed NPY.
    import retrieve_weights as retrieval
    manifest = retrieval.validate_manifest(spec['parameters'])
    projection = None
    coefficients = None
    counts = {'originalParameterBytes': 0, 'originalParameters': 0, 'denseMatmuls': 0,
              'fullCheckpointBytesHashed': Path(spec['checkpointFile']).stat().st_size}
    for entry, stream, size in retrieval._source_entries(manifest, spec['checkpointRoot']):
        raw = b''.join(retrieval._artifact_chunks(entry, stream, size))
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('Reconstructed parameter differs')
        word = np.load(io.BytesIO(raw), allow_pickle=False)
        weight = (word.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        del raw, word
        counts['originalParameterBytes'] += size
        counts['originalParameters'] += weight.size
        if entry['source']['tensor'].endswith('linear_in.weight'):
            combined = np.concatenate([x0, x1 - x0], axis=0)
            projection = (combined.reshape(-1, combined.shape[-1]) @ weight.T).reshape(2, x0.shape[1], -1)
            counts['denseMatmuls'] += 1
        elif entry['source']['tensor'].endswith('linear_out.weight') and projection is not None:
            a0, v0 = np.split(projection[:1], 2, axis=-1)
            a1, v1 = np.split(projection[1:], 2, axis=-1)
            hidden = hidden_coefficients(a0, a1, v0, v1)
            coefficients = (hidden.reshape(-1, hidden.shape[-1]) @ weight.T).reshape(5, *x0.shape)
            counts['denseMatmuls'] += 1
            del hidden
        else:
            raise ValueError('Unexpected parameter order/name')
        del weight
    if coefficients is None or coefficients.shape != (5, *x0.shape) or not np.isfinite(coefficients).all():
        raise ValueError('Coefficient contract differs')
    output = Path(output)
    np.save(output / 'coefficients.npy', coefficients, allow_pickle=False)
    np.savez(output / 'consumer.npz', parent=parent, gate=gate)
    write(output / 'compile-receipt.json', {
        'schema': 'tce-compiled-mlp-line-v1', 'compileSpecSHA256': sha(spec_path),
        'coefficientsSHA256': sha(output / 'coefficients.npy'), 'consumerSHA256': sha(output / 'consumer.npz'),
        'coefficientShape': list(coefficients.shape), 'coefficientDtype': str(coefficients.dtype),
        'coefficientPayloadBytes': coefficients.nbytes, 'consumerPayloadBytes': parent.nbytes + gate.nbytes,
        'wallSeconds': time.perf_counter() - started, 'cpuSeconds': time.process_time() - cpu,
        'peakRSSMiB': peak_mib(), 'opened': sorted(opened), 'offlineWork': counts,
        'inputLineToBF16Displacement': input_rounding,
        'numericScope': 'Smooth input-line polynomial vs staged CPU BF16 reference; input, projection, SiLU, product and output rounding differences remain in endpoint error. Independently computed c0 is not a native-output anchor.',
        'scope': spec['scope'], 'neuralFrameworkCalls': 0,
        'offlineDenseLearnedWeightMath': True, 'fitToNativeOutputs': False})


def serve(spec_path, output):
    started, cpu = time.perf_counter(), time.process_time()
    spec = read(spec_path)
    primary = spec['mode'] == 'primary'
    if spec['mode'] not in {'primary', 'controls'}:
        raise ValueError('Unknown serving mode')
    bundle_keys = ['coefficients', 'consumer'] if primary else ['controls', 'consumer']
    allowed = [spec[k]['path'] for k in bundle_keys]
    opened = audit_reads(allowed, spec['sourceRoots'], serving=True)
    for key in bundle_keys:
        if sha(spec[key]['path']) != spec[key]['sha256']:
            raise ValueError('Serving bundle changed')
    with np.load(spec['consumer']['path'], allow_pickle=False) as ctx:
        parent, gate = ctx['parent'], ctx['gate']
    if primary:
        coefficients = np.load(spec['coefficients']['path'], allow_pickle=False)
        if coefficients.shape != (5, *parent.shape):
            raise ValueError('Serving coefficient shape differs')
        arrays = [coefficients, parent, gate]
    else:
        with np.load(spec['controls']['path'], allow_pickle=False) as ctx:
            base, changed = ctx['base'], ctx['changed']
        if base.shape != parent.shape or changed.shape != base.shape:
            raise ValueError('Serving control shape differs')
        arrays = [parent, gate, base, changed]
    if not all(np.isfinite(v).all() for v in arrays):
        raise ValueError('Serving bundle not finite')
    output = Path(output)
    rows = []
    for alpha in spec['alphas']:
        if not 0 <= alpha <= 1:
            raise ValueError('Dose outside supplied line')
        for method in (['polynomial'] if primary else ['endpoint_linear', 'copy_base']):
            def calculate():
                raw = horner(coefficients, alpha) if method == 'polynomial' else base + alpha * (changed - base) if method == 'endpoint_linear' else base.copy()
                rounded, state = consume(parent, gate, raw)
                return raw, rounded, state
            calculate()  # declared warmup, charged to process/end-to-end time
            samples, cpu_samples = [], []
            for _ in range(spec['repeats']):
                wall0, cpu0 = time.perf_counter(), time.process_time()
                raw, rounded, state = calculate()
                samples.append(time.perf_counter() - wall0)
                cpu_samples.append(time.process_time() - cpu0)
            if not all(np.isfinite(v).all() for v in [raw, rounded, state]):
                raise ValueError('Predicted write/state is not finite')
            path = output / f'{method}-{alpha:g}.npz'
            np.savez(path, raw=raw, rounded=rounded, state=state)
            rows.append({'method': method, 'alpha': alpha, 'path': path.name, 'sha256': sha(path),
                         'medianArithmeticSeconds': statistics.median(samples), 'arithmeticSeconds': samples,
                         'medianArithmeticCPUSeconds': statistics.median(cpu_samples),
                         'roundingRms': rms(rounded.astype(np.float64) - raw)})
    seal_name = 'prediction-seal.json' if primary else 'control-seal.json'
    write(output / seal_name, {'schema': 'tce-compiled-mlp-seal-v1', 'servingSpecSHA256': sha(spec_path),
          'rows': rows, 'wallSecondsAfterImports': time.perf_counter() - started,
          'cpuSecondsAfterImports': time.process_time() - cpu, 'peakRSSMiB': peak_mib(), 'opened': sorted(opened),
          'originalCheckpointBytesRead': 0, 'originalDenseMatmuls': 0, 'neuralEvaluations': 0,
          'scope': spec['scope'], 'mode': spec['mode'], 'nativeEndpointOutputsSupplied': not primary,
          'repeatsPerCase': spec['repeats'], 'warmupsPerCase': 1})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['compile', 'serve'])
    parser.add_argument('spec', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    (compile_line if args.action == 'compile' else serve)(args.spec, args.output)
