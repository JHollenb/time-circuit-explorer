"""Fresh stdlib-only compiler/serving children for the address-program assay."""
import argparse
import gzip
import hashlib
import importlib.abc
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_history_address import METHODS, fit, run_packet


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def write_rows(path, rows):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode())


class NoEvaluation(importlib.abc.MetaPathFinder):
    roots = {'torch', 'transformers', 'diffusers', 'jax', 'tensorflow', 'onnxruntime', 'numpy'}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in self.roots:
            raise ImportError('Numerical/model runtime excluded from integer program child')


def isolate(spec, output):
    if NoEvaluation.roots.intersection(sys.modules):
        raise RuntimeError('Child must start without numerical/model runtime')
    sys.meta_path.insert(0, NoEvaluation())
    allowed = {str(Path(p).resolve()) for p in spec['readAllow']}
    library = [str(Path(p).resolve())+os.sep for p in {sys.base_prefix, sys.prefix}]
    output = str(Path(output).resolve())+os.sep
    opens = set()

    def audit(event, args):
        if event in {'socket.connect', 'socket.bind', 'subprocess.Popen', 'os.system'}:
            raise PermissionError('No network or child process in program execution')
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = str(Path(os.fsdecode(args[0])).resolve())
        mode = args[1]
        writing = any(a in mode for a in 'wax+') if isinstance(mode, str) else bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        in_output = path.startswith(output)
        permitted = in_output or (not writing and (path in allowed or any(path.startswith(a) for a in library)))
        if not permitted:
            raise PermissionError('Outside declared clean program inputs/output: '+path)
        if not writing:
            opens.add(path)

    sys.addaudithook(audit)
    denials = []
    for path in spec['forbiddenPaths']:
        try:
            with open(path, 'rb'):
                pass
        except PermissionError:
            denials.append(path)
        else:
            raise RuntimeError('File exclusion canary unexpectedly admitted')
    imports = []
    for name in ('numpy', 'torch'):
        try:
            importlib.import_module(name)
        except ImportError:
            imports.append(name)
        else:
            raise RuntimeError('Numerical import canary unexpectedly admitted')
    return opens, denials, imports


def execute(action, spec_path, output):
    started = time.perf_counter()
    cpu = time.process_time()
    spec = read(spec_path)
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    opens, denied, imports = isolate(spec, output)
    if action == 'compile':
        packets = [read(a['tokens']) for a in spec['cases']]
        labels = [read(a['supports']) for a in spec['cases']]
        ready = time.perf_counter()
        program, receipt = fit(packets, labels)
        computed = time.perf_counter()
        save(output/'program.json', program)
        save(output/'discovery.json', receipt)
        result = {'programSHA256': sha(output/'program.json'), 'discoverySHA256': sha(output/'discovery.json'),
            'computeSeconds': computed-ready, 'inputSeconds': ready-started, 'cases': len(packets)}
    else:
        if sha(spec['program']) != spec['programSHA256']:
            raise ValueError('Frozen program changed')
        program = read(spec['program'])
        result = {'programSHA256': spec['programSHA256'], 'rows': []}
        for item in spec['cases']:
            packet = read(item['tokens'])
            for method in METHODS:
                tick = time.perf_counter()
                rows, work = run_packet(packet, program, method)
                computed = time.perf_counter()
                path = output/(item['id']+'-'+method+'.jsonl.gz')
                write_rows(path, rows)
                result['rows'].append({'id': item['id'], 'method': method, 'path': path.name,
                    'sha256': sha(path), 'bytes': path.stat().st_size, 'queries': len(rows)*32,
                    'computeSeconds': computed-tick, 'serializationAndHashSeconds': time.perf_counter()-computed, **work})
    result.update(schema='tce-token-history-child-v1', action=action, fileCanaryDenials=denied,
        blockedImportCanaries=imports, successfulReadPaths=sorted(opens),
        loadedNumericalRuntimeRoots=sorted(NoEvaluation.roots.intersection(sys.modules)),
        CPUSeconds=time.process_time()-cpu, wallSecondsBeforeReceipt=time.perf_counter()-started,
        highWaterRSSKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Integer/equality/position/copy program only. Library/Python/hash/JSON/gzip/guard overhead paid; historical preparation excluded, not free.')
    save(output/'receipt.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['compile', 'serve'])
    parser.add_argument('spec')
    parser.add_argument('output')
    args = parser.parse_args()
    execute(args.action, args.spec, args.output)
