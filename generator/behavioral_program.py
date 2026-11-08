"""Fresh-process byte copy/constant interpreter. No oracle or matrix evaluator."""
import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import time


DENIED = {'torch', 'transformers', 'diffusers', 'tensorflow', 'jax', 'onnxruntime', 'numpy'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(program):
    if program['schema'] != 'tce-byte-copy-program-v1' or program['status'] != 'lowered':
        raise ValueError('Only a lowered byte copy program can execute')
    count = program['persistentRegisters']
    if type(count) is not int or count < 0 or len(program['initial']) != count:
        raise ValueError('Invalid register domain')
    if any(type(x) is not int or not 0 <= x <= 255 for x in program['initial']):
        raise ValueError('Byte registers required')
    if set(program['transitions']) != set(program['alphabet']):
        raise ValueError('Symbol dispatch must be total on the declared alphabet')
    for operations in [*program['transitions'].values(), program['readout']]:
        for op in operations:
            if op.get('op') == 'copy':
                if set(op) != {'op', 'register'} or type(op['register']) is not int or not 0 <= op['register'] < count:
                    raise ValueError('Invalid register-copy instruction')
            elif op.get('op') == 'constant':
                if set(op) != {'op', 'value'} or type(op['value']) is not int or not 0 <= op['value'] <= 255:
                    raise ValueError('Invalid byte-constant instruction')
            else:
                raise ValueError('Coefficient arithmetic is outside this instruction set')
    if any(len(ops) != count for ops in program['transitions'].values()):
        raise ValueError('Every transition must write the full next register state')


def operation(op, old):
    return old[op['register']] if op['op'] == 'copy' else op['value']


def run_word(program, tokens):
    if any(token not in program['alphabet'] for token in tokens):
        return {'status': 'outside-declared-alphabet', 'output': None, 'transitionSteps': 0}
    state = list(program['initial'])
    copies = constants = 0
    for token in tokens:
        ops = program['transitions'][token]
        # Materialize the next tuple from old-state reads before changing state.
        following = [operation(op, state) for op in ops]
        copies += sum(op['op'] == 'copy' for op in ops)
        constants += sum(op['op'] == 'constant' for op in ops)
        state = following
    return {'status': 'executed', 'output': [operation(op, state) for op in program['readout']],
            'finalRegisters': state, 'transitionSteps': len(tokens), 'registerCopies': copies,
            'constantWrites': constants, 'readoutCopies': sum(op['op'] == 'copy' for op in program['readout'])}


def serve(program_path, request_path, output_path):
    class NoNeural(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in DENIED:
                raise ImportError('Runtime outside the copy interpreter: ' + fullname)
    if DENIED.intersection(sys.modules):
        raise RuntimeError('Fresh dependency-light interpreter required')
    sys.meta_path.insert(0, NoNeural())
    request_sha = sha(request_path)
    request = json.loads(Path(request_path).read_text())
    forbidden = {str(Path(path).resolve()) for path in request['forbiddenReads']}
    reads = []
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(Path(args[0]).resolve())
            if path in forbidden:
                raise PermissionError('Source/oracle/matrix input prohibited')
            reads.append(path)
        if event in {'socket.connect', 'socket.bind'}:
            raise PermissionError('No network in copy interpreter')
    sys.addaudithook(audit)
    canaries = []
    for path in request['forbiddenReads']:
        try:
            with open(path, 'rb') as stream:
                stream.read(1)
        except PermissionError:
            canaries.append({'path': path, 'blocked': True})
        else:
            raise RuntimeError('Forbidden-read instrument did not run')
    import_canaries = []
    for name in ['torch', 'numpy']:
        try:
            __import__(name)
        except ImportError:
            import_canaries.append({'module': name, 'blocked': True})
        else:
            raise RuntimeError('Runtime-import exclusion instrument did not run')
    started = time.perf_counter()
    if sha(program_path) != request['programSHA256']:
        raise ValueError('Compiled program identity differs')
    program = json.loads(Path(program_path).read_text())
    validate(program)
    rows = [{'id': item['id'], **run_word(program, item['tokens'])} for item in request['cases']]
    if sha(request_path) != request_sha or sha(program_path) != request['programSHA256']:
        raise ValueError('Program/request changed during serving')
    result = {'schema': 'tce-copy-program-serving-v1', 'programSHA256': sha(program_path),
              'requestSHA256': request_sha, 'serverSourceSHA256': sha(__file__),
              'rows': rows, 'canaries': canaries, 'importCanaries': import_canaries, 'successfulFileOpens': sorted(set(reads)),
              'persistentRegisters': program['persistentRegisters'],
              'interpreterTemporaryPolicy': 'Next-state list and Python dispatch objects additional to persistent byte registers',
              'coefficientMatrixEvaluations': 0, 'sourceOracleCalls': 0,
              'neuralRuntimeImports': sorted(DENIED.intersection(sys.modules)),
              'servingSecondsBeforeReceiptWrite': time.perf_counter() - started,
              'scope': 'Trusted-code toy byte-machine serving; audit canaries are not an adversarial sandbox or hardware operation counter.'}
    with Path(output_path).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('program')
    parser.add_argument('request')
    parser.add_argument('output')
    args = parser.parse_args()
    result = serve(args.program, args.request, args.output)
    print(json.dumps({'rows': len(result['rows']), 'persistentRegisters': result['persistentRegisters'],
                      'servingSecondsBeforeReceiptWrite': result['servingSecondsBeforeReceiptWrite']}))
