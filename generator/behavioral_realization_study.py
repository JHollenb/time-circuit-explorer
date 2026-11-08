"""One tiny exact compiler mechanics study; no native model or scientific arrays."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from behavioral_realization import ARITHMETIC, FiniteMachine, WordOracle, certify_finite, compile_panel, evaluate, identity, lower_copy_program, transformed, words


def toy_source():
    return {'schema': 'tce-supplied-two-bit-machine-v1', 'alphabet': ['a', 'b'], 'initial': '00',
            'states': {str(v) + str(h): {'output': [v], 'next': {'a': str(h) + '0', 'b': str(h) + '1'}}
                       for v in (0, 1) for h in (0, 1)},
            'interface': 'Input is an ordered symbol list; empty word is identity; unknown symbols refused. Readout once at the end; no EOS symbol.',
            'scope': 'Completely supplied finite toy. No natural-language parser, model-derived semantics, image renderer or pixel claim.'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def execute(destination):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    linear_macs_before = ARITHMETIC['exactLinearMACs']
    machine = FiniteMachine(toy_source())
    initial_source_id = identity(machine.data)
    save(root / 'source.json', machine.data)
    prefixes = words(machine.alphabet, 2)
    tests = [(suffix, 0) for suffix in words(machine.alphabet, 2)]
    oracle = WordOracle(machine.output)
    realization = compile_panel(oracle, machine.alphabet, prefixes, tests)
    discovery = oracle.accounting()
    certificate = certify_finite(machine, realization)
    lowered = lower_copy_program(realization, certificate)
    if lowered['status'] != 'lowered':
        raise RuntimeError('Tiny compiler mechanics did not emit a program')
    shallow_oracle = WordOracle(machine.output)
    shallow = compile_panel(shallow_oracle, machine.alphabet, prefixes, [((), 0)])
    shallow_certificate = certify_finite(machine, shallow)
    for name, recorded in [('discovery-queries.json', oracle), ('shallow-queries.json', shallow_oracle)]:
        save(root / name, {'sourceID': initial_source_id, 'accounting': recorded.accounting(),
                           'rows': [{'tokens': list(word), 'output': list(map(str, value))}
                                    for word, value in recorded.cache.items()],
                           'scope': 'Complete immutable physical word-query cache; repeated requests counted separately.'})
    shallow_words = words(machine.alphabet, 5)
    failures = [word for word in shallow_words if tuple(evaluate(shallow, word)) != machine.output(word)]
    gauge = transformed(realization, [[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    gauge_certificate = certify_finite(machine, gauge)
    gauge_lowering = lower_copy_program(gauge, gauge_certificate)
    for name, value in [('realization.json', realization), ('certificate.json', certificate),
                        ('program.json', lowered), ('shallow-realization.json', shallow),
                        ('shallow-certificate.json', shallow_certificate), ('gauge-realization.json', gauge),
                        ('gauge-certificate.json', gauge_certificate), ('gauge-lowering.json', gauge_lowering)]:
        save(root / name, value)
    cases = [{'id': 'word-' + (''.join(word) or 'empty'), 'tokens': list(word)} for word in words(machine.alphabet, 4)]
    cases += [{'id': 'long-' + str(i), 'tokens': list(word)} for i, word in enumerate(['a' * 32, 'b' * 32, 'ab' * 16, 'ba' * 16])]
    cases.append({'id': 'invalid', 'tokens': ['a', '?']})
    server = Path(__file__).resolve().with_name('behavioral_program.py')
    forbidden = [p for p in sorted(root.iterdir()) if p.is_file() and p.name != 'program.json']
    forbidden += [Path(__file__).resolve().with_name('behavioral_realization.py'), Path(__file__).resolve()]
    request = {'schema': 'tce-copy-program-request-v1', 'programSHA256': sha(root / 'program.json'),
               'cases': cases, 'forbiddenReads': list(map(str, forbidden))}
    save(root / 'request.json', request)
    child = subprocess.run([sys.executable, '-I', str(server), str(root / 'program.json'),
                            str(root / 'request.json'), str(root / 'serving.json')], capture_output=True, text=True, timeout=30)
    save(root / 'child-exit.json', {'returncode': child.returncode, 'stdout': child.stdout, 'stderr': child.stderr})
    if child.returncode:
        raise RuntimeError('Fresh copy-program child did not complete; retain its exit record')
    serving_sha = sha(root / 'serving.json')
    save(root / 'prediction-seal.json', {'programSHA256': sha(root / 'program.json'),
                                        'servingSHA256': serving_sha, 'requestSHA256': sha(root / 'request.json')})
    serving = json.loads((root / 'serving.json').read_text())
    if (len(serving['rows']) != len(cases) or [row['id'] for row in serving['rows']] != [item['id'] for item in cases]
            or serving['requestSHA256'] != sha(root / 'request.json')
            or serving['programSHA256'] != sha(root / 'program.json') or serving['serverSourceSHA256'] != sha(server)):
        raise RuntimeError('Complete fresh serving handoff differs')
    scored = []
    for item, row in zip(cases, serving['rows']):
        valid = set(item['tokens']).issubset(machine.alphabet)
        expected = list(machine.output(item['tokens'])) if valid else None
        expected_status = 'executed' if valid else 'outside-declared-alphabet'
        scored.append({'id': item['id'], 'expected': expected, 'actual': row['output'],
                       'equal': row['output'] == expected and row['status'] == expected_status,
                       'status': 'in-domain' if valid else 'explicit-invalid-symbol-control'})
    if identity(machine.data) != initial_source_id or sha(root / 'serving.json') != serving_sha:
        raise RuntimeError('Source or sealed serving result changed during scoring')
    result = {'schema': 'tce-exact-behavioral-compiler-study-v1', 'cycle': 92,
              'sourceID': initial_source_id, 'discovery': discovery,
              'panelRank': realization['rank'],
              'globalRankProof': {'sampledMinorLowerBound': realization['rank'], 'completeRealizationUpperBound': realization['rank'],
                                  'authority': 'Exact nonsingular minor plus complete finite-graph commuting/output/initial certificate'},
              'completeCertificateStatus': certificate['closureStatus'],
              'sourceStates': len(certificate['reachableStates']), 'stateSymbolEqualities': len(certificate['transitions']),
              'outputEqualities': len(certificate['outputs']), 'initialEquality': certificate['initialEqual'],
              'programPersistentRegisters': lowered['persistentRegisters'],
              'shallowRank': shallow['rank'], 'shallowStatus': shallow_certificate['closureStatus'],
              'shallowFirstWholeWordFailure': list(failures[0]) if failures else None,
              'shallowOracleAccounting': shallow_oracle.accounting(),
              'gaugeClosure': gauge_certificate['closureStatus'], 'gaugeLoweringStatus': gauge_lowering['status'],
              'orderContrast': {'ab': list(machine.output(('a', 'b'))), 'ba': list(machine.output(('b', 'a')))},
              'sourceExecutionAccounting': {'outputCallsWholeStudyBeforeReport': machine.output_calls,
                                            'symbolTransitionsWholeStudyBeforeReport': machine.symbol_steps,
                                            'exactRationalLinearMACsBeforeReport': ARITHMETIC['exactLinearMACs'] - linear_macs_before,
                                            'scope': 'Main-process supplied-source queries and row-times-matrix loops only. Static graph traversal, exact elimination/pivots, JSON, guards and child dispatch additional.'},
              'servingRows': scored, 'servingSHA256': serving_sha, 'predictionSealSHA256': sha(root / 'prediction-seal.json'),
              'guardCanaries': len(serving['canaries']), 'coefficientMatrixEvaluationsInCopyChild': serving['coefficientMatrixEvaluations'],
              'importGuardCanaries': len(serving['importCanaries']),
              'sourceOracleCallsInCopyChild': serving['sourceOracleCalls'], 'neuralRuntimeImportsInCopyChild': serving['neuralRuntimeImports'],
              'newNeuralModelEvaluations': 0, 'newImages': 0, 'scientificArrayReads': 0,
              'nominalDenseCoefficientMACsPerSymbol': realization['rank'] ** 2,
              'nominalDenseCoefficientMACsPerReadout': realization['rank'] * realization['outputWidth'],
              'sourceHashes': {p.name: sha(p) for p in [Path(__file__).resolve(), server,
                               Path(__file__).resolve().with_name('behavioral_realization.py')]},
              'elapsedSecondsBeforeReportWrite': time.perf_counter() - started,
              'mechanics_status': 'complete', 'instrument_status': 'supplied finite-machine exact compiler mechanics',
              'trend_status': 'Future tests recover ordered memory; exact copy lowering removes coefficient arithmetic for this supplied toy',
              'terminal_status': 'bounded finite-machine closure only; arbitrary-prompt image goal unestablished',
              'scope': 'All source behavior, alphabet and scalar readout supplied. Complete proof applies only to this finite graph. No learned image state, raw-prompt interpreter, general pixel producer or neural-model compression established.'}
    save(root / 'report.json', result)
    save(root / 'files.json', {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                               for p in sorted(root.iterdir()) if p.is_file()})
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('destination')
    result = execute(parser.parse_args().destination)
    print(json.dumps({k: result[k] for k in ['panelRank', 'completeCertificateStatus', 'programPersistentRegisters',
                                            'shallowFirstWholeWordFailure', 'gaugeLoweringStatus', 'elapsedSecondsBeforeReportWrite']}, indent=2))
