"""Source-bound prompt collisions and bounds from published scalar evidence only.

This is not an image generator or a neural-state quotient learner. No scientific
tensors, images, tokenizer tables, model weights or neural runtimes are loaded.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time


PINS = {
    'docs/results/fresh-prompt-square/request.json': 'c8544002b36980941b50f3c9899db93fb16a82c03da69ee7d3463aadf45dc288',
    'docs/results/fresh-prompt-square/report.json': 'db131593e32969d38c3345834493fe914ad888bbd913730daf793f10127d22f9',
    'generator/fresh_square_study.py': '5c0084346395cdb70066372428ed400ecb38cc06b3d9f38d3195ba8535b33c14',
    'generator/fresh_square_encoder.py': '5a1104c76dcb597edd9a31540b86be3a0657c2b17ebd70d1fe33c2053f583265',
    'generator/fresh_square_images.py': '4cdc4ce242cb670efdef2cab485914363dc7ad55e88f3abc0515fbdc6048179d',
    'generator/encoder_atoms.py': '6889cd42e46b9881a6568b8378edef0488f651d6867bdb1dcf7b0a79266c6922',
    'generator/response_chart.py': '1d0febad9df24642badb474a521ee819ffb667b4cc22269a8603d36fee22589d',
    'docs/raw-prompt-encoder-atoms.md': '3f6faba5ff0f36f7179fe428588984cdd6cf67bb8aad5cad2dfa33b7f7f8dd4d',
    'docs/zero-network-image-response.md': '2f445aec723b6a123cf5bed74c089cd8dbffc58704cc008db883c8708c9782f4',
    'docs/fresh-prompt-square.md': '3d862bdf0e1e0c011065183b6982225e6ebeefc1ffbf167dbe7f7ba44ee86bbf',
    'docs/plans/complete-serving-construction.md': 'd44bb8e51f2ac013bff0f91888014b34dacad92b2a22ab11582e28dbbb7781b1',
    'generator/render_response.py': '2e6b0af97bfd422a002a47d6eedd04c67c60aba9e2f79e3dcc036bd729d2025e',
    'docs/results/fresh-prompt-square/noise-seal.json': '3971848ac18fbef7d6ff616284fedd20e07fffd4c313be2297087bf7740336d8',
    'docs/results/fresh-prompt-square/full-seal.json': '4c1bc60b3981a5ce8a6466a63079a04415e0b2a1ac57d4278ff17d68d1c4b29b',
    'docs/results/fresh-prompt-square/images-full.json': '602146e108d7976615c82a4695047c2c338a2f9a189f20a4b398caba189f639f',
    'docs/results/fresh-prompt-square/retained-array-verification.json': 'bd8c8872b06301012f510e4117fdb68826b85b758d744b8de26803b127a827d0',
    'docs/results/fresh-prompt-square/provenance.json': 'e0077bc7eb4372928080fbc3e86cba3c015734b4371c219f3102e109ac447184',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def word_bag(prompt):
    """Explicit raw-whitespace-word representation; not BPE or learned atoms."""
    return tuple(sorted(Counter(prompt.split()).items()))


def collision_classes(prompts, encoder):
    groups = defaultdict(list)
    for item in prompts:
        groups[encoder(item['prompt'])].append(item['id'])
    return sorted(sorted(group) for group in groups.values() if len(group) > 1)


def pair_bounds(distance):
    if not math.isfinite(distance) or distance < 0:
        raise ValueError('Finite nonnegative RMS distance required')
    return {'minimumWorstEndpointRMS': distance / 2,
            'minimumMeanEndpointSquaredRMS': distance * distance / 4,
            'minimumPairPooledRMS': distance / 2}


def comparison_status(same_code, common_controls, distance):
    if common_controls is None or same_code is None or distance is None:
        return 'missing-evidence'
    if not common_controls:
        return 'incomparable-controls'
    pair_bounds(distance)
    return 'counterexample' if same_code and distance > 0 else 'no-counterexample-in-inspected-records'


def closure_witnesses(rows):
    """Observed successor conflicts, never a global completeness certificate."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row['clock'], row['control'], tuple(row['code']))].append(row)
    witnesses = []
    for group in groups.values():
        for index, left in enumerate(group):
            for right in group[index + 1:]:
                if left['nextCode'] != right['nextCode']:
                    witnesses.append({'clock': left['clock'], 'control': left['control'],
                                      'states': [left['state'], right['state']],
                                      'code': left['code'],
                                      'successorCodes': [left['nextCode'], right['nextCode']]})
    return witnesses


def known_wiring():
    """Complete four-state/two-control toy; no learned or image claim."""
    rows = {'visibleOnly': [], 'visibleAndHidden': []}
    for visible in (0, 1):
        for hidden in (0, 1):
            for control in ('hold', 'reveal'):
                following = visible if control == 'hold' else hidden
                for name, code, next_code in [
                    ('visibleOnly', [visible], [following]),
                    ('visibleAndHidden', [visible, hidden], [following, hidden]),
                ]:
                    rows[name].append(dict(clock=0, control=control, state=[visible, hidden],
                                           code=code, nextCode=next_code))
    return {'stateDomain': [[v, h] for v in (0, 1) for h in (0, 1)],
            'controlDomain': ['hold', 'reveal'], 'transition': '(v,h)->(v,h) for hold; (h,h) for reveal',
            'readout': 'visible register v', 'transitionsPerRepresentation': 8,
            'visibleOnlyWitnesses': closure_witnesses(rows['visibleOnly']),
            'refinedObservedWitnesses': closure_witnesses(rows['visibleAndHidden']),
            'scope': 'Exhaustive only for this supplied four-state toy. No FLUX/Qwen quotient inferred.'}


def execute(destination):
    started = time.perf_counter()
    root = Path(__file__).resolve().parent.parent
    for path, expected in PINS.items():
        if sha(root / path) != expected:
            raise ValueError('Source evidence changed: ' + path)
    request = json.loads((root / 'docs/results/fresh-prompt-square/request.json').read_text())
    report = json.loads((root / 'docs/results/fresh-prompt-square/report.json').read_text())
    noise = json.loads((root / 'docs/results/fresh-prompt-square/noise-seal.json').read_text())
    full = json.loads((root / 'docs/results/fresh-prompt-square/full-seal.json').read_text())
    images = json.loads((root / 'docs/results/fresh-prompt-square/images-full.json').read_text())
    provenance = json.loads((root / 'docs/results/fresh-prompt-square/provenance.json').read_text())
    if report['requestSHA256'] != request['requestSHA256']:
        raise ValueError('Public report/request association differs')
    request_id = request['requestSHA256']
    if any(item['requestSHA256'] != request_id for item in [noise, full, images, provenance]):
        raise ValueError('Reference/control metadata association differs')
    if (report['fullSealSHA256'] != sha(root / 'docs/results/fresh-prompt-square/full-seal.json')
            or full['imagesSHA256'] != sha(root / 'docs/results/fresh-prompt-square/images-full.json')
            or full['noiseSealSHA256'] != images['noiseSealSHA256']
            or images['noiseSealSHA256'] != sha(root / 'docs/results/fresh-prompt-square/noise-seal.json')
            or noise['newNoiseDraws'] != 1):
        raise ValueError('Reference/shared-noise seal association differs')
    schedules = list(images['schedules'].values())
    if set(images['schedules']) != {'00', '10', '01', '11'} or any(s != schedules[0] for s in schedules):
        raise ValueError('Historical common schedule differs')
    first_inputs = [r for r in images['actual'] if r['kind'] == 'denoiser-input' and r['step'] == 0]
    if (len(first_inputs) != 4 or {r['case'] for r in first_inputs} != {'00', '10', '01', '11'}
            or any(r['latentPayloadSHA256'] != noise['packed']['payloadSHA256'] for r in first_inputs)):
        raise ValueError('Historical initial packed-noise join differs')
    if request['seed'] != 271828 or (request['height'], request['width'], request['steps']) != (512, 512, 4):
        raise ValueError('Historical specimen differs')
    prompts = request['prompts']
    bag_classes = collision_classes(prompts, word_bag)
    ordered_classes = collision_classes(prompts, lambda p: tuple(p.split()))
    if bag_classes != [['00', '10'], ['01', '11']] or ordered_classes:
        raise ValueError('Nominated raw-word representation association differs')
    by_id = {item['id']: item for item in prompts}
    witnesses = []
    for cases, kind in [(['00', '10'], 'color-left'), (['01', '11'], 'color-right')]:
        matches = [row for row in report['factorialResponses']
                   if (row['kind'], row['port'], row['region']) == (kind, 'pngRGB', 'all')]
        if len(matches) != 1 or matches[0]['scalars'] != 512 * 512 * 3:
            raise ValueError('Unique complete published PNG response required')
        row = matches[0]
        distance = row['referenceRms']
        witnesses.append({'cases': cases, 'publishedContrast': kind,
                          'sourceDenominator': 'referenceRms; full-reference PNG response',
                          'nativeImageDistanceRMS': distance, 'pixelScale': '0..255',
                          'commonSeed': request['seed'], 'sameWordBag': True,
                          'status': comparison_status(True, True, distance),
                          'orderedWordsDiffer': tuple(by_id[cases[0]]['prompt'].split()) != tuple(by_id[cases[1]]['prompt'].split()),
                          **pair_bounds(distance)})
    result = {'schema': 'tce-serving-collision-witness-v1', 'cycle': 91,
              'analyzerSourceSHA256': sha(__file__), 'sourceSHA256': PINS,
              'historicalRequestSHA256': request['requestSHA256'],
              'historicalCandidateSealSHA256': report['candidateSealSHA256'],
              'historicalFullSealSHA256': report['fullSealSHA256'],
              'historicalJobID': provenance['jobID'],
              'commonControls': {'newHistoricalNoiseDraws': noise['newNoiseDraws'],
                                 'rawNoisePayloadSHA256': noise['raw']['payloadSHA256'],
                                 'packedNoisePayloadSHA256': noise['packed']['payloadSHA256'],
                                 'schedule': schedules[0], 'numericalProfile': images['numericalProfile'],
                                 'scope': 'Current local metadata joins/source clone rule; historical raw bytes are not freshly rehashed or numerically replayed.'},
              'representation': 'Sorted Counter(prompt.split()) over raw whitespace-delimited words, case and punctuation retained',
              'promptWordCounts': {item['id']: len(item['prompt'].split()) for item in prompts},
              'bagCollisionClasses': bag_classes, 'orderedWordCollisionClasses': ordered_classes,
              'witnesses': witnesses, 'knownWiring': known_wiring(),
              'dependencyTable': [
                  {'function': 'c lexical prefix', 'status': 'implemented-bounded-prefix',
                   'source': 'generator/encoder_atoms.py:Prefix.tokenize/forward',
                   'provided': 'Ordered IDs/mask/positions and compiled token-local E/Q/K/V/RoPE',
                   'missing': 'First contextual attention and remaining contextual conditioning'},
                  {'function': 'c complete conditioning', 'status': 'unresolved-without-neural-evaluation',
                   'source': 'generator/fresh_square_encoder.py:execute',
                   'provided': 'Own-state native-weight encoder continuation',
                   'missing': 'Executable non-neural contextual producer from ordered lexical operands'},
                  {'function': 'I latent initializer', 'status': 'specified-conditional-on-conditioning',
                   'source': 'generator/fresh_square_study.py:noise;generator/fresh_square_images.py:execute',
                   'provided': 'Declared RNG draw/packing plus conditioning and finite schedule state',
                   'missing': 'General conditioning is still an unresolved input; learned image projections belong to T'},
                  {'function': 'T own-state transitions', 'status': 'unresolved-without-neural-evaluation',
                   'source': 'generator/fresh_square_images.py:execute',
                   'provided': 'Original learned denoiser and actual scheduler commits',
                   'missing': 'Executable general transition independent of original learned maps'},
                  {'function': 'R general pixels', 'status': 'unresolved-without-neural-evaluation',
                   'source': 'generator/fresh_square_images.py:execute;generator/response_chart.py:evaluate',
                   'provided': 'Native VAE or RGB interpolation on four supplied context curves',
                   'missing': 'Pixel producer for candidate-generated general states without a supplied scene renderer'},
              ],
              'newImages': 0, 'newModelEvaluations': 0, 'newLearnedMatrixMACs': 0,
              'scientificTensorOrImageReads': 0, 'numericalScope': 'Published scalar JSON, raw text and tiny supplied finite-state mechanics only',
              'mechanics_status': 'complete', 'instrument_status': 'applicable to two historical prompt pairs',
              'trend_status': 'Both raw-word-bag collisions have positive native-image distance and a necessary endpoint-error floor',
              'terminal_status': 'unassessed for arbitrary-prompt quality or generation',
              'scope': 'Retrospective exact-native factoring exclusion for this nominated representation and common controls. Not an exclusion of ordered atoms, position-aware representations, all compact states, or prompt-faithful generation without native pixel parity.',
              'analysisSecondsBeforeReceiptWrite': time.perf_counter() - started}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination')
    result = execute(parser.parse_args().destination)
    print(json.dumps({k: result[k] for k in ['bagCollisionClasses', 'witnesses', 'analysisSecondsBeforeReceiptWrite']}, indent=2))
