"""Source-bound integer/rational feasibility; no model, array or learned products."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parent.parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def census(alphabet, length, vocabulary, heads, dimension):
    if any(type(v) is not int or v <= 0 for v in [alphabet, length, vocabulary, heads, dimension]) or length < 2:
        raise ValueError('Positive integer geometry required')
    triangle = length*(length-1)//2
    finite = alphabet*triangle
    fixed_query_full_vocabulary = vocabulary*triangle
    all_query_strict_past = vocabulary*fixed_query_full_vocabulary
    legal_diagonal = vocabulary*length
    return {'strictPastPositionPairs': triangle, 'currentDomainScores': finite,
        'currentDomainFP64Bytes': 8*finite, 'projectedCompileCoordinateProducts': finite*dimension,
        'idealRelativeOffsetScores': alphabet*(length-1), 'idealRelativeOffsetFP64Bytes': 8*alphabet*(length-1),
        'fixedQueryFullVocabularyScoresPerHead': fixed_query_full_vocabulary,
        'fixedQueryFullVocabularyFP64BytesPerHead': 8*fixed_query_full_vocabulary,
        'fixedQueryFullVocabularyFP64BytesAllHeads': 8*fixed_query_full_vocabulary*heads,
        'allQueryIDsStrictPastScoresPerHead': all_query_strict_past,
        'allQueryIDsStrictPastFP64BytesAllHeads': 8*all_query_strict_past*heads,
        'additionalLegalActiveDiagonalScoresPerHead': legal_diagonal,
        'allQueryIDsWithLegalDiagonalFP64BytesAllHeads': 8*(all_query_strict_past+legal_diagonal)*heads,
        'scope': 'Literal dense FP64 layouts,not information-theoretic lower bounds,measured compressed sizes or native finite-parity certification. Diagonal assumes legal same-token current query/key;strict-past census alone excludes it.'}


def score_index(alphabet, length, token_index, query_position, key_position):
    if any(type(v) is not int for v in [alphabet, length, token_index, query_position, key_position]):
        raise ValueError('Integer index geometry')
    if alphabet <= 0 or length < 2 or not (0 <= token_index < alphabet and 0 < query_position < length and 0 <= key_position < query_position):
        raise ValueError('Outside strict-past finite domain')
    return alphabet*query_position*(query_position-1)//2+token_index*query_position+key_position


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    rounded = ((bits+0x7fff+((bits >> 16) & 1)) >> 16) & 0xffff
    return Fraction(struct.unpack('<f', struct.pack('<I', rounded << 16))[0])


def rotary_witness():
    c = bf16(math.cos(math.pi/8)); s = bf16(math.sin(math.pi/8)); a = bf16(math.cos(math.pi/4))
    # Supplied q0=k0=(1,0):each source-style product and sum is already BF16-exact here.
    before = c; after = a*c+a*s
    return {'suppliedUnrotatedQueryAndKey': ['1', '0'], 'angleStep': 'pi/8', 'unscaled': True,
        'bf16CosStep': str(c), 'bf16SinStep': str(s), 'bf16CosAndSinTwoSteps': str(a),
        'firstQueryKeyPositions': [1, 0], 'translatedQueryKeyPositions': [2, 1], 'relativeOffsetBoth': 1,
        'firstScore': str(before), 'translatedScore': str(after), 'signedDifference': str(after-before),
        'scope': 'Exact rational supplied2-D rounded-rotary toy,not native Qwen measurements. Refutes universal exact relative-offset reuse for this finite policy;does not quantify source-model error or exclude other compact evaluators.'}


def weight_witness():
    a = [8, 4, 2, 1]; b = [64, 16, 4, 1]
    return {'positiveMassesA': a, 'positiveMassesB': b, 'sameStrictDescendingOrder': True,
        'normalizedA': [str(Fraction(v, sum(a))) for v in a], 'normalizedB': [str(Fraction(v, sum(b))) for v in b],
        'scope': 'Supplied rational counterexample:ordering does not determine normalized masses. No fitted coefficients or source-model observation.'}


def analyze():
    started = time.perf_counter()
    metadata_path = ROOT/'docs/results/source-order-program/compile-program.json'
    manifest_path = ROOT/'docs/results/source-order-program/atom-manifest.json'
    metadata = json.loads(metadata_path.read_text()); manifest = json.loads(manifest_path.read_text())
    alphabet = metadata['alphabet']; length = metadata['length']; dimension = manifest['headDim']; heads = manifest['qWidth']//dimension
    if len(alphabet) != 30 or alphabet != sorted(set(alphabet)) or length != 512 or dimension != 128 or heads != 32 or manifest['vocabSize'] != 151936 or metadata['paddingQueryID'] != 151643:
        raise ValueError('Qualified source geometry differs')
    counts = census(len(alphabet), length, manifest['vocabSize'], heads, dimension)
    if counts['currentDomainScores'] != metadata['ordinalCount']:
        raise ValueError('Original rank-domain census differs')
    paths = ['generator/source_order_compile.py', 'generator/source_order_program.py', 'generator/encoder_atoms.py',
        'docs/results/source-order-program/compile-program.json', 'docs/results/source-order-program/atom-manifest.json',
        'docs/plans/first-read-relative-mass-producer.md', 'generator/relative_mass_feasibility.py', 'generator/test_relative_mass_feasibility.py']
    return {'schema': 'tce-relative-mass-feasibility-v1', 'sourceHashes': {name: sha(ROOT/name) for name in paths},
        'sourceDomain': {'alphabet': alphabet, 'length': length, 'head': 0, 'paddingQueryID': 151643, 'dimension': dimension, 'vocabulary': manifest['vocabSize'], 'heads': heads},
        'literalLayoutCensus': counts, 'rotaryWitness': rotary_witness(), 'weightWitness': weight_witness(),
        'candidate': {'kind': 'offline token/absolute-position scalar-score partial evaluation',
            'scoreContract': metadata['scoreContract'], 'historyDependentTable': False,
            'serving': 'Reuse95 own integer supports;look up up to4fixed FP64 scores;execute declared4-way normalization. Values/mixing remain separately charged.',
            'onlineQKCoordinateProducts': 0, 'normalizationStillExecuted': True, 'implementationExists': False,
            'scope': 'One finite learned primitive compiled offline,not a full zero-neural reader/generator. Scores absent from95 retained asset;rank/hash receipts cannot provide their magnitudes.'},
        'alternatives': [
            {'kind': 'ideal-relative-position Fourier law', 'directCoefficientProductsPerScore': dimension, 'scope': 'Direct64-frequency real rewrite still pays128learned coefficient products;not a lower bound on all possible programs. Source BF16 rotations do not inherit ideal shift symmetry.'},
            {'kind': 'direct trigonometric clock recurrence', 'scope': 'Advancing phases can avoid trig recomputation but does not eliminate coefficient weighting or final sums;exact finite source equivalence and a genuinely cheaper lowering remain unestablished.'}],
        'newModelEvaluations': 0, 'scientificArrayReads': 0, 'newLearnedScoreProducts': 0, 'newImages': 0,
        'mechanics_status': 'complete', 'instrument_status': 'source-bound-counts-and-supplied-rational-witnesses',
        'trend_status': 'Finite partial evaluation is concrete;literal all-vocabulary/active-query scaling motivates explicit economy constraints',
        'terminal_status': 'No weight producer executed;arbitrary-prompt zero-neural images unestablished',
        'analysisSecondsBeforeReportWrite': time.perf_counter()-started,
        'scope': 'Local source/public-metadata and tiny integer/rational mechanics only. Byte counts are hypothetical uncompressed layouts;no table compiled/no mrun/native/model/scientific-array work. No universal lower-bound or impossibility claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('output'); args = parser.parse_args(); value = analyze()
    with Path(args.output).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps({'counts': value['literalLayoutCensus'], 'witness': value['rotaryWitness'], 'seconds': value['analysisSecondsBeforeReportWrite']}))
