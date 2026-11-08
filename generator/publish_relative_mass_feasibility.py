"""Independent integer/rational, source and release-link audit; no array/model run."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value):
    with Path(path).open('x') as stream: json.dump(value, stream, indent=2); stream.write('\n')


def audit():
    result = ROOT/'docs/results/relative-mass-feasibility'; report = read(result/'report.json'); checks = []
    def check(ok, name):
        if not ok: raise ValueError(name)
        checks.append(name)
    for name, digest in report['sourceHashes'].items(): check(sha(ROOT/name) == digest, 'Frozen source/public input '+name)
    geometry = report['sourceDomain']; a = len(geometry['alphabet']); length = geometry['length']; v = geometry['vocabulary']; heads = geometry['heads']; dim = geometry['dimension']
    pairs = sum(range(length)); count = a*pairs; fixed = v*pairs; all_queries = v*v*pairs; diagonal = v*length
    expected = {'strictPastPositionPairs': pairs, 'currentDomainScores': count, 'currentDomainFP64Bytes': 8*count,
        'projectedCompileCoordinateProducts': dim*count, 'idealRelativeOffsetScores': a*(length-1), 'idealRelativeOffsetFP64Bytes': 8*a*(length-1),
        'fixedQueryFullVocabularyScoresPerHead': fixed, 'fixedQueryFullVocabularyFP64BytesPerHead': 8*fixed,
        'fixedQueryFullVocabularyFP64BytesAllHeads': 8*fixed*heads, 'allQueryIDsStrictPastScoresPerHead': all_queries,
        'allQueryIDsStrictPastFP64BytesAllHeads': 8*all_queries*heads, 'additionalLegalActiveDiagonalScoresPerHead': diagonal,
        'allQueryIDsWithLegalDiagonalFP64BytesAllHeads': 8*(all_queries+diagonal)*heads}
    for name, value in expected.items(): check(report['literalLayoutCensus'][name] == value, 'Independent integer census '+name)
    witness = report['rotaryWitness']; c = Fraction(witness['bf16CosStep']); s = Fraction(witness['bf16SinStep']); q = Fraction(witness['bf16CosAndSinTwoSteps'])
    check(c == Fraction(237, 256) and s == Fraction(98, 256) and q == Fraction(181, 256), 'Supplied rounded operands')
    check(Fraction(witness['firstScore']) == c and Fraction(witness['translatedScore']) == q*(c+s), 'Independent rational contractions')
    check(Fraction(witness['signedDifference']) == q*(c+s)-c == -Fraction(37, 65536), 'Exact signed same-offset defect')
    weights = report['weightWitness']; masses = [weights['positiveMassesA'], weights['positiveMassesB']]
    check(weights['sameStrictDescendingOrder'] is True and all(all(x > 0 for x in values) and all(x > y for x, y in zip(values, values[1:])) for values in masses), 'Positive masses with same strict descending order')
    left = [Fraction(x, sum(masses[0])) for x in masses[0]]; right = [Fraction(x, sum(masses[1])) for x in masses[1]]
    check(list(map(str, left)) == weights['normalizedA'] and list(map(str, right)) == weights['normalizedB'] and left != right, 'Independent order/magnitude ambiguity')
    check(report['newModelEvaluations'] == report['scientificArrayReads'] == report['newLearnedScoreProducts'] == report['newImages'] == 0 and not report['candidate']['implementationExists'], 'No producer/model/array execution claim')
    mechanics = read(result/'mechanics.json'); check(mechanics['returncode'] == 0 and 'Ran 3 tests' in mechanics['stderr'], 'Original three mechanics results')
    for name, digest in mechanics['sourceHashes'].items(): check(sha(ROOT/name) == digest, 'Mechanics source '+name)
    check(read(result/'execution.json')['returncode'] == 0, 'Sole successful report process')
    before = read(result/'advisor-source.json')
    for name, digest in before['pins'].items(): check(sha(ROOT/name) == digest, 'Preexecution approved source '+name)
    for name, digest in before['old95Bindings'].items(): check(sha(ROOT/'docs/results/source-order-program'/name) == digest, 'Observed95 metadata pin '+name)
    final = read(result/'advisor-final.json')
    for name, item in final['snapshots'].items(): check(sha(item['path']) == item['sha256'], 'Current final advisor snapshot '+name)
    progress = read(ROOT/'docs/zero-neural-progress.json')
    check(progress['status'] == 'active' and progress['researchCycle'] == progress['thisGoalTurn']['cycle'] == 98 and progress['liveJobs'] == [], 'Current active goal/no live jobs')
    check(progress['previousGoalTurn']['cycle'] == 97 and progress['previousGoalTurn']['commit'] == 'afd2f78bf0ee312a9697850d85ab5519f3b9343a' and progress['previousGoalTurn']['pushed'], 'Prior97 completed/pushed')
    check(progress['thisGoalTurn']['classification'] == 'progress', 'Actual current feasibility progress')
    check(progress['reviews']['cycle98FinalStatus'] == 'approved' and progress['reviews']['cycle98Final']['sha256'] == sha(result/'advisor-final.json'), 'Final review metadata join')
    check(progress['nextCut']['planSHA256'] == sha(ROOT/'docs/plans/compiled-selected-mass.md') and progress['nextCut']['noImplementationQueued'] and progress['nextCut']['noJobSubmitted'], 'Next plan bound/unexecuted')
    check(progress['nextCut']['requiresFullReviewBeforeImplementationCycle'] == 99 and progress['reviews']['lastFullSurveyCycle'] == 96, 'Full review99 before implementation')
    check(sha(ROOT/'docs/results/research-review-cycle96/advisor-survey.json') == 'ae88d23e38916a5b5ba6a1ce0bb54f420343a308d30d5fb9ab70257142e3d27e', 'Historical full research96')
    check(sha(ROOT/'docs/results/storage-audit-cycle96/publication.json') == 'd2d31cf9f367a61c01a521e17856b0e5971ae38d1bce92906a104432c6b128fc', 'Historical full storage96')
    texts = {}
    for name in ['docs/relative-mass-producer-feasibility.md', 'docs/plans/compiled-selected-mass.md']:
        path = ROOT/name; texts[path] = path.read_text()
    for name in ['README.md', 'generator/README.md']:
        path = ROOT/name; texts[path] = path.read_text().split('\n\n')[1]
    path = ROOT/'docs/zero-neural-research.md'; texts[path] = next(x for x in path.read_text().splitlines() if x.startswith('Latest cycle98:'))
    blog = ROOT.parent/'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'; content = blog.read_text()
    check(sha(blog) == progress['findingsLog']['sha256'] and progress['findingsLog']['lastUpdatedCycle'] == 98, 'Current living blog')
    check('timestamp: 2026-11-06T00:42:04-08:00' in content and content.count('## Cycle 98:') == 1 and 'The latest cycle 98 study' in content, 'Original blog timestamp/current unique lead-section')
    texts[blog] = content.split('## Cycle 98:', 1)[1].split('## What would close the original goal', 1)[0]
    links = []
    for path, text in texts.items():
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if '://' in link or link.startswith('#'): continue
            check((path.parent/link.split('#', 1)[0]).resolve().exists(), 'Local link '+str(path.relative_to(ROOT.parent))+':'+link)
            links.append({'source': str(path), 'target': link})
    return {'schema': 'tce-relative-mass-release-audit-v1', 'checks': len(checks), 'passedChecks': checks, 'localLinks': len(links), 'links': links,
        'reportSHA256': sha(result/'report.json'), 'progressSHA256': sha(ROOT/'docs/zero-neural-progress.json'), 'auditorSHA256': sha(__file__),
        'scope': 'Independent integer/rational/source/public-metadata/release-link verification. No tiny test/helper/model/scientific-array rerun;no new timing,compression or general capability claim.'}


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args(); result = ROOT/'docs/results/relative-mass-feasibility'; value = audit(); save(result/'release-qa.json', value)
    save(result/'publication.json', {'schema': 'tce-relative-mass-publication-v1', 'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(result.iterdir()) if p.is_file()},
        'auditorSHA256': sha(__file__), 'scope': value['scope']})
    print(json.dumps({k: value[k] for k in ['checks', 'localLinks', 'scope']}))
