"""Release identity/state/link QA for97; no scientific or test rerun."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    checks = []
    def check(ok, name):
        if not ok: raise ValueError(name)
        checks.append(name)
    result = ROOT/'docs/results/first-read-route-payload'; evidence = ROOT/'docs/results/first-read-route-payload-evidence'
    run = ROOT/'artifacts/first-read-route-payload-v1/run'
    publication = read(result/'publication.json'); collection = read(result/'collection.json'); request = read(result/'request.json')
    for name, item in publication['files'].items():
        check(sha(result/name) == item['sha256'] and (result/name).stat().st_size == item['bytes'], 'Publication byte '+name)
    for name, item in collection['inventory']['files'].items():
        check(sha(result/name) == sha(run/'result'/name) == item['sha256'], 'Raw byte copy '+name)
    check(publication['rawFiles'] == len(collection['inventory']['files']) == 13 and publication['rawBytes'] == collection['inventory']['bytes'] == 252819, 'Original compact census')
    for name in ['first_read_payload.py', 'first_read_payload_study.py', 'test_first_read_payload.py', 'encoder_atoms.py']:
        check(sha(ROOT/'generator'/name) == request['sourceHashes'][name], 'Frozen executed source '+name)
    for original, staged in [('first-read-route-payload.md', 'plan.md'), ('first-read-route-payload-implementation.md', 'implementation.md')]:
        check(sha(ROOT/'docs/plans'/original) == request['sourceHashes'][staged], 'Frozen executed plan '+original)
    source = read(evidence/'advisor-source.json')
    for name, digest in source['finalPins'].items(): check(sha(ROOT/name) == digest, 'Exact pre-execution review '+name)
    mechanics = read(result/'mechanics.json')
    check(mechanics['returncode'] == 0 and 'Ran 9 tests in 0.025s' in mechanics['stderr'], 'Nine original targeted mechanics')
    for name, digest in mechanics['sourceHashes'].items(): check(digest == request['sourceHashes'][name], 'Test source join '+name)
    actual = read(evidence/'advisor-actual-result.json')
    for name, digest in actual['pins'].items(): check(sha(result/name) == digest, 'Actual-result reviewed byte '+name)
    for name, digest in actual['sourcePins'].items(): check(sha(ROOT/name) == digest, 'Actual-result reviewed helper '+name)
    scalar = read(result/'scalar-audit.json'); report = read(result/'report.json')
    check(scalar['auditorSHA256'] == publication['auditorSHA256'] == sha(ROOT/'generator/publish_first_read_payload.py'), 'Independent publisher identity')
    check(scalar['scalarRows'] == 477 and scalar['panels'] == 3 and scalar['allScalarAggregatesMatch'], 'Complete independent scalar account')
    check(scalar['retainedArrayBytes'] == 15641082 and scalar['freshRemoteByteVerification']['bytes'] == 89081598, 'Fresh remote byte verification scope')
    check(report['crossReadProducts'] == report['endpointReplayProducts'] == 488448, 'Two distinct product costs')
    check(report['newQKProducts'] == report['newSoftmaxEvaluations'] == report['newNeuralModelEncoderDenoiserVAEEvaluations'] == report['newImages'] == 0, 'No new neural/image work')
    for name, item in report['endpointReplay'].items():
        check(item['coordinates'] == 61056 and item['rawBitDifferences'] == item['bf16WordDifferences'] == item['retainedRawAdmissionWordDifferences'] == 0, 'Recorded endpoint profile '+name)
    summary = read(result/'summary-publication.json'); common = read(result/'common-weight-publication.json')
    check(summary['publicationSHA256'] == common['publicationSHA256'] == sha(result/'publication.json'), 'Additive supplements bind original publication')
    for name, digest in summary['sourceHashes'].items(): check(sha(result/name) == digest, 'Scalar summary input/output '+name)
    check(summary['summarizerSHA256'] == sha(ROOT/'generator/summarize_first_read_payload.py'), 'Scalar summary helper')
    check(common['resultSHA256'] == sha(result/'common-weight-ratio.json') and common['helperSHA256'] == sha(ROOT/'generator/common_weight_ratio.py'), 'Common-weight scalar helper/result')
    check(common['sourceRowsSHA256'] == sha(result/'rows.jsonl.gz') and common['actualAdvisorSHA256'] == sha(evidence/'advisor-actual-result.json'), 'Common-weight source/review')
    final = read(evidence/'advisor-final-narratives.json')
    for name, item in final['currentSnapshots'].items(): check(sha(item['path']) == item['sha256'], 'Current final Astra snapshot '+name)
    progress = read(ROOT/'docs/zero-neural-progress.json')
    check(progress['status'] == 'active' and progress['researchCycle'] == progress['thisGoalTurn']['cycle'] == 97 and progress['liveJobs'] == [], 'Active strict goal/no live job')
    check(progress['thisGoalTurn']['classification'] == 'progress' and progress['previousGoalTurn']['cycle'] == 96 and progress['previousGoalTurn']['commit'] == 'bbc1a97ac4a4c1a81084e10b318156c07bac50de' and progress['previousGoalTurn']['pushed'], 'Prior completed/current progress chronology')
    check(progress['reviews']['cycle97FinalStatus'] == 'approved' and progress['reviews']['cycle97Final']['sha256'] == sha(evidence/'advisor-final-narratives.json'), 'Final current review state')
    check(progress['nextCut']['noJobSubmitted'] and progress['nextCut']['noImplementationQueued'] and progress['nextCut']['noAutomaticExpansion'], 'Design-only next question')
    check(progress['nextCut']['planSHA256'] == sha(ROOT/'docs/plans/first-read-relative-mass-producer.md'), 'Current next design identity')
    check(progress['nextCut']['articleSHA256'] == sha(ROOT/'docs/first-read-route-payload.md'), 'Current article identity')
    check(progress['reviews']['lastFullSurveyCycle'] == progress['nextCut']['fullReviewCurrentCycle'] == 96 and progress['nextCut']['nextPeriodicFullReviewCycle'] == 99, 'Periodic survey schedule')
    check(sha(ROOT/'docs/results/research-review-cycle96/advisor-survey.json') == 'ae88d23e38916a5b5ba6a1ce0bb54f420343a308d30d5fb9ab70257142e3d27e', 'Historical full survey96 preserved')
    check(sha(ROOT/'docs/results/storage-audit-cycle96/publication.json') == 'd2d31cf9f367a61c01a521e17856b0e5971ae38d1bce92906a104432c6b128fc', 'Full storage96 receipt preserved')
    check(progress['storageMaintenance']['cycle97Practice']['newArchivedFiles'] == progress['storageMaintenance']['cycle97Practice']['newDeletedFiles'] == 0, 'No invented cleanup')
    texts = {}
    for name in ['docs/first-read-route-payload.md', 'docs/plans/first-read-relative-mass-producer.md', 'docs/results/first-read-route-payload-evidence/README.md']:
        path = ROOT/name; texts[path] = path.read_text()
    for name in ['README.md', 'generator/README.md']:
        path = ROOT/name; texts[path] = path.read_text().split('\n\n')[1]
    path = ROOT/'docs/zero-neural-research.md'; texts[path] = next(line for line in path.read_text().splitlines() if line.startswith('Latest cycle97:'))
    blog = WORKSPACE/'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'; content = blog.read_text()
    check('timestamp: 2026-11-06T00:42:04-08:00' in content and 'tags: [blog, research, interpretability,' in content, 'Blog original metadata')
    check(progress['findingsLog']['lastUpdatedCycle'] == 97 and progress['findingsLog']['sha256'] == sha(blog), 'Current living blog identity')
    check(content.count('## Cycle 97:') == 1 and 'The latest cycle 97 assay' in content, 'Current blog lead/unique section')
    texts[blog] = content.split('## Cycle 97:', 1)[1].split('## What would close the original goal', 1)[0]
    links = []
    for path, text in texts.items():
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if '://' in link or link.startswith('#'): continue
            check((path.parent/link.split('#', 1)[0]).resolve().exists(), 'Local backlink '+str(path.relative_to(WORKSPACE))+':'+link)
            links.append({'source': str(path), 'target': link})
    return {'schema': 'tce-first-read-release-qa-v1', 'checks': len(checks), 'passedChecks': checks, 'localLinks': len(links), 'links': links,
        'progressSHA256': sha(ROOT/'docs/zero-neural-progress.json'), 'finalAdvisorSHA256': sha(evidence/'advisor-final-narratives.json'), 'auditorSHA256': sha(__file__),
        'scope': 'Release byte/metadata/current-state/backlink QA only. Original nine tests, one CPU vector assay and independent publication retained; no scientific/vector/model/image/test rerun.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('output'); args = parser.parse_args(); value = audit()
    with Path(args.output).open('x') as stream: json.dump(value, stream, indent=2); stream.write('\n')
    print(json.dumps({k: value[k] for k in ['checks', 'localLinks', 'scope']}))
