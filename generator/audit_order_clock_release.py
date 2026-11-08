"""Cycle96 release bytes/current-state/backlinks; no scientific rerun."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit():
    checks = []
    def check(ok, name):
        if not ok: raise ValueError(name)
        checks.append(name)
    r = ROOT/'docs/results/order-clock-recurrence'; e = ROOT/'docs/results/order-clock-recurrence-evidence'
    run = ROOT/'artifacts/order-clock-recurrence-v1'; pub = read(r/'publication.json'); collection = read(r/'collection.json')
    for name, item in pub['files'].items():
        check(sha(r/name) == item['sha256'] and (r/name).stat().st_size == item['bytes'], 'Published member '+name)
    for name, item in collection['inventory']['files'].items():
        check(sha(run/'result'/name) == sha(r/name) == item['sha256'], 'Raw byte copy '+name)
    check(len(collection['inventory']['files']) == pub['rawFiles'] == 64 and collection['inventory']['bytes'] == pub['rawBytes'] == 8558900, 'Complete raw census')
    request = read(r/'request.json'); source_review = read(e/'advisor-source.json')
    for name in request['ownedSources']:
        check(sha(ROOT/'generator'/name) == request['sourceHashes'][name], 'Frozen executed source '+name)
    check(sha(ROOT/'docs/plans/order-clock-recurrence.md') == request['sourceHashes']['plan.md'], 'Frozen executed plan')
    for name, digest in source_review['finalPins'].items(): check(sha(ROOT/name) == digest, 'Final pre-execution source review '+name)
    tests = read(r/'mechanics.json')
    check(tests['returncode'] == 0 and 'Ran 8 tests in 0.050s' in tests['stderr'], 'Eight targeted prior tests')
    for name, digest in tests['sourceHashes'].items(): check(digest == request['sourceHashes'][name], 'Test/source '+name)
    scalar = read(r/'scalar-audit.json'); receipt = read(r/'prediction-receipt.json'); report = read(r/'report.json')
    check(scalar['auditorSHA256'] == pub['auditorSHA256'] == sha(ROOT/'generator/publish_order_clock.py'), 'Independent publisher identity')
    check((scalar['clockHashes'], scalar['independentlyDecodedOrdinals'], scalar['methodRows']) == (512, 3924480, 26624), 'Complete ordinal/method proof')
    check(scalar['methodMismatches'] == scalar['sparseProof']['mismatchedOrdinals'] == report['servingRowMismatches'] == 0, 'Recorded exact encoding')
    check(sum(receipt['indexPayload'].values()) == scalar['indexPayloadBytes'] == 15640256, 'Resident index cost retained')
    check(report['program']['rawBytes'] == 19555087 and report['program']['compressedBytes'] == 8062001, 'Adverse bytecode size retained')
    check(report['newLearnedScoreProducts'] == report['newOriginalModelEncoderDenoiserVAEEvaluations'] == report['newImages'] == 0, 'No new neural/image capability')
    final = read(e/'advisor-final-narratives.json')
    for name, item in final['currentSnapshots'].items(): check(sha(item['path']) == item['sha256'], 'Current final Astra '+name)
    actual = read(e/'advisor-actual-result.json')
    for name, digest in actual['pins'].items(): check(sha(r/name) == digest, 'Actual-result review '+name)
    survey_path = ROOT/'docs/results/research-review-cycle96/advisor-survey.json'; survey = read(survey_path)
    check(sha(survey_path) == 'ae88d23e38916a5b5ba6a1ce0bb54f420343a308d30d5fb9ab70257142e3d27e', 'Full survey96 pin')
    check(len(survey['contentReadSnapshots']) == 33 and survey['priorReview']['pinCount'] == 49 and survey['priorReview']['unchangedCount'] == 48, 'Full review scope')
    historical_blog = read(ROOT/'docs/results/source-order-program-evidence/advisor-final-narratives.json')['currentSnapshots']['livingBlog']['sha256']
    for item in survey['contentReadSnapshots']:
        if item['path'].endswith('2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'):
            check(item['sha256'] == historical_blog, 'Historical surveyed95 blog approval join')
        else: check(sha(WORKSPACE/item['path']) == item['sha256'], 'Current survey content pin '+item['path'])
    for item in survey['priorReview']['rechecks']:
        if item['path'].endswith('2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'):
            check(item['currentSHA256'] == historical_blog, 'Historical prior-comparison95 blog join')
        else: check(sha(WORKSPACE/item['path']) == item['currentSHA256'], 'Prior source recheck '+item['path'])
    root_review = read(ROOT/'docs/results/research-review-cycle96/root-pin-verification.json')
    check(root_review['advisorSurveySHA256'] == sha(survey_path) and len(root_review['currentReadPins']) == 33, 'Root survey check joins')
    storage = read(ROOT/'docs/results/storage-audit-cycle96/publication.json')
    check(storage['totalRestorePointers'] == 93 and storage['olderTerminalPointers'] == 84 and storage['currentByteSnapshotPointers'] == 9, 'Storage custody classes')
    check(storage['freshFullColdBytesVerified'] == 3806240603 and storage['remaining100MBCandidates'] == 0 and storage['newArchivedFiles'] == storage['newDeletedFiles'] == 0, 'Storage outcome')
    for name, digest in storage['files'].items(): check(sha(ROOT/'docs/results/storage-audit-cycle96'/name) == digest, 'Storage bytes '+name)
    for name, digest in storage['auditorSources'].items(): check(sha(ROOT/'generator'/name) == digest, 'Frozen storage auditor '+name)
    progress_path = ROOT/'docs/zero-neural-progress.json'; progress = read(progress_path)
    check(progress['status'] == 'active' and progress['researchCycle'] == 96 and progress['liveJobs'] == [], 'Active strict goal current state')
    check(progress['previousGoalTurn']['cycle'] == 95 and progress['previousGoalTurn']['commit'] == '00edc07eed4ccde4580387a70c57878e820d47cc' and progress['previousGoalTurn']['pushed'], 'Previous95 completed and pushed')
    check(progress['thisGoalTurn']['classification'] == 'progress' and progress['thisGoalTurn']['cycle'] == 96, 'Current96 actual progress')
    check(progress['reviews']['cycle96FinalStatus'] == 'approved' and progress['reviews']['cycle96Final']['sha256'] == sha(e/'advisor-final-narratives.json'), 'Current final review progress join')
    check(progress['findingsLog']['lastUpdatedCycle'] == 96 and progress['findingsLog']['sha256'] == final['currentSnapshots']['livingBlog']['sha256'], 'Current96 living blog distinct from surveyed95')
    check(progress['nextCut']['noJobSubmitted'] and progress['nextCut']['noImplementationQueued'] and progress['nextCut']['noAutomaticExpansion'], 'No automatic next campaign')
    check(progress['nextCut']['planSHA256'] == sha(ROOT/'docs/plans/first-read-route-payload.md'), 'Next question current plan')
    check(progress['nextCut']['fullReviewCurrentCycle'] == 96 and progress['nextCut']['nextPeriodicFullReviewCycle'] == storage['nextStorageReviewDueCycle'] == 99, 'Next review99')
    texts = {}
    for rel in ['docs/order-clock-recurrence.md', 'docs/results/order-clock-recurrence-evidence/README.md', 'docs/plans/first-read-route-payload.md', 'docs/results/research-review-cycle96/README.md', 'docs/results/storage-audit-cycle96/README.md']:
        path = ROOT/rel; texts[path] = path.read_text()
    for rel in ['README.md', 'generator/README.md']:
        path = ROOT/rel; texts[path] = path.read_text().split('\n\n')[1]
    path = ROOT/'docs/zero-neural-research.md'; texts[path] = next(x for x in path.read_text().splitlines() if x.startswith('Latest cycle96:'))
    blog = Path(final['currentSnapshots']['livingBlog']['path']); text = blog.read_text()
    texts[blog] = text.split('## Cycle 96:', 1)[1].split('## What would close the original goal', 1)[0]
    links = []
    for path, text in texts.items():
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if '://' in link or link.startswith('#'): continue
            check((path.parent/link.split('#', 1)[0]).resolve().exists(), 'Local backlink '+str(path.relative_to(WORKSPACE))+':'+link)
            links.append({'source': str(path), 'target': link})
    return {'schema': 'tce-order-clock-release-qa-v1', 'checks': len(checks), 'passedChecks': checks, 'localLinks': len(links), 'links': links,
            'progressSHA256': sha(progress_path), 'finalAdvisorSHA256': sha(e/'advisor-final-narratives.json'), 'auditorSHA256': sha(__file__),
            'scope': 'Release byte/integrity/metadata/current-state/link checks. Original integer proof/independent publication/eight targeted tests completed separately; no scientific/model/score/image/test rerun.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('output'); args = p.parse_args(); result = audit()
    with Path(args.output).open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k: result[k] for k in ['checks', 'localLinks', 'scope']}))
