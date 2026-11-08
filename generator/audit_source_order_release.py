"""Release QA for cycle95 published bytes, evidence and current narratives."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def audit():
    checks = []
    def check(condition, name):
        if not condition:
            raise ValueError(name)
        checks.append(name)
    r = ROOT/'docs/results/source-order-program'
    e = ROOT/'docs/results/source-order-program-evidence'
    raw = ROOT/'artifacts/source-order-program-v1/result'
    pub = read(r/'publication.json')
    for name, item in pub['files'].items():
        check((r/name).stat().st_size == item['bytes'] and sha(r/name) == item['sha256'], 'Published member '+name)
    collection = read(r/'collection.json')
    for name, item in collection['inventory']['files'].items():
        check(sha(raw/name) == sha(r/name) == item['sha256'], 'Raw byte copy '+name)
    check(pub['rawFiles'] == len(collection['inventory']['files']) == 86 and pub['rawBytes'] == collection['inventory']['bytes'] == 9523323, 'Raw census')
    request = read(r/'request.json')
    for name in request['ownedSources']:
        check(sha(ROOT/'generator'/name) == request['sourceHashes'][name], 'Frozen executed source '+name)
    for name, staged in [('source-ordering-program-feasibility.md', 'frozen-plan.md'), ('source-ordering-implementation.md', 'implementation.md')]:
        check(sha(ROOT/'docs/plans'/name) == sha(r/staged) == request['sourceHashes'][staged], 'Frozen executed plan '+name)
    tests = read(r/'mechanics.json')
    check(tests['returncode'] == 0 and 'Ran 8 tests in 0.046s' in tests['stderr'], 'Eight targeted prior mechanics checks')
    for name, digest in tests['sourceHashes'].items():
        check(digest == request['sourceHashes'][name], 'Test source '+name)
    scalar = read(r/'scalar-audit.json')
    check(scalar['auditorSHA256'] == sha(ROOT/'generator/publish_source_order_v2.py') == pub['auditorSHA256'], 'Successful independent publisher')
    check((scalar['clockPermutations'], scalar['ordinals'], scalar['headQueryRows'], scalar['methodRows'], scalar['panels']) == (512, 3924480, 6656, 26624, 136), 'Complete integer/scalar audit census')
    check(scalar['supportedPrimaryByGroup'] == {'paddingNontrivial': 3265, 'paddingShort': 523}, 'Primary supported denominators')
    check(scalar['historicalEdit'] == {'supportedRows': 477, 'changedAddresses': 426, 'distinctSupportsA': 319, 'distinctSupportsB': 280}, 'Historical response denominators')
    check(scalar['syntheticEditChangedReferenceRows'] == 0 and scalar['supportedEmptyPrefixRows'] == 0, 'Weak and empty applicability retained')
    correction = read(e/'publication-instrument-correction.json')
    check(correction['originalSHA256'] == sha(ROOT/correction['originalAuditor']), 'Original publisher refusal source retained')
    check(correction['correctedSHA256'] == pub['auditorSHA256'], 'Separate publication correction')
    review = read(e/'advisor-final-narratives.json')
    for name, item in review['currentSnapshots'].items():
        check(sha(item['path']) == item['sha256'], 'Current final Astra snapshot '+name)
    actual_review = read(e/'advisor-actual-result.json')
    for name, digest in actual_review['pins'].items():
        check(sha(r/name) == digest, 'Actual result Astra snapshot '+name)
    source_review = read(e/'advisor-source.json')
    for name, digest in source_review['sourceHashes'].items():
        check(digest == request['sourceHashes'][name], 'Source review identity '+name)
    check('overlapped' in source_review['chronology'], 'Source review chronology explicit')
    storage = read(e/'storage-practice.json')
    check(storage['fullAuditCurrentCycle'] == 93 and storage['nextFullReviewCycle'] == 96 and storage['lastActualArchiveCycle'] == 78, 'Storage schedule')
    check(storage['newArchiveFiles'] == storage['newDeletedFiles'] == 0, 'No invented cleanup')
    check(sha(e/storage['priorAudit']) == storage['priorAuditSHA256'], 'Prior storage audit pin')
    check(sha(ROOT/'docs/results/research-review-cycle93/advisor-survey.json') == 'de69dc5433955ad9cbd54e9f261f6804fbb833928b4c044cd36fea2228d66d7d', 'Prior full research survey pin')
    p = read(ROOT/'docs/zero-neural-progress.json')
    check(p['status'] == 'active' and p['researchCycle'] == 95 and p['liveJobs'] == [], 'Active goal current cycle no pending jobs')
    check(p['previousGoalTurn']['cycle'] == 94 and p['previousGoalTurn']['commit'] == '82b1e3fca33034b9400a3a086014e92c38982ac5' and p['previousGoalTurn']['pushed'], 'Previous completed turn preserved')
    check(p['thisGoalTurn']['classification'] == 'progress' and p['thisGoalTurn']['cycle'] == 95, 'This turn bounded progress')
    check(p['reviews']['cycle95FinalStatus'] == 'approved' and p['reviews']['cycle95Final']['sha256'] == sha(e/'advisor-final-narratives.json'), 'Current review progress join')
    check(p['findingsLog']['sha256'] == review['currentSnapshots']['livingBlog']['sha256'] and p['findingsLog']['lastUpdatedCycle'] == 95, 'Living blog current join')
    check(p['nextCut']['nextPeriodicFullReviewCycle'] == 96 and p['nextCut']['noAutomaticExpansion'] and p['nextCut']['noImplementationQueued'], 'Conditional next question')
    check(p['sourceOrderProgramEvidence']['articleSHA256'] == review['currentSnapshots']['article']['sha256'], 'Current article progress join')
    texts = {}
    for rel in ['docs/source-ordering-program.md', 'docs/results/source-order-program-evidence/README.md']:
        path = ROOT/rel; texts[path] = path.read_text()
    for rel in ['README.md', 'generator/README.md']:
        path = ROOT/rel; texts[path] = path.read_text().split('\n\n')[1]
    path = ROOT/'docs/zero-neural-research.md'
    texts[path] = next(line for line in path.read_text().splitlines() if line.startswith('Latest cycle95:'))
    blog = Path(review['currentSnapshots']['livingBlog']['path']); text = blog.read_text()
    texts[blog] = text.split('## Cycle 95:', 1)[1].split('## What would close the original goal', 1)[0]
    links = []
    for path, text in texts.items():
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if '://' in link or link.startswith('#'):
                continue
            target = link.split('#', 1)[0]
            check((path.parent/target).resolve().exists(), 'Local backlink '+str(path.relative_to(ROOT.parent))+':'+target)
            links.append({'source': str(path), 'target': target})
    return {'schema': 'tce-source-order-release-qa-v1', 'checks': len(checks), 'passedChecks': checks, 'localLinks': len(links),
            'links': links, 'rawFiles': 86, 'rawBytes': 9523323, 'auditorSHA256': sha(__file__),
            'progressSHA256': sha(ROOT/'docs/zero-neural-progress.json'), 'finalAdvisorSHA256': sha(e/'advisor-final-narratives.json'),
            'scope': 'Byte/integrity/current-state/link release checks only. Independent scientific scalar/integer audit and eight targeted tests already completed separately; no test/model/scientific-score/image rerun.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('output'); args = p.parse_args()
    result = audit()
    with Path(args.output).open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k: result[k] for k in ['checks', 'localLinks', 'rawFiles', 'rawBytes', 'scope']}))
