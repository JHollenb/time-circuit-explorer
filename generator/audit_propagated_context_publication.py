"""Compact engineering/publication joins; no model or activation execution."""
import gzip
import json
from pathlib import Path
import re
from encoder_atoms import read, save, sha
from publish_propagated_context import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/propagated-context'
    e = root / 'docs/results/propagated-context-evidence'
    mechanics = audit(p)
    report, progress, summary, advice, tests, instrument = [read(x) for x in
        [p / 'report.json', root / 'docs/zero-neural-progress.json', e / 'scalar-summary.json', e / 'advisor.json', e / 'tests.json', e / 'saturn-instrument.json']]
    if (progress['researchCycle'], progress['status'], progress['liveJobs']) != (56, 'active', []):
        raise ValueError('Active goal/completed job state differs')
    if progress['lastCompletedResearchTurn']['commit'] != '587a3ae3f41b82756d63ffb03a012c23d50e4a73' or any(progress[k]['classification'] != 'progress' for k in ['thisGoalTurn', 'previousGoalTurn']):
        raise ValueError('Previous/current progress evidence differs')
    recorded = progress['propagatedContextEvidence']
    if recorded['reportSHA256'] != sha(p / 'report.json') or recorded['scalarSummarySHA256'] != sha(e / 'scalar-summary.json'):
        raise ValueError('Progress evidence snapshots differ')
    for name in ['effects', 'signedWriteComposition', 'totalNewMatrixMACs', 'distinctBF16ParameterBytes', 'processBF16ParameterBytes', 'affineCalls', 'attentionCalls', 'observations']:
        if recorded[name] != report[name]:
            raise ValueError('Progress/result field differs: ' + name)
    request, inventory = read(p / 'request.json'), read(p / 'byte-inventory.json')
    if sha(p / 'frozen-plan.md') != request['sourceHashes']['frozen-plan.md'] or inventory['sourceMetadataSHA256'] != sha(p / 'source-metadata.json') or inventory['verifierSourceSHA256'] != sha(root / 'generator/propagated_context_intake.py'):
        raise ValueError('Original plan/source-byte inventory changed')
    rows = [json.loads(x) for x in gzip.decompress((p / 'observations.jsonl.gz').read_bytes()).splitlines()]
    expected = {(field, 'whole-row', row, None) for field in report['effects'] for row in request['support']}
    expected |= {(field, 'query-head', row, head) for field in ['nextQProjected', 'nextQNormalized', 'nextQ'] for row in request['support'] for head in range(24)}
    actual = [(x['field'], x['panel'], x['row'], x.get('head')) for x in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Complete unique field/row/head coverage differs')
    if summary['newArrays'] != 45 or summary['sourceChangedOutsideSupportRows'] != {'image': 1024, 'text': 508}:
        raise ValueError('Source/storage summary differs')
    for name, digest in advice['reviewedSources'].items():
        if sha(root / 'generator' / name) != digest:
            raise ValueError('Advisor source snapshot differs')
    for name, digest in advice['reviewedResults'].items():
        if sha(p / name) != digest:
            raise ValueError('Advisor result snapshot differs')
    if tests['sourceSHA256'] != sha(root / tests['source']) or (tests['passed'], tests['exitCode']) != (284, 0):
        raise ValueError('Actual test snapshot differs')
    if (instrument['records'], instrument['predicates'], instrument['runExit'], instrument['verifyExit'], instrument['formalReceipts']) != (348, 11, 0, 0, 0):
        raise ValueError('Actual instrument coverage differs')
    blog = root / progress['findingsLog']['path'];text = blog.read_text()
    if progress['findingsLog']['lastUpdatedCycle'] != 56 or progress['findingsLog']['sha256'] != sha(blog):
        raise ValueError('Living blog snapshot differs')
    section = text.split('## Cycle 56:', 1)[1].split('## What would close the original goal', 1)[0]
    for n in advice['reviewedSources']:
        if '../../time-circuit-explorer/generator/' + n not in text.split('---', 2)[1]:
            raise ValueError('Executed-source blog backlink absent')
    plan = root / 'docs' / progress['nextCut']['plan']
    for path, contents in [(blog, section), (root / 'docs/propagated-context-query.md', None),
                           (plan, None), (e / 'README.md', None)]:
        for link in re.findall(r'\]\(([^)]+)\)', contents if contents is not None else path.read_text()):
            if link.startswith(('https:', 'http:')):
                continue
            target = (path.parent / link.split('#', 1)[0]).resolve()
            if target != (e / 'publication-qa.json').resolve() and not target.exists():
                raise ValueError('Publication backlink absent: ' + link)
    if progress['nextCut']['status'] != 'written-unexecuted' or 'written, unexecuted' not in plan.read_text() or progress['nextCut']['totalNewMatrixMACs'] != 905969664:
        raise ValueError('Prespecified next-plan scope differs')
    figure = root / 'docs/results/propagated-context-figure/propagated-context.png'
    frec = read(figure.parent / 'figure.json')
    if sha(figure) != sha(root.parent / 'obsidian/images/zero-neural-cycle56-propagated-context.png') or sha(figure) != frec['imageSHA256'] or frec['reportSHA256'] != sha(p / 'report.json') or frec['plotSourceSHA256'] != sha(root / 'generator/plot_propagated_context.py'):
        raise ValueError('Figure source/data/notebook copy differs')
    if (progress['storageMaintenance']['lastActualArchiveCycle'], progress['storageMaintenance']['lastAuditCycle'], progress['storageMaintenance']['nextDueByCycle'], progress['reviews']['lastCycle'], progress['reviews']['nextDueByCycle']) != (54, 54, 57, 56, 59):
        raise ValueError('Maintenance/advisor cadence differs')
    result = {**mechanics, 'schema': 'tce-propagated-context-publication-qa-v1', 'reportSHA256': sha(p / 'report.json'),
              'progressSHA256': sha(root / 'docs/zero-neural-progress.json'), 'blogSHA256': sha(blog), 'figureSHA256': sha(figure),
              'advisorSHA256': sha(e / 'advisor.json'), 'nextPlanSHA256': sha(plan), 'testsPassed': 284,
              'nextStorageCycle': 57, 'nextAdvisorCycle': 59, 'auditSourceSHA256': sha(__file__),
              'mutableAdvisorPostObservations': [{**x, 'currentSHA256': sha((e / x['path']).resolve())} for x in advice['blogSnapshots']],
              'scope': 'Engineering source/seal/byte/typed-ABI/complete-address/narrative consistency. Not a full-width native parity, semantic or arbitrary-prompt zero-neural certification.'}
    save(e / 'publication-qa.json', result)
    return {k: result[k] for k in ['ownedSourcePins', 'typedActs', 'explicitOutputABIs', 'ownStateQueryJoins', 'observations', 'nextStorageCycle', 'nextAdvisorCycle']}


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
