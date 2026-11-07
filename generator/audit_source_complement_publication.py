"""Check compact publication joins; no model or tensor execution."""
import json
from pathlib import Path
import re
from encoder_atoms import read, save, sha
from publish_source_complement import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/source-complement'
    e = root / 'docs/results/source-complement-evidence'
    mechanics = audit(p)
    r, summary, progress, advice, tests, instrument = [read(path) for path in
        [p / 'report.json', e / 'scalar-summary.json', root / 'docs/zero-neural-progress.json',
         e / 'advisor.json', e / 'tests.json', e / 'saturn-instrument.json']]
    if (progress['researchCycle'], progress['status'], progress['liveJobs']) != (55, 'active', []):
        raise ValueError('Active goal/completed job state differs')
    if progress['lastCompletedResearchTurn']['commit'] != 'fc8541018553f46a973c68998f0bd0b0383820c3':
        raise ValueError('Previous completed research turn differs')
    recorded = progress['sourceComplementEvidence']
    if recorded['reportSHA256'] != sha(p / 'report.json') or recorded['scalarSummarySHA256'] != sha(e / 'scalar-summary.json'):
        raise ValueError('Progress evidence snapshot differs')
    for name in ['responses', 'endpoints', 'queryPanels', 'sourceMetrics', 'sourceRowMetrics',
                 'contextContrast', 'selectedOperandFingerprints', 'totalNewMatrixMACs', 'work']:
        if recorded[name] != r[name]:
            raise ValueError('Progress report field differs: ' + name)
    if summary['instrument_status'] != 'context-contrast-inapplicable-identical-complements' or any(not x['byteIdentical'] for x in summary['complementByteIdentity'].values()):
        raise ValueError('Additive unchanged-context diagnosis differs')
    receipt = read(p / 'retained-array-verification.json')
    for field in ['K', 'V']:
        witnesses = {x['arm']: x for x in receipt['bankSpliceWitnesses'] if x['field'] == field}
        actual = summary['complementByteIdentity'][field]
        if actual['primarySHA256'] != witnesses['primary']['outsideSupportSHA256'] or actual['controlSHA256'] != witnesses['control']['outsideSupportSHA256']:
            raise ValueError('Actual complement byte witness differs')
    for name, digest in {**tests['sources'], **advice['reviewedSources']}.items():
        if sha(root / 'generator' / name) != digest:
            raise ValueError('Test/advisor source snapshot differs')
    for name, digest in advice['reviewedResults'].items():
        if sha(p / name) != digest:
            raise ValueError('Advisor result snapshot differs')
    if (tests['passed'], tests['exitCode'], instrument['runExit'], instrument['verifyExit'], instrument['records']) != (280, 0, 0, 0, 3072):
        raise ValueError('Actual tests/instrument scope differs')
    if [(x['records'], x['predicates']) for x in instrument['perStream']] != [(1024, 7), (2048, 19)]:
        raise ValueError('Actual stream adaptation differs')
    blog = root / progress['findingsLog']['path']
    text = blog.read_text()
    if progress['findingsLog']['lastUpdatedCycle'] != 55 or progress['findingsLog']['sha256'] != sha(blog):
        raise ValueError('Living findings snapshot differs')
    section = text.split('## Cycle 55:', 1)[1].split('## What would close the original goal', 1)[0]
    for name in advice['reviewedSources']:
        if '../../time-circuit-explorer/generator/' + name not in text.split('---', 2)[1]:
            raise ValueError('Executed-source blog backlink absent')
    docs = ['docs/source-complement-transfer.md', 'docs/plans/propagated-context-support.md',
            'docs/results/source-complement-evidence/README.md']
    for path, content in [(blog, section)] + [(root / name, None) for name in docs]:
        for target in re.findall(r'\]\(([^)]+)\)', content if content is not None else path.read_text()):
            if target.startswith(('https:', 'http:')):
                continue
            resolved = (path.parent / target.split('#', 1)[0]).resolve()
            if resolved != (e / 'publication-qa.json').resolve() and not resolved.exists():
                raise ValueError('Publication backlink absent: ' + target)
    figure = root / 'docs/results/source-complement-figure/source-complement.png'
    frec = read(figure.parent / 'figure.json')
    if sha(figure) != sha(root.parent / 'obsidian/images/zero-neural-cycle55-source-complement.png') or sha(figure) != frec['imageSHA256'] or frec['reportSHA256'] != sha(p / 'report.json') or frec['plotSourceSHA256'] != sha(root / 'generator/plot_source_complement.py'):
        raise ValueError('Figure data/source/copy differs')
    cut = progress['nextCut']
    plan = root / 'docs' / cut['plan']
    if cut['status'] != 'written-unexecuted' or 'written, unexecuted' not in plan.read_text() or sum(cut[k] for k in ['twoFullKVBanksMACs', 'sharedSelectedQueryMACs', 'twoAttentionMACs', 'twoOutputMLPMACs', 'twoNextQueryMACs']) != cut['totalNewMatrixMACs']:
        raise ValueError('Next plan/work scope differs')
    if (progress['storageMaintenance']['lastActualArchiveCycle'], progress['storageMaintenance']['lastAuditCycle'], progress['storageMaintenance']['nextDueByCycle'], progress['reviews']['lastCycle'], progress['reviews']['nextDueByCycle']) != (54, 54, 57, 55, 58):
        raise ValueError('Actual maintenance/advisor cadence differs')
    post_observations = [{**x, 'currentSHA256': sha((e / x['path']).resolve()), 'scope': 'Mutable post; original reviewed snapshot retained.'} for x in advice['blogSnapshots']]
    result = {**mechanics, 'schema': 'tce-source-complement-publication-qa-v1',
              'reportSHA256': sha(p / 'report.json'), 'progressSHA256': sha(root / 'docs/zero-neural-progress.json'),
              'blogSHA256': sha(blog), 'figureSHA256': sha(figure), 'advisorSHA256': sha(e / 'advisor.json'),
              'nextPlanSHA256': sha(plan), 'mutableAdvisorPostObservations': post_observations,
              'testsPassed': 280, 'contextQuestion': 'inapplicable-identical-complement',
              'nextStorageCycle': 57, 'nextAdvisorCycle': 58, 'auditSourceSHA256': sha(__file__),
              'scope': 'Engineering custody and publication consistency; the unchanged context contrast cannot establish sufficiency. No arbitrary-prompt zero-neural certification.'}
    save(e / 'publication-qa.json', result)
    return {k: result[k] for k in ['ownedSourcePins', 'typedSharedActs', 'typedSourceActs', 'typedRecipientActs',
                                 'queryRecords', 'contextEffectRecords', 'contextQuestion', 'nextStorageCycle', 'nextAdvisorCycle']}


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
