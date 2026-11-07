"""Check published metadata joins without executing models or activation arithmetic."""
import json
from pathlib import Path
import re

from encoder_atoms import identity, read, save, sha
from publish_stale_key_context import audit
from stale_key_context_analysis import NAMES


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/stale-key-context'
    e = root / 'docs/results/stale-key-context-evidence'
    parent = root / 'docs/results/propagated-context'
    mechanics = audit(p)
    report, progress, summary, advice, tests, instrument = [read(x) for x in [
        p / 'report.json', root / 'docs/zero-neural-progress.json',
        e / 'scalar-summary.json', e / 'advisor.json', e / 'tests.json',
        e / 'saturn-instrument.json']]
    if (progress['researchCycle'], progress['status'], progress['liveJobs']) != (57, 'active', []):
        raise ValueError('Active goal/completed job state differs')
    if progress['lastCompletedResearchTurn']['commit'] != 'eaac20b2fc45e36bf44f322e7222cf1fe07cd92b':
        raise ValueError('Previous committed result differs')
    if any(progress[k]['classification'] != 'progress' for k in ['thisGoalTurn', 'previousGoalTurn']):
        raise ValueError('Current/previous progress classification differs')
    recorded = progress['staleKeyContextEvidence']
    if recorded['reportSHA256'] != sha(p / 'report.json') or recorded['scalarSummarySHA256'] != sha(e / 'scalar-summary.json'):
        raise ValueError('Progress evidence snapshots differ')
    for name in ['scores', 'totalNewMatrixMACs', 'distinctBF16ParameterBytes',
                 'processBF16ParameterBytes', 'affineCalls', 'attentionCalls', 'observations',
                 'primaryCandidate', 'responseReference', 'interactionLaw',
                 'conditionalFutureQKeyFormationOpportunityMACs']:
        if recorded[name] != report[name]:
            raise ValueError('Progress/result field differs: ' + name)
    if recorded['queryHeadResponseTrends'] != summary['queryHeadResponseTrends']:
        raise ValueError('Complete final-head trends differ')
    if recorded['fullInteractionArrayCount'] != len(report['interactions']) or len(report['interactions']) != 15:
        raise ValueError('Complete interaction account differs')
    request, inputs, targets = [read(p / n) for n in ['request.json', 'inputs.json', 'targets.json']]
    if sha(p / 'frozen-plan.md') != request['sourceHashes']['frozen-plan.md']:
        raise ValueError('Frozen executed plan differs')

    # Independently bind each target label to the parent's complete published record.
    for name, record in inputs['parentRecords'].items():
        if sha(parent / name) != record['sha256']:
            raise ValueError('Parent metadata snapshot differs: ' + name)
    banks, bank_seal = read(parent / 'banks.json'), read(parent / 'banks-seal.json')
    if bank_seal['banksSHA256'] != sha(parent / 'banks.json') or inputs['parentBankSealSHA256'] != sha(parent / 'banks-seal.json'):
        raise ValueError('Parent bank seal join differs')
    if inputs['selected'] != banks['selected']['fields']:
        raise ValueError('Shared selected operands differ from parent')
    for role in ['P', 'Q']:
        if inputs['banks'][role] != banks['banks'][role]['fields']:
            raise ValueError('Complementary bank origin differs: ' + role)
    for corner, phase in [('PP', 'control'), ('QQ', 'primary')]:
        arm, seal = read(parent / (phase + '.json')), read(parent / (phase + '-seal.json'))
        if seal[phase + 'SHA256'] != sha(parent / (phase + '.json')):
            raise ValueError('Parent diagonal seal differs: ' + corner)
        fields = {**arm['source']['fields'], **arm['nextQuery']['fields']}
        if targets['diagonals'][corner] != {name: fields[name] for name in NAMES}:
            raise ValueError('Parent diagonal target labels differ: ' + corner)
    if (summary['newArrays'], summary['newArrayBytes'], summary['fullByteFiles'],
        summary['NPYPayloadWitnesses']) != (49, 40999040, 115, 90):
        raise ValueError('Array/custody summary differs')
    for name, digest in advice['reviewedSources'].items():
        if sha(root / 'generator' / name) != digest:
            raise ValueError('Advisor source snapshot differs: ' + name)
    for name, digest in advice['reviewedResults'].items():
        if sha(p / name) != digest:
            raise ValueError('Advisor result snapshot differs: ' + name)
    if tests['sourceSHA256'] != sha(root / tests['source']) or (tests['passed'], tests['seconds'], tests['exitCode']) != (286, 18.19, 0):
        raise ValueError('Actual test snapshot differs')
    if (instrument['records'], instrument['predicates'], instrument['runExit'],
        instrument['verifyExit'], instrument['formalReceipts'], instrument['boundXrefAddresses']) != (348, 26, 0, 0, 0, 0):
        raise ValueError('Actual instrument coverage differs')

    blog = root / progress['findingsLog']['path']
    text = blog.read_text()
    if progress['findingsLog']['lastUpdatedCycle'] != 57 or progress['findingsLog']['sha256'] != sha(blog):
        raise ValueError('Living blog snapshot differs')
    section = text.split('## Cycle 57:', 1)[1].split('## What would close the original goal', 1)[0]
    for name in advice['reviewedSources']:
        if '../../time-circuit-explorer/generator/' + name not in text.split('---', 2)[1]:
            raise ValueError('Executed-source blog backlink absent: ' + name)
    plan = root / 'docs' / progress['nextCut']['plan']
    for path, contents in [(blog, section), (root / 'docs/stale-key-context-response.md', None),
                           (plan, None), (e / 'README.md', None),
                           (root / 'docs/results/storage-cleanup-cycle57/README.md', None),
                           (root / 'docs/results/storage-audit-cycle57/README.md', None)]:
        for link in re.findall(r'\]\(([^)]+)\)', contents if contents is not None else path.read_text()):
            if link.startswith(('https:', 'http:')):
                continue
            target = (path.parent / link.split('#', 1)[0]).resolve()
            if target != (e / 'publication-qa.json').resolve() and not target.exists():
                raise ValueError('Publication backlink absent: ' + link)
    next_cut = progress['nextCut']
    if next_cut['status'] != 'written-unexecuted' or 'written, unexecuted' not in plan.read_text():
        raise ValueError('Prospective plan status differs')
    for name, value in advice['plannedCosts'].items():
        if next_cut[name] != value:
            raise ValueError('Prospective cost differs: ' + name)
    planned_sum = sum(next_cut[name] for name in ['PDenominatorCompilerMACs', 'dualCompilerMACs',
                                               'actorMatrixMACs', 'postSealQProjectionMACs'])
    if planned_sum != next_cut['totalNewMatrixMACs'] or planned_sum != 21296578560 or next_cut['dualTableFP64Bytes'] != 153354240:
        raise ValueError('Prospective cost/space arithmetic differs')
    figure = root / 'docs/results/stale-key-context-figure/stale-key-context.png'
    frec = read(figure.parent / 'figure.json')
    if sha(figure) != sha(root.parent / 'obsidian/images/zero-neural-cycle57-stale-key-context.png') or sha(figure) != frec['imageSHA256'] or frec['reportSHA256'] != sha(p / 'report.json') or frec['plotSourceSHA256'] != sha(root / 'generator/plot_stale_key_context.py'):
        raise ValueError('Figure source/data/notebook copy differs')

    cleanup = root / 'docs/results/storage-cleanup-cycle57'
    storage_audit = root / 'docs/results/storage-audit-cycle57'
    for directory in [cleanup, storage_audit]:
        for member in directory.iterdir():
            if member.name == 'README.md':
                continue
            if sha(member) != sha(root / 'artifacts' / directory.name / member.name):
                raise ValueError('Exact storage receipt copy differs: ' + str(member))
    clean, post = read(cleanup / 'result.json'), read(storage_audit / 'result.json')
    cleanup_plan = read(cleanup / 'plan.json')
    plan_identity = identity({k: v for k, v in cleanup_plan.items() if k != 'planSHA256'})
    if plan_identity != cleanup_plan['planSHA256'] or clean['planSHA256'] != plan_identity or (clean['files'], clean['archivedBytes']) != (4, 626249557):
        raise ValueError('Actual cleanup identity/count differs')
    if len(post['pointerChecks']) != 60 or len(post['remoteMetadataChecks']) != 60 or (post['candidateFiles'], post['candidateLogicalBytes']) != (31, 10149805832):
        raise ValueError('Post-cleanup pointer/candidate account differs')
    maintenance, review = progress['storageMaintenance'], progress['reviews']
    if (maintenance['lastActualArchiveCycle'], maintenance['lastAuditCycle'],
        maintenance['nextDueByCycle'], review['lastCycle'], review['nextDueByCycle']) != (57, 57, 60, 57, 60):
        raise ValueError('Maintenance/advisor cadence differs')
    if review['latestAdvisorSHA256'] != sha(e / 'advisor.json'):
        raise ValueError('Latest advisor receipt differs')
    result = {**mechanics, 'schema': 'tce-stale-key-context-publication-qa-v1',
              'reportSHA256': sha(p / 'report.json'),
              'progressSHA256': sha(root / 'docs/zero-neural-progress.json'),
              'blogSHA256': sha(blog), 'figureSHA256': sha(figure),
              'advisorSHA256': sha(e / 'advisor.json'), 'nextPlanSHA256': sha(plan),
              'parentDiagonalJoins': ['PP=parent-control', 'QQ=parent-primary'],
              'testsPassed': 286, 'archivedTransportFiles': clean['files'],
              'archivedTransportBytes': clean['archivedBytes'], 'postCleanupPointerJoins': 60,
              'nextStorageCycle': 60, 'nextAdvisorCycle': 60, 'auditSourceSHA256': sha(__file__),
              'mutableAdvisorPostObservations': [{**x, 'currentSHA256': sha((e / x['path']).resolve())}
                                                  for x in advice['blogSnapshots']],
              'scope': 'Engineering source/seal/byte/typed-own-state/target/narrative consistency. '
                       'Effect sizes remain observations. Not native parity, semantic fidelity, '
                       'a serving saving, or arbitrary-prompt zero-neural certification.'}
    save(e / 'publication-qa.json', result)
    return {k: result[k] for k in ['ownedSourcePins', 'typedActs', 'explicitOutputABIs',
                                  'ownStateQueryJoins', 'observations', 'parentDiagonalJoins',
                                  'nextStorageCycle', 'nextAdvisorCycle']}


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
