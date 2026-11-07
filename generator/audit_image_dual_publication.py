"""Publication and historical target joins; no model/activation execution."""
import json
from pathlib import Path
import re
from encoder_atoms import read, save, sha
from image_dual_analysis import NAMES
from publish_image_dual import audit


def execute():
    root = Path(__file__).resolve().parent.parent;p = root / 'docs/results/image-dual';e = root / 'docs/results/image-dual-evidence'
    mechanics = audit(p)
    report, progress, summary, advice, pre, tests, instrument = [read(x) for x in [p / 'report.json',
        root / 'docs/zero-neural-progress.json', e / 'scalar-summary.json', e / 'advisor.json',
        e / 'pre-stage-advisor.json', e / 'tests.json', e / 'saturn-instrument.json']]
    if (progress['researchCycle'], progress['status'], progress['liveJobs']) != (58, 'active', []):
        raise ValueError('Active goal/completed job state differs')
    if progress['lastCompletedResearchTurn']['commit'] != '9e471ed8c6c717a3fd45836b6a4c23036eadedd6' or any(progress[n]['classification'] != 'progress' for n in ['thisGoalTurn', 'previousGoalTurn']):
        raise ValueError('Current/previous committed progress differs')
    recorded = progress['imageDualEvidence']
    if recorded['reportSHA256'] != sha(p / 'report.json') or recorded['scalarSummarySHA256'] != sha(e / 'scalar-summary.json'):
        raise ValueError('Progress evidence snapshots differ')
    for name in ['scores', 'diagnostics', 'headErrorDecomposition', 'headErrorCrossEnergy', 'workByPhase',
                 'totalNewMatrixMACs', 'distinctBF16ParameterBytes', 'processBF16ParameterBytes',
                 'observations', 'primaryCandidate', 'decompositionLaw']:
        if recorded[name] != report[name]:
            raise ValueError('Progress/result field differs: ' + name)
    for name in ['queryHeadResponseTrends', 'headsBetterThanStale', 'newArrays', 'newArrayBytes', 'dualFP64PayloadBytes', 'dualNPYBytes']:
        if recorded[name] != summary[name]:
            raise ValueError('Progress/scalar field differs: ' + name)
    request, inputs, targets = [read(p / n) for n in ['request.json', 'inputs.json', 'targets.json']]
    for name in ['image-dual-score-producer.md', 'image-dual-execution-addendum.md']:
        if sha(p / name) != request['sourceHashes'][name] or sha(root / 'docs/plans' / name) != request['sourceHashes'][name]:
            raise ValueError('Frozen original plan/addendum differs')
    parent = root / 'docs/results/stale-key-context'
    for name, value in inputs['latestParentRecords'].items():
        if sha(parent / name) != value['sha256']:
            raise ValueError('Latest parent metadata snapshot differs: ' + name)
    diagonals = read(parent / 'targets.json')['diagonals']
    if any(targets[n] != diagonals[n] for n in ['PP', 'QQ']):
        raise ValueError('PP/QQ historical target labels differ')
    stale = read(parent / 'primary.json');fields = {**stale['source']['fields'], **stale['nextQuery']['fields']}
    if targets['stale'] != {n: fields[n] for n in NAMES}:
        raise ValueError('Stale target origin differs')
    old_inputs = read(root / 'docs/results/propagated-context/inputs.json')
    if inputs['sourceCuts'] != old_inputs['sourceCuts'] or inputs['textKeySlice']['source'] != inputs['banks']['Q']['K']:
        raise ValueError('Native source/text-prefix origin differs')
    inventory = inputs['positionByteInventory']['axes']
    groups = read(p / 'compiler.json')['groupInventory']
    if [x['imageDistinctByteGroups'] for x in inventory] != [1, 32, 32, 1] or [x['textDistinctByteGroups'] for x in inventory] != [1, 1, 1, 512] or groups['constantAxes'] != [0, 3] or groups['varyingAxes'] != [1, 2]:
        raise ValueError('Stored-byte/native THWL correction differs')
    if (summary['newArrays'], summary['newArrayBytes'], summary['dualFP64PayloadBytes'], summary['dualNPYBytes']) != (46, 238519040, 153354240, 153354368):
        raise ValueError('Complete new array/header accounting differs')
    if pre['reviewedSources'] != advice['reviewedSources']:
        raise ValueError('Pre-stage/actual advisor source snapshots differ')
    for name, digest in advice['reviewedSources'].items():
        if sha(root / 'generator' / name) != digest or request['sourceHashes'][name] != digest:
            raise ValueError('Advisor/executed source differs: ' + name)
    for name, digest in advice['reviewedResults'].items():
        if sha(p / name) != digest:
            raise ValueError('Advisor actual result snapshot differs: ' + name)
    if tests['sourceSHA256'] != sha(root / tests['source']) or (tests['passed'], tests['seconds'], tests['exitCode']) != (290, 19.78, 0):
        raise ValueError('Actual test source/receipt differs')
    if (instrument['records'], instrument['predicates'], instrument['runExit'], instrument['verifyExit'], instrument['formalReceipts'], instrument['boundXrefAddresses']) != (444, 20, 0, 0, 0, 0):
        raise ValueError('Actual MDB instrument scope differs')
    blog = root / progress['findingsLog']['path'];text = blog.read_text()
    if progress['findingsLog']['lastUpdatedCycle'] != 58 or progress['findingsLog']['sha256'] != sha(blog):
        raise ValueError('Living blog snapshot differs')
    section = text.split('## Cycle 58:', 1)[1].split('## What would close the original goal', 1)[0]
    front = text.split('---', 2)[1]
    for name in advice['reviewedSources']:
        if '../../time-circuit-explorer/generator/' + name not in front:
            raise ValueError('Executed-source blog backlink absent')
    plan = root / 'docs' / progress['nextCut']['plan']
    if progress['nextCut']['status'] != 'written-unexecuted' or 'written, unexecuted' not in plan.read_text() or progress['nextCut']['totalNewMatrixMACs'] != 1346371584 + 75497472 or progress['nextCut']['coefficientFP64PayloadBytes'] != 24 * 3072 * 8:
        raise ValueError('Prospective diagonal norm scope/counts differ')
    for path, contents in [(blog, section), (root / 'docs/image-dual-score-response.md', None), (plan, None), (e / 'README.md', None)]:
        for link in re.findall(r'\]\(([^)]+)\)', contents if contents is not None else path.read_text()):
            if link.startswith(('http:', 'https:')):
                continue
            target = (path.parent / link.split('#', 1)[0]).resolve()
            if target != (e / 'publication-qa.json').resolve() and not target.exists():
                raise ValueError('Publication backlink absent: ' + link)
    figure = root / 'docs/results/image-dual-figure/image-dual.png';frec = read(figure.parent / 'figure.json')
    if sha(figure) != sha(root.parent / 'obsidian/images/zero-neural-cycle58-image-dual.png') or sha(figure) != frec['imageSHA256'] or frec['reportSHA256'] != sha(p / 'report.json') or frec['plotSourceSHA256'] != sha(root / 'generator/plot_image_dual.py'):
        raise ValueError('Figure source/data/notebook copy differs')
    maintenance, review = progress['storageMaintenance'], progress['reviews']
    if (maintenance['lastActualArchiveCycle'], maintenance['lastAuditCycle'], maintenance['nextDueByCycle'], review['lastCycle'], review['nextDueByCycle']) != (57, 57, 60, 58, 60) or review['latestAdvisorSHA256'] != sha(e / 'advisor.json'):
        raise ValueError('Storage/advisor actual cadence differs')
    result = {**mechanics, 'schema': 'tce-image-dual-publication-qa-v1',
        'reportSHA256': sha(p / 'report.json'), 'progressSHA256': sha(root / 'docs/zero-neural-progress.json'),
        'blogSHA256': sha(blog), 'figureSHA256': sha(figure), 'advisorSHA256': sha(e / 'advisor.json'),
        'nextPlanSHA256': sha(plan), 'auditSourceSHA256': sha(__file__), 'testsPassed': 290,
        'historicalTargetJoins': ['PP/QQ=cycle57 bound parent diagonals', 'stale=cycle57 prespecified PQ'],
        'positionByteGroups': [1, 32, 32, 1], 'nextStorageCycle': 60, 'nextAdvisorCycle': 60,
        'mutableAdvisorPostObservations': [{**x, 'currentSHA256': sha((e / x['path']).resolve())} for x in advice['blogSnapshots']],
        'scope': 'Source/seal/bytes/typed-own-state/target/narrative consistency only. No effect-size gate, isolated image-numerator attribution, semantic/generalization/speedup or arbitrary-prompt zero-neural certification.'}
    save(e / 'publication-qa.json', result)
    return {n: result[n] for n in ['ownedSourcePins', 'typedActs', 'explicitOutputABIs', 'ownHeadStateQueryJoins', 'observations', 'nextStorageCycle', 'nextAdvisorCycle']}


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
