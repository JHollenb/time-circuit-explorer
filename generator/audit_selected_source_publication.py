"""Final compact publication joins; no numerical model/array work."""
import json
from pathlib import Path
import re
from encoder_atoms import read, save, sha
from audit_selected_source import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/selected-source'
    e = root / 'docs/results/selected-source-evidence'
    mechanics = audit(p)
    report = read(p / 'report.json')
    progress_path = root / 'docs/zero-neural-progress.json'
    progress = read(progress_path)
    if (progress['researchCycle'], progress['status'], progress['liveJobs']) != (54, 'active', []):
        raise ValueError('Active/completed-job state differs')
    if any(progress[k]['classification'] != 'progress' for k in ['thisGoalTurn', 'previousGoalTurn']):
        raise ValueError('Goal-turn evidence classification differs')
    recorded = progress['selectedSourceEvidence']
    if recorded['reportSHA256'] != sha(p / 'report.json'):
        raise ValueError('Progress/report binding differs')
    for name in ['responses', 'endpoints', 'queryPanels', 'sourceMetrics', 'selectedTextMetrics',
                 'upstreamBoundaryIdentity', 'work', 'sourceMatrixMACs', 'actorsMatrixMACs', 'totalIncrementalMatrixMACs']:
        if recorded[name] != report[name]:
            raise ValueError('Progress field differs: ' + name)
    blog = root / progress['findingsLog']['path']
    text = blog.read_text()
    if progress['findingsLog']['lastUpdatedCycle'] != 54 or progress['findingsLog']['sha256'] != sha(blog):
        raise ValueError('Living blog snapshot differs')
    section = text.split('## Cycle 54:', 1)[1].split('## What would close the original goal', 1)[0]
    for name in ['selected_source_v2.py', 'selected_source_worker_v2.py', 'selected_source_study_v2.py', 'selected_source_analysis.py']:
        if '../../time-circuit-explorer/generator/' + name not in text.split('---', 2)[1]:
            raise ValueError('Blog executed-source backlink absent')
    docs = ['docs/selected-source-text-producer.md', 'docs/plans/source-complement-transfer.md',
            'docs/results/selected-source-evidence/README.md', 'docs/results/storage-cleanup-cycle54/README.md',
            'docs/results/storage-audit-cycle54/README.md']
    for path, content in [(blog, section)] + [(root / name, None) for name in docs]:
        for target in re.findall(r'\]\(([^)]+)\)', content if content is not None else path.read_text()):
            if target.startswith(('https:', 'http:')):
                continue
            resolved = (path.parent / target.split('#', 1)[0]).resolve()
            if resolved != (e / 'publication-qa.json').resolve() and not resolved.exists():
                raise ValueError('New publication backlink absent: ' + target)
    figure = root / 'docs/results/selected-source-figure/selected-source.png'
    figure_receipt = read(figure.parent / 'figure.json')
    notebook_figure = root.parent / 'obsidian/images/zero-neural-cycle54-selected-source.png'
    if (sha(figure) != sha(notebook_figure) or sha(figure) != figure_receipt['imageSHA256']
            or sha(p / 'report.json') != figure_receipt['reportSHA256']
            or sha(root / 'generator/plot_selected_source.py') != figure_receipt['plotSourceSHA256']):
        raise ValueError('Figure source/copy differs')
    advice = read(e / 'advisor.json')
    for name, digest in advice['reviewedSources'].items():
        source = p / name if name.endswith('.json') else root / 'generator' / name
        if sha(source) != digest:
            raise ValueError('Advisor source/result snapshot differs')
    snapshots = []
    for post in advice['blogSnapshots']:
        current = sha((e / post['path']).resolve())
        snapshots.append({**post, 'currentSHA256': current, 'unchangedSinceReview': current == post['sha256'],
                          'scope': 'Review binds the declared snapshot; mutable-post updates do not rewrite it.'})
    plan = root / 'docs' / progress['nextCut']['plan']
    if progress['nextCut']['status'] != 'written-unexecuted' or 'written, unexecuted' not in plan.read_text():
        raise ValueError('Next-cut status differs')
    cut = progress['nextCut']
    if sum(cut[k] for k in ['sharedSelectedQKVMACs', 'twoAttentionMACs', 'twoOutputMLPMACs', 'twoReaderMACs']) != cut['totalNewMatrixMACs']:
        raise ValueError('Next-cut operation sum differs')
    if (progress['storageMaintenance']['lastActualArchiveCycle'], progress['storageMaintenance']['lastAuditCycle'],
            progress['storageMaintenance']['nextDueByCycle'], progress['reviews']['lastCycle'], progress['reviews']['nextDueByCycle']) != (54, 54, 57, 54, 57):
        raise ValueError('Actual maintenance/review cadence differs')
    cleanup = root / 'docs/results/storage-cleanup-cycle54/result.json'
    storage_audit = root / 'docs/results/storage-audit-cycle54/result.json'
    archived, storage = read(cleanup), read(storage_audit)
    if archived['files'] != 4 or archived['archivedBytes'] != 617842388 or len(storage['pointerChecks']) != 56:
        raise ValueError('Actual storage receipt differs')
    for name in ['storage-cleanup-cycle54', 'storage-audit-cycle54']:
        if sha(root / 'artifacts' / name / 'result.json') != sha(root / 'docs/results' / name / 'result.json'):
            raise ValueError('Public storage copy differs')
    tests = read(e / 'tests.json')
    if (tests['passed'], tests['exitCode']) != (276, 0) or tests['sourceSHA256'] != sha(root / 'generator/test_selected_source.py'):
        raise ValueError('Test source/result differs')
    instrument = read(e / 'saturn-instrument.json')
    if (instrument['runExit'], instrument['verifyExit'], instrument['records'], instrument['predicates'], instrument['formalReceipts']) != (0, 0, 1024, 1, 0):
        raise ValueError('Actual instrument scope differs')
    diagnostic = read(e / 'publication-diagnostics.json')
    if diagnostic['verifierSourceSHA256'] != sha(root / 'generator/publish_selected_source.py') or diagnostic['correctedAuditSourceSHA256'] != sha(root / 'generator/audit_selected_source.py'):
        raise ValueError('Publication correction source differs')
    unexecuted = read(e / 'unexecuted-v1.json')
    if unexecuted['status'] != 'staged-unsubmitted' or unexecuted['jobFilePresent'] is not False or len(unexecuted['actualMetadataJoins']) != 6:
        raise ValueError('Unexecuted-stage preservation differs')
    result = {**mechanics, 'schema': 'tce-selected-source-publication-qa-v1',
              'reportSHA256': sha(p / 'report.json'), 'progressSHA256': sha(progress_path), 'blogSHA256': sha(blog),
              'figureSHA256': sha(figure), 'advisorSHA256': sha(e / 'advisor.json'), 'advisorMutablePostObservations': snapshots,
              'nextPlanSHA256': sha(plan), 'cleanupSHA256': sha(cleanup), 'storageAuditSHA256': sha(storage_audit),
              'publicationDiagnosticSHA256': sha(e / 'publication-diagnostics.json'), 'tests': tests,
              'nextStorageCycle': 57, 'nextAdvisorCycle': 57, 'auditSourceSHA256': sha(__file__),
              'scope': 'Engineering source/seal/complete-byte/typed-input/cost/narrative consistency. Goal remains active; no effect-size, image inheritance or arbitrary-prompt zero-neural certification.'}
    save(e / 'publication-qa.json', result)
    return {k: result[k] for k in ['ownedSourcePins', 'sourcePageMembershipJoins', 'typedSourceActs', 'typedSpliceActs', 'typedReaderActs', 'queryRecords', 'nextStorageCycle']}


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
