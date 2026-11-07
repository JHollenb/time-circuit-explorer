"""Check publication bindings using compact metadata; no model computation."""
import json
from pathlib import Path
import re

from encoder_atoms import read, save, sha
from publish_projected_complement import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    result_root = root / 'docs/results/projected-complement'
    evidence = root / 'docs/results/projected-complement-evidence'
    mechanics = audit(result_root)
    report = read(result_root / 'report.json')
    progress_path = root / 'docs/zero-neural-progress.json'
    progress = read(progress_path)
    if (progress['researchCycle'], progress['status'], progress['liveJobs']) != (53, 'active', []):
        raise ValueError('Active goal or completed-job state differs')
    if any(progress[k]['classification'] != 'progress' for k in ['thisGoalTurn', 'previousGoalTurn']):
        raise ValueError('Scientific-turn evidence classification differs')
    recorded = progress['projectedComplementEvidence']
    if recorded['reportSHA256'] != sha(result_root / 'report.json') or recorded['responses'] != report['responses']:
        raise ValueError('Progress/report binding differs')
    for key in ['queryPanels', 'endpoints', 'upstreamBoundaryIdentity', 'scalarWorkPerActor',
                'compilerMatrixMACs', 'actorsMatrixMACs', 'totalIncrementalMatrixMACs', 'cacheLogicalBytes']:
        if recorded[key] != report[key]:
            raise ValueError('Progress field differs: ' + key)

    blog = root / progress['findingsLog']['path']
    text = blog.read_text()
    if progress['findingsLog']['lastUpdatedCycle'] != 53 or progress['findingsLog']['sha256'] != sha(blog):
        raise ValueError('Living blog snapshot differs')
    section = text.split('## Cycle 53:', 1)[1].split('## What would close the original goal', 1)[0]
    for source in ['projected_complement.py', 'projected_complement_worker.py',
                   'projected_complement_study.py', 'projected_complement_analysis.py']:
        if '../../time-circuit-explorer/generator/' + source not in text.split('---', 2)[1]:
            raise ValueError('Blog source backlink missing: ' + source)
    for path, content in [(blog, section), (root / 'docs/projected-complement-reader.md', None),
                          (root / 'docs/plans/selected-source-text-producer.md', None),
                          (evidence / 'README.md', None)]:
        for target in re.findall(r'\]\(([^)]+)\)', content if content is not None else path.read_text()):
            if target.startswith(('http:', 'https:')):
                continue
            resolved = (path.parent / target.split('#', 1)[0]).resolve()
            if resolved != (evidence / 'publication-qa.json').resolve() and not resolved.exists():
                raise ValueError('Publication backlink missing: ' + target)

    figure = root / 'docs/results/projected-complement-figure/projected-complement.png'
    copied_figure = root.parent / 'obsidian/images/zero-neural-cycle53-projected-complement.png'
    figure_receipt = read(figure.parent / 'figure.json')
    if (sha(figure) != sha(copied_figure) or sha(figure) != figure_receipt['imageSHA256']
            or sha(result_root / 'report.json') != figure_receipt['reportSHA256']
            or sha(root / 'generator/plot_projected_complement.py') != figure_receipt['plotSourceSHA256']):
        raise ValueError('Figure/source/notebook copy differs')

    advice = read(evidence / 'advisor.json')
    for name, digest in advice['reviewedSources'].items():
        source = result_root / name if name.endswith('.json') else root / 'generator' / name
        if sha(source) != digest:
            raise ValueError('Advisor source/result snapshot differs: ' + name)
    snapshots = []
    for stage, posts in [('pre-run', advice['preRunBlogSnapshots']), ('post-run', advice['blogSnapshots'])]:
        for post in posts:
            current = sha((evidence / post['path']).resolve())
            snapshots.append({**post, 'reviewStage': stage, 'currentSHA256': current,
                              'unchangedSinceReview': current == post['sha256'],
                              'scope': 'Review binds its declared snapshot; mutable-post drift does not rewrite history.'})
    next_plan = root / 'docs' / progress['nextCut']['plan']
    if 'written, unexecuted' not in next_plan.read_text() or progress['nextCut']['status'] != 'written-unexecuted':
        raise ValueError('Next-cut state differs')
    for identity in ['8ccb66606eb879c23f819b49060f23624ae7bd523ad1aface825af3119d88bfa',
                     '504bcfb16f8a8063162dca0cd318b760cbfc7a1dd3afecdda5f7e66c925d9f2d',
                     '3b07b8a59185736d31a1d2005c66296899a3335f6a0d83078ba1bf008fe727ac',
                     '5b7e018ead0011bea66318383e9e80a434dc2c7a7efeca426e2346ecb5af34c5']:
        if identity not in next_plan.read_text():
            raise ValueError('Next-plan source identity spelling differs')
    if (progress['storageMaintenance']['lastActualArchiveCycle'],
            progress['storageMaintenance']['nextDueByCycle'],
            progress['reviews']['lastCycle'], progress['reviews']['nextDueByCycle']) != (51, 54, 53, 56):
        raise ValueError('Maintenance/review cadence differs')
    cleanup = root / 'docs/results/storage-cleanup-cycle51/result.json'
    storage_audit = root / 'docs/results/storage-audit-cycle51/result.json'
    if not cleanup.exists() or not storage_audit.exists():
        raise ValueError('Prior actual storage receipts absent')
    instrument = read(evidence / 'saturn-instrument.json')
    if (instrument['runExit'], instrument['verifyExit'], instrument['records'], instrument['predicates'],
            instrument['formalReceipts'], instrument['instrumentStatus']) != (0, 0, 1024, 1, 0, 'partial-nested-metric-coverage'):
        raise ValueError('Actual evidence-instrument coverage differs')
    tests = read(evidence / 'tests.json')
    if (tests['passed'], tests['exitCode']) != (272, 0) or tests['sourceSHA256'] != sha(root / 'generator/test_projected_complement.py'):
        raise ValueError('Final test source/result binding differs')
    for name in ['compiler', 'primary']:
        execution = read(result_root / (name + '-execution.json'))
        if sum(x['nominalMACs'] for x in execution['linears'] + execution['contractions']) != execution['nominalMatrixMACs']:
            raise ValueError('Independent operation-cost sum differs')
    result = {**mechanics, 'schema': 'tce-projected-complement-publication-qa-v1',
              'reportSHA256': sha(result_root / 'report.json'), 'progressSHA256': sha(progress_path),
              'blogSHA256': sha(blog), 'figureSHA256': sha(figure),
              'advisorSHA256': sha(evidence / 'advisor.json'), 'advisorMutablePostObservations': snapshots,
              'nextPlanSHA256': sha(next_plan), 'priorStorageCleanupSHA256': sha(cleanup),
              'priorStorageAuditSHA256': sha(storage_audit), 'newArchiveOrDeletion': False,
              'nextStorageCycle': 54, 'nextAdvisorCycle': 56, 'instrumentStatus': instrument['instrumentStatus'],
              'tests': tests, 'auditSourceSHA256': sha(__file__),
              'scope': 'Publication/source/seal/byte/physical-head/typed-input/cost/narrative consistency only. No scientific effect threshold, image inheritance or arbitrary-prompt zero-neural certification.'}
    save(evidence / 'publication-qa.json', result)
    return {k: result[k] for k in ['ownedSourcePins', 'typedActors', 'queryRecords',
                                 'compilerPhysicalHeadBlocks', 'actorPhysicalHeadBlocks', 'nextStorageCycle']}


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
