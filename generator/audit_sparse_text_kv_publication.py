"""Final source/receipt/narrative consistency audit; no model computation."""
import hashlib
import json
from pathlib import Path
import re
from encoder_atoms import read, save, sha
from publish_sparse_text_kv import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/sparse-text-kv'
    e = root / 'docs/results/sparse-text-kv-evidence'
    mechanics = audit(p)
    report, r, progress = read(p / 'report.json'), read(p / 'request.json'), read(root / 'docs/zero-neural-progress.json')
    if progress['researchCycle'] != 52 or progress['status'] != 'active' or progress['liveJobs']:
        raise ValueError('Active/completed-job state differs')
    if progress['thisGoalTurn']['classification'] != 'progress' or progress['previousGoalTurn']['classification'] != 'no-progress':
        raise ValueError('Goal-turn evidence classification differs')
    if progress['sparseTextKVEvidence']['reportSHA256'] != sha(p / 'report.json') or report['response']['relativeRms'] != progress['sparseTextKVEvidence']['response']['relativeRms']:
        raise ValueError('Progress/report binding differs')
    for name in ['compiler', 'primary']:
        seal = read(p / (name + '-seal.json'))
        if seal[name + 'SHA256'] != sha(p / (name + '.json')) or seal['requestSHA256'] != r['requestSHA256']:
            raise ValueError('Explicit complete JSON/request-to-seal join differs')
    blog = root / progress['findingsLog']['path']
    text = blog.read_text()
    if progress['findingsLog']['lastUpdatedCycle'] != 52 or progress['findingsLog']['sha256'] != sha(blog):
        raise ValueError('Living blog snapshot differs')
    section = text.split('## Cycle 52:', 1)[1].split('## What would close the original goal', 1)[0]
    for target in re.findall(r'\]\(([^)]+)\)', section):
        if target.startswith(('http:', 'https:')):
            continue
        if not (blog.parent / target).resolve().exists():
            raise ValueError('New blog backlink missing: ' + target)
    for path in ['generator/sparse_text_kv.py', 'generator/sparse_text_kv_worker.py', 'generator/sparse_text_kv_study.py', 'generator/sparse_text_kv_analysis.py']:
        if '../../time-circuit-explorer/' + path not in text.split('---', 2)[1]:
            raise ValueError('Blog source backlink missing')
    figure = root / 'docs/results/sparse-text-kv-figure/sparse-text-kv.png'
    copy = root.parent / 'obsidian/images/zero-neural-cycle52-sparse-text-kv.png'
    figure_receipt = read(figure.parent / 'figure.json')
    if sha(figure) != sha(copy) or sha(figure) != figure_receipt['imageSHA256'] or sha(p / 'report.json') != figure_receipt['sourceReportSHA256'] or sha(root / 'generator/plot_sparse_text_kv.py') != figure_receipt['plotSourceSHA256']:
        raise ValueError('Figure/source/notebook copy differs')
    advice = read(e / 'advisor.json')
    snapshots = []
    for name, digest in advice['reviewedSources'].items():
        source = p / name if name.endswith('.json') else root / 'generator' / name
        if sha(source) != digest:
            raise ValueError('Advisor source/result snapshot differs')
    for post in advice['blogSnapshots']:
        path = (e / post['path']).resolve()
        current = sha(path)
        snapshots.append({**post, 'currentSHA256': current, 'unchangedSinceReview': current == post['sha256'],
                          'scope': 'Review binds the declared snapshot; later mutable-post drift does not rewrite it.'})
    next_plan = root / 'docs' / progress['nextCut']['plan']
    if not next_plan.exists() or 'written, unexecuted' not in next_plan.read_text() or progress['nextCut']['status'] != 'written-unexecuted':
        raise ValueError('Next-cut state differs')
    if progress['storageMaintenance']['lastActualArchiveCycle'] != 51 or progress['storageMaintenance']['nextDueByCycle'] != 54:
        raise ValueError('Storage cadence differs')
    cleanup = root / 'docs/results/storage-cleanup-cycle51/result.json'
    storage_audit = root / 'docs/results/storage-audit-cycle51/result.json'
    if not cleanup.exists() or not storage_audit.exists() or progress['reviews']['lastCycle'] != 52 or progress['reviews']['nextDueByCycle'] != 55:
        raise ValueError('Maintenance/review evidence missing')
    instrument = read(e / 'saturn-instrument.json')
    if instrument['runExit'] != 0 or instrument['verifyExit'] != 0 or instrument['records'] != 1024 or instrument['formalReceipts'] != 0:
        raise ValueError('Evidence instrument scope differs')
    tests = read(e / 'tests.json')
    if tests['passed'] != 270 or tests['exitCode'] != 0 or tests['sourceSHA256'] != sha(root / 'generator/test_sparse_text_kv.py'):
        raise ValueError('Final test source/result binding differs')
    for name in ['compiler', 'primary']:
        execution = read(p / (name + '-execution.json'))
        if sum(x['nominalMACs'] for x in execution['linears'] + execution['contractions']) != execution['nominalMACs']:
            raise ValueError('Independent operation-cost sum differs')
    result = {**mechanics, 'schema': 'tce-sparse-text-kv-publication-qa-v1', 'reportSHA256': sha(p / 'report.json'),
              'progressSHA256': sha(root / 'docs/zero-neural-progress.json'), 'blogSHA256': sha(blog), 'figureSHA256': sha(figure),
              'advisorSHA256': sha(e / 'advisor.json'), 'advisorMutablePostObservations': snapshots, 'nextPlanSHA256': sha(next_plan),
              'priorStorageCleanupSHA256': sha(cleanup), 'priorStorageAuditSHA256': sha(storage_audit),
              'newArchiveOrDeletion': False, 'nextStorageCycle': 54, 'nextAdvisorCycle': 55,
              'tests': tests, 'auditSourceSHA256': sha(__file__),
              'scope': 'Engineering publication/source/seal/byte/cost/typed-input/query/narrative consistency. Goal remains active; no effect-size, image-inheritance or general zero-neural certification.'}
    save(e / 'publication-qa.json', result)
    return {k: result[k] for k in ['ownedSourcePins', 'typedActors', 'queryRecords', 'textEndpointBF16WordsExact', 'independentNPYPayloadWitnesses', 'nextStorageCycle']}


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
