"""Publication links/identities, preserved failure, storage and next-cut work."""
import ast
from pathlib import Path
import re
from encoder_atoms import read, save, sha
from publish_value_innovation import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/value-innovation'
    out = root / 'docs/results/value-innovation-evidence'
    target = out / 'publication-qa.json'
    blog = root.parent / 'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'
    names = ['README.md', 'generator/README.md', 'docs/addressed-value-innovation.md',
             'docs/plans/addressed-value-innovation.md', 'docs/plans/sparse-text-kv-innovation.md',
             'docs/results/value-innovation-evidence/README.md', 'docs/results/value-innovation-failed-v1/README.md',
             'docs/results/storage-cleanup-cycle51/README.md', 'docs/results/storage-audit-cycle51/README.md']
    docs = [root / name for name in names] + [blog]
    broken = []
    links = metadata = self_links = 0
    for doc in docs:
        for name in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            name = name.split('#')[0].strip('<>')
            if not name or '://' in name or name.startswith(('mailto:', 'data:')):
                continue
            links += 1
            file = (doc.parent / name).resolve()
            if file == target.resolve():
                self_links += 1
            elif not file.exists():
                broken.append({'source': str(doc), 'target': name})
    for line in blog.read_text().split('---', 2)[1].splitlines():
        if line.startswith('  - ../../'):
            metadata += 1
            file = (blog.parent / line[4:]).resolve()
            if file == target.resolve():
                self_links += 1
            elif not file.exists():
                broken.append({'source': str(blog), 'target': line[4:]})
    if broken:
        raise ValueError(broken)
    mechanics = audit(p)
    if mechanics != read(out / 'mechanics.json'):
        raise ValueError('Published mechanics differ')
    progress = read(root / 'docs/zero-neural-progress.json')
    if progress['researchCycle'] != 51 or progress['status'] != 'active' or progress['findingsLog']['sha256'] != sha(blog):
        raise ValueError('Living-log/progress identity differs')
    figure = root / 'docs/results/value-innovation-figure/addressed-value-innovation.png'
    copy = root.parent / 'obsidian/images/tce-value-innovation-20261106/addressed-value-innovation.png'
    f = read(figure.parent / 'figure.json')
    if sha(figure) != sha(copy) or f['figureSHA256'] != sha(figure) or f['reportSHA256'] != sha(p / 'report.json') or f['plotterSourceSHA256'] != sha(root / 'generator/plot_value_innovation.py'):
        raise ValueError('Figure identity differs')
    advisor = read(out / 'advisor.json')
    drift = []
    for item in advisor['blogSnapshots']:
        file = (out / item['path']).resolve()
        if sha(file) != item['sha256']:
            drift.append({'path': item['path'], 'reviewedSHA256': item['sha256'], 'currentSHA256': sha(file)})
    for name, digest in advisor['reviewedSources'].items():
        file = p / name if (p / name).exists() else root / 'generator' / name
        if sha(file) != digest:
            raise ValueError('Advisor execution snapshot differs')
    failed = root / 'docs/results/value-innovation-failed-v1'
    for name, digest in read(failed / 'executed-sources.json')['sources'].items():
        if sha(root / 'generator' / name) != digest:
            raise ValueError('Preserved failed execution source changed')
    failure = read(failed / 'failure.json')
    if failure['candidateSealPresent'] is not False or sha(failed / 'failed-worker.log') != failure['workerLogSHA256'] or failure['state'] != 'failed':
        raise ValueError('Preserved failed-run identity differs')
    h, q, d, c, s = 24, 1024, 3072, 128, 4
    actor = 2 * s * d * d + 2 * h * q * s * c + q * d * d
    compiler = 3 * q * d * d + 2 * 512 * d * d + 2 * h * q * 1536 * c
    cache = (2 * h * q + h * q * c) * 8
    work = progress['nextCut']['physicalWork']
    if actor != 9764339712 or compiler != 48318382080 or cache != 25559040 or work['perActorMACs'] != actor or work['twoActorMACs'] != 2 * actor or work['fullBaselineCompilerMACs'] != compiler or work['minimalFP64ComplementBytes'] != cache:
        raise ValueError('Sparse K/V plan integer work differs')
    plan = root / 'docs/plans/sparse-text-kv-innovation.md'
    if progress['nextCut']['status'] != 'written-unexecuted' or 'Status: **written, unexecuted**' not in plan.read_text() or progress['nextCut']['case'] != 'sphere-cube-plain-2027' or progress['nextCut']['support'] != [4, 11, 12, 13]:
        raise ValueError('Single next event/status differs')
    request = read(out / 'planning-request.json')
    import json
    rows = [json.loads(line) for line in (out / request['sourcePath']).resolve().read_text().splitlines() if line.strip()]
    item = next(x for x in rows if x.get('handle') == request['handle'])
    if request['researchNote'] not in item.get('research_notes', []) or item['vote count'] < request['votesAfterThisChange']:
        raise ValueError('Specific planner request update missing')
    cleanup = root / 'docs/results/storage-cleanup-cycle51/result.json'
    storage = read(root / 'docs/results/storage-audit-cycle51/result.json')
    if sha(cleanup) != sha(root / 'artifacts/storage-cleanup-cycle51/result.json') or read(cleanup)['archivedBytes'] != 800539996 or len(storage['pointerChecks']) != 52 or len(storage['remoteMetadataChecks']) != 52:
        raise ValueError('Actual storage transaction/audit differs')
    if progress['storageMaintenance']['lastActualArchiveCycle'] != 51 or progress['storageMaintenance']['nextDueByCycle'] != 54 or progress['reviews']['lastCycle'] != 51 or progress['reviews']['nextDueByCycle'] != 54:
        raise ValueError('Periodic schedules differ')
    syntax = list((root / 'generator').glob('*value_innovation*.py'))
    for file in syntax:
        ast.parse(file.read_text(), filename=str(file))
    result = {'schema': 'tce-value-innovation-publication-qa-v1', 'researchCycle': 51, 'markdownFiles': len(docs), 'localLinks': links, 'metadataLinks': metadata,
              'brokenLinks': broken, 'selfLinksClosedOnSave': self_links, 'blogSHA256': sha(blog), 'figureSHA256': sha(figure), 'notebookFigureByteIdentical': True,
              'executedSourcesUnchanged': mechanics['ownedSourcePins'], 'failedSourcesPreserved': len(read(failed / 'executed-sources.json')['sources']), 'pythonSyntaxFiles': len(syntax),
              'advisorPostDrift': drift, 'plannedPerActorMACs': actor, 'plannedBaselineCompilerMACs': compiler, 'plannedMinimalComplementBytes': cache, 'planSHA256': sha(plan),
              'plannerRequestNoteRetained': True, 'lastActualCleanupCycle': 51, 'nextStorageReviewCycle': 54, 'nextAdvisorReviewCycle': 54,
              'qaSourceSHA256': sha(Path(__file__)), 'goalStatus': 'active-unsolved',
              'scope': 'Publication/source/seal/figure/log/link/failure/storage identities and proposed integer work. Mutable post drift retained separately. No scientific effect-size, whole-image or generator certification.'}
    save(target, result)
    if not target.is_file():
        raise ValueError('QA self-link did not close')
    return result


if __name__ == '__main__':
    print(execute())
