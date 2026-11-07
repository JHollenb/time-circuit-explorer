"""Check publication identities, links, storage receipts and proposed work."""
import ast
from pathlib import Path
import re
from encoder_atoms import read, save, sha
from publish_first_consumer import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    published = root / 'docs/results/first-consumer'
    out = root / 'docs/results/first-consumer-evidence'
    blog = root.parent / 'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'
    target = out / 'publication-qa.json'
    docs = [root / name for name in [
        'README.md', 'generator/README.md', 'docs/first-image-consumer.md',
        'docs/plans/first-image-consumer.md', 'docs/plans/receiver-conditioned-product.md',
        'docs/results/first-consumer-evidence/README.md',
        'docs/results/storage-cleanup-cycle48/README.md',
        'docs/results/storage-audit-cycle48/README.md',
    ]] + [blog]
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
    mechanics = audit(published)
    if mechanics != read(out / 'mechanics.json'):
        raise ValueError('Published mechanics changed')
    progress = read(root / 'docs/zero-neural-progress.json')
    if progress['researchCycle'] != 48 or progress['status'] != 'active' or progress['findingsLog']['sha256'] != sha(blog):
        raise ValueError('Living-log/progress identity differs')
    figure = root / 'docs/results/first-consumer-figure/first-image-consumer.png'
    figure_copy = root.parent / 'obsidian/images/tce-first-consumer-20261106/first-image-consumer.png'
    receipt = read(figure.parent / 'figure.json')
    if sha(figure_copy) != sha(figure) or receipt['figureSHA256'] != sha(figure) or receipt['reportSHA256'] != sha(published / 'report.json') or receipt['plotterSourceSHA256'] != sha(root / 'generator/plot_first_consumer.py'):
        raise ValueError('Figure identity differs')
    advisor = read(out / 'advisor.json')
    drift = []
    for item in advisor['blogSnapshots']:
        file = (out / item['path']).resolve()
        if sha(file) != item['sha256']:
            drift.append({'path': item['path'], 'reviewedSHA256': item['sha256'], 'currentSHA256': sha(file)})
    for name, digest in advisor['reviewedSources'].items():
        file = published / name if (published / name).exists() else root / 'generator' / name
        if sha(file) != digest:
            raise ValueError('Advisor-reviewed execution source changed')
    macs = 1024 * 9216 * 3072
    parameter_bytes = 2 * 3072 * 9216
    if macs != 28991029248 or parameter_bytes != 56623104 or progress['nextCut']['physicalWork']['nominalAffineMACs'] != macs or progress['nextCut']['physicalWork']['selectedBF16ParameterBytes'] != parameter_bytes:
        raise ValueError('Proposed one-affine geometry differs')
    cleanup = read(root / 'docs/results/storage-cleanup-cycle48/result.json')
    storage_audit = read(root / 'docs/results/storage-audit-cycle48/result.json')
    for name in ['plan.json', 'result.json', 'events.jsonl']:
        if sha(root / 'docs/results/storage-cleanup-cycle48' / name) != sha(root / 'artifacts/storage-cleanup-cycle48' / name):
            raise ValueError('Cleanup publication differs')
    if sha(root / 'docs/results/storage-audit-cycle48/result.json') != sha(root / 'artifacts/storage-audit-cycle48/result.json') or storage_audit['auditorSHA256'] != sha(root / 'generator/storage_transports.py'):
        raise ValueError('Storage audit source/receipt differs')
    if cleanup['files'] != 4 or cleanup['archivedBytes'] != 600212652 or len(cleanup['rows']) != 4 or len(storage_audit['pointerChecks']) != 48 or len(storage_audit['remoteMetadataChecks']) != 48:
        raise ValueError('Storage counts differ')
    for item in cleanup['rows']:
        source = Path(item['path'])
        if source.exists() or read(str(source) + '.cold.json') != item or not Path(item['extractedRoot']).is_dir() or sha(item['custodyPath']) != item['custodySHA256']:
            raise ValueError('New transport restoration/custody closure differs')
        if item['remoteVerification'] != {'path': item['remote'], 'bytes': item['bytes'], 'sha256': item['sha256'], 'verified': True}:
            raise ValueError('Full cold-byte verification missing')
    for pointer, remote in zip(storage_audit['pointerChecks'], storage_audit['remoteMetadataChecks'], strict=True):
        item = read(pointer['pointer'])
        if item['sha256'] != pointer['sha256'] or sha(item['custodyPath']) != pointer['custodySHA256'] or item['remote'] != pointer['remote'] or remote != {'remote': pointer['remote'], 'bytes': pointer['bytes'], 'regular': True}:
            raise ValueError('Historical pointer/metadata receipt differs')
    if progress['storageMaintenance']['nextDueByCycle'] != 51 or progress['storageMaintenance']['lastActualArchiveCycle'] != 48:
        raise ValueError('Storage schedule differs')
    syntax = list((root / 'generator').glob('*first_consumer*.py')) + [root / 'generator/storage_transports.py']
    for file in syntax:
        ast.parse(file.read_text(), filename=str(file))
    result = {
        'schema': 'tce-first-consumer-publication-qa-v1', 'researchCycle': 48,
        'markdownFiles': len(docs), 'localLinks': links, 'metadataLinks': metadata,
        'brokenLinks': broken, 'selfLinksClosedOnSave': self_links,
        'blogSHA256': sha(blog), 'figureSHA256': sha(figure), 'notebookFigureByteIdentical': True,
        'executedSourcesUnchanged': mechanics['ownedSourcePins'], 'pythonSyntaxFiles': len(syntax),
        'advisorPostDrift': drift, 'plannedAffineMACs': macs, 'plannedSelectedParameterBytes': parameter_bytes,
        'cleanupFiles': 4, 'cleanupBytes': 600212652, 'postAuditPointerJoins': 48,
        'nextStorageReviewCycle': 51, 'goalStatus': 'active-unsolved',
        'scope': 'Publication links, immutable execution/figure/log identities, retained storage receipt joins and proposed integer work. Historical remote checks are not repeated full-byte hashes. This self-link closes on exclusive save and final existence check. No scientific quality or generator certification.',
    }
    save(target, result)
    if not target.is_file():
        raise ValueError('QA self-link did not close')
    return result


if __name__ == '__main__':
    print(execute())
