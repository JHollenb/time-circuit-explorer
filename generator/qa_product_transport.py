"""Publication identities, log links, bounded storage account and next-cut work."""
import ast
from pathlib import Path
import re
from encoder_atoms import read, save, sha
from publish_product_transport import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/product-transport'
    out = root / 'docs/results/product-transport-evidence'
    target = out / 'publication-qa.json'
    blog = root.parent / 'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'
    docs = [root / n for n in ['README.md', 'generator/README.md', 'docs/recipient-conditioned-product.md', 'docs/plans/receiver-conditioned-product.md', 'docs/plans/shared-normalization-innovation.md', 'docs/results/product-transport-evidence/README.md']] + [blog]
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
    if read(p / 'primary.json')['fields']['residualCommit']['payloadSHA256'] != read(p / 'targets.json')['residualR1B']['payloadSHA256']:
        raise ValueError('Native residual payload bits differ')
    mechanics['nativeResidualPayloadExact'] = True
    if mechanics != read(out / 'mechanics.json'):
        raise ValueError('Published mechanics differ')
    progress = read(root / 'docs/zero-neural-progress.json')
    if progress['researchCycle'] != 49 or progress['status'] != 'active' or progress['findingsLog']['sha256'] != sha(blog):
        raise ValueError('Living-log/progress identity differs')
    figure = root / 'docs/results/product-transport-figure/recipient-product-transport.png'
    copy = root.parent / 'obsidian/images/tce-product-transport-20261106/recipient-product-transport.png'
    f = read(figure.parent / 'figure.json')
    if sha(figure) != sha(copy) or f['figureSHA256'] != sha(figure) or f['plotterSourceSHA256'] != sha(root / 'generator/plot_product_transport.py') or f['reportSHA256'] != sha(p / 'report.json'):
        raise ValueError('Figure identity differs')
    advisor = read(out / 'advisor.json')
    drift = []
    for item in advisor['blogSnapshots']:
        file = (out / item['path']).resolve()
        if sha(file) != item['sha256']:
            drift.append({'path': item['path'], 'reviewedSHA256': item['sha256'], 'currentSHA256': sha(file)})
    for name, digest in advisor['reviewedSources'].items():
        source = p / name if (p / name).exists() else root / 'generator' / name
        if sha(source) != digest:
            raise ValueError('Advisor execution snapshot differs')
    shared = 1024 * 3072 * 18432
    clock = 3072 * 18432
    downs = 2 * 1024 * 9216 * 3072
    total = shared + clock + downs
    parameters = 2 * (3072 * 18432 + 3072 * 9216)
    work = progress['nextCut']['physicalWork']
    if total != 116020740096 or parameters != 169869312 or work['nominalAffineMACs'] != total or work['selectedBF16ParameterBytes'] != parameters or work['affineCalls'] != 4:
        raise ValueError('Shared innovation plan geometry differs')
    if progress['storageMaintenance']['lastActualArchiveCycle'] != 48 or progress['storageMaintenance']['nextDueByCycle'] != 51 or progress['reviews']['lastCycle'] != 49 or progress['reviews']['nextDueByCycle'] != 52:
        raise ValueError('Periodic schedules differ')
    cleanup = root / 'docs/results/storage-cleanup-cycle48/result.json'
    if sha(cleanup) != sha(root / 'artifacts/storage-cleanup-cycle48/result.json') or read(cleanup)['archivedBytes'] != 600212652:
        raise ValueError('Last actual cleanup receipt differs')
    syntax = list((root / 'generator').glob('*product_transport*.py'))
    for file in syntax:
        ast.parse(file.read_text(), filename=str(file))
    result = {'schema': 'tce-product-transport-publication-qa-v1', 'researchCycle': 49, 'markdownFiles': len(docs), 'localLinks': links, 'metadataLinks': metadata, 'brokenLinks': broken, 'selfLinksClosedOnSave': self_links, 'blogSHA256': sha(blog), 'figureSHA256': sha(figure), 'notebookFigureByteIdentical': True, 'executedSourcesUnchanged': mechanics['ownedSourcePins'], 'nativeResidualPayloadExact': True, 'pythonSyntaxFiles': len(syntax), 'advisorPostDrift': drift, 'plannedAffineMACs': total, 'plannedSelectedParameterBytes': parameters, 'lastActualCleanupCycle': 48, 'nextStorageReviewCycle': 51, 'nextAdvisorReviewCycle': 52, 'goalStatus': 'active-unsolved', 'scope': 'Publication links, frozen source/seal/full-byte/figure/log identities, previous cleanup receipt and proposed integer work. Self-link closes on exclusive save and final existence check. No scientific quality, whole-future or generator certification.'}
    save(target, result)
    if not target.is_file():
        raise ValueError('QA self-link did not close')
    return result


if __name__ == '__main__':
    print(execute())
