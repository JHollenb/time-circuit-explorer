"""Check publication identities and the next plan's integer work account."""
import ast
import json
from pathlib import Path
import re
from encoder_atoms import read, save, sha
from publish_normalization_innovation import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    p = root / 'docs/results/normalization-innovation'
    out = root / 'docs/results/normalization-innovation-evidence'
    target = out / 'publication-qa.json'
    blog = root.parent / 'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'
    docs = [root / n for n in [
        'README.md', 'generator/README.md',
        'docs/shared-normalization-innovation.md',
        'docs/plans/shared-normalization-innovation.md',
        'docs/plans/addressed-value-innovation.md',
        'docs/results/normalization-innovation-evidence/README.md',
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
    mechanics = audit(p)
    if mechanics != read(out / 'mechanics.json'):
        raise ValueError('Published mechanics differ')
    if mechanics['nativeResidualPrefixesExact'] != 4:
        raise ValueError('Native residual prefix account differs')
    progress = read(root / 'docs/zero-neural-progress.json')
    if progress['researchCycle'] != 50 or progress['status'] != 'active' or progress['findingsLog']['sha256'] != sha(blog):
        raise ValueError('Living-log/progress identity differs')
    figure = root / 'docs/results/normalization-innovation-figure/shared-normalization-innovation.png'
    copy = root.parent / 'obsidian/images/tce-normalization-innovation-20261106/shared-normalization-innovation.png'
    f = read(figure.parent / 'figure.json')
    if sha(figure) != sha(copy) or f['figureSHA256'] != sha(figure) or f['plotterSourceSHA256'] != sha(root / 'generator/plot_normalization_innovation.py') or f['reportSHA256'] != sha(p / 'report.json'):
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
    h, q, k, c, d, fdim, m = 24, 1024, 1536, 128, 3072, 18432, 9216
    components = {
        'freshPrefixAffineMACs': 51136954368,
        'QKScoreMACs': h * q * k * c,
        'headBlockWoMACs': h * c * d,
        'directionFusedMACs': h * d * fdim,
        'assembleCarrierMACs': q * h * d,
        'assembleJMACs': q * h * fdim,
        'oneDownMACs': q * m * d,
    }
    incremental = sum(v for key, v in components.items() if key not in ['freshPrefixAffineMACs', 'QKScoreMACs'])
    candidate = sum(components.values())
    native_attention = 2 * h * q * k * c
    native_wo = q * d * d
    native_mlp = q * d * fdim + q * m * d
    validation = 2 * native_attention + 2 * native_wo + native_mlp
    parameter_bytes = 142344960 + 2 * d * d + 2 * (d * fdim + d * m)
    work = progress['nextCut']['physicalWork']
    expected = {
        **components, 'innovationAndDownMACs': incremental,
        'candidateMACs': candidate, 'normalValidationMACs': validation,
        'normalTotalMACs': candidate + validation,
        'conditionalAdditionalAMLPReferenceMACs': native_mlp,
        'selectedBF16ParameterBytes': parameter_bytes,
    }
    if any(work[key] != value for key, value in expected.items()):
        raise ValueError('Addressed-value plan geometry differs')
    if incremental != 30887903232 or candidate != 86856695808 or validation != 125627793408 or parameter_bytes != 331088640:
        raise ValueError('Independent integer work differs')
    plan = root / 'docs/plans/addressed-value-innovation.md'
    if progress['nextCut']['status'] != 'written-unexecuted' or 'Status: **written, unexecuted**' not in plan.read_text() or progress['nextCut']['event']['imageRows'] != [264, 792]:
        raise ValueError('Next event/status differs')
    request = read(out / 'planning-request.json')
    request_file = (out / request['sourcePath']).resolve()
    matching = [json.loads(line) for line in request_file.read_text().splitlines() if line.strip()]
    matching = [item for item in matching if item.get('handle') == request['handle']]
    if len(matching) != 1 or request['researchNote'] not in matching[0].get('research_notes', []) or matching[0]['vote count'] < request['votesAfterThisChange']:
        raise ValueError('Specific planner-request update is missing')
    if progress['storageMaintenance']['lastActualArchiveCycle'] != 48 or progress['storageMaintenance']['nextDueByCycle'] != 51 or progress['reviews']['lastCycle'] != 50 or progress['reviews']['nextDueByCycle'] != 53:
        raise ValueError('Periodic schedules differ')
    cleanup = root / 'docs/results/storage-cleanup-cycle48/result.json'
    if sha(cleanup) != sha(root / 'artifacts/storage-cleanup-cycle48/result.json') or read(cleanup)['archivedBytes'] != 600212652:
        raise ValueError('Last actual cleanup receipt differs')
    syntax = list((root / 'generator').glob('*normalization_innovation*.py'))
    for file in syntax:
        ast.parse(file.read_text(), filename=str(file))
    result = {
        'schema': 'tce-normalization-innovation-publication-qa-v1',
        'researchCycle': 50, 'markdownFiles': len(docs), 'localLinks': links,
        'metadataLinks': metadata, 'brokenLinks': broken,
        'selfLinksClosedOnSave': self_links, 'blogSHA256': sha(blog),
        'figureSHA256': sha(figure), 'notebookFigureByteIdentical': True,
        'executedSourcesUnchanged': mechanics['ownedSourcePins'],
        'nativeResidualPrefixesExact': 4, 'pythonSyntaxFiles': len(syntax),
        'advisorPostDrift': drift, 'proposedWork': expected,
        'planSHA256': sha(plan), 'plannerRequestNoteRetained': True,
        'lastActualCleanupCycle': 48, 'nextStorageReviewCycle': 51,
        'nextAdvisorReviewCycle': 53, 'goalStatus': 'active-unsolved',
        'qaSourceSHA256': sha(Path(__file__)),
        'scope': 'Publication links, source/seal/figure/log identities, planner-note retention, previous cleanup receipt and proposed integer work only. Scientific effect sizes and whole-future/generator certification remain separate. Mutable post drift is recorded without rewriting reviewed snapshots.',
    }
    save(target, result)
    if not target.is_file():
        raise ValueError('QA self-link did not close')
    return result


if __name__ == '__main__':
    print(execute())
