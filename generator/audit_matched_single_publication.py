"""Read-only final source, evidence, synthesis and local-link checks for cycle73."""
import json
from pathlib import Path
import re
from urllib.parse import unquote
from encoder_atoms import read, sha
from publish_matched_single import audit


def execute():
    root = Path(__file__).resolve().parent.parent
    docs = root / 'docs'
    p = docs / 'results/matched-single-write'
    e = docs / 'results/matched-single-write-evidence'
    mechanics = audit(p)
    progress = read(docs / 'zero-neural-progress.json')
    advisor = read(e / 'advisor.json')
    article = docs / 'matched-single-write.md'
    plan = docs / 'plans/current-stream-mean-up.md'
    blog = root.parent / 'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'
    if progress['researchCycle'] != 73 or progress['status'] != 'active' or progress['liveJobs'] or progress['thisGoalTurn']['classification'] != 'progress':
        raise ValueError('Goal/result state differs')
    if progress['findingsLog']['sha256'] != sha(blog) or progress['findingsLog']['lastUpdatedCycle'] != 73 or progress['nextCut']['status'] != 'written-unexecuted' or progress['nextCut']['planSHA256'] != sha(plan):
        raise ValueError('Living log or next plan differs')
    if progress['matchedSingleWriteEvidence']['reportSHA256'] != sha(p / 'report.json') or progress['reviews']['latestReviewReceiptSHA256'] != sha(e / 'advisor.json'):
        raise ValueError('Current result/review identity differs')
    for path, digest in advisor['finalReviewedSynthesisSnapshots'].items():
        if sha(root / path) != digest:
            raise ValueError('Reviewed synthesis differs: ' + path)
    if advisor['reviewedResultSnapshots']['report.json'] != sha(p / 'report.json') or advisor['lastFullNewestPostSurveyCycle'] != 72 or advisor['nextFullSurveyDueCycle'] != 75 or advisor['newFullNewestPostSurvey']:
        raise ValueError('Actual review scope differs')
    figure_dir = docs / 'results/matched-single-write-figure'
    meta = read(figure_dir / 'figure.json')
    figure = figure_dir / 'matched-single-write.png'
    if meta['plotSourceSHA256'] != sha(root / 'generator/plot_matched_single.py') or meta['reportSHA256'] != sha(p / 'report.json') or meta['imageSHA256'] != sha(figure) or sha(meta['notebookCopy']) != sha(figure):
        raise ValueError('Figure source/output/notebook copy differs')
    row_meta = read(e / 'row-heterogeneity.json')
    if row_meta['sourceSHA256'] != sha(p / 'anatomy-observations.jsonl.gz') or row_meta['summarizerSHA256'] != sha(root / 'generator/summarize_matched_single_rows.py'):
        raise ValueError('Published scalar descriptions differ')
    tests = read(e / 'tests.json')
    if tests['passed'] != 337 or tests['exitCode'] != 0 or tests['sourceHashes'] != {n: sha(root / 'generator' / n) for n in read(p / 'request.json')['ownedSources']}:
        raise ValueError('Final mechanics source snapshot differs')
    body = blog.read_text().split('## Cycle 73:', 1)[1].split('## What would close the original goal', 1)[0]
    files = [(article, article.read_text()), (plan, plan.read_text()), (e / 'README.md', (e / 'README.md').read_text()), (blog, body)]
    count = 0
    for source, text in files:
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith(('https:', 'http:', '#')):
                continue
            path = unquote(target.split('#', 1)[0]).strip('<>')
            if not (source.parent / path).resolve().exists():
                raise ValueError('Broken local link: ' + str(source) + ' → ' + target)
            count += 1
    return {'schema': 'tce-matched-single-publication-qa-v1', 'status': 'passed',
        'localLinksChecked': count, 'articleSHA256': sha(article), 'nextPlanSHA256': sha(plan),
        'livingBlogSHA256': sha(blog), 'auditSourceSHA256': sha(__file__), 'mechanics': mechanics,
        'scope': 'Read-only executed-source, evidence, final synthesis, figure/scalar provenance and local-link checks. No new model run or semantic certification; strict goal remains active.'}


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2, sort_keys=True))
