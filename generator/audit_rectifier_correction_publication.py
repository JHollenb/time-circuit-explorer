"""Read-only joins for the completed assay, reviews, living log and publication."""
import json
from pathlib import Path
import re
from urllib.parse import unquote

from encoder_atoms import read, sha
from publish_rectifier_correction import audit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def execute():
    root = Path(__file__).resolve().parent.parent
    docs = root / 'docs'
    result = docs / 'results/rectifier-correction'
    evidence = docs / 'results/rectifier-correction-evidence'
    mechanics = audit(result)
    progress = read(docs / 'zero-neural-progress.json')
    advisor = read(evidence / 'advisor.json')
    report = read(result / 'report.json')
    request = read(result / 'request.json')
    article = docs / 'combined-rectifier.md'
    plan = docs / 'plans/rectifier-sign-support.md'
    blog = root.parent / 'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'
    body = blog.read_text()
    require(progress['researchCycle'] == 80 and progress['status'] == 'active'
            and not progress['liveJobs'] and progress['thisGoalTurn']['classification'] == 'progress',
            'Current goal differs')
    require(progress['findingsLog']['lastUpdatedCycle'] == 80
            and progress['findingsLog']['sha256'] == sha(blog)
            and progress['nextCut']['status'] == 'written-unexecuted'
            and progress['nextCut']['planSHA256'] == sha(plan)
            and progress['nextCut']['requiresNewestPostAndStorageReviewsBeforeStage']
            and progress['nextCut']['prospectiveLearnedMatrixMACs'] == 0,
            'Living log or unexecuted next cut differs')
    for name, fingerprint in advisor['finalReviewedSynthesisSnapshots'].items():
        require(sha(root / name) == fingerprint, 'Final reviewed synthesis differs: ' + name)
    require(len(advisor['finalReviewedSynthesisSnapshots']) == 4, 'Four final syntheses required')
    for name, fingerprint in advisor['reviewedResultSnapshots'].items():
        require(sha(result / name) == fingerprint, 'Reviewed actual result differs: ' + name)
    require(not advisor['newFullNewestPostSurvey']
            and advisor['lastFullNewestPostSurveyCycle'] == 78
            and advisor['nextFullSurveyDueCycle'] == 81
            and advisor['fullSurveySHA256'] == sha(docs / 'results/research-review-cycle78/advisor-survey.json'),
            'Periodic review scope differs')
    source = read(result / 'inputs.json')
    for name, fingerprint in source['preStageReviewSnapshots'].items():
        require(sha(root / name) == fingerprint, 'Pre-stage review dependency differs: ' + name)
    tests = read(evidence / 'tests.json')
    pre = read(evidence / 'advisor-prestage.json')
    require(pre == advisor['preStageReview'] and pre['planSHA256'] == sha(result / 'frozen-plan.md'),
            'Pre-stage source/plan review differs')
    require(tests['passed'] == 347 and tests['exitCode'] == 0
            and tests['sourceHashes'] == {name: request['sourceHashes'][name] for name in request['ownedSources']},
            'Tests and executed sources differ')
    for name, fingerprint in pre['reviewedSources'].items():
        require(sha(root / 'generator' / name) == fingerprint, 'Pre-stage source review differs: ' + name)
    current = progress['rectifierCorrectionEvidence']
    for metric in ['fullResidualOverTotalRMS', 'centeredResidualOverCenteredTotalRMS', 'rowsVs']:
        require(current[metric] == {name: values[metric] for name, values in report['panels'].items()},
                'Progress numerical account differs: ' + metric)
    require(current['work']['matrixMACs'] == report['totalMatrixMACs'] == 43486543872,
            'Paid learned work differs')
    concentration = read(evidence / 'concentration.json')
    rows = read(evidence / 'row-heterogeneity.json')
    require(advisor['concentrationReceiptSHA256'] == sha(evidence / 'concentration.json')
            and concentration['summarizerSHA256'] == sha(root / 'generator/summarize_rectifier_concentration.py')
            and rows['summarizerSHA256'] == sha(root / 'generator/summarize_rectifier_correction.py')
            and concentration['sourceSHA256'] == rows['sourceSHA256']
            and current['concentration'] == concentration,
            'Concentration source or retained countertrend differs')
    require(concentration['otherRowsNetEnergyGainVsQuadratic'] < 0
            and concentration['historicalTwoRowsShareOfNetEnergyGain'] > 1
            and concentration['worseVsZeroRows'] == [34, 39, 40, 41, 43, 44],
            'Adverse row findings differ')
    figure_root = docs / 'results/rectifier-correction-figure'
    figure = read(figure_root / 'figure.json')
    render = read(evidence / 'render-qa.json')
    require(figure['plotSourceSHA256'] == sha(root / 'generator/plot_rectifier_correction.py')
            and figure['sourceReportSHA256'] == sha(result / 'report.json')
            and figure['imageSHA256'] == sha(figure_root / 'rectifier-correction.png')
            and figure['imageSHA256'] == sha(Path(figure['notebookCopy']))
            and figure['imageSHA256'] == render['imageSHA256'] == advisor['figureSHA256'],
            'Inspected figure, source or notebook copy differs')
    arrays = [*read(result / 'candidate.json')['fields'].values(),
              *read(result / 'analysis-fields.json')['fields'].values()]
    storage = read(evidence / 'storage.json')
    require(len(arrays) == storage['newRawArrays'] == 21
            and sum(x['bytes'] for x in arrays) == storage['newRawArrayBytes'] == 868912896
            and storage['lastActualCleanupCycle'] == progress['storageMaintenance']['lastActualArchiveCycle'] == 78
            and storage['nextCleanupDueCycle'] == 81,
            'Storage scope differs')
    scheduler = read(result / 'scheduler.json')
    require(scheduler['state'] == 'succeeded' and scheduler['result']['returncode'] == 0
            and progress['inFlightResearchCycle80']['status'] == 'completed-authoritatively'
            and progress['inFlightResearchCycle80']['noLiveJob'],
            'Completed job status differs')
    hashes = scheduler['payload_custody']
    require(len({hashes[k]['sha256'] for k in ['declared', 'sealed', 'executed']}) == 1,
            'Executed payload custody differs')
    count = 0
    sources = [(article, article.read_text()), (plan, plan.read_text()),
               (evidence / 'README.md', (evidence / 'README.md').read_text()),
               (blog, body.split('## Cycle 80:', 1)[1].split('## What would close the original goal', 1)[0])]
    for source_path, text in sources:
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith(('https:', 'http:', '#')):
                continue
            path = unquote(target.split('#', 1)[0]).strip('<>')
            require((source_path.parent / path).resolve().exists(), 'Broken link: ' + target)
            count += 1
    for section, size in [('source_docs', 3), ('source_code', 10), ('source_artifacts', 7)]:
        entries = body.split(section + ':\n', 1)[1].splitlines()[:size]
        require(len(entries) == size and all(x.startswith('  - ') and (blog.parent / x[4:]).resolve().exists() for x in entries),
                'New frontmatter links differ: ' + section)
        count += size
    return {
        'schema': 'tce-rectifier-correction-publication-qa-v1',
        'status': 'passed',
        'localLinksChecked': count,
        'articleSHA256': sha(article),
        'nextPlanSHA256': sha(plan),
        'livingBlogSHA256': sha(blog),
        'auditSourceSHA256': sha(__file__),
        'mechanics': mechanics,
        'scope': 'Publication, source, review, current-status, countertrend and render joins only. No new real-array numerical replay, full newest-post survey, storage cleanup, model computation or image generation.'
    }


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
