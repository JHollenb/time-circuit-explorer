"""Read-only publication/review/current-status/storage/render joins for cycle81."""
import json
from pathlib import Path
import re
from urllib.parse import unquote

from encoder_atoms import read, sha
from publish_rectifier_sign_support import audit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def execute():
    root = Path(__file__).resolve().parent.parent
    docs = root / 'docs'
    result = docs / 'results/rectifier-sign-support'
    evidence = docs / 'results/rectifier-sign-support-evidence'
    mechanics = audit(result)
    progress = read(docs / 'zero-neural-progress.json')
    advisor = read(evidence / 'advisor.json')
    report = read(result / 'report.json')
    request = read(result / 'request.json')
    article = docs / 'rectifier-liveness.md'
    plan = docs / 'plans/current-state-gate-up-basis.md'
    blog = root.parent / 'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'
    body = blog.read_text()
    require(progress['researchCycle'] == 81 and progress['status'] == 'active'
            and not progress['liveJobs'] and progress['thisGoalTurn']['classification'] == 'progress',
            'Current goal differs')
    require(progress['findingsLog']['lastUpdatedCycle'] == 81
            and progress['findingsLog']['sha256'] == sha(blog)
            and progress['nextCut']['status'] == 'written-unexecuted'
            and progress['nextCut']['noJobSubmitted']
            and progress['nextCut']['planSHA256'] == sha(plan)
            and progress['nextCut']['prospectiveMatrixMACs'] == 77309411328,
            'Living log or unexecuted next cut differs')
    for name, fingerprint in advisor['finalReviewedSynthesisSnapshots'].items():
        require(sha(root / name) == fingerprint, 'Reviewed final synthesis differs: ' + name)
    require(len(advisor['finalReviewedSynthesisSnapshots']) == 4, 'Four final syntheses required')
    for name, fingerprint in advisor['reviewedResultSnapshots'].items():
        require(sha(result / name) == fingerprint, 'Actual result snapshot differs: ' + name)
    survey_path = docs / 'results/research-review-cycle81/advisor-survey.json'
    survey = read(survey_path)
    require(advisor['newFullNewestPostSurvey'] and advisor['lastFullNewestPostSurveyCycle'] == 81
            and advisor['nextFullSurveyDueCycle'] == 84 and advisor['fullSurveySHA256'] == sha(survey_path)
            and survey['newFullNewestPostSurvey'] and survey['cycle'] == 81,
            'Full newest-post review scope differs')
    source = read(result / 'inputs.json')
    for name, fingerprint in source['preStageReviewSnapshots'].items():
        require(sha(root / name) == fingerprint, 'Pre-stage review dependency differs: ' + name)
    tests = read(evidence / 'tests.json')
    pre = read(evidence / 'advisor-prestage.json')
    require(pre == advisor['preStageReview'] and pre['planSHA256'] == sha(result / 'frozen-plan.md'),
            'Pre-stage source/plan review differs')
    require(tests['passed'] == 348 and tests['exitCode'] == 0
            and tests['sourceHashes'] == {n: request['sourceHashes'][n] for n in request['ownedSources']},
            'Tests and executed sources differ')
    for name, fingerprint in pre['reviewedSources'].items():
        require(sha(root / 'generator' / name) == fingerprint, 'Reviewed source differs: ' + name)
    current = progress['rectifierSignSupportEvidence']
    require(current['coefficientPanels'] == report['panels']
            and current['denseStreamFold'] == report['denseStreamFold']
            and current['work'] == report['physicalWork'],
            'Progress numerical account differs')
    require(report['panels']['all']['expressions']['combined']['coefficientDensity'] == 1
            and report['panels']['all']['expressions']['combined']['fixedTileSavingsMACs'] == 0
            and not report['denseStreamFold']['constructed'], 'Measured demand scope differs')
    figure_root = docs / 'results/rectifier-sign-support-figure-v2'
    figure = read(figure_root / 'figure.json')
    render = read(evidence / 'render-qa.json')
    require(figure['plotSourceSHA256'] == sha(root / 'generator/plot_rectifier_sign_support_v2.py')
            and figure['sourceReportSHA256'] == sha(result / 'report.json')
            and figure['imageSHA256'] == sha(figure_root / 'rectifier-sign-support.png')
            and figure['imageSHA256'] == sha(Path(figure['notebookCopy']))
            and figure['imageSHA256'] == render['imageSHA256'] == advisor['figureSHA256'],
            'Actual corrected figure/source/notebook copy differs')
    arrays = read(result / 'fields.json')['fields']
    storage = read(evidence / 'storage.json')
    storage_root = docs / 'results/storage-audit-cycle81'
    review = read(storage_root / 'publication.json')
    require(len(arrays) == storage['newRawArrays'] == 6
            and sum(x['bytes'] for x in arrays.values()) == storage['newRawArrayBytes'] == 6124800
            and storage['lastActualArchiveCycle'] == progress['storageMaintenance']['lastActualArchiveCycle'] == 78
            and storage['fullStorageReviewCycle'] == review['cycle'] == 81
            and review['totalRestorePointers'] == 93 and review['remaining100MBCandidates'] == 0
            and review['newArchivedBytes'] == 0 and review['freshFullColdBytesVerified'] == 3806240603
            and review['nextStorageReviewDueCycle'] == 84,
            'Storage review/retention scope differs')
    for name, fingerprint in review['files'].items():
        require(sha(storage_root / name) == fingerprint, 'Storage metadata snapshot differs')
    instrument = read(evidence / 'saturn-instrument.json')
    sweep = read(evidence / 'saturn-run.json')
    verify = read(evidence / 'saturn-verify.json')
    require(sweep['exit_code'] == verify['exit_code'] == 0
            and sweep['steps']['watch']['records'] == instrument['records'] == 1632
            and sweep['steps']['watch']['streams'] == instrument['streams'] == 2
            and sweep['steps']['watch']['fires'] == 0
            and verify['steps']['watch']['ledgers_missing_expected'] == 2,
            'Actually applied Saturn instrument differs')
    for name, fingerprint in instrument['rawReceipts'].items():
        require(sha(evidence / name) == fingerprint, 'Instrument receipt differs')
    scheduler = read(result / 'scheduler.json')
    require(scheduler['state'] == 'succeeded' and scheduler['result']['returncode'] == 0
            and progress['inFlightResearchCycle81']['status'] == 'completed-authoritatively'
            and progress['inFlightResearchCycle81']['noLiveJob'], 'Completed job state differs')
    custody = scheduler['payload_custody']
    require(custody['required'] and len({custody[k]['sha256'] for k in ['declared', 'sealed', 'executed']}) == 1,
            'Executed payload custody differs')
    count = 0
    sources = [(article, article.read_text()), (plan, plan.read_text()),
               (evidence / 'README.md', (evidence / 'README.md').read_text()),
               (storage_root / 'README.md', (storage_root / 'README.md').read_text()),
               (blog, body.split('## Cycle 81:', 1)[1].split('## What would close the original goal', 1)[0])]
    for source_path, text in sources:
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith(('https:', 'http:', '#')):
                continue
            path = unquote(target.split('#', 1)[0]).strip('<>')
            require((source_path.parent / path).resolve().exists(), 'Broken local link: ' + target)
            count += 1
    for section, size in [('source_docs', 5), ('source_code', 6), ('source_artifacts', 9)]:
        entries = body.split(section + ':\n', 1)[1].splitlines()[:size]
        require(len(entries) == size and all(x.startswith('  - ') and (blog.parent / x[4:]).resolve().exists() for x in entries),
                'New frontmatter links differ: ' + section)
        count += size
    return {'schema': 'tce-rectifier-sign-support-publication-qa-v1', 'status': 'passed',
            'localLinksChecked': count, 'articleSHA256': sha(article), 'nextPlanSHA256': sha(plan),
            'livingBlogSHA256': sha(blog), 'auditSourceSHA256': sha(__file__), 'mechanics': mechanics,
            'scope': 'Publication/review/current-goal/scalar-result/storage/render joins only. No independent real-array replay, sparse lowering, new learned product, model state update or image generation.'}


if __name__ == '__main__':
    print(json.dumps(execute(), indent=2))
