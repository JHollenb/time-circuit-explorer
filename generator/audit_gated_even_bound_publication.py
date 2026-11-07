"""Read-only evidence/review/current-status/backlink/render publication joins."""
import json
from pathlib import Path
import re
from urllib.parse import unquote
from encoder_atoms import read,sha
from publish_gated_even_bound import audit


def execute():
    root=Path(__file__).resolve().parent.parent;docs=root/'docs';p=docs/'results/gated-even-bound';e=docs/'results/gated-even-bound-evidence'
    mechanics=audit(p);progress=read(docs/'zero-neural-progress.json');advisor=read(e/'advisor.json');report=read(p/'report.json');r=read(p/'request.json')
    article=docs/'gated-even-bound.md';plan=docs/'plans/combined-rectifier-correction.md';blog=root.parent/'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md';body=blog.read_text()
    if progress['researchCycle']!=79 or progress['status']!='active' or progress['liveJobs'] or progress['thisGoalTurn']['classification']!='progress':raise ValueError('Current goal differs')
    if progress['findingsLog']['lastUpdatedCycle']!=79 or progress['findingsLog']['sha256']!=sha(blog) or progress['nextCut']['status']!='written-unexecuted' or progress['nextCut']['planSHA256']!=sha(plan):raise ValueError('Livinglog/nextcut differs')
    for name,h in advisor['finalReviewedSynthesisSnapshots'].items():
        if sha(root/name)!=h:raise ValueError('Final reviewed synthesis differs:'+name)
    for name,h in advisor['reviewedResultSnapshots'].items():
        if sha(p/name)!=h:raise ValueError('Actual result snapshot differs')
    if advisor['newFullNewestPostSurvey'] or advisor['lastFullNewestPostSurveyCycle']!=78 or advisor['nextFullSurveyDueCycle']!=81 or advisor['fullSurveySHA256']!=sha(docs/'results/research-review-cycle78/advisor-survey.json'):raise ValueError('Periodic advisory scope differs')
    tests=read(e/'tests.json');pre=read(e/'advisor-prestage.json')
    if tests['passed']!=346 or tests['sourceHashes']!={n:r['sourceHashes'][n] for n in r['ownedSources']}:raise ValueError('Tests/source snapshot differs')
    for name,h in pre['reviewedSources'].items():
        if sha(root/'generator'/name)!=h:raise ValueError('Source review snapshot differs')
    if progress['gatedEvenBoundEvidence']['etaOverRemainingNormUpper']!={n:x['etaOverRemainingNormUpper'] for n,x in report['panels'].items()}:raise ValueError('Progress numerical summary differs')
    figure=read(docs/'results/gated-even-bound-figure/figure.json');render=read(e/'render-qa.json')
    if figure['plotSourceSHA256']!=sha(root/'generator/plot_gated_even_bound.py') or figure['imageSHA256']!=sha(docs/'results/gated-even-bound-figure/gated-even-bound.png') or figure['imageSHA256']!=sha(Path(figure['notebookCopy'])) or render['imageSHA256']!=figure['imageSHA256']:raise ValueError('Actual render/source/copy differs')
    storage=read(e/'storage.json')
    if storage['newRawArrays']!=1 or storage['newRawArrayBytes']!=123008 or storage['lastActualCleanupCycle']!=78 or storage['nextCleanupDueCycle']!=81 or progress['storageMaintenance']['lastActualArchiveCycle']!=78:raise ValueError('Storage scope differs')
    count=0
    sources=[(article,article.read_text()),(plan,plan.read_text()),(e/'README.md',(e/'README.md').read_text()),(blog,body.split('## Cycle 79:',1)[1].split('## What would close the original goal',1)[0])]
    for source,text in sources:
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('https:','http:','#')):continue
            path=unquote(target.split('#',1)[0]).strip('<>')
            if not (source.parent/path).resolve().exists():raise ValueError('Broken local link:'+str(source)+' → '+target)
            count+=1
    for section,n in [('source_docs',3),('source_code',6),('source_artifacts',7)]:
        entries=body.split(section+':\n',1)[1].splitlines()[:n]
        if len(entries)!=n or any(not x.startswith('  - ') or not (blog.parent/x[4:]).resolve().exists() for x in entries):raise ValueError('New frontmatter link differs')
        count+=n
    return {'schema':'tce-gated-even-bound-publication-qa-v1','status':'passed','localLinksChecked':count,'articleSHA256':sha(article),'nextPlanSHA256':sha(plan),'livingBlogSHA256':sha(blog),'auditSourceSHA256':sha(__file__),'mechanics':mechanics,
        'scope':'Executed evidence/source/review/status/storage/render/backlink joins only. No new model/real-array arithmetic or semantic/generalization/zero-neural serving certification; goal active.'}


if __name__=='__main__':print(json.dumps(execute(),indent=2,sort_keys=True))
