"""Read-only final source/evidence/synthesis/figure/backlink checks for cycle74."""
import json
from pathlib import Path
import re
from urllib.parse import unquote
from encoder_atoms import read,sha
from publish_mean_up import audit


def execute():
    root=Path(__file__).resolve().parent.parent;docs=root/'docs';p=docs/'results/current-stream-mean-up';e=docs/'results/current-stream-mean-up-evidence'
    mechanics=audit(p);progress=read(docs/'zero-neural-progress.json');advisor=read(e/'advisor.json')
    article=docs/'current-stream-mean-up.md';plan=docs/'plans/mean-up-own-trajectory.md';blog=root.parent/'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'
    if progress['researchCycle']!=74 or progress['status']!='active' or progress['liveJobs'] or progress['thisGoalTurn']['classification']!='progress' or 'inFlightResearchCycle' in progress:raise ValueError('Current goal/result phase differs')
    if progress['findingsLog']['sha256']!=sha(blog) or progress['findingsLog']['lastUpdatedCycle']!=74 or progress['nextCut']['status']!='written-unexecuted' or progress['nextCut']['planSHA256']!=sha(plan):raise ValueError('Livinglog/nextplan differs')
    if progress['meanUpCompletionEvidence']['reportSHA256']!=sha(p/'report.json') or progress['reviews']['latestReviewReceiptSHA256']!=sha(e/'advisor.json'):raise ValueError('Actual result/review identity differs')
    for path,digest in advisor['finalReviewedSynthesisSnapshots'].items():
        if sha(root/path)!=digest:raise ValueError('Reviewed synthesis differs:'+path)
    if advisor['reviewedResultSnapshots']['report.json']!=sha(p/'report.json') or advisor['lastFullNewestPostSurveyCycle']!=72 or advisor['nextFullSurveyDueCycle']!=75 or advisor['newFullNewestPostSurvey']:raise ValueError('Actual review scope differs')
    f=docs/'results/current-stream-mean-up-figure';meta=read(f/'figure.json');image=f/'current-stream-mean-up.png'
    if meta['plotSourceSHA256']!=sha(root/'generator/plot_mean_up.py') or meta['reportSHA256']!=sha(p/'report.json') or meta['rowStreamSHA256']!=sha(p/'write-observations.jsonl.gz') or meta['imageSHA256']!=sha(image) or sha(meta['notebookCopy'])!=sha(image):raise ValueError('Figure source/output/copy differs')
    rows=read(e/'row-heterogeneity.json')
    if rows['sourceSHA256']!=sha(p/'write-observations.jsonl.gz') or rows['summarizerSHA256']!=sha(root/'generator/summarize_mean_up.py'):raise ValueError('Scalar-description provenance differs')
    tests=read(e/'tests.json')
    if tests['passed']!=338 or tests['exitCode']!=0 or tests['sourceHashes']!={n:sha(root/'generator'/n) for n in read(p/'request.json')['ownedSources']}:raise ValueError('Final mechanics source snapshot differs')
    body=blog.read_text().split('## Cycle 74:',1)[1].split('## What would close the original goal',1)[0]
    count=0
    for source,text in [(article,article.read_text()),(plan,plan.read_text()),(e/'README.md',(e/'README.md').read_text()),(blog,body)]:
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('https:','http:','#')):continue
            path=unquote(target.split('#',1)[0]).strip('<>')
            if not (source.parent/path).resolve().exists():raise ValueError('Broken local link:'+str(source)+' → '+target)
            count+=1
    return {'schema':'tce-mean-up-publication-qa-v1','status':'passed','localLinksChecked':count,'articleSHA256':sha(article),'nextPlanSHA256':sha(plan),
        'livingBlogSHA256':sha(blog),'auditSourceSHA256':sha(__file__),'mechanics':mechanics,'scope':'Read-only executedsource/evidence/final synthesis/figure/scalarprovenance/local-links. No newmodel or semantic certification;strictgoalactive.'}


if __name__=='__main__':print(json.dumps(execute(),indent=2,sort_keys=True))
