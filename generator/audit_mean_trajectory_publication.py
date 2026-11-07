"""Read-only cycle75 mechanics, source, synthesis and local-link validation."""
import json
from pathlib import Path
import re
from urllib.parse import unquote
from encoder_atoms import read,sha
from publish_mean_trajectory import audit


def execute():
    root=Path(__file__).resolve().parent.parent;docs=root/'docs';p=docs/'results/mean-up-trajectory';e=docs/'results/mean-up-trajectory-evidence'
    mechanics=audit(p);progress=read(docs/'zero-neural-progress.json');advisor=read(e/'advisor.json')
    article=docs/'mean-up-trajectory.md';plan=docs/'plans/own-state-gated-covariance.md';blog=root.parent/'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'
    if progress['researchCycle']!=75 or progress['status']!='active' or progress['liveJobs'] or progress['thisGoalTurn']['classification']!='progress':raise ValueError('Goal/result state differs')
    if progress['findingsLog']['sha256']!=sha(blog) or progress['findingsLog']['lastUpdatedCycle']!=75 or progress['nextCut']['status']!='written-unexecuted' or progress['nextCut']['planSHA256']!=sha(plan):raise ValueError('Livinglog/nextplan differs')
    for path,digest in advisor['finalReviewedSynthesisSnapshots'].items():
        if sha(root/path)!=digest:raise ValueError('Final reviewed synthesis differs:'+path)
    if advisor['reviewedResultSnapshots']['report.json']!=sha(p/'report.json') or advisor['lastFullNewestPostSurveyCycle']!=75 or advisor['nextFullSurveyDueCycle']!=78 or not advisor['newFullNewestPostSurvey']:raise ValueError('Actual review scope differs')
    f=docs/'results/mean-up-trajectory-figure';meta=read(f/'figure.json')
    if meta['plotSourceSHA256']!=sha(root/'generator/plot_mean_trajectory.py') or meta['imageSHA256']!=sha(f/'mean-up-trajectory.png'):raise ValueError('Figure source/output differs')
    for n in ['plan.json','result.json','events.jsonl']:
        if sha(docs/'results/storage-cleanup-cycle75'/n)!=sha(root/'artifacts/storage-cleanup-cycle75'/n):raise ValueError('Storage receipt copy differs')
    if sha(docs/'results/storage-audit-cycle75/result.json')!=sha(root/'artifacts/storage-audit-cycle75/result.json'):raise ValueError('Storage audit copy differs')
    body=blog.read_text().split('## Cycle 75:',1)[1].split('## What would close the original goal',1)[0]
    files=[(article,article.read_text()),(plan,plan.read_text()),(e/'README.md',(e/'README.md').read_text()),(blog,body),(docs/'results/storage-cleanup-cycle75/README.md',(docs/'results/storage-cleanup-cycle75/README.md').read_text()),(docs/'results/storage-audit-cycle75/README.md',(docs/'results/storage-audit-cycle75/README.md').read_text())];count=0
    for source,text in files:
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('https:','http:','#')):continue
            path=unquote(target.split('#',1)[0]).strip('<>')
            if not (source.parent/path).resolve().exists():raise ValueError('Broken local link: '+str(source)+' → '+target)
            count+=1
    for role in ['candidate','native']:
        if sha(p/(role+'-A.png'))!=sha(root.parent/'obsidian/images'/('zero-neural-cycle75-'+role+'-A.png')):raise ValueError('Notebook image copy differs')
    return {'schema':'tce-mean-trajectory-publication-qa-v1','status':'passed','localLinksChecked':count,'articleSHA256':sha(article),'nextPlanSHA256':sha(plan),'livingBlogSHA256':sha(blog),
        'auditSourceSHA256':sha(__file__),'mechanics':mechanics,'scope':'Read-only source/evidence/publication/backlink joins. No new model/numerical run or semantic certification; original goal remains active.'}


if __name__=='__main__':print(json.dumps(execute(),indent=2,sort_keys=True))
