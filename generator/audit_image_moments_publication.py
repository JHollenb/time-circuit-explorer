"""Read-only cycle70 mechanics, source, synthesis and local-link validation."""
import json
from pathlib import Path
import re
from urllib.parse import unquote
from encoder_atoms import read,sha
from publish_image_moments import audit


def execute():
    root=Path(__file__).resolve().parent.parent;docs=root/'docs';p=docs/'results/image-partition-moments';e=docs/'results/image-partition-moments-evidence'
    mechanics=audit(p);progress=read(docs/'zero-neural-progress.json');advisor=read(e/'advisor.json')
    article=docs/'image-partition-moments.md';plan=docs/'plans/gate-first-terminal-mlp.md';blog=root.parent/'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'
    if progress['researchCycle']!=70 or progress['status']!='active' or progress['liveJobs'] or progress['thisGoalTurn']['classification']!='progress':raise ValueError('Goal/result state differs')
    if progress['findingsLog']['sha256']!=sha(blog) or progress['findingsLog']['lastUpdatedCycle']!=70 or progress['nextCut']['status']!='written-unexecuted' or progress['nextCut']['planSHA256']!=sha(plan):raise ValueError('Livinglog/nextplan differs')
    for path,digest in advisor['finalReviewedSynthesisSnapshots'].items():
        if sha(root/path)!=digest:raise ValueError('Final reviewed synthesis differs:'+path)
    if advisor['reviewedResultSnapshots']['report.json']!=sha(p/'report.json') or advisor['lastFullNewestPostSurveyCycle']!=69 or advisor['nextFullSurveyDueCycle']!=72 or advisor['newFullNewestPostSurvey']:raise ValueError('Actual review scope differs')
    f=docs/'results/image-partition-moments-figure';meta=read(f/'figure.json')
    if meta['plotSourceSHA256']!=sha(root/'generator/plot_image_moments.py') or meta['imageSHA256']!=sha(f/'image-partition-moments.png'):raise ValueError('Figure source/output differs')
    allocation=read(e/'affine-allocation.json')
    if allocation['accountingSourceSHA256']!=sha(root/'generator/account_image_moments_affines.py') or allocation['executionSHA256']!=sha(p/'execution.json') or allocation['MLPMACs']!=12872016986112:raise ValueError('Affine allocation differs')
    body=blog.read_text().split('## Cycle 70:',1)[1].split('## What would close the original goal',1)[0]
    files=[(article,article.read_text()),(plan,plan.read_text()),(e/'README.md',(e/'README.md').read_text()),(blog,body)];count=0
    for source,text in files:
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('https:','http:','#')):continue
            path=unquote(target.split('#',1)[0]).strip('<>')
            if not (source.parent/path).resolve().exists():raise ValueError('Broken local link: '+str(source)+' → '+target)
            count+=1
    for role in ['candidate','native']:
        if sha(p/(role+'-A.png'))!=sha(root.parent/'obsidian/images'/('zero-neural-cycle70-'+role+'-A.png')):raise ValueError('Notebook image copy differs')
    return {'schema':'tce-image-moments-publication-qa-v1','status':'passed','localLinksChecked':count,'articleSHA256':sha(article),'nextPlanSHA256':sha(plan),'livingBlogSHA256':sha(blog),
        'auditSourceSHA256':sha(__file__),'mechanics':mechanics,'scope':'Read-only source/evidence/publication/backlink joins. No new model/numerical run or semantic certification; original goal remains active.'}


if __name__=='__main__':print(json.dumps(execute(),indent=2,sort_keys=True))
