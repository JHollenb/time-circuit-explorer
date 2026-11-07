"""Read-only cycle69 source/evidence/synthesis/local-backlink validation."""
import json
from pathlib import Path
import re
from urllib.parse import unquote
from encoder_atoms import read,sha
from publish_partition_feedback import audit


def execute():
    root=Path(__file__).resolve().parent.parent;docs=root/'docs';p=docs/'results/partition-feedback';e=docs/'results/partition-feedback-evidence'
    mechanics=audit(p);progress=read(docs/'zero-neural-progress.json');advisor=read(e/'advisor.json')
    article=docs/'partition-feedback.md';plan=docs/'plans/image-partition-moments.md';blog=root.parent/'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'
    if progress['researchCycle']!=69 or progress['status']!='active' or progress['liveJobs'] or progress['thisGoalTurn']['classification']!='progress':raise ValueError('Goal/result state differs')
    if progress['findingsLog']['sha256']!=sha(blog) or progress['findingsLog']['lastUpdatedCycle']!=69 or progress['nextCut']['status']!='written-unexecuted' or progress['nextCut']['planSHA256']!=sha(plan):raise ValueError('Livinglog/nextplan differs')
    if advisor['reviewedArticleSHA256']!=sha(article) or advisor['reviewedPlanSHA256']!=sha(plan) or advisor['reviewedBlogSHA256']!=sha(blog) or advisor['reportSHA256']!=sha(p/'report.json'):raise ValueError('Reviewed synthesis differs')
    if advisor['lastFullNewestPostSurveyCycle']!=69 or advisor['nextFullSurveyDueCycle']!=72:raise ValueError('Actual survey cadence differs')
    for folder,code,file in [('partition-feedback-figure','plot_partition_feedback.py','partition-feedback.png'),('partition-mass-figure-v2','plot_partition_mass_v2.py','partition-mass-v2.png')]:
        f=docs/'results'/folder;meta=read(f/'figure.json')
        if meta['plotSourceSHA256']!=sha(root/'generator'/code) or meta['imageSHA256']!=sha(f/file):raise ValueError('Figure source/output differs')
    for name in ['storage-cleanup-cycle69','storage-audit-cycle69']:
        f=docs/'results'/name;meta=read(f/'publication.json')
        if meta['archiverSourceSHA256']!=sha(root/'generator/storage_transports.py'):raise ValueError('Storage source differs')
        for n,h in meta['files'].items():
            if sha(f/n)!=h:raise ValueError('Storage publication differs')
    body=blog.read_text().split('## Cycle 69:',1)[1].split('## What would close the original goal',1)[0]
    files=[(article,article.read_text()),(plan,plan.read_text()),(e/'README.md',(e/'README.md').read_text()),(blog,body)]
    files += [(docs/'results'/n/'README.md',(docs/'results'/n/'README.md').read_text()) for n in ['storage-cleanup-cycle69','storage-audit-cycle69']]
    count=0
    for source,text in files:
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('https:','http:','#')):continue
            path=unquote(target.split('#',1)[0]).strip('<>')
            if not (source.parent/path).resolve().exists():raise ValueError('Broken local link: '+str(source)+' → '+target)
            count+=1
    for role in ['candidate','native']:
        if sha(p/(role+'-A.png'))!=sha(root.parent/'obsidian/images'/('zero-neural-cycle69-'+role+'-A.png')):raise ValueError('Notebook image copy differs')
    return {'schema':'tce-partition-feedback-publication-qa-v1','status':'passed','localLinksChecked':count,'articleSHA256':sha(article),'nextPlanSHA256':sha(plan),'livingBlogSHA256':sha(blog),
        'mechanics':mechanics,'scope':'Read-only source/evidence/publication/storage/backlink joins. No new model/numerical run or semantic certification; original goal remains active.'}


if __name__=='__main__':print(json.dumps(execute(),indent=2,sort_keys=True))
