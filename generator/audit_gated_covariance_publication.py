"""Read-only cycle76 source/evidence/narrative/backlink and status joins."""
import json
from pathlib import Path
import re
from urllib.parse import unquote
from encoder_atoms import read,sha
from publish_gated_covariance import audit


def execute():
    root=Path(__file__).resolve().parent.parent;docs=root/'docs';p=docs/'results/own-state-gated-covariance';e=docs/'results/own-state-gated-covariance-evidence'
    mechanics=audit(p);progress=read(docs/'zero-neural-progress.json');advisor=read(e/'advisor.json');article=docs/'own-state-gated-covariance.md';plan=docs/'plans/mean-gate-variable-gate.md'
    blog=root.parent/'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md';body=blog.read_text()
    if progress['researchCycle']!=76 or progress['status']!='active' or progress['liveJobs'] or progress['thisGoalTurn']['classification']!='progress':raise ValueError('Current goal/result differs')
    if progress['findingsLog']['lastUpdatedCycle']!=76 or progress['findingsLog']['sha256']!=sha(blog) or progress['nextCut']['status']!='written-unexecuted' or progress['nextCut']['planSHA256']!=sha(plan):raise ValueError('Livinglog/unexecutedplan differs')
    for name,h in advisor['finalReviewedSynthesisSnapshots'].items():
        if sha(root/name)!=h:raise ValueError('Final reviewed synthesis differs:'+name)
    if advisor['newFullNewestPostSurvey'] or advisor['lastFullNewestPostSurveyCycle']!=75 or advisor['nextFullSurveyDueCycle']!=78 or advisor['reviewedResultSnapshots']['report.json']!=sha(p/'report.json'):raise ValueError('Actual review scope differs')
    f=docs/'results/own-state-gated-covariance-figure-v2';meta=read(f/'figure.json');render=read(e/'render-qa.json')
    if meta['plotSourceSHA256']!=sha(root/'generator/plot_gated_covariance_v2.py') or meta['imageSHA256']!=sha(f/'own-state-gated-covariance.png') or meta['imageSHA256']!=sha(Path(meta['notebookCopy'])) or render['publishedImageSHA256']!=meta['imageSHA256']:raise ValueError('Actual figure/source/copy differs')
    if render['supersededImageSHA256']!=sha(docs/'results/own-state-gated-covariance-figure/own-state-gated-covariance.png'):raise ValueError('Original render finding lost')
    count=0
    files=[(article,article.read_text()),(plan,plan.read_text()),(e/'README.md',(e/'README.md').read_text()),(blog,body.split('## Cycle 76:',1)[1].split('## What would close the original goal',1)[0])]
    for source,text in files:
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('https:','http:','#')):continue
            path=unquote(target.split('#',1)[0]).strip('<>')
            if not (source.parent/path).resolve().exists():raise ValueError('Broken local link:'+str(source)+' → '+target)
            count+=1
    for section,n in [('source_docs',3),('source_code',10),('source_artifacts',7)]:
        entries=body.split(section+':\n',1)[1].splitlines()[:n]
        if len(entries)!=n or any(not line.startswith('  - ') or not (blog.parent/line[4:]).resolve().exists() for line in entries):raise ValueError('New frontmatter source link differs')
        count+=n
    return {'schema':'tce-gated-covariance-publication-qa-v1','status':'passed','localLinksChecked':count,'articleSHA256':sha(article),'nextPlanSHA256':sha(plan),'livingBlogSHA256':sha(blog),'auditSourceSHA256':sha(__file__),'mechanics':mechanics,
        'scope':'Source/evidence/currentstatus/review/narrative/backlink/rendercustody only. No new realarray/modelrun or semantic/generalization/zero-neuralcertification; goalactive.'}


if __name__=='__main__':print(json.dumps(execute(),indent=2,sort_keys=True))
