"""Read-only joins of executed evidence, current status, figures and backlinks."""
import json
from pathlib import Path
import re
from urllib.parse import unquote
from encoder_atoms import read,sha
from publish_mean_gate import audit


def execute():
    root=Path(__file__).resolve().parent.parent;docs=root/'docs';p=docs/'results/mean-gate';e=docs/'results/mean-gate-evidence'
    mechanics=audit(p);progress=read(docs/'zero-neural-progress.json');advisor=read(e/'advisor.json');article=docs/'mean-gate.md';plan=docs/'plans/odd-gate-quadratic.md'
    blog=root.parent/'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md';body=blog.read_text()
    if progress['researchCycle']!=77 or progress['status']!='active' or progress['liveJobs'] or progress['thisGoalTurn']['classification']!='progress':raise ValueError('Current goal/result differs')
    if progress['findingsLog']['lastUpdatedCycle']!=77 or progress['findingsLog']['sha256']!=sha(blog) or progress['nextCut']['status']!='written-unexecuted' or progress['nextCut']['planSHA256']!=sha(plan):raise ValueError('Living log or unexecuted plan differs')
    for name,h in advisor['finalReviewedSynthesisSnapshots'].items():
        if sha(root/name)!=h:raise ValueError('Final reviewed synthesis differs:'+name)
    for name,h in advisor['reviewedResultSnapshots'].items():
        if sha(p/name)!=h:raise ValueError('Reviewed actual result differs:'+name)
    if advisor['newFullNewestPostSurvey'] or advisor['lastFullNewestPostSurveyCycle']!=75 or advisor['nextFullSurveyDueCycle']!=78:raise ValueError('Actual periodic review scope differs')
    request=read(p/'request.json');tests=read(e/'tests.json')
    if tests['passed']!=341 or tests['sourceHashes']!={n:request['sourceHashes'][n] for n in request['ownedSources']}:raise ValueError('Test/source snapshot differs')
    f=docs/'results/mean-gate-figure';figure=read(f/'figure.json');render=read(e/'render-qa.json')
    if figure['plotSourceSHA256']!=sha(root/'generator/plot_mean_gate.py') or figure['imageSHA256']!=sha(f/'mean-gate.png') or figure['imageSHA256']!=sha(Path(figure['notebookCopy'])) or render['imageSHA256']!=figure['imageSHA256']:raise ValueError('Actual rendered figure/copy differs')
    count=0
    files=[(article,article.read_text()),(plan,plan.read_text()),(e/'README.md',(e/'README.md').read_text()),(blog,body.split('## Cycle 77:',1)[1].split('## What would close the original goal',1)[0])]
    for source,text in files:
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('https:','http:','#')):continue
            path=unquote(target.split('#',1)[0]).strip('<>')
            if not (source.parent/path).resolve().exists():raise ValueError('Broken local link:'+str(source)+' → '+target)
            count+=1
    for section,n in [('source_docs',3),('source_code',9),('source_artifacts',7)]:
        entries=body.split(section+':\n',1)[1].splitlines()[:n]
        if len(entries)!=n or any(not line.startswith('  - ') or not (blog.parent/line[4:]).resolve().exists() for line in entries):raise ValueError('New frontmatter source link differs')
        count+=n
    return {'schema':'tce-mean-gate-publication-qa-v1','status':'passed','localLinksChecked':count,'articleSHA256':sha(article),'nextPlanSHA256':sha(plan),
        'livingBlogSHA256':sha(blog),'auditSourceSHA256':sha(__file__),'mechanics':mechanics,
        'scope':'Executed source/evidence/current-status/advisor/render/backlink joins only. No new scientific array/model run or semantic/generalization/zero-neural certification; goal active.'}


if __name__=='__main__':print(json.dumps(execute(),indent=2,sort_keys=True))
