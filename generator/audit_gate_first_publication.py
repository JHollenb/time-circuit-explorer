"""Read-only cycle71 source/evidence/synthesis and local-link verification."""
import json
from pathlib import Path
import re
from urllib.parse import unquote
from encoder_atoms import read,sha
from publish_gate_first import audit


def execute():
    root=Path(__file__).resolve().parent.parent;docs=root/'docs';p=docs/'results/gate-first-terminal';e=docs/'results/gate-first-terminal-evidence'
    mechanics=audit(p);progress=read(docs/'zero-neural-progress.json');advisor=read(e/'advisor.json');article=docs/'gate-first-terminal-mlp.md';plan=docs/'plans/gate-first-single-trajectory.md'
    blog=root.parent/'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'
    if progress['researchCycle']!=71 or progress['status']!='active' or progress['liveJobs'] or progress['thisGoalTurn']['classification']!='progress':raise ValueError('Goal/result state differs')
    if progress['findingsLog']['sha256']!=sha(blog) or progress['findingsLog']['lastUpdatedCycle']!=71 or progress['nextCut']['status']!='written-unexecuted' or progress['nextCut']['planSHA256']!=sha(plan):raise ValueError('Livinglog/nextplan differs')
    for path,digest in advisor['finalReviewedSynthesisSnapshots'].items():
        if sha(root/path)!=digest:raise ValueError('Final reviewed synthesis differs:'+path)
    if advisor['reviewedResultSnapshots']['report.json']!=sha(p/'report.json') or advisor['lastFullNewestPostSurveyCycle']!=69 or advisor['nextFullSurveyDueCycle']!=72 or advisor['newFullNewestPostSurvey']:raise ValueError('Actual review scope differs')
    f=docs/'results/gate-first-terminal-figure';meta=read(f/'figure.json')
    if meta['plotSourceSHA256']!=sha(root/'generator/plot_gate_first.py') or meta['imageSHA256']!=sha(f/'gate-first-terminal.png') or meta['imageSHA256']!=sha(root.parent/'obsidian/images/zero-neural-cycle71-gate-first-terminal.png'):raise ValueError('Figure source/output/copy differs')
    body=blog.read_text().split('## Cycle 71:',1)[1].split('## What would close the original goal',1)[0]
    files=[(article,article.read_text()),(plan,plan.read_text()),(e/'README.md',(e/'README.md').read_text()),(blog,body)];count=0
    for source,text in files:
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('https:','http:','#')):continue
            path=unquote(target.split('#',1)[0]).strip('<>')
            if not (source.parent/path).resolve().exists():raise ValueError('Broken local link: '+str(source)+' → '+target)
            count+=1
    return {'schema':'tce-gate-first-publication-qa-v1','status':'passed','localLinksChecked':count,'articleSHA256':sha(article),'nextPlanSHA256':sha(plan),'livingBlogSHA256':sha(blog),'auditSourceSHA256':sha(__file__),'mechanics':mechanics,
        'scope':'Read-only source/evidence/publication/backlink joins. No new model/numerical run or semantic certification; original goal remains active.'}


if __name__=='__main__':print(json.dumps(execute(),indent=2,sort_keys=True))
