"""Read-only publication joins and local backlinks for the cycle68 synthesis."""
import json
from pathlib import Path
import re
from urllib.parse import unquote
from encoder_atoms import read,sha
from publish_prompt_program import audit


def execute():
    root=Path(__file__).resolve().parent.parent;docs=root/'docs';p=docs/'results/prompt-program';e=docs/'results/prompt-program-evidence'
    mechanics=audit(p);progress=read(docs/'zero-neural-progress.json');advisor=read(e/'advisor.json');plot=read(docs/'results/prompt-program-figure/figure.json')
    article=docs/'prompt-program.md';plan=docs/'plans/text-partition-feedback.md'
    blog=root.parent/'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md'
    if progress['researchCycle']!=68 or progress['status']!='active' or progress['liveJobs'] or progress['thisGoalTurn']['classification']!='progress':raise ValueError('Active goal/result status differs')
    if progress['findingsLog']['sha256']!=sha(blog) or progress['findingsLog']['lastUpdatedCycle']!=68:raise ValueError('Living findings log differs')
    if progress['nextCut']['status']!='written-unexecuted' or progress['nextCut']['planSHA256']!=sha(plan):raise ValueError('Next plan status/source differs')
    if advisor['reviewedArticleSHA256']!=sha(article) or advisor['reviewedPlanSHA256']!=sha(plan) or advisor['reviewedBlogSHA256']!=sha(blog):raise ValueError('Reviewed synthesis differs')
    if advisor['reportSHA256']!=sha(p/'report.json') or advisor['lastFullNewestPostSurveyCycle']!=67 or advisor['nextFullSurveyDueCycle']!=70:raise ValueError('Advisory evidence/survey scope differs')
    if plot['reportSHA256']!=sha(p/'report.json') or plot['plotSourceSHA256']!=sha(root/'generator/plot_prompt_program.py') or plot['imageSHA256']!=sha(docs/'results/prompt-program-figure/prompt-program.png'):raise ValueError('Plot source/evidence differs')
    text=blog.read_text();section=text.split('## Cycle 68:',1)[1].split('## What would close the original goal',1)[0]
    sources=[(article,article.read_text()),(plan,plan.read_text()),(e/'README.md',(e/'README.md').read_text()),(blog,section)]
    links=[]
    for source,body in sources:
        for target in re.findall(r'\]\(([^)]+)\)',body):
            if target.startswith(('https:','http:','#')):continue
            path=unquote(target.split('#',1)[0]).strip('<>');resolved=(source.parent/path).resolve()
            if not resolved.exists():raise ValueError('Broken local link: '+str(source)+' → '+target)
            links.append({'source':str(source.relative_to(root) if source.is_relative_to(root) else source),'target':target})
    for role in ['candidate','native']:
        src=p/(role+'-A.png');dst=root.parent/'obsidian/images'/('zero-neural-cycle68-'+role+'-A.png')
        if sha(src)!=sha(dst):raise ValueError('Notebook image copy differs')
    return {'schema':'tce-prompt-program-publication-qa-v1','localLinksChecked':len(links),'articleSHA256':sha(article),'nextPlanSHA256':sha(plan),'livingBlogSHA256':sha(blog),
        'mechanics':mechanics,'scope':'Read-only source/evidence/publication/backlink joins. No new numerical/model run or semantic certification; original goal remains active.'}


if __name__=='__main__':print(json.dumps(execute(),indent=2,sort_keys=True))
