"""Read-only joins of scientific evidence, archive scopes, reviews and prose."""
import json
from pathlib import Path
import re
from urllib.parse import unquote
from encoder_atoms import read,sha
from publish_odd_gate import audit


def execute():
    root=Path(__file__).resolve().parent.parent;docs=root/'docs';p=docs/'results/odd-gate';e=docs/'results/odd-gate-evidence'
    mechanics=audit(p);progress=read(docs/'zero-neural-progress.json');advisor=read(e/'advisor.json');article=docs/'odd-gate.md';plan=docs/'plans/gated-even-operator-bound.md'
    blog=root.parent/'obsidian/blog/2026-11-06-004204-toward-arbitrary-prompt-images-without-neural-evaluations.md';body=blog.read_text()
    if progress['researchCycle']!=78 or progress['status']!='active' or progress['liveJobs'] or progress['thisGoalTurn']['classification']!='progress':raise ValueError('Current goal/result differs')
    if progress['findingsLog']['lastUpdatedCycle']!=78 or progress['findingsLog']['sha256']!=sha(blog) or progress['nextCut']['status']!='written-unexecuted' or progress['nextCut']['planSHA256']!=sha(plan):raise ValueError('Livinglog/nextplan differs')
    for n,h in advisor['finalReviewedSynthesisSnapshots'].items():
        if sha(root/n)!=h:raise ValueError('Final reviewed synthesis differs:'+n)
    for n,h in advisor['reviewedResultSnapshots'].items():
        if sha(p/n)!=h:raise ValueError('Actual reviewed result differs:'+n)
    survey=docs/'results/research-review-cycle78/advisor-survey.json'
    if not advisor['newFullNewestPostSurvey'] or advisor['lastFullNewestPostSurveyCycle']!=78 or advisor['nextFullSurveyDueCycle']!=81 or advisor['fullSurveySHA256']!=sha(survey):raise ValueError('Periodic review scope differs')
    r=read(p/'request.json');tests=read(e/'tests.json');inputs=read(p/'inputs.json')
    if tests['passed']!=342 or tests['sourceHashes']!={n:r['sourceHashes'][n] for n in r['ownedSources']}:raise ValueError('Test/source snapshot differs')
    for n,h in inputs['preStageReviewSnapshots'].items():
        if sha(root/n)!=h:raise ValueError('Due pre-stage review snapshot differs')
    figure=read(docs/'results/odd-gate-figure-v2/figure.json');render=read(e/'render-qa.json')
    if figure['plotSourceSHA256']!=sha(root/'generator/plot_odd_gate_v2.py') or figure['imageSHA256']!=sha(docs/'results/odd-gate-figure-v2/odd-gate.png') or figure['imageSHA256']!=sha(Path(figure['notebookCopy'])) or render['imageSHA256']!=figure['imageSHA256'] or render['supersededImageSHA256']!=sha(docs/'results/odd-gate-figure/odd-gate.png'):raise ValueError('Render/copy/original finding differs')
    cleanup=docs/'results/storage-cleanup-cycle78';cp=read(cleanup/'publication.json');result=read(cleanup/'result.json');storage=docs/'results/storage-audit-cycle78';sp=read(storage/'publication.json')
    for directory,publication in [(cleanup,cp),(storage,sp)]:
        for n,h in publication['files'].items():
            if sha(directory/n)!=h:raise ValueError('Storage publication differs')
    if result['files']!=9 or result['archivedBytes']!=3806240603 or not result['remoteDurabilityVerifiedBeforeUnlink'] or result['executorSHA256']!=sha(root/'generator/storage_local_snapshots_v2.py') or sp['totalRestorePointers']!=93 or sp['olderTerminalPointers']!=84 or sp['newCurrentByteSnapshotPointers']!=9 or sp['remaining100MBCandidates']!=0:raise ValueError('Actual archival scope differs')
    if any(Path(x['path']).exists() or read(Path(x['path']+'.cold.json'))!=x for x in result['rows']):raise ValueError('Current-byte snapshot restore pointer differs')
    count=0
    sources=[(article,article.read_text()),(plan,plan.read_text()),(e/'README.md',(e/'README.md').read_text()),(cleanup/'README.md',(cleanup/'README.md').read_text()),(blog,body.split('## Cycle 78:',1)[1].split('## What would close the original goal',1)[0])]
    for source,text in sources:
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('https:','http:','#')):continue
            path=unquote(target.split('#',1)[0]).strip('<>')
            if not (source.parent/path).resolve().exists():raise ValueError('Broken local link:'+str(source)+' → '+target)
            count+=1
    for section,n in [('source_docs',6),('source_code',13),('source_artifacts',7)]:
        entries=body.split(section+':\n',1)[1].splitlines()[:n]
        if len(entries)!=n or any(not x.startswith('  - ') or not (blog.parent/x[4:]).resolve().exists() for x in entries):raise ValueError('New frontmatter link differs')
        count+=n
    return {'schema':'tce-odd-gate-publication-qa-v1','status':'passed','localLinksChecked':count,'articleSHA256':sha(article),'nextPlanSHA256':sha(plan),'livingBlogSHA256':sha(blog),
        'auditSourceSHA256':sha(__file__),'mechanics':mechanics,'archiveScope':{'currentByteSnapshots':9,'logicalArchivedBytes':3806240603,'totalRestorePointers':93,'olderPointersNotFreshlyFullRehashed':84},
        'scope':'Executed source/evidence/archiveclass/currentstatus/reviews/render/backlink joins only. No new real-array/modelrun or semantic/generalization/zero-neural certification; goal active.'}


if __name__=='__main__':print(json.dumps(execute(),indent=2,sort_keys=True))
