"""Final publication/goal bookkeeping checks on byte-verified scalar evidence."""
import json
from pathlib import Path
import re
from encoder_atoms import read,save,sha
from publish_gate_up_basis import audit


def execute():
    from mrun.client.api import Api
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/current-state-gate-up';e=root/'docs/results/current-state-gate-up-evidence'
    mechanics=audit(p);progress=read(root/'docs/zero-neural-progress.json');report=read(p/'report.json');scheduler=read(p/'scheduler.json');tests=read(e/'tests.json')
    actual=Api().json('GET','/api/jobs/'+scheduler['job_id'])
    if actual['state']!='succeeded' or actual['result']['returncode']!=0 or actual['payload_custody']!=scheduler['payload_custody']:raise ValueError('Authoritative completed job differs')
    if progress['status']!='active' or progress['researchCycle']!=82 or progress['liveJobs']:raise ValueError('Goal scope/live state differs')
    if progress['nextCut']['status']!='written-unexecuted' or not progress['nextCut']['noJobSubmitted'] or not progress['nextCut']['noImplementation']:raise ValueError('Next plan execution state differs')
    plan=root/'docs'/progress['nextCut']['plan']
    if sha(plan)!=progress['nextCut']['planSHA256'] or report['newImages']!=0 or report['newContinuations']!=0:raise ValueError('Plan/image boundary differs')
    if tests['passed']!=351 or tests['exitCode']!=0:raise ValueError('Actual mechanics suite missing')
    for name,digest in tests['sourceHashes'].items():
        if sha(root/'generator'/name)!=digest:raise ValueError('Tested source changed:'+name)
    for name in ['frozen-plan.md','implementation-notes.md']:
        if sha(root/'artifacts/current-state-gate-up-v1/payload'/name)!=sha(p/name):raise ValueError('Executed input copy changed')
    if sha(root/'docs/plans/current-state-gate-up-implementation.md')!=sha(p/'implementation-notes.md'):raise ValueError('Source implementation note differs from frozen input')
    advisor=read(e/'advisor-synthesis.json')
    if advisor['decision']!='approved-exact-snapshots':raise ValueError('Final synthesis review missing')
    for name,digest in advisor['reviewedSHA256'].items():
        if sha(root/name)!=digest:raise ValueError('Reviewed synthesis changed:'+name)
    blog=root/progress['findingsLog']['path']
    if sha(blog)!=progress['findingsLog']['sha256'] or progress['findingsLog']['lastUpdatedCycle']!=82:raise ValueError('Living log identity differs')
    paths=[root/'docs/current-state-gate-up.md',plan,e/'README.md'];links=[]
    for path in paths+[blog]:
        content=path.read_text()
        if path==blog:content=content.split('## Cycle 82:',1)[1].split('## What would close the original goal',1)[0]
        for target in re.findall(r'\]\(([^)]+)\)',content):
            if target.startswith(('http:','https:','#')):continue
            file=target.split('#',1)[0]
            if not (path.parent/file).resolve().exists():raise ValueError('Broken publication link:'+target)
            links.append(str((path.parent/file).resolve()))
    fig=root/'docs/results/current-state-gate-up-figure';fr=read(fig/'figure.json')
    if sha(fig/'gate-up-basis.png')!=fr['imageSHA256'] or sha(root.parent/'obsidian/blog/images/current-state-gate-up-cycle82.png')!=fr['imageSHA256'] or sha(root/'generator/plot_gate_up_basis.py')!=fr['plotSourceSHA256']:raise ValueError('Inspected scalar figure identity differs')
    instrument=read(e/'saturn-instrument.json')
    if instrument['correctedRawResultRunReturnCode']!=0 or instrument['correctedRawResultVerifyReturnCode']!=0 or instrument['records']!=1536:raise ValueError('Applied Saturn scope differs')
    if read(e/'saturn-run-v2.json')['returncode']!=0 or read(e/'saturn-verify-v2.json')['returncode']!=0:raise ValueError('Actual instrument receipts differ')
    storage=read(e/'storage.json')
    if storage['newArrayCount']!=22 or storage['newArrayFileBytes']!=468652800 or storage['newArchivedFiles']!=0 or progress['storageMaintenance']['lastActualArchiveCycle']!=78 or progress['reviews']['lastFullNewestPostSurveyCycle']!=81:raise ValueError('Storage/survey scope differs')
    result={'schema':'tce-current-state-gate-up-publication-QA-v1','jobID':actual['job_id'],'authoritativeState':actual['state'],'mechanics':mechanics,
        'tests':351,'verifiedLocalLinks':len(links),'reviewedSynthesisFiles':len(advisor['reviewedSHA256']),'goalStatus':'active','nextCut':'written-unexecuted',
        'figureSHA256':fr['imageSHA256'],'blogSHA256':sha(blog),'auditorSourceSHA256':sha(__file__),
        'scope':'Actual source/core publisher/test/custody/typed/work/row/bookkeeping/links/figure/applied-instrument/survey/storage checks. No model, quality threshold, independent real-tensor replay, runtime/trajectory/generalization or zero-neural certification.'}
    save(e/'publication-qa.json',result);return result


if __name__=='__main__':print(json.dumps(execute(),indent=2))
