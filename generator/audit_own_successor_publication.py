"""Audit publication bookkeeping using exact sources, bytes and public scalars."""
import json
from pathlib import Path
import re
from encoder_atoms import read,save,sha
from own_successor_study import OWNED
from publish_own_successor import audit


def execute():
    from mrun.client.api import Api
    root=Path(__file__).resolve().parent.parent
    public=root/'docs/results/own-successor'
    evidence=root/'docs/results/own-successor-evidence'
    mechanics=audit(public)
    progress=read(root/'docs/zero-neural-progress.json')
    saved=read(public/'scheduler.json')
    actual=Api().json('GET','/api/jobs/'+saved['job_id'])
    if actual['state']!='succeeded' or actual['result']['returncode']!=0 or actual['payload_custody']!=saved['payload_custody']:
        raise ValueError('Authoritative job/custody differs')
    custody=actual['payload_custody']
    if not custody['required'] or len({custody[k]['sha256'] for k in ['declared','sealed','executed']})!=1 or custody['executed']['size_bytes']!=3189925:
        raise ValueError('Guarded executed payload differs')
    if {k:actual['reservation'][k] for k in ['ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s']}!={'ram_mb':6144,'vram_mb':4096,'cpu_threads':4,'disk_gb':4,'est_wall_s':120} or actual['timeout_s']!=360:
        raise ValueError('Declared resource exception differs')
    if (actual['result']['elapsed_s'],actual['result']['peak_rss_mb'],actual['result']['peak_vram_mb'])!=(53.43,2717.2,2554.0):
        raise ValueError('Observed execution scope differs')
    if progress['status']!='active' or progress['researchCycle']!=87 or progress['liveJobs'] or progress['inFlightResearchCycle87']['status']!='completed' or not progress['inFlightResearchCycle87']['noLiveJob']:
        raise ValueError('Goal/current cycle differs')
    cut=progress['nextCut'];plan=root/'docs'/cut['plan']
    if cut['status']!='written-unexecuted' or not cut['noImplementation'] or not cut['noJobSubmitted'] or sha(plan)!=cut['planSHA256']:
        raise ValueError('Next plan/execution boundary differs')
    tests=read(evidence/'tests.json')
    if tests['passed']!=362 or tests['exitCode']!=0 or len(tests['sourceHashes'])!=24:
        raise ValueError('Actual tiny tests missing')
    for name in OWNED:
        if sha(root/'generator'/name)!=tests['sourceHashes'][name]:
            raise ValueError('Tested executed source changed:'+name)
    for name,original in [('frozen-plan.md','receiver-own-state-successor.md'),('implementation-notes.md','own-successor-implementation.md')]:
        if sha(public/name)!=sha(root/'docs/plans'/original) or sha(public/name)!=sha(root/'artifacts/own-successor-v1/payload'/name):
            raise ValueError('Frozen historical contract differs')
    pre=read(evidence/'advisor-prestage.json')
    for path,digest in pre['sourceHashes'].items():
        if sha(root/path)!=digest:raise ValueError('Exact pre-stage reviewed member changed:'+path)
    for name in ['advisor-result.json','advisor-synthesis.json']:
        review=read(evidence/name)
        for path,digest in review['reviewedSHA256'].items():
            if sha(root/path)!=digest:raise ValueError('Reviewed result/synthesis changed:'+path)
    synthesis=read(evidence/'advisor-synthesis.json')
    if synthesis['decision']!='approved-exact-snapshots' or len(synthesis['reviewedSHA256'])!=4:
        raise ValueError('Final exact synthesis review missing')
    blog=root/progress['findingsLog']['path']
    if progress['findingsLog']['lastUpdatedCycle']!=87 or sha(blog)!=progress['findingsLog']['sha256']:
        raise ValueError('Living log identity differs')
    links=[]
    for path in [root/'docs/own-state-successor.md',plan,evidence/'README.md',blog]:
        text=path.read_text()
        if path==blog:text=text.split('## Cycle 87:',1)[1].split('## What would close the original goal',1)[0]
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('http:','https:','#')):continue
            resolved=(path.parent/target.split('#',1)[0]).resolve()
            if not resolved.exists() and resolved!=evidence/'publication-qa.json':raise ValueError('Broken local link:'+target)
            links.append(str(resolved))
    figdir=root/'docs/results/own-successor-figure';figure=read(figdir/'figure.json')
    if sha(figdir/'own-successor.png')!=figure['imageSHA256'] or sha(root.parent/'obsidian/blog/images/own-successor-cycle87.png')!=figure['imageSHA256']:
        raise ValueError('Figure copy differs')
    if sha(root/'generator/plot_own_successor.py')!=figure['plotSourceSHA256'] or sha(public/'report.json')!=figure['sourceReportSHA256'] or read(evidence/'render-qa.json')['imageSHA256']!=figure['imageSHA256']:
        raise ValueError('Inspected figure/source differs')
    for action in ['run','verify']:
        if read(evidence/('saturn-'+action+'.json'))['returncode']!=0:raise ValueError('Actual Saturn instrument code differs')
        value=json.loads((evidence/('saturn-'+action+'-stdout.txt')).read_text())
        if Path(value['target'])!=root/'artifacts/own-successor-v1/result' or value['instrument_status']!='target-streams-only':
            raise ValueError('Applied raw target differs')
    if read(evidence/'saturn-instrument.json')['records']!=1536:raise ValueError('Instrument row scope differs')
    storage=read(evidence/'storage.json')
    if (storage['newArrayCount'],storage['newArrayFileBytes'],storage['newArchivedFiles'],storage['newDeletedFiles'])!=(52,648557056,0,0):
        raise ValueError('New retention differs')
    if progress['reviews']['lastFullNewestPostSurveyCycle']!=87 or progress['reviews']['nextDueByCycle']!=90 or progress['storageMaintenance']['lastReviewCycle']!=87 or progress['storageMaintenance']['lastActualArchiveCycle']!=78 or progress['storageMaintenance']['nextDueByCycle']!=90:
        raise ValueError('Scheduled review/archive boundaries differ')
    survey=read(root/'docs/results/research-review-cycle87/advisor-survey.json')
    historical=survey['reviewedWorkingSnapshots']['obsidian/blog/'+blog.name]
    if historical!='4f5c664ed0f59559709087f5c4a242aaa4dd17c32f91fee0f23bb6b897155d0d' or historical==sha(blog):
        raise ValueError('Historical survey/final log distinction lost')
    result={'schema':'tce-own-successor-publication-QA-v1','jobID':actual['job_id'],'authoritativeState':actual['state'],
            'mechanics':mechanics,'tests':362,'verifiedLocalLinks':len(links),'reviewedSynthesisFiles':4,
            'goalStatus':'active','nextCut':'written-unexecuted','surveyedHistoricalBlogSHA256':historical,
            'blogSHA256':sha(blog),'figureSHA256':figure['imageSHA256'],'auditorSourceSHA256':sha(__file__),
            'scope':'Source/custody/resources/byte/scalar/narrative/figure/links/reviews/storage/goal bookkeeping. No real-array numerical replay,native propagation attribution,compact closure,image/generalization/zero-neural certification.'}
    save(evidence/'publication-qa.json',result)
    if not all(Path(path).exists() for path in links):raise ValueError('Final linked publication member absent')
    return result


if __name__=='__main__':print(json.dumps(execute(),indent=2))
