"""Audit publication identities and scalar bookkeeping without scientific tensor replay."""
import json
import math
from pathlib import Path
import re
from encoder_atoms import read,save,sha
from dense_corner_study import OWNED
from publish_dense_corner import audit


def execute():
    from mrun.client.api import Api
    root=Path(__file__).resolve().parent.parent
    public=root/'docs/results/dense-corner';evidence=root/'docs/results/dense-corner-evidence'
    mechanics=audit(public)
    if mechanics!=read(evidence/'mechanics.json'):raise ValueError('Recorded mechanics differs')
    progress=read(root/'docs/zero-neural-progress.json')
    saved=read(public/'scheduler.json');actual=Api().json('GET','/api/jobs/'+saved['job_id'])
    if actual['state']!='succeeded' or actual['result']['returncode']!=0 or actual['payload_custody']!=saved['payload_custody']:
        raise ValueError('Authoritative job/custody differs')
    custody=actual['payload_custody']
    if not custody['required'] or len({custody[k]['sha256'] for k in ['declared','sealed','executed']})!=1 or custody['executed']['size_bytes']!=3214973:
        raise ValueError('Guarded executed payload differs')
    if {k:actual['reservation'][k] for k in ['ram_mb','vram_mb','cpu_threads','disk_gb','est_wall_s']}!={'ram_mb':4096,'vram_mb':3072,'cpu_threads':4,'disk_gb':3,'est_wall_s':90} or actual['timeout_s']!=300:
        raise ValueError('Declared allocation-based resource exception differs')
    if (actual['result']['elapsed_s'],actual['result']['peak_rss_mb'],actual['result']['peak_vram_mb'])!=(23.9,2286.5,1164.0):
        raise ValueError('Observed execution scope differs')
    if progress['status']!='active' or progress['researchCycle']!=89 or progress['liveJobs'] or progress['inFlightResearchCycle89']['status']!='completed' or not progress['inFlightResearchCycle89']['noLiveJob']:
        raise ValueError('Active goal/completed cycle differs')
    cut=progress['nextCut'];plan=root/'docs'/cut['plan']
    if cut['status']!='written-unexecuted' or not cut['noImplementation'] or not cut['noJobSubmitted'] or sha(plan)!=cut['planSHA256'] or cut['candidateMACs']!=72277819392:
        raise ValueError('Next plan/execution boundary differs')
    if cut['fullReviewBeforeStagingCycle']!=90:raise ValueError('Scheduled full review prerequisite missing')
    tests=read(evidence/'tests.json')
    if tests['passed']!=368 or tests['exitCode']!=0 or len(tests['sourceHashes'])!=36 or len(OWNED)!=36:
        raise ValueError('Actual tiny tests/source census missing')
    for name in OWNED:
        if sha(root/'generator'/name)!=tests['sourceHashes'][name]:raise ValueError('Tested executed source changed:'+name)
    for name,original in [('frozen-plan.md','dense-own-state-corner.md'),('implementation-notes.md','dense-corner-implementation.md')]:
        if sha(public/name)!=sha(root/'docs/plans'/original) or sha(public/name)!=sha(root/'artifacts/dense-corner-v1/payload'/name):
            raise ValueError('Frozen historical contract differs')
    pre=read(evidence/'advisor-prestage.json')
    for path,digest in pre['sourceHashes'].items():
        if sha(root/path)!=digest:raise ValueError('Exact pre-stage reviewed member changed:'+path)
    for name in ['advisor-result.json','advisor-synthesis.json']:
        for path,digest in read(evidence/name)['reviewedSHA256'].items():
            if sha(root/path)!=digest:raise ValueError('Reviewed result/synthesis changed:'+path)
    synthesis=read(evidence/'advisor-synthesis.json')
    if synthesis['decision']!='approved-exact-snapshots' or len(synthesis['reviewedSHA256'])!=4:
        raise ValueError('Final four-file synthesis review missing')
    blog=root/progress['findingsLog']['path']
    if progress['findingsLog']['lastUpdatedCycle']!=89 or sha(blog)!=progress['findingsLog']['sha256']:
        raise ValueError('Living log identity differs')
    report=read(public/'report.json');supplement=read(evidence/'increment-normalized.json')
    if supplement['sourceReportSHA256']!=sha(public/'report.json') or supplement['summarizerSourceSHA256']!=sha(root/'generator/summarize_dense_corner.py'):
        raise ValueError('Scalar supplement identity differs')
    for name,panel in report['panels'].items():
        scalar=supplement['panels'][name];a=panel['signedAccount']
        for centered,prefix in [(False,''),(True,'centered')]:
            energy='centeredEnergy' if centered else 'energy'
            denominator='centeredCandidateEnergy' if centered else 'candidateEnergy'
            target='centeredTargetEnergy' if centered else 'targetEnergy'
            key=prefix+'E1OverActualDenseOwnIncrementRMS'
            if scalar[key]!=math.sqrt(a['E1_'+energy]/panel['denseIncrement']['energies'][denominator]):
                raise ValueError('Actual own-increment denominator differs')
            key=prefix+'E0OverDenseParentIncrementRMS'
            if scalar[key]!=math.sqrt(a['E0_'+energy]/panel['ownIncrement']['energies'][target]):
                raise ValueError('Parent-increment denominator differs')
    links=[]
    for path in [root/'docs/dense-own-state-corner.md',plan,evidence/'README.md',blog]:
        content=path.read_text()
        if path==blog:content=content.split('## Cycle 89:',1)[1].split('## What would close the original goal',1)[0]
        for target in re.findall(r'\]\(([^)]+)\)',content):
            if target.startswith(('http:','https:','#')):continue
            resolved=(path.parent/target.split('#',1)[0]).resolve()
            if not resolved.exists() and resolved!=evidence/'publication-qa.json':raise ValueError('Broken local link:'+target)
            links.append(str(resolved))
    figdir=root/'docs/results/dense-corner-figure';figure=read(figdir/'figure.json')
    if sha(figdir/'dense-corner.png')!=figure['imageSHA256'] or sha(root.parent/'obsidian/blog/images/dense-corner-cycle89.png')!=figure['imageSHA256']:
        raise ValueError('Figure/notebook copy differs')
    if sha(root/'generator/plot_dense_corner.py')!=figure['plotSourceSHA256'] or sha(public/'report.json')!=figure['sourceReportSHA256'] or sha(evidence/'increment-normalized.json')!=figure['supplementSHA256'] or read(evidence/'render-qa.json')['imageSHA256']!=figure['imageSHA256']:
        raise ValueError('Inspected figure/source differs')
    for action in ['run','verify']:
        if read(evidence/('saturn-'+action+'.json'))['returncode']!=0:raise ValueError('Applied Saturn instrument differs')
        value=json.loads((evidence/('saturn-'+action+'-stdout.txt')).read_text())
        if Path(value['target'])!=root/'artifacts/dense-corner-v1/result' or value['instrument_status']!='target-streams-only':
            raise ValueError('Actual raw target scope differs')
    instrument=read(evidence/'saturn-instrument.json')
    if [instrument[k] for k in ['records','fires','formalClaims','qualifiedAddresses','expectedMissingNoFireLedgers']]!=[1536,0,0,0,1]:
        raise ValueError('Instrument row scope differs')
    storage=read(evidence/'storage.json')
    if [storage[k] for k in ['newArrayCount','newArrayFileBytes','compactCoreFiles','compactCoreBytes','newArchivedFiles','newDeletedFiles']]!=[32,503019520,18,1974956,0,0]:
        raise ValueError('New retention scope differs')
    if progress['reviews']['lastFullNewestPostSurveyCycle']!=87 or progress['reviews']['nextDueByCycle']!=90 or progress['storageMaintenance']['lastReviewCycle']!=87 or progress['storageMaintenance']['lastActualArchiveCycle']!=78 or progress['storageMaintenance']['nextDueByCycle']!=90:
        raise ValueError('Scheduled review/archive boundaries differ')
    survey=read(root/'docs/results/research-review-cycle87/advisor-survey.json')
    historical=survey['reviewedWorkingSnapshots']['obsidian/blog/'+blog.name]
    if historical!='4f5c664ed0f59559709087f5c4a242aaa4dd17c32f91fee0f23bb6b897155d0d' or historical==sha(blog):
        raise ValueError('Historical survey/current log distinction lost')
    result={'schema':'tce-dense-corner-publication-QA-v1','jobID':actual['job_id'],'authoritativeState':actual['state'],
        'mechanics':mechanics,'tests':368,'verifiedLocalLinks':len(links),'reviewedSynthesisFiles':4,
        'goalStatus':'active','nextCut':'written-unexecuted','fullReviewBeforeNextStage':90,'surveyedHistoricalBlogSHA256':historical,
        'blogSHA256':sha(blog),'figureSHA256':figure['imageSHA256'],'auditorSourceSHA256':sha(__file__),
        'scope':'Actual custody/resources/frozen sources/contracts/intake/byte/scalar/figure/narrative/review/storage/goal bookkeeping. No independent real-array replay,original full-attention sensitivity,compact closure,image/generalization/zero-neural certification.'}
    save(evidence/'publication-qa.json',result)
    if not all(Path(path).exists() for path in links):raise ValueError('Final linked member absent')
    return result


if __name__=='__main__':print(json.dumps(execute(),indent=2))
