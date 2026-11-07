"""Bind source-reader execution, publication, review, living log and storage receipts."""
import json
from pathlib import Path
import re
from encoder_atoms import read,save,sha
from image_dual_analysis import NAMES
from publish_source_reader import audit


def require(condition,message):
    if not condition:raise ValueError(message)


def execute():
    from mrun.client.api import Api
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/source-reader';e=root/'docs/results/source-reader-evidence'
    mechanics=audit(p);require(mechanics==read(e/'mechanics.json'),'Mechanics snapshot differs')
    r,s,report,actor,summary,advice,progress=[read(x) for x in [p/'request.json',p/'inputs.json',p/'report.json',p/'actor.json',
        e/'scalar-summary.json',e/'advisor.json',root/'docs/zero-neural-progress.json']]
    require((progress['researchCycle'],progress['status'],progress['liveJobs'])==(63,'active',[]),'Active goal/cycle differs')
    require(progress['lastCompletedResearchTurn']['commit']=='29fd91cd4524e029e0b965cdd4a04aa3a9f03c91'
        and progress['lastCompletedResearchTurn']['cycle']==62
        and all(progress[n]['classification']=='progress' for n in ['previousGoalTurn','thisGoalTurn']),'Previous/current progress differs')
    current=progress['sourceReaderEvidence']
    require(current['reportSHA256']==summary['reportSHA256']==sha(p/'report.json') and current['nextQ']==report['scores']['nextQ']
        and current['diagnostics']==report['diagnostics'] and current['scalarSummarySHA256']==sha(e/'scalar-summary.json'),'Progress/result differs')
    old=root/'docs/results/deferred-value';old_actor=read(old/'actor.json');targets=read(p/'targets.json')
    for n,x in s['sourceReaderParentRecords'].items():require(sha(old/n)==x['sha256'],'Assisted parent record differs:'+n)
    old_fields={**old_actor['source']['fields'],**old_actor['next_query']['fields']}
    require(targets['assisted']=={n:old_fields[n] for n in NAMES},'Assisted actual consumer origin differs')
    require(all(targets[n]==read(old/'targets.json')[n] for n in ['PP','QQ','stale','retained']),'Historical targets differ')
    require(sha(root/'docs/plans/source-input-deferred-reader.md')==sha(p/'frozen-plan.md')==r['sourceHashes']['frozen-plan.md']
        ==advice['executedPlanSHA256'],'Executed plan differs')
    for key,n in [('reportSHA256','report.json'),('actorSealSHA256','actor-seal.json'),('sealedActorSHA256','actor.json'),('actorExecutionSHA256','actor-execution.json')]:
        require(advice[key]==sha(p/n),'Advisor result differs:'+n)
    require(advice['articleSHA256']==sha(root/'docs/source-state-reader.md'),'Advisor article binding differs')
    for n,h in advice['reviewedSources'].items():require(sha(root/'generator'/n)==h==r['sourceHashes'][n],'Reviewed/staged source differs:'+n)
    tests=read(e/'tests.json');require((tests['passed'],tests['seconds'],tests['exitCode'])==(301,18.10,0)
        and tests['sourceSHA256']==advice['reviewedTestSourceSHA256']==sha(root/tests['source']),'Tests/source receipt differs')
    instrument=read(e/'saturn-instrument.json');watch=instrument['run']['steps']['watch'];xref=instrument['run']['steps']['xref']
    require((watch['records'],watch['per_stream'][0]['predicates'])==(37308,38) and instrument['runExit']==instrument['verifyExit']==0
        and instrument['run']['steps']['claims']['receipts_found']==xref['distinct_addresses']==0,'Instrument scope differs')
    job=read(p/'scheduler.json');live=Api().json('GET','/api/jobs/'+job['job_id'])
    require(job['job_id']=='job-b064ad1f4b15' and live['state']=='succeeded' and live['payload_custody']==job['payload_custody']
        and job['result']['elapsed_s']==41.11,'Terminal job/custody differs')
    require(report['totalNewMatrixMACs']==15495856128 and report['distinctBF16ParameterBytes']==301990912
        and summary['newArrays']=={'count':47,'NPYBytes':66305408},'Work/array account differs')
    require(summary['headPanels']['nextQ']['assisted']=={'improve':92,'worsen':4,'equal':0}
        and summary['headPanels']['nextQ']['retained']=={'improve':90,'worsen':6,'equal':0},'Head countertrends differ')
    figure=root/'docs/results/source-reader-figure/source-reader.png';fr=read(figure.parent/'figure.json')
    require(sha(figure)==fr['imageSHA256']==sha(root.parent/'obsidian/images/zero-neural-cycle63-source-reader.png')
        and fr['reportSHA256']==sha(p/'report.json') and fr['plotSourceSHA256']==sha(root/'generator/plot_source_reader.py'),'Figure bindings differ')
    future=progress['nextCut'];plan=root/'docs'/future['plan']
    require(future['status']==advice['nextCut']['status']=='written-unexecuted'
        and future['planSHA256']==advice['nextCut']['sha256']==sha(plan),'Next plan binding differs')
    blog=root/progress['findingsLog']['path'];require(progress['findingsLog']['lastUpdatedCycle']==63
        and progress['findingsLog']['sha256']==sha(blog),'Living log binding differs')
    section=blog.read_text().split('## Cycle 63:',1)[1].split('## What would close the original goal',1)[0];links=0
    for path,text in [(blog,section),(root/'docs/source-state-reader.md',None),(e/'README.md',None),(plan,None)]:
        for link in re.findall(r'\]\(([^)]+)\)',text if text is not None else path.read_text()):
            if not link.startswith(('http:','https:')):
                target=(path.parent/link.split('#',1)[0]).resolve();require(target==(e/'publication-qa.json').resolve() or target.exists(),'Missing local backlink:'+link);links+=1
    sm,reviews=progress['storageMaintenance'],progress['reviews']
    require((sm['lastActualArchiveCycle'],sm['lastAuditCycle'],sm['nextDueByCycle'],reviews['lastCycle'],reviews['nextDueByCycle'],reviews['latestActualResultCycle'])
        ==(63,63,66,62,65,63),'Storage/advisor cadence differs')
    require(reviews['latestAdvisorSHA256']==sha(e/'advisor.json'),'Latest review receipt differs')
    for kind in ['cleanup','audit']:
        d=root/'docs/results'/('storage-'+kind+'-cycle63');publication=read(d/'publication.json')
        require(publication['archiverSourceSHA256']==sha(root/'generator/storage_transports.py'),'Storage source differs')
        for n,h in publication['files'].items():require(sha(d/n)==h==sha(root/publication['source']/n),'Storage copy differs')
    cleanup=read(root/'docs/results/storage-cleanup-cycle63/result.json')
    require((cleanup['files'],cleanup['archivedBytes'])==(4,1346226331),'Actual cleanup differs')
    rejection=read(p/'prelease-note-rejection.json');require(rejection['HTTPStatus']==422 and not rejection['jobCreated']
        and rejection['replacementLauncherSHA256']==sha(root/'generator/source_reader_submit_v2.py'),'Note-only launch correction differs')
    result={**mechanics,'schema':'tce-source-reader-publication-qa-v1','reportSHA256':sha(p/'report.json'),
        'progressSHA256':sha(root/'docs/zero-neural-progress.json'),'blogSHA256':sha(blog),'figureSHA256':sha(figure),
        'advisorSHA256':sha(e/'advisor.json'),'nextPlanSHA256':sha(plan),'auditSourceSHA256':sha(__file__),'testsPassed':301,
        'localLinksChecked':links,'historicalTrajectoryJoins':['PP','QQ','stale','retained','assisted'],'currentTerminalJob':job['job_id'],
        'nextStorageCycle':66,'nextNewestPostReviewCycle':65,
        'scope':'Named source/byte/typed-own-state/publication/narrative mechanics; no effect-size, image/generalization/speedup/arbitrary-prompt zero-neural certification.'}
    save(e/'publication-qa.json',result)
    return {k:result[k] for k in ['ownedSourcePins','typedActs','explicitOutputABIs','ownStateJoins','observations','fullByteFiles','NPYPayloadWitnesses','localLinksChecked','nextStorageCycle','nextNewestPostReviewCycle']}


if __name__=='__main__':print(json.dumps(execute(),indent=2))
