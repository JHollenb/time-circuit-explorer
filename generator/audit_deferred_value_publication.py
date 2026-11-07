"""Verify named publication joins and active-goal narrative, without model work."""
import json
from pathlib import Path
import re
from encoder_atoms import read, save, sha
from image_dual_analysis import NAMES
from publish_deferred_value import audit


def require(condition,message):
    if not condition:raise ValueError(message)


def execute():
    from mrun.client.api import Api
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/deferred-value';e=root/'docs/results/deferred-value-evidence'
    mechanics=audit(p);require(mechanics==read(e/'mechanics.json'),'Mechanics snapshot differs')
    r,s,report,actor,summary,advice,progress=[read(x) for x in [p/'request.json',p/'inputs.json',p/'report.json',p/'actor.json',
        e/'scalar-summary.json',e/'advisor.json',root/'docs/zero-neural-progress.json']]
    require((progress['researchCycle'],progress['status'],progress['liveJobs'])==(62,'active',[]),'Active goal/cycle differs')
    require(progress['lastCompletedResearchTurn']['commit']=='6981dc01c52aaa6bd2fa77e06abbcc42382b6549'
        and progress['lastCompletedResearchTurn']['cycle']==61
        and all(progress[n]['classification']=='progress' for n in ['previousGoalTurn','thisGoalTurn']),
        'Previous/publishing progress differs')
    current=progress['deferredValueEvidence']
    require(current['reportSHA256']==summary['reportSHA256']==sha(p/'report.json')
        and current['nextQ']==report['scores']['nextQ'] and current['diagnostics']==report['diagnostics']
        and current['matchedControlPayloads']==report['matchedControlPayloads']
        and current['scalarSummarySHA256']==sha(e/'scalar-summary.json'),'Progress/scalar/result differs')
    parent=root/'docs/results/image-dual';old_actor=read(parent/'actor.json')
    for n,x in s['deferredParentRecords'].items():require(sha(parent/n)==x['sha256'],'Historical record differs:'+n)
    for n in s['matchedControlFields']:
        require(s['matchedControlFields'][n]==old_actor['read']['fields'][n],'Allocation control origin differs')
    require(s['controlModLN']==old_actor['modln']['fields']['x'],'Source input control origin differs')
    targets=read(p/'targets.json');old_targets=read(parent/'targets.json')
    require(all(targets[n]==old_targets[n] for n in ['PP','QQ','stale']),'P/Q/stale origin differs')
    old_fields={**old_actor['source']['fields'],**old_actor['next_query']['fields']}
    require(targets['retained']=={n:old_fields[n] for n in NAMES},'Retained-V actual consumer origin differs')
    require(sha(root/'docs/plans/deferred-value-projection.md')==sha(p/'frozen-plan.md')==r['sourceHashes']['frozen-plan.md']
        ==advice['executedPlanSHA256'],'Original executed plan differs')
    require(advice['reportSHA256']==sha(p/'report.json') and advice['actorSealSHA256']==sha(p/'actor-seal.json')
        and advice['sealedActorSHA256']==sha(p/'actor.json') and advice['actorExecutionSHA256']==sha(p/'actor-execution.json')
        and advice['articleSHA256']==sha(root/'docs/deferred-value-projection.md'),'Advisor result/source/article binding differs')
    for n,h in advice['reviewedSources'].items():
        require(sha(root/'generator'/n)==h==r['sourceHashes'][n],'Reviewed/staged/executed source differs:'+n)
    tests=read(e/'tests.json')
    require((tests['passed'],tests['seconds'],tests['exitCode'])==(299,17.81,0)
        and tests['sourceSHA256']==advice['reviewedTestSourceSHA256']==sha(root/tests['source']),'Actual test/source receipt differs')
    instrument=read(e/'saturn-instrument.json')
    require((instrument['records'],instrument['predicates'])==(444,29)
        and all(instrument[n]==0 for n in ['runExit','verifyExit','formalReceipts','boundXrefAddresses']),
        'Instrument coverage/scope differs')
    job=read(p/'scheduler.json');live=Api().json('GET','/api/jobs/'+job['job_id'])
    require(job['job_id']=='job-c5be79476937' and live['state']=='succeeded'
        and live['payload_custody']==job['payload_custody'] and job['result']['elapsed_s']==41.23,
        'Current terminal job/custody/timing differs')
    require(report['totalNewMatrixMACs']==1874853888 and summary['valueMatrixMACs']==528482304
        and summary['newArrays']==37 and summary['newArrayBytes']==28489856,'Work/array account differs')
    figure=root/'docs/results/deferred-value-figure/deferred-value.png';fr=read(figure.parent/'figure.json')
    require(sha(figure)==fr['imageSHA256']==sha(root.parent/'obsidian/images/zero-neural-cycle62-deferred-value.png')
        and fr['reportSHA256']==sha(p/'report.json') and fr['plotSourceSHA256']==sha(root/'generator/plot_deferred_value.py'),
        'Figure/data/source/notebook copy differs')
    future=progress['nextCut'];plan=root/'docs'/future['plan']
    require(future['status']==advice['nextCut']['status']=='written-unexecuted'
        and future['planSHA256']==advice['nextCut']['sha256']==sha(plan)
        and future['totalNewMatrixMACs']==advice['nextCut']['actorMatrixMACs']==15495856128,
        'Next source-state interface/plan/work differs')
    blog=root/progress['findingsLog']['path']
    require(progress['findingsLog']['lastUpdatedCycle']==62 and progress['findingsLog']['sha256']==sha(blog),
        'Living blog binding differs')
    section=blog.read_text().split('## Cycle 62:',1)[1].split('## What would close the original goal',1)[0]
    links=0
    for path,contents in [(blog,section),(root/'docs/deferred-value-projection.md',None),(e/'README.md',None),(plan,None)]:
        for link in re.findall(r'\]\(([^)]+)\)',contents if contents is not None else path.read_text()):
            if not link.startswith(('http:','https:')):
                target=(path.parent/link.split('#',1)[0]).resolve()
                require(target==(e/'publication-qa.json').resolve() or target.exists(),'Local backlink missing:'+link);links+=1
    sm,reviews=progress['storageMaintenance'],progress['reviews']
    require((sm['lastActualArchiveCycle'],sm['lastAuditCycle'],sm['nextDueByCycle'],reviews['lastCycle'],
        reviews['nextDueByCycle'],reviews['latestActualResultCycle'])==(60,60,63,62,65,62),'Storage/advisor cadence differs')
    require(reviews['latestAdvisorSHA256']==sha(e/'advisor.json') and reviews['cycle62Snapshots']==advice['newestPostSnapshots'],
        'Scheduled newest-post/source/result snapshot binding differs')
    result={**mechanics,'schema':'tce-deferred-value-publication-qa-v1','reportSHA256':sha(p/'report.json'),
        'progressSHA256':sha(root/'docs/zero-neural-progress.json'),'blogSHA256':sha(blog),'figureSHA256':sha(figure),
        'advisorSHA256':sha(e/'advisor.json'),'nextPlanSHA256':sha(plan),'auditSourceSHA256':sha(__file__),
        'testsPassed':299,'localLinksChecked':links,'historicalTrajectoryJoins':['PP','QQ','stale','retained'],
        'currentTerminalJob':job['job_id'],'nextStorageCycle':63,'nextNewestPostReviewCycle':65,
        'mutableAdvisorPostObservations':[{**x,'publicationCurrentSHA256':sha((e/x['path']).resolve())} for x in advice['newestPostSnapshots']],
        'scope':'Named publication/source/bytes/typed-own-state/narrative mechanics; no effect-size gate or semantic/image/generalization/speedup/arbitrary-prompt zero-neural certification.'}
    save(e/'publication-qa.json',result)
    return {k:result[k] for k in ['ownedSourcePins','typedActs','explicitOutputABIs','ownStateJoins','observations',
        'fullByteFiles','NPYPayloadWitnesses','localLinksChecked','nextStorageCycle','nextNewestPostReviewCycle']}


if __name__=='__main__':print(json.dumps(execute(),indent=2))
