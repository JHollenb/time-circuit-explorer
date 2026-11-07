"""Check the named publication joins; no model or saved tensor execution."""
import json
from pathlib import Path
import re
from encoder_atoms import identity, read, save, sha
from publish_spatial_norm import audit


def require(value, message):
    if not value:
        raise ValueError(message)


def execute():
    from mrun.client.api import Api
    root=Path(__file__).resolve().parent.parent
    p=root/'docs/results/spatial-norm-capacity';e=root/'docs/results/spatial-norm-evidence'
    mechanics=audit(p)
    require(mechanics==read(e/'mechanics.json'),'Mechanics snapshot differs')
    request,inputs,report,intake,candidate,advice,summary,progress=[read(x) for x in [p/'request.json',p/'inputs.json',
        p/'report.json',p/'intake.json',p/'candidate.json',e/'advisor.json',e/'scalar-summary.json',root/'docs/zero-neural-progress.json']]
    require((progress['researchCycle'],progress['status'],progress['liveJobs'])==(61,'active',[]),'Active goal state differs')
    require(progress['lastCompletedResearchTurn']['commit']=='24aca52ae18d623f05d8e9a0deaeb24c4df38107'
        and progress['lastCompletedResearchTurn']['cycle']==60
        and all(progress[n]['classification']=='progress' for n in ['previousGoalTurn','thisGoalTurn']),
        'Previous/publishing goal progress differs')
    current=progress['spatialNormEvidence']
    require(current['reportSHA256']==summary['reportSHA256']==sha(p/'report.json')
        and current['norms']==summary['norms']
        and current['norms']=={k:v for k,v in report['norms'].items() if k!='heads'}
        and current['scalarSummarySHA256']==sha(e/'scalar-summary.json'),'Progress/scalar/actual result differs')
    require([x['phase'] for x in report['phases']]==['intake','candidate','diagnose']
        and report['newMatrixMACs']==report['newOriginalProjections']==report['newConsumers']==0,
        'Actual phase order or work scope differs')
    require(sha(root/'docs/plans/spatial-key-norm-capacity.md')==sha(p/'frozen-plan.md')
        ==request['sourceHashes']['frozen-plan.md']==advice['executedPlanSHA256'],'Executed plan changed')
    parent=root/'docs/results/coherent-norm';dual=root/'docs/results/image-dual'
    pa,ps=read(parent/'actor.json'),read(parent/'inputs.json')
    expected={'pNorm':read(dual/'compiler.json')['compile']['fields']['pDenominator'],
        'qNorm':read(dual/'reference.json')['reference']['fields']['qDenominator'],
        'numerator':pa['read']['fields']['imageNumerator'],'textLogits':pa['read']['fields']['textLogits']}
    require(all(inputs[n]==v for n,v in expected.items())
        and inputs['matchedControlPayloads']==read(parent/'report.json')['matchedControlPayloads']
        and inputs['parentReportSHA256']==sha(parent/'report.json'),'Actual saved input/control origin differs')
    require(inputs['sourceCuts']=={n:{k:c[k] for k in ['cut','cutFile','cutID']} for n,c in ps['sourceCuts'].items()},
        'Native source cut origin differs')
    for cut in inputs['sourceCuts'].values():
        require(intake['nativeFiles'][cut['cutFile']['path']]==cut['cutFile']['sha256'],
            'Actual native cut byte witness absent')
    require(advice['reportSHA256']==sha(p/'report.json')
        and advice['candidateSealSHA256']==sha(p/'candidate-seal.json')
        and advice['reviewedArticleSHA256']==sha(root/'docs/spatial-key-norm-capacity.md'),
        'Reviewed actual result/seal/article differs')
    for n,h in advice['reviewedSources'].items():
        require(sha(root/'generator'/n)==h==request['sourceHashes'][n],'Reviewed/staged/current source differs')
    tests=read(e/'tests.json')
    require((tests['passed'],tests['seconds'],tests['exitCode'])==(297,17.86,0)
        and tests['sourceSHA256']==advice['reviewedTestSourceSHA256']==sha(root/tests['source']),
        'Actual test/source receipt differs')
    instrument=read(e/'saturn-instrument.json')
    require((instrument['records'],instrument['predicates'])==(24672,10)
        and all(instrument[k]==0 for k in ['runExit','verifyExit','formalReceipts','boundXrefAddresses']),
        'Actual instrument coverage/scope differs')
    scheduler=read(p/'scheduler.json');live=Api().json('GET','/api/jobs/'+scheduler['job_id'])
    require(live['state']=='succeeded' and live['payload_custody']==scheduler['payload_custody'],
        'Current terminal scheduler custody differs')
    require(scheduler['job_id']=='job-a8cbc7c17161' and scheduler['result']['elapsed_s']==1.44,
        'Actual job/timing differs')
    figure=root/'docs/results/spatial-norm-figure/spatial-norm.png';fr=read(figure.parent/'figure.json')
    require(sha(figure)==fr['imageSHA256']==sha(root.parent/'obsidian/images/zero-neural-cycle61-spatial-norm.png')
        and fr['reportSHA256']==sha(p/'report.json')
        and fr['plotSourceSHA256']==sha(root/'generator/plot_spatial_norm.py'),'Figure/source/data/blog copy differs')
    plan=root/'docs'/progress['nextCut']['plan'];future=progress['nextCut']
    require(future['status']==advice['nextCut']['status']=='written-unexecuted'
        and future['planSHA256']==advice['nextCut']['sha256']==sha(plan)
        and future['totalNewMatrixMACs']==1874853888,'Next unexecuted plan/work differs')
    blog=root/progress['findingsLog']['path']
    require(progress['findingsLog']['lastUpdatedCycle']==61 and progress['findingsLog']['sha256']==sha(blog),
        'Living blog binding differs')
    section=blog.read_text().split('## Cycle 61:',1)[1].split('## What would close the original goal',1)[0]
    links=0
    for path,contents in [(blog,section),(root/'docs/spatial-key-norm-capacity.md',None),(e/'README.md',None),(plan,None)]:
        for link in re.findall(r'\]\(([^)]+)\)',contents if contents is not None else path.read_text()):
            if not link.startswith(('http:','https:')):
                target=(path.parent/link.split('#',1)[0]).resolve()
                require(target==(e/'publication-qa.json').resolve() or target.exists(),'Missing local backlink: '+link)
                links+=1
    sm,reviews=progress['storageMaintenance'],progress['reviews']
    require((sm['lastActualArchiveCycle'],sm['lastAuditCycle'],sm['nextDueByCycle'],reviews['lastCycle'],
        reviews['nextDueByCycle'],reviews['latestActualResultCycle'])==(60,60,63,59,62,61),
        'Storage/full-post-review/result-review cadence differs')
    require(reviews['latestAdvisorSHA256']==sha(e/'advisor.json'),'Latest advisor binding differs')
    result={**mechanics,'schema':'tce-spatial-norm-publication-qa-v1','reportSHA256':sha(p/'report.json'),
        'progressSHA256':sha(root/'docs/zero-neural-progress.json'),'blogSHA256':sha(blog),'figureSHA256':sha(figure),
        'advisorSHA256':sha(e/'advisor.json'),'nextPlanSHA256':sha(plan),'auditSourceSHA256':sha(__file__),
        'testsPassed':297,'localLinksChecked':links,'savedInputJoins':list(expected),'currentTerminalJob':scheduler['job_id'],
        'nextStorageCycle':63,'nextFullNewestPostReviewCycle':62,
        'scope':'Named source, byte, seal, geometry, narrative and cadence consistency; no effect-size gate or semantic/image/generalization/speedup/arbitrary-prompt zero-neural certification.'}
    save(e/'publication-qa.json',result)
    return {k:result[k] for k in ['ownedSourcePins','normObservations','queryHeadObservations','fullByteFiles',
        'NPYPayloadWitnesses','localLinksChecked','nextStorageCycle','nextFullNewestPostReviewCycle']}


if __name__=='__main__':print(json.dumps(execute(),indent=2))
