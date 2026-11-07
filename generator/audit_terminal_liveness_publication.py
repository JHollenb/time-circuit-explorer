"""Bind terminal reduction evidence, narrative, review and next trajectory plan."""
import json
from pathlib import Path
import re
from encoder_atoms import read,save,sha
from publish_terminal_liveness import audit


def require(value,message):
    if not value:raise ValueError(message)


def execute():
    from mrun.client.api import Api
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/terminal-liveness';ev=root/'docs/results/terminal-liveness-evidence'
    m=audit(p);require(m==read(ev/'mechanics.json'),'Mechanics receipt differs')
    r,s,report,advice,progress=[read(x) for x in [p/'request.json',p/'inputs.json',p/'report.json',ev/'advisor.json',root/'docs/zero-neural-progress.json']]
    parent=root/'docs/results/terminal-native-capture'
    for name,h in s['parentRecords'].items():require(sha(parent/name)==h,'Native parent record differs:'+name)
    c=read(parent/'capture.json');targets=read(p/'targets.json')
    require(targets['fields']=={n:c['targetFields'][n] for n in ['image','velocity']} and targets['sourceCombined']==c['sourceFields']['combined']
        and targets['sourceCutID']==c['source']['cutID'] and targets['targetCutID']==c['target']['cutID'],'Native comparison/source origin differs')
    require(s['sourceCut']==read(parent/'state-cut-records.json')['source'],'Complete input origin differs')
    for key,name in [('reportSHA256','report.json'),('candidateSealSHA256','candidate-seal.json'),('executionSHA256','execution.json')]:
        require(advice[key]==sha(p/name),'Advisor actual result differs:'+name)
    for name,h in advice['reviewedSources'].items():require(sha(root/'generator'/name)==h,'Reviewed source differs:'+name)
    article=root/'docs/terminal-output-liveness.md';require(advice['articleSHA256']==sha(article) and advice['publicationAuditorSHA256']==sha(__file__),'Reviewed narrative/auditor differs')
    require((progress['researchCycle'],progress['status'],progress['liveJobs'])==(65,'active',[]),'Active goal/cycle differs')
    require(progress['lastCompletedResearchTurn']['cycle']==64 and progress['lastCompletedResearchTurn']['commit']=='d59359591f3e3061ba8591484daf6d7251a776a9'
        and progress['thisGoalTurn']['classification']=='progress','Previous/current progress differs')
    current=progress['terminalLivenessEvidence']
    require(current['reportSHA256']==sha(p/'report.json') and current['scores']==report['scores'] and current['newMatrixMACs']==145376673792
        and current['avoidedMatrixMACs']==57982058496,'Progress/result differs')
    tests=read(ev/'tests.json');require((tests['passed'],tests['seconds'],tests['exitCode'])==(308,18.40,0),'Executed tests differ')
    for name,h in tests['sourceHashes'].items():require(sha(root/'generator'/name)==h,'Test source differs:'+name)
    inst=read(ev/'saturn-instrument.json');require(inst['runExit']==inst['verifyExit']==0 and inst['records']==5248 and inst['applicablePredicates']==28
        and inst['fires']==inst['formalClaims']==inst['qualifiedAddresses']==0,'Standard instrument scope differs')
    figures=root/'docs/results/terminal-liveness-figure-v2';fr=read(figures/'figure.json');image=figures/'terminal-liveness.png'
    require(fr['reportSHA256']==sha(p/'report.json') and fr['executionSHA256']==sha(p/'execution.json')
        and fr['plotSourceSHA256']==sha(root/'generator/plot_terminal_liveness_v2.py') and fr['imageSHA256']==sha(image)
        ==sha(root.parent/'obsidian/images/zero-neural-cycle65-terminal-liveness-v2.png'),'Figure/source/result copy differs')
    render=read(ev/'render-qa.json');require(render['latestImageSHA256']==sha(image) and render['visualInspection']=='complete; no title clipping or legend/axis overlap'
        and render['previousImageSHA256']==sha(root/'docs/results/terminal-liveness-figure/terminal-liveness.png'),'Actual visual QA differs')
    future=progress['nextCut'];plan=root/'docs'/future['plan']
    require(future['status']=='written-unexecuted' and future['planSHA256']==sha(plan)==advice['nextPlanSHA256']
        and future['prospectiveAvoidedMatrixMACs']==231928233984 and future['newNativeReferences']==0,'Next trajectory scope differs')
    historic=root/'artifacts/complete-masked-image-v1/result'
    require(sha(historic/'images-full.json')==future['nativeComparatorRecords']['images-full.json']
        and sha(historic/'full-seal.json')==future['nativeComparatorRecords']['full-seal.json'],'Next comparator metadata differs')
    require((progress['reviews']['lastCycle'],progress['reviews']['nextDueByCycle'],progress['reviews']['latestActualResultCycle'])==(65,68,65)
        and progress['reviews']['latestAdvisorSHA256']==sha(ev/'advisor.json'),'Advisor cadence/current binding differs')
    require((progress['storageMaintenance']['lastActualArchiveCycle'],progress['storageMaintenance']['lastAuditCycle'],progress['storageMaintenance']['nextDueByCycle'])==(63,63,66),'Actual storage cadence differs')
    blog=root/progress['findingsLog']['path'];require(progress['findingsLog']['lastUpdatedCycle']==65 and progress['findingsLog']['sha256']==sha(blog),'Living log binding differs')
    section=blog.read_text().split('## Cycle 65:',1)[1].split('## What would close the original goal',1)[0];links=0
    for path,text in [(blog,section),(article,None),(ev/'README.md',None),(plan,None)]:
        for link in re.findall(r'\]\(([^)]+)\)',text if text is not None else path.read_text()):
            if not link.startswith(('http:','https:')):
                target=(path.parent/link.split('#',1)[0]).resolve();require(target==(ev/'publication-qa.json').resolve() or target.exists(),'Missing local backlink:'+link);links+=1
    job=read(p/'scheduler.json');live=Api().json('GET','/api/jobs/'+job['job_id'])
    require(live['state']=='succeeded' and live['payload_custody']==job['payload_custody'] and job['job_id']=='job-07e24bebce9b'
        and job['result']['elapsed_s']==38.07,'Authoritative terminal job differs')
    result={**m,'schema':'tce-terminal-liveness-publication-qa-v1','reportSHA256':sha(p/'report.json'),'articleSHA256':sha(article),
        'blogSHA256':sha(blog),'progressSHA256':sha(root/'docs/zero-neural-progress.json'),'advisorSHA256':sha(ev/'advisor.json'),
        'figureSHA256':sha(image),'auditSourceSHA256':sha(__file__),'nextPlanSHA256':sha(plan),'testsPassed':308,'localLinksChecked':links,
        'nextNewestPostReviewCycle':68,'nextStorageCycle':66,'nextComparatorScope':'Local sealed native metadata only; complete source/target byte qualification required before staging.',
        'scope':'Source/custody/physicalwork/ownconsumer/publication mechanics. No effect-size, matchedlatency, imagequality, arbitrary-prompt or zero-neural certification.'}
    save(ev/'publication-qa.json',result)
    return {k:result[k] for k in ['ownedSourcePins','typedActs','explicitOutputABIs','ownImageConsumerJoins','physicalSliceByteWitnesses','observations','fullByteFiles','NPYPayloadWitnesses','localLinksChecked']}


if __name__=='__main__':print(json.dumps(execute(),indent=2))
