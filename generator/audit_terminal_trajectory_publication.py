"""Bind the actual trajectory, immutable failure, living log and next hypothesis."""
import json
from pathlib import Path
import re
from encoder_atoms import read,save,sha
from publish_terminal_trajectory import audit


def require(value,message):
    if not value:raise ValueError(message)


def execute():
    from mrun.client.api import Api
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/terminal-trajectory';ev=root/'docs/results/terminal-trajectory-evidence'
    mechanics=audit(p);require(mechanics==read(ev/'mechanics.json'),'Actual mechanics receipt differs')
    report=read(p/'report.json');advice=read(ev/'advisor.json');progress=read(root/'docs/zero-neural-progress.json')
    for key,name in [('reportSHA256','report.json'),('candidateSealSHA256','candidate-seal.json'),('executionSHA256','execution.json')]:
        require(advice[key]==sha(p/name),'Actual reviewed result differs:'+name)
    for n,h in advice['reviewedSources'].items():require(sha(root/'generator'/n)==h,'Reviewed source differs:'+n)
    article=root/'docs/terminal-own-state-trajectory.md';plan=root/'docs/plans/fixed-text-memory-trajectory.md'
    require(advice['articleSHA256']==sha(article) and advice['publicationAuditorSHA256']==sha(__file__) and advice['nextPlanSHA256']==sha(plan),'Reviewed narrative/source/plan differs')
    require((progress['researchCycle'],progress['status'],progress['liveJobs'])==(66,'active',[]),'Active goal/cycle differs')
    require(progress['lastCompletedResearchTurn']['cycle']==65 and progress['lastCompletedResearchTurn']['commit']=='c5d8d4b27228481a19ad21a7f9f32afef4e09655'
        and progress['thisGoalTurn']['classification']=='progress','Prior/current progress differs')
    current=progress['terminalTrajectoryEvidence']
    require(current['reportSHA256']==sha(p/'report.json') and current['scores']==report['scores'] and current['trajectory']==report['trajectory'] and current['work']==report['work'],'Progress/result differs')
    require(progress['nextCut']['status']=='written-unexecuted' and progress['nextCut']['planSHA256']==sha(plan)
        and progress['nextCut']['prospectiveAvoidedMatrixMACs']==5798205849600 and progress['nextCut']['incrementalAvoidedMatrixMACs']==5566277615616,'Next intervention scope differs')
    tests=read(ev/'tests.json');require((tests['passed'],tests['seconds'],tests['exitCode'])==(313,18.59,0),'Executed mechanics tests differ')
    for n,h in tests['sourceHashes'].items():require(sha(root/'generator'/n)==h,'Tested source differs:'+n)
    instrument=read(ev/'saturn-instrument.json');require(instrument['runExit']==instrument['verifyExit']==0 and instrument['streams']==2 and instrument['records']==11845
        and instrument['fires']==instrument['formalClaims']==instrument['qualifiedAddresses']==0,'Standard instrument scope differs')
    failure=root/'docs/results/terminal-trajectory-failed-v1';fp=read(failure/'publication.json');f=read(failure/'failure.json')
    require(f['jobID']=='job-f5d33304c764' and f['state']=='failed' and f['result']['elapsed_s']==61.76 and f['observedArrays']==f['observedImages']==[],'Failed predecessor scope differs')
    for n,h in fp['files'].items():require(sha(failure/n)==h,'Failed predecessor changed:'+n)
    require(fp['publisherSourceSHA256']==sha(root/'generator/publish_terminal_trajectory_failure.py'),'Failed publisher changed')
    figure=root/'docs/results/terminal-trajectory-figure-v2';fr=read(figure/'figure.json');image=figure/'terminal-trajectory.png';qa=read(ev/'render-qa.json')
    require(fr['reportSHA256']==sha(p/'report.json') and fr['plotSourceSHA256']==sha(root/'generator/plot_terminal_trajectory_v2.py')
        and fr['imageSHA256']==sha(image)==sha(root.parent/'obsidian/images/zero-neural-cycle66-terminal-trajectory-v2.png')==qa['latestImageSHA256'],'Figure/copy/result differs')
    require(qa['previousImageSHA256']==sha(root/'docs/results/terminal-trajectory-figure/terminal-trajectory.png'),'Original plot not preserved')
    for role in ['native','candidate']:
        require(qa['images'][role]['publishedSHA256']==sha(p/(role+'-A.png'))==sha(root.parent/'obsidian/images'/('zero-neural-cycle66-'+role+'-A.png')),'Actual PNG/copy differs')
    archive=read(root/'docs/results/storage-cleanup-cycle66/result.json');storage=read(root/'docs/results/storage-audit-cycle66/result.json')
    require(archive['files']==4 and archive['archivedBytes']==1296894473 and len(storage['pointerChecks'])==len(storage['remoteMetadataChecks'])==72,'Actual cleanup/audit differs')
    for row in archive['rows']:require(row['sha256']==row['remoteVerification']['sha256'] and row['remoteVerification']['verified'] is True,'Cold byte witness absent')
    require((progress['storageMaintenance']['lastActualArchiveCycle'],progress['storageMaintenance']['lastAuditCycle'],progress['storageMaintenance']['nextDueByCycle'])==(66,66,69),'Storage cadence differs')
    require((progress['reviews']['lastCycle'],progress['reviews']['nextDueByCycle'],progress['reviews']['latestActualResultCycle'])==(65,68,66)
        and progress['reviews']['latestAdvisorSHA256']==sha(ev/'advisor.json'),'Periodic/actual advisor cadence differs')
    blog=root/progress['findingsLog']['path'];require(progress['findingsLog']['lastUpdatedCycle']==66 and progress['findingsLog']['sha256']==sha(blog),'Living findings log differs')
    section=blog.read_text().split('## Cycle 66:',1)[1].split('## What would close the original goal',1)[0];links=0
    for path,text in [(blog,section),(article,None),(ev/'README.md',None),(plan,None)]:
        for link in re.findall(r'\]\(([^)]+)\)',text if text is not None else path.read_text()):
            if not link.startswith(('http:','https:')):
                target=(path.parent/link.split('#',1)[0]).resolve();require(target==(ev/'publication-qa.json').resolve() or target.exists(),'Missing local backlink:'+link);links+=1
    job=read(p/'scheduler.json');live=Api().json('GET','/api/jobs/'+job['job_id'])
    require(live['state']=='succeeded' and live['payload_custody']==job['payload_custody'] and job['job_id']=='job-8b0c4f4bcc44' and job['result']['elapsed_s']==58.75,'Authoritative completed job differs')
    result={**mechanics,'schema':'tce-terminal-trajectory-publication-qa-v1','reportSHA256':sha(p/'report.json'),'articleSHA256':sha(article),
        'blogSHA256':sha(blog),'progressSHA256':sha(root/'docs/zero-neural-progress.json'),'advisorSHA256':sha(ev/'advisor.json'),
        'figureSHA256':sha(image),'auditSourceSHA256':sha(__file__),'nextPlanSHA256':sha(plan),'testsPassed':313,'localLinksChecked':links,
        'nextNewestPostReviewCycle':68,'nextStorageCycle':69,'scope':'Actual source/byte/work/ownstate/publication mechanics only. No scientific quality threshold, generalization, matched speedup or arbitrary-prompt zero-neural certification.'}
    save(ev/'publication-qa.json',result)
    return {k:result[k] for k in ['typedActs','explicitOutputABIs','ownStateConsumptionJoins','physicalSliceByteWitnesses','observations','fullByteFiles','NPYPayloadWitnesses','localLinksChecked']}


if __name__=='__main__':print(json.dumps(execute(),indent=2))
