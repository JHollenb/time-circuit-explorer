"""Bind the adverse image result, own-state evidence and next source question."""
import json
from pathlib import Path
import re
from encoder_atoms import read,save,sha
from publish_fixed_text_memory import audit


def require(value,message):
    if not value:raise ValueError(message)


def execute():
    from mrun.client.api import Api
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/fixed-text-memory';ev=root/'docs/results/fixed-text-memory-evidence'
    mechanics=audit(p);require(mechanics==read(ev/'mechanics.json'),'Actual mechanics receipt differs')
    report=read(p/'report.json');advice=read(ev/'advisor.json');progress=read(root/'docs/zero-neural-progress.json')
    for key,name in [('reportSHA256','report.json'),('candidateSealSHA256','candidate-seal.json'),('executionSHA256','execution.json')]:require(advice[key]==sha(p/name),'Reviewed actual result differs:'+name)
    for n,h in advice['reviewedSources'].items():require(sha(root/'generator'/n)==h,'Reviewed source differs:'+n)
    article=root/'docs/fixed-text-memory.md';plan=root/'docs/plans/prompt-only-text-compiler.md'
    require(advice['articleSHA256']==sha(article) and advice['publicationAuditorSHA256']==sha(__file__) and advice['nextPlanSHA256']==sha(plan),'Reviewed narrative/auditor/plan differs')
    require((progress['researchCycle'],progress['status'],progress['liveJobs'])==(67,'active',[]),'Active goal/cycle differs')
    require(progress['lastCompletedResearchTurn']['cycle']==66 and progress['lastCompletedResearchTurn']['commit']=='9146367cfaa270c5627d8a8f838ff15d7a99cb27'
        and progress['previousGoalTurn']['classification']==progress['thisGoalTurn']['classification']=='progress','Prior/current progress differs')
    current=progress['fixedTextMemoryEvidence'];require(current['reportSHA256']==sha(p/'report.json') and current['scores']==report['scores'] and current['trajectory']==report['trajectory'] and current['work']==report['work'],'Progress/result differs')
    require(current['terminal_status']=='unassessed' and current['mechanics_status']=='complete','Mechanics/terminal separation differs')
    tests=read(ev/'tests.json');require((tests['passed'],tests['seconds'],tests['exitCode'])==(315,18.78,0),'Executed mechanics tests differ')
    for n,h in tests['sourceHashes'].items():require(sha(root/'generator'/n)==h,'Test snapshot differs:'+n)
    instrument=read(ev/'saturn-instrument.json');require(instrument['runExit']==instrument['verifyExit']==0 and instrument['streams']==3 and instrument['records']==11945
        and instrument['fires']==instrument['formalClaims']==instrument['qualifiedAddresses']==0,'Standard instrument scope differs')
    figure=root/'docs/results/fixed-text-memory-figure';fr=read(figure/'figure.json');image=figure/'fixed-text-memory.png';qa=read(ev/'render-qa.json')
    require(fr['reportSHA256']==sha(p/'report.json') and fr['plotSourceSHA256']==sha(root/'generator/plot_fixed_text_memory.py')
        and fr['imageSHA256']==sha(image)==sha(root.parent/'obsidian/images/zero-neural-cycle67-fixed-text-memory.png')==qa['imageSHA256'],'Figure/source/result copy differs')
    for role in ['native','candidate']:require(qa['images'][role]['publishedSHA256']==sha(p/(role+'-A.png'))==sha(root.parent/'obsidian/images'/('zero-neural-cycle67-'+role+'-A.png')),'Actual PNG/copy differs')
    require((progress['storageMaintenance']['lastActualArchiveCycle'],progress['storageMaintenance']['lastAuditCycle'],progress['storageMaintenance']['nextDueByCycle'])==(66,66,69),'Actual storage cadence differs')
    storage=read(ev/'storage.json');require(storage['candidateArrays']+storage['differenceArrays']==56 and storage['candidateBytes']+storage['differenceBytes']==143236096,'New raw-array footprint differs')
    require((progress['reviews']['lastCycle'],progress['reviews']['nextDueByCycle'],progress['reviews']['latestActualResultCycle'])==(67,70,67)
        and progress['reviews']['latestAdvisorSHA256']==sha(ev/'advisor.json'),'Full/latest-result review cadence differs')
    for post in advice['latestPosts']:require(sha(root/post['path'])==post['sha256'],'Reviewed neighboring snapshot changed:'+post['path'])
    future=progress['nextCut'];nt,dim,mlp=512,3072,9216
    compiler=100*2*nt*dim**2+96*(nt*(2*dim**2+3*dim*mlp)+2*nt**2*dim);cache=100*2*nt*dim*2
    new_account=report['work']['modelMatrixAndConvolutionMACs']-100*2*nt*dim**2+compiler
    require(future['status']=='written-unexecuted' and future['planSHA256']==sha(plan) and future['prospectiveCompilerMatrixMACs']==compiler==6223407611904
        and future['prospectiveKVCacheBytes']==cache==629145600 and future['prospectiveCompilerPlusOneImageMACs']==new_account==21062568902656
        and future['prospectiveAvoidedMatrixMACs']==541165879296,'Prospective compiler/cache account differs')
    blog=root/progress['findingsLog']['path'];require(progress['findingsLog']['lastUpdatedCycle']==67 and progress['findingsLog']['sha256']==sha(blog),'Living findings log differs')
    section=blog.read_text().split('## Cycle 67:',1)[1].split('## What would close the original goal',1)[0];links=0
    for path,text in [(blog,section),(article,None),(ev/'README.md',None),(plan,None)]:
        for link in re.findall(r'\]\(([^)]+)\)',text if text is not None else path.read_text()):
            if not link.startswith(('http:','https:')):
                target=(path.parent/link.split('#',1)[0]).resolve();require(target==(ev/'publication-qa.json').resolve() or target.exists(),'Missing local backlink:'+link);links+=1
    job=read(p/'scheduler.json');live=Api().json('GET','/api/jobs/'+job['job_id'])
    require(live['state']=='succeeded' and live['payload_custody']==job['payload_custody'] and job['job_id']=='job-96e74e161844' and job['result']['elapsed_s']==119.85,'Authoritative terminal job differs')
    result={**mechanics,'schema':'tce-fixed-text-memory-publication-qa-v1','reportSHA256':sha(p/'report.json'),'articleSHA256':sha(article),
        'blogSHA256':sha(blog),'progressSHA256':sha(root/'docs/zero-neural-progress.json'),'advisorSHA256':sha(ev/'advisor.json'),'figureSHA256':sha(image),
        'auditSourceSHA256':sha(__file__),'nextPlanSHA256':sha(plan),'testsPassed':315,'localLinksChecked':links,'nextNewestPostReviewCycle':70,'nextStorageCycle':69,
        'scope':'Source/custody/physicalwork/ownstate/publication mechanics. Adverse image remains evidence; no quality threshold, sole-cause attribution, matched speedup or arbitrary-prompt zero-neural certification.'}
    save(ev/'publication-qa.json',result)
    return {k:result[k] for k in ['typedActs','explicitOutputABIs','ownStateConsumptionJoins','fixedTextOwnImageBlockJoins','physicalSliceByteWitnesses','observations','fullByteFiles','NPYPayloadWitnesses','localLinksChecked']}


if __name__=='__main__':print(json.dumps(execute(),indent=2))
