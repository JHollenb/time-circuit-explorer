"""Qualify complete terminal-cut lineage and its publication, without tensor work."""
import json
from pathlib import Path
import re
from encoder_atoms import read,save,sha,identity
from publish_terminal_capture import audit


def require(condition,message):
    if not condition:raise ValueError(message)


def mapping(value):
    if value['kind']!='mapping':raise ValueError('Expected typed mapping')
    return {x['key']['value']:x['value'] for x in value['items']}


def execute():
    from mrun.client.api import Api
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/terminal-native-capture';out=root/'docs/results/terminal-capture-evidence'
    m=audit(p);require(m==read(out/'mechanics.json'),'Base mechanics receipt differs')
    r,s,c,e,advice,progress=[read(x) for x in [p/'request.json',p/'inputs.json',p/'capture.json',p/'execution.json',out/'advisor.json',root/'docs/zero-neural-progress.json']]
    original=s['sourceCut']['cut']['manifest'];original_slots=mapping(original['payload']);joins=[]
    untouched=set(original_slots)-{'image','text','cursor','phase'}
    require(len(untouched)==9 and len(original_slots)==13,'Original complete schema differs')
    for role,phase,cursor,index,next_operation,parent in [
        ('source','single',19,19,'single_block.19',s['sourceCut']['cutID']),
        ('target','readout',20,None,'final_velocity_head',c['source']['cutID'])]:
        cut=c[role]['manifest'];slots=mapping(cut['payload']);point={k:v['value'] for k,v in mapping(cut['execution_point']).items()}
        require(set(slots)==set(original_slots) and all(slots[n]==original_slots[n] for n in untouched),'Untouched source descriptors differ:'+role)
        require(cut['parent']==parent and cut['model_identity']==original['model_identity'] and cut['boundary']=='terminal-'+role,'Cut parent/model/boundary differs:'+role)
        require(cut['execution']['requestSHA256']==r['requestSHA256'] and cut['execution']['sourceExecutionSHA256']==identity(original['execution'])
            and cut['execution']['nativeAdapterSessionCompatible'] is False,'Cut execution contract differs:'+role)
        require(slots['phase']['value']==phase and slots['cursor']['value']==cursor and slots['step']==original_slots['step'],'Phase/cursor/step differs:'+role)
        require(point=={'edge':'before','family':'flux2-klein','index':index,'local_clock':cursor,'logical_step':2,'next_operation':next_operation,'operator':phase,'phase':phase},'Execution point differs:'+role)
        require(slots['text']['shape']==[1,512,3072] and slots['image']['shape']==[1,1024,3072]
            and slots['text']['stride']==slots['image']['stride']==[4718592,3072,1],'Residual view ABI differs:'+role)
        joins.append({'role':role,'unchangedDescriptors':sorted(untouched),'parent':parent,'phase':phase,'cursor':cursor,'modelIdentity':cut['model_identity']})
    require(read(p/'capture-mechanics.json')['requestSHA256']==r['requestSHA256'],'Actual wheel mechanics request differs')
    head=[x for x in e['operations'] if x['stage']=='velocity']
    require([(x['parameter'],x['inputShape'],x['weightShape'],x['nominalMACs']) for x in head]==[
        ('norm_out.linear.weight',[1,3072],[6144,3072],18874368),('proj_out.weight',[1,1024,3072],[128,3072],402653184)],'Actual head accounting differs')
    future=progress['nextCut'];plan=root/'docs'/future['plan'];addendum=root/'docs'/future['executionIntake']
    require(future['status']=='written-unexecuted' and future['planSHA256']==sha(plan)==advice['nextPlanSHA256']
        and future['executionIntakeSHA256']==sha(addendum)==advice['executionIntakeSHA256'],'Next reduction binding differs')
    require(future['sourceCutID']==c['source']['cutID'] and future['targetCutID']==c['target']['cutID']
        and future['candidateBlockAndHeadMACs']==145376673792 and future['nativeBlockAndHeadMACs']==203358732288,'Next input/work differs')
    require((progress['researchCycle'],progress['status'],progress['liveJobs'])==(64,'active',[]),'Active cycle differs')
    require(progress['lastCompletedResearchTurn']['commit']=='ce26ccafbbc494d843b1cf8cb93f3e31423bef6b'
        and progress['lastCompletedResearchTurn']['cycle']==63 and progress['thisGoalTurn']['classification']=='progress','Previous/current progress differs')
    report_hash=sha(p/'report.json');require(progress['terminalCaptureEvidence']['reportSHA256']==report_hash==advice['reportSHA256'],'Result binding differs')
    article=root/'docs/terminal-native-capture.md';require(advice['articleSHA256']==sha(article),'Advisor article differs')
    for name,expected in advice['reviewedSources'].items():require(sha(root/'generator'/name)==expected,'Reviewed source differs:'+name)
    require(advice['publicationAuditorSHA256']==sha(__file__),'Reviewed auditor differs')
    tests=read(out/'tests.json');require((tests['passed'],tests['seconds'],tests['exitCode'])==(305,18.84,0),'Test outcome differs')
    for name,expected in tests['sourceHashes'].items():require(sha(root/'generator'/name)==expected,'Test source differs:'+name)
    inst=read(out/'saturn-instrument.json');require(inst['runExit']==inst['verifyExit']==0 and inst['records']==25 and inst['applicablePredicates']==3
        and inst['fires']==inst['formalClaims']==inst['qualifiedAddresses']==0 and inst['instrument_status']=='target-streams-only','Instrument scope differs')
    census=root/'docs/results/terminal-capture-intake/state-cut-census.json';cin=read(census)
    require(sha(census)==r['sourceHashes']['state-cut-census.json'] and cin['intakeSourceSHA256']==sha(root/'generator/terminal_liveness_intake.py')
        and len(cin['rows'])==342 and cin['terminalSingle19Cuts']==[],'Bounded original census differs')
    failure=root/'docs/results/terminal-native-capture-failed-v1';fp=read(failure/'publication.json')
    require(fp['publisherSourceSHA256']==sha(root/'generator/publish_terminal_capture_failure.py'),'Failure publisher differs')
    for name,expected in fp['files'].items():require(sha(failure/name)==expected,'Failure member differs:'+name)
    for name,expected in fp['sourceHashes'].items():
        if name.endswith('.py'):require(sha(root/'generator'/name)==expected,'Failed scientific source changed:'+name)
    job=read(p/'scheduler.json');live=Api().json('GET','/api/jobs/'+job['job_id']);old=Api().json('GET','/api/jobs/job-c266fd593526')
    require(job['job_id']=='job-69f0736b2924' and live['state']=='succeeded' and live['payload_custody']==job['payload_custody']
        and job['result']['elapsed_s']==79.94 and old['state']=='failed' and old['result']['elapsed_s']==69.80,'Authoritative jobs differ')
    blog=root/progress['findingsLog']['path'];require(progress['findingsLog']['lastUpdatedCycle']==64 and progress['findingsLog']['sha256']==sha(blog),'Living log binding differs')
    section=blog.read_text().split('## Cycle 64:',1)[1].split('## What would close the original goal',1)[0];links=0
    for path,text in [(blog,section),(article,None),(out/'README.md',None),(addendum,None),(failure/'README.md',None)]:
        for link in re.findall(r'\]\(([^)]+)\)',text if text is not None else path.read_text()):
            if not link.startswith(('http:','https:')):
                target=(path.parent/link.split('#',1)[0]).resolve();require(target==(out/'publication-qa.json').resolve() or target.exists(),'Missing backlink:'+link);links+=1
    sm,reviews=progress['storageMaintenance'],progress['reviews']
    require((sm['lastActualArchiveCycle'],sm['lastAuditCycle'],sm['nextDueByCycle'],reviews['lastCycle'],reviews['nextDueByCycle'],reviews['latestActualResultCycle'])
        ==(63,63,66,62,65,64),'Storage/advisor cadence differs')
    require(reviews['latestAdvisorSHA256']==sha(out/'advisor.json'),'Current advisor differs')
    result={**m,'schema':'tce-terminal-capture-publication-qa-v1','completeCutJoins':joins,'headAffineMACs':421527552,'reportSHA256':report_hash,
        'articleSHA256':sha(article),'blogSHA256':sha(blog),'progressSHA256':sha(root/'docs/zero-neural-progress.json'),'advisorSHA256':sha(out/'advisor.json'),
        'auditSourceSHA256':sha(__file__),'nextPlanSHA256':sha(plan),'executionIntakeSHA256':sha(addendum),'testsPassed':305,'localLinksChecked':links,
        'nextNewestPostReviewCycle':65,'nextStorageCycle':66,'scope':'Complete typed component-capture/custody/publication mechanics. No independent full-native parity, reduced-producer/image/speedup or arbitrary-prompt zero-neural certification.'}
    save(out/'publication-qa.json',result)
    return {k:result[k] for k in ['typedNativeActs','explicitOutputABIs','completeStateCuts','fullByteFiles','NPYPayloadWitnesses','headAffineMACs','localLinksChecked']}


if __name__=='__main__':print(json.dumps(execute(),indent=2))
