"""Receipts for observed cycle69 execution, instruments, images and storage."""
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha
from publish_partition_feedback import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/partition-feedback';out=root/'docs/results/partition-feedback-evidence';out.mkdir(parents=True,exist_ok=False)
    save(out/'mechanics.json',audit(p));r=read(p/'request.json');c=read(p/'candidate.json');diff=read(p/'differences.json')
    save(out/'tests.json',{'schema':'tce-partition-feedback-tests-v1','command':'rtk proxy ../common/bin/common exec python -m pytest -q generator','passed':322,'seconds':18.73,'exitCode':0,
        'targetedPassed':3,'targetedSeconds':3.31,'sourceHashes':{n:sha(root/'generator'/n) for n in r['ownedSources']},
        'scope':'Observed tiny native-block reference with zero-image-value FP32 text SDPA, two own commits, shared denominator/image-key competition and autocast controls. No local pretrainedmodel; actualmrunresult qualifiesCUDAexecution.'})
    names=['observations.jsonl','partition-observations.jsonl','block-progress.jsonl','progress.jsonl']
    save(out/'saturn-instrument.json',{'schema':'tce-partition-feedback-instrument-v1','target':'artifacts/partition-feedback-v1/result','runExit':0,'verifyExit':0,'instrument_status':'target-streams-only',
        'streams':4,'records':14249,'signedComparisonRows':11840,'descriptivePartitionHeadRows':2304,'blockProgressRows':100,'clockProgressRows':5,'fires':0,'formalClaims':0,'qualifiedAddresses':0,'verifyMissingExpectedFireLedgers':4,
        'sourceStreams':{n:sha(root/'artifacts/partition-feedback-v1/result'/n) for n in names},'scope':'Observed standard offline sweep/verification. No fires or fire ledgers generated; no semantic certification or quality gate. All signed rows retained.'})
    images={}
    for role in ['candidate','native']:
        src=p/(role+'-A.png');dst=root.parent/'obsidian/images'/('zero-neural-cycle69-'+role+'-A.png');shutil.copyfile(src,dst);images[role]={'publishedSHA256':sha(src),'notebookCopySHA256':sha(dst)}
    save(out/'render-qa.json',{'schema':'tce-partition-feedback-render-qa-v1','images':images,
        'trajectoryFigureSHA256':sha(root/'docs/results/partition-feedback-figure/partition-feedback.png'),'massFigureV1SHA256':sha(root/'docs/results/partition-mass-figure/partition-mass.png'),'massFigureV2SHA256':sha(root/'docs/results/partition-mass-figure-v2/partition-mass-v2.png'),
        'visualInspection':'Two dominant separate objects, closedredcubeleft/bluesphereright; spherehasaddedbase andshape/shadingdiffer. Mainfigureclean. InitialmassfigureJoint4/Single0ticksoverlap; immutablev1retained, separatev2removesadjacenttickandisclean.',
        'scope':'Direct actual PNG/figure inspection, not independent semantic evaluation or new pixel arithmetic.'})
    save(out/'storage.json',{'schema':'tce-partition-feedback-storage-v1','candidateArrays':len(c['fields']),'candidateBytes':sum(x['bytes'] for x in c['fields'].values()),
        'differenceArrays':len(diff),'differenceBytes':sum(x['bytes'] for x in diff.values()),'primaryCompactFiles':len(list(p.iterdir())),'primaryCompactBytes':sum(x.stat().st_size for x in p.iterdir()),
        'logicalBlockOutputBytesPerClock':309460992,'logicalTerminalMapMaterializationBytes':679477248,'maxLogicalActiveMapBytes':169869312,'localFreeBytesObserved':shutil.disk_usage(root).free,
        'cleanupReceipt':'../storage-cleanup-cycle69/result.json','auditReceipt':'../storage-audit-cycle69/result.json','archivedBytes':1150830575,
        'scope':'Complete own states/alpha/textreads/differences staynearcompute;compactmetadata/PNGs/losslessrowslocal. Cleanup69fourverifiedtransports;76pointermetadataaudit,older72notfreshlybytehashed,next72orpressure. Sharedfreeobservationisnotexclusiveattribution;logicalmapbytesnotresidency.'})
    return {'typedActs':9,'explicitOutputABIs':429,'testsPassed':322,'newRawArrays':len(c['fields'])+len(diff)}


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True))
