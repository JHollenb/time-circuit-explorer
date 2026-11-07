"""Record observed cycle70 mechanics, instrument, images and storage receipts."""
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha
from publish_image_moments import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/image-partition-moments';out=root/'docs/results/image-partition-moments-evidence';out.mkdir(parents=True,exist_ok=False)
    save(out/'mechanics.json',audit(p));r=read(p/'request.json');c=read(p/'candidate.json');diff=read(p/'differences.json')
    save(out/'tests.json',{'schema':'tce-image-moments-tests-v1','command':'rtk proxy ../common/bin/common exec python -m pytest -q generator','passed':327,'seconds':18.43,'exitCode':0,
        'targetedPassed':5,'targetedSeconds':3.26,'sourceHashes':{n:sha(root/'generator'/n) for n in r['ownedSources']},
        'scope':'Observed population/off-diagonal/translation/constant-key/autocast fixtures and independent direct-score moment reference through native tiny blocks over two own commits. Zero pretrained local model evaluations; no native BF16 parity or quality claim.'})
    names=['observations.jsonl','partition-observations.jsonl','moment-observations.jsonl','block-progress.jsonl','progress.jsonl']
    save(out/'saturn-instrument.json',{'schema':'tce-image-moments-instrument-v1','target':'artifacts/image-partition-moments-v1/result','runExit':0,'verifyExit':0,'instrument_status':'target-streams-only',
        'streams':5,'records':16553,'signedComparisonRows':11840,'descriptivePartitionHeadRows':2304,'descriptiveMomentHeadRows':2304,'blockProgressRows':100,'clockProgressRows':5,
        'fires':0,'formalClaims':0,'qualifiedAddresses':0,'verifyMissingExpectedFireLedgers':5,'sourceStreams':{n:sha(root/'artifacts/image-partition-moments-v1/result'/n) for n in names},
        'scope':'Observed standard offline sweep/verification. No fire ledgers generated and no semantic certification or quality gate. All signed rows retained.'})
    images={}
    for role in ['candidate','native']:
        src=p/(role+'-A.png');dst=root.parent/'obsidian/images'/('zero-neural-cycle70-'+role+'-A.png');shutil.copyfile(src,dst);images[role]={'publishedSHA256':sha(src),'notebookCopySHA256':sha(dst)}
    save(out/'render-qa.json',{'schema':'tce-image-moments-render-qa-v1','images':images,'figureSHA256':sha(root/'docs/results/image-partition-moments-figure/image-partition-moments.png'),
        'visualInspection':'Two separate dominant objects, closed red cube left and blue sphere right; sphere still has an added base. Proportions, lighting and shading differ from native. Figure labels/legend clean and legible.',
        'scope':'Direct actual PNG/figure inspection, not independent semantic evaluation or a new pixel comparison.'})
    save(out/'storage.json',{'schema':'tce-image-moments-storage-v1','candidateArrays':len(c['fields']),'candidateBytes':sum(x['bytes'] for x in c['fields'].values()),
        'differenceArrays':len(diff),'differenceBytes':sum(x['bytes'] for x in diff.values()),'primaryCompactFiles':len(list(p.iterdir())),'primaryCompactBytes':sum(x.stat().st_size for x in p.iterdir()),
        'logicalBlockOutputBytesPerClock':351043584,'extraMomentBytesPerClock':41582592,'meanCovarianceBytesPerBlock':1585152,
        'logicalTerminalMapMaterializationBytes':679477248,'maxLogicalActiveMapBytes':169869312,'localFreeBytesObserved':shutil.disk_usage(root).free,
        'lastActualCleanupCycle':69,'nextCleanupDueCycle':72,'scope':'924 new raw arrays remain near compute. Compact metadata/PNGs/lossless scalar streams local. No new archive/deletion this cycle; cleanup69 receipts/restore pointers retained. Shared free observation has no exclusive attribution; logical map bytes are not residency.'})
    return {'typedActs':9,'explicitOutputABIs':909,'testsPassed':327,'newRawArrays':len(c['fields'])+len(diff)}


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True))
