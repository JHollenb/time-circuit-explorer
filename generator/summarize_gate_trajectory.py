"""Record observed cycle72 mechanics, instrument, images and storage receipts."""
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha
from publish_gate_trajectory import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/gate-first-trajectory';out=root/'docs/results/gate-first-trajectory-evidence';out.mkdir(parents=True,exist_ok=False)
    save(out/'mechanics.json',audit(p));r=read(p/'request.json');c=read(p/'candidate.json');diff=read(p/'differences.json')
    save(out/'tests.json',{'schema':'tce-gate-trajectory-tests-v1','command':'rtk proxy ../common/bin/common exec python -m pytest -q generator','passed':335,'seconds':18.97,'exitCode':0,
        'newTrajectoryTests':2,'sourceHashes':{n:sha(root/'generator'/n) for n in r['ownedSources']},
        'scope':'Observed335-suite including independent complete/selected two-stream refs and whole tiny full-support model vs70 through twoownclocks, plus inherited score/moment/autocast checks. Zero pretrained local model evaluations; no native BF16 parity or quality claim.'})
    names=['observations.jsonl','partition-observations.jsonl','moment-observations.jsonl','selection-observations.jsonl','block-progress.jsonl','progress.jsonl']
    save(out/'saturn-instrument.json',{'schema':'tce-gate-trajectory-instrument-v1','target':'artifacts/gate-first-trajectory-v1/result','runExit':0,'verifyExit':0,'instrument_status':'target-streams-only',
        'streams':6,'records':753833,'signedComparisonRows':11840,'descriptivePartitionHeadRows':2304,'descriptiveMomentHeadRows':2304,'blockProgressRows':100,'clockProgressRows':5,
        'fires':0,'formalClaims':0,'qualifiedAddresses':0,'verifyMissingExpectedFireLedgers':6,'sourceStreams':{n:sha(root/'artifacts/gate-first-trajectory-v1/result'/n) for n in names},
        'scope':'Observed standard offline sweep/verification. No fire ledgers generated and no semantic certification or quality gate. All signed rows retained.'})
    images={}
    for role in ['candidate','native']:
        src=p/(role+'-A.png');dst=root.parent/'obsidian/images'/('zero-neural-cycle72-'+role+'-A.png');shutil.copyfile(src,dst);images[role]={'publishedSHA256':sha(src),'notebookCopySHA256':sha(dst)}
    save(out/'render-qa.json',{'schema':'tce-gate-trajectory-render-qa-v1','images':images,'figureSHA256':sha(root/'docs/results/gate-first-trajectory-figure/gate-first-trajectory.png'),
        'visualInspection':'Two dominant objects retain left/right red/blue order, but redcube becomes a rounded red form and blue/cyan object is blurred/distorted; native square geometry/sharpness lost. Figure legible, labels/legend unclipped.',
        'scope':'Direct actual PNG/figure inspection, not independent semantic evaluation or a new pixel comparison.'})
    save(out/'storage.json',{'schema':'tce-gate-trajectory-storage-v1','candidateArrays':len(c['fields']),'candidateBytes':sum(x['bytes'] for x in c['fields'].values()),
        'differenceArrays':len(diff),'differenceBytes':sum(x['bytes'] for x in diff.values()),'primaryCompactFiles':len(list(p.iterdir())),'primaryCompactBytes':sum(x.stat().st_size for x in p.iterdir()),
        'logicalBlockOutputBytesPerClock':351043584,'extraMomentBytesPerClock':41582592,'meanCovarianceBytesPerBlock':1585152,
        'logicalMapMaterializationBytes':12834570240,'maxLogicalActiveMapBytes':113246208,'localFreeBytesObserved':shutil.disk_usage(root).free,
        'lastActualCleanupCycle':72,'nextCleanupDueCycle':75,'scope':'1486 newrawarrays remainnearcompute. Compactmetadata/PNGs/lossless scalarstreams local. Cleanup72 full-SHA archives6redundant pairedtransports991780817B,82totalrestorepointers;11remaining4.484GBsize/name candidatesnotdeletioncertificates. Shared free observation has no exclusive attribution; logical map bytes are not residency.'})
    save(out/'timing-qualification.json',{'schema':'tce-gate-trajectory-timing-qualification-v1','ledgerSourceSHA256':sha(root/'generator/gate_trajectory_ledger.py'),
        'qualification':'Frozen perAct scope text says excludes only final output fingerprint, but timer also stops before post-operation ABI and finite-output checks. Treat these as synchronized operation intervals, not complete perAct overhead. Worker after-import/setup/CPU measurements include those checks and all fingerprint/I/O work. No historical execution rewritten.',
        'sourceNormOperationSeconds':read(p/'execution.json')['timings'][0]['instrumentedSynchronizedSecondsBeforeFinalOutputFingerprint']})
    return {'typedActs':10,'explicitOutputABIs':1471,'testsPassed':335,'newRawArrays':len(c['fields'])+len(diff)}


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True))
