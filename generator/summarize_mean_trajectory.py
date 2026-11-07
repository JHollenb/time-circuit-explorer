"""Publish observed mechanics, phase scopes and storage without tensor arithmetic."""
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha
from publish_mean_trajectory import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/mean-up-trajectory';out=root/'docs/results/mean-up-trajectory-evidence';out.mkdir(parents=True,exist_ok=False)
    save(out/'mechanics.json',audit(p));r=read(p/'request.json');c=read(p/'candidate.json');e=read(p/'execution.json');scheduler=read(p/'scheduler.json');diff=read(p/'differences.json')
    save(out/'tests.json',{'schema':'tce-mean-trajectory-tests-v1','command':'rtk proxy ../common/bin/common exec python -m pytest -q generator','passed':339,'seconds':18.99,'exitCode':0,
        'newTests':1,'sourceHashes':{n:sha(root/'generator'/n) for n in r['ownedSources']},
        'scope':'Independent tiny quarter/fullUp references, nontrivialRoPE/absoluteimageQpositions, terminalstride, exactnonterminalagreementwithfrozen74, fullsupporttwoownclocks vs70. Inherited mechanics included. No local pretrained model evaluations.'})
    source=root/'artifacts/mean-up-trajectory-v1/result';names=['observations.jsonl','partition-observations.jsonl','moment-observations.jsonl','selection-observations.jsonl','block-progress.jsonl','progress.jsonl','completion-progress.jsonl']
    counts={name:sum(1 for _ in (source/name).open()) for name in names}
    save(out/'saturn-instrument.json',{'schema':'tce-mean-trajectory-instrument-v1','target':str(source.relative_to(root)),'runExit':0,'verifyExit':0,'instrument_status':'target-streams-only',
        'streams':7,'records':753913,'sourceStreamRows':counts,'sourceStreams':{n:sha(source/n) for n in names},'fires':0,'formalClaims':0,'qualifiedAddresses':0,'verifyMissingExpectedFireLedgers':7,
        'scope':'Observed standard offline run/verify returned0. No fireledgers written,7missingexpected. No semantic, nativeequivalence or quality certification.'})
    save(out/'phase-account.json',{'schema':'tce-mean-trajectory-phases-v1','scheduler':{k:scheduler['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'reservation':scheduler['reservation'],
        'actor':{k:e[k] for k in ['wallSecondsAfterImports','setupSecondsAfterImports','CPUSecondsAfterImports','timings','completionTimings','peakCUDAAllocatedBytes','peakCUDAReservedBytes']},'analysisSeconds':read(p/'report.json')['analysisSeconds'],
        'scope':'Nestedcompletionintervalsoverlapdenoiserintervals; do notsum. Operationtimers exclude subsequentABI/finite/outputhash work. Afterimportswall includeschecks/capture/I/O. Scheduler includesorchestration/imports. No matched serving benchmark.'})
    save(out/'storage.json',{'schema':'tce-mean-trajectory-storage-v1','candidateArrays':len(c['fields']),'candidateBytes':sum(x['bytes'] for x in c['fields'].values()),'differenceArrays':len(diff),'differenceBytes':sum(x['bytes'] for x in diff.values()),
        'newRawArrays':len(c['fields'])+len(diff),'newRawArrayBytes':sum(x['bytes'] for x in [*c['fields'].values(),*diff.values()]),'primaryCompactFiles':len(list(p.iterdir())),'primaryCompactBytes':sum(n.stat().st_size for n in p.iterdir()),
        'additionalCompletionLogicalBytes':8729346048,'activationAndProductLogicalBytes':4454350848,'maxLogicalActiveMapBytes':113246208,'localFreeBytesObserved':shutil.disk_usage(root).free,'lastActualCleanupCycle':75,'nextCleanupDueCycle':78,
        'scope':'Completearrays nearcompute; onlycompactmetadata/losslessscalarstreams/actualPNGs local.8.729GBadditional capture not simultaneous GPU residency. Cleanup75 archived677763634Bof2failedjobtransports losslessly,failedoutcomepreserved.84pointerschecked,9remainingname/sizecandidates notdeletioncertificates.'})
    images={}
    for role in ['candidate','native']:
        src=p/(role+'-A.png');dst=root.parent/'obsidian/images'/('zero-neural-cycle75-'+role+'-A.png');shutil.copyfile(src,dst);images[role]={'publishedSHA256':sha(src),'notebookCopySHA256':sha(dst)}
    save(out/'render-qa.json',{'schema':'tce-mean-trajectory-render-qa-v1','images':images,'figureSHA256':sha(root/'docs/results/mean-up-trajectory-figure/mean-up-trajectory.png'),
        'visualInspection':'ActualPNG has redfacet/cubelikegeometry strongerthan72roundedred, butdeformedroundedcorners/darkfaces;bluefaceted/lobedobject. Originalcube/spheregeometrynotrestored. Publishedscalarfigurelegible,legend/labelsunclipped.',
        'scope':'DirectactualPNGandfigureinspection,not independent semantic evaluation or a new pixel comparison.'})
    return {'typedActs':90,'explicitOutputABIs':3139,'testsPassed':339,'newRawArrays':len(c['fields'])+len(diff),'streamRows':counts}


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True))
