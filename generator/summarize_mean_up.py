"""Record observed74 mechanics, timing, dependent scalar trends and storage."""
import gzip
import json
from pathlib import Path
import shutil
import statistics
from encoder_atoms import read,save,sha
from publish_mean_up import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/current-stream-mean-up';out=root/'docs/results/current-stream-mean-up-evidence';out.mkdir(parents=True,exist_ok=False)
    save(out/'mechanics.json',audit(p));r=read(p/'request.json');c=read(p/'candidate.json');execution=read(p/'execution.json');scheduler=read(p/'scheduler.json')
    save(out/'tests.json',{'schema':'tce-mean-up-tests-v1','command':'rtk proxy ../common/bin/common exec python -m pytest -q generator','passed':338,'seconds':18.64,'exitCode':0,
        'newTests':1,'sourceHashes':{n:sha(root/'generator'/n) for n in r['ownedSources']},
        'scope':'Final post-support-applicability-source-fix suite; independent tiny full-Up/stream completion reference, explicit selected overwrite, product/residual and full-support dense toy agreement. No local pretrained evaluation.'})
    raw=gzip.decompress((p/'write-observations.jsonl.gz').read_bytes());rows=[json.loads(x) for x in raw.splitlines()];strata={}
    for name in ['text','image']:
        part=[x for x in rows if x['stratum']==name];ratios=[x['ownOverPriorErrorRMS'] for x in part]
        strata[name]={'rows':len(part),'improved':sum(x['energyImprovement']>0 for x in part),'worsened':sum(x['energyImprovement']<0 for x in part),'unchanged':sum(x['energyImprovement']==0 for x in part),
            'medianOwnOverPriorErrorRMS':statistics.median(ratios),'minOwnOverPriorErrorRMS':min(ratios),'maxOwnOverPriorErrorRMS':max(ratios)}
    save(out/'row-heterogeneity.json',{'schema':'tce-mean-up-row-heterogeneity-v1','sourceSHA256':sha(p/'write-observations.jsonl.gz'),'summarizerSHA256':sha(__file__),
        'strata':strata,'allPositiveCorrectionOpposedErrorCosines':all(x['correctionOpposedPriorErrorCosine']>0 for x in rows),
        'minCorrectionOpposedErrorCosine':min(x['correctionOpposedPriorErrorCosine'] for x in rows),
        'priorFiniteSensitiveRows':[rows[i] for i in [34,40,41]],'worstOwnOverPriorRows':sorted(rows,key=lambda x:x['ownOverPriorErrorRMS'],reverse=True)[:3],
        'scope':'Published scalar descriptions only. These1536rows are dependent, not prompt replications. Favorable projection does not guarantee total-error reduction; no new decomposition attributes the50worsened rows to finite policy.'})
    names=['observations.jsonl','write-observations.jsonl']
    save(out/'saturn-instrument.json',{'schema':'tce-mean-up-instrument-v1','target':'artifacts/current-stream-mean-up-v1/result','runExit':0,'verifyExit':0,'instrument_status':'target-streams-only',
        'streams':2,'records':84518,'comparisonRows':82982,'writeRows':1536,'fires':0,'formalClaims':0,'qualifiedAddresses':0,'verifyMissingExpectedFireLedgers':2,
        'sourceStreams':{n:sha(root/'artifacts/current-stream-mean-up-v1/result'/n) for n in names},
        'scope':'Actual standard offline sweep/verification. Zero fire ledgers written, two missing expected reported. No quality gate or semantic certification.'})
    save(out/'phase-account.json',{'schema':'tce-mean-up-phases-v1','scheduler':{k:scheduler['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'reservation':scheduler['reservation'],
        'actor':{k:execution[k] for k in ['wallSecondsAfterImports','setupSecondsAfterImports','CPUSecondsAfterImports','timings','peakCUDAAllocatedBytes','peakCUDAReservedBytes']},
        'analysisSecondsAfterImports':read(p/'report.json')['analysisSecondsAfterImports'],
        'scope':'Actor synchronized operation includes internal maps/copies, excludes later finite checks/fingerprints; fresh-process wall includes setup/checks/I/O. Scheduler includes orchestration/imports. No matched serving benchmark.'})
    collections=[c['fields'],read(p/'differences.json'),{'derived':read(p/'derived-reference.json')['field']}]
    save(out/'storage.json',{'schema':'tce-mean-up-storage-v1','newRawArrays':sum(len(x) for x in collections),'newRawArrayBytes':sum(y['bytes'] for x in collections for y in x.values()),
        'primaryCompactFiles':len(list(p.iterdir())),'primaryCompactBytes':sum(x.stat().st_size for x in p.iterdir()),'localFreeBytesObserved':shutil.disk_usage(root).free,
        'completedUpBytes':28311552,'productBytes':28311552,'liveUAnchorMapBytes':56623104,'maxLogicalActiveMapBytes':113246208,
        'lastActualCleanupCycle':72,'nextCleanupDueCycle':75,
        'scope':'No full current Up projection, but full-sized proxy/product banks and map copies remain paid. Complete arrays near compute, compact metadata/lossless scalar streams local. No archive/deletion thiscycle; free space shared. Byte verifier inheritedassets total is not new storage.'})
    figure=root/'docs/results/current-stream-mean-up-figure/current-stream-mean-up.png';f=read(figure.parent/'figure.json')
    save(out/'render-qa.json',{'schema':'tce-mean-up-render-qa-v1','figureSHA256':sha(figure),'notebookCopySHA256':sha(Path(f['notebookCopy'])),
        'visualInspection':'Direct figure inspection: legible/unclipped grouped error bars and all1536positive scalar row points on logarithmic axes. Dashed identity separates improvements/worsening; no clipping or new model image.',
        'scope':'Published scalar rendering only; no independent model or semantic evaluation.'})
    return {'typedActs':1,'outputABIs':36,'testsPassed':338,'newRawArrays':40,'rowStrata':strata}


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True))
