"""Actual receipt and published-scalar summary; never loads real tensors."""
import gzip
import json
from pathlib import Path
import shutil
import statistics
from encoder_atoms import read,save,sha
from publish_gated_even_bound import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/gated-even-bound';out=root/'docs/results/gated-even-bound-evidence';report=read(p/'report.json');scheduler=read(p/'scheduler.json')
    save(out/'mechanics.json',audit(p))
    rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()];strata={}
    for name in ['all','text','image']:
        selected=rows if name=='all' else [r for r in rows if r['stratum']==name];ratios=[r['etaOverRemainingNormUpper'] for r in selected if r['etaOverRemainingNormUpper'] is not None]
        strata[name]={'rows':len(selected),'negativeAlignmentRowsConditional':sum(r['conditionalNegativeAlignment'] for r in selected),'otherRowsSignUnresolved':sum(not r['conditionalNegativeAlignment'] for r in selected),
            'positiveEvenNormLowerRows':sum(r['evenOutputNormInterval'][0]>0 for r in selected),'etaRatioMedian':statistics.median(ratios),'etaRatioMin':min(ratios),'etaRatioMax':max(ratios),'undefinedRatios':len(selected)-len(ratios)}
    save(out/'row-heterogeneity.json',{'schema':'tce-gated-even-bound-row-summary-v1','sourceSHA256':sha(p/'observations.jsonl.gz'),'summarizerSHA256':sha(__file__),'strata':strata,'adverseTextRows':report['retainedAdverseTextRows'],
        'scope':'Scalar-only localization, no trimming. Conditional sign-unresolved does not mean absent cancellation; all dependent rows remain. No prompt replication or fitted correction.'})
    source=root/'artifacts/gated-even-bound-v1/result'
    save(out/'saturn-instrument.json',{'schema':'tce-gated-even-bound-instrument-v1','target':str(source.relative_to(root)),'runExit':0,'verifyExit':0,'instrument_status':'target-streams-only','streams':1,'records':1536,'fires':0,'formalClaims':0,'qualifiedAddresses':0,'verifyMissingExpectedFireLedgers':1,'sourceStreamSHA256':sha(source/'observations.jsonl'),
        'scope':'Observed standard offline run/verify0. No fireledger written,one missingexpected. No numerical/semantic/image/generalization certification.'})
    save(out/'phase-account.json',{'schema':'tce-gated-even-bound-phases-v1','scheduler':{k:scheduler['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'reservation':scheduler['reservation'],'workerWallSeconds':report['workerWallSeconds'],'physicalWork':report['physicalWork'],
        'normCalls':report['numericalDiagnostics']['normCalls'],'scope':'Worker3.562s begins after imports/source verification, includes source hashes/scans/row reductions/retention/final source hashes. Scheduler5.43s includes more startup/finalization. Numerical thread environments1; scheduler reserves4CPU. Zero new model matrix MACs does not mean zero scalar arithmetic; no matched serving speed claim.'})
    save(out/'storage.json',{'schema':'tce-gated-even-bound-storage-v1','sourceInputFiles':16,'sourceFileBytes':1245734912,'newRawArrays':1,'newRawArrayBytes':report['rowBounds']['bytes'],'primaryCompactFiles':len(list(p.iterdir())),'primaryCompactBytes':sum(x.stat().st_size for x in p.iterdir()),'lastActualCleanupCycle':78,'nextCleanupDueCycle':81,'restorePointers':93,'localFreeBytesObserved':shutil.disk_usage(root).free,
        'scope':'Existing scientific products/maps read nearcompute by mmap; one123008Bscalar-array retained there. Compact scalar/evidence only local. No new archive/deletion/coldrehash thiscycle. Prior78cleanup9currentbyte snapshots3.806GB retained;84oldmetadata/9newfullSHA custodyclasses distinct. Shared free-space observation only.'})
    return strata


if __name__=='__main__':print(json.dumps(execute(),indent=2))
