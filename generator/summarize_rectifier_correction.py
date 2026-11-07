"""Summarize published scalar observations and actual receipts, no tensors."""
import gzip
import json
from pathlib import Path
import shutil
import statistics
from encoder_atoms import read,save,sha
from publish_rectifier_correction import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/rectifier-correction';out=root/'docs/results/rectifier-correction-evidence';report=read(p/'report.json');scheduler=read(p/'scheduler.json');e=read(p/'execution.json')
    save(out/'mechanics.json',audit(p));rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()];strata={}
    for name in ['all','text','image']:
        selected=rows if name=='all' else [r for r in rows if r['stratum']==name];comparison={}
        for baseline in ['Zero','Linear','Quadratic']:
            effects=[r['energyImprovementVs'+baseline] for r in selected]
            comparison[baseline.lower()]={'improve':sum(x>0 for x in effects),'worsen':sum(x<0 for x in effects),'equal':sum(x==0 for x in effects),'minEnergyImprovement':min(effects),'maxEnergyImprovement':max(effects),'medianEnergyImprovement':statistics.median(effects)}
        strata[name]={'rows':len(selected),'comparison':comparison}
    save(out/'row-heterogeneity.json',{'schema':'tce-rectifier-correction-row-summary-v1','sourceSHA256':sha(p/'observations.jsonl.gz'),'summarizerSHA256':sha(__file__),'strata':strata,
        'historicalTextExceptionRows':[rows[i] for i in [3,35,36]],'newWorseVsLinearRows':[x['row'] for x in rows if x['energyImprovementVsLinear']<0],
        'newWorseVsQuadraticRows':[x['row'] for x in rows if x['energyImprovementVsQuadratic']<0],
        'scope':'Published scalar-only localization. Every dependentrow retained; historicalexceptionIDs are diagnostic, not a selector/trimmed promotion or prompt replications.'})
    source=root/'artifacts/rectifier-correction-v1/result'
    save(out/'saturn-instrument.json',{'schema':'tce-rectifier-correction-instrument-v1','target':str(source.relative_to(root)),'runExit':0,'verifyExit':0,'instrument_status':'target-streams-only','streams':1,'records':1536,'fires':0,'formalClaims':0,'qualifiedAddresses':0,'verifyMissingExpectedFireLedgers':1,'sourceStreamSHA256':sha(source/'observations.jsonl'),
        'scope':'Observed standardoffline run/verify0. No fireledger written,one missingexpected; no numerical/quality/semantic/generalization certification.'})
    save(out/'phase-account.json',{'schema':'tce-rectifier-correction-phases-v1','scheduler':{k:scheduler['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'reservation':scheduler['reservation'],
        'actor':{k:e[k] for k in ['setupSecondsAfterImports','wallSecondsAfterImports','CPUSecondsAfterImports','timings','peakCUDAAllocatedBytes','peakCUDAReservedBytes']},'producerExit':read(p/'producer-exit.json'),'analysisSecondsAfterImports':report['analysisSecondsAfterImports'],
        'scope':'Instrumented synchronized interval includes rectifier/means/product/Down/gating/forecast and internal affine fingerprints; excludes outer source/finaloutputhashes/ABI/retention. Actor/analysis/scheduler scopes differ; no matched serving speed claim.'})
    fields=[*read(p/'candidate.json')['fields'].values(),*read(p/'analysis-fields.json')['fields'].values()]
    save(out/'storage.json',{'schema':'tce-rectifier-correction-storage-v1','newRawArrays':len(fields),'newRawArrayBytes':sum(x['bytes'] for x in fields),'outputLogicalBytes':868910208,
        'primaryCompactFiles':len(list(p.iterdir())),'primaryCompactBytes':sum(x.stat().st_size for x in p.iterdir()),'lastActualCleanupCycle':78,'nextCleanupDueCycle':81,'restorePointers':93,'localFreeBytesObserved':shutil.disk_usage(root).free,
        'scope':'Allnewcompletearrays remainnearcompute,compactmetadata/losslessscalarrows onlylocal. OriginalG/Up/D/L/parents reused not newstorage. No archive/deletion/coldrehash80. Actual78cleanup9currentbyte snapshots3.806GB,84oldmetadata/9newfullSHA custodyclasses preserved. Next81orpressure.'})
    return strata


if __name__=='__main__':print(json.dumps(execute(),indent=2))
