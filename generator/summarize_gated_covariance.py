"""Observed evidence and published scalar heterogeneity; no real tensor math."""
import gzip
import json
from pathlib import Path
import shutil
import statistics
from encoder_atoms import read,save,sha
from publish_gated_covariance import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/own-state-gated-covariance';out=root/'docs/results/own-state-gated-covariance-evidence';out.mkdir(parents=True,exist_ok=False)
    save(out/'mechanics.json',audit(p));r=read(p/'request.json');c=read(p/'candidate.json');e=read(p/'execution.json');scheduler=read(p/'scheduler.json')
    save(out/'tests.json',{'schema':'tce-gated-covariance-tests-v1','command':'rtk proxy ../common/bin/common exec python -m pytest -q generator','passed':340,'seconds':18.89,'exitCode':0,'newTests':1,
        'sourceHashes':{name:sha(root/'generator'/name) for name in r['ownedSources']},'scope':'Independent tiny scalar Up/Down, omittedcovariance, productpolicy expansion, fullABI and constantfamily Pythagoreanreference. No local pretrained model evaluations.'})
    rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()];strata={}
    for name in ['all','text','image']:
        selected=rows if name=='all' else [x for x in rows if x['stratum']==name]
        dots=[x['totalEnergy']+x['meanEnergy']-x['meanRResidualEnergy'] for x in selected]
        strata[name]={'rows':len(selected),'improved':sum(x['meanCorrectionEnergyImprovement']>0 for x in selected),'worsened':sum(x['meanCorrectionEnergyImprovement']<0 for x in selected),'positiveMeanProjection':sum(x>0 for x in dots),'negativeMeanProjection':sum(x<0 for x in dots),
            'minTwiceTotalMeanInner':min(dots),'medianTwiceTotalMeanInner':statistics.median(dots),'maxTwiceTotalMeanInner':max(dots),'minEnergyImprovement':min(x['meanCorrectionEnergyImprovement'] for x in selected),'medianEnergyImprovement':statistics.median(x['meanCorrectionEnergyImprovement'] for x in selected),'maxEnergyImprovement':max(x['meanCorrectionEnergyImprovement'] for x in selected)}
    save(out/'row-heterogeneity.json',{'schema':'tce-gated-covariance-row-summary-v1','sourceSHA256':sha(p/'observations.jsonl.gz'),'summarizerSHA256':sha(__file__),'strata':strata,'scope':'Published scalars only. Twice<total,mean>=totalEnergy+meanEnergy-residualEnergy; positive projection neednot outweigh correctionenergy. Dependentrows,notpromptreplications.'})
    source=root/'artifacts/own-state-gated-covariance-v1/result';names=['observations.jsonl','channel-observations.jsonl']
    save(out/'saturn-instrument.json',{'schema':'tce-gated-covariance-instrument-v1','target':str(source.relative_to(root)),'runExit':0,'verifyExit':0,'instrument_status':'target-streams-only','streams':2,'records':19968,'fires':0,'formalClaims':0,'qualifiedAddresses':0,'verifyMissingExpectedFireLedgers':2,
        'sourceStreams':{n:sha(source/n) for n in names},'scope':'Observed standard offline run/verify0. No fireledgers written,two missingexpected. No nativeequivalence/semantic/quality certification.'})
    save(out/'phase-account.json',{'schema':'tce-gated-covariance-phases-v1','scheduler':{k:scheduler['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'reservation':scheduler['reservation'],
        'actor':{k:e[k] for k in ['setupSecondsAfterImports','wallSecondsAfterImports','CPUSecondsAfterImports','timings','peakCUDAAllocatedBytes','peakCUDAReservedBytes']},'analysisSecondsAfterImports':read(p/'report.json')['analysisSecondsAfterImports'],
        'scope':'4.530s synchronized operation includes internalaffine fingerprints; excludesouterinput/postchecks/finalhash/arraywrite.37.147sactorwall includes those overheads. Notservingtime or speedgain.'})
    save(out/'storage.json',{'schema':'tce-gated-covariance-storage-v1','newRawArrays':len(c['fields']),'newRawArrayBytes':sum(x['bytes'] for x in c['fields'].values()),'outputLogicalBytes':1803313368,'originalParameterBytes':245366784,'inputLogicalBytes':101548032,
        'primaryCompactFiles':len(list(p.iterdir())),'primaryCompactBytes':sum(x.stat().st_size for x in p.iterdir()),'lastActualCleanupCycle':75,'nextCleanupDueCycle':78,'localFreeBytesObserved':shutil.disk_usage(root).free,
        'scope':'28completeFP64 arrays nearcompute,includes453MBpromotedU/Dmaps.10oldsourcefields reusednotnewstorage. CompactJSON/losslessscalarstreams/figure local. No newarchive/deletion76; sharedfreeobservation not attribution.'})
    old=root/'docs/results/own-state-gated-covariance-figure/own-state-gated-covariance.png';new=root/'docs/results/own-state-gated-covariance-figure-v2/own-state-gated-covariance.png'
    save(out/'render-qa.json',{'schema':'tce-gated-covariance-render-qa-v1','supersededImageSHA256':sha(old),'publishedImageSHA256':sha(new),'plotSourceSHA256':sha(root/'generator/plot_gated_covariance_v2.py'),
        'originalFinding':'Firstfigurecolorbarlabel clippedright,legendoverlaidbars; originalexecutedplot/figurepreserved.',
        'repair':'Newv2plot withmoremarginandseparatelegend; actualv2directlyinspected, labels/legend/colorbar unclipped. SignedsmallGramterms retained inlabels.',
        'scope':'Display-only correction on publishedscalars; no scientificrun/arraycomparison changed.'})
    return {'typedActs':1,'outputABIs':28,'testsPassed':340,'rows':strata,'newRawArrays':len(c['fields'])}


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True))
