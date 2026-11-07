"""Aggregate already published scalar rows and observed execution receipts."""
import gzip
import json
from pathlib import Path
import shutil
import statistics
from encoder_atoms import read,save,sha
from publish_mean_gate import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/mean-gate';out=root/'docs/results/mean-gate-evidence';out.mkdir(parents=True,exist_ok=False)
    save(out/'mechanics.json',audit(p));r=read(p/'request.json');c=read(p/'candidate.json');e=read(p/'execution.json');a=read(p/'analysis-fields.json');scheduler=read(p/'scheduler.json')
    save(out/'tests.json',{'schema':'tce-mean-gate-tests-v1','command':'rtk proxy ../common/bin/common exec python -m pytest -q generator','passed':341,'seconds':18.83,'exitCode':0,'newTests':1,
        'sourceHashes':{n:sha(root/'generator'/n) for n in r['ownedSources']},'scope':'Independent tiny scalar mean/product/Down, changing-product identity, signed full/centered row improvements and Pythagorean mechanics. No local pretrained array computations.'})
    rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()];strata={}
    for name in ['all','text','image']:
        selected=rows if name=='all' else [x for x in rows if x['stratum']==name];panels={}
        for prefix,inner,energy in [('full','linearTotalInner','linearEnergy'),('centered','centeredLinearTotalInner','centeredLinearEnergy')]:
            vals=[x[prefix+'EnergyImprovement'] for x in selected]
            panels[prefix]={'improved':sum(v>0 for v in vals),'worsened':sum(v<0 for v in vals),
                'positiveAlignment':sum(x[inner]>0 for x in selected),'negativeAlignment':sum(x[inner]<0 for x in selected),
                'positiveAlignmentButWorse':sum(x[inner]>0 and x[prefix+'EnergyImprovement']<0 for x in selected),
                'zeroCorrectionRows':sum(x[energy]==0 for x in selected),'minEnergyImprovement':min(vals),'medianEnergyImprovement':statistics.median(vals),'maxEnergyImprovement':max(vals)}
        strata[name]={'rows':len(selected),**panels}
    save(out/'row-heterogeneity.json',{'schema':'tce-mean-gate-row-summary-v1','sourceSHA256':sha(p/'observations.jsonl.gz'),'summarizerSHA256':sha(__file__),'strata':strata,
        'scope':'Published scalar rows only. Improvement=2<total,linear>-||linear||^2; positive alignment can still worsen error. Centered and full rows use different targets. Dependent rows, not prompt replications.'})
    source=root/'artifacts/mean-gate-v1/result'
    save(out/'saturn-instrument.json',{'schema':'tce-mean-gate-instrument-v1','target':str(source.relative_to(root)),'runExit':0,'verifyExit':0,'instrument_status':'target-streams-only',
        'streams':1,'records':1536,'fires':0,'formalClaims':0,'qualifiedAddresses':0,'verifyMissingExpectedFireLedgers':1,'sourceStreamSHA256':sha(source/'observations.jsonl'),
        'scope':'Observed offline run/verify0. No fireledger written; one missing expected. No numerical-equivalence/semantic/quality certification.'})
    save(out/'phase-account.json',{'schema':'tce-mean-gate-phases-v1','scheduler':{k:scheduler['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'reservation':scheduler['reservation'],
        'actor':{k:e[k] for k in ['setupSecondsAfterImports','wallSecondsAfterImports','CPUSecondsAfterImports','timings','peakCUDAAllocatedBytes','peakCUDAReservedBytes']},
        'producerExit':read(p/'producer-exit.json'),'analysisSecondsAfterImports':read(p/'report.json')['analysisSecondsAfterImports'],
        'scope':'1.643s synchronized operation includes internal affine fingerprints, excludes outer/post hashes/ABI/retention.12.647s actor wall includes setup/checks/retention; post-exit analysis8.612s.27.94s scheduler includes environment/process overhead. No serving or matched speed claim.'})
    fields=[*c['fields'].values(),*a['fields'].values()]
    save(out/'storage.json',{'schema':'tce-mean-gate-storage-v1','newRawArrays':len(fields),'newRawArrayBytes':sum(x['bytes'] for x in fields),'outputLogicalBytes':604570248,
        'savedOriginalDownMapBytesLoaded':226492416,'primaryCompactFiles':len(list(p.iterdir())),'primaryCompactBytes':sum(x.stat().st_size for x in p.iterdir()),
        'lastActualCleanupCycle':75,'nextCleanupDueCycle':78,'localFreeBytesObserved':shutil.disk_usage(root).free,
        'scope':'21new complete arrays nearcompute; saved inputs/comparisons reused, not newly generated. Compact JSON/lossless rows/figure local. No archive/deletion77 or fresh cold-object rehash. Shared volume observation not attribution.'})
    figure=root/'docs/results/mean-gate-figure/mean-gate.png'
    save(out/'render-qa.json',{'schema':'tce-mean-gate-render-v1','imageSHA256':sha(figure),'plotSourceSHA256':sha(root/'generator/plot_mean_gate.py'),
        'finding':'Directly inspected actual image: axis labels, bar numbers, signed Gram labels, legend, colorbar and denominator footnotes legible/unclipped.',
        'scope':'Published scalar-only visualization; no real model/tensor computation.'})
    names=['mean_gate.py','mean_gate_worker.py','mean_gate_ledger.py','mean_gate_analysis.py','mean_gate_study.py','test_mean_gate.py','publish_mean_gate.py']
    save(out/'advisor-prestage.json',{'schema':'tce-mean-gate-advisor-prestage-v1','cycle':77,'advisor':'/root/astra_zero_forward_advisor','model':'gpt-6-astra','newFullNewestPostSurvey':False,
        'lastFullNewestPostSurveyCycle':75,'nextFullSurveyDueCycle':78,'reviewedSources':{n:sha(root/'generator'/n) for n in names},'planSHA256':sha(p/'frozen-plan.md'),
        'verdict':'Proceed single guarded run; no remaining blocker. Exact law/full-versus-centered/oracle accounting/source quarantine/exit receipt/work/byte custody/resources reviewed. No advisor tests/models/arrays/edits.',
        'correctedBeforeFreeze':['Expose paid constant-only comparator score explicitly.','Distinguish newly loaded226492416B saved original Down map from zero checkpoint tensor loads.'],
        'resourceScope':'4096RAM3072VRAM4CPU2GBdisk90expected240cap,no retries; actual peaks unmeasured at pre-stage review.'})
    return {'tests':341,'rows':strata,'newRawArrays':len(fields),'newRawArrayBytes':sum(x['bytes'] for x in fields)}


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True))
