"""Aggregate published scalar rows and observed receipts without real tensor math."""
import gzip
import json
from pathlib import Path
import shutil
import statistics
from encoder_atoms import read,save,sha
from publish_odd_gate import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/odd-gate';out=root/'docs/results/odd-gate-evidence';out.mkdir(parents=True,exist_ok=False)
    save(out/'mechanics.json',audit(p));r=read(p/'request.json');c=read(p/'candidate.json');a=read(p/'analysis-fields.json');e=read(p/'execution.json');scheduler=read(p/'scheduler.json')
    save(out/'tests.json',{'schema':'tce-odd-gate-tests-v1','command':'rtk proxy ../common/bin/common exec python -m pytest -q generator','passed':342,'seconds':18.51,'exitCode':0,'newTests':1,
        'sourceHashes':{n:sha(root/'generator'/n) for n in r['ownedSources']},'scope':'Independent tiny scalar gate/centeredproduct/Down/forecast, finite admitted activation odd/even product closure, fullGram and stream-centered Pythagorean reference. No local pretrained array computations.'})
    rows=[json.loads(x) for x in gzip.decompress((p/'observations.jsonl.gz').read_bytes()).splitlines()];strata={}
    for name in ['all','text','image']:
        selected=rows if name=='all' else [x for x in rows if x['stratum']==name]
        strata[name]={'rows':len(selected),'improvedVsZero':sum(x['energyImprovementVsZero']>0 for x in selected),'worsenedVsZero':sum(x['energyImprovementVsZero']<0 for x in selected),
            'improvedVsLinear':sum(x['energyImprovementVsLinear']>0 for x in selected),'worsenedVsLinear':sum(x['energyImprovementVsLinear']<0 for x in selected),
            'positiveQuadraticBaselineAlignment':sum(x['quadraticBaselineInner']>0 for x in selected),'negativeQuadraticBaselineAlignment':sum(x['quadraticBaselineInner']<0 for x in selected),
            'positiveAlignmentButWorseVsLinear':sum(x['quadraticBaselineInner']>0 and x['energyImprovementVsLinear']<0 for x in selected),
            'medianEnergyImprovementVsLinear':statistics.median(x['energyImprovementVsLinear'] for x in selected),
            'minEnergyImprovementVsLinear':min(x['energyImprovementVsLinear'] for x in selected),'maxEnergyImprovementVsLinear':max(x['energyImprovementVsLinear'] for x in selected)}
    adverse=[x for x in rows if x['energyImprovementVsLinear']<0];text=[x for x in rows if x['stratum']=='text'];share=sum(x['fullResidualEnergy'] for x in adverse)/sum(x['fullResidualEnergy'] for x in text)
    save(out/'row-heterogeneity.json',{'schema':'tce-odd-gate-row-summary-v1','sourceSHA256':sha(p/'observations.jsonl.gz'),'summarizerSHA256':sha(__file__),'strata':strata,'adverseVsLinear':adverse,
        'adverseRowsShareOfFullTextResidualEnergy':share,'worsenedVsZeroRows':[x['row'] for x in rows if x['energyImprovementVsZero']<0],
        'scope':'Published scalar-only localization. All rows remain in primary aggregate; adverse-subset share is retrospective diagnosis, not trimmed accuracy/promotion. Dependent rows, not prompt replications. Positive alignment need not outweigh correction energy; finite forecast regrouping separately measured.'})
    source=root/'artifacts/odd-gate-v1/result'
    save(out/'saturn-instrument.json',{'schema':'tce-odd-gate-instrument-v1','target':str(source.relative_to(root)),'runExit':0,'verifyExit':0,'instrument_status':'target-streams-only','streams':1,'records':1536,
        'fires':0,'formalClaims':0,'qualifiedAddresses':0,'verifyMissingExpectedFireLedgers':1,'sourceStreamSHA256':sha(source/'observations.jsonl'),
        'scope':'Observed standard offline run/verify0. No fireledger written,one missingexpected. No numerical/semantic/generalization/quality certification.'})
    save(out/'phase-account.json',{'schema':'tce-odd-gate-phases-v1','scheduler':{k:scheduler['result'][k] for k in ['elapsed_s','peak_rss_mb','peak_vram_mb']},'reservation':scheduler['reservation'],
        'actor':{k:e[k] for k in ['setupSecondsAfterImports','wallSecondsAfterImports','CPUSecondsAfterImports','timings','peakCUDAAllocatedBytes','peakCUDAReservedBytes']},'producerExit':read(p/'producer-exit.json'),
        'analysisSecondsAfterImports':read(p/'report.json')['analysisSecondsAfterImports'],
        'scope':'0.817s instrumented interval includes internal affine fingerprints, excludes outer/post hashes/ABI/retention; actor14.855s afterimports,analysis15.578s,scheduler43.15s differ in scope. No serving or matched speed claim.'})
    fields=[*c['fields'].values(),*a['fields'].values()]
    save(out/'storage.json',{'schema':'tce-odd-gate-storage-v1','newRawArrays':len(fields),'newRawArrayBytes':sum(x['bytes'] for x in fields),'outputLogicalBytes':1133213016,
        'primaryCompactFiles':len(list(p.iterdir())),'primaryCompactBytes':sum(x.stat().st_size for x in p.iterdir()),'lastActualCleanupCycle':78,'nextCleanupDueCycle':81,
        'localFreeBytesObserved':shutil.disk_usage(root).free,'cleanupFiles':9,'cleanupBytes':3806240603,'restorePointers':93,
        'scope':'25newcompletearrays staynearcompute. Priorstates/maps reusednotnewstorage. Cleanup78 current-byte snapshots explicitly do not repair incomplete historical scientific custody. Older84coldobjectsmetadata-only,new9fullSHA/durability/restorechecks. Sharedfreeobservation not exclusive attribution.'})
    old=root/'docs/results/odd-gate-figure/odd-gate.png';figure=root/'docs/results/odd-gate-figure-v2/odd-gate.png'
    save(out/'render-qa.json',{'schema':'tce-odd-gate-render-v1','supersededImageSHA256':sha(old),'imageSHA256':sha(figure),'plotSourceSHA256':sha(root/'generator/plot_odd_gate_v2.py'),
        'originalFinding':'Text full/centered bar labels nearly touch; originalexecutedplot/source retained.',
        'repair':'Separatev2staggeredcenteredlabels; actualv2directlyinspected,labels/legend/Gram/colorbar/footnotes legible andunclipped. No scientificrun/data change.'})
    names=['odd_gate.py','odd_gate_worker.py','odd_gate_ledger.py','odd_gate_analysis.py','odd_gate_study.py','test_odd_gate.py','publish_odd_gate.py']
    save(out/'advisor-prestage.json',{'schema':'tce-odd-gate-advisor-prestage-v1','cycle':78,'advisor':'/root/astra_zero_forward_advisor','model':'gpt-6-astra','reviewedSources':{n:sha(root/'generator'/n) for n in names},
        'planSHA256':sha(p/'frozen-plan.md'),'verdict':'Proceed single guarded assay; no material pre-freeze blocker. Exact gate slice/factor/centering/mask/fullforecast/finiteH-F boundary/isolation/exit/work/ABIs/lineage/byteaudit/resources reviewed. No advisor models/tests/arrays/edits.',
        'resourceScope':'6144RAM4096VRAM4CPU4GBdisk120expected360cap,no retries; actualpeaksunmeasured atreview.',
        'fullNewestPostSurvey':'../research-review-cycle78/advisor-survey.json','nextFullSurveyDueCycle':81})
    return {'rows':strata,'adverseRowsShareOfTextResidualEnergy':share,'newRawArrays':len(fields),'newRawArrayBytes':sum(x['bytes'] for x in fields)}


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True))
