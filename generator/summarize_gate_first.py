"""Record observed cycle71 mechanics, tests, instrument, figure and retention."""
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha
from publish_gate_first import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/gate-first-terminal';out=root/'docs/results/gate-first-terminal-evidence';out.mkdir(parents=True,exist_ok=False)
    save(out/'mechanics.json',audit(p));request=read(p/'request.json');c=read(p/'candidate.json');d=read(p/'differences.json')
    save(out/'tests.json',{'schema':'tce-gate-first-tests-v1','command':'rtk proxy ../common/bin/common exec python -m pytest -q generator','passed':333,'seconds':19.06,'exitCode':0,
        'gateFirstFixtures':6,'sourceHashes':{n:sha(root/'generator'/n) for n in request['ownedSources']},
        'scope':'Observed tiny complete/selected dense references, real-score bound, physical source intervals, stable ties/support alignment, FP32 autocast and nonfinite-ranking refusal. No local pretrained model or native BF16 quality certification.'})
    names=['observations.jsonl','selection-observations.jsonl','progress.jsonl']
    save(out/'saturn-instrument.json',{'schema':'tce-gate-first-instrument-v1','target':'artifacts/gate-first-terminal-v1/result','runExit':0,'verifyExit':0,'instrument_status':'target-streams-only','streams':3,'records':14470,
        'signedComparisonRows':5248,'selectionRows':9216,'progressRows':6,'fires':0,'formalClaims':0,'qualifiedAddresses':0,'verifyMissingExpectedFireLedgers':3,
        'sourceStreams':{n:sha(root/'artifacts/gate-first-terminal-v1/result'/n) for n in names},'scope':'Observed standard offline sweep/verify, no fire ledgers generated; no semantic certification or effect-size gate.'})
    figure=root/'docs/results/gate-first-terminal-figure/gate-first-terminal.png'
    save(out/'render-qa.json',{'schema':'tce-gate-first-render-qa-v1','figureSHA256':sha(figure),'notebookFigureSHA256':sha(root.parent/'obsidian/images/zero-neural-cycle71-gate-first-terminal.png'),
        'visualInspection':'Work bars and three consumer error bars match published scalars; title, annotations, legend, axes and footer legible with no overlap. No new image was generated.',
        'scope':'Direct figure inspection, no new scientific numerical comparison.'})
    save(out/'storage.json',{'schema':'tce-gate-first-storage-v1','candidateArrays':len(c['fields']),'candidateBytes':sum(x['bytes'] for x in c['fields'].values()),'differenceArrays':len(d),'differenceBytes':sum(x['bytes'] for x in d.values()),
        'primaryCompactFiles':len(list(p.iterdir())),'primaryCompactBytes':sum(x.stat().st_size for x in p.iterdir()),'localFreeBytesObserved':shutil.disk_usage(root).free,
        'logicalPhysicalMapBytes':160432128,'preparedBackingLogicalBytes':72351744,'lastActualCleanupCycle':69,'nextCleanupDueCycle':72,
        'scope':'26 raw arrays remain near compute; compact metadata/lossless scalar streams local. No new archive/deletion; actual cleanup69 and76restore pointers retained, due72orpressure. Shared free observation not exclusive attribution; logical bytes not residency.'})
    return {'typedActs':6,'explicitOutputABIs':24,'testsPassed':333,'newRawArrays':len(c['fields'])+len(d)}


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True))
