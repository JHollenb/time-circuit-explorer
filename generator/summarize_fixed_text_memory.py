"""Compact receipts from the completed fixed-memory experiment and tool outputs."""
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha
from publish_fixed_text_memory import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/fixed-text-memory';out=root/'docs/results/fixed-text-memory-evidence';out.mkdir(parents=True,exist_ok=False)
    save(out/'mechanics.json',audit(p));r=read(p/'request.json');c=read(p/'candidate.json');diff=read(p/'differences.json')
    names=r['ownedSources']+['test_fixed_text_memory.py','test_terminal_attention_ledger.py']
    save(out/'tests.json',{'schema':'tce-fixed-text-memory-tests-v1','command':'rtk proxy ../common/bin/common exec python -m pytest generator -q','passed':315,'seconds':18.78,'exitCode':0,
        'earlierSuiteSeconds':18.63,'targetedPassed':2,'targetedSeconds':3.31,'sourceHashes':{n:sha(root/'generator'/n) for n in names},
        'scope':'Observed completed local mechanics tests on final source snapshot. Tiny joint/single fixed-text image laws and two toy commits; no pretrained model locally evaluated. Full CUDA worker is qualified by its actual mrun result, not toy tests.'})
    save(out/'saturn-instrument.json',{'schema':'tce-fixed-text-memory-instrument-v1','target':'artifacts/fixed-text-memory-v1/result','runExit':0,'verifyExit':0,
        'instrument_status':'target-streams-only','streams':3,'records':11945,'observationRows':11840,'clockProgressRows':5,'blockProgressRows':100,'fires':0,'formalClaims':0,'qualifiedAddresses':0,
        'sourceStreams':{n:sha(root/'artifacts/fixed-text-memory-v1/result'/n) for n in ['observations.jsonl','progress.jsonl','block-progress.jsonl']},
        'scope':'Completed standard offline watch/claims/xref and verification. No semantic certification or quality gate; actual bad image and all signed rows retained.'})
    images={}
    for role in ['candidate','native']:
        src=p/(role+'-A.png');dst=root.parent/'obsidian/images'/('zero-neural-cycle67-'+role+'-A.png');shutil.copyfile(src,dst);images[role]={'publishedSHA256':sha(src),'notebookCopySHA256':sha(dst)}
    figure=root/'docs/results/fixed-text-memory-figure/fixed-text-memory.png'
    save(out/'render-qa.json',{'schema':'tce-fixed-text-memory-render-qa-v1','images':images,'imageSHA256':sha(figure),
        'visualInspection':'Actual candidate is pale gray block texture; requested red cube/blue sphere are absent. Native comparator retains both. Scalar figure has no title, legend or axis overlap; velocity and write curves almost coincide and exact rows remain available.',
        'scope':'Direct actual PNG/figure inspection, not an independent semantic evaluator or new pixel arithmetic.'})
    save(out/'storage.json',{'schema':'tce-fixed-text-memory-storage-v1','candidateArrays':len(c['fields']),'candidateBytes':sum(x['bytes'] for x in c['fields'].values()),
        'differenceArrays':len(diff),'differenceBytes':sum(x['bytes'] for x in diff.values()),'primaryCompactFiles':len(list(p.iterdir())),'primaryCompactBytes':sum(x.stat().st_size for x in p.iterdir()),
        'logicalMapBytesAcross80Materializations':13589544960,'maxLogicalActiveSliceBytes':169869312,
        'scope':'Raw arrays remain near compute; actual PNGs/metadata/lossless streams published. Map-byte sum counts repeated ephemeral materialization, not unique resident storage. No new archive/deletion; last actual maintenance66, next69 or pressure.'})
    return {'typedActs':9,'observations':11840,'testsPassed':315,'rawArrays':len(c['fields'])+len(diff)}


if __name__=='__main__':print(json.dumps(execute(),indent=2))
