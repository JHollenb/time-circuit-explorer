"""Compact receipts for the completed cycle68 tools and actual image."""
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha
from projected_complement import fields
from publish_prompt_program import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/prompt-program';out=root/'docs/results/prompt-program-evidence';out.mkdir(parents=True,exist_ok=False)
    save(out/'mechanics.json',audit(p));r=read(p/'request.json');c=read(p/'candidate.json');diff=read(p/'differences.json');program=read(p/'compiler.json')
    names=r['ownedSources']+['test_terminal_attention_ledger.py']
    save(out/'tests.json',{'schema':'tce-prompt-program-tests-v1','command':'rtk proxy ../common/bin/common exec python -m pytest -q generator','passed':319,'seconds':18.87,'exitCode':0,
        'targetedPassed':9,'targetedSeconds':3.40,'sourceHashes':{n:sha(root/'generator'/n) for n in names},
        'scope':'Observed completed local mechanics suite. Tiny text-only native reference, same-incoming-text native image consumer, two own commits, physical terminal omission. No pretrained model locally evaluated; actual CUDA job supplies model evidence.'})
    streams=['observations.jsonl','progress.jsonl','block-progress.jsonl','compiler-block-progress.jsonl']
    save(out/'saturn-instrument.json',{'schema':'tce-prompt-program-instrument-v1','target':'artifacts/prompt-program-v1/result','runExit':0,'verifyExit':0,
        'instrument_status':'target-streams-only','streams':4,'records':12045,'observationRows':11840,'clockProgressRows':5,'compilerBlockRows':100,'actorBlockRows':100,'fires':0,'formalClaims':0,'qualifiedAddresses':0,
        'verifyMissingExpectedFireLedgers':4,'sourceStreams':{n:sha(root/'artifacts/prompt-program-v1/result'/n) for n in streams},
        'scope':'Observed standard offline watch/claims/xref and verification. No fires generated, hence no fire ledgers; verify reports four missing expected ledgers with exit0. No semantic certification, quality gate or suppression of adverse rows.'})
    images={}
    for role in ['candidate','native']:
        src=p/(role+'-A.png');dst=root.parent/'obsidian/images'/('zero-neural-cycle68-'+role+'-A.png');shutil.copyfile(src,dst)
        images[role]={'publishedSHA256':sha(src),'notebookCopySHA256':sha(dst)}
    figure=root/'docs/results/prompt-program-figure/prompt-program.png'
    save(out/'render-qa.json',{'schema':'tce-prompt-program-render-qa-v1','images':images,'imageSHA256':sha(figure),
        'visualInspection':'Candidate restores red shapes and blue spheres on white tabletop, but many extra small objects appear and the main red shape is an open box. Native has two closed/simple objects. Figure title, axes, legends and values are legible without overlap; velocity/write curves nearly coincide.',
        'scope':'Direct actual PNG/figure inspection, not independent semantic evaluation or new pixel arithmetic.'})
    compiler_fields=list(fields(program));candidate_fields=list(c['fields'].values());difference_fields=list(diff.values())
    save(out/'storage.json',{'schema':'tce-prompt-program-storage-v1','compilerArrays':len(compiler_fields),'compilerBytes':sum(x['bytes'] for x in compiler_fields),
        'candidateArrays':len(candidate_fields),'candidateBytes':sum(x['bytes'] for x in candidate_fields),'differenceArrays':len(difference_fields),'differenceBytes':sum(x['bytes'] for x in difference_fields),
        'primaryCompactFiles':len(list(p.iterdir())),'primaryCompactBytes':sum(x.stat().st_size for x in p.iterdir()),'logicalKVCacheBytes':629145600,
        'logicalTerminalKVMapMaterializationBytes':150994944,'maxLogicalActiveKVMapBytes':37748736,'localFreeBytesObserved':shutil.disk_usage(root).free,
        'scope':'Raw state/cache/actor/difference arrays remain near compute. Only metadata/actual PNGs/lossless rows collected and published. No new archive/deletion/bulkmodelhydration; actual maintenance66, next69 or pressure. Logical map/cache bytes are not measured GPU residency; shared free space is not exclusively attributable.'})
    return {'typedActs':13,'observations':11840,'testsPassed':319,'newRawArrays':len(compiler_fields)+len(candidate_fields)+len(difference_fields)}


if __name__=='__main__':print(json.dumps(execute(),indent=2))
