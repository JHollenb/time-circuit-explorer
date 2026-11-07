"""Compact receipts from actual published metadata and completed tool observations."""
import json
from pathlib import Path
import shutil
from encoder_atoms import read,save,sha
from publish_terminal_trajectory import audit


def execute():
    root=Path(__file__).resolve().parent.parent;p=root/'docs/results/terminal-trajectory';out=root/'docs/results/terminal-trajectory-evidence';out.mkdir(parents=True,exist_ok=False)
    save(out/'mechanics.json',audit(p));r=read(p/'request.json');c=read(p/'candidate.json');diff=read(p/'differences.json')
    names=r['ownedSources']+['test_terminal_trajectory.py','test_terminal_attention_ledger.py']
    save(out/'tests.json',{'schema':'tce-terminal-trajectory-tests-v1','command':'rtk proxy ../common/bin/common exec python -m pytest generator -q',
        'passed':313,'seconds':18.59,'exitCode':0,'targetedPassed':5,'targetedSeconds':3.20,
        'sourceHashes':{n:sha(root/'generator'/n) for n in names},'scope':'Observed completed local mechanics suite; includes native keyword/positional/mixed observer forwarding and tiny two-clock toy. No pretrained model locally evaluated.'})
    save(out/'saturn-instrument.json',{'schema':'tce-terminal-trajectory-instrument-v1','target':'artifacts/terminal-trajectory-v2/result','runExit':0,'verifyExit':0,
        'instrument_status':'target-streams-only','streams':2,'records':11845,'observationRows':11840,'progressRows':5,'fires':0,'formalClaims':0,'qualifiedAddresses':0,
        'sourceStreams':{n:sha(root/'artifacts/terminal-trajectory-v2/result'/n) for n in ['observations.jsonl','progress.jsonl']},
        'scope':'Completed standard offline watch/claims/xref and verification. No fire ledgers expected without fires; dedicated audits qualify bytes/work/ownstate. No semantic certification.'})
    images={}
    for role in ['candidate','native']:
        src=p/(role+'-A.png');dst=root.parent/'obsidian/images'/('zero-neural-cycle66-'+role+'-A.png');shutil.copyfile(src,dst);images[role]={'publishedSHA256':sha(src),'notebookCopySHA256':sha(dst)}
    figure=root/'docs/results/terminal-trajectory-figure-v2/terminal-trajectory.png'
    save(out/'render-qa.json',{'schema':'tce-terminal-trajectory-render-qa-v1','images':images,'latestImageSHA256':sha(figure),
        'previousImageSHA256':sha(root/'docs/results/terminal-trajectory-figure/terminal-trajectory.png'),
        'visualInspection':'Actual native/candidate images retain red cube left of blue sphere. First plot legend lies over data; separate v2 moves it outside axes, with no title/legend/axis overlap.',
        'scope':'Direct visual inspection of actual PNGs and scalar plots, not independent prompt-quality scoring or new pixel arithmetic.'})
    save(out/'storage.json',{'schema':'tce-terminal-trajectory-storage-v1','candidateArrays':len(c['fields']),'candidateBytes':sum(x['bytes'] for x in c['fields'].values()),
        'differenceArrays':len(diff),'differenceBytes':sum(x['bytes'] for x in diff.values()),'primaryCompactFiles':len(list(p.iterdir())),
        'primaryCompactBytes':sum(x.stat().st_size for x in p.iterdir()),'scope':'Raw arrays remain near compute; original candidate/native PNGs, metadata and lossless streams published. No model/table hydration.'})
    return {'typedActs':9,'observations':11840,'testsPassed':313,'rawArrays':len(c['fields'])+len(diff)}


if __name__=='__main__':print(json.dumps(execute(),indent=2))
