"""Publish exact cycle69 archive and read-only audit receipts."""
from pathlib import Path
import json
import shutil
from encoder_atoms import read,save,sha


def execute():
    root=Path(__file__).resolve().parent.parent;archiver=root/'generator/storage_transports.py';result=read(root/'artifacts/storage-cleanup-cycle69/result.json')
    if result['files']!=4 or result['archivedBytes']!=1150830575:raise ValueError('Actual archive identity differs')
    for name,files in [('storage-cleanup-cycle69',['plan.json','result.json','events.jsonl']),('storage-audit-cycle69',['result.json'])]:
        source=root/'artifacts'/name;out=root/'docs/results'/name;out.mkdir(parents=True,exist_ok=False)
        for n in files:
            shutil.copyfile(source/n,out/n)
            if sha(source/n)!=sha(out/n):raise ValueError('Published storage receipt differs')
        save(out/'publication.json',{'source':'artifacts/'+name,'archiverSourceSHA256':sha(archiver),'publisherSourceSHA256':sha(__file__),
            'files':{n:sha(out/n) for n in files},'scope':'Four new cold objects full-SHA verified before durable restore pointers and local transport unlink. All extracted evidence/custody retained. Post-audit76pointer/custody+remote regular-file/size joins; older72cold objects not freshly byte-rehashed. Remaining17name/size candidates are not deletion certificates.'})
    return {'archivedFiles':4,'archivedBytes':1150830575,'postAuditPointers':76,'remainingCandidates':17}


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True))
