"""Publish exact cycle72 archive and read-only audit receipts."""
from pathlib import Path
import json
import shutil
from encoder_atoms import read,save,sha


def execute():
    root=Path(__file__).resolve().parent.parent;archiver=root/'generator/storage_transports.py';result=read(root/'artifacts/storage-cleanup-cycle72/result.json')
    if result['files']!=6 or result['archivedBytes']!=991780817:raise ValueError('Actual archive identity differs')
    for name,files in [('storage-cleanup-cycle72',['plan.json','result.json','events.jsonl']),('storage-audit-cycle72',['result.json'])]:
        source=root/'artifacts'/name;out=root/'docs/results'/name;out.mkdir(parents=True,exist_ok=False)
        for n in files:
            shutil.copyfile(source/n,out/n)
            if sha(source/n)!=sha(out/n):raise ValueError('Published storage receipt differs')
        save(out/'publication.json',{'source':'artifacts/'+name,'archiverSourceSHA256':sha(archiver),'publisherSourceSHA256':sha(__file__),
            'files':{n:sha(out/n) for n in files},'scope':'Six new cold objects full-SHA verified before durable restore pointers and local transport unlink. All extracted evidence/custody retained. Post-audit82pointer/custody+remote regular-file/size joins; older76cold objects not freshly byte-rehashed. Remaining11name/size candidates are not deletion certificates.'})
    return {'archivedFiles':6,'archivedBytes':991780817,'postAuditPointers':82,'remainingCandidates':11}


if __name__=='__main__':print(json.dumps(execute(),sort_keys=True))
