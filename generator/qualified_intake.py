"""Publish an independently verified metadata-only component evidence pack."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def publish(collection, verification, pointer, destination):
    collection = Path(collection).resolve(); verification = Path(verification).resolve()
    pointer = Path(pointer).resolve(); out = Path(destination).resolve()
    pack = collection / 'extracted/binding-minimal-result'; v = read(verification / 'verification.json')
    receipt = read(collection / 'collection-receipt.json')
    if sha(pack / 'files.json') != v['compact']['manifest_sha256'] or sha(pack / 'audit.json') != v['audit_sha256']:
        raise ValueError('Verified compact manifest or audit changed')
    files = {str(p.relative_to(pack)): sha(p) for p in pack.rglob('*') if p.is_file() and p != pack / 'files.json'}
    if files != read(pack / 'files.json') or v['compact']['retained_array_members']:
        raise ValueError('Metadata-only compact inventory differs')
    if sha(pointer) != receipt['pointer_sha256'] or sha(collection / 'collection-receipt.json') != v['compact']['collection_receipt_sha256']:
        raise ValueError('Transport receipt or custody pointer differs')
    if any(j['status'] != 'authenticated-local-capture' for j in v['jobs'].values()):
        raise ValueError('Complete captured job/log authentication is required')
    calls = v['actual_installed_cli']
    for row in calls:
        if row['contrast'] == 'exact' and row['matches'] != 1:
            raise ValueError('Exact occurrence retrieval instrument differs')
        if row['contrast'] != 'exact' and row['matches'] != 0:
            raise ValueError('Source/profile abstention instrument differs')
    out.mkdir(parents=True, exist_ok=False)
    for p in pack.rglob('*'):
        if p.is_file():
            if p.is_symlink() or p.suffix not in {'.json', '.jsonl', '.py'}: raise ValueError('Unexpected nonmetadata member')
            target = out / 'compact' / p.relative_to(pack); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
            if sha(target) != sha(p): raise ValueError('Copied compact differs')
    shutil.copytree(verification, out / 'verification')
    shutil.copyfile(pointer, out / 'custody-pointer.json')
    shutil.copyfile(collection / 'collection-receipt.json', out / 'collection-receipt.json')
    plan = read(pack / 'source-plan.json')
    provenance = {'schema': 'tce-qualified-component-intake-v1', 'publisherSHA256': sha(__file__),
        'verificationSHA256': sha(verification / 'verification.json'),
        'jobs': v['jobs'], 'scope': 'Independent compact report/custody/installed-CLI verification. No new scientific or neural execution, raw arrays or parameter payload hydration.',
        'rawEvidence': {'jobID': v['jobs']['native']['job_id'], 'root': plan['output'],
                        'manifestSHA256': sha(pack / 'source-files.json')},
        'copiedFiles': {str(p.relative_to(out)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in out.rglob('*') if p.is_file()}}
    (out / 'provenance.json').write_text(json.dumps(provenance, sort_keys=True, indent=2) + '\n')
    (out / 'files.json').write_text(json.dumps({str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}, sort_keys=True, indent=2) + '\n')
    return {'files': sum(p.is_file() for p in out.rglob('*')), 'bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
            'verificationSHA256': sha(verification / 'verification.json')}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('collection'); p.add_argument('verification'); p.add_argument('pointer'); p.add_argument('destination')
    a = p.parse_args(); print(json.dumps(publish(a.collection, a.verification, a.pointer, a.destination)))
