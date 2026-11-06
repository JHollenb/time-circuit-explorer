"""Read-only cold-pointer audit and bounded transport-name inventory."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(destination, receipts, roots, minimum_bytes=128 << 20):
    pointers = []; jobs = set()
    for receipt in receipts:
        result = read(receipt)
        if result['status'] != 'complete': raise ValueError('Cleanup receipt is not complete')
        for row in result['rows']:
            path = Path(row['path']); pointer = path.with_name(path.name + '.cold.json')
            if path.exists() or read(pointer) != row or sha(row['custodyPath']) != row['custodySHA256']:
                raise ValueError('Transport presence, restoration pointer or custody differs')
            pointers.append({'receipt': str(Path(receipt).resolve()), 'pointer': str(pointer),
                             'transportAbsent': True, 'recordMatchesArchiveReceipt': True,
                             'remote': row['remote'], 'bytes': row['bytes'], 'sha256': row['sha256']})
            jobs.add(row['jobID'])
    # Recheck type/size for all objects. Full-byte verification belongs to the
    # archive transaction and is not falsely claimed again for historical copies.
    remote = '''import json,stat,sys
from pathlib import Path
out=[]
for row in json.load(sys.stdin):
 p=Path(row['remote'])
 if not p.is_relative_to('/mnt/hdd8tb/cold-storage/mbp1/time-circuit-explorer/transports/sha256') or p.is_symlink() or not stat.S_ISREG(p.stat().st_mode) or p.stat().st_size!=row['bytes']:raise ValueError('Cold object identity/type/size differs')
 out.append({'remote':str(p),'bytes':p.stat().st_size,'regular':True})
print(json.dumps(out))
'''
    proc = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast', 'python3', '-c',
                           __import__('shlex').quote(remote)], input=json.dumps(pointers), text=True,
                          capture_output=True, check=True, timeout=60)
    checked = json.loads(proc.stdout)
    scan = subprocess.run(['rg', '--files', '--hidden', '--no-ignore', '-g', 'archive.tar.gz', '-g', 'worker.log.gz',
                           *[str(Path(p).resolve()) for p in roots]], text=True, capture_output=True, check=True, timeout=60)
    candidates = []
    for name in scan.stdout.splitlines():
        p = Path(name)
        if p.is_file() and not p.is_symlink() and p.stat().st_size >= minimum_bytes:
            candidates.append({'path': str(p), 'bytes': p.stat().st_size})
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=False)
    value = {'schema': 'tce-storage-audit-v1', 'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'auditorSHA256': sha(__file__), 'scope': 'Read-only transport name/size review, restoration-pointer/custody joins and fresh remote type/size checks. Candidates are not deletion certificates or unique physical bytes.',
        'artifactRoots': [str(Path(p).resolve()) for p in roots], 'minimumBytes': minimum_bytes,
        'candidateFiles': len(candidates), 'candidateLogicalBytes': sum(r['bytes'] for r in candidates),
        'candidates': sorted(candidates, key=lambda r: r['path']), 'pointerChecks': pointers, 'remoteMetadataChecks': checked,
        'remoteCheckScope': 'Fresh regular-file/size checks for all listed cold objects. Historical objects are not full-byte rehashed by this audit; each original archive transaction retains its own full-SHA receipt.',
        'localFreeBytesObserved': shutil.disk_usage(destination).free}
    with (destination / 'result.json').open('x') as stream: json.dump(value, stream, indent=2, sort_keys=True)
    return {'candidateFiles': len(candidates), 'candidateLogicalBytes': value['candidateLogicalBytes'],
            'pointerChecks': len(pointers), 'remoteMetadataChecks': len(checked), 'localFreeBytesObserved': value['localFreeBytesObserved']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('destination'); p.add_argument('--receipt', action='append', required=True)
    p.add_argument('--root', action='append', required=True); p.add_argument('--minimum-bytes', type=int, default=128 << 20)
    a = p.parse_args(); print(json.dumps(audit(a.destination, a.receipt, a.root, a.minimum_bytes)))
