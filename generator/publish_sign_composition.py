"""Byte-verify saved composition fields; publish compact reports without tensors."""
import argparse
import gzip
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from encoder_atoms import read, save, sha
from publish_context_statistics import checked_collection
from publish_source_context import REMOTE


def verify(root):
    root = Path(root).resolve(); c, j, r = checked_collection(root)
    p = root / 'result'; seal = read(p / 'analysis-seal.json')
    base = j['result']['diagnostics']['cwd']
    source = read(p / 'source.json')
    roots = [base, str(Path(source['rawManifest']['path']).parent), str(Path(source['parentManifest']['path']).parent)]
    items = [*seal['verifiedReads'], *seal['arrays']]
    items += [{'path': base + '/response-result/' + f.name, 'sha256': sha(f)} for f in p.iterdir()]
    proc = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'beast',
                           'python3 -c ' + shlex.quote(REMOTE)], input=json.dumps({'allowedRoots': roots, 'items': items}),
                          text=True, capture_output=True, check=True, timeout=120)
    value = json.loads(proc.stdout)
    value.update(schema='tce-sign-composition-retained-bytes-v1', jobID=j['job_id'],
                 analysisSealSHA256=sha(p / 'analysis-seal.json'), verifierSourceSHA256=sha(__file__),
                 scope='Full byte hashes of every consumed saved source and complete derived defect field. No model access or local tensor hydration.')
    save(root / ('retained-array-verification-' + sha(__file__)[:12] + '.json'), value)
    return {'files': len(value['files']), 'bytes': value['totalBytesVerified']}


def publish(root, destination):
    root = Path(root).resolve(); c, j, r = checked_collection(root)
    p = root / 'result'; verified = root / ('retained-array-verification-' + sha(__file__)[:12] + '.json'); v = read(verified)
    if v['analysisSealSHA256'] != sha(p / 'analysis-seal.json') or v['verifierSourceSHA256'] != sha(__file__):
        raise ValueError('Retained byte verification differs')
    out = Path(destination).resolve(); out.mkdir(parents=True, exist_ok=False); transports = []
    for f in p.iterdir():
        if f.suffix == '.jsonl':
            data = f.read_bytes(); dest = out / (f.name + '.gz'); dest.write_bytes(gzip.compress(data, mtime=0))
            if gzip.decompress(dest.read_bytes()) != data: raise ValueError('Gzip differs')
            transports.append({'source': f.name, 'sourceSHA256': sha(f), 'sourceBytes': len(data),
                               'published': dest.name, 'publishedSHA256': sha(dest), 'publishedBytes': dest.stat().st_size})
        else: shutil.copyfile(f, out / f.name)
    shutil.copyfile(verified, out / 'retained-array-verification.json')
    save(out / 'provenance.json', {'schema': 'tce-sign-composition-public-v1', 'jobID': j['job_id'],
         'requestSHA256': r['requestSHA256'], 'payloadCustody': c['custody'], 'schedulerResult': c['result'],
         'completeCollectedInventory': c['inventory'], 'losslessStreams': transports,
         'scope': 'Retrospective composition geometry, no predictive next-block or arbitrary-prompt image claim. Raw fields stay near compute.'})
    save(out / 'files.json', {f.name: sha(f) for f in out.iterdir()})
    return {'files': len(list(out.iterdir())), 'bytes': sum(f.stat().st_size for f in out.iterdir())}


def intake(collection, verification, destination):
    """Publish the already authenticated metadata and installed-CLI audit."""
    collection = Path(collection).resolve(); verification = Path(verification).resolve()
    pack = collection / 'extracted/binding-minimal-result'; v = read(verification / 'verification.json')
    if sha(pack / 'audit.json') != v['audit_sha256'] or sha(pack / 'files.json') != v['compact']['manifest_sha256']:
        raise ValueError('Qualified intake differs')
    expected = read(pack / 'files.json')
    actual = {str(f.relative_to(pack)): sha(f) for f in pack.rglob('*') if f.is_file() and f.name != 'files.json'}
    if actual != expected or v['compact']['retained_array_members']:
        raise ValueError('Compact inventory differs or contains arrays')
    out = Path(destination).resolve(); out.mkdir(parents=True, exist_ok=False)
    for f in pack.rglob('*'):
        if f.is_file():
            if f.suffix not in {'.json', '.jsonl', '.py'}: raise ValueError('Unexpected nonmetadata member')
            dest = out / 'compact' / f.relative_to(pack); dest.parent.mkdir(exist_ok=True, parents=True); shutil.copyfile(f, dest)
    shutil.copytree(verification, out / 'verification')
    for f in collection.iterdir():
        if f.is_file() and f.suffix == '.json': shutil.copyfile(f, out / f.name)
    pointer = collection.parent / (collection.name + '-custody-pointer.json')
    if sha(pointer) != read(collection / 'collection-receipt.json')['pointer_sha256']:
        raise ValueError('Custody pointer differs')
    shutil.copyfile(pointer, out / 'custody-pointer.json')
    save(out / 'provenance.json', {'schema': 'tce-row-sign-intake-public-v1', 'verificationSHA256': sha(verification / 'verification.json'),
         'scope': 'Neighboring native/audit evidence independently collected and verified. No new native work, activations or weight payloads copied.',
         'rawEvidence': {'jobID': 'job-74efe5506ecf', 'root': '/mnt/big/mdb/research/job-74efe5506ecf/qwen-row-sign-result',
                         'manifestSHA256': sha(pack / 'source-files.json')},
         'copiedFiles': {str(f.relative_to(out)): {'bytes': f.stat().st_size, 'sha256': sha(f)} for f in out.rglob('*') if f.is_file()}})
    save(out / 'files.json', {str(f.relative_to(out)): sha(f) for f in out.rglob('*') if f.is_file()})
    return {'files': sum(f.is_file() for f in out.rglob('*')), 'bytes': sum(f.stat().st_size for f in out.rglob('*') if f.is_file())}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['verify', 'publish', 'intake']); p.add_argument('source')
    p.add_argument('destination', nargs='?'); p.add_argument('--verification'); a = p.parse_args()
    if a.action == 'verify': result = verify(a.source)
    elif a.action == 'publish': result = publish(a.source, a.destination)
    else: result = intake(a.source, a.verification, a.destination)
    print(json.dumps(result))
